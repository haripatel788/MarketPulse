import logging
import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

from app.auth import get_optional_user
from app.forecast_tool import register_forecast_runner
from app.news import fetch_live_headlines
from app.routes import admin as admin_routes
from app.routes import bootstrap as bootstrap_routes
from app.routes import chat as chat_routes
from app.routes import forecast_history as forecast_history_routes
from app.routes import users as users_routes
from app.routes import watchlist as watchlist_routes
from app.security.audit import AuditEvent, audit_log
from app.security.headers import SecurityHeadersMiddleware
from app.security.middleware import AdminPathGuardMiddleware, RequestContextMiddleware
from app.security.rate_limiter import enforce_predict_limits
from app.security.sanitizer import sanitize_ticker
from app.sentiment import compute_weighted_sentiment, score_headlines_groq
from app.ticker import normalize_symbol
from app.tiers import (
    assert_horizon_allowed,
    effective_tier,
    enforce_authenticated_daily_forecast,
    record_forecast_usage,
    save_forecast_row,
)

logger = logging.getLogger(__name__)

# Price-only features (27). Sentiment is applied after each 1-day return (no train/serve skew).
FEATURE_COLS = [
    "close",
    "volume",
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "ma_5",
    "ma_10",
    "ma_20",
    "ma_50",
    "price_vs_ma_10",
    "price_vs_ma_20",
    "price_vs_ma_50",
    "volatility_10",
    "volatility_20",
    "close_lag_1",
    "close_lag_2",
    "close_lag_3",
    "close_lag_5",
    "close_lag_10",
    "volume_lag_1",
    "volume_lag_5",
    "volume_lag_10",
    "volume_change_1d",
    "intraday_range",
    "high_close_ratio",
    "low_close_ratio",
]

MODEL_CACHE_TTL_SEC = int(os.getenv("MODEL_CACHE_TTL_SEC", "3600"))
MODEL_CACHE_MAX_KEYS = int(os.getenv("MODEL_CACHE_MAX_KEYS", "24"))
SENTIMENT_RETURN_COEF = float(os.getenv("SENTIMENT_RETURN_COEF", "0.004"))

RF_PARAMS = {
    "n_estimators": 300,
    "random_state": 42,
    "min_samples_leaf": 12,
    "max_depth": 14,
    "max_features": "sqrt",
    "n_jobs": -1,
}

# In-memory model cache: symbol -> (trained_at_epoch, model, mae, mae_pct, baseline_mae, feature_cols)
_model_cache: OrderedDict[str, tuple[float, RandomForestRegressor, float, float, float, list[str]]] = (
    OrderedDict()
)
_model_cache_lock = threading.Lock()
_symbol_train_locks: dict[str, threading.Lock] = {}


def _symbol_lock(symbol: str) -> threading.Lock:
    with _model_cache_lock:
        if symbol not in _symbol_train_locks:
            _symbol_train_locks[symbol] = threading.Lock()
        return _symbol_train_locks[symbol]


class PredictionRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=10)
    horizon_days: int = Field(default=7, ge=1, le=30)


class PredictionResponse(BaseModel):
    symbol: str
    data_source: str
    last_close: float
    predicted_prices: list[float]
    predicted_dates: list[str]
    predicted_price_low: list[float]
    predicted_price_high: list[float]
    model_mae: float
    model_mae_pct: float
    baseline_mae: float


class NyseHolidayCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday, start_date="2022-01-01"),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


NYSE_BDAY = CustomBusinessDay(calendar=NyseHolidayCalendar())


def standardize_dataframe(frame: pd.DataFrame, rows: int) -> pd.DataFrame | None:
    needed = {"date", "open", "high", "low", "close", "volume"}
    if not needed.issubset(frame.columns):
        return None
    work = frame[["date", "open", "high", "low", "close", "volume"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["date", "close", "volume"]).sort_values("date").tail(rows)
    if len(work) < 120:
        return None
    return work.reset_index(drop=True)


def try_fetch_stooq(symbol: str, rows: int = 1200) -> pd.DataFrame | None:
    symbol_variants = [symbol.lower()]
    if "." not in symbol:
        symbol_variants.append(f"{symbol.lower()}.us")
    for variant in symbol_variants:
        url = f"https://stooq.com/q/d/l/?s={variant}&i=d"
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            df = pd.read_csv(pd.io.common.StringIO(response.text))
        except Exception:
            continue
        if "Date" not in df.columns or "Close" not in df.columns or df.empty:
            continue
        normalized = standardize_dataframe(df.rename(columns=str.lower), rows)
        if normalized is not None:
            return normalized
    return None


def try_fetch_yahoo_chart(symbol: str, rows: int = 1200) -> pd.DataFrame | None:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    params = {"range": "10y", "interval": "1d", "events": "div,splits"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=20)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None
    chart = payload.get("chart", {})
    result = chart.get("result")
    if not result:
        return None
    item = result[0]
    timestamps = item.get("timestamp")
    quote = item.get("indicators", {}).get("quote", [])
    if not timestamps or not quote:
        return None
    quote0 = quote[0]
    length = min(
        len(timestamps),
        len(quote0.get("open", [])),
        len(quote0.get("high", [])),
        len(quote0.get("low", [])),
        len(quote0.get("close", [])),
        len(quote0.get("volume", [])),
    )
    if length == 0:
        return None
    df = pd.DataFrame({
        "date": pd.to_datetime(timestamps[:length], unit="s", utc=True).tz_localize(None),
        "open": quote0["open"][:length],
        "high": quote0["high"][:length],
        "low": quote0["low"][:length],
        "close": quote0["close"][:length],
        "volume": quote0["volume"][:length],
    })
    return standardize_dataframe(df, rows)


def fetch_history(symbol: str, rows: int = 1200) -> tuple[pd.DataFrame, str]:
    normalized = normalize_symbol(symbol)
    stooq_data = try_fetch_stooq(normalized, rows)
    if stooq_data is not None:
        return stooq_data, "Stooq"
    yahoo_data = try_fetch_yahoo_chart(normalized, rows)
    if yahoo_data is not None:
        return yahoo_data, "Yahoo Finance"
    raise ValueError("Unable to load data for this ticker right now.")


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["return_1d"] = work["close"].pct_change()
    work["return_3d"] = work["close"].pct_change(3)
    work["return_5d"] = work["close"].pct_change(5)
    work["return_10d"] = work["close"].pct_change(10)
    work["ma_5"] = work["close"].rolling(5).mean()
    work["ma_10"] = work["close"].rolling(10).mean()
    work["ma_20"] = work["close"].rolling(20).mean()
    work["ma_50"] = work["close"].rolling(50).mean()
    work["price_vs_ma_10"] = work["close"] / work["ma_10"] - 1.0
    work["price_vs_ma_20"] = work["close"] / work["ma_20"] - 1.0
    work["price_vs_ma_50"] = work["close"] / work["ma_50"] - 1.0
    work["volatility_10"] = work["return_1d"].rolling(10).std()
    work["volatility_20"] = work["return_1d"].rolling(20).std()
    work["close_lag_1"] = work["close"].shift(1)
    work["close_lag_2"] = work["close"].shift(2)
    work["close_lag_3"] = work["close"].shift(3)
    work["close_lag_5"] = work["close"].shift(5)
    work["close_lag_10"] = work["close"].shift(10)
    work["volume_lag_1"] = work["volume"].shift(1)
    work["volume_lag_5"] = work["volume"].shift(5)
    work["volume_lag_10"] = work["volume"].shift(10)
    work["volume_change_1d"] = work["volume"].pct_change()
    work["intraday_range"] = (work["high"] - work["low"]) / work["close"]
    work["high_close_ratio"] = work["high"] / work["close"] - 1.0
    work["low_close_ratio"] = work["low"] / work["close"] - 1.0
    return work


def build_training_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-day forward return only; iterated rollout happens in forecast_next_days."""
    work = build_feature_frame(df)
    target_cols = ["target_return_1"]
    work["target_return_1"] = work["close"].shift(-1) / work["close"] - 1.0
    required_cols = FEATURE_COLS + target_cols
    return work.dropna(subset=required_cols).reset_index(drop=True), target_cols


def latest_feature_vector(history: pd.DataFrame) -> pd.Series:
    features = build_feature_frame(history).dropna(subset=FEATURE_COLS)
    if features.empty:
        raise ValueError("Insufficient data to build forecasting features")
    return features.iloc[-1][FEATURE_COLS].copy()


def train_model(
    feature_df: pd.DataFrame, target_cols: list[str]
) -> tuple[RandomForestRegressor, float, float, float, list[str]]:
    x = feature_df[FEATURE_COLS]
    y = feature_df[target_cols]
    # Last 20% is a single temporal holdout (shuffle=False): no future rows in train.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    params = RF_PARAMS
    eval_model = RandomForestRegressor(**params)
    eval_model.fit(x_train, y_train)
    preds = np.asarray(eval_model.predict(x_test), dtype=float)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)
    actual_return_1d = y_test[target_cols[0]].to_numpy(dtype=float)
    predicted_return_1d = preds[:, 0]
    current_close = x_test["close"].to_numpy(dtype=float)
    actual_next_close = current_close * (1.0 + actual_return_1d)
    model_next_close = current_close * (1.0 + predicted_return_1d)
    baseline_next_close = current_close
    mae = float(mean_absolute_error(actual_next_close, model_next_close))
    baseline_mae = float(mean_absolute_error(actual_next_close, baseline_next_close))
    safe_actual = np.where(np.abs(actual_next_close) < 1e-9, 1.0, actual_next_close)
    mae_pct = float(np.mean(np.abs((actual_next_close - model_next_close) / safe_actual)) * 100.0)
    final_model = RandomForestRegressor(**params)
    final_model.fit(x, y)
    importances = np.asarray(final_model.feature_importances_, dtype=float)
    top_idx = np.argsort(importances)[::-1][:10]
    top_pairs = [f"{FEATURE_COLS[i]}={importances[i]:.4f}" for i in top_idx]
    logger.info("Top mean feature importances: %s", ", ".join(top_pairs))
    return final_model, mae, mae_pct, baseline_mae, FEATURE_COLS


def _append_synthetic_trading_day(history: pd.DataFrame, next_close: float, next_date: pd.Timestamp) -> pd.DataFrame:
    h = history.copy()
    last_close = float(h["close"].iloc[-1])
    o = last_close
    c = float(next_close)
    hi = max(o, c) * 1.002
    lo = min(o, c) * 0.998
    vol = float(h["volume"].tail(20).mean())
    if not np.isfinite(vol) or vol <= 0:
        vol = float(h["volume"].iloc[-1])
    new_row = pd.DataFrame(
        [{"date": next_date, "open": o, "high": hi, "low": lo, "close": c, "volume": vol}]
    )
    return pd.concat([h, new_row], ignore_index=True)


def forecast_next_days(
    history_df: pd.DataFrame,
    model: RandomForestRegressor,
    feature_cols: list[str],
    horizon_days: int,
    weighted_sentiment: float = 0.0,
    volume_ratio: float = 1.0,
) -> tuple[list[float], list[str]]:
    """
    Recursive one-day-ahead forecast: each step updates history with the predicted close,
    then rebuilds features from the extended series.
    """
    history = history_df.copy()
    prices: list[float] = []
    dates: list[str] = []
    current_date = pd.Timestamp(history["date"].iloc[-1])
    vr = float(np.clip(volume_ratio, 0.25, 3.0))
    sent_adj = SENTIMENT_RETURN_COEF * float(np.clip(weighted_sentiment, -1.0, 1.0)) * vr

    for _ in range(horizon_days):
        latest = latest_feature_vector(history)
        raw = np.asarray(model.predict(latest[feature_cols].to_frame().T), dtype=float)
        r_raw = float(raw.reshape(-1)[0])
        r_raw = float(np.clip(r_raw, -0.30, 0.30))
        r = float(np.clip(r_raw + sent_adj, -0.30, 0.30))

        last_close = float(history["close"].iloc[-1])
        next_close = round(last_close * (1.0 + r), 2)
        current_date = current_date + NYSE_BDAY
        prices.append(next_close)
        dates.append(current_date.strftime("%Y-%m-%d"))
        history = _append_synthetic_trading_day(history, next_close, current_date)

    return prices, dates


def _train_and_metrics(raw_df: pd.DataFrame) -> tuple[RandomForestRegressor, float, float, float, list[str]]:
    feature_df, target_cols = build_training_features(raw_df)
    if len(feature_df) < 80:
        raise ValueError("Not enough clean data after preprocessing")
    return train_model(feature_df, target_cols)


def get_or_train_cached_model(symbol: str, raw_df: pd.DataFrame) -> tuple[RandomForestRegressor, float, float, float, list[str]]:
    key = symbol.upper()
    now = time.time()
    with _model_cache_lock:
        hit = _model_cache.get(key)
        if hit is not None and now - hit[0] < MODEL_CACHE_TTL_SEC:
            _model_cache.move_to_end(key)
            logger.info("Model cache hit for %s (age %.0fs)", key, now - hit[0])
            return hit[1], hit[2], hit[3], hit[4], hit[5]

    sym_lock = _symbol_lock(key)
    with sym_lock:
        with _model_cache_lock:
            hit = _model_cache.get(key)
            if hit is not None and time.time() - hit[0] < MODEL_CACHE_TTL_SEC:
                _model_cache.move_to_end(key)
                return hit[1], hit[2], hit[3], hit[4], hit[5]

        logger.info("Training model for %s", key)
        model, mae, mae_pct, baseline_mae, cols = _train_and_metrics(raw_df)
        trained_at = time.time()
        with _model_cache_lock:
            _model_cache[key] = (trained_at, model, mae, mae_pct, baseline_mae, cols)
            _model_cache.move_to_end(key)
            while len(_model_cache) > MODEL_CACHE_MAX_KEYS:
                _model_cache.popitem(last=False)
        return model, mae, mae_pct, baseline_mae, cols


def run_prediction(symbol: str, horizon_days: int) -> PredictionResponse:
    if os.getenv("KILL_SWITCH", "").lower() in ("1", "true", "yes"):
        raise HTTPException(status_code=503, detail="Forecasting temporarily disabled")

    sym = normalize_symbol(symbol.strip())
    weighted_sentiment = 0.0
    volume_ratio = 1.0
    try:
        raw_df, source = fetch_history(sym)
        logger.info("forecast symbol=%s horizon=%s", sym, horizon_days)

        model, mae, mae_pct, baseline_mae, feature_cols = get_or_train_cached_model(sym, raw_df)

        try:
            headlines = fetch_live_headlines(sym)
            if headlines:
                scored = score_headlines_groq(headlines, sym)
                weighted_sentiment = compute_weighted_sentiment(scored)
                recent_volume = float(raw_df["volume"].iloc[-1])
                avg_volume = float(raw_df["volume"].tail(20).mean())
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
                logger.info(
                    "sentiment symbol=%s headlines=%s weighted=%.4f vol_ratio=%.3f",
                    sym,
                    len(headlines),
                    weighted_sentiment,
                    volume_ratio,
                )
        except Exception:
            logger.exception("sentiment pipeline failed for %s", sym)

        prices, dates = forecast_next_days(
            raw_df,
            model,
            feature_cols,
            horizon_days,
            weighted_sentiment=weighted_sentiment,
            volume_ratio=volume_ratio,
        )
        low = [round(p - mae, 2) for p in prices]
        high = [round(p + mae, 2) for p in prices]
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("predict failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictionResponse(
        symbol=sym,
        data_source=source,
        last_close=round(float(raw_df["close"].iloc[-1]), 2),
        predicted_prices=prices,
        predicted_dates=dates,
        predicted_price_low=low,
        predicted_price_high=high,
        model_mae=round(mae, 4),
        model_mae_pct=round(mae_pct, 4),
        baseline_mae=round(baseline_mae, 4),
    )


register_forecast_runner(run_prediction)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("logs", exist_ok=True)
    logger.info("MarketPulse startup")
    yield
    logger.info("MarketPulse shutdown")


_env = (os.getenv("ENVIRONMENT") or "").lower()
_docs = None if _env in ("production", "prod") else "/docs"
_redoc = None if _env in ("production", "prod") else "/redoc"
_openapi = None if _env in ("production", "prod") else "/openapi.json"

app = FastAPI(title="Market Pulse", lifespan=lifespan, docs_url=_docs, redoc_url=_redoc, openapi_url=_openapi)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(AdminPathGuardMiddleware)
app.add_middleware(RequestContextMiddleware)

if _env in ("production", "prod"):
    _hosts = [h.strip() for h in os.getenv("ALLOWED_HOSTS", "").split(",") if h.strip()]
    if _hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=_hosts)

_cors = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors if _cors != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)

app.include_router(bootstrap_routes.router, prefix="/api")
app.include_router(chat_routes.router, prefix="/api")
app.include_router(admin_routes.router, prefix="/api")
app.include_router(users_routes.router, prefix="/api")
app.include_router(watchlist_routes.router, prefix="/api")
app.include_router(forecast_history_routes.router, prefix="/api")


@app.get("/")
def root() -> FileResponse:
    return FileResponse("public/index.html")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.post("/api/predict", response_model=PredictionResponse)
def predict(
    request: Request,
    payload: PredictionRequest,
    user: dict | None = Depends(get_optional_user),
) -> PredictionResponse:
    tier = effective_tier(user)
    enforce_predict_limits(request, tier_public=(user is None))
    symbol = sanitize_ticker(payload.symbol)
    assert_horizon_allowed(tier, payload.horizon_days)
    if user:
        enforce_authenticated_daily_forecast(user)

    result = run_prediction(symbol, payload.horizon_days)

    if user:
        try:
            record_forecast_usage(user)
        except Exception:
            logger.exception("failed to update daily forecast usage")
        try:
            save_forecast_row(
                user,
                {
                    "symbol": result.symbol,
                    "horizon": payload.horizon_days,
                    "last_close": result.last_close,
                    "predicted_prices": result.predicted_prices,
                    "predicted_dates": result.predicted_dates,
                    "model_mae": result.model_mae,
                },
            )
        except Exception:
            logger.exception("failed to persist forecast row")

    audit_log(
        AuditEvent.FORECAST_RUN,
        request,
        user_id=user.get("id") if user else None,
        details={"symbol": result.symbol, "horizon": payload.horizon_days, "tier": tier},
    )
    return result


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}
