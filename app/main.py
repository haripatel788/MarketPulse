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

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

from app.auth import get_optional_user
from app.forecast_tool import register_forecast_runner
from app.routes import admin as admin_routes
from app.routes import bootstrap as bootstrap_routes
from app.routes import chat as chat_routes
from app.routes import forecast_history as forecast_history_routes
from app.routes import quotes as quotes_routes
from app.routes import users as users_routes
from app.routes import watchlist as watchlist_routes
from app.security.audit import AuditEvent, audit_log
from app.security.headers import SecurityHeadersMiddleware
from app.security.middleware import AdminPathGuardMiddleware, RequestContextMiddleware
from app.security.rate_limiter import enforce_predict_limits
from app.security.sanitizer import sanitize_ticker
from app.ticker import normalize_symbol
from app.tiers import (
    assert_horizon_allowed,
    effective_tier,
    enforce_authenticated_daily_forecast,
    record_forecast_usage,
    save_forecast_row,
)
from app.volatility import (
    HarModel,
    daily_variance,
    fit_har,
    har_features,
    load_band_quantiles,
    price_band,
)

logger = logging.getLogger(__name__)

MODEL_CACHE_TTL_SEC = int(os.getenv("MODEL_CACHE_TTL_SEC", "3600"))
MODEL_CACHE_MAX_KEYS = int(os.getenv("MODEL_CACHE_MAX_KEYS", "24"))
HISTORY_ROWS = 2520
MAX_HORIZON_DAYS = 30
MIN_HAR_ROWS = 250
BAND_QUANTILES = load_band_quantiles()

# In-memory model cache: symbol -> (trained_at_epoch, HAR model)
_model_cache: OrderedDict[str, tuple[float, HarModel]] = OrderedDict()
_model_cache_lock = threading.Lock()
_symbol_train_locks: dict[str, threading.Lock] = {}


def _symbol_lock(symbol: str) -> threading.Lock:
    with _model_cache_lock:
        if symbol not in _symbol_train_locks:
            _symbol_train_locks[symbol] = threading.Lock()
        return _symbol_train_locks[symbol]


class PredictionRequest(BaseModel):
    symbol: str = Field(min_length=1, max_length=10)
    horizon_days: int = Field(default=7, ge=1, le=MAX_HORIZON_DAYS)


class PredictionResponse(BaseModel):
    symbol: str
    data_source: str
    last_close: float
    model: str
    # No direction forecast: the center line is the last close, the honest point estimate.
    predicted_prices: list[float]
    predicted_dates: list[str]
    predicted_price_low: list[float]
    predicted_price_high: list[float]
    predicted_price_low_95: list[float]
    predicted_price_high_95: list[float]
    expected_move_pct: list[float]
    daily_vol_pct: float
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
            response = requests.get(url, timeout=8)
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
    # Yahoo first: it is the source the volatility model and bands were validated on, and Stooq
    # can stall behind a bot-check page.
    normalized = normalize_symbol(symbol)
    yahoo_data = try_fetch_yahoo_chart(normalized, rows)
    if yahoo_data is not None:
        return yahoo_data, "Yahoo Finance"
    stooq_data = try_fetch_stooq(normalized, rows)
    if stooq_data is not None:
        return stooq_data, "Stooq"
    raise ValueError("Unable to load data for this ticker right now.")


def train_har(raw_df: pd.DataFrame) -> HarModel:
    rv = daily_variance(raw_df)
    if har_features(rv).dropna().shape[0] < MIN_HAR_ROWS:
        raise ValueError("Not enough clean price history to fit the volatility model")
    return fit_har(rv, list(range(1, MAX_HORIZON_DAYS + 1)))


def get_or_train_cached_model(symbol: str, raw_df: pd.DataFrame) -> HarModel:
    key = symbol.upper()
    now = time.time()
    with _model_cache_lock:
        hit = _model_cache.get(key)
        if hit is not None and now - hit[0] < MODEL_CACHE_TTL_SEC:
            _model_cache.move_to_end(key)
            logger.info("Model cache hit for %s (age %.0fs)", key, now - hit[0])
            return hit[1]

    sym_lock = _symbol_lock(key)
    with sym_lock:
        with _model_cache_lock:
            hit = _model_cache.get(key)
            if hit is not None and time.time() - hit[0] < MODEL_CACHE_TTL_SEC:
                _model_cache.move_to_end(key)
                return hit[1]

        logger.info("Training model for %s", key)
        model = train_har(raw_df)
        with _model_cache_lock:
            _model_cache[key] = (time.time(), model)
            _model_cache.move_to_end(key)
            while len(_model_cache) > MODEL_CACHE_MAX_KEYS:
                _model_cache.popitem(last=False)
        return model


def naive_one_day_error(close: pd.Series) -> tuple[float, float]:
    """MAE in dollars and percent of "tomorrow = today" over the most recent 20% of history."""
    tail = close.tail(max(int(len(close) * 0.2), 2))
    err = tail.diff().abs().dropna()
    return float(err.mean()), float((err / tail.loc[err.index]).mean() * 100.0)


def forecast_bands(raw_df: pd.DataFrame, model: HarModel, horizon_days: int) -> dict[str, list[float]]:
    features = har_features(daily_variance(raw_df)).dropna()
    if features.empty:
        raise ValueError("Insufficient data to build volatility features")
    latest = features.iloc[[-1]]
    last_close = float(raw_df["close"].iloc[-1])
    out: dict[str, list[float]] = {"low": [], "high": [], "low_95": [], "high_95": [], "move_pct": []}
    for h in range(1, horizon_days + 1):
        variance = float(model.predict(latest, h)[0])
        lo, hi = price_band(last_close, variance, h, BAND_QUANTILES[h], "80")
        lo95, hi95 = price_band(last_close, variance, h, BAND_QUANTILES[h], "95")
        out["low"].append(round(lo, 2))
        out["high"].append(round(hi, 2))
        out["low_95"].append(round(lo95, 2))
        out["high_95"].append(round(hi95, 2))
        out["move_pct"].append(round(float(np.sqrt(h * variance)) * 100.0, 3))
    return out


def run_prediction(symbol: str, horizon_days: int) -> PredictionResponse:
    if os.getenv("KILL_SWITCH", "").lower() in ("1", "true", "yes"):
        raise HTTPException(status_code=503, detail="Forecasting temporarily disabled")

    sym = normalize_symbol(symbol.strip())
    try:
        raw_df, source = fetch_history(sym, rows=HISTORY_ROWS)
        logger.info("forecast symbol=%s horizon=%s", sym, horizon_days)
        model = get_or_train_cached_model(sym, raw_df)
        bands = forecast_bands(raw_df, model, horizon_days)
        mae, mae_pct = naive_one_day_error(raw_df["close"].astype(float))
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("predict failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    last_close = round(float(raw_df["close"].iloc[-1]), 2)
    current_date = pd.Timestamp(raw_df["date"].iloc[-1])
    dates = []
    for _ in range(horizon_days):
        current_date = current_date + NYSE_BDAY
        dates.append(current_date.strftime("%Y-%m-%d"))

    return PredictionResponse(
        symbol=sym,
        data_source=source,
        last_close=last_close,
        model="HAR volatility (per-stock), calibrated bands",
        predicted_prices=[last_close] * horizon_days,
        predicted_dates=dates,
        predicted_price_low=bands["low"],
        predicted_price_high=bands["high"],
        predicted_price_low_95=bands["low_95"],
        predicted_price_high_95=bands["high_95"],
        expected_move_pct=bands["move_pct"],
        daily_vol_pct=bands["move_pct"][0],
        model_mae=round(mae, 4),
        model_mae_pct=round(mae_pct, 4),
        baseline_mae=round(mae, 4),
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
app.include_router(quotes_routes.router, prefix="/api")
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
