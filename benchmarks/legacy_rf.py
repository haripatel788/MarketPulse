"""
Retired random-forest price forecaster, kept so run_benchmark.py and extended.py stay reproducible.
It lost to the naive "tomorrow = today" baseline (see README); the app now serves HAR volatility bands.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from app.main import NYSE_BDAY

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

SENTIMENT_RETURN_COEF = float(os.getenv("SENTIMENT_RETURN_COEF", "0.004"))

RF_PARAMS = {
    "n_estimators": 300,
    "random_state": 42,
    "min_samples_leaf": 12,
    "max_depth": 14,
    "max_features": "sqrt",
    "n_jobs": -1,
}


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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    eval_model = RandomForestRegressor(**RF_PARAMS)
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
    final_model = RandomForestRegressor(**RF_PARAMS)
    final_model.fit(x, y)
    return final_model, mae, mae_pct, baseline_mae, FEATURE_COLS


def train_and_metrics(raw_df: pd.DataFrame) -> tuple[RandomForestRegressor, float, float, float, list[str]]:
    feature_df, target_cols = build_training_features(raw_df)
    if len(feature_df) < 80:
        raise ValueError("Not enough clean data after preprocessing")
    return train_model(feature_df, target_cols)


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
