"""Range-based realized variance, HAR volatility forecasts, and calibrated price bands."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

GK_COEF = 2.0 * np.log(2.0) - 1.0
VAR_FLOOR = 1e-7
HAR_COLS = ["log_rv_d", "log_rv_w", "log_rv_m"]
BAND_QUANTILES_PATH = Path(__file__).with_name("vol_band_quantiles.json")
BAND_LEVELS = {"80": ("q10", "q90"), "95": ("q025", "q975")}
BAND_PROBS = {"q025": 0.025, "q10": 0.10, "q90": 0.90, "q975": 0.975}


def daily_variance(df: pd.DataFrame) -> pd.Series:
    """Overnight gap plus Garman-Klass intraday variance, one value per row (first row is NaN)."""
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    lo = df["low"].astype(float)
    c = df["close"].astype(float)
    prev_c = c.shift(1)

    close_to_close = np.log(c / prev_c) ** 2
    overnight = np.log(o / prev_c) ** 2
    garman_klass = 0.5 * np.log(h / lo) ** 2 - GK_COEF * np.log(c / o) ** 2

    tol_lo, tol_hi = lo * 0.999, h * 1.001
    valid = (
        (lo > 0)
        & (h >= lo)
        & o.between(tol_lo, tol_hi)
        & c.between(tol_lo, tol_hi)
        & overnight.notna()
        & garman_klass.notna()
    )
    rv = pd.Series(np.where(valid, overnight + garman_klass, close_to_close), index=df.index)
    rv = rv.clip(lower=VAR_FLOOR)
    # Zero-range or zero-volume rows are placeholder quotes (e.g. pre-listing history), not calm days.
    placeholder = (h <= lo) | (df["volume"].astype(float) <= 0)
    rv[placeholder] = np.nan
    if len(rv):
        rv.iloc[0] = np.nan
    return rv


def forward_mean_variance(rv: pd.Series, horizon: int) -> pd.Series:
    """Average of rv over the next `horizon` rows, aligned to the origin row."""
    return rv.rolling(horizon).mean().shift(-horizon)


def har_features(rv: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "log_rv_d": np.log(rv),
        "log_rv_w": np.log(rv.rolling(5).mean()),
        "log_rv_m": np.log(rv.rolling(22).mean()),
    }, index=rv.index)


@dataclass
class HarModel:
    """Per-stock log-HAR, one OLS fit per horizon; forecasts mean daily variance over h days."""

    coefs: dict[int, np.ndarray]
    resid_vars: dict[int, float]

    def predict(self, features: pd.DataFrame, horizon: int) -> np.ndarray:
        x = np.column_stack([np.ones(len(features)), features[HAR_COLS].to_numpy()])
        return np.exp((x * self.coefs[horizon]).sum(axis=1) + 0.5 * self.resid_vars[horizon])


def fit_har(rv: pd.Series, horizons: list[int]) -> HarModel:
    features = har_features(rv)
    coefs: dict[int, np.ndarray] = {}
    resid_vars: dict[int, float] = {}
    for h in horizons:
        frame = features.assign(y=forward_mean_variance(rv, h)).dropna()
        x = np.column_stack([np.ones(len(frame)), frame[HAR_COLS].to_numpy()])
        y = np.log(frame["y"].to_numpy())
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        coefs[h] = coef
        resid_vars[h] = float(np.var(y - (x * coef).sum(axis=1), ddof=x.shape[1]))
    return HarModel(coefs, resid_vars)


def standardized_moves(close: pd.Series, variance: pd.Series, horizon: int) -> pd.Series:
    """h-day log return divided by the forecast h-day standard deviation, aligned to the origin row."""
    move = np.log(close.shift(-horizon) / close)
    return move / np.sqrt(horizon * variance)


def load_band_quantiles(path: Path = BAND_QUANTILES_PATH) -> dict[int, dict[str, float]]:
    raw = json.loads(path.read_text())
    return {int(h): q for h, q in raw["quantiles"].items()}


def price_band(last_close: float, variance: float, horizon: int, quantiles: dict[str, float],
               level: str = "80") -> tuple[float, float]:
    lo_key, hi_key = BAND_LEVELS[level]
    sd = np.sqrt(horizon * variance)
    return last_close * float(np.exp(quantiles[lo_key] * sd)), last_close * float(np.exp(quantiles[hi_key] * sd))


MARKET_COLS = ["mkt_log_rv_d", "mkt_log_rv_w", "mkt_log_rv_m"]
POOLED_COLS = HAR_COLS + [
    "log_rv_q",
    "neg_ret_1d",
    "neg_ret_5d",
    "log_overnight",
    "log_volume_ratio",
    *MARKET_COLS,
    "rel_log_rv_m",
]


def market_features(market_prices: pd.DataFrame) -> pd.DataFrame:
    """HAR components of the market index (SPY), keyed by date."""
    har = har_features(daily_variance(market_prices))
    har.columns = MARKET_COLS
    return pd.concat([market_prices[["date"]], har], axis=1)


def pooled_features(prices: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Per-day inputs for the cross-stock volatility model; `market` comes from market_features."""
    rv = daily_variance(prices)
    close = prices["close"].astype(float)
    log_ret = np.log(close / close.shift(1))
    volume = prices["volume"].astype(float)
    f = pd.concat([prices[["date"]], har_features(rv)], axis=1)
    f["log_rv_q"] = np.log(rv.rolling(66).mean())
    f["neg_ret_1d"] = log_ret.clip(upper=0.0)
    f["neg_ret_5d"] = log_ret.rolling(5).sum().clip(upper=0.0)
    f["log_overnight"] = np.log(np.log(prices["open"].astype(float) / close.shift(1)) ** 2 + VAR_FLOOR)
    f["log_volume_ratio"] = np.log((volume / volume.rolling(22).mean()).clip(lower=1e-3))
    f = f.merge(market, on="date", how="left")
    f["rel_log_rv_m"] = f["log_rv_m"] - f["mkt_log_rv_m"]
    return f.drop(columns="date")
