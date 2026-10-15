import numpy as np
import pandas as pd

from app.volatility import (
    HAR_COLS,
    daily_variance,
    fit_har,
    har_features,
    load_band_quantiles,
    price_band,
)


def _prices(n: int = 600, daily_sd: float = 0.02, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = 50.0 * np.exp(np.cumsum(rng.normal(0.0, daily_sd, n)))
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.full(n, 1_000_000),
        }
    )


def test_placeholder_rows_are_missing_not_calm():
    prices = _prices(10)
    prices.loc[5, ["open", "high", "low", "close"]] = prices.loc[4, "close"]
    prices.loc[7, "volume"] = 0
    rv = daily_variance(prices)
    assert np.isnan(rv.iloc[0])
    assert np.isnan(rv.iloc[5])
    assert np.isnan(rv.iloc[7])
    assert rv.drop([0, 5, 7]).gt(0).all()


def test_har_forecast_tracks_volatility_level():
    calm = fit_har(daily_variance(_prices(daily_sd=0.01)), [1, 5])
    wild = fit_har(daily_variance(_prices(daily_sd=0.04)), [1, 5])
    calm_last = har_features(daily_variance(_prices(daily_sd=0.01))).dropna().iloc[[-1]]
    wild_last = har_features(daily_variance(_prices(daily_sd=0.04))).dropna().iloc[[-1]]
    assert list(calm_last.columns) == HAR_COLS
    assert wild.predict(wild_last, 1)[0] > 4 * calm.predict(calm_last, 1)[0]


def test_band_quantiles_cover_every_horizon_and_widen():
    quantiles = load_band_quantiles()
    assert sorted(quantiles) == list(range(1, 31))
    for q in quantiles.values():
        assert q["q025"] < q["q10"] < 0 < q["q90"] < q["q975"]

    lo1, hi1 = price_band(100.0, 0.0004, 1, quantiles[1])
    lo20, hi20 = price_band(100.0, 0.0004, 20, quantiles[20])
    lo95, hi95 = price_band(100.0, 0.0004, 1, quantiles[1], "95")
    assert lo95 < lo1 < 100.0 < hi1 < hi95
    assert lo20 < lo1 and hi20 > hi1
