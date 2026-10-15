import numpy as np
import pandas as pd

from benchmarks.legacy_rf import FEATURE_COLS, build_feature_frame, forecast_next_days


class ReturnFromCloseModel:
    def predict(self, features: pd.DataFrame):
        returns = (features["close"] / 10000.0).to_numpy(dtype=float)
        return returns.reshape(-1, 1)


class ConstantModel:
    def predict(self, features: pd.DataFrame):
        return np.zeros((len(features), 1), dtype=float)


def make_history(end_date: str, periods: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range(end=end_date, periods=periods)
    closes = [100.0 + i for i in range(periods)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
            "volume": [1_000_000 + i for i in range(periods)],
        }
    )


def test_forecast_uses_most_recent_day_features():
    history = make_history("2025-02-14")
    prices, dates = forecast_next_days(history, ReturnFromCloseModel(), FEATURE_COLS, 1)

    last_close = history["close"].iloc[-1]
    assert prices[0] == round(last_close * (1.0 + (last_close / 10000.0)), 2)
    assert dates[0] == "2025-02-18"


def test_forecast_skips_nyse_holidays():
    history = make_history("2025-12-24")
    prices, dates = forecast_next_days(history, ConstantModel(), FEATURE_COLS, 1)

    assert prices[0] == round(history["close"].iloc[-1], 2)
    assert dates[0] == "2025-12-26"


def test_forecast_iterates_multi_day_constant():
    history = make_history("2025-12-24")
    prices, dates = forecast_next_days(history, ConstantModel(), FEATURE_COLS, 2)
    last = round(history["close"].iloc[-1], 2)
    assert prices == [last, last]
    assert dates[0] == "2025-12-26"
    assert dates[1] == "2025-12-29"


def test_feature_frame_has_expected_columns():
    history = make_history("2025-02-14", periods=100)
    frame = build_feature_frame(history).dropna(subset=FEATURE_COLS)
    assert len(frame) > 0
    assert set(FEATURE_COLS).issubset(frame.columns)