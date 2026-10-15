from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import _model_cache, app


def _history(end: str = "2025-12-24", n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=end, periods=n)
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.015, n)))
    opens = closes * np.exp(rng.normal(0.0, 0.004, n))
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": np.maximum(opens, closes) * 1.008,
            "low": np.minimum(opens, closes) * 0.992,
            "close": closes,
            "volume": rng.integers(900_000, 1_100_000, n),
        }
    )


@patch("app.main.fetch_history")
def test_predict_returns_calibrated_bands(mock_fetch):
    _model_cache.clear()
    history = _history()
    mock_fetch.return_value = (history, "Test")

    client = TestClient(app)
    res = client.post(
        "/api/predict",
        json={"symbol": "AAPL", "horizon_days": 3},
        headers={"User-Agent": "pytest"},
    )
    assert res.status_code == 200
    data = res.json()
    last = round(float(history["close"].iloc[-1]), 2)

    assert data["predicted_prices"] == [last, last, last]
    assert data["predicted_dates"] == ["2025-12-26", "2025-12-29", "2025-12-30"]
    for key in ("predicted_price_low", "predicted_price_high", "predicted_price_low_95",
                "predicted_price_high_95", "expected_move_pct"):
        assert len(data[key]) == 3

    lo, hi = data["predicted_price_low"], data["predicted_price_high"]
    lo95, hi95 = data["predicted_price_low_95"], data["predicted_price_high_95"]
    for i in range(3):
        assert lo95[i] < lo[i] < last < hi[i] < hi95[i]
    assert hi[2] - lo[2] > hi[0] - lo[0]
    assert data["expected_move_pct"] == sorted(data["expected_move_pct"])
    assert data["daily_vol_pct"] == data["expected_move_pct"][0]
    assert 0.5 < data["daily_vol_pct"] < 4.0
