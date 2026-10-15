# MarketPulse

A FastAPI + browser app that forecasts how far a stock is likely to move, with calibrated price ranges.

Repository: [github.com/haripatel788/MarketPulse](https://github.com/haripatel788/MarketPulse)

## What it does

- Fits a per-stock HAR volatility model on up to 10 years of daily OHLC data from Stooq or Yahoo.
- Returns the expected move and 80% / 95% price ranges for each of the next 1-30 trading days.
- Does **not** predict direction. The center line is the last close, because no model tested here
  beat "tomorrow = today" for the price itself (see [Price direction](#price-direction-retired-model)).

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## API

`POST /api/predict`

```json
{ "symbol": "AAPL", "horizon_days": 5 }
```

Response (illustrative values):

```json
{
  "symbol": "AAPL",
  "data_source": "Yahoo Finance",
  "last_close": 342.96,
  "model": "HAR volatility (per-stock), calibrated bands",
  "predicted_prices": [342.96, 342.96],
  "predicted_dates": ["2026-10-15", "2026-10-16"],
  "predicted_price_low": [336.16, 333.16],
  "predicted_price_high": [350.15, 353.27],
  "predicted_price_low_95": [330.0, 325.9],
  "predicted_price_high_95": [355.4, 360.6],
  "expected_move_pct": [1.65, 2.33],
  "daily_vol_pct": 1.65,
  "model_mae": 4.1,
  "model_mae_pct": 1.3,
  "baseline_mae": 4.1
}
```

- `predicted_price_low/high` is the 80% range and `_95` the 95% range, for each date.
- `expected_move_pct` is a one-standard-deviation move from the last close to each date.
- `model_mae` / `baseline_mae` is the "tomorrow = today" one-day error over the last 20% of
  history, in dollars. They are equal because the center line is the naive forecast. Kept for
  compatibility with saved forecast history.

## How the forecast works

Code: `app/volatility.py`.

1. **Daily variance.** Each day's variance is the overnight gap plus the Garman-Klass intraday
   range estimator: `ln(open/prev_close)^2 + 0.5 ln(high/low)^2 - (2 ln 2 - 1) ln(close/open)^2`.
   Days with zero range or zero volume are placeholder quotes and are treated as missing.
2. **HAR model.** For each horizon `h`, OLS regresses the log of the next `h` days' average
   variance on the log of the last day's, 5-day and 22-day average variance. Forecasts use a
   lognormal bias correction.
3. **Price ranges.** The range is `last_close * exp(q * sqrt(h * forecast_variance))`. The `q`
   multipliers are percentiles of realized standardized moves, frozen in
   `app/vol_band_quantiles.json` from development data only.

## Evaluation

Code: `benchmarks/vol_benchmark.py`. The rules were written down before any results, in
`benchmarks/PREREGISTRATION.md`. That file has dated amendments, all made before the final run.

- **Development set:** 20 large caps, walk-forward test years from Oct 2018 to Oct 2025.
  All tuning happened here.
- **Final test:** run once, on 480 other S&P 500 stocks, for forecasts from Oct 14, 2025 to
  Oct 13, 2026. 474 stocks had enough history to score. Nothing was refit or re-tuned after it.

Loss is QLIKE on variance; lower is better. DM is a Diebold-Mariano test on the daily
cross-sectional mean loss difference, with Newey-West lag `h`.

### Volatility accuracy, final test

| Horizon | HAR QLIKE | EWMA QLIKE | Stocks where HAR beats EWMA | DM p-value | Pass (≥70%, p<0.05) |
|---|---|---|---|---|---|
| 1 day | 0.454 | 0.524 | 86% | 1e-16 | yes |
| 5 days | 0.273 | 0.336 | 87% | 8e-8 | yes |
| 20 days | 0.164 | 0.234 | 87% | 0.004 | yes |

HAR also beats GARCH(1,1) and a 20-day historical average at every horizon. The app previously
used the 20-day historical average.

### Range calibration, final test

| Horizon | 80% range held the close | 95% range held the close | Pass (78-82%) |
|---|---|---|---|
| 1 day | 81.6% | 95.8% | yes |
| 5 days | 81.7% | 95.9% | yes |
| 20 days | 80.9% | 96.1% | yes |

### Secondary result: pooled model (not shipped)

A linear model trained across the 20 development stocks used extra inputs: recent losses,
overnight gap, volume, and SPY volatility. It was pre-registered against HAR at 1 and 5 days.
On the final test:

- **1 day:** passed. It beat HAR on 83% of stocks, p = 6e-8, with about 9% lower QLIKE.
- **5 days:** failed. It beat HAR on 58% of stocks against a 60% threshold, although DM p = 0.013.

The app ships HAR. Gradient boosting was dropped on development data. It is still printed in the
final output but makes no claim.

## Price direction (retired model)

The previous version predicted prices with a 27-feature random forest. It is kept in
`benchmarks/legacy_rf.py` so `python -m benchmarks.run_benchmark` and `benchmarks/extended.py`
still reproduce these results (20 large caps, Yahoo daily data):

| Setup | Model MAE% | Naive MAE% | Tickers beating naive |
|---|---|---|---|
| API holdout (last 20% of 1,200 days) | 1.49 | 1.46 | 4 / 20 |
| Walk-forward, 120 windows, Apr 2021 to Oct 2026 | 1.46 | 1.42 | 12 / 120 windows |
| 5-day recursive rollout | 3.55 | 3.29 | 6 / 20 |

On 490 S&P 500 stocks it was 2.5% worse than naive on one-day error. Removing the 10 raw price
features roughly halved that gap but did not close it.

News sentiment (NewsAPI headlines scored by Llama 3.3 70B) was tested on 20 tickers, 2009-2020,
10,840 news days:

- It correlates with same-day returns (rank IC 0.19).
- It does not correlate with next-day returns (rank IC 0.005, p = 0.59).
- It is no longer applied to forecasts.
