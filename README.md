# MarketPulse

A complete backend + frontend stock prediction app using Python, FastAPI, pandas, and scikit-learn.

Repository: [github.com/haripatel788/MarketPulse](https://github.com/haripatel788/MarketPulse)

## Features

- Free market data from Stooq's public CSV endpoint
- ML forecasting pipeline built with scikit-learn `RandomForestRegressor`
- REST endpoint for predictions
- Browser UI for symbol input, horizon selection, and forecast table
- Built-in validation and health endpoint

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

Request body:

```json
{
  "symbol": "AAPL",
  "horizon_days": 7
}
```

Response body:

```json
{
  "symbol": "AAPL",
  "data_source": "Stooq",
  "last_close": 235.1,
  "predicted_prices": [236.2, 236.7],
  "predicted_dates": ["2026-02-13", "2026-02-16"],
  "model_mae": 2.0831,
  "model_mae_pct": 0.89,
  "baseline_mae": 2.4512
}
```

`model_mae` is the one-day holdout MAE (in dollars).  
`model_mae_pct` is one-day holdout MAE in percent.  
`baseline_mae` is the one-day MAE from a naive baseline (`tomorrow = today`).

The values above are illustrative. Metrics are computed per request on that ticker's last 20% of history and are not a benchmark.

## Evaluation

`python -m benchmarks.run_benchmark` reuses the API's features, hyperparameters, and rollout, and compares the model with two baselines: `tomorrow = today` and "predict the training-window mean return". Results are written to `benchmarks/results/`.

Results from the October 2026 run, using 20 large-cap tickers and Yahoo daily data:

| Setup | Model MAE% | Naive MAE% | Tickers beating naive |
|---|---|---|---|
| API holdout (last 20% of 1,200 days) | 1.49 | 1.46 | 4 / 20 |
| Walk-forward, 120 windows, Apr 2021 to Oct 2026 | 1.46 | 1.42 | 12 / 120 windows |
| 5-day recursive rollout | 3.55 | 3.29 | 6 / 20 |

The random forest does not beat either baseline on daily price error. The mean-return baseline performs about the same as the naive one, so this gap is not explained by drift.

Sentiment is not a model input. It is applied after prediction as `return + SENTIMENT_RETURN_COEF × sentiment × volume_ratio`, with a fixed, untuned coefficient of 0.004. A test on `data/sentiment_cache.csv` (20 tickers, 2009–2020, 10,840 news days) found the following:

- Sentiment correlates with same-day returns (rank IC 0.19) but not with next-day returns (rank IC 0.005, p = 0.59).
- The fixed overlay increases error by 0.7%.
- Using sentiment as model features changes error by +0.2%, which is not significant (p = 0.08).
