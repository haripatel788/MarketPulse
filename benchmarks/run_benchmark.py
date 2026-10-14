"""
Offline benchmark for the MarketPulse forecaster.

Reuses the exact feature pipeline, hyperparameters, and recursive rollout from app.main,
and compares the random forest against two baselines on 1-day (and 5-day) price error:
  - naive:      tomorrow's close = today's close
  - train_mean: tomorrow's return = mean daily return of the training window

Experiments:
  1. api_holdout   : same setup as /api/predict (last 1,200 rows, last 20% holdout), N tickers
  2. walk_forward  : sliding 1,200-row windows over ~10y of history, non-overlapping test blocks
  3. horizon_5d    : recursive 5-day rollout (forecast_next_days) on the api_holdout test split
  4. sentiment     : historical sentiment (data/sentiment_cache.csv, 2009-2020) tested three ways:
                     production fixed-weight overlay, tuned overlay, and as model input features

Usage:
  .venv/bin/python -m benchmarks.run_benchmark            # all experiments
  .venv/bin/python -m benchmarks.run_benchmark --only api_holdout walk_forward
  .venv/bin/python -m benchmarks.run_benchmark --refresh  # re-download prices
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

from app.main import (
    FEATURE_COLS,
    RF_PARAMS,
    SENTIMENT_RETURN_COEF,
    build_training_features,
    forecast_next_days,
    standardize_dataframe,
)

logging.getLogger("app").setLevel(logging.WARNING)

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "price_cache"
RESULTS_DIR = ROOT / "results"
SENTIMENT_PATH = ROOT.parent / "data" / "sentiment_cache.csv"

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "BAC", "XOM",
    "CVX", "JNJ", "PFE", "KO", "PG", "WMT", "DIS", "INTC", "BA", "F",
]
API_ROWS = 1200
TEST_FRAC = 0.2
SENTIMENT_COLS = ["daily_sentiment", "sentiment_3d_avg", "sentiment_momentum"]

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


# --------------------------------------------------------------------------- data


def fetch_yahoo(symbol: str, start: str | None, end: str | None, refresh: bool) -> pd.DataFrame | None:
    tag = f"{symbol}_{start or 'na'}_{end or '10y'}".replace("-", "")
    path = CACHE_DIR / f"{tag}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["date"])
    params: dict[str, str | int] = {"interval": "1d", "events": "div,splits"}
    if start:
        params["period1"] = int(pd.Timestamp(start).timestamp())
        params["period2"] = int(pd.Timestamp(end).timestamp())
    else:
        params["range"] = "10y"
    try:
        r = requests.get(YAHOO_URL.format(symbol=symbol), params=params, headers=YAHOO_HEADERS, timeout=20)
        r.raise_for_status()
        item = r.json()["chart"]["result"][0]
        q = item["indicators"]["quote"][0]
        df = pd.DataFrame({
            "date": pd.to_datetime(item["timestamp"], unit="s", utc=True).tz_localize(None).normalize(),
            "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"], "volume": q["volume"],
        })
    except Exception as exc:
        print(f"  ! {symbol}: fetch failed ({exc.__class__.__name__})")
        return None
    df = standardize_dataframe(df, rows=100_000)
    if df is None:
        return None
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    time.sleep(0.3)
    return df


# --------------------------------------------------------------------------- metrics


def fit_rf(x: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(x, y)
    return model


def mae_pct(actual_close: np.ndarray, pred_close: np.ndarray) -> float:
    """Same definition as app.main.train_model: mean(|actual - pred| / actual) * 100."""
    return float(np.mean(np.abs(actual_close - pred_close) / actual_close) * 100.0)


def score(close: np.ndarray, actual_r: np.ndarray, preds: dict[str, np.ndarray]) -> dict[str, float]:
    """preds maps name -> predicted 1-day return. Returns MAE% per predictor plus skill stats."""
    actual_close = close * (1.0 + actual_r)
    out: dict[str, float] = {"n_test": int(len(actual_r))}
    for name, pr in preds.items():
        out[f"mae_pct_{name}"] = mae_pct(actual_close, close * (1.0 + pr))
        out[f"mae_usd_{name}"] = float(np.mean(np.abs(actual_close - close * (1.0 + pr))))
    nz = actual_r != 0
    m = preds["model"]
    out["dir_acc_model"] = float(np.mean(np.sign(m[nz]) == np.sign(actual_r[nz])))
    out["up_day_share"] = float(np.mean(actual_r[nz] > 0))
    out["holdout_mean_return"] = float(np.mean(actual_r))
    out["pred_mean_return"] = float(np.mean(m))
    out["pred_abs_mean"] = float(np.mean(np.abs(m)))
    out["actual_abs_mean"] = float(np.mean(np.abs(actual_r)))
    out["ic_pearson"] = float(np.corrcoef(m, actual_r)[0, 1]) if np.std(m) > 0 else 0.0
    out["ic_spearman"] = float(stats.spearmanr(m, actual_r).statistic) if np.std(m) > 0 else 0.0
    # Out-of-sample R^2 on returns relative to each baseline (Campbell-Thompson style).
    sse_m = np.sum((actual_r - m) ** 2)
    out["r2_oos_vs_naive"] = float(1 - sse_m / np.sum(actual_r ** 2))
    if "train_mean" in preds:
        out["r2_oos_vs_mean"] = float(1 - sse_m / np.sum((actual_r - preds["train_mean"]) ** 2))
    # Diebold-Mariano style paired t-stat on absolute price errors (1-step => no HAC lag needed).
    for base in ("naive", "train_mean"):
        if base in preds:
            d = np.abs(actual_close - close * (1 + m)) - np.abs(actual_close - close * (1 + preds[base]))
            out[f"dm_t_vs_{base}"] = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
    return out


def one_split(feat: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, cols: list[str]):
    tr, te = feat.iloc[train_idx], feat.iloc[test_idx]
    model = fit_rf(tr[cols], tr["target_return_1"])
    pred = np.asarray(model.predict(te[cols]), dtype=float)
    return model, tr, te, pred


def improvement(row: dict, base: str) -> float:
    return (row[f"mae_pct_{base}"] - row["mae_pct_model"]) / row[f"mae_pct_{base}"] * 100.0


# --------------------------------------------------------------------------- experiments


def exp_api_holdout(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("\n[1] API-equivalent holdout (last 1,200 rows, last 20% held out)")
    rows, models = [], {}
    for sym, full in prices.items():
        raw = full.tail(API_ROWS).reset_index(drop=True)
        feat, _ = build_training_features(raw)
        n = len(feat)
        split = int(np.floor(n * (1 - TEST_FRAC)))  # matches train_test_split(test_size=0.2, shuffle=False)
        model, tr, te, pred = one_split(feat, np.arange(split), np.arange(split, n), FEATURE_COLS)
        mu = float(tr["target_return_1"].mean())
        actual = te["target_return_1"].to_numpy(float)
        res = score(te["close"].to_numpy(float), actual,
                    {"model": pred, "naive": np.zeros_like(actual), "train_mean": np.full_like(actual, mu)})
        res.update(symbol=sym, train_start=str(tr["date"].iloc[0].date()), test_start=str(te["date"].iloc[0].date()),
                   test_end=str(te["date"].iloc[-1].date()), n_train=len(tr), train_mean_return=mu)
        res["impr_vs_naive_pct"] = improvement(res, "naive")
        res["impr_vs_mean_pct"] = improvement(res, "train_mean")
        top = np.argsort(model.feature_importances_)[::-1][:5]
        res["top_features"] = ",".join(FEATURE_COLS[i] for i in top)
        models[sym] = (model, raw, feat, split)
        rows.append(res)
        print(f"  {sym:6s} model {res['mae_pct_model']:.3f}%  naive {res['mae_pct_naive']:.3f}%  "
              f"mean {res['mae_pct_train_mean']:.3f}%  vs naive {res['impr_vs_naive_pct']:+.2f}%  "
              f"vs mean {res['impr_vs_mean_pct']:+.2f}%  dir {res['dir_acc_model']:.3f}")
    exp_api_holdout.models = models
    return pd.DataFrame(rows)


def exp_walk_forward(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("\n[2] Walk-forward: sliding 1,200-row windows, non-overlapping 20% test blocks over ~10y")
    rows = []
    for sym, full in prices.items():
        feat, _ = build_training_features(full)
        n = len(feat)
        win = API_ROWS - 50  # ~rows left after the 50-day MA warmup in the API
        test_len = int(win * TEST_FRAC)
        train_len = win - test_len
        start = n - win
        folds = []
        while start >= 0:
            folds.append(start)
            start -= test_len
        for k, s in enumerate(sorted(folds)):
            tr_idx = np.arange(s, s + train_len)
            te_idx = np.arange(s + train_len, s + win)
            _, tr, te, pred = one_split(feat, tr_idx, te_idx, FEATURE_COLS)
            mu = float(tr["target_return_1"].mean())
            actual = te["target_return_1"].to_numpy(float)
            res = score(te["close"].to_numpy(float), actual,
                        {"model": pred, "naive": np.zeros_like(actual), "train_mean": np.full_like(actual, mu)})
            res.update(symbol=sym, fold=k, test_start=str(te["date"].iloc[0].date()),
                       test_end=str(te["date"].iloc[-1].date()), train_mean_return=mu)
            res["impr_vs_naive_pct"] = improvement(res, "naive")
            res["impr_vs_mean_pct"] = improvement(res, "train_mean")
            rows.append(res)
        sub = pd.DataFrame([r for r in rows if r["symbol"] == sym])
        print(f"  {sym:6s} folds={len(sub)}  mean impr vs naive {sub['impr_vs_naive_pct'].mean():+.2f}%  "
              f"vs mean {sub['impr_vs_mean_pct'].mean():+.2f}%")
    return pd.DataFrame(rows)


def exp_horizon(prices: dict[str, pd.DataFrame], horizon: int = 5, stride: int = 5) -> pd.DataFrame:
    print(f"\n[3] Recursive {horizon}-day rollout via forecast_next_days (API holdout, every {stride}th day)")
    rows = []
    for sym, (model, raw, feat, split) in exp_api_holdout.models.items():
        model.set_params(n_jobs=1)
        mu = float(feat["target_return_1"].iloc[:split].mean())
        date_to_pos = {d: i for i, d in enumerate(raw["date"])}
        err = {"model": [], "naive": [], "train_mean": []}
        err1 = {"model": [], "naive": []}
        for i in range(split, len(feat), stride):
            pos = date_to_pos[feat["date"].iloc[i]]
            if pos + horizon >= len(raw):
                break
            hist = raw.iloc[: pos + 1]
            preds, _ = forecast_next_days(hist, model, FEATURE_COLS, horizon)
            last = float(hist["close"].iloc[-1])
            actual_h = float(raw["close"].iloc[pos + horizon])
            actual_1 = float(raw["close"].iloc[pos + 1])
            err["model"].append(abs(actual_h - preds[-1]) / actual_h)
            err["naive"].append(abs(actual_h - last) / actual_h)
            err["train_mean"].append(abs(actual_h - last * (1 + mu) ** horizon) / actual_h)
            err1["model"].append(abs(actual_1 - preds[0]) / actual_1)
            err1["naive"].append(abs(actual_1 - last) / actual_1)
        model.set_params(n_jobs=RF_PARAMS["n_jobs"])
        r = {f"mae_pct_{k}": float(np.mean(v) * 100) for k, v in err.items()}
        r.update(symbol=sym, n_origins=len(err["model"]),
                 mae_pct_1d_model_rollout=float(np.mean(err1["model"]) * 100),
                 mae_pct_1d_naive_rollout=float(np.mean(err1["naive"]) * 100))
        r["impr_vs_naive_pct"] = improvement(r, "naive")
        r["impr_vs_mean_pct"] = improvement(r, "train_mean")
        rows.append(r)
        print(f"  {sym:6s} {horizon}d model {r['mae_pct_model']:.3f}%  naive {r['mae_pct_naive']:.3f}%  "
              f"vs naive {r['impr_vs_naive_pct']:+.2f}%  vs mean {r['impr_vs_mean_pct']:+.2f}%")
    return pd.DataFrame(rows)


def load_sentiment_for(sym: str, sent: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    """Align news dates to the next trading day (weekend/holiday news -> next session)."""
    s = sent[sent["stock"] == sym].sort_values("date")
    td = pd.DataFrame({"trade_date": trading_dates.sort_values().values})
    m = pd.merge_asof(s, td, left_on="date", right_on="trade_date", direction="forward").dropna(subset=["trade_date"])
    agg = m.groupby("trade_date").agg(
        daily_sentiment=("daily_sentiment", "mean"),
        sentiment_3d_avg=("sentiment_3d_avg", "last"),
        sentiment_momentum=("sentiment_momentum", "last"),
    )
    return agg


def exp_sentiment(refresh: bool, n_tickers: int) -> tuple[pd.DataFrame, dict]:
    print("\n[4] Sentiment test on historical sentiment file (2009-2020)")
    sent = pd.read_csv(SENTIMENT_PATH, parse_dates=["date"])
    ranking = (sent.assign(nz=sent["daily_sentiment"] != 0).groupby("stock")["nz"].sum()
               .sort_values(ascending=False))
    panels = []
    for sym in ranking.index:
        if len(panels) >= n_tickers:
            break
        df = fetch_yahoo(sym, "2009-01-01", "2020-07-01", refresh)
        if df is None or len(df) < 1500:
            continue
        feat, _ = build_training_features(df)
        agg = load_sentiment_for(sym, sent, feat["date"])
        feat = feat.merge(agg, left_on="date", right_index=True, how="left")
        feat[SENTIMENT_COLS] = feat[SENTIMENT_COLS].fillna(0.0)
        vol20 = df.set_index("date")["volume"].rolling(20).mean()
        feat["volume_ratio"] = (feat["volume"] / feat["date"].map(vol20)).clip(0.25, 3.0).fillna(1.0)
        feat["sent_x_vr"] = feat["daily_sentiment"].clip(-1, 1) * feat["volume_ratio"]
        feat["symbol"] = sym
        panels.append(feat)
    print(f"  tickers with price data: {[p['symbol'].iloc[0] for p in panels]}")

    # Pooled tuned coefficient, fit only on training portions (target demeaned per ticker).
    xs, ys = [], []
    for p in panels:
        split = int(np.floor(len(p) * (1 - TEST_FRAC)))
        tr = p.iloc[:split]
        xs.append(tr["sent_x_vr"].to_numpy())
        ys.append((tr["target_return_1"] - tr["target_return_1"].mean()).to_numpy())
    x_all, y_all = np.concatenate(xs), np.concatenate(ys)
    tuned_coef = float(np.dot(x_all, y_all) / np.dot(x_all, x_all))
    slope_res = stats.linregress(x_all, y_all)

    all_sent = pd.concat(panels)
    has = all_sent["daily_sentiment"] != 0
    ic_next = stats.spearmanr(all_sent.loc[has, "daily_sentiment"], all_sent.loc[has, "target_return_1"])
    ic_same = stats.spearmanr(all_sent.loc[has, "daily_sentiment"], all_sent.loc[has, "return_1d"])

    rows = []
    for p in panels:
        sym = p["symbol"].iloc[0]
        n = len(p)
        split = int(np.floor(n * (1 - TEST_FRAC)))
        tr_idx, te_idx = np.arange(split), np.arange(split, n)
        _, tr, te, pred_price = one_split(p, tr_idx, te_idx, FEATURE_COLS)
        _, _, _, pred_sent = one_split(p, tr_idx, te_idx, FEATURE_COLS + SENTIMENT_COLS + ["sent_x_vr"])
        sx = te["sent_x_vr"].to_numpy(float)
        actual = te["target_return_1"].to_numpy(float)
        close = te["close"].to_numpy(float)
        preds = {
            "model": pred_price,
            "naive": np.zeros_like(actual),
            "train_mean": np.full_like(actual, float(tr["target_return_1"].mean())),
            "overlay_fixed": np.clip(pred_price + SENTIMENT_RETURN_COEF * sx, -0.3, 0.3),
            "overlay_tuned": np.clip(pred_price + tuned_coef * sx, -0.3, 0.3),
            "sent_features": pred_sent,
        }
        res = score(close, actual, preds)
        mask = te["daily_sentiment"].to_numpy() != 0
        res.update(symbol=sym, test_start=str(te["date"].iloc[0].date()), test_end=str(te["date"].iloc[-1].date()),
                   sentiment_day_share_test=float(mask.mean()))
        actual_close = close * (1 + actual)
        for name in ("model", "overlay_fixed", "overlay_tuned", "sent_features", "naive"):
            res[f"mae_pct_{name}_on_news_days"] = mae_pct(actual_close[mask], close[mask] * (1 + preds[name][mask]))
            res[f"dir_acc_{name}_on_news_days"] = float(
                np.mean(np.sign(preds[name][mask]) == np.sign(actual[mask]))) if name != "naive" else np.nan
        rows.append(res)
        print(f"  {sym:6s} price-only {res['mae_pct_model']:.4f}%  +fixed overlay {res['mae_pct_overlay_fixed']:.4f}%  "
              f"+tuned {res['mae_pct_overlay_tuned']:.4f}%  +sent features {res['mae_pct_sent_features']:.4f}%  "
              f"news days {mask.mean():.0%}")
    meta = {
        "tuned_coef": tuned_coef,
        "tuned_coef_pvalue": float(slope_res.pvalue),
        "production_coef": SENTIMENT_RETURN_COEF,
        "rank_ic_sentiment_vs_next_day_return": float(ic_next.statistic),
        "rank_ic_sentiment_vs_next_day_pvalue": float(ic_next.pvalue),
        "rank_ic_sentiment_vs_same_day_return": float(ic_same.statistic),
        "rank_ic_sentiment_vs_same_day_pvalue": float(ic_same.pvalue),
        "n_news_days": int(has.sum()),
    }
    print(f"  tuned coef {tuned_coef:.5f} (p={slope_res.pvalue:.3f}) vs production {SENTIMENT_RETURN_COEF}")
    print(f"  rank IC sentiment->next-day return {ic_next.statistic:+.4f} (p={ic_next.pvalue:.3f}); "
          f"same-day {ic_same.statistic:+.4f} (p={ic_same.pvalue:.3g})")
    return pd.DataFrame(rows), meta


# --------------------------------------------------------------------------- summary


def summarize_rel(df: pd.DataFrame, label: str) -> dict:
    out = {"label": label, "n_rows": len(df)}
    for base in ("naive", "mean"):
        col = f"impr_vs_{base}_pct"
        if col not in df:
            continue
        v = df[col].to_numpy()
        out[f"mean_impr_vs_{base}_pct"] = float(v.mean())
        out[f"median_impr_vs_{base}_pct"] = float(np.median(v))
        out[f"wins_vs_{base}"] = int((v > 0).sum())
        out[f"ttest_p_vs_{base}"] = float(stats.ttest_1samp(v, 0).pvalue) if len(v) > 2 else np.nan
        out[f"sign_test_p_vs_{base}"] = float(stats.binomtest(int((v > 0).sum()), len(v)).pvalue)
    for c in ("mae_pct_model", "mae_pct_naive", "mae_pct_train_mean", "dir_acc_model", "up_day_share",
              "ic_spearman", "r2_oos_vs_naive", "r2_oos_vs_mean", "pred_abs_mean", "actual_abs_mean"):
        if c in df:
            out[f"avg_{c}"] = float(df[c].mean())
    if "holdout_mean_return" in df and "impr_vs_naive_pct" in df:
        rho = stats.spearmanr(df["holdout_mean_return"], df["impr_vs_naive_pct"])
        out["drift_corr_spearman"] = float(rho.statistic)
        out["drift_corr_p"] = float(rho.pvalue)
        rho2 = stats.spearmanr(df["holdout_mean_return"], df["impr_vs_mean_pct"])
        out["drift_corr_vs_mean_spearman"] = float(rho2.statistic)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["api_holdout", "walk_forward", "horizon_5d", "sentiment"])
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--sentiment-tickers", type=int, default=20)
    args = ap.parse_args(argv)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {"tickers": TICKERS, "rf_params": RF_PARAMS, "n_features": len(FEATURE_COLS)}

    need_prices = {"api_holdout", "walk_forward", "horizon_5d"} & set(args.only)
    prices = {}
    if need_prices:
        print("Fetching ~10y daily history from Yahoo Finance (cached under benchmarks/price_cache)")
        for sym in TICKERS:
            df = fetch_yahoo(sym, None, None, args.refresh)
            if df is not None:
                prices[sym] = df
        summary["price_range"] = {s: [str(d["date"].iloc[0].date()), str(d["date"].iloc[-1].date())]
                                  for s, d in prices.items()}

    if "api_holdout" in args.only or "horizon_5d" in args.only:
        api = exp_api_holdout(prices)
        api.to_csv(RESULTS_DIR / "api_holdout.csv", index=False)
        summary["api_holdout"] = summarize_rel(api, "API-equivalent single holdout")
    if "walk_forward" in args.only:
        wf = exp_walk_forward(prices)
        wf.to_csv(RESULTS_DIR / "walk_forward.csv", index=False)
        summary["walk_forward"] = summarize_rel(wf, "walk-forward folds (ticker x fold)")
        per_ticker = wf.groupby("symbol")[["impr_vs_naive_pct", "impr_vs_mean_pct"]].mean()
        summary["walk_forward"]["tickers_beating_naive_on_avg"] = int((per_ticker["impr_vs_naive_pct"] > 0).sum())
        summary["walk_forward"]["tickers_beating_mean_on_avg"] = int((per_ticker["impr_vs_mean_pct"] > 0).sum())
        summary["walk_forward"]["test_period"] = [wf["test_start"].min(), wf["test_end"].max()]
    if "horizon_5d" in args.only:
        hz = exp_horizon(prices)
        hz.to_csv(RESULTS_DIR / "horizon_5d.csv", index=False)
        summary["horizon_5d"] = summarize_rel(hz, "5-day recursive rollout")
    if "sentiment" in args.only:
        se, meta = exp_sentiment(args.refresh, args.sentiment_tickers)
        se.to_csv(RESULTS_DIR / "sentiment.csv", index=False)
        s = {"n_tickers": len(se), **meta}
        for name in ("model", "overlay_fixed", "overlay_tuned", "sent_features", "naive", "train_mean"):
            s[f"avg_mae_pct_{name}"] = float(se[f"mae_pct_{name}"].mean())
        for name in ("model", "overlay_fixed", "overlay_tuned", "sent_features", "naive"):
            s[f"avg_mae_pct_{name}_on_news_days"] = float(se[f"mae_pct_{name}_on_news_days"].mean())
        for name in ("overlay_fixed", "overlay_tuned", "sent_features"):
            d = (se["mae_pct_model"] - se[f"mae_pct_{name}"]) / se["mae_pct_model"] * 100
            s[f"impr_{name}_vs_price_only_pct"] = float(d.mean())
            s[f"wins_{name}_vs_price_only"] = int((d > 0).sum())
            s[f"ttest_p_{name}_vs_price_only"] = float(stats.ttest_1samp(d, 0).pvalue)
        s["avg_dir_acc_model_on_news_days"] = float(se["dir_acc_model_on_news_days"].mean())
        s["avg_dir_acc_sent_features_on_news_days"] = float(se["dir_acc_sent_features_on_news_days"].mean())
        s["avg_dir_acc_overlay_tuned_on_news_days"] = float(se["dir_acc_overlay_tuned_on_news_days"].mean())
        summary["sentiment"] = s

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\nSummary written to", RESULTS_DIR / "summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("price_range", "tickers")}, indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv[1:])
