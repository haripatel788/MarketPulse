"""
Extended experiments for the MarketPulse forecaster (run after benchmarks.run_benchmark).

  variants     : walk-forward (120 windows) for model variants, shrinkage, shuffled-target check,
                 volatility model, forward band coverage, regime breakdown, trading simulation
  window       : training-window length (250 / 500 / 1000 / 1900 days), fixed 2-year test, quarterly refits
  horizons     : recursive rollout to 30 days, error and +/-MAE band coverage at 1/5/10/20/30 days
  importance   : permutation importance on the API holdout test sets
  sentiment2   : lagged sentiment, extreme-sentiment days, event study
  sp500        : API holdout on S&P 500 constituents (production model and no-price-level variant)
  speed        : data fetch, training, rollout, and end-to-end run_prediction latency; model size

Usage:
  .venv/bin/python -m benchmarks.extended                 # everything
  .venv/bin/python -m benchmarks.extended --only window horizons
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import app.main as api
from benchmarks import legacy_rf
from benchmarks.legacy_rf import FEATURE_COLS, RF_PARAMS, build_training_features, forecast_next_days
from benchmarks.run_benchmark import (
    API_ROWS,
    RESULTS_DIR,
    TEST_FRAC,
    TICKERS,
    build_sentiment_panels,
    exp_api_holdout,
    fetch_yahoo,
    fit_rf,
    mae_pct,
)

PRICE_LEVEL_COLS = ["close", "ma_5", "ma_10", "ma_20", "ma_50",
                    "close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5", "close_lag_10"]
VOLUME_LEVEL_COLS = ["volume", "volume_lag_1", "volume_lag_5", "volume_lag_10"]
NO_PRICE_COLS = [c for c in FEATURE_COLS if c not in PRICE_LEVEL_COLS]
STATIONARY_COLS = [c for c in NO_PRICE_COLS if c not in VOLUME_LEVEL_COLS]
SHRINK = (0.1, 0.25, 0.5)
COST_PER_TRADE = 0.0005  # 5 bps per unit change in position
SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"


def load_prices(symbols: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for s in symbols:
        df = fetch_yahoo(s, None, None, refresh=False)
        if df is not None:
            out[s] = df
    return out


def add_extra(feat: pd.DataFrame) -> pd.DataFrame:
    feat = feat.copy()
    feat["abs_target"] = feat["target_return_1"].abs()
    feat["hist_vol20"] = feat["return_1d"].abs().rolling(20).mean()
    return feat


def wf_folds(n: int) -> list[tuple[np.ndarray, np.ndarray, int]]:
    win = API_ROWS - 50
    test_len = int(win * TEST_FRAC)
    train_len = win - test_len
    starts, s = [], n - win
    while s >= 0:
        starts.append(s)
        s -= test_len
    return [(np.arange(s, s + train_len), np.arange(s + train_len, s + win), s) for s in sorted(starts)]


def pct_impr(model_err: float, base_err: float) -> float:
    return (base_err - model_err) / base_err * 100.0


def ttest_p(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(stats.ttest_1samp(v, 0).pvalue) if len(v) > 2 else float("nan")


# --------------------------------------------------------------------------- 1. variants (walk-forward)


def exp_variants(prices: dict[str, pd.DataFrame]) -> dict:
    print("\n[variants] walk-forward over 120 windows")
    fold_rows, day_rows, cover_rows = [], [], []
    rng = np.random.default_rng(42)
    for sym, full in prices.items():
        feat, _ = build_training_features(full)
        feat = add_extra(feat)
        folds = wf_folds(len(feat))
        for k, (tr_idx, te_idx, s) in enumerate(folds):
            tr, te = feat.iloc[tr_idx], feat.iloc[te_idx]
            ytr = tr["target_return_1"]
            actual = te["target_return_1"].to_numpy(float)
            close = te["close"].to_numpy(float)
            actual_close = close * (1 + actual)
            mu = float(ytr.mean())

            preds: dict[str, np.ndarray] = {}
            rf_full = fit_rf(tr[FEATURE_COLS], ytr)
            preds["rf_27_production"] = rf_full.predict(te[FEATURE_COLS])
            preds["rf_17_no_price_levels"] = fit_rf(tr[NO_PRICE_COLS], ytr).predict(te[NO_PRICE_COLS])
            preds["rf_13_stationary"] = fit_rf(tr[STATIONARY_COLS], ytr).predict(te[STATIONARY_COLS])
            ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(tr[STATIONARY_COLS], ytr)
            preds["ridge_13_stationary"] = ridge.predict(te[STATIONARY_COLS])
            hgb = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, random_state=42)
            preds["gboost_27"] = hgb.fit(tr[FEATURE_COLS], ytr).predict(te[FEATURE_COLS])
            preds["rf_27_shuffled_target"] = fit_rf(tr[FEATURE_COLS], rng.permutation(ytr.to_numpy())).predict(
                te[FEATURE_COLS])
            for a in SHRINK:
                preds[f"rf_27_shrink_{a}"] = a * preds["rf_27_production"]
            preds["naive"] = np.zeros_like(actual)
            preds["train_mean"] = np.full_like(actual, mu)

            row = {"symbol": sym, "fold": k, "test_start": str(te["date"].iloc[0].date())}
            for name, p in preds.items():
                row[f"mae_pct_{name}"] = mae_pct(actual_close, close * (1 + p))
                nz = actual != 0
                row[f"dir_{name}"] = float(np.mean(np.sign(p[nz]) == np.sign(actual[nz])))

            # Volatility model: predict |next-day return|.
            vol_rf = fit_rf(tr[FEATURE_COLS], tr["abs_target"])
            vol_pred = vol_rf.predict(te[FEATURE_COLS])
            abs_act = te["abs_target"].to_numpy(float)
            hv = te["hist_vol20"].to_numpy(float)
            row["vol_mae_model"] = float(np.mean(np.abs(abs_act - vol_pred)))
            row["vol_mae_hist20"] = float(np.mean(np.abs(abs_act - hv)))
            row["vol_mae_train_mean"] = float(np.mean(np.abs(abs_act - tr["abs_target"].mean())))
            row["vol_corr_model"] = float(stats.spearmanr(vol_pred, abs_act).statistic)
            row["vol_corr_hist20"] = float(stats.spearmanr(hv, abs_act).statistic)
            fold_rows.append(row)

            usd_err = np.abs(actual_close - close * (1 + preds["rf_27_production"]))
            band_usd = float(usd_err.mean())
            day_rows.append(pd.DataFrame({
                "symbol": sym, "date": te["date"].to_numpy(), "actual": actual,
                "pred_prod": preds["rf_27_production"], "pred_noprice": preds["rf_17_no_price_levels"],
                "err_model": np.abs(actual_close - close * (1 + preds["rf_27_production"])) / actual_close,
                "err_naive": np.abs(actual_close - close) / actual_close,
                "hist_vol20": hv,
            }))

            # Forward band coverage: the API trains on the whole window, reports the holdout MAE as +/- band,
            # and serves forecasts for the days after the window.
            nxt = np.arange(s + len(tr_idx) + len(te_idx), min(s + len(tr_idx) + 2 * len(te_idx), len(feat)))
            if len(nxt) > 20:
                win_df = feat.iloc[np.concatenate([tr_idx, te_idx])]
                final = fit_rf(win_df[FEATURE_COLS], win_df["target_return_1"])
                fx = feat.iloc[nxt]
                fc = fx["close"].to_numpy(float)
                fa = fc * (1 + fx["target_return_1"].to_numpy(float))
                fp = fc * (1 + final.predict(fx[FEATURE_COLS]))
                ratio = np.abs(fa - fp) / band_usd
                cover_rows.append(pd.DataFrame({"symbol": sym, "ratio": ratio}))
        print(f"  {sym:6s} done ({len(folds)} folds)")

    folds_df = pd.DataFrame(fold_rows)
    days = pd.concat(day_rows, ignore_index=True)
    cover = pd.concat(cover_rows, ignore_index=True)
    folds_df.to_csv(RESULTS_DIR / "ext_variants_folds.csv", index=False)
    days.to_csv(RESULTS_DIR / "ext_walk_forward_days.csv", index=False)

    out: dict = {"n_windows": len(folds_df), "n_test_days": int(len(days))}
    names = [c[len("mae_pct_"):] for c in folds_df.columns if c.startswith("mae_pct_")]
    variants = {}
    for name in names:
        if name in ("naive", "train_mean"):
            continue
        impr = (folds_df["mae_pct_naive"] - folds_df[f"mae_pct_{name}"]) / folds_df["mae_pct_naive"] * 100
        impr_m = (folds_df["mae_pct_train_mean"] - folds_df[f"mae_pct_{name}"]) / folds_df["mae_pct_train_mean"] * 100
        variants[name] = {
            "avg_mae_pct": float(folds_df[f"mae_pct_{name}"].mean()),
            "mean_impr_vs_naive_pct": float(impr.mean()),
            "median_impr_vs_naive_pct": float(impr.median()),
            "wins_vs_naive": int((impr > 0).sum()),
            "p_vs_naive": ttest_p(impr),
            "mean_impr_vs_mean_pct": float(impr_m.mean()),
            "wins_vs_mean": int((impr_m > 0).sum()),
            "dir_acc": float(folds_df[f"dir_{name}"].mean()),
        }
    variants["naive"] = {"avg_mae_pct": float(folds_df["mae_pct_naive"].mean())}
    variants["train_mean"] = {"avg_mae_pct": float(folds_df["mae_pct_train_mean"].mean()),
                              "dir_acc": float(folds_df["dir_train_mean"].mean())}
    out["variants"] = variants

    v_impr_h = (folds_df["vol_mae_hist20"] - folds_df["vol_mae_model"]) / folds_df["vol_mae_hist20"] * 100
    v_impr_m = (folds_df["vol_mae_train_mean"] - folds_df["vol_mae_model"]) / folds_df["vol_mae_train_mean"] * 100
    out["volatility"] = {
        "avg_mae_model": float(folds_df["vol_mae_model"].mean()),
        "avg_mae_hist20": float(folds_df["vol_mae_hist20"].mean()),
        "avg_mae_train_mean": float(folds_df["vol_mae_train_mean"].mean()),
        "mean_impr_vs_hist20_pct": float(v_impr_h.mean()), "wins_vs_hist20": int((v_impr_h > 0).sum()),
        "p_vs_hist20": ttest_p(v_impr_h),
        "mean_impr_vs_train_mean_pct": float(v_impr_m.mean()), "wins_vs_train_mean": int((v_impr_m > 0).sum()),
        "avg_rank_corr_model": float(folds_df["vol_corr_model"].mean()),
        "avg_rank_corr_hist20": float(folds_df["vol_corr_hist20"].mean()),
    }

    r = cover["ratio"].to_numpy()
    out["band_coverage_1d_forward"] = {
        "n_days": int(len(r)),
        "coverage_pct": float(np.mean(r <= 1.0) * 100),
        "band_multiple_for_80pct": float(np.quantile(r, 0.80)),
        "band_multiple_for_90pct": float(np.quantile(r, 0.90)),
        "band_multiple_for_95pct": float(np.quantile(r, 0.95)),
    }

    # Regimes
    def regime(mask) -> dict:
        d = days[mask]
        return {"n_days": int(len(d)), "mae_pct_model": float(d["err_model"].mean() * 100),
                "mae_pct_naive": float(d["err_naive"].mean() * 100),
                "impr_vs_naive_pct": pct_impr(d["err_model"].mean(), d["err_naive"].mean())}

    days["year"] = pd.to_datetime(days["date"]).dt.year
    med = days.groupby("symbol")["hist_vol20"].transform("median")
    reg = {"high_volatility": regime(days["hist_vol20"] > med), "low_volatility": regime(days["hist_vol20"] <= med)}
    dt = pd.to_datetime(days["date"])
    reg["bear_2022"] = regime((dt >= "2022-01-03") & (dt <= "2022-10-12"))
    reg["outside_bear_2022"] = regime(~((dt >= "2022-01-03") & (dt <= "2022-10-12")))
    for y in sorted(days["year"].unique()):
        reg[f"year_{y}"] = regime(days["year"] == y)
    big = days["actual"].abs() > 2 * days["hist_vol20"]
    reg["big_move_days"] = regime(big)
    reg["normal_days"] = regime(~big)
    out["regimes"] = reg

    out["trading"] = trading_sim(days)
    print(json.dumps({k: out[k] for k in ("volatility", "band_coverage_1d_forward")}, indent=2))
    return out


def trading_sim(days: pd.DataFrame) -> dict:
    res = {}

    def stats_of(daily: pd.Series, exposure: float | None = None, trades_per_year: float | None = None) -> dict:
        eq = (1 + daily).cumprod()
        years = len(daily) / 252
        ann_ret = float(eq.iloc[-1] ** (1 / years) - 1)
        vol = float(daily.std() * np.sqrt(252))
        dd = float((eq / eq.cummax() - 1).min())
        d = {"annual_return_pct": ann_ret * 100, "annual_vol_pct": vol * 100,
             "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)), "max_drawdown_pct": dd * 100,
             "total_return_pct": float(eq.iloc[-1] - 1) * 100}
        if exposure is not None:
            d["time_in_market_pct"] = exposure * 100
        if trades_per_year is not None:
            d["position_changes_per_year_per_stock"] = trades_per_year
        return d

    days = days.sort_values(["symbol", "date"])
    port = {}
    for label, col, kind in [("long_flat_production", "pred_prod", "lf"), ("long_short_production", "pred_prod", "ls"),
                             ("long_flat_no_price_levels", "pred_noprice", "lf")]:
        pos = (days[col] > 0).astype(float) if kind == "lf" else np.sign(days[col])
        change = pos.groupby(days["symbol"]).diff().abs().fillna(pos.abs())
        ret_gross = pos * days["actual"]
        ret_net = ret_gross - change * COST_PER_TRADE
        g = pd.DataFrame({"date": days["date"], "gross": ret_gross, "net": ret_net, "pos": pos, "chg": change})
        daily = g.groupby("date")[["gross", "net"]].mean()
        n_years = days.groupby("symbol").size().mean() / 252
        tpy = float(g.groupby(days["symbol"])["chg"].sum().mean() / n_years)
        port[label] = {"before_costs": stats_of(daily["gross"], float(g["pos"].abs().mean()), tpy),
                       "after_5bp_costs": stats_of(daily["net"])}
    bh = days.groupby("date")["actual"].mean()
    port["buy_and_hold"] = stats_of(bh, 1.0, 0.0)

    # Control: random long/flat positions with the same time-in-market as the production model.
    rng = np.random.default_rng(42)
    exposure = float((days["pred_prod"] > 0).mean())
    codes, _ = pd.factorize(days["date"])
    counts = np.bincount(codes)
    actual = days["actual"].to_numpy()
    sharpes = []
    for _ in range(500):
        pos = rng.random(len(actual)) < exposure
        daily = np.bincount(codes, weights=pos * actual) / counts
        sharpes.append(daily.mean() / daily.std() * np.sqrt(252))
    sharpes = np.asarray(sharpes)
    model_sharpe = port["long_flat_production"]["before_costs"]["sharpe"]
    port["random_long_flat_control"] = {
        "n_simulations": 500, "time_in_market_pct": exposure * 100,
        "median_sharpe": float(np.median(sharpes)), "p95_sharpe": float(np.quantile(sharpes, 0.95)),
        "share_of_random_runs_beating_model_pct": float(np.mean(sharpes >= model_sharpe) * 100),
    }
    res["equal_weight_20_stocks"] = port
    res["period"] = [str(pd.Timestamp(days["date"].min()).date()), str(pd.Timestamp(days["date"].max()).date())]
    return res


# --------------------------------------------------------------------------- 2. training window length


def exp_window(prices: dict[str, pd.DataFrame]) -> dict:
    print("\n[window] training-window length, fixed last-504-day test, refit every 63 days")
    windows, test_days, step = (250, 500, 1000, 1900), 504, 63
    rows = []
    for sym, full in prices.items():
        feat, _ = build_training_features(full)
        n = len(feat)
        t0 = n - test_days
        for w in windows:
            errs_m, errs_n, errs_mu = [], [], []
            for r in range(t0, n, step):
                tr = feat.iloc[r - w:r]
                te = feat.iloc[r:min(r + step, n)]
                model = fit_rf(tr[FEATURE_COLS], tr["target_return_1"])
                c = te["close"].to_numpy(float)
                a = c * (1 + te["target_return_1"].to_numpy(float))
                errs_m.append(np.abs(a - c * (1 + model.predict(te[FEATURE_COLS]))) / a)
                errs_n.append(np.abs(a - c) / a)
                errs_mu.append(np.abs(a - c * (1 + tr["target_return_1"].mean())) / a)
            m, nv, mu = (float(np.concatenate(e).mean() * 100) for e in (errs_m, errs_n, errs_mu))
            rows.append({"symbol": sym, "window": w, "mae_pct_model": m, "mae_pct_naive": nv, "mae_pct_mean": mu,
                         "impr_vs_naive_pct": pct_impr(m, nv), "impr_vs_mean_pct": pct_impr(m, mu)})
        print(f"  {sym:6s} " + "  ".join(f"{r['window']}d {r['impr_vs_naive_pct']:+.2f}%" for r in rows[-len(windows):]))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "ext_window.csv", index=False)
    feat_dates = build_training_features(next(iter(prices.values())))[0]["date"]
    out = {"test_period": [str(feat_dates.iloc[-test_days].date()), str(feat_dates.iloc[-1].date())]}
    for w, g in df.groupby("window"):
        out[f"{w}_days"] = {"avg_mae_pct_model": float(g["mae_pct_model"].mean()),
                            "avg_mae_pct_naive": float(g["mae_pct_naive"].mean()),
                            "mean_impr_vs_naive_pct": float(g["impr_vs_naive_pct"].mean()),
                            "wins_vs_naive": int((g["impr_vs_naive_pct"] > 0).sum()),
                            "mean_impr_vs_mean_pct": float(g["impr_vs_mean_pct"].mean())}
    return out


# --------------------------------------------------------------------------- 3. long horizons + band coverage


def exp_horizons(prices: dict[str, pd.DataFrame], api_df: pd.DataFrame) -> dict:
    print("\n[horizons] recursive rollout to 30 days")
    hs, max_h, stride = (1, 5, 10, 20, 30), 30, 10
    rec = []
    for sym, (model, raw, feat, split) in exp_api_holdout.models.items():
        model.set_params(n_jobs=1)
        mu = float(feat["target_return_1"].iloc[:split].mean())
        band = float(api_df.set_index("symbol").loc[sym, "mae_usd_model"])
        pos_of = {d: i for i, d in enumerate(raw["date"])}
        for i in range(split, len(feat), stride):
            pos = pos_of[feat["date"].iloc[i]]
            if pos + max_h >= len(raw):
                break
            hist = raw.iloc[: pos + 1]
            preds, _ = forecast_next_days(hist, model, FEATURE_COLS, max_h)
            last = float(hist["close"].iloc[-1])
            for h in hs:
                a = float(raw["close"].iloc[pos + h])
                p = preds[h - 1]
                rec.append({"symbol": sym, "h": h, "err_model": abs(a - p) / a, "err_naive": abs(a - last) / a,
                            "err_mean": abs(a - last * (1 + mu) ** h) / a, "in_band": abs(a - p) <= band})
        model.set_params(n_jobs=RF_PARAMS["n_jobs"])
        print(f"  {sym:6s} done")
    df = pd.DataFrame(rec)
    out = {"n_origins": int((df["h"] == 1).sum())}
    for h, g in df.groupby("h"):
        per = g.groupby("symbol")[["err_model", "err_naive", "err_mean"]].mean()
        impr = (per["err_naive"] - per["err_model"]) / per["err_naive"] * 100
        out[f"{h}_day"] = {"mae_pct_model": float(g["err_model"].mean() * 100),
                           "mae_pct_naive": float(g["err_naive"].mean() * 100),
                           "mae_pct_mean": float(g["err_mean"].mean() * 100),
                           "mean_impr_vs_naive_pct": float(impr.mean()),
                           "wins_vs_naive": int((impr > 0).sum()),
                           "band_coverage_pct": float(g["in_band"].mean() * 100)}
    df.to_csv(RESULTS_DIR / "ext_horizons.csv", index=False)
    return out


# --------------------------------------------------------------------------- 4. permutation importance


def exp_importance() -> dict:
    print("\n[importance] permutation importance on API holdout test sets")
    rows = []
    for sym, (model, raw, feat, split) in exp_api_holdout.models.items():
        te = feat.iloc[split:]
        base = float(np.mean(np.abs(te["target_return_1"] - model.predict(te[FEATURE_COLS]))))
        pi = permutation_importance(model, te[FEATURE_COLS], te["target_return_1"],
                                    scoring="neg_mean_absolute_error", n_repeats=10, random_state=42)
        for f, m in zip(FEATURE_COLS, pi.importances_mean):
            rows.append({"symbol": sym, "feature": f, "mae_increase_pct": m / base * 100})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "ext_importance.csv", index=False)
    agg = df.groupby("feature")["mae_increase_pct"].agg(["mean", lambda s: int((s > 0).sum())])
    agg.columns = ["avg_mae_change_when_shuffled_pct", "tickers_where_feature_helps"]
    agg = agg.sort_values("avg_mae_change_when_shuffled_pct", ascending=False)
    return {"n_features_helping_on_avg": int((agg["avg_mae_change_when_shuffled_pct"] > 0).sum()),
            "n_features_hurting_on_avg": int((agg["avg_mae_change_when_shuffled_pct"] < 0).sum()),
            "by_feature": agg.round(4).reset_index().to_dict(orient="records")}


# --------------------------------------------------------------------------- 5. sentiment follow-ups


def exp_sentiment2() -> dict:
    print("\n[sentiment2] lagged sentiment, extreme days, event study")
    panels = build_sentiment_panels(False, 20)
    rows = []
    for p in panels:
        p = p.reset_index(drop=True).copy()
        p["sent_lag1"] = p["daily_sentiment"].shift(1)
        for k in range(1, 6):
            p[f"fwd_{k}"] = p["close"].shift(-k) / p["close"] - 1
        rows.append(p)
    d = pd.concat(rows, ignore_index=True)
    news = d[d["daily_sentiment"] != 0]
    lag = d[(d["sent_lag1"].fillna(0) != 0)]
    out = {"n_news_days": int(len(news))}
    ic_lag = stats.spearmanr(lag["sent_lag1"], lag["target_return_1"])
    out["lagged_rank_ic_next_day"] = {"ic": float(ic_lag.statistic), "p": float(ic_lag.pvalue), "n": int(len(lag))}

    hi, lo = news["daily_sentiment"].quantile(0.9), news["daily_sentiment"].quantile(0.1)
    ext = news[(news["daily_sentiment"] >= hi) | (news["daily_sentiment"] <= lo)]
    sgn = np.sign(ext["daily_sentiment"])
    nz_next, nz_same = ext["target_return_1"] != 0, ext["return_1d"] != 0
    ic_e = stats.spearmanr(ext["daily_sentiment"], ext["target_return_1"])
    out["extreme_days"] = {
        "thresholds": [float(lo), float(hi)], "n_days": int(len(ext)),
        "next_day_rank_ic": float(ic_e.statistic), "next_day_p": float(ic_e.pvalue),
        "next_day_direction_hit_pct": float((sgn[nz_next] == np.sign(ext["target_return_1"][nz_next])).mean() * 100),
        "same_day_direction_hit_pct": float((sgn[nz_same] == np.sign(ext["return_1d"][nz_same])).mean() * 100),
    }

    pos, neg = news[news["daily_sentiment"] >= hi], news[news["daily_sentiment"] <= lo]
    ev = {"n_positive": int(len(pos)), "n_negative": int(len(neg))}
    for label, col in [("day_0", "return_1d")] + [(f"day_+{k}", f"fwd_{k}") for k in range(1, 6)]:
        a, b, u = pos[col].dropna(), neg[col].dropna(), d[col].dropna()
        t = stats.ttest_ind(a, b, equal_var=False)
        ev[label] = {"positive_news_avg_pct": float(a.mean() * 100), "negative_news_avg_pct": float(b.mean() * 100),
                     "all_days_avg_pct": float(u.mean() * 100), "spread_pct": float((a.mean() - b.mean()) * 100),
                     "p": float(t.pvalue)}
    out["event_study"] = ev
    print(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- 6. S&P 500


def load_sp500() -> pd.DataFrame:
    r = requests.get(SP500_URL, timeout=20)
    r.raise_for_status()
    return pd.read_csv(pd.io.common.StringIO(r.text))


def exp_sp500() -> dict:
    print("\n[sp500] API holdout on S&P 500 constituents")
    cons = load_sp500()
    rows, crashes = [], []
    for i, (sym, sector) in enumerate(zip(cons["Symbol"], cons["GICS Sector"])):
        ysym = sym.replace(".", "-")
        full = fetch_yahoo(ysym, None, None, refresh=False)
        if full is None or len(full) < API_ROWS:
            continue
        raw = full.tail(API_ROWS).reset_index(drop=True)
        feat, _ = build_training_features(raw)
        split = int(np.floor(len(feat) * (1 - TEST_FRAC)))
        tr, te = feat.iloc[:split], feat.iloc[split:]
        if not np.isfinite(feat[FEATURE_COLS].to_numpy(float)).all():
            crashes.append(ysym)
            continue
        c = te["close"].to_numpy(float)
        a = c * (1 + te["target_return_1"].to_numpy(float))
        p27 = fit_rf(tr[FEATURE_COLS], tr["target_return_1"]).predict(te[FEATURE_COLS])
        p17 = fit_rf(tr[NO_PRICE_COLS], tr["target_return_1"]).predict(te[NO_PRICE_COLS])
        nv = mae_pct(a, c)
        mu = mae_pct(a, c * (1 + tr["target_return_1"].mean()))
        m27, m17 = mae_pct(a, c * (1 + p27)), mae_pct(a, c * (1 + p17))
        rows.append({"symbol": ysym, "sector": sector, "mae_pct_production": m27, "mae_pct_no_price_levels": m17,
                     "mae_pct_naive": nv, "mae_pct_mean": mu,
                     "impr_production_vs_naive": pct_impr(m27, nv), "impr_no_price_vs_naive": pct_impr(m17, nv),
                     "impr_production_vs_mean": pct_impr(m27, mu), "impr_no_price_vs_mean": pct_impr(m17, mu)})
        if i % 50 == 0:
            print(f"  {i}/{len(cons)}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "ext_sp500.csv", index=False)
    out = {"n_tickers": len(df), "n_test_predictions": int(len(df) * 230),
           "tickers_that_crash_api_training_inf_features": crashes}
    for col in ("impr_production_vs_naive", "impr_no_price_vs_naive", "impr_production_vs_mean", "impr_no_price_vs_mean"):
        v = df[col]
        out[col] = {"mean_pct": float(v.mean()), "median_pct": float(v.median()), "wins": int((v > 0).sum()),
                    "p": ttest_p(v),
                    "sign_test_p": float(stats.binomtest(int((v > 0).sum()), len(v)).pvalue)}
    for c in ("mae_pct_production", "mae_pct_no_price_levels", "mae_pct_naive", "mae_pct_mean"):
        out[f"avg_{c}"] = float(df[c].mean())
    sec = df.groupby("sector").agg(n=("symbol", "size"), impr_production=("impr_production_vs_naive", "mean"),
                                   impr_no_price=("impr_no_price_vs_naive", "mean"),
                                   wins_production=("impr_production_vs_naive", lambda s: int((s > 0).sum())))
    out["by_sector"] = sec.round(3).reset_index().to_dict(orient="records")
    print(json.dumps({k: v for k, v in out.items() if k != "by_sector"}, indent=2))
    return out


# --------------------------------------------------------------------------- 7. speed


def exp_speed() -> dict:
    # Stooq now serves a bot-check page that can stall for minutes, so it is skipped here;
    # timings below cover the Yahoo fallback path the API actually ends up using.
    print("\n[speed] latency (Stooq skipped, live Yahoo fetch; news/LLM sentiment disabled)")
    stooq_off = mock.patch.object(api, "try_fetch_stooq", return_value=None)
    stooq_off.start()
    t_fetch, t_train, t_roll, cold, warm, sizes = [], [], [], [], [], []
    for sym in TICKERS:
        t = time.perf_counter(); raw, _ = api.fetch_history(sym); t_fetch.append(time.perf_counter() - t)
        t = time.perf_counter(); model, *_ = legacy_rf.train_and_metrics(raw); t_train.append(time.perf_counter() - t)
        sizes.append(len(pickle.dumps(model)) / 1e6)
        t = time.perf_counter(); forecast_next_days(raw, model, FEATURE_COLS, 7); t_roll.append(time.perf_counter() - t)
        api._model_cache.clear()
        t = time.perf_counter(); api.run_prediction(sym, 7); cold.append(time.perf_counter() - t)
        for _ in range(3):
            t = time.perf_counter(); api.run_prediction(sym, 7); warm.append(time.perf_counter() - t)
        print(f"  {sym:6s} cold {cold[-1]:.2f}s warm {np.median(warm[-3:]):.2f}s")

    def q(v):
        v = np.asarray(v)
        return {"p50_s": float(np.median(v)), "p95_s": float(np.quantile(v, 0.95)), "mean_s": float(v.mean())}

    stooq_off.stop()
    return {"fetch_history_yahoo": q(t_fetch), "train_eval_plus_final": q(t_train),
            "rollout_7_days": q(t_roll), "run_prediction_cold": q(cold), "run_prediction_warm_cache": q(warm),
            "model_size_mb": {"mean": float(np.mean(sizes)), "max": float(np.max(sizes))},
            "cache_max_models": api.MODEL_CACHE_MAX_KEYS, "n_cpus_used": "all (n_jobs=-1)"}


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*",
                    default=["variants", "window", "horizons", "importance", "sentiment2", "sp500", "speed"])
    args = ap.parse_args(argv)
    path = RESULTS_DIR / "extended_summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    prices = load_prices(TICKERS)

    def save():
        path.write_text(json.dumps(summary, indent=2, default=float))

    if "variants" in args.only:
        summary["variants_walk_forward"] = exp_variants(prices); save()
    if "window" in args.only:
        summary["training_window"] = exp_window(prices); save()
    if {"horizons", "importance"} & set(args.only):
        api_df = exp_api_holdout(prices)
        if "horizons" in args.only:
            summary["horizons"] = exp_horizons(prices, api_df); save()
        if "importance" in args.only:
            summary["permutation_importance"] = exp_importance(); save()
    if "sentiment2" in args.only:
        summary["sentiment_followups"] = exp_sentiment2(); save()
    if "sp500" in args.only:
        summary["sp500"] = exp_sp500(); save()
    if "speed" in args.only:
        summary["speed"] = exp_speed(); save()
    print("\nWritten", path)


if __name__ == "__main__":
    main(sys.argv[1:])
