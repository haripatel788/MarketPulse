"""
Volatility forecasting benchmark. Rules are fixed in benchmarks/PREREGISTRATION.md.

Forecast target: mean daily realized variance over the next h days (h in 1, 5, 20), using the
overnight + Garman-Klass proxy from app.volatility. Models are scored with QLIKE against HAR.

Usage:
  .venv/bin/python -m benchmarks.vol_benchmark            # development set walk-forward
  .venv/bin/python -m benchmarks.vol_benchmark --bands    # calibrate price bands on the development set
  .venv/bin/python -m benchmarks.vol_benchmark --final    # one-shot final test (refuses to rerun)
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

from app.volatility import (
    BAND_LEVELS,
    BAND_PROBS,
    BAND_QUANTILES_PATH,
    HAR_COLS,
    POOLED_COLS,
    VAR_FLOOR,
    daily_variance,
    fit_har,
    forward_mean_variance,
    load_band_quantiles,
    market_features,
    pooled_features,
    standardized_moves,
)

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "price_cache"
RESULTS_DIR = ROOT / "results"
FINAL_PATH = RESULTS_DIR / "vol_final.json"

DEV_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "BAC", "XOM",
    "CVX", "JNJ", "PFE", "KO", "PG", "WMT", "DIS", "INTC", "BA", "F",
]
CUTOFF = pd.Timestamp("2025-10-14")
DEV_TEST_STARTS = [pd.Timestamp(f"{y}-10-14") for y in range(2018, 2025)]
HORIZONS = [1, 5, 20]
EWMA_LAMBDA = 0.94
MIN_TRAIN_ROWS = 400

BASELINES = ["hist20", "ewma", "garch", "har"]
POOLED_MODELS = ["pooled_har", "pooled_linear", "pooled_gbm"]
CANDIDATE: str | None = "pooled_linear"
CLAIMED_HORIZONS = [1, 5]
MIN_SHARE_BEATING_HAR = 0.60
MAX_DM_P = 0.05
SHIPPED_MODEL = "har"
SHIPPED_BASELINE = "ewma"
MIN_SHARE_BEATING_BASELINE = 0.70
BAND_HORIZONS = list(range(1, 31))
COVERAGE_TARGETS = {"80": (0.78, 0.82), "95": (0.935, 0.965)}
GBM_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 200,
    "l2_regularization": 1.0,
    "random_state": 0,
}
GBM_CALIBRATION_FRAC = 0.15
POOL_EXTRA: list[str] = []
MIX_SOURCE = {1: "pooled_linear", 5: "pooled_linear", 20: "pooled_har"}


# --------------------------------------------------------------------------- data


def load_prices(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(CACHE_DIR / f"{symbol}_na_10y.csv", parse_dates=["date"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def all_cached_symbols() -> list[str]:
    return sorted(p.name.removesuffix("_na_10y.csv") for p in CACHE_DIR.glob("*_na_10y.csv"))


def build_frame(prices: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Causal per-day quantities; targets are added separately so they respect truncation."""
    f = pd.DataFrame({"date": prices["date"], "close": prices["close"].astype(float)})
    f["r"] = np.log(f["close"] / f["close"].shift(1))
    f["rv"] = daily_variance(prices)
    f = pd.concat([f, pooled_features(prices, market)], axis=1)
    r2 = f["r"] ** 2
    f["hist20"] = r2.rolling(20).mean().clip(lower=VAR_FLOOR)
    f["ewma"] = r2.ewm(alpha=1.0 - EWMA_LAMBDA, adjust=False).mean().clip(lower=VAR_FLOOR)
    return f


def load_frame(symbol: str, data_end: pd.Timestamp | None) -> pd.DataFrame:
    prices = load_prices(symbol)
    spy = load_prices("SPY")
    if data_end is not None:
        prices = prices[prices["date"] < data_end].reset_index(drop=True)
        spy = spy[spy["date"] < data_end].reset_index(drop=True)
    return build_frame(prices, market_features(spy))


def add_targets(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for h in HORIZONS:
        out[f"y{h}"] = forward_mean_variance(out["rv"], h)
    return out


# --------------------------------------------------------------------------- per-ticker models


def garch_forecasts(train: pd.DataFrame, test: pd.DataFrame, full: pd.DataFrame) -> dict[int, np.ndarray]:
    """Fit GARCH(1,1) on training returns, then filter the full series with fixed parameters."""
    r_train = train["r"].dropna().to_numpy() * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = arch_model(r_train, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
    mu, omega, alpha, beta = (float(res.params[k]) for k in ("mu", "omega", "alpha[1]", "beta[1]"))
    persistence = alpha + beta

    eps2 = ((full["r"].fillna(0.0).to_numpy() * 100.0) - mu) ** 2
    s2_next = np.empty(len(full))
    s2 = float(np.var(r_train))
    for i in range(len(full)):
        s2 = omega + alpha * eps2[i] + beta * s2
        s2_next[i] = s2
    s2_next = pd.Series(s2_next, index=full.index).loc[test.index].to_numpy()

    out = {}
    if persistence < 1.0:
        uncond = omega / (1.0 - persistence)
        for h in HORIZONS:
            decay = persistence ** np.arange(h)
            out[h] = (uncond + np.outer(s2_next - uncond, decay)).mean(axis=1) / 1e4
    else:
        for h in HORIZONS:
            out[h] = s2_next / 1e4
    return out


def har_forecasts(train: pd.DataFrame, test: pd.DataFrame) -> dict[int, np.ndarray]:
    model = fit_har(train["rv"], HORIZONS)
    return {h: model.predict(test, h) for h in HORIZONS}


def per_ticker_forecasts(train: pd.DataFrame, test: pd.DataFrame, full: pd.DataFrame) -> dict[str, dict[int, np.ndarray]]:
    preds: dict[str, dict[int, np.ndarray]] = {
        "hist20": {h: test["hist20"].to_numpy() for h in HORIZONS},
        "ewma": {h: test["ewma"].to_numpy() for h in HORIZONS},
        "har": har_forecasts(train, test),
    }
    try:
        preds["garch"] = garch_forecasts(train, test, full)
    except Exception as exc:
        print(f"    garch failed ({exc.__class__.__name__}); skipping")
    return preds


# --------------------------------------------------------------------------- pooled models


class PooledLinear:
    """OLS on log target across stocks, rescaled so the variance forecast is QLIKE-unbiased."""

    def __init__(self, cols: list[str]):
        self.cols = cols

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.ones(len(frame)), frame[self.cols].to_numpy()])

    def fit(self, frame: pd.DataFrame, target: str) -> "PooledLinear":
        x = self._design(frame)
        y = frame[target].to_numpy()
        self.coef, *_ = np.linalg.lstsq(x, np.log(y), rcond=None)
        self.scale = float(np.mean(y / np.exp((x * self.coef).sum(axis=1))))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp((self._design(frame) * self.coef).sum(axis=1)) * self.scale


class PooledGBM:
    """Gradient boosting on log target; the variance scale is set on a later, unseen slice."""

    def __init__(self, cols: list[str]):
        self.cols = cols

    def fit(self, frame: pd.DataFrame, target: str) -> "PooledGBM":
        dates = np.sort(frame["date"].unique())
        cut = dates[int(len(dates) * (1.0 - GBM_CALIBRATION_FRAC))]
        purge_start = dates[max(0, np.searchsorted(dates, cut) - max(HORIZONS))]
        early = frame[frame["date"] < purge_start]
        late = frame[frame["date"] >= cut]
        probe = HistGradientBoostingRegressor(**GBM_PARAMS).fit(early[self.cols], np.log(early[target]))
        self.scale = float(np.mean(late[target] / np.exp(probe.predict(late[self.cols]))))
        self.model = HistGradientBoostingRegressor(**GBM_PARAMS).fit(frame[self.cols], np.log(frame[target]))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.exp(self.model.predict(frame[self.cols])) * self.scale


def fit_pooled(train: pd.DataFrame) -> dict[str, dict[int, PooledLinear | PooledGBM]]:
    fitted: dict[str, dict[int, PooledLinear | PooledGBM]] = {m: {} for m in POOLED_MODELS}
    for h in HORIZONS:
        tr = train.dropna(subset=POOLED_COLS + [f"y{h}"])
        fitted["pooled_har"][h] = PooledLinear(HAR_COLS).fit(tr, f"y{h}")
        fitted["pooled_linear"][h] = PooledLinear(POOLED_COLS).fit(tr, f"y{h}")
        fitted["pooled_gbm"][h] = PooledGBM(POOLED_COLS).fit(tr, f"y{h}")
    return fitted


# --------------------------------------------------------------------------- scoring


def qlike(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    ratio = y / np.clip(f, VAR_FLOOR, None)
    return ratio - np.log(ratio) - 1.0


def dm_test(diff_by_date: pd.Series, lag: int) -> tuple[float, float]:
    d = diff_by_date.to_numpy(dtype=float)
    n = len(d)
    centered = d - d.mean()
    var = float(np.sum(centered * centered)) / n
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1)) * float(np.sum(centered[k:] * centered[:-k])) / n
    if var <= 0:
        return float("nan"), float("nan")
    t = d.mean() / np.sqrt(var / n)
    return float(t), float(2.0 * stats.norm.sf(abs(t)))


def score_rows(symbol: str, fold: str, test: pd.DataFrame, preds: dict[str, dict[int, np.ndarray]]) -> list[pd.DataFrame]:
    out = []
    for h in HORIZONS:
        y = test[f"y{h}"].to_numpy()
        mask = np.isfinite(y)
        for name, by_h in preds.items():
            f = by_h[h]
            mask &= np.isfinite(f)
        if not mask.any():
            continue
        rows = pd.DataFrame({"symbol": symbol, "fold": fold, "h": h, "date": test["date"].to_numpy()[mask], "y": y[mask]})
        for name, by_h in preds.items():
            rows[f"f_{name}"] = by_h[h][mask]
        out.append(rows)
    return out


def summarize(daily: pd.DataFrame) -> dict:
    models = [c.removeprefix("f_") for c in daily.columns if c.startswith("f_")]
    summary: dict = {}
    for h, g in daily.groupby("h"):
        g = g.copy()
        for m in models:
            g[f"L_{m}"] = qlike(g["y"].to_numpy(), g[f"f_{m}"].to_numpy())
        per_ticker = g.groupby("symbol")[[f"L_{m}" for m in models]].mean()
        block = {}
        for m in models:
            entry = {
                "mean_qlike": float(g[f"L_{m}"].mean()),
                "median_ticker_qlike": float(per_ticker[f"L_{m}"].median()),
                "mean_log_bias": float(np.mean(np.log(g["y"] / g[f"f_{m}"]))),
                "spearman": float(g.groupby("symbol").apply(
                    lambda s, m=m: stats.spearmanr(s[f"f_{m}"], s["y"]).statistic, include_groups=False
                ).mean()),
            }
            for ref in ("har", "ewma"):
                if m == ref:
                    continue
                entry[f"share_tickers_beating_{ref}"] = float((per_ticker[f"L_{m}"] < per_ticker[f"L_{ref}"]).mean())
                diff = (g[f"L_{m}"] - g[f"L_{ref}"]).groupby(g["date"]).mean().sort_index()
                entry[f"dm_t_vs_{ref}"], entry[f"dm_p_vs_{ref}"] = dm_test(diff, lag=int(h))
            block[m] = entry
        summary[f"h{h}"] = {"n_rows": int(len(g)), "n_tickers": int(g["symbol"].nunique()), "models": block}
    return summary


# --------------------------------------------------------------------------- experiments


def run_split(symbols: list[str], test_start: pd.Timestamp, test_end: pd.Timestamp, fold: str,
              data_end: pd.Timestamp | None) -> list[pd.DataFrame]:
    pool_train = []
    for sym in DEV_TICKERS + POOL_EXTRA:
        base = load_frame(sym, data_end)
        if len(base) < MIN_TRAIN_ROWS:
            continue
        pool_train.append(add_targets(base[base["date"] < test_start]))
    pooled = fit_pooled(pd.concat(pool_train, ignore_index=True))

    rows: list[pd.DataFrame] = []
    for sym in symbols:
        base = load_frame(sym, data_end)
        full = add_targets(base)
        train = add_targets(base[base["date"] < test_start])
        test = full[(full["date"] >= test_start) & (full["date"] < test_end)]
        test = test.dropna(subset=POOLED_COLS + ["hist20", "ewma"])
        if len(train.dropna(subset=HAR_COLS + ["y20"])) < MIN_TRAIN_ROWS or test.empty:
            continue
        preds = per_ticker_forecasts(train, test, full)
        for name, by_h in pooled.items():
            preds[name] = {h: model.predict(test) for h, model in by_h.items()}
        preds["pooled_mix"] = {h: preds[MIX_SOURCE[h]][h] for h in HORIZONS}
        rows.extend(score_rows(sym, fold, test, preds))
    return rows


def run_dev() -> None:
    rows: list[pd.DataFrame] = []
    for start in DEV_TEST_STARTS:
        end = start + pd.DateOffset(years=1)
        print(f"fold {start.date()} -> {end.date()}")
        rows.extend(run_split(DEV_TICKERS, start, end, fold=str(start.year), data_end=CUTOFF))
    daily = pd.concat(rows, ignore_index=True)
    summary = summarize(daily)
    RESULTS_DIR.mkdir(exist_ok=True)
    daily.to_csv(RESULTS_DIR / "vol_dev_daily.csv.gz", index=False)
    (RESULTS_DIR / "vol_dev_summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(summary)


def accuracy_verdict(summary: dict) -> dict:
    verdict = {}
    for h in CLAIMED_HORIZONS:
        e = summary[f"h{h}"]["models"][CANDIDATE]
        verdict[f"h{h}"] = {
            "share_tickers_beating_har": e["share_tickers_beating_har"],
            "dm_p_vs_har": e["dm_p_vs_har"],
            "pass": bool(
                e["share_tickers_beating_har"] >= MIN_SHARE_BEATING_HAR
                and e["dm_p_vs_har"] < MAX_DM_P
                and e["dm_t_vs_har"] < 0
            ),
        }
    verdict["pass"] = all(v["pass"] for v in verdict.values())
    return verdict


def shipped_verdict(summary: dict) -> dict:
    verdict = {}
    for h in HORIZONS:
        e = summary[f"h{h}"]["models"][SHIPPED_MODEL]
        share = e[f"share_tickers_beating_{SHIPPED_BASELINE}"]
        t, p = e[f"dm_t_vs_{SHIPPED_BASELINE}"], e[f"dm_p_vs_{SHIPPED_BASELINE}"]
        verdict[f"h{h}"] = {
            f"share_tickers_beating_{SHIPPED_BASELINE}": share,
            "dm_t": t,
            "dm_p": p,
            "pass": bool(share >= MIN_SHARE_BEATING_BASELINE and p < MAX_DM_P and t < 0),
        }
    verdict["pass"] = all(v["pass"] for v in verdict.values())
    return verdict


# --------------------------------------------------------------------------- price bands


def band_moves(symbols: list[str], test_start: pd.Timestamp, test_end: pd.Timestamp, fold: str,
               data_end: pd.Timestamp | None) -> pd.DataFrame:
    """Realized h-day log moves divided by the HAR forecast h-day standard deviation."""
    out = []
    for sym in symbols:
        base = load_frame(sym, data_end)
        train = base[base["date"] < test_start]
        if len(train.dropna(subset=HAR_COLS)) < MIN_TRAIN_ROWS:
            continue
        model = fit_har(train["rv"], BAND_HORIZONS)
        in_test = (base["date"] >= test_start) & (base["date"] < test_end) & base[HAR_COLS].notna().all(axis=1)
        test = base[in_test]
        if test.empty:
            continue
        for h in BAND_HORIZONS:
            variance = pd.Series(np.nan, index=base.index)
            variance[in_test] = model.predict(test, h)
            z = standardized_moves(base["close"], variance, h)[in_test].dropna()
            out.append(pd.DataFrame({"symbol": sym, "fold": fold, "h": h, "date": base.loc[z.index, "date"], "z": z}))
    return pd.concat(out, ignore_index=True)


def band_quantiles(z: pd.DataFrame) -> dict[int, dict[str, float]]:
    return {
        int(h): {name: float(g["z"].quantile(p)) for name, p in BAND_PROBS.items()}
        for h, g in z.groupby("h")
    }


def band_coverage(z: pd.DataFrame, quantiles: dict[int, dict[str, float]], horizons: list[int]) -> dict:
    out = {}
    for h in horizons:
        zh = z.loc[z["h"] == h, "z"]
        q = quantiles[h]
        out[f"h{h}"] = {
            level: float(((zh >= q[lo]) & (zh <= q[hi])).mean()) for level, (lo, hi) in BAND_LEVELS.items()
        }
        out[f"h{h}"]["n"] = int(len(zh))
    return out


def run_bands() -> None:
    """Calibrate band quantiles on the development set and write them for the app."""
    folds = []
    for start in DEV_TEST_STARTS:
        end = start + pd.DateOffset(years=1)
        print(f"bands fold {start.date()} -> {end.date()}")
        folds.append(band_moves(DEV_TICKERS, start, end, fold=str(start.year), data_end=CUTOFF))
    z = pd.concat(folds, ignore_index=True)

    walk_forward = {}
    for i in range(1, len(folds)):
        past = pd.concat(folds[:i], ignore_index=True)
        walk_forward[folds[i]["fold"].iloc[0]] = band_coverage(folds[i], band_quantiles(past), HORIZONS)
    pooled = {
        f"h{h}": {lvl: float(np.mean([wf[f"h{h}"][lvl] for wf in walk_forward.values()])) for lvl in BAND_LEVELS}
        for h in HORIZONS
    }
    report = {"walk_forward_by_year": walk_forward, "walk_forward_mean": pooled}
    (RESULTS_DIR / "vol_bands_dev.json").write_text(json.dumps(report, indent=2))

    quantiles = band_quantiles(z)
    BAND_QUANTILES_PATH.write_text(json.dumps({
        "source": "benchmarks/vol_benchmark.py --bands: per-stock HAR, development tickers, "
                  f"walk-forward test years {DEV_TEST_STARTS[0].year}-{DEV_TEST_STARTS[-1].year + 1}",
        "quantiles": {str(h): q for h, q in quantiles.items()},
    }, indent=2))

    print(f"\n{'year':<6}" + "".join(f"{f'h{h} 80%':>10}{f'h{h} 95%':>10}" for h in HORIZONS))
    for year, cov in walk_forward.items():
        print(f"{year:<6}" + "".join(f"{cov[f'h{h}']['80']:>10.1%}{cov[f'h{h}']['95']:>10.1%}" for h in HORIZONS))
    print(f"{'mean':<6}" + "".join(f"{pooled[f'h{h}']['80']:>10.1%}{pooled[f'h{h}']['95']:>10.1%}" for h in HORIZONS))
    print(f"\nwrote {BAND_QUANTILES_PATH}")


def calibration_verdict(coverage: dict) -> dict:
    verdict = {}
    for h in HORIZONS:
        lo, hi = COVERAGE_TARGETS["80"]
        verdict[f"h{h}"] = {**coverage[f"h{h}"], "pass": bool(lo <= coverage[f"h{h}"]["80"] <= hi)}
    verdict["pass"] = all(v["pass"] for v in verdict.values())
    return verdict


def run_final() -> None:
    if FINAL_PATH.exists():
        raise SystemExit(f"{FINAL_PATH} exists. The final test runs once; see PREREGISTRATION.md.")
    if not BAND_QUANTILES_PATH.exists():
        raise SystemExit("Band quantiles missing; run --bands on the development set first.")
    holdout = [s for s in all_cached_symbols() if s not in DEV_TICKERS and s != "SPY"]
    print(f"final test: {len(holdout)} tickers, origins from {CUTOFF.date()}")

    rows = run_split(holdout, CUTOFF, pd.Timestamp.max, fold="final", data_end=None)
    daily = pd.concat(rows, ignore_index=True)
    summary = summarize(daily)
    z = band_moves(holdout, CUTOFF, pd.Timestamp.max, fold="final", data_end=None)
    coverage = band_coverage(z, load_band_quantiles(), HORIZONS)

    result = {
        "shipped_model": {"model": SHIPPED_MODEL, "vs": SHIPPED_BASELINE, **shipped_verdict(summary)},
        "calibration": calibration_verdict(coverage),
        "secondary_candidate": {"model": CANDIDATE, **accuracy_verdict(summary)},
        "summary": summary,
    }
    daily.to_csv(RESULTS_DIR / "vol_final_daily.csv.gz", index=False)
    z.to_csv(RESULTS_DIR / "vol_final_band_z.csv.gz", index=False)
    FINAL_PATH.write_text(json.dumps(result, indent=2))
    print_summary(summary)
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}, indent=2))


def print_summary(summary: dict) -> None:
    for hk, block in summary.items():
        print(f"\n{hk}  rows={block['n_rows']}  tickers={block['n_tickers']}")
        print(f"  {'model':<10}{'QLIKE':>9}{'spearman':>10}{'log_bias':>10}{'beat_HAR':>10}{'DM_t':>8}{'DM_p':>9}")
        for m, e in sorted(block["models"].items(), key=lambda kv: kv[1]["mean_qlike"]):
            beat = e.get("share_tickers_beating_har")
            print(
                f"  {m:<10}{e['mean_qlike']:>9.4f}{e['spearman']:>10.3f}{e['mean_log_bias']:>10.3f}"
                f"{'' if beat is None else f'{beat:.0%}':>10}"
                f"{'' if beat is None else f'{e['dm_t_vs_har']:.2f}':>8}"
                f"{'' if beat is None else f'{e['dm_p_vs_har']:.3g}':>9}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--bands", action="store_true")
    args = parser.parse_args()
    if args.final:
        run_final()
    elif args.bands:
        run_bands()
    else:
        run_dev()


if __name__ == "__main__":
    main()
