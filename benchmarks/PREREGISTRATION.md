# Volatility forecaster: pre-registered evaluation

Written 2026-10-15, before any volatility model was built or any baseline was scored.
Do not edit the success criteria after results exist. If something must change, add a dated
amendment at the bottom explaining why, and keep the original text.

## What is being forecast

At the close of trading day `t`, forecast the average daily variance of a stock over the next
`h` trading days, for `h` in {1, 5, 20}.

Realized variance proxy for day `t` (computed in `app/volatility.py`):

- `overnight_t = ln(open_t / close_{t-1})^2`
- `garman_klass_t = 0.5 * ln(high_t / low_t)^2 - (2 ln 2 - 1) * ln(close_t / open_t)^2`
- `rv_t = overnight_t + garman_klass_t`, falling back to `ln(close_t / close_{t-1})^2` when OHLC is
  missing or inconsistent, floored at `1e-7`.

Target: `mean(rv_{t+1} ... rv_{t+h})`.

## Data

- Prices: Yahoo daily OHLCV in `benchmarks/price_cache/*_na_10y.csv`, 500 current S&P 500 members,
  2016-10-14 to 2026-10-13. SPY from the same source is used only as a market-level input.
- Known bias: current index members only (survivorship). It affects every model equally and is
  reported, not corrected.

### Development set (tune freely)

- Tickers: AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM, BAC, XOM, CVX, JNJ, PFE, KO, PG, WMT,
  DIS, INTC, BA, F.
- Forecast origins strictly before **2025-10-14**.
- Walk-forward: one-year test blocks starting each Oct 14 from 2018 to 2024; training uses all
  earlier data minus the last 20 rows (purge gap so no training target overlaps the test block).

### Final test set (run once)

- Tickers: the other 480 cached tickers.
- Forecast origins on or after **2025-10-14** whose `h`-day target is fully observed.
- Models are trained only on data whose targets end before 2025-10-14.
- The final run is executed once with `--final`. The script refuses to overwrite an existing
  final result.

## Baselines

1. `hist20`: mean of squared close-to-close log returns over the last 20 days.
2. `ewma`: RiskMetrics EWMA of squared close-to-close log returns, lambda = 0.94.
3. `garch`: GARCH(1,1), constant mean, normal errors, fitted per ticker on the training window;
   multi-step forecasts averaged over the horizon.
4. `har`: log-HAR per ticker, OLS of `log(target)` on `log rv_t`, `log mean(rv, 5d)`,
   `log mean(rv, 22d)`, with a lognormal bias correction. **This is the bar to beat.**

## Candidate model

Exactly one candidate goes to the final test. It is chosen on the development set by lowest
mean QLIKE averaged over the three horizons, and its name and configuration are recorded in the
amendment section below before `--final` is run.

## Success criteria (final test set only)

Primary loss: QLIKE on variance, `L = y/f - ln(y/f) - 1`, where `y` is the realized target and `f`
the forecast.

**Accuracy passes** if, at each horizon h in {1, 5, 20}:

- the candidate's mean QLIKE is lower than HAR's on at least 60% of final-test tickers, and
- a Diebold-Mariano test vs HAR is significant at p < 0.05 (two-sided). The test uses the daily
  cross-sectional mean of loss differences, with Newey-West standard errors at lag `h`.

**Calibration passes** if the 80% price band contains the realized close `h` days ahead on
78%-82% of final-test origins, at each horizon. 95% band coverage is reported, target 93.5%-96.5%.

## Outcomes

- Both pass: ship the candidate and claim it beats HAR out of sample.
- Calibration passes, accuracy fails: ship HAR with calibrated bands. No "beats HAR" claim.
- Calibration fails: fix the band method on the development set, then add an amendment.
  The accuracy result stands as is.

## Amendments

### 2026-10-15: placeholder rows are missing data (before any final-test run)

Rows with `high <= low` or `volume <= 0` now give `rv_t = NaN` instead of the `1e-7` floor. Any
feature window or target that touches such a row is dropped for every model alike.

Why: 21 cached tickers have these rows, all outside the development set. SW (1,645 rows),
FERG (987), AMCR (645), VRT (60) and HWM (35) carry long stretches of flat pre-listing quotes.
Treating them as near-zero variance made `y / f` ratios above 10,000 when they entered the
training pool. Found while testing a larger training pool on the development set; no final-test
data was looked at.

### 2026-10-15: candidate locked, accuracy claim limited to h = 1 and h = 5 (before any final-test run)

Candidate: `pooled_linear` in `benchmarks/vol_benchmark.py`. One OLS model per horizon on
`log(target)`, trained on the 20 development tickers only, with inputs `POOLED_COLS` from
`app/volatility.py` (HAR terms, 66-day level, negative 1-day and 5-day returns, overnight gap,
volume ratio, SPY HAR terms, stock-minus-market monthly level). The variance forecast is
rescaled by `mean(y / exp(fit))` on the training rows.

The accuracy criterion now applies only at h = 1 and h = 5. At h = 20 the app uses per-ticker
HAR, which is reported but makes no "beats HAR" claim. Calibration still applies at all three
horizons.

Why: on the development walk-forward, the candidate beat HAR in every test year at h = 1
(+4% to +19%), was close to even in 2020-2024 at h = 5 (-0.1% to +5%), and lost to HAR in four
of the five 2020-2024 years at h = 20 (-5% to -24%). Its overall h = 20 advantage came only
from 2018-2019, when per-ticker HAR had 2-3 years of training data. The final test period
resembles the later years. A one-year h = 20 test also has only about 12 non-overlapping
windows, too few for the DM test.

Rejected on the development set: gradient boosting (worse than HAR at h = 5 and h = 20),
a 120-ticker training pool (no better than 20), and a one-year volatility level input
(no consistent change).

### 2026-10-15: shipped model is per-ticker HAR; primary test is HAR vs EWMA (before any final-test run)

Product decision: the app ships per-ticker HAR (`app.volatility.fit_har`), and HAR scales the
price bands. The `pooled_linear` test from the previous amendment is kept as a **secondary**
result. It is graded and reported, but it does not decide what ships.

**Primary accuracy test (replaces "Success criteria" above).** On the 480 final-test tickers,
at each of h = 1, 5 and 20:

- HAR's mean QLIKE is lower than EWMA's on at least **70%** of tickers, and
- a Diebold-Mariano test of HAR vs EWMA gives p < 0.05 with HAR better (daily
  cross-sectional mean of loss differences, Newey-West lag `h`).

All three horizons must pass.

**Price bands.** The band for horizon `h` is `last_close * exp(q * sqrt(h * f_h))`, where `f_h`
is HAR's forecast of mean daily variance over the next `h` days. `q` comes from the 10/90
(80% band) and 2.5/97.5 (95% band) percentiles of `ln(close_{t+h} / close_t) / sqrt(h * f_h)`.
Those percentiles were computed once, on the development tickers' walk-forward test years
2018-2025, and frozen in `app/vol_band_quantiles.json` for h = 1 to 30. They are not refit on
final-test data.

**Calibration test (unchanged).** The 80% band covers 78%-82% of final-test origins at each of
h = 1, 5 and 20. 95% coverage is reported against 93.5%-96.5%.

Development-set evidence at the time of writing:

- HAR vs EWMA beat share was 95%, 90% and 85%; DM p was 6e-20, 5e-4 and 0.026 for h = 1, 5, 20.
- Walk-forward 80% band coverage, using only earlier years' percentiles, averaged 81.5%,
  81.3% and 81.7%. Single years ranged from 75% to 88%.

Known risk, stated before the run: the final test is a single year, and individual years
missed 78%-82% on the development set, so calibration can fail on market regime alone.

No model, input, percentile or threshold will be changed after the final result is seen.
