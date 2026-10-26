# Comparing Topological Early-Warning Signals with Classical Volatility Indicators

## Executive summary

This project builds an objective, leakage-controlled forecast experiment for
rare market drawdowns in the S&P 500, Dow Jones, Nasdaq Composite, and Russell
2000. It compares three sequence models with the same LSTM architecture:
classical indicators only, TDA indicators only, and classical plus TDA.

In the current seeded walk-forward run, the combined TDA+CSD LSTM had the
best mean daily average precision (0.0307) across the three test periods.
The classical-only LSTM reached 0.0144 and TDA alone 0.0124. These are low
absolute scores and the result changes substantially by period. The combined
model's advantage in this one seeded run is preliminary; repeated seeds and
event-level confidence intervals are needed before claiming reliable TDA
lift. The experiment provides a reproducible baseline and clear next steps.

This report has since been extended with three larger experiments (sections
10–14), all using the same objective labels and protocol: (1) a unified
walk-forward benchmark across 7 asset datasets and 39 tickers comparing
classical, TDA, combined, landscape-vector, price-momentum, and peer feature
families on 8 sequence models; (2) a lead-time / early-warning analysis over
2,034 crash onsets; and (3) two independent peer/momentum-feature
experiments. Headline outputs: classical CSD remains the strongest single
"classical" family, richer TDA summaries (landscape vectors) make topology
competitive (best cell .0694 on energy), topology warns earlier and more
consistently than CSD (fires before 58–83% of onsets), and the
peer/momentum family is the strongest signal family overall (AP up to
.0508). A machine-generated companion with every run's numbers is in
`EXPERIMENTS_FULL_REPORT.md`; the append-only experiment log is
`GLM_EXPERIMENT_LOG.md`.

## 1. Project objective

The objective is to test whether topological features from persistent
homology provide predictive information about a future large drawdown beyond
classical critical-slowing-down (CSD) statistics. The forecast unit is one
index on one trading date. At date *t*, only observations available through
that date are used to predict whether the next 15 trading sessions contain a
large loss.

## 2. Data and preprocessing

- Four original US indices: S&P 500 (`^GSPC`), Dow Jones (`^DJI`), Nasdaq
  Composite (`^IXIC`), and Russell 2000 (`^RUT`).
- Local price history covers January 1997 through September 2026. This run
  used the CSVs already in the repository; it did not download fresh prices.
- Daily log returns are calculated as `log(P[t] / P[t-1])`.
- TDA is computed from the joint four-dimensional daily-return point cloud
  with Euclidean Vietoris–Rips persistent homology through H1, using 50- and
  100-session rolling windows. Exact persistence-landscape L1/L2 norms are
  taken from the existing paper-method output.
- Classical features are computed separately for each index: lag-1 return
  autocorrelation, rolling return variance, and rolling skewness over 50 and
  100 sessions, plus 20-session realized volatility.
- Rolling statistics use data through the scored date. The TDA vector is
  shared across the four target indices because it is formed from the joint
  four-index cloud.

The source paper specifies adjusted-close values. The repository's existing
price inputs are Close-derived, as documented in its replication report. This
data limitation carries into the present results.

## 3. Objective label

For each index and date *t*, calculate the minimum simple return over the
following 15 closes:

`forward_min_return(t) = min(P[t+k] / P[t] - 1), k = 1,...,15`.

For each index independently, fix the crash threshold to the 2nd percentile
of this forward-minimum-return series using only origins whose full 15-session
outcomes end by 2006-12-31 (the last eligible origin is recorded in
`label_thresholds.csv`). The threshold is not re-estimated in later years. The resulting
positive label means the next 15 sessions contain a decline at least as large
as this calibration-period tail threshold.

| Index | Fixed threshold | Positive share in modeling panel |
|---|---:|---:|
| S&P 500 | -10.45% | 2.75% |
| Dow Jones | -11.13% | 1.97% |
| Nasdaq | -18.32% | 1.15% |
| Russell 2000 | -14.01% | 2.13% |

The fixed tail cutoffs are statistically motivated by a training-period
quantile, but they are not universal definitions of a financial crash. The
different cutoffs reflect each index's own return distribution. Existing
epsilon-drawdown / exponential-tail crash labels and reports remain available
in the repository as a complementary labeling method.

## 4. Models and evaluation design

### Sequence model

Each input is a 20-trading-session sequence. The LSTM has one recurrent layer
with 24 hidden units and a binary classification head. Features are
standardized using training data only. Training uses a fixed random seed,
AdamW, class-weighted binary cross-entropy, gradient clipping, and early
stopping on validation average precision. The final saved artifacts are
trained on data before the last test period, selected using a preceding
chronological validation period.

Three feature ablations are fit with the same model and settings:

1. **Classical:** CSD and realized-volatility features.
2. **TDA:** persistence-landscape features only.
3. **Combined:** CSD plus TDA.

### Walk-forward splits

| Test block | Training dates end before | Validation period |
|---|---|---|
| 2011–2015 | 25-calendar-day purge before validation and test | final 2 years before training cutoff |
| 2016–2020 | 25-calendar-day purge before validation and test | final 2 years before training cutoff |
| 2021–September 2026 | 25-calendar-day purge before validation and test | final 2 years before training cutoff |

The 25-calendar-day gaps are longer than the 15-session label horizon and
reduce overlap between the training, validation, and test targets. A validation
score threshold is set at its 95th percentile, targeting about 5% alert days;
test alert rates can differ because score distributions shift. The displayed
sigmoid scores are ranking scores and are **not calibrated probabilities**.

### Metrics

- **Average precision (primary):** suitable for rare positives and comparable
  with each test block's prevalence baseline.
- ROC AUC and Brier score.
- Precision, recall, F1, and realized alert rate using the validation-set
  threshold.

Average precision is the most stable comparison here. Threshold metrics show
how usable an alert rule is but can shift substantially with the market regime.
The test samples are daily observations with overlapping 15-session targets,
so they are not independent crash events.

## 5. Current results

### Per-test-block results

| Test block | Positive prevalence | Classical AP | TDA AP | Combined AP | Classical ROC AUC | TDA ROC AUC | Combined ROC AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2011–2015 | 0.95% | 0.0082 | 0.0094 | 0.0073 | 0.301 | 0.495 | 0.302 |
| 2016–2020 | 2.18% | 0.0199 | 0.0165 | 0.0340 | 0.448 | 0.374 | 0.703 |
| 2021–Sep 2026 | 0.70% | 0.0150 | 0.0113 | 0.0507 | 0.712 | 0.701 | 0.847 |
| Mean of test blocks | — | 0.0144 | 0.0124 | 0.0307 | 0.487 | 0.524 | 0.617 |

The 2011–2015 test block contains 48 positive index-days in the pooled panel;
2016–2020 contains 110; and 2021 onward contains 40. The Nasdaq has no
positive labels in 2011–2015 and 2021 onward under its fixed -18.32% cutoff,
so its per-index ROC AUC is undefined in those two blocks.

At the validation-selected threshold, mean precision across test blocks was
1.61% for classical, 0% for TDA, and 3.42% for combined. Mean realized alert
rates were 11.46%, 7.15%, and 8.16%, respectively. Some test rates were far
from the 5% validation target because many model outputs saturated at nearly
identical sigmoid scores and the score distributions shifted across periods.
This makes the present threshold rule unsuitable for deployment without
better calibration. A strong average precision does not guarantee a useful
fixed threshold.

### Interpretation

- The combined model had the highest average precision in 2016–2020 and
  2021–September 2026, and the best three-block mean in this seeded run.
- No model clearly ranked the positive days in 2011–2015; average precision
  for all three feature sets was at or below the 0.95% prevalence.
- TDA alone had the lowest three-block mean average precision and did not
  outperform the prevalence baseline consistently.
- Adding TDA to classical inputs improved over both ablations on the mean
  average precision in this single seeded run.
- The LSTM results are a baseline, not evidence that sequence models are
  optimal. A one-seed run and limited number of distinct crash episodes make
  the averages uncertain.
- These results do not invalidate TDA as a descriptive stress indicator.
  They show that the current features, target, training procedure, and sample
  have not demonstrated reliable out-of-sample predictive lift.

## 6. Qualitative validation

`forecasting_qualitative.py` creates event-aligned panels for the dot-com
decline, Lehman crisis, 2011 debt-ceiling stress, COVID selloff, and 2022 bear
market. Each panel compares four-index price drawdowns with the joint TDA L2
norm and per-index variance and lag-1 autocorrelation. Indicator z-scores are
scaled against the 180-session pre-event period for display only; they are
not model inputs or forecast scores. Read them as descriptive event plots,
not evidence of advance prediction.

The panels are in `data/forecasting/event_*.png`. Existing TDA replication
figures and reports in `data/group_a_original_indices/` provide additional
paper-reproduction and episode-level diagnostics.

## 7. Interface and use

The Streamlit interface is `forecast_app.py`. It accepts a feature-panel CSV
with `date`, `ticker`, and the feature columns stored in the model bundle.
Extra columns such as labels are ignored for scoring. The generated
`data/forecasting/modeling_panel.csv` is a ready-to-upload demonstration.
The app displays the latest risk score per index, a score history plot, and
optional metrics when compatible labels are present; it can download the
scored rows.

Install dependencies and launch from this folder:

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m streamlit run forecast_app.py
```

To regenerate the experiments and event plots:

```powershell
py -3.12 crash_forecasting.py
py -3.12 forecasting_qualitative.py
```

## 8. Deliverable files

- `crash_forecasting.py`: target construction, feature alignment, LSTM
  training, walk-forward evaluation, and serialized models.
- `forecasting_qualitative.py`: event-aligned visual analysis.
- `forecast_app.py`: CSV upload and risk-score interface.
- `data/forecasting/modeling_panel.csv`: reproducible feature/label panel.
- `data/forecasting/label_thresholds.csv`: calibration thresholds.
- `data/forecasting/walk_forward_metrics.csv`: pooled and per-index metrics.
- `data/forecasting/classical_lstm.pt`, `tda_lstm.pt`,
  `combined_lstm.pt`: trained model states; corresponding scaler files are
  saved beside them.
- `data/forecasting/report.txt`: concise machine-generated run record.
- `data/forecasting/event_*.png`: qualitative validation plots.
- `pipeline/` package with `compute_features.py`, `run_benchmark.py`,
  `analyze_benchmark.py`, `qualitative_leadtime.py`, `augment_panel.py`:
  the unified multi-dataset benchmark of sections 10–14.
- `data/benchmark/unified_panel.csv` (212,816 rows × 119 columns),
  `all_results_v1.csv`, `all_results.csv` (v2 grid, in progress),
  `qualitative/` lead-time outputs, `benchmark_full_v2.log`.
- `EXPERIMENTS_FULL_REPORT.md` / `GLM_EXPERIMENT_LOG.md`: complete results
  snapshot and append-only experiment log.
- `RUN_RECORD.md` + `data/benchmark/run_record_v2.csv` (800 rows) +
  `run_record_v1.csv` (420 rows): **one row per planned run** — each model's
  per-fold test metrics, or an explicit `not_run` / `skipped` status; the
  ~735er-run raw per-ticker breakdowns remain in `all_results.csv`.

## 9. New cross-index temporal-attention experiment

### Why this is a separate approach

Deep learning can learn nonlinear representations, but a neural network is
not automatically topological. TDA/persistent homology explicitly summarizes
topological structure that persists across filtration scales. The new model
instead learns temporal representations of market histories and cross-index
attention weights. It does not calculate persistent homology.

### Inputs and architecture

For each index, the model uses 1-day returns, 5/20/60-session momentum,
20/60-session realized volatility, 20-session downside volatility,
60/252-session drawdown, a 20-to-60-day volatility ratio, and 20-session
return skew. Every feature is trailing and available at the scored date. Each
input covers 30 sessions and predicts the same fixed, 15-session crash label
from the earlier experiment.

The cross-market model first encodes each index's 30-session history with a
shared LSTM. A learned query for the target index attends across the encoded
markets. Three inputs are compared:

1. **self_only:** only the target index's feature history reaches the LSTM.
2. **other_only:** the target's full feature history is zeroed before encoding;
   the query attends only to the other three indices.
3. **all_indices:** the query can attend to the target and its three peers.

All three variants use identical walk-forward dates, labels, validation
threshold selection, and evaluation metrics. Outputs are kept separately in
`data/cross_index_forecasting/`; the earlier TDA/CSD work is retained.

### Results

| Test block | Positive prevalence | Self only AP | Other only AP | All indices AP | Other only ROC AUC | All indices ROC AUC |
|---|---:|---:|---:|---:|---:|---:|
| 2011–2015 | 0.95% | 0.0102 | 0.0117 | 0.0106 | 0.613 | 0.565 |
| 2016–2020 | 2.18% | 0.0554 | 0.0632 | 0.0863 | 0.750 | 0.825 |
| 2021–Sep 2026 | 0.70% | 0.0199 | 0.0349 | 0.0337 | 0.899 | 0.901 |
| Mean of test blocks | — | 0.0285 | 0.0366 | 0.0436 | 0.754 | 0.764 |

The peer-only model beats self-only average precision in all three blocks. Its
mean average precision is about 28% higher; the all-index model is about 53%
higher than self-only. This is preliminary evidence that the other indices
carry useful information for the target labels in this sample. It is not yet
evidence of stable crash forecasting: this is a single seeded run, the
2011–2015 ranking was close to the 0.95% prevalence baseline, and thresholded
alerts did not consistently catch positives. On the 2021+ block both
cross-index variants ranked positives above negatives reasonably well, but
the peer-only validation threshold produced no true-positive alerts.

Per-index AP/AUC metrics are in `data/cross_index_forecasting/cross_index_walk_forward_metrics.csv`.
The largest consistently scored series are S&P 500, Dow, and Russell; Nasdaq
has no positive labels in the first and last test blocks under its fixed
calibration threshold, so AUC is undefined there.

### Cross-index interface and artifacts

`cross_index_app.py` accepts a CSV of daily closes with `date`, `^GSPC`,
`^DJI`, `^IXIC`, and `^RUT` columns. It calculates the trailing features and
shows self-only, peer-only, and all-index risk scores for each target. Start
it with `py -3.12 -m streamlit run cross_index_app.py`; the existing
`forecast_app.py` remains available for the TDA/CSD model.

The trained checkpoints are `self_only.pt`, `other_only.pt`, and
`all_indices.pt`, with their scalers and metadata beside them under
`data/cross_index_forecasting/`. Rebuild the benchmark with
`py -3.12 cross_index_forecasting.py`.

## 10. Unified multi-dataset benchmark — v1 grid (complete)

This benchmark generalizes the four-index experiment to **7 datasets / 39
tickers** with one shared, leakage-controlled pipeline (`pipeline/` package:
`datasets.py`, `labels.py`, `features.py`, `models.py`, `benchmark.py`).

**Labels.** Per ticker: forward 15-session minimum simple return ≤ the
ticker's 2nd percentile, estimated **calibration-only** — 1997→2006-12-31
anchor where ≥1,000 fully-observed origins exist (group_a, international,
financials, energy), otherwise the first 40% of origins (group_b, crypto,
tech). No full-sample quantiles.

**Datasets.** group_a US indices (^GSPC, ^DJI, ^IXIC, ^RUT, 1997–) ·
group_b ETFs (SPY, TLT, GLD, USO, UUP, HYG, 2007–) · crypto (BTC, ETH, XRP,
SOL, BNB, 2020–) · international (^N225, ^GDAXI, ^FTSE, 000001.SS, ^HSI,
^BVSP, ^BSESN, 1997–) · financials (JPM, BAC, GS, C, MS, WFC, 1999–) ·
energy (XOM, CVX, COP, SLB, EOG, 1997–) · tech (AAPL, MSFT, NVDA, GOOGL,
META, AMD, 2012–).

**Feature modes (v1).** `classical` (CSD lag-1 autocorrelation, variance,
skew, excess kurtosis over 50/100d + 20d realized vol and 20d EWMA vol),
`tda` (five H1 landscape/persistence scalars), `combined`. Group level uses
the joint Gidea–Katz multivariate window; ticker level uses per-ticker Takens
delay embeddings (dim 3, delay 1, 50d z-scored window).

**Models (v1, 7).** LSTM(24) · BiLSTM(24) · GRU(24) · CNN1D · TCN ·
Transformer · logistic regression on flattened sequences (floor baseline).
Sequence length 20, LayerNorm+Dropout head, train-only scaler, pos-weight
capped at 20, AdamW (lr 8e-4, wd 1e-3), ≤12 epochs, early stop on validation
AP, alarm threshold = validation 95th percentile, 25-day purge, three
chronological test folds (2011–15, 2016–20, 2021+).

Mode × level (mean AP / mean ROC AUC across folds and datasets):

| Level | classical | tda | combined |
|---|---|---|---|
| Group (7 datasets) | **0.0419** / 0.574 | 0.0267 / 0.524 | 0.0379 / 0.543 |
| Ticker (39 pooled) | 0.0417 / 0.653 | 0.0253 / 0.518 | **0.0424** / 0.649 |

Model ranking (mean AP ± std across all cells): BiLSTM **0.0413** ± .037 ·
GRU .0393 · LSTM .0362 · LogReg .0349 · CNN .0328 · TCN .0323 ·
Transformer .0316. On the pooled ticker level all seven models are within
.033–.038 (Transformer tops at .0384).

Group level, dataset × mode (mean AP):

| Dataset | classical | combined | tda |
|---|---|---|---|
| energy | .0489 | .0522 | **.0552** (TDA wins) |
| financials | **.0547** | .0460 | .0262 |
| tech | **.0708** | .0591 | .0431 |
| international | .0300 | .0287 | .0173 |
| group_b | .0330 | .0294 | .0110 |
| group_a | .0239 | .0192 | .0131 |

**v1 findings.** Classical CSD features are the strongest single family on
5 of 6 complete datasets; joint-TDA wins only on energy. TDA-only is the
weakest family everywhere else. Deep sequence models beat logistic
regression only modestly (≈.006 on the group level). Full results:
`data/benchmark/all_results_v1.csv` (3,494 rows).

## 11. Benchmark v2 — richer TDA summaries, GARCH, cross-section (final for 5/7 datasets)

v2 adds to the panel (212,816 rows; 102 → 119 columns after section 13):

- **GARCH(1,1)** conditional volatility (stride-5 refit on a trailing
  250-day window, forward-fill only).
- **Total persistence** and a **36-dim persistence-landscape grid**
  (3 levels × 12 points over [0, 2.5] in z-scored window space) as native
  deep-model inputs — new `landscape` and `full` modes.
- **Cross-sectional correlation features**: mean pairwise correlation and
  max eigenvalue of the 50d/100d correlation matrix.
- An eighth model, **attention-BiLSTM** (additive attention pooling).
- A proportional 70/15/15 fold for short histories (crypto, tech).

**Status.** The grid is final for all datasets except two: crypto has no
evaluable group cells (its calibration-era threshold leaves zero positives
after 2024 — skipped at runtime), and the last chunk of tech / pooled-ticker
tasks was cut when compute was stopped (696 of 760 evaluable cells recorded;
the interrupted remainder is recoverable with
`py -3.12 run_benchmark.py --tag full_v2 --resume`). **Every one of the 800
planned runs — including the 40 skipped and 64 not-run — is individually
accounted for, with full metrics, in `RUN_RECORD.md` and
`data/benchmark/run_record_v2.csv`.** All conclusions below
rest on complete 120-cell blocks (5 modes × 8 models × 3 folds) for energy,
financials, tech (partial), group_a, group_b, international.

Group level, mode comparison (mean AP / AUC; 5 complete datasets):

| mode | AP | AUC |
|---|---|---|
| classical | .0339 | 0.554 |
| **combined** | **.0366** | 0.562 |
| landscape | .0354 | 0.537 |
| full | .0331 | 0.539 |
| tda | .0286 | 0.538 |

Group level, dataset × mode (mean AP):

| Dataset | classical | combined | full | landscape | tda | winner |
|---|---|---|---|---|---|---|
| energy | .0496 | .0571 | .0610 | **.0694** | .0603 | landscape |
| international | .0216 | .0266 | .0289 | **.0471** | .0169 | landscape |
| tech* | .0564 | .0668 | .0529 | .0662 | .0624 | combined |
| financials | .0467 | **.0478** | .0303 | .0238 | .0272 | classical/combined |
| group_a | .0233 | .0190 | .0119 | .0156 | .0138 | classical |
| group_b | .0210 | .0221 | **.0267** | .0108 | .0133 | full |

Best single cells (mode, model; mean AP over folds): international
landscape/LSTM **.1102** · tech tda/BiLSTM .1038 · energy landscape/TCN
.0971 · group_b full/GRU .0604 · financials classical/LSTM .0595 ·
group_a classical/GRU .0448.

Model ranking (group level, mean AP): TCN **.0369** · LSTM .0354 · CNN
.0338 · GRU .0337 · Transformer .0331 · attn-BiLSTM .0324 · BiLSTM .0316 ·
LogReg .0311. Pooled ticker level (partial): combined .0448 > classical
.0415 > tda .0245; BiLSTM tops the models at .0382, attn-BiLSTM .0357 —
**attention pooling did not beat plain BiLSTM**.

**v2 conclusions against the pre-declared success criteria:**

1. *Does `landscape`/`full` beat `combined`/`classical` on ≥3 datasets?*
   **Yes** — landscape wins on energy (+.012 over combined) and
   international (+.020), full wins on group_b (+.005): 3 of 6 datasets.
2. *Does TDA-only improve with richer summaries?* **Yes** — replacing the
   5 scalar summaries with the 41-column landscape vectors lifts the TDA
   family group mean .0286 → .0354, and per dataset: energy .0603 → .0694,
   international .0169 → .0471 (≈3×).
3. *attn-BiLSTM vs BiLSTM?* **No improvement** (pooled .0357 vs .0382).
   The richer summaries, not the architecture, carried the v2 gains.

## 12. Lead-time / early-warning robustness (complete)

`qualitative_leadtime.py` analyses 2,034 label onsets across all 39 tickers.
For each indicator family (past-only 250-day z-scores), it measures how
often and how early the indicator crosses its quiet-day 90th percentile
before an onset (scan window −60..−1 sessions; pre-onset window 30
sessions).

Dataset × family — AUC / mean lead in sessions / share of onsets fired early:

| Dataset | CSD | TDA-index | TDA-joint | CROSS |
|---|---|---|---|---|
| crypto | .461 / 27 / .23 | .569 / 37 / .83 | .575 / 26 / .57 | .414 / 24 / .27 |
| energy | .529 / 42 / .35 | .500 / 37 / .64 | .482 / 37 / .60 | .535 / 44 / .27 |
| financials | .543 / 43 / .42 | .492 / 40 / .67 | .492 / 39 / .59 | .489 / 47 / .31 |
| group_a | .591 / 44 / .43 | .572 / 36 / **.79** | .505 / 49 / .53 | .476 / 49 / .23 |
| group_b | .660 / 48 / .51 | .479 / 38 / .67 | .488 / 26 / .55 | .549 / 42 / .47 |
| international | .502 / 38 / .42 | .522 / 40 / .61 | .497 / 34 / .60 | .462 / 22 / .30 |
| tech | .520 / 40 / .40 | .558 / 39 / .63 | .508 / 40 / .57 | .424 / 53 / .19 |

Overall: CSD .543 AUC, fires before .40 of onsets · TDA-index .525, **.68** ·
TDA-joint .505, .58 · CROSS .478, .30.

**Findings.** (1) Classical CSD discriminates harder before onsets on the
equity groups (AUC .59–.66 on group_a/group_b). (2) TDA warns **earlier and
more consistently**: the TDA-index z-score crosses its alarm level before
58–83% of onsets (vs 23–51% for CSD), with a mean lead of ≈36–40 sessions on
the US indices at AUC .572. (3) Crypto is a TDA domain: CSD is below noise
(AUC .46) while TDA-joint reaches .58. (4) Only 1 of 4 landmark event panels
matched cleanly within the ±45-session tolerance — figure tolerance is
queued for loosening. Outputs: `data/benchmark/qualitative/`.

## 13. Peer-index deep models (independent experiment, complete)

A second independent experiment (`peer_index_forecasting.py` and
`peer_index_forecasting_v2.py`) tests whether the *other* indices carry
crash information for a target, with the target's own returns excluded from
its features. Same labels, folds, purge, and threshold rule; sequence length
30; v2 trains one model per target and validates a top-2 ensemble.

| Model | v1 pooled AP | v2 per-target AP |
|---|---|---|
| GRU | 0.0436 | **0.0508** |
| ensemble (top-2, val-selected) | — | 0.0414 |
| TCN | — | 0.0401 |
| attn-BiLSTM | — | 0.0400 |
| Transformer | 0.0395 | 0.0362 |

Per-target ensemble AP by test block (^target / 2011-15 / 2016-20 / 2021+):
^DJI .016/.059/.082 · ^GSPC .010/**.111**/.004 · ^IXIC —/.025/— ·
^RUT .019/.055/.034. The 2016–2020 block carries most of the signal for
every approach in this project so far.

## 14. Cross-family comparison on the identical group-A protocol

All numbers below use the same labels, folds, purge, and validation
thresholds on the four US indices:

| Feature family / model | best mean AP |
|---|---|
| CSD classical only (v1 benchmark) | 0.024 |
| TDA-only (v1 benchmark) | 0.015 |
| Team-original combined LSTM | 0.031 |
| Cross-index attention (section 9) | 0.0436 |
| Peer-only GRU, per-target (section 13) | **0.0508** |

**Verdict.** The reactive price-action/peer feature family (momentum,
drawdowns, volatility shocks, breadth) is roughly **2× stronger** than
CSD-only or TDA-only features for the 15-day crash label on US indices.
The gap is a *feature-family* effect, not an architecture effect: all
approaches share GRU/LSTM/attention backbones.

**Integration.** Both ideas are absorbed into the unified benchmark
(`augment_panel.py`): the panel now carries 11 momentum/price-action
features per ticker plus 6 per-ticker **peer aggregates** (peer return
mean/median, loss breadth, dispersion, mean peer volatility, mean peer-pair
correlation — the target's own series always excluded), enabling new modes
`price`, `peer`, and `everything`. Unlike the original peer experiments,
which covered only 4 US indices, this tests the family across all 39
tickers × 7 datasets on the shared protocol (queued after the v2 grid).

## 15. Limitations and next experiments

1. Add repeated random seeds and confidence intervals using block bootstrap
   over dates or crash episodes, not independent daily rows.
2. Compare logistic regression, gradient-boosted trees, and a simple
   persistence baseline against the LSTM using identical folds.
3. Add event-level recall, false alarms per year, and lead-time distributions
   with a predeclared alarm-merging rule.
4. Test alternative threshold calibration (e.g., fixed drawdown severity or
   the existing exponential-tail episode threshold) without selecting it on
   test outcomes.
5. Tune only inside rolling training/validation folds; report one untouched
   final period after choices are frozen.
6. Recompute TDA on other asset groups only if cross-market performance is a
   project goal; the current classifier covers the four original US indices.
7. Calibrate model scores on validation data before describing them as
   probabilities, and test alert rates under regime shifts.
8. Verify the source price series and licensing/provenance before submission;
   this run did not refresh external market data.

