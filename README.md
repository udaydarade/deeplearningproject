# Financial Crash Detection and TDA/CSD Research

This repository contains Python scripts and data for reproducing a financial
crash-detection study based on:

- persistent-homology / topological data analysis (TDA),
- critical-slowing-down (CSD) indicators,
- drawdown-based crash episodes and forward-return labels, and
- cross-asset extensions covering equities, crypto, and international markets.

The project is a research workspace rather than an installable Python package.
Most scripts read and write files below `data/`; plots and presentation
figures are written to `data/plots/` or `assets/`.

## Quick start

### 1. Create a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the scripts with the interpreter directly:

```powershell
py -3.12 <script-name>.py
```

Python 3.10+ is recommended. The dependency list includes the numerical,
plotting, data-download, GARCH, and persistent-homology libraries used by the
scripts.

### 2. Run the main cross-asset extension

```powershell
py -3.12 run_cross_asset_extension.py
```

This command downloads the sector data needed for the extension, reads the
existing Group B and Group C price files, computes the rolling features, and
writes results to `data/extension/`.

The download step needs an internet connection. `yfinance` is attempted first
and Stooq is used as a fallback. Failed tickers are reported rather than
being filled with synthetic or forward-filled values.

## Repository layout

```text
.
├── assets/                         Final figures and presentation assets
├── data/
│   ├── group_a_original_indices/   Original-index analysis outputs
│   ├── group_b_cross_asset/       Cross-asset equity inputs and outputs
│   ├── group_c_panel/              Crypto/international panel files
│   ├── plots/                      Group-level diagnostic plots
│   └── extension/                  Generated cross-asset extension outputs
├── paper_source/                   Reference figures and source material
├── crash_labels.py                 Drawdown episodes and crash labels
├── csd_baseline.py                 Rolling CSD indicators
├── download_market_data.py         Market-data download and preparation
├── episode_extension.py            Episode-level extension diagnostics
├── generate_final_assets.py        Final presentation figure generation
├── ismail_2022_reproduction.py     Reproduction-oriented analysis
├── paper_exact_detection.py        Paper-method detection diagnostics
├── paper_extension.py              TDA extension analysis
├── paper_reproduction_figures.py   Reproduction figures
├── paper_style_figures.py          Paper-style figures
├── presentation_figures.py         Presentation dashboards and figures
├── run_cross_asset_extension.py    Current n=50 cross-asset pipeline
├── tda_replication.py              Rolling TDA landscape features
└── two_period_paper_figures.py     Two-period comparison figures
```

## Recommended workflow

Run the scripts in this order when rebuilding the project from raw market
data:

1. `download_market_data.py` — download and normalize the market price data.
2. `tda_replication.py` — calculate rolling TDA landscape norms.
3. `csd_baseline.py` — calculate rolling CSD indicators, including GARCH
   volatility.
4. `crash_labels.py` — identify drawdown episodes and generate N10/N15/N20/N40
   labels.
5. `episode_extension.py` — produce episode-level diagnostics and lead-time
   comparisons.
6. The paper/reproduction scripts — generate analyses tied to the reference
   methodology.
7. `generate_final_assets.py` — regenerate the final figures in
   `assets/final/`.
8. `run_cross_asset_extension.py` — run the newer cross-asset extension.

Most scripts contain a `main()` function and can be run directly from the
repository root. They use paths relative to the script location, so running
from the root is recommended for consistent output locations.

## Cross-asset extension details

`run_cross_asset_extension.py` uses a 50-observation rolling window (`N = 50`)
for each market group:

- **Sector groups:** Financials, energy, and technology stocks downloaded from
  yfinance/Stooq, starting at `1997-01-01`.
- **Group B:** Existing `data/group_b_cross_asset/prices.csv`, using `SPY` as
  the reference return.
- **Group C crypto:** Existing `data/group_c_panel/crypto_prices.csv`, using
  `BTC-USD` as the reference return.
- **Group C international:** Existing
  `data/group_c_panel/international_prices.csv`, using `^N225` as the
  reference return.

For every complete rolling return window, the script writes:

- `TDA_L1_norm`, `TDA_L2_norm`, `TDA_entropy`, and `TDA_betti1_count`;
- `CSD_AC1`, `CSD_variance`, `CSD_skew`, and `CSD_garch_vol`; and
- `crash_label_N15`, based on the reference price's own empirical 2nd
  percentile of forward 15-trading-day minimum returns.

The output files are:

```text
data/extension/
├── cross_asset_panel.csv          Main long-format feature panel
├── data_download_summary.csv      Per-ticker download status and date ranges
├── group_summary.csv              Row counts, thresholds, and missing GARCH counts
├── report.txt                     Human-readable run summary
├── *_prices.csv                   Downloaded sector prices
├── *_returns.csv                  Downloaded sector log returns
└── *_diagnostics.png              TDA/CSD diagnostic plots
```

The extension intentionally uses complete observations only. It does not
forward-fill missing prices, fabricate failed tickers, or use a common crash
threshold across unrelated reference assets.

## Data and reproducibility notes

- Market data is external and can change when providers revise history.
- Record the run date and provider status from
  `data/extension/data_download_summary.csv` when sharing results.
- Existing CSV inputs under `data/` are treated as source data by the scripts;
  do not overwrite them unless you intentionally want to rebuild the study.
- Generated output can be safely recreated by rerunning the relevant script.
- The repository currently has no automated test suite. A basic validation is
  to run the intended script and inspect its report, row counts, and output
  files.

## Collaboration guidelines

Before opening a pull request:

1. Explain which script(s) were run.
2. Include the relevant report or summary output.
3. State whether data was downloaded again and which provider was used.
4. Avoid committing `.venv/`, Python caches, or other machine-local files.
5. Keep large source datasets and reference material separate from generated
   outputs when adding new files.

## Citation and reference material

Reference figures and the paper source used for reproduction work are stored
under `paper_source/`. Preserve the accompanying citation and licensing
information when redistributing those materials.

## Out-of-sample crash forecasting and interface

The forecasting deliverables use the four original US indices. Generate the
TDA/CSD feature panel, train three LSTM feature ablations, and create the
walk-forward results with:

```powershell
py -3.12 crash_forecasting.py
py -3.12 forecasting_qualitative.py
```

The target threshold for each index is its 2nd percentile forward 15-session
minimum return over the 1997–2006 calibration period. The threshold is held
fixed afterward. Test blocks are 2011–2015, 2016–2020, and 2021 onward; all
model features use information through the forecast date, and training is
purged 25 calendar days before each test block. Reports, trained model
weights, scalers, panel CSV, test metrics, threshold CSV, and event-aligned
figures are written under `data/forecasting/`.

To train on the larger cross-asset extension panel instead of only the four
original indices, first regenerate the panel if needed and then run:

```text
py -3.12 run_cross_asset_extension.py
py -3.12 cross_asset_forecasting.py
```

This produces `data/forecasting/cross_asset_walk_forward_metrics.csv`, saved
cross-asset LSTM artifacts, and a report with per-group row and label counts.
The cross-asset benchmark keeps the same chronological holdouts and boundary
purge used by the original benchmark.

### Peer-index deep-learning forecast

The separate `peer_index_forecasting.py` experiment tests whether the other
indices can warn about a target index crash. For each target, its own returns
are excluded from the input; the model receives only the other indices'
recent returns, volatility, dispersion, and correlation. It evaluates the
existing GRU and Transformer implementations without changing the original
CSD/TDA or model-zoo outputs:

```text
py -3.12 peer_index_forecasting.py
```

Results are written to `data/forecasting/peer_index_deep_learning_metrics.csv`.

The target-specific v2 experiment goes further by training one model per
target, adding richer peer-only dynamics, and selecting a validation ensemble
from GRU, attention-BiLSTM, TCN, and Transformer models:

```text
py -3.12 peer_index_forecasting_v2.py
```

Its results are written to
`data/forecasting/peer_index_deep_learning_v2_metrics.csv`.

To launch the CSV upload interface:

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m streamlit run forecast_app.py
```

Upload the generated `data/forecasting/modeling_panel.csv` as a demo, or a CSV
with `date`, `ticker`, and all feature columns listed in the saved model
bundle. The interface reports a risk score, not a calibrated probability.
See `FINAL_PROJECT_REPORT.md` for the methodology and current findings.

### Cross-index deep learning experiment

The independent cross-market experiment compares a target-only sequence model,
a peer-only temporal attention model that masks the target index's full input
history, and an all-index temporal attention model. It uses return, momentum,
volatility, downside-risk, and drawdown features. Run:

```powershell
py -3.12 cross_index_forecasting.py
py -3.12 -m streamlit run cross_index_app.py
```

This writes to `data/cross_index_forecasting/` without replacing the existing
TDA/CSD experiment. The app accepts daily closing-price CSVs with `date`,
`^GSPC`, `^DJI`, `^IXIC`, and `^RUT` columns. Results are documented in the
cross-index addendum in `FINAL_PROJECT_REPORT.md`. The interface validates and
charts the uploaded prices, compares target-history, peer-only, and all-index
scores, and provides a scored-CSV download. For a quick demo, choose the
bundled sample; for inference, the three trained checkpoints and their scalers
must be present under `data/cross_index_forecasting/`.

## V5, V6, and V7 synthesis

The later experiments remain additive: V5/V6/V7 outputs do not delete or
replace the earlier TDA/CSD, benchmark, or peer-model results.

### V5 full-history topology work

V5 inventories the global panel at approximately 159 assets and 1.17 million
asset-date rows spanning roughly 1927–2026. It evaluates scale-normalized
TDA, landscape dynamics, clustering/network blocks, and PRICE1D features. The
completed era results are in `data/v5/e1_results.csv`; E2 remains resumable
from `data/v5/e2_v7_final_checkpoint.csv`, and E3 deep-model output is written
to `data/v5/e3_results.csv` while its process runs.

### V6 asymmetric and real-time results

V6 adds asymmetric SEMIVAR/momentum features and evaluates causal Scheme W
alarms. The archived result tables under
`data/v6/archive_pre_forward_label_fix_20261025/` show the strongest verified
downside result in the available Scheme W summary for
`BEST_SET_SEMIVAR` at N=15: mean AP approximately 0.462 and mean ROC AUC
approximately 0.935 across seven datasets. At N=5, the corresponding mean AP
is approximately 0.339 and mean ROC AUC approximately 0.892. These are
summary values across folds/datasets, not per-ticker guarantees.

### V7 final synthesis

V7 completed all 240 configured cells:

- Horizons: N=5 and N=15, 2nd-percentile downside labels.
- Feature ladder: `CLASSICAL_CSD`, `SEMIVAR_MOM`, `PEER_CROSS`,
  `TDA_ONLY_NORM`, and `GRAND_COMBINED`.
- Models: Logistic Regression, LightGBM, GRU, and TCN.
- Deep-model seeds: 42, 43, and 44.
- Folds: 2011–2015, 2016–2020, and 2021+ with a 25-session purge.
- Alarm rule: causal rolling 250-session 95th-percentile threshold.

The complete V7 table is
`data/v7_final/v7_ablation_results.csv`. The corrected event-level forecast
ledger is under `data/v7_final/precrash_ledger/`. In that ledger, a model is
counted as detecting a crash only when an alarm occurs strictly before
`crash_start`; an alarm during an already active crash is not a successful
prediction.

The V7 results are complete for their configured folds, but they should not be
described as a full-history test over every date from 1927 onward until the
additional historical V7 folds are run. The full-history observed crash
population and the model-score coverage population are reported separately in
`data/v7_final/crash_stats/observed_crash_events.csv`.

### Headline findings currently supported by saved outputs

- The V6 real-time downside SEMIVAR configuration is the strongest broad
  result in the saved Scheme W summary: at N=15 it reaches mean AP about
  **0.462** and mean ROC AUC about **0.935** across seven dataset folds.
- At N=5, the same V6 configuration reaches mean AP about **0.339** and mean
  ROC AUC about **0.892**.
- V7 contains a complete 240-cell ablation matrix. One illustrative
  N=15/2016–2020 cell, `PEER_CROSS` + TCN, has AP **0.0555**, ROC AUC
  **0.7893**, and event recall **1.000** in that fold; this is a fold-specific
  result, not a universal or full-history rate.
- The V7 strict pre-crash ledger evaluates alarms before onset in 1-, 5-, 10-,
  20-, and 60-session windows. It deliberately excludes alarms that occur
  during an already active crash.
- The original TDA replication remains descriptive: its S&P 500 H1 landscape
  norms show their largest saved peaks around the 2000 dot-com period, but
  that file is not a causal out-of-sample forecasting claim.

## Updated Streamlit demos

Launch the original-indicator demo:

```powershell
py -3.12 -m streamlit run forecast_app.py
```

It now shows the V7 ablation summary and a strict pre-crash event lookup before
the upload scorer. Upload scoring still uses the saved LSTM artifact and is
clearly labeled as a score, not a calibrated probability.

Launch the peer/cross-index demo:

```powershell
py -3.12 -m streamlit run cross_index_app.py
```

It now shows the related V7 summary before accepting a four-index price CSV.
Both apps keep the original upload workflows and do not retrain models.

## Reproducibility and report index

- [MASTER_DATA_AND_METHODS_BIBLE.md](MASTER_DATA_AND_METHODS_BIBLE.md) —
  compact evidence-backed methods, environment, status, and result index.
- [V6_EXPERIMENT_REPORT.md](V6_EXPERIMENT_REPORT.md) — V6-only report.
- [PRECRASH_MODEL_LEDGER.md](PRECRASH_MODEL_LEDGER.md) — strict forecast
  detection definition and event-level outputs.
- [CRASH_BY_CRASH_MODEL_LEDGER.md](CRASH_BY_CRASH_MODEL_LEDGER.md) — legacy
  in-crash audit ledger, retained for transparency but not used as evidence of
  advance prediction.
