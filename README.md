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
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the scripts with the interpreter directly:

```powershell
.\.venv\Scripts\python.exe <script-name>.py
```

Python 3.10+ is recommended. The dependency list includes the numerical,
plotting, data-download, GARCH, and persistent-homology libraries used by the
scripts.

### 2. Run the main cross-asset extension

```powershell
python run_cross_asset_extension.py
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
