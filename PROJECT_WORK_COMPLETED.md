# Project work completed

Status date: 2026-10-25

This document records work supported by saved files in the repository. It does
not treat a source panel, a partial CSV, or an in-crash alarm as proof of a
completed forecasting experiment.

## 1. Data and universe

- V5/V6 global source panel: approximately 159 assets and 1.17 million
  asset-date rows, spanning approximately 1927–2026.
- V6 asset classes are represented in `data/v6/assets_by_class.csv`:
  equity indices, country ETFs, US ETFs, commodities, stocks, FX, bonds, and
  crypto.
- Full-history observed crash population:
  - 8,099 observed event rows.
  - N=5 and N=15 downside labels at the 2nd percentile.
  - 159-asset label population.
- Full event population is separate from model-score coverage. An observed
  event without a saved model score is not counted as a model miss.

## 2. V1–V4 work

- Original CSD and TDA benchmark pipelines were built and retained.
- V2 landscape experiments established the 36-dimensional landscape feature
  representation.
- Peer/cross-index experiments were added without deleting the earlier models.
- V3/V3.1 repaired the earlier excessive alarm-rate problem and retained leak
  tests and score files.
- V4 audit and rebuild completed:
  - Metric tests passed.
  - Causal leak test passed.
  - PRICE1D hand checks and brute-force comparisons passed.
  - Scale-law checks passed.
- The original TDA replication report remains descriptive and is preserved in
  `data/group_a_original_indices/tda_replication_report.txt`.

## 3. V5 completed work

- V5 inventory and era design were completed.
- V5 E1 contains 572 saved result rows and evaluates the defined historical
  eras E0–E10.
- V5 feature blocks were generated or retained for:
  - Scale-normalized TDA landscapes.
  - Landscape dynamics.
  - Cluster/network features.
  - PRICE1D topology.
  - Takens/volatility features.
- V5 diagnostics include raw-scaling, proxy-redundancy, signed-AUC, quality,
  leak, and scale checks.
- V5 E3 has 116 saved result rows and a validation file, but no completion
  marker proving that the entire intended matrix finished.

## 4. V6 completed work

The authoritative corrected V6 outputs are under:

`data/v6/archive_pre_forward_label_fix_20261025/`

Completed items:

- Corrected forward crash labels.
- Blocked E-severity sweep:
  - 110 saved cells.
  - Horizons include N=3, 5, 10, 15, and 30.
- Real-time Scheme W:
  - 168 saved cells.
  - N=5, 15, and 30.
  - Downside and upside directions.
  - B3 and SEMIVAR feature sets.
  - Causal alarm evaluation.
- Native down/up comparison.
- SEMIVAR value comparison against B3.
- Corrected V6 result tables and summaries.

### Strongest saved V6 headline

In the corrected Scheme W summary, the strongest broad downside result is:

- `BEST_SET_SEMIVAR`, N=15, E-bin:
  - Mean AP approximately 0.462.
  - Mean ROC AUC approximately 0.935.
- At N=5:
  - Mean AP approximately 0.339.
  - Mean ROC AUC approximately 0.892.

These are aggregate summary values across the saved seven-dataset Scheme W
rows, not guarantees for every asset or every historical date.

## 5. V7 completed work

V7 status file reports 240/240 configured cells complete:

- Horizons: N=5 and N=15.
- Feature sets:
  - `CLASSICAL_CSD`
  - `SEMIVAR_MOM`
  - `PEER_CROSS`
  - `TDA_ONLY_NORM`
  - `GRAND_COMBINED`
- Models:
  - Logistic Regression.
  - LightGBM.
  - GRU.
  - TCN.
- GRU/TCN seeds: 42, 43, and 44.
- Folds: 2011–2015, 2016–2020, and 2021+.
- Purge: 25 sessions.
- Alarm rule: causal rolling-250-session 95th-percentile rule.

V7 outputs:

- `data/v7_final/v7_ablation_results.csv`
- `data/v7_final/v7_bootstrap_significance.csv`
- `data/v7_final/scores/`

V7 is complete for its configured folds. It is not a complete test over every
historical date from 1927 onward.

## 6. Corrected forecast-event interpretation

The strict pre-crash ledger was created under:

`data/v7_final/precrash_ledger/`

A model detection is valid only when an alarm occurs strictly before
`crash_start`. Alarms during an active crash are not counted as advance
predictions.

The original in-crash ledger remains preserved for auditability, but it must
not be used as evidence of successful early warning.

## 7. Reports and interfaces completed

- `README.md` updated with project workflow, V5/V6/V7 findings, limitations,
  and demo instructions.
- `MASTER_DATA_AND_METHODS_BIBLE.md` compact version generated and kept below
  5 MB.
- `forecast_app.py` updated with V7 summary and strict pre-crash event lookup.
- `cross_index_app.py` updated with related V7 benchmark summary.
- V6, interruption, full-history, comparative, crash-statistics, and
  crash-by-crash reports are preserved.
