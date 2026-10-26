# AUDIT NOTICE (2026-10-25)

The original V6 results below used forward labels shifted by the horizon. They are preserved for history in `data/v6/archive_pre_forward_label_fix_20261025/` and must not be used as early-warning evidence. Corrected labels and model reruns are in progress; see `data/v6/current_status.csv` and the task logs for live status. This report will be replaced with corrected results after reruns finish.

---

# V6 experiment record

This document records only the V6 work. V5 files, outputs, and running processes
are not part of this report and were not modified by V6.

## Scope

V6 tested whether crash and upward-move early warning behaves differently when:

- the horizon is varied across `N = 3, 5, 10, 15, 30`;
- the severity percentile is varied across `p = 2, 3, 4, 5, 6, 8, 10%`;
- upward moves are treated separately from crashes;
- past-only semivariance and classical features are used;
- blocked historical validation is compared with real-time walk-forward validation.

The evaluator used alarm-day budgets rather than event-count budgets. Metrics include
event recall, random-baseline event recall, excess event recall, AP, AUC, alarm-day
rate, and off-budget status.

## Experiment inventory

| Experiment | Status | Main output |
|---|---|---|
| V6-0 inventory | Complete | `data/v6/label_inventory.csv`, `assets_alive_by_year.csv` |
| V6-1 severity labels | Complete | `data/v6/labels/`, `label_prevalence.csv` |
| V6-2 SEMIVAR block | Complete | `data/v6/semivar.parquet`, `panel.parquet` |
| V6-3 blocked E-severity sweep | Complete | `data/v6/sweep_e_sev.csv` |
| V6-4 real-time Scheme-W | Complete | `data/v6/scheme_w_results.csv` |
| V6-T1 native down/up comparison | Complete | `data/v6/t1_native_direction.csv` |
| V6-T2 N=5 transfer | Complete | `data/v6/t2_cross_transfer_n5.csv` |
| V6-S5 SEMIVAR value comparison | Complete through B3 vs SEMIVAR | `data/v6/scheme_w_summary.csv` |
| External universe downloads | Not run | No V6 download output |
| T3 conditional direction model | Not run | Requires a separate direction-conditioned fit |
| S1 ensemble | Not run | Component score histories were not saved |
| S2 two-stage model | Not run | Component score histories were not saved |
| S3 per-class models | Not run | Not launched |
| S4 multitask GRU | Not run | Not launched |

The machine-readable status is in `data/v6/current_status.csv`,
`data/v6/v6_remaining_status.csv`, and `data/v6/completion_status.csv`.

## V6-0: inventory

Script: `v6_0_inventory.py`

Observed V3 source data reused by V6:

- 1,168,380 panel rows.
- 159 assets.
- Date range: 1927-12-30 through 2026-10-23.
- 1,044,519 rows with valid BASE labels.
- 20,445 BASE positive asset-days.
- BASE prevalence among valid rows: 1.957%.

Asset classes:

| Class | Assets |
|---|---:|
| bond | 11 |
| commodity | 19 |
| country ETF | 29 |
| crypto | 11 |
| equity index | 32 |
| FX | 14 |
| stock | 17 |
| US ETF | 26 |

Outputs:

- `data/v6/assets_by_class.csv`
- `data/v6/asset_first_year.csv`
- `data/v6/assets_alive_by_year.csv`
- `data/v6/label_inventory.csv`
- `data/v6/prior_outputs_inventory.csv`
- `data/v6/v3_feature_sets.csv`

## V6 evaluator tests

Scripts: `v6_3_eval_tests.py`, `pipeline_v6/v5_eval_copy.py`

The copied V5 evaluator was kept inside `pipeline_v6/`. The blocked-split smoke
test passed:

- 11 era labels recognized.
- Training, validation, and test sets were non-empty.
- Training dates did not overlap test dates.

The test output was printed by `v6_3_eval_tests.py`; V6 did not alter the V5
evaluator.

## V6-1: severity labels

Scripts: `pipeline_v6/labels.py`, `v6_1_labels.py`

For every asset and horizon, V6 computed a past-only severity rank for the worst
future return (`u_down`) and best future return (`u_up`). It then derived down,
up, and two-sided labels at all requested percentiles.

Generated:

- 159 per-asset Parquet label files.
- 5 horizons.
- 7 percentile levels.
- Down, up, and move labels.
- 4,099,179 asset-date rows represented in the label files.

Selected prevalence examples:

| N | p | Direction | Prevalence |
|---:|---:|---|---:|
| 3 | 2% | down | 0.465% |
| 3 | 2% | up | 0.455% |
| 3 | 2% | move | 0.915% |
| 3 | 5% | down | 1.136% |
| 3 | 5% | up | 1.125% |

The complete table is `data/v6/label_prevalence.csv`.

Unit checks saved in `data/v6/label_unit_tests.csv`:

- Toy severity calculation produced finite ranks.
- Five symmetry checks passed for `u_up(P)` versus `u_down(1/P)`.

### Label overlap

Output: `data/v6/labels_overlap.csv`

For N=15, p=2%:

- `P(up | down) = 1.265%`.
- `P(down | up) = 1.302%`.
- Rank correlation between down and up severity: `-0.6414`.
- Both-direction label share: `0.0207%`.
- Mismatch against the V3 BASE label: `13.43%` of shared rows.
- Shared rows: 1,168,380.

The V6 label is therefore not silently treated as identical to V3 BASE.

Class-level overlap is in `data/v6/labels_overlap_by_class.csv`. The highest
N=15,p=2 down prevalence was stocks at 2.126%; the lowest was crypto at 0.345%.

## V6-2: SEMIVAR block

Script: `v6_2_semivar.py`

Generated 18 past-only feature columns for all 159 assets, including:

- positive and negative semivolatility over 20 and 60 sessions;
- up/down log ratios;
- 20/60-session skew;
- largest gain and loss;
- 60/252-session close position;
- days since rolling high/low;
- volatility compression percentile;
- up and down streak lengths.

Output:

- `data/v6/semivar.parquet`
- `data/v6/panel.parquet`
- `data/v6/registry_v6.csv`

The feature block contained 4,099,179 asset-date rows. V6 did not add TDA
columns.

## V6-3: blocked E-severity sweep

Script: `v6_5_sweep.py`

Configuration:

- Scheme B blocked historical validation.
- 11 eras.
- GBM-lite regression on `-log(u)`.
- SEMIVAR plus classical features.
- 5% alarm-day budget.
- 200 random event-baseline simulations.

There were 110 completed cells: 2 directions Ã— 5 horizons Ã— 11 eras.
All completed cells were on-budget.

| N | Direction | Mean excess event recall | Mean AP | Mean AUC |
|---:|---|---:|---:|---:|
| 3 | down | 0.476 | 0.480 | 0.926 |
| 3 | up | 0.419 | 0.361 | 0.886 |
| 5 | down | 0.645 | 0.685 | 0.963 |
| 5 | up | 0.668 | 0.621 | 0.957 |
| 10 | down | 0.364 | 0.445 | 0.919 |
| 10 | up | 0.280 | 0.287 | 0.865 |
| 15 | down | 0.286 | 0.503 | 0.939 |
| 15 | up | 0.235 | 0.326 | 0.889 |
| 30 | down | 0.201 | 0.489 | 0.935 |
| 30 | up | 0.148 | 0.279 | 0.855 |

The strongest blocked-CV horizon was N=5 for both directions. These results
must not be interpreted as real-time performance because Scheme B uses later
historical eras in training.

Outputs:

- `data/v6/sweep_e_sev.csv`
- `data/v6/sweep_e_sev_summary.csv`

## V6-4: real-time Scheme-W

Script: `v6_6_realtime.py`

Configuration:

- Expanding walk-forward Scheme W.
- Test eras E4 through E10.
- N = 5, 15, 30.
- Down and up directions.
- B3 baseline:
  - 20-session realized volatility;
  - 60-session drawdown;
  - 1-session return.
- BEST_SET_SEMIVAR:
  - B3 plus SEMIVAR columns.
- E-severity GBM-lite.
- E-bin GBM-lite.
- RQ previous-250-session alarm rule.
- 5% alarm-day budget.

There were 168 completed cells:

`3 horizons Ã— 2 directions Ã— 2 feature sets Ã— 2 engines Ã— 7 eras`.

All completed cells were on-budget.

### Scheme-W aggregate results

| N | Direction | Feature set | Engine | Excess recall | AP | AUC |
|---:|---|---|---|---:|---:|---:|
| 5 | down | B3 | E-sev | 0.358 | 0.239 | 0.851 |
| 5 | down | SEMIVAR | E-sev | 0.317 | 0.335 | 0.886 |
| 5 | down | SEMIVAR | E-bin | 0.301 | 0.339 | 0.892 |
| 5 | up | B3 | E-sev | 0.209 | 0.144 | 0.763 |
| 5 | up | SEMIVAR | E-sev | 0.230 | 0.184 | 0.813 |
| 5 | up | SEMIVAR | E-bin | 0.232 | 0.195 | 0.832 |
| 15 | down | B3 | E-sev | 0.187 | 0.276 | 0.884 |
| 15 | down | SEMIVAR | E-sev | 0.236 | 0.462 | 0.935 |
| 15 | down | SEMIVAR | E-bin | 0.229 | 0.456 | 0.935 |
| 15 | up | B3 | E-sev | 0.149 | 0.153 | 0.780 |
| 15 | up | SEMIVAR | E-sev | 0.204 | 0.259 | 0.881 |
| 15 | up | SEMIVAR | E-bin | 0.208 | 0.263 | 0.888 |
| 30 | down | B3 | E-sev | 0.121 | 0.326 | 0.903 |
| 30 | down | SEMIVAR | E-sev | 0.110 | 0.434 | 0.928 |
| 30 | down | SEMIVAR | E-bin | 0.097 | 0.424 | 0.925 |
| 30 | up | B3 | E-sev | 0.070 | 0.144 | 0.766 |
| 30 | up | SEMIVAR | E-sev | 0.138 | 0.217 | 0.850 |
| 30 | up | SEMIVAR | E-bin | 0.126 | 0.216 | 0.853 |

Interpretation:

- SEMIVAR improved AP/AUC in every listed comparison.
- SEMIVAR improved excess event recall for up moves at N=5 and N=30 and for
  both directions at N=15.
- B3 remained slightly stronger on down excess recall at N=5 and N=30.
- No single feature set dominated every horizon and direction.

Outputs:

- `data/v6/scheme_w_results.csv`
- `data/v6/scheme_w_summary.csv`

## V6-T1: native direction comparison

Output: `data/v6/t1_native_direction.csv`

Native scores were evaluated against their own labels. At N=5 with SEMIVAR:

- Down E-severity: excess recall 0.317, AUC 0.886.
- Up E-severity: excess recall 0.230, AUC 0.813.

At N=15:

- Down E-severity: excess recall 0.236, AUC 0.935.
- Up E-severity: excess recall 0.204, AUC 0.881.

At N=30:

- Down E-severity: excess recall 0.110, AUC 0.928.
- Up E-severity: excess recall 0.138, AUC 0.850.

The relative advantage changes with horizon; crashes are not uniformly easier
than surges.

## V6-T2: cross-direction transfer

Script: `v6_10_transfer.py`

This was run for N=5 using real-time Scheme-W scores from both directions.
Per-row scores were saved before calculating transfer metrics.

Output:

- `data/v6/scheme_w_n5_scores.parquet`
- `data/v6/t2_cross_transfer_n5.csv`

Pooled transfer results:

| Source score | Target label | Rows | AUC |
|---|---|---:|---:|
| Down score | Up label | 894,760 | 0.188 |
| Up score | Down label | 894,760 | 0.172 |

Class-level AUCs were also saved. Transfer was below random in every listed
class, indicating that the directional scores are not interchangeable.

## S5: SEMIVAR value

The B3-versus-SEMIVAR comparison is contained in the Scheme-W summary.
SEMIVAR substantially increased AP and AUC, especially at N=15, but its event
recall advantage was not universal. Therefore the correct conclusion is:

**SEMIVAR adds ranking information, but its operational event-recall benefit
depends on direction and horizon.**

## Work explicitly not run

### External universe extension

Not run. V6 reused the complete V3 universe of 159 assets. No external data was
downloaded, and no new asset labels or features were added.

### T3 conditional direction

Not run. This requires a separate model trained only on non-both-direction
move rows, with its own saved causal scores.

### S1 ensemble

Not run. The required component score histories were not available in the
earlier runs, so no validation-weighted rank ensemble was fabricated.

### S2 two-stage model

Not run. It requires saved `P(move)` and conditional-direction scores.

### S3 per-class models

Not run. The completed run evaluated class-level diagnostics but did not fit
separate per-class production models.

### S4 multitask GRU

Not run. No V6 multitask sequence model was trained.

These omissions are recorded in `data/v6/v6_remaining_status.csv`; they are not
hidden or counted as completed experiments.

## Final V6 conclusion

The strongest reproducible real-time result is not the very high blocked-CV N=5
score. The more reliable conclusion comes from Scheme W:

1. N=5 is the most promising horizon overall.
2. SEMIVAR improves ranking quality substantially.
3. Operational event recall is regime-dependent.
4. Crash and surge scores do not transfer to one another.
5. V6 does not establish that one universal model or feature set wins across
   every horizon, direction, asset class, and era.

The experiment log was updated in `GLM_EXPERIMENT_LOG.md`.

