# TDA vs Classical Crash Prediction â€” Experiment Log

> Living log of experiments, kept so markers/results survive model or agent switches.
> **Agent:** GLM-5.3-Flash Free (OpenCode harness) Â· Dates: Oct 2026.
> **Rule:** append every experiment here with config, raw result pointers, and findings.

Full project context: `README.md`, `FINAL_PROJECT_REPORT.md` (replication stage work by the team). This file covers the **prediction benchmark** phase built by the agent.

---

## 1. Environment & how to reproduce

- **Python:** `py -3.12` (global env) â€” torch 2.8.0+cpu, sklearn 1.9.1, pandas 2.3.3, numpy 2.4.6, ripser (installed Oct 2026), arch 8.0.0 (installed Oct 2026).
- **Note:** the repo's `.venv` has ripser/yfinance/arch but NO torch â€” the benchmark runs entirely in `py -3.12`.
- **20-core CPU machine**; worker pools use 8 processes, each with `torch.set_num_threads(1)`.

```powershell
py -3.12 compute_features.py        # rebuild data/benchmark/unified_panel.csv (background, ~30 min with GARCH)
py -3.12 run_benchmark.py           # full grid, appends to data/benchmark/all_results.csv
py -3.12 analyze_benchmark.py       # tables + figures from all_results.csv
py -3.12 qualitative_leadtime.py    # early-warning lead-time analysis
```

---

## 2. Pipeline architecture (built Oct 2026)

| File | Role |
|---|---|
| `pipeline/datasets.py` | Dataset registry: 7 market datasets (name â†’ prices CSV). Add a market = add a line. |
| `pipeline/labels.py` | Objective crash labels: forward N15-session min simple return â‰¤ per-ticker 2nd pct. Calibration rule: 1997â€“2006 anchor if â‰¥1k origins, else first 40% of origins (crypto/tech). |
| `pipeline/features.py` | CSD block (AC1/variance/skew/kurt w50+w100, realized vol w20, EWMA vol, GARCH(1,1) stride-refit); TDA joint + per-ticker Takens (summaries + 36-dim landscape grid); cross-sectional (mean pairwise corr, max eig). |
| `pipeline/models.py` | Model zoo: logreg, LSTM, BiLSTM, attn-BiLSTM (additive attention pooling), GRU, CNN, TCN, Transformer. Shared trainer: train-only scaler, pos_weight cap 20, early-stop on val AP (patience 4). |
| `pipeline/benchmark.py` | Walk-forward evaluation: 3 calendar folds (2011â€“15, 2016â€“20, 2021+) + proportional 70/15/15 fold for short-history datasets (crypto, tech); 25-day purge; 2-y val; threshold = val 95th pct (~5% alarm days). |
| `compute_features.py` | Builds `data/benchmark/unified_panel.csv` (parallel: 7 dataset tasks + 39 ticker tasks). |
| `run_benchmark.py` | CLI. Levels: `group` (per dataset) and `ticker` (pooled across all 39 tickers, per-ticker breakdown). |
| `analyze_benchmark.py` | Summary CSVs + heatmaps + model-ranking bars. |
| `qualitative_leadtime.py` | Event-study: discrimination AUC of family z-scores pre-onset + lead time (sessions before onset where family z exceeds quiet-day 90th pct). Events: dotcom 2000, GFC 2008, covid 2020, rate-shock 2022. |

**Datasets** (all under `data/`): group_a 4 US indices (1997â€“), group_b 6 cross-asset ETFs (2007â€“), crypto 5 coins (2020â€“), international 7 indices (1997â€“), financials 6 (1999â€“), energy 5 (1997â€“), tech 6 (2012â€“). 39 tickers, 212k panel rows.

**Feature modes** (v2): `classical` (+cross-corr features at group level), `tda` (5 scalars), `landscape` (41: scalars + 3Ã—12 landscape grid), `combined` (classical + tda scalars), `full` (everything, 61 features).

---

## 3. Experiment log

### EXP-1 Â· v1 baseline grid (7 models Ã— 3 modes Ã— 2 levels) â€” COMPLETE (490/504 tasks; last ~14 ticker tasks not run)

- **Config:** unified panel v1 (no GARCH, no landscape grid, no cross-corr features). Modes: classical / tda / combined. Seq len 20, hidden 24, 12 epochs, batch 512. Crypto group skipped (calendar folds infeasible â€” fixed in v2 with proportional fold).
- **Raw results:** `data/benchmark/all_results_v1.csv` (3494 rows), transcript `data/benchmark/benchmark_full.log`.
- **Label prevalence â‰ˆ 2%** â†’ AP is the metric of record (baseline = prevalence).

**Mean test AP by mode/level (across models & folds):**

| Level | classical | tda | combined |
|---|---|---|---|
| group | **0.0419** | 0.0267 | 0.0379 |
| ticker (pooled) | 0.0417 | 0.0253 | **0.0424** |

**Mean test AP by model (across datasets/levels/folds):**

| model | AP |
|---|---|
| bilstm | **0.0413** |
| gru | 0.0393 |
| lstm | 0.0362 |
| logreg | 0.0349 |
| cnn | 0.0328 |
| tcn | 0.0323 |
| transformer | 0.0316 |

**Group level, dataset Ã— mode (mean AP):**

| dataset | classical | combined | tda |
|---|---|---|---|
| energy | 0.0489 | 0.0522 | **0.0552** |
| financials | **0.0547** | 0.0460 | 0.0262 |
| tech | **0.0708** | 0.0591 | 0.0431 |
| international | 0.0300 | 0.0287 | 0.0173 |
| group_b | 0.0330 | 0.0294 | 0.0110 |
| group_a | 0.0239 | 0.0192 | 0.0131 |

**Findings (v1):**
1. Classical CSD indicators are the strongest single feature family on 5/6 datasets; joint TDA-only is weak except **energy**, where TDA-only is best (0.055) and the combined mode also improves (0.052 > 0.049 classical).
2. Combined (CSD+TDA) â‰ˆ classical on average â€” TDA adds value only on commodity-heavy/energy segments so far.
3. Recurrent models lead; the Transformer underperforms at 2% prevalence on this data volume. LogReg stays close deep models (gap â‰ˆ 0.006 AP) â€” modest genuine temporal signal.
4. tech is the most predictable dataset (possible survivorship/listing effects â€” flagged for the report).

### EXP-2 Â· v2 feature upgrade + full re-run â€” RUNNING â†’ results pending

- **Added** per the team's plan: GARCH(1,1) conditional vol (stride-5 refit on trailing 250d, ffill-forward only), **total persistence**, **36-dim landscape-grid vectors** (3 levels Ã— 12 points, z-scored window space) as deep-model input channels, **cross-sectional correlation features** (mean pairwise corr + max eigenvalue of the correlation matrix, w50/w100), **attn-BiLSTM** model (additive attention pooling), **proportional 70/15/15 fold** for short histories (crypto, tech). Panel v2: 212,816 rows Ã— 102 cols (before augmentation).
- Main grid (5 modes Ã— 8 models Ã— 2 levels Ã— folds = 800 tasks) running in background; interim (first ~60% of tasks): classical 0.0328 / combined 0.0365 / full 0.0325 / landscape 0.0299 / tda 0.0271 mean AP.
- Success criteria: does `landscape` or `full` beat `combined`/`classical` on â‰¥3 datasets; does TDA-only improve with richer summaries; attn-BiLSTM vs BiLSTM pooled.

### EXP-3 Â· qualitative lead-time / early-warning robustness â€” COMPLETE (Oct 24)

- **Raw outputs:** `data/benchmark/qualitative/` (`family_stats.csv`, `family_stats_dataset.csv`, `leadtime_report.txt`, event-aligned figures â€” only 1 landmark matched an onset cleanly; 3 others missed the Â±45-session tolerance â€” revisit later).
- **Setup:** past-only 250d z-scoring; each label onset (quiet 30-session lead-in) â†’ discrimination AUC of family z-score pre-onset vs quiet days + lead time (first quiet-day-90th-pct crossing scanned over 60 sessions before onset). 2,034 onsets across 39 tickers.

**Per dataset (mean AUC / mean lead in sessions / share of onsets fired early):**

| dataset | CSD | TDA-index | TDA-joint | CROSS |
|---|---|---|---|---|
| crypto | .461 / 27 / .23 | **.569 / 37 / .83** | **.575 / 26 / .57** | .414 / 24 / .27 |
| energy | .529 / 42 / .35 | .500 / 37 / .64 | .482 / 37 / .60 | **.535 / 44 / .27** |
| financials | **.543 / 43 / .42** | .492 / 40 / .67 | .492 / 39 / .59 | .489 / 47 / .31 |
| group_a | **.591 / 44 / .43** | .572 / **36** / **.79** | .505 / 49 / .53 | .476 / 49 / .23 |
| group_b | **.660 / 48 / .51** | .479 / 38 / .67 | .488 / 26 / .55 | .549 / 42 / .47 |
| international | .502 / 38 / .42 | .522 / 40 / .61 | .497 / 34 / .60 | .462 / 22 / .30 |
| tech | .520 / 40 / .40 | **.558 / 39 / .63** | .508 / 40 / .57 | .424 / 53 / .19 |

**Findings (EXP-3):**
1. **CSD discriminates more strongly on the traditional equity datasets** (group_a .591, group_b .660), consistent with the AUC ordering in EXP-1.
2. **TDA-index fires early much more consistently** (lead_share .57â€“.83 vs CSD .23â€“.51) â€” topology is the "watches quietly but lights up" detector; CSD is the "discriminates hard when it fires" detector. On **group_a**, TDA-index achieves AUC .572 with a mean onset lead of **36 sessions** and fires before **79%** of crashes â€” the strongest "earlier warning" argument for topology on the paper's original markets.
3. **Crypto is a TDA domain**: with no volatility classical benchmark equivalent, TDA families get AUC â‰ˆ .57 (best for crypto) while CSD fails (AUC .46 Â« 0.5).
4. CROSS (correlation-structure) features are weak in raw event-z terms except energy/group_b.

### EXP-4 Â· independent agent approaches (peer/momentum features) â€” AGENT RUNS REVIEWED

Two other agents explored non-CSD/non-TDA feature families **on the same group-A protocol** (identical labels via `data/forecasting/label_thresholds.csv`, same 3 folds, same purge, val-95th-pct thresholds):

**Agent A â€” `cross_index_forecasting.py` (temporal cross-attention over the 4 US indices):**

| model | AP (mean 3 folds) | AUC |
|---|---|---|
| all_indices (attention over all 4) | **0.0436** | 0.7637 |
| other_only (peer-only, target zeroed) | 0.0366 | 0.7538 |
| self_only (target LSTM) | 0.0285 | 0.7179 |

**Agent B â€” `peer_index_forecasting.py` (pooled) and `peer_index_forecasting_v2.py` (per-target + val-selected top-2 ensemble):**

| model | AP (v1 pooled) | AP (v2 per-target) |
|---|---|---|
| GRU | 0.0436 | **0.0508** |
| ensemble_top2 | â€” | 0.0414 |
| attn_bilstm | â€” | 0.0400 |
| transformer | 0.0395 | 0.0362 |
| tcn | â€” | 0.0401 |

**My unified benchmark on the same group-A data (v1/v2 modes):** classical 0.023â€“0.024, combined 0.019, tda/landscape/full 0.012â€“0.016 â†’ **the peer/momentum family beats my CSD-only and TDA-only feature sets by ~2x on group A.**

**Findings (EXP-4):**
1. The winning ingredient is the **feature family, not the architecture:** reactive price-action features (momentum 5/20/60d, drawdowns, vol shocks, breadth) carry far more 15-day-horizon signal than slow CSD rolling stats or H1 TDA summaries.
2. Cross-index information genuinely helps: other_only > self_only (0.0366 vs 0.0285), all_indices 0.0436, per-target peer GRU 0.0508.
3. High variance across folds (e.g. peer-v2 ^GSPC fold-2016-20 AP=0.11 vs â‰ˆ0 in fold-2011-15) â€” crashes cluster in a couple of regimes; report should emphasize fold-level dispersion, not just means.

**Integration decision:** absorbed both ideas into the unified framework via `augment_panel.py` â€” panel now 212,816 rows Ã— 119 cols, adding `price_*` (11 momentum features) and per-ticker `peer_*` (6 peer aggregates, own series excluded) columns; new benchmark modes `price`, `peer`, `everything`. This Test them across all 39 tickers Ã— 7 datasets (the agents only ran 4 indices), scheduled as the supplement run (EXP-5) after the main grid finishes.

### Engineering fixes & housekeeping (Oct 24â€“25)

- Ticker-level task duplication fixed (`build_tasks` now uses one pooled `ALL` dataset for the ticker level; 1520 â†’ 800 tasks).
- Zero-positive test windows are skipped (`run_task` early-out) â€” crypto group labels calibrated on the 2020â€“22 crash era leave no positives after 2024-09 (documented caveat, not a bug).
- `--resume` flag added: rerun the same log-suffixed command and completed overall cells are skipped.
- A stalled/partial v2 run was archived as `data/benchmark/all_results_v2_partial.csv`; the definitive run is `data/benchmark/all_results.csv` (fresh, tagged `full_v2`).
- `augment_panel.py` verified: all 17 new columns 0% NaN on recent dates.

---

## 4. Dataset/engineering quirks & decisions (append freely)

- Old `data/extension/cross_asset_panel.csv` uses FULL-sample label quantiles (slight leakage) + reference-only labels; the new pipeline per-ticker labels with calibration-only thresholds and is the benchmark of record.
- Group-level features (joint TDA, cross-corr) exist only on inner-join dates of the group; ticker-level runs ignore them.
- GARCH refits every 5 sessions and ffills forward â€” the standing estimate is always a past-only quantity (no leak).
- Landscape grid fixed at [0, 2.5] in z-scored return units; H1 bars from Vietorisâ€“Rips on window-standardised points; windows with <3 bars are zero-padded to the 36-dim vector.
- `py -3.12` was missing `ripser`/`arch` and both were installed there (Oct 2026).

## 5. Open TODO queue

- [ ] v2 benchmark run + analysis (EXP-2)
- [ ] Qualitative lead-time analysis (EXP-3)
- [ ] TDA signal lead-time comparison figure for the final report
- [ ] Interface (`forecast_app.py`) update to the new models/modes
- [ ] README/report refresh with v2 numbers

---

## CLOSE-OUT (Oct 25) â€” session ended; no further runs

**Final grid state (v2, tag `full_v2`):** 696 of 800 planned runs completed
with metrics; 40 crypto group runs skipped (zero positives in the 2024â€“26
test window â€” structurally unevaluable, documented above); 64 pooled-ticker
runs enqueued but not executed (compute stopped at user request). The tech
group is complete (40/40). **v1 final state:** 406/420 done; the 14 not-run
are the last pooled-ticker cells.

**Per-run records (one row per planned run, every model, every fold):**
`RUN_RECORD.md` (human-readable tables: AP per fold + mean + status),
`data/benchmark/run_record_v2.csv` (800 rows, full metrics or explicit
status), `data/benchmark/run_record_v1.csv` (420 rows). Raw per-ticker
breakdowns: `all_results.csv` (5,160 rows) / `all_results_v1.csv` (3,494).
`final_analysis.py` regenerates the summary aggregates; `dump_all_results.py`
and `build_run_record.py` regenerate the other artifacts from the raw CSVs.

**Final v2 numbers (group level, complete cells):** combined .0366 best
overall mode; landscape wins energy (.0694) and international (.0471) and
full wins group_b (.0267) â†’ TDA-rich modes beat classical/combined on 3 of 6
datasets (success criterion #1 met); landscape > tda scalars everywhere
(criterion #2 met); attn-BiLSTM did not beat BiLSTM (criterion #3: no).
Best cells: international landscape/LSTM .1102, tech tda/BiLSTM .1038,
energy landscape/TCN .0971.

**Final cross-family verdict (group A, same labels/folds):** peer/momentum
features (agents' experiments: cross-attention .0436, per-target peer GRU
.0508) â‰ˆ 2Ã— CSD-only (.024) and TDA-only (.015). Both ideas were absorbed
into the unified panel (price/peer columns + `price`/`peer`/`everything`
modes in the code), but those modes were **not benchmarked** â€” the
supplement run never executed (user stopped compute). Code is ready;
rerunning `run_benchmark.py --level both --tag supplement --resume` after
`compute_features.py` covers all three modes in ~30 min.

**Remaining open items (for whoever picks this up):** the 64 not-run pooled
ticker cells + price/peer/everything supplement; `analyze_benchmark.py`
figure pass; interface refresh; README/FINAL_PROJECT_REPORT final polish
(final report already carries the v2 numbers and the run-record files).

---

## EXP-V3 work resumed (2026-10-25; experiments only)

**Scope guard:** New code/artifacts are confined to `pipeline_v3/`, `v3_*.py`,
and `data/v3/`; this log is appended. No `data/benchmark/`, V1/V2 output,
README, report, app, or figure was edited by the V3 continuation.

### Phase 0 â€” environment and event metrics

- `py -3.12`: sklearn, torch, ripser, pandas, pyarrow, lightgbm, and yfinance
  were already installed; no packages were added.
- Available resources: 20 logical CPUs, approximately 15.6 GiB RAM. Runtime
  work is limited to eight workers as requested.
- Repaired `pipeline_v3/eval_events.py`: crash catches now require an alarm
  on an actually positive label day; the random baseline samples exactly the
  observed number of distinct alarm days without replacement and reports
  expected event counts (not fractions). Per-asset random seeds are stable.
- Replaced contradictory synthetic checks with seven unit tests covering the
  five-row merge boundary, event/day counts, bridged alarm gaps, missing-date
  session gaps, deterministic random baselines, no-crash assets, and a
  rolling-threshold past-only check. Command:
  `py -3.12 -m unittest pipeline_v3.test_eval_events -v` â€” **7 passed**.
- Validation searches R1 and all three R3 budgets share one compact quantile
grid (at most 67 candidate thresholds) so event metrics are computed once per
  candidate. Missing price dates are padded as non-event sessions when
  grouping events; asset-year denominators still use observed rows.
- Added validation-only R1/R3 threshold selection and a past-only pooled R2
  threshold implementation in `pipeline_v3/operating_rules.py`.

### Phase 1 â€” data snapshot and labels

- Existing V3 Yahoo snapshot: 159 series, 26,639 source dates, 1927-12-30 to
  2026-10-24; two requested symbols (`^OMXS30`, `^XU100`) were listed as
  unavailable. All eight original local price CSVs were inventoried (50
  local asset series).
- Wikipedia constituent-list retrieval returned HTTP 403; per brief fallback,
  the existing 17 project stocks were downloaded successfully. No current
  S&P 500 survivor list was added.
- `v3_prepare_data.py` produced a 25,781-row US-business-day cleaned panel
  with 159 price targets and 15 state series retained after the 15-year rule.
  State inputs are lagged by one business day (daily FRED) or five days
  (weekly FRED); short gaps are forward-filled at most three sessions.
- Outputs under `data/v3/`: `prices.parquet`, `returns.parquet`,
  `states.parquet`, `registry.csv`, `cleaning_log.csv`,
  `large_moves_gt_40pct.csv`, `labels.parquet`, and `label_prevalence.csv`.
  Ten >40% daily moves outside crypto are listed for review and were retained.
- Seven per-asset label configurations were generated. For each origin,
  thresholds use only fully observed past forward-min outcomes and start only
  after 750 such origins. Label summary is `data/v3/label_prevalence.csv`.
- GARCH pilot: 20 refits on one long-history series took about 3.6 seconds
  including interpreter startup, projecting beyond the 25-minute fit budget
  for all non-stock assets; GARCH is omitted and this is logged.

### Phase 3 â€” feature computation status

- `v3_build_features.py --pilot --bags 3 --stride 10`: 20 representative
  market dates took 1.22 seconds (about 26.3 minutes projected for all daily
  market/clustering features); 20 representative TDA dates took 0.32 seconds
  (about 0.7 minutes projected at stride 10). Default bag count/stride were
  retained. The H2 pilot had non-empty bars on 4.2% of eligible k=12 windows
  and 28.6% of eligible k=24 windows, for both T and A clouds.
- Full feature pass is currently running; latest progress observed was
  7,000 / 25,781 market dates. No model results are available yet.

### Phase 3 â€” completed features and point-in-time audit

- Full feature build completed in 1,284.8 seconds using the requested 3 TDA bags and stride 10. It wrote 1,168,380 asset-date rows with 86 panel columns and 25,781 date-level rows with 1,475 shared feature columns. The post-pruning manifests retain F0=42, F1=61, F2=114, F3=1,027, F4=1,109 features before region metadata is refreshed. Panel and shared parquet sizes are 199 MB and 5.5 MB, respectively; no V3 file exceeds 2 GB.
- Re-ran preparation after feature extraction to refresh registry classes/regions. Cleaned prices cover 159 assets and 25,781 business dates; 15 state features remain. BASE labels have 1,044,519 valid outcomes, 20,445 positive outcomes, and 1.96% pooled prevalence. The seven label configuration prevalence values are recorded in `data/v3/label_prevalence.csv`.
- `v3_finalize_features.py` refreshed the asset class, source-dataset, and region metadata; both H2 configurations were nonempty (7.9% of eligible windows for k=12; 44.3% for k=24), so no all-empty H2 feature columns were removed.
- The first leakage-test attempt exposed a NumPy date conversion issue and an incorrect threshold reconstruction in the audit itself. Fixed the audit to call the same forward-min definition as label generation and to use only outcomes ending at least 15 asset sessions before the origin. Final `py -3.12 v3_leak_test.py` passed 10 seeded dates, 11,884 numeric comparisons (11,440 TDA), maximum absolute difference 2.9e-6 (<1e-5). This is a point-in-time consistency check, not a proof against every possible leak.
- `py -3.12 -m unittest pipeline_v3.test_eval_events -v`: all 7 tests passed.

### EXP-V3-A smoke run (not a final result)

- Command: `py -3.12 v3_run_experiments.py --feature-sets F0 --models logreg,gbm --folds 0`; completed in 133 seconds and wrote 160 rows to `data/v3/exp_a_results.csv`.
- On F0, 2000â€“2004, all-assets test subset, R1 validation-calibrated results were logistic regression AP=0.0309, event recall=0.230, event precision=0.133; LightGBM AP=0.0316, event recall=0.188, event precision=0.073. Test prevalence was 2.81%. These two exploratory numbers should not be treated as evidence of superiority; the complete walk-forward grid is required.
- EXP-V3-A full walk-forward run has started with resumable fold-level output. Results will be appended here after the grid is complete or a time-box is reached.

### EXP-V3-A and EXP-V3-D time-box status

- EXP-V3-A ran for its 60-minute box. Completed all five Scheme-W folds for F0 and all four models (logreg, GBM, GRU, TCN); F1 began at the boundary but produced no rows. F1â€“F4 comparisons therefore remain unrun. F0 fold tables and every result subset/rule are in `data/v3/exp_a_results.csv` (1,660 rows at stop). Runtime was 3,646.98 process CPU seconds; peak working set was approximately 5.6 GB. The per-fold train/validation/test counts are in the runner output and the CSV.
- For F0, R3 b=1 mean event recall by model was GBM 0.734 (range 0.245â€“1.000), logreg 0.999 (0.995â€“1.000), GRU 1.000, and TCN 1.000. Mean event precision was 0.422, 0.785, 0.725, and 0.731, respectively. These headline recalls are misleading in isolation: mean R3 test alarm-day rates were 64.2% for GBM, 99.95% for logreg, 99.97% for GRU, and 99.99% for TCN. In several folds the 200-simulation random baseline caught nearly as many crash events as the model. Broad alarms merge into a few long alarm events, so event-level wrong-alarm budgets can be met while the system alarms almost every day. Keep day-level alarm rate beside event recall/precision; do not call this a useful early-warning result.
- F0 all-assets AP averaged 0.0429 (GBM), 0.0603 (logreg), 0.0416 (GRU), 0.0431 (TCN); this compares to test prevalence that varies substantially by era. No reliable architecture winner is established: folds have high alarm-rate instability, event counts differ sharply by era, and the A feature-set grid is incomplete.
- EXP-V3-D used F0 because the time-boxed A run had no completed alternative feature set across all folds. Script assumption for the approximate-30 universe: 20 longest-history equity indices plus 10 longest-history US ETFs (30 assets, no stocks). `data/v3/exp_d_results.csv` contains all five folds for US4, approx30_index_etf, all_nonstock, and all_assets. The 1997-history universe has only the 2000â€“04 fold; later folds were stopped at D's 20-minute boundary. The 2000â€“04 history-cut model had only 8,969 training rows and selected no alarms at R3 b=1, so it is not interpretable as a history-cut conclusion.
- Across the five completed D folds, R3 b=1 results: US4 event recall 0.600 / precision 0.203 (random caught 9.49 events on average; test alarm-day rate 57.7%); approx30 event recall 0.873 / precision 0.445 (random caught 80.55; alarm-day rate 77.2%); all non-stock 0.739 / 0.368 (random caught 257.33; alarm-day rate 63.5%); all assets 0.849 / 0.465 (random caught 318.56; alarm-day rate 79.3%). These universes have different event totals, so compare against each universe's random baseline. Again, the very high alarm-day rates prevent interpreting event recall as practical lead-warning quality.
- P1 time boxes are exhausted. P2 EXP-B, EXP-F, and final-model export were not started; P3 EXP-C/EXP-E were not started. No final model is trained or frozen. The 1997-history D continuation and EXP-A F1â€“F4 grid remain resumable; use `py -3.12 v3_run_experiments.py --resume` to continue A. Do not count the current V3-A CSV as a complete architecture/feature comparison.
- Final checks: event metric tests still pass (7/7); the point-in-time audit remains 10 dates / 11,884 comparisons / 11,440 TDA comparisons / max absolute difference 2.9e-6. All V3 files remain under the 2 GB intermediate limit. The non-V3 project files and `data/benchmark/` were not edited in this continuation.

- Row-count correction after verifying the final CSVs: `exp_a_results.csv` has 1,760 rows (not 1,660); `exp_d_results.csv` has 1,160 rows, including the completed 2000â€“04 row for the history-cut universe.

## EXP-V3.1 â€” metric repair, Step 0/1 (2026-10-25)
- Step 0 inventory: BASE test folds contain (rows / positive labels / merged crash events): 2000-04 97,521/2,744/361; 2005-09 145,453/6,391/642; 2010-14 181,175/1,461/220; 2015-19 191,813/1,199/212; 2020-latest 275,917/5,195/606. Frozen feature manifest currently has F0=51, F1=70, F2=123, F3=1,036, F4=1,118 columns.
- Saved-score audit found no per-row old EXP-A/EXP-D predictions in CSV/parquet/NPZ/PKL. The metric-only rescore path is unavailable; both old aggregate files were copied to `data/v3/exp_a_results_old.csv` and `data/v3/exp_d_results_old.csv`; fresh F0 fitting is required.
- A-vs-D code note: A fits the full panel at seed 1701+fold; D fits registry-filtered universes at seed 2718+fold. Both use F0, the same LightGBM helper and chronological split; stochastic seeds and the universe construction differ. The old CSVs lack row scores, so the precise source of the reported alarm-rate/recall gap cannot be isolated retrospectively.
- Implemented `v3_1_eval.py`: alarm-day budgets RQ/RV, event bridging and 15-session alarm-run cap, daily/event metrics, random event-recall baseline and off-budget guard. Four required unit tests pass: oracle, random-vs-baseline, always-alarm off-budget, hand-counted toy. Command: `py -3.12 -m unittest v3_1_test_eval -v`.
- TDA block inventory from frozen F4: T_H0=23, T_H1=34, T_H2=9, A_H0=4, A_H1=0, A_H2=0; T dynamics split into bounded groups of 60/60/8. These sets reflect only columns available in the existing manifest; absent A-cloud blocks are not recomputed.

## EXP-V4 progress and crash-count clarification (2026-10-25)
- User requested an audit/rebuild of TDA features (V4), preservation of prior work, and a maintained record of requests and completion status. V4 work is isolated in `v4_*.py` and `data/v4/`; no V3 source data or old result files were overwritten.
- Completed so far: `v4_1_audit.py` wrote `data/v4/audit_old_tda.csv` and `audit_market_crash_runs.csv`; `v4_3_build_raw.py` wrote `blocks/V4_RAW.parquet`; `v4_4_build_dyn.py` wrote `blocks/V4_DYN.parquet`. Current V4_RAW covers 1,399 dates (2000-01-03 to 2026-10-19), 161 feature columns, and the A_corr/A_raw/T_raw blocks over windows 20/60/120. The build used stride=5 and 2 bags (feature_blocks.json), so it is a reduced run and does not match the requested 3-bag configuration. V4 feature diagnostics, any required full-config rerun, leak/scale checks, and model comparisons remain unfinished.
- Corrected label interpretation: BASE is an asset-specific forward 15-session minimum-return event, thresholded at that asset's expanding historical 2nd percentile after 750 prior outcomes. It is not a manually curated catalogue of named market crashes.
- Across all available BASE labels (159 assets; data date range 1927-12-30 through 2026-10-23), there are 20,445 positive asset-days and 2,484 asset-specific event runs when positive runs separated by at most five sessions are merged. The 2000-latest out-of-sample test folds contain 16,990 positive asset-days and 2,041 such events: fold event counts 361, 642, 220, 212, 606. These are pooled per-asset events, so simultaneous asset events can refer to the same broad market episode.
- The V4 market-wide audit uses a separate proxy: a date is flagged when at least 20% of assets have positive BASE labels. This yields 26 market-wide proxy runs across the five test folds (9, 10, 1, 1, 5). This proxy is not a count of all historically named crashes and is not interchangeable with the 2,041 asset-event count.
- Run state at this update: V3.1 Step 2 completed all five folds for B1/B2/B3 logistic regression and F0 logistic regression/GBM. Step 3 classical ladder is still running (process `v3_1_run_steps.py 3`); its CSV currently has through fold 2015-2019, and validation has four folds. V4 dynamic block has finished; no V4 modeling results exist yet.

## EXP-V5 progress: inventory, evaluator, E1 smoke (2026-10-25)
- V5 supersedes prior restrictions where stated and keeps all work isolated under `v5_*.py`, `pipeline_v5/`, and `data/v5/`. Existing data and prior result files remain untouched.
- Phase 0 inventory completed: panel has 1,168,380 rows and 159 assets, 1927-12-30 through 2026-10-23. Per-year asset counts, valid BASE prevalence, positive days, and event counts are in `data/v5/inventory_by_year.csv`; per-era summaries are in `inventory_by_era.csv`. The labels reuse V3's per-asset expanding historical 2nd-percentile threshold after 750 prior outcomes. V3 feature counts are F0=51, F1=70, F2=123, F3=1,036, F4=1,118; score inventory found 80 parquet files. V4 currently has RAW and DYN parquets.
- Resource snapshot: 15.6 GiB RAM total, about 3.3 GiB available at inventory time, 14 physical cores / 20 logical CPUs. All V5 model jobs are set to one worker; requested cap is 8.
- `v5_eval.py` implements the 11 fixed eras, Scheme B blocked splits with embargo, Scheme W expanding splits, RE/RQ/RV thresholds, random-baseline event metrics, and off-budget checks. `py -3.12 -m unittest v5_test_eval -v`: 6/6 passed (oracle, random scores, always-alarm budget, hand-counted toy, embargo separation, exact era coverage).
- E1 smoke: one Scheme B/F2/GBM/E0 fit completed in 0.4 min and wrote fold scores/results. Full E1 grid is 216 model-era tasks (6 sets x 2 models x [11 B eras + 7 W folds]).
- First E1 full pass reached 71/216 tasks in 62.7 elapsed minutes, then hit the timebox. Audit caught validation-to-train embargo used N+40 rather than N+25 (test-era embargo was correctly N+40); that pass is preserved under `data/v5/archive_e1_val_purge55/`.
- A second short pass reached 32 tasks before audit found a_rel was based on test prevalence. It is preserved under `data/v5/archive_e1_test_prevalence/`.
- A third short pass (17 tasks) had correct embargoes and validation-only a_rel, but lacked the >20% off-budget early stop; it is preserved under `data/v5/archive_e1_without_stoprule/`.
- The current runner adds the >20% early-stop guard and resume-aware off-budget counts. Only its outputs may feed E1 summaries; a fourth audited pass will use the full 60-minute box.
- Status: stride-1 CLUSTER completed 25,780 dates; PRICE1D completed all 159 assets. V5 big-cloud remains active; deep models and label sweep remain pending.
- Expanded V5 checks: all 15 unit tests across evaluation, topology scale laws, PRICE1D (including 200 brute-force comparisons), and cluster/network metrics pass. `requirements.txt` now lists the explicitly allowed `numba` dependency used by PRICE1D.
- The prior V3.1 Step 3 runner was stopped when V5 took priority, with its incremental outputs preserved. At stop: 150 result rows across F0/F1 (both models, 5 folds each) and partial F2 (logreg 3 folds, GBM 2); 26 validation rows. No V3.1 Steps 4-8 were started. The post-queue score pack/log job is allowed to finish.
- Before the V5 topology timing run: one-window smoke produced 1,324 features in 5.7 seconds. The 20-window pilot measured 0.8 min and projected stride-5/B=3 full-history cost at 209.8 min. Following the required reduction order, B=2 projects to 139.9 min; stride=10 then projects to 69.9 min (already using specified landmarks for >100-point clouds). Full run is launched at stride 10/B=2 with a 100-minute hard box; no k or cloud type was dropped.
- Cluster block three-date smoke took 0.23s; extrapolated 25,781-date serial runtime is ~33 minutes, launched with a 100-minute checkpoint limit, stride 1, and no ripser. PRICE1D uses the allowed Numba package; union-find matches both required bar examples and an independent brute-force implementation on 200 random windows.
- PRICE1D 250-row smoke on the longest index series took 0.68s. Scaling against 1.17M labeled asset-date rows gives a rough 53-minute estimate; launching with a 100-minute checkpoint limit, one worker, and per-asset checkpoints.
### EXP-V4-AUDIT â€” V3 TDA audit and metric tests

- Reused `data/v3/` prices, returns, labels, folds, universe, and frozen feature outputs; no V3 files were modified.
- Audited 1,420 old TDA columns: V3 row-wise z-scored clouds, stride 10, and forward-filled snapshots; market crash runs were 9/10/1/1/5 across the five folds.
- The four required V4 metric tests passed: oracle, random, always-alarm off-budget, and the hand-counted 30-day event toy.
- Outputs: `data/v4/audit_old_tda.csv`, `data/v4/audit_market_crash_runs.csv`.

### EXP-V4-BUILD â€” Rebuilt P1 topology blocks and checks

- Built `V4_RAW` (A_corr, A_raw, T_raw; raw and scale-free summaries) and `V4_DYN` using past-only windows and date-seeded bagging. Adaptive runtime used stride 5 and two bags; raw block has 161 columns and dynamic block 122 columns over 1,399 post-2000 dates.
- PRICE1D union-find unit tests passed, including both hand cases and 200 brute-force random comparisons.
- Point-in-time leak test passed: 10 dates, 670 comparisons, maximum absolute difference 0. Scale-law test passed with maximum error about 3.55e-15.
- Outputs: `data/v4/blocks/V4_RAW.parquet`, `data/v4/blocks/V4_DYN.parquet`, `data/v4/leak_test.csv`, `data/v4/scale_test.csv`.

### EXP-V4-D1 â€” Fair paired model comparison

- Compared logistic regression and LightGBM using the requested V3 Scheme-W fold dates and the V4 alarm-day/event evaluator; every completed run wrote scores under `data/v4/scores/`.
- Because the current P1 blocks begin in 2000, fold 0 has no pre-2000 training rows after the inner block join and was skipped; folds 1-4 have complete paired results. This is a limitation, not a five-fold claim.
- At 5% nominal budget, the available-fold mean excess-event differences versus M0 were negative for OLD_TDA, V4_RAW, and V4_DYN in both models; ALL_V4 was negative for logistic regression and near zero for GBM. Most runs were off-budget and therefore do not enter a strict headline claim.
- Outputs: `data/v4/d1_paired.csv`, `data/v4/d1_paired_summary.csv`, and `data/v4/scores/`.

## EXP-V3.1-Step-2 (2026-10-25)
- Recorded rows: 150; completed folds/cells: 25; on-budget headline rows: 24/25.
- On-budget RQ@5% means by feature set/model: B1/logreg excess_event_recall=-0.1222, event_recall=0.2596, event_precision=0.0775, day_lift=2.6161, rand_event_recall=0.3818, realized_alarm_day_rate=0.07; B2/logreg excess_event_recall=-0.1692, event_recall=0.2845, event_precision=0.0802, day_lift=2.5348, rand_event_recall=0.4537, realized_alarm_day_rate=0.0885; B3/logreg excess_event_recall=-0.1378, event_recall=0.2841, event_precision=0.0923, day_lift=2.9893, rand_event_recall=0.4219, realized_alarm_day_rate=0.0783; F0/gbm excess_event_recall=-0.0676, event_recall=0.1746, event_precision=0.0604, day_lift=2.3848, rand_event_recall=0.2422, realized_alarm_day_rate=0.0416; F0/logreg excess_event_recall=-0.0901, event_recall=0.2848, event_precision=0.0854, day_lift=2.7873, rand_event_recall=0.3749, realized_alarm_day_rate=0.0736.

## EXP-V3.1-Step-3 (2026-10-25)
- Recorded rows: 150; completed folds/cells: 25; on-budget headline rows: 23/25.
- On-budget RQ@5% means by feature set/model: F0/gbm excess_event_recall=-0.0766, event_recall=0.1764, event_precision=0.0645, day_lift=2.4877, rand_event_recall=0.253, realized_alarm_day_rate=0.0449; F0/logreg excess_event_recall=-0.0901, event_recall=0.2848, event_precision=0.0854, day_lift=2.7873, rand_event_recall=0.3749, realized_alarm_day_rate=0.0736; F1/gbm excess_event_recall=-0.0843, event_recall=0.207, event_precision=0.0581, day_lift=2.0415, rand_event_recall=0.2913, realized_alarm_day_rate=0.0527; F1/logreg excess_event_recall=-0.0729, event_recall=0.3216, event_precision=0.0654, day_lift=2.6873, rand_event_recall=0.3945, realized_alarm_day_rate=0.0828; F2/gbm excess_event_recall=-0.0702, event_recall=0.1673, event_precision=0.1736, day_lift=3.473, rand_event_recall=0.2375, realized_alarm_day_rate=0.034; F2/logreg excess_event_recall=-0.3058, event_recall=0.1587, event_precision=0.0464, day_lift=0.8152, rand_event_recall=0.4645, realized_alarm_day_rate=0.0909.
### EXP-V6-0/1 â€” V6 inventory, SEMIVAR, and severity-label sweep

- V6 was started in isolated `pipeline_v6/`, `v6_*.py`, and `data/v6/` paths while existing V5 processes continued untouched.
- Inventory confirmed 159 assets, 1,168,380 V3 panel rows, and 20,445 BASE positive asset-days over 1927-2026.
- Copied the V5 evaluator to `pipeline_v6/v5_eval_copy.py`; evaluator smoke tests passed with blocked era separation.
- Built the past-only SEMIVAR block for all 159 assets: 4,099,179 asset-date rows and 18 feature columns; saved `data/v6/semivar.parquet` and `data/v6/panel.parquet`.
- Implemented checkpointed severity ranks for down, up, and two-sided labels over N={3,5,10,15,30} and p={2,3,4,5,6,8,10}%; per-asset outputs are under `data/v6/labels/`.
- Label prevalence and overlap outputs are `data/v6/label_prevalence.csv`, `labels_overlap.csv`, and `labels_overlap_by_class.csv`; the N=15,p=2 down-label mismatch against V3 BASE is recorded rather than silently substituted.
- Started the resumable V6 E-severity blocked-era sweep in `v6_5_sweep.py`; it writes `data/v6/sweep_e_sev.csv` after each completed era.

### EXP-V6-3 â€” E-severity blocked-era sweep

- Completed 110 cells: both directions Ã— N={3,5,10,15,30} Ã— 11 eras, using GBM-lite regression on severity and the fixed 5% alarm-day budget.
- No completed cell was off-budget. Mean excess event recall was highest at N=5: 0.645 down and 0.668 up; N=3 was 0.476 down and 0.419 up.
- These are blocked-CV results and are not a real-time Scheme-W claim. The strongest cells should still be checked against B3, E-bin, and the capped event definitions before interpreting them as predictive value.
- Outputs: `data/v6/sweep_e_sev.csv` and `data/v6/sweep_e_sev_summary.csv`.

### EXP-V6-4 - Scheme-W real-time validation and completion status

- Completed 168 causal Scheme-W cells: N={5,15,30}, down/up, B3 and SEMIVAR feature sets, E-severity and E-bin engines, across E4-E10.
- At N=5 with SEMIVAR, mean excess event recall was 0.317 down and 0.230 up for E-severity; B3 was 0.358 down and 0.209 up. At N=15, SEMIVAR was 0.236 down and 0.204 up. At N=30, SEMIVAR was 0.110 down and 0.138 up.
- The SEMIVAR block improved AP and AUC over B3 in the real-time results, but excess-event recall did not improve uniformly; results are regime-dependent.
- Native down/up comparison is saved in `data/v6/t1_native_direction.csv` and `t1_down_up_comparison.csv`. Remaining work is explicitly recorded in `data/v6/v6_remaining_status.csv`; cross-transfer and multi-strategy tests were not claimed because per-row component score histories were not saved.
- N=5 cross-direction transfer is now complete. Saved per-row Scheme-W scores in `data/v6/scheme_w_n5_scores.parquet` and transfer AUCs in `data/v6/t2_cross_transfer_n5.csv`. Native-direction scores do not transfer: pooled AUC was 0.188 for down-score to up labels and 0.172 for up-score to down labels.

### EXP-V5 continuation (2026-10-25)

- E1 corrected pass checkpoint: 113/216 planned cells, 320 aggregate metric rows. The run has correctly stopped B1/B2/B3/F0/F1 GBM experiments after more than 20% of inspected Scheme-B primary folds fell outside the preregistered alarm-day band. Strict within-era `score > q95` ties are the main observed cause: some E0 baseline GBMs had zero alarm days; inspected baseline assets had about 9-17% of scores tied at q95. The rule was not altered. A fold diagnostic utility is `v5_1_diagnose_offbudget.py`.
- The first full-history TDA pass finished 2,578 dates at stride 10 and two bags in 73.3 minutes. Its scale audit found the specified L2 / variance^1.5 ratio is not scale-free given L2 persistence units; corrected to L2 / variance^0.75 (standard deviation^1.5). Raw distances are normalized before ripser and diagrams restored to physical units, keeping greedy landmark selection scale-invariant. Old feature/quality outputs are retained in `data/v5/archive_tda_scale_v1/`; corrected rebuild is active.
- Enforced TDA missingness for any universe/window with fewer than eight complete-window assets. Quality gate will be rerun on regenerated feature values; the earlier mismatch was from applying the corrected formula to old saved scores.
- 20-test suite passed before the scale correction; 8 topology/dynamics/sequence tests passed after it. E2 runner/paired event bootstrap, E3 BEST_ALL selection, and E4 minimal label sweep are implemented and compile; not yet launched.
- E1 headline and F1/F2-vs-B3 summarizer now compares only common on-budget Scheme-B eras. Full E1, corrected topology, dynamics, quality gate, E2-E4 are pending.
- Corrected topology preflight scale check on 2024-03-04: 2,044 feature values (all generated raw/normalized summaries and landscapes) had maximum relative scale-law error 3.85e-13. The registered 20-date scale gate still runs after rebuild completion.

### EXP-V5 continuation (live checkpoint, 2026-10-25)

The audited E1 job is at 135/216 planned cells (elapsed 39.1/60 minutes) with 380 aggregate rows and 93 validation rows at the last count; previously observed GBM STOPs follow the preregistered off-budget guard. The corrected big-cloud TDA rebuild is at 2,320/2,578 dates. The previous `quality_gate_results.csv` belongs to the archived, scale-incorrect build and is not a valid result for the active rebuild.

A hidden continuation queue (`v5_13_continue_queue.ps1`, PID 30752) waits for both active jobs, then runs the E1 off-budget diagnosis/summary, eight-asset TDA enforcement, and fresh quality gates. It halts downstream work if any required gate fails. If they pass, it runs dynamics, P1 diagnostics, E2 smoke/full, E3 smoke/full, minimal E4 smoke/full, and the historical episode table with separate step logs under `data/v5/`. This is queued work, not a claim of completion; V6 is untouched.

Queue safety adjustment: the initial idle queue (PID 30752) was stopped before executing any stage to preserve the existing pre-rebuild quality-gate CSVs. The restarted queue (PID 21744) first archives those files to `data/v5/archive_quality_pre_corrected_rebuild/`, then writes fresh checks. At restart, E1 was at 138/216 cells (42.9 minutes) and corrected TDA at 2,480/2,578 sampled dates.

Corrected TDA builder has now completed within its 100-minute hard box with 2,578/2,578 sampled dates saved (stride 10, two bags) and daily block files present. It emitted NumPy invalid-divide warnings on zero-variance correlation windows; treat missingness and the post-rebuild quality gates as required checks. E1 remains active at 141/216 cells, 45.7 elapsed minutes at the last checkpoint. The continuation queue is waiting for E1 before it archives and recomputes quality gates.

Queue status update: idle queue instances 30752 and 21744 were stopped before launching experiment stages while I added archiving and reordered independent work. Active queue PID 28928 has moved all five prior quality-gate CSVs to `data/v5/archive_quality_pre_corrected_rebuild/` and is now running universe enforcement in parallel with E1. No experiment stage was lost or replaced by the idle queues.

Quality gates on the corrected, eight-asset-masked topology passed: leak check 10/10 dates, maximum absolute difference 0.0; scale check 39,876 values, maximum relative error 6.65e-12. The block has 2,044 columns and 98 H1/H2 groups. Mean sampled NaN share is 60.8%. Bar presence is 29.4% of all sampled rows on average (H1 30.4%, H2 23.2%); conditional on non-NaN rows it is 81.2% (H1 83.9%, H2 65.3%).

The first DYN build hit an empty PCA input in the pre-window boundary. Added an `index < 59` guard, validated the earliest date directly, and reran. The rerun completed landscapes for 2,578/2,578 sampled dates and saved `V5_DYN` (25,781 dates x 962 features, `landscapes_complete=True`). The 3 dynamics unit tests passed. The first failed attempt remains logged.

E1 corrected run reached its hard timebox and checkpointed at 160/216 planned cells in 60.9 minutes, with 572 metric rows and 112 per-row score files. The last task began before the deadline and committed before the next deadline check. W/B2 logistic regression stopped after 2/7 primary folds violated the alarm budget. The E1 diagnostic and validation-only BASE_SET/headline summaries remain queued after the P1 diagnostics stage; no E1 ranking result is claimed yet.

Topology universe audit: 142 non-stock assets plus 17 stocks (159 total). ALL uses all available assets per date; ALLS cap=250 exceeded the universe, excluding no stocks. Corrected manifest: 2,044 features, stride 10, two date-seeded bags, minimum eight available assets, sampled range 1928-01-02–2026-10-12; daily forward-filled block covers through 2026-10-23.

P1 descriptive diagnostics completed in 26.6/30 minutes: 268 C1 scaling rows, 2,044 C2 proxy-R2 rows, and 90,068 C3 era/volatility-tercile signed-AUC rows. Outputs are `diagnostic_c1_raw_scaling.csv`, `diagnostic_c2_proxy_r2.csv`, `diagnostic_c2_block_share.csv`, `diagnostic_c3_auc.csv`, and `diagnostic_c3_top15.csv` under `data/v5/`. The queue has advanced to E1 off-budget diagnosis and will then write BASE_SET/headlines before E2.

Diagnostic preview (descriptive only): proxy redundancy flags >R² 0.8 for 1/2,044 topology features (A_corr 1/724; other families 0). Median C1 slopes for raw max persistence are close to volatility-only expectation (A_raw 0.47, T_raw 0.48 vs 0.5); landscape scaling varies. C3 feature AUCs vary by era, and extreme effects can be confined to one era or inverse-signed; this is not evidence of stable forecast skill.

E1 summary (2026-10-25): fixed the validation-schema mismatch and selected BASE_SET=F0 from 14 common Scheme-B validation cells. Validation means: F0 0.1335, F1 0.0978, F2 -0.0199 excess event recall. Pooled on-budget Scheme-B rows all have negative excess event recall; F0/logreg event recall 0.251 vs random 0.303 (excess -0.0519). Scheme-W rows are also negative. F1/logreg beats B3 in 9/11 eras (82%); F2/logreg 4/11 (36%); no common on-budget GBM eras, so no GBM verdict. E1 itself ended at 160/216 cells, 572 metric rows, 112 score files, after 60.9 minutes. Off-budget diagnostic: 21 primary folds, median q95 tie share 9.7%, 11 asset skips.

E2 smoke passed after E1 summary: B/M0/logreg/E0, 240,490 score rows, 3,342 positives, 6 crash events. Its first queued smoke had failed only because BASE_SET did not yet exist; the later E3 smoke was terminated before it produced output. A resumable E2-E4 queue (PID 2288) is active and starts with the 90-minute E2 run.

E2 full-grid checkpoint at 15:03: nine of 2,448 planned jobs have been reached in under one minute; initial Scheme-B M0/logistic folds are writing score parquets and report per-era event totals (E1=59, E2=171, E4=325, E5=677, E6=220, E7=123, E8=91). The 90-minute run continues in the background. The continuation queue then summarizes E2 and attempts E3 smoke/full, E4 minimal smoke/full, and the episode table. P2 E5-E7 are still pending.

E2 live checkpoint at 15:04: 11/2,448 jobs reached through Scheme-B M0/logistic E10; E10 contains 391 BASE event runs. Full run remains active under the 90-minute box.

E2 live checkpoint (2026-10-25 15:27 local): 51/2,448 jobs completed at 24.1 elapsed minutes; process PID 24140 remains active under the 90-minute hard box (queue PID 2288). The last committed job is W/B3/logreg/E7. E2 per-job scores are checkpointed. Queue order remains E2 summary, E3 smoke/full (150-minute box), E4 smoke/minimal full (90-minute box), then episode table; E5-E7 P2 are not included. Do not treat partial E2 results as final.

E2 live checkpoint (2026-10-25 15:29 local): 53/2,448 jobs, elapsed 26.0 minutes; PIDs 24140/2288 active. This is a timeboxed sweep, so report actual completed combinations separately from planned grid after summarization.

E2 live checkpoint (2026-10-25 15:30 local): 56/2,448 jobs, elapsed 27.6 minutes; entered B/B3/GBM (E0 and E1 completed), active PIDs unchanged. LightGBM emits an eval_set deprecation warning; no run failure observed. Per-job results continue to checkpoint.

E2 live checkpoint (2026-10-25 15:31 local): log advanced to 66/2,448 jobs at 28.4 minutes and is in W/B3/GBM. Queue PID 2288 and Python 24140 remain active. Results are partial until timebox completion and E2 summary.

V6 audit update (2026-10-25 15:49 local): found that the old V6 severity labels were shifted forward by N rows and their expanding reference ranks included outcomes not yet known at forecast origin. Prior V6 labels, dependent model outputs, and report were preserved under data/v6/archive_pre_forward_label_fix_20261025; old performance results are invalid as early-warning evidence. Corrected origin-aligned, past-only, full-window labels are rebuilding (71/159 asset files at the latest check); label and evaluator audit tests pass. Corrected V6 model reruns are pending. V5 E2 remains active at 74/2448 jobs, 43.6 minutes into its 90-minute timebox; queue PID 2288 remains active.
