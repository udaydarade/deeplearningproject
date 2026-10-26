# Project request and completion log

This is a running status record of the user's project requests and the work completed. Detailed experiment configurations and numerical results remain in `GLM_EXPERIMENT_LOG.md`; this file tracks scope and state so later updates can be appended without losing the request history.

## Request history

| Request | Work completed | Current state |
|---|---|---|
| Build a complete crash early-warning project comparing topological indicators with classical indicators, with objective labels, models, evaluation, interface, and report. | Existing project includes V3 data preparation, labels, features, walk-forward experiments and prior project reports. Earlier V1/V2 and V3 work is preserved in its existing files. | Core historical experiments are documented in the existing reports/logs; model comparisons have known time-boxed gaps. |
| Add cross-index/deep-learning prediction using information from other indices, without removing other experiments. | Cross-index/peer-index experiment scripts and results exist in the project history. | Preserved; no files from these experiments were removed as part of current V3.1/V4 work. |
| Record prior work and results in project documents. | `EXPERIMENT_RESULTS.md`, `ALL_EXPERIMENT_RESULTS.md`, `GLM_EXPERIMENT_LOG.md`, and other existing report artifacts are present. | This task log supplements those artifacts with request/status tracking. |
| Finish V3 pipeline work and document its results. | Existing V3 datasets, feature manifests, experiment outputs, and V3 run history are retained. | V3 historical results are recorded in `GLM_EXPERIMENT_LOG.md`; see its V3 sections for time-box limits. |
| V3.1: audit saved EXP-A/EXP-D row scores; rescore if available, otherwise retrain; repair alarm-budget metrics; run experiments without blocking subsequent coding. | No saved per-row EXP-A/EXP-D predictions were found, so fresh F0 training was required. Old aggregate result CSVs were copied to `*_old.csv`. Metric evaluator and four unit tests were added; Step 2 and Step 3 pipelines were started. | Step 2 is complete. Step 3 was stopped when V5 took priority; F0/F1 are complete across five folds for both models, F2 is partial, and Steps 4-8 were not started. Incremental outputs remain saved. |
| V4: audit/rebuild TDA blocks fairly while preserving V3 and other teams' experiments; append results to the experiment log. | V4 old-feature audit, market-wide proxy counts, V4_RAW, and V4_DYN artifacts are complete under `data/v4/`. V4 source files use the `v4_*.py` naming convention. | V4 feature diagnostics, remaining P1/P2 blocks, tests, and fair model comparisons remain to be run. No V4 model results are available yet. |
| Clarify whether the current labels include every crash and give counts. | Verified the exact BASE label definition and counted positive days/events across the full asset panel, walk-forward test folds, each of the four U.S. indices, and the V4 market-wide proxy. | Counts and important interpretation caveat are recorded in `GLM_EXPERIMENT_LOG.md` under EXP-V4 progress. |
| V5: full-history blocked evaluation, larger TDA clouds, cluster/price topology, sequence/cross-asset models, and label sweeps. | V5 Phase 0 annual/era inventory is saved; the new evaluator and six required unit tests pass; a Scheme B F2/GBM/E0 smoke run completed. | E1 full grid is authorized and time-boxed at 60 minutes; big-cloud and remaining P1/P2 experiments still pending. V5 instructions and initial status are recorded in `GLM_EXPERIMENT_LOG.md`. |

## Current verified counts and run state (updated 2026-10-25)

- BASE is an asset-specific 15-session forward minimum-return label, with a point-in-time expanding 2nd-percentile threshold after 750 prior outcomes. It is not a curated list of named historical crashes.
- Full available sample: 159 assets, 1927-12-30 to 2026-10-23; 1,044,519 valid asset-days, 20,445 positive asset-days, and 2,484 merged asset-level event runs (runs within an asset separated by at most five sessions are joined).
- Five out-of-sample folds, 2000-latest: 16,990 positive asset-days and 2,041 merged asset-level events. Per-fold events: 361, 642, 220, 212, 606.
- V4's separate market-wide proxy (at least 20% of assets labeled positive on a day): 26 merged episodes across those folds, distributed 9, 10, 1, 1, 5. This is not the same unit as asset-level events and is not a complete catalogue of historical market crashes.
- V3.1 Step 2 complete: five folds for B1/B2/B3 logistic regression and F0 logistic regression/GBM. Step 3 was stopped for V5; partial results and validation rows are preserved.
- V4: old TDA audit, V4_RAW, and V4_DYN are complete. V4_RAW currently has 1,399 dates and 161 features, built at stride 5 with 2 bags; the requested 3-bag configuration is not yet met. V4 model comparisons and remaining diagnostics are pending.
- V5: Phase 0 and all 18 current unit tests are complete. Three E1 drafts are preserved separately after audits of validation embargo, test-prevalence use, and the off-budget stop rule. The next E1 pass includes all three checks. Big-cloud TDA is running; CLUSTER (25,780 dates) and PRICE1D (159 assets) are complete. See `GLM_EXPERIMENT_LOG.md` for checkpoints.

## Update protocol

When a requested experiment starts, completes, fails, or changes scope, add a dated row/update here and the matching methodological/result details to `GLM_EXPERIMENT_LOG.md`. Keep “complete,” “running,” and “pending” distinct; do not report a run as complete solely because an output file exists.

## EXP-V5 continuation update (2026-10-25)

- Corrected V5 E1 pass is still running under its 60-minute box. At the last checkpoint it had completed 113/216 model-era cells (320 result rows; several GBM set/model runs correctly stopped after >20% off-budget folds). The independent logs are `data/v5/e1.stdout.log` and `data/v5/e1_results.csv`.
- Diagnosed early GBM off-budget results: strict `score > within-era q95` creates too few alarm days in low-cardinality score distributions; the mean fraction tied at q95 was 9-17% in inspected baseline sets. The preregistered rule remains unchanged; `v5_1_diagnose_offbudget.py` creates the fold diagnostic CSV.
- First stride-10, two-bag topology run completed 2,578 dates in 73.3 minutes and was preserved under `data/v5/archive_tda_scale_v1/` after the scale audit exposed a dimensionally inconsistent L2 ratio and raw-landmark scale sensitivity. The corrected scale-normalized rebuild is active; it retains all k/clouds and enforces the eight-asset minimum.
- Updated topology checks pass (8 topology/dynamics/sequence tests); the prior complete suite had 20/20 passing. Final leak/scale checks are pending the corrected topology rebuild.
- Implemented E2 block comparisons and paired event-bootstrap summary (`v5_9_e2.py`, `v5_9_summarize_e2.py`), expanded E3 to validation-selected BEST_ALL (`v5_8_e3_deep.py`), and added the minimal frozen-label sweep (`v5_10_e4_labels.py`). They compile; execution awaits E1/TDA and the dynamics block.
- Next phases: finish E1/TDA; enforce minimum-universe masks and run 10-date leak/20-date scale gates; backfill landscape/dynamics; then run E2 smoke/full timebox, E3, and the E4 minimal label sweep as time allows. No V3/V4 results have been overwritten.
- Corrected topology preflight scale check on 2024-03-04: 2,044 feature values had maximum relative scale-law error 3.85e-13. The registered 20-date scale gate remains pending the full rebuild.

## EXP-V5 continuation update (2026-10-25, live checkpoint)

- E1 corrected run is still active under the 60-minute hard box: 135/216 cells written at elapsed 39.1 minutes (380 aggregate rows and 93 validation rows at the last file count). This excludes the 3 GBM set/scheme runs already stopped by the registered >20% off-budget rule. Remaining cells continue until completion or the hard box.
- Corrected stride-10/two-bag TDA rebuild is active at 2,320/2,578 sampled dates. It is generating the full requested cloud-size grid; no cloud-size family has been removed. The old quality-gate CSV is from the archived build and is not being treated as the corrected build's result.
- Started `v5_13_continue_queue.ps1` (hidden PowerShell process PID 30752). It waits for E1 and TDA, then runs E1 diagnostics/summary, enforces the eight-asset TDA rule, reruns leak/scale/quality gates, and stops before downstream comparisons if those gates fail. If gates pass it continues with dynamics, diagnostics, E2, E3, minimal E4, and episode table; each step has its own log in `data/v5/`.
- Queue script syntax was checked. No downstream result is claimed yet. V6 process and files were left untouched.

- Queue safety adjustment (2026-10-25): stopped the first idle queue before it launched any stage, so no results were affected. It would have replaced the existing pre-rebuild quality CSVs. The restarted queue (PID 21744) now archives those files to `data/v5/archive_quality_pre_corrected_rebuild/` before writing fresh gates. Live checkpoint when restarted: E1 138/216 cells at 42.9 minutes; corrected TDA 2,480/2,578 sampled dates.

- Corrected TDA builder completed (2026-10-25): saved all 2,578/2,578 stride-10 sampled dates and the daily forward-filled block under `data/v5/blocks/`. It emitted NumPy warnings for zero-variance correlation windows; the quality audit is queued to quantify their missingness and run the required gates. E1 continued separately and reached 141/216 cells at 45.7 minutes when checked.

- Queue scheduling update (2026-10-25): two idle queue instances were stopped before launching experiment stages while I tightened the no-overwrite handling and parallel order. The active continuation queue is PID 28928. It has preserved the five old quality-gate CSVs under `data/v5/archive_quality_pre_corrected_rebuild/` and is running the eight-asset enforcement now, while E1 continues. No experiment result was lost or overwritten by the idle queues.

- Quality gates passed on the corrected features: leakage check 10 dates, max difference 0.0; scale check 39,876 comparisons, max relative error 6.65e-12. The topology block has 2,044 columns and 98 H1/H2 groups. Mean sampled NaN share is 60.8% (mostly no eight-asset history early in the panel). Bar presence is 29.4% of all sampled rows on average across groups (H1 30.4%, H2 23.2%); among non-NaN rows it averages 81.2% (H1 83.9%, H2 65.3%).
- Dynamics initially failed on an early date because a negative slice produced an empty PCA input. Added an explicit index<59 guard; the early-window smoke and all 3 dynamics unit tests passed. Rerun produced landscapes for 2,578/2,578 dates and `V5_DYN` with 25,781 rows/962 columns; `landscapes_complete=True`. The failed first attempt is retained in its log.

- E1 timebox ended and checkpointed at 160/216 task cells, 572 metric rows, 112 saved score parquets, elapsed 60.9 minutes (last fold began just before cutoff). W/B2 logistic stopped after 2/7 primary folds off-budget; other STOP lines are in `data/v5/e1.stdout.log`. E1 off-budget diagnosis and validation-only summary are queued after the P1 diagnostics run; no BASE_SET or E1 headline is claimed until those outputs are written.

Topology universe audit: registry has 142 non-stock assets and 17 stocks (159 total). The ALL cloud uses every available asset at each date; the ALLS stock-inclusive cap of 250 exceeds the full universe size, so no actual stock was excluded. Feature manifest records 2,044 topology columns, stride 10, two date-seeded bags, minimum available universe 8, dates 1928-01-02 through 2026-10-12 sampled; daily ffilled coverage extends through 2026-10-23.

P1 feature diagnostics completed in 26.6 minutes (within 30): C1 268 raw-scale rows, C2 2,044 proxy-R2 rows, C3 90,068 signed-AUC/volatility-tercile rows. Outputs: `data/v5/diagnostic_c1_raw_scaling.csv`, `diagnostic_c2_proxy_r2.csv`, `diagnostic_c2_block_share.csv`, `diagnostic_c3_auc.csv`, `diagnostic_c3_top15.csv`. The queue is now running the E1 off-budget diagnosis, followed by its validation-selected summary.

Diagnostic preview (descriptive, not model evidence): C2 flags 1/2,044 topology features at proxy R�>0.8 (A_corr 1/724; other cloud families 0). In C1, median log-log slopes are close to volatility-only expectations for raw max persistence (A_raw 0.47 vs 0.5; T_raw 0.48 vs 0.5), while landscape L1/L2 slopes vary by cloud. C3 finite signed-AUC rows vary substantially by era; some extreme feature values are single-era and often inversely signed, so do not interpret them as stable predictive signals.

E1 summary completed after correcting a validation-column name (`excess_event_recall`, not `validation_excess_event_recall`). BASE_SET=F0 from 14 common Scheme-B validation cells: means F0=0.1335, F1=0.0978, F2=-0.0199. On-budget pooled Scheme-B excess event recall is negative for every summarized row; F0/logreg catches 25.1% of events vs 30.3% random (excess -0.0519). F1/logreg beats B3 in 9/11 eras; F2/logreg in 4/11; there are no common on-budget GBM eras for those comparisons. Walk-forward headline excess is also negative. Off-budget diagnostic covered 21 folds, median q95 tie share 9.7%, 11 skipped assets. E2 smoke now passes: B/M0/logreg/E0, 240,490 rows, 3,342 positives, 6 crash events. E2 90-minute continuation is active (PID 2288); the earlier failed smoke/E3 smoke attempts stopped before E3 wrote outputs.

E2 run checkpoint (2026-10-25 15:03): full run is active under its 90-minute box; 9/2,448 planned jobs reached with the first scheme-B M0/logreg folds saved, including event counts E1=59, E2=171, E4=325, E5=677, E6=220, E7=123, E8=91. Per-row score files are checkpointed under `data/v5/scores/`. The resumed queue will summarize E2 then attempt E3 deep smoke/full, minimal E4 smoke/full, and the episode table. P2 E5-E7 remain unrun.

E2 live checkpoint at 15:04: eleven of 2,448 jobs reached (B/M0/logreg through E10), including 391 BASE label events in E10. The process and 90-minute timer remain active.

E2 live checkpoint (2026-10-25 15:27 local): 51/2,448 jobs completed at 24.1 elapsed minutes; process PID 24140 remains active under the 90-minute hard box (queue PID 2288). The last committed job is W/B3/logreg/E7. E2 per-job scores are checkpointed. Queue order remains E2 summary, E3 smoke/full (150-minute box), E4 smoke/minimal full (90-minute box), then episode table; E5-E7 P2 are not included. Do not treat partial E2 results as final.

E2 live checkpoint (2026-10-25 15:29 local): 53/2,448 jobs, elapsed 26.0 minutes; PIDs 24140/2288 active. This is a timeboxed sweep, so report actual completed combinations separately from planned grid after summarization.

E2 live checkpoint (2026-10-25 15:30 local): 56/2,448 jobs, elapsed 27.6 minutes; entered B/B3/GBM (E0 and E1 completed), active PIDs unchanged. LightGBM emits an eval_set deprecation warning; no run failure observed. Per-job results continue to checkpoint.

E2 live checkpoint (2026-10-25 15:31 local): log advanced to 66/2,448 jobs at 28.4 minutes and is in W/B3/GBM. Queue PID 2288 and Python 24140 remain active. Results are partial until timebox completion and E2 summary.

V6 audit update (2026-10-25 15:49 local): found that the old V6 severity labels were shifted forward by N rows and their expanding reference ranks included outcomes not yet known at forecast origin. Prior V6 labels, dependent model outputs, and report were preserved under data/v6/archive_pre_forward_label_fix_20261025; old performance results are invalid as early-warning evidence. Corrected origin-aligned, past-only, full-window labels are rebuilding (71/159 asset files at the latest check); label and evaluator audit tests pass. Corrected V6 model reruns are pending. V5 E2 remains active at 74/2448 jobs, 43.6 minutes into its 90-minute timebox; queue PID 2288 remains active.
