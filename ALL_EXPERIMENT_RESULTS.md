# Consolidated Experiment Results

**Project:** TDA vs classical early-warning indicators and deep learning for market-crash prediction  
**Snapshot date:** 25 October 2026  
**Workspace:** this project directory

This is the top-level record of the experiments and results currently present in the workspace. It reports headline results and points to the run-level tables and raw outputs so each result can be traced. “Completed” refers to recorded runs; it does not imply that every proposed experiment or grid cell was finished.

## Executive summary

- The project explored classical critical-slowing-down (CSD) indicators, persistent-homology TDA features, sequence models, cross-index/peer information, broad asset panels, and the newer V3 feature/asset-universe experiments.
- Across the four-index CSD/TDA LSTM comparison, combined features had the best mean average precision (AP): **0.0307**, versus **0.0144** for classical features and **0.0124** for TDA alone. Results varied sharply by era; this is not evidence that TDA alone is a reliable crash warning system.
- In a separate four-index cross-attention experiment, mean AP was **0.0436** with all index inputs, **0.0366** with only the other indices, and **0.0285** with only the target index. A separate target-specific peer GRU reported **0.0508** mean AP. These are single-run/regime-sensitive experiments and should not be treated as definitive superiority.
- The broad V2 benchmark’s best complete feature mode was combined (mean AP **0.0366** overall); landscape was strongest for energy and international datasets, and full features for group B cross-asset. Across complete dataset/mode comparisons, TDA-rich modes beat classical/combined on 3 of 6 evaluable datasets. Results are highly dataset- and period-dependent.
- V3 preparation, feature generation, leakage audit, and limited experiments completed. V3-A covered **F0 only**; F1–F4 were not evaluated. V3-D covered four asset universes for five folds, while its since-1997/history-cut universe has only one usable fold. Its event-recall values came with very high alarm-day rates, so they do not establish useful practical warning performance.
- The available results do **not** support claiming a dependable crash-prediction system or a deployable trading signal. Positive labels are rare, many results are based on few crash episodes or a single seed, and temporal regime changes are substantial.

## 1. Task, labels, and metric notes

The main forecasting task labels a date positive when the target asset’s minimum simple return in the following 15 trading sessions crosses a fixed downside threshold. The unified benchmark estimates per-asset thresholds using calibration/training history rather than the held-out test period. The V3 pipeline has its own past-only rolling label configurations; do not combine its scores numerically with the earlier benchmark scores.

Chronological test folds generally cover 2011–2015, 2016–2020, and 2021 onward, with proportional splits for shorter histories. Validation/test boundaries use a purge gap to limit overlap from forward labels. Average precision is the primary ranking metric because crash labels are rare; compare AP with the positive prevalence in the same test fold. Thresholded precision, recall, and alarm rates depend on the validation-selected operating threshold. V3 additionally reports event-level catches and random alarm baselines.

**Cross-experiment caution:** thresholds, samples, features, split designs, alarm rules, and evaluation units differ. AP values are comparable only within their experiment/protocol unless the underlying rows and labels are shown to match. A high event recall with alarms on most days is not a useful warning result by itself.

## 2. Main results by experiment family

### 2.1 Original four-index CSD vs TDA LSTM

| Feature family | Mean AP | Mean ROC AUC | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Classical CSD | 0.0144 | 0.4872 | 1.61% | 12.30% | 0.0244 | 0.0879 |
| TDA | 0.0124 | 0.5235 | 0.00% | 0.00% | 0.0000 | 0.0739 |
| **Combined CSD + TDA** | **0.0307** | **0.6174** | **3.42%** | **18.28%** | **0.0567** | **0.0745** |

Combined AP by test era: **0.0073** (2011–2015), **0.0340** (2016–2020), **0.0508** (2021 onward). The TDA model produced no positive alerts at the selected cutoff in the aggregate rows. The combined model’s performance is not stable across periods.

### 2.2 Extended cross-asset CSD/TDA comparison

Reported mean AP/AUC: classical **0.0268 / 0.5845**; TDA **0.0201 / 0.5393**; combined **0.0198 / 0.5380**. At the selected threshold, reported precision/recall were classical **3.45% / 16.07%**, TDA **0.91% / 3.83%**, and combined **2.21% / 5.35%**. This extension did not show a consistent gain from adding TDA.

### 2.3 Peer-index and cross-index neural models

| Experiment / input | Mean AP | Mean AUC | Additional recorded result |
|---|---:|---:|---|
| Cross-index temporal attention, target index only (`self_only`) | 0.0285 | 0.7179 | Mean recall 10.83% |
| Cross-index temporal attention, other three indices only (`other_only`) | 0.0366 | 0.7538 | Mean recall 14.55% |
| Cross-index temporal attention, all four indices | **0.0436** | **0.7637** | Mean precision 3.45%, recall 17.45%, alarm rate 5.47% |
| Pooled peer GRU | 0.0357 | 0.6775 | Approx pooled precision 6.90%, recall 29.29% |
| Target-specific peer GRU v2 | **0.0508** | **0.7022** | Approx pooled precision 4.92%, recall 18.69% |
| Target-specific top-two ensemble v2 | 0.0414 | 0.6949 | Approx pooled precision 5.89%, recall 21.72% |

The cross-index attention results are means over three chronological folds. The peer GRU v2 is a separate experiment and its mean is not directly interchangeable with the attention model’s mean. Peer-model results are strongly regime-dependent, especially around 2016–2020; they need repeated-seed and episode-level validation.

### 2.4 Unified broad benchmark V1 and V2

V2 covers classical, TDA scalar, landscape, combined, and full feature modes across eight model families and group/ticker levels. In the complete group-level comparisons, overall mean AP by mode was **0.0366 combined**, **0.0323 classical**, **0.0316 landscape**, **0.0267 full**, and **0.0245 TDA** (summaries are from the run log; see the per-run tables for dataset and fold context). Landscape beat scalar TDA for each dataset. TDA-rich modes exceeded classical/combined on 3 of 6 datasets with complete feature-mode coverage. The best individual completed mean-AP cells recorded were international landscape/LSTM **0.1102**, tech TDA/BiLSTM **0.1038**, and energy landscape/TCN **0.0971**. Attention-BiLSTM did not consistently outperform BiLSTM.

| Grid | Planned cells | Completed | Not run | Skipped for insufficient data |
|---|---:|---:|---:|---:|
| V1 | 420 | 406 | 14 | 0 |
| V2 | 800 | 696 | 64 | 40 |

All 40 skipped V2 cells are crypto group-level runs, structurally unevaluable because the relevant test interval had no positives. The not-run cells are pooled ticker-level tasks left incomplete when compute stopped. V1’s 14 not-run cells are also pooled ticker-level tasks. These grids are **not fully complete**.

### 2.5 Qualitative and episode-level indicator checks

An existing diagnostic reports indicator threshold crossings around 31 objective crash episodes: TDA L1 crossed in **8/31 (25.8%)**, rolling variance in **24/31 (77.4%)**, and GARCH volatility in **23/31 (74.2%)**. This is a retrospective episode crossing count, not a full forecast evaluation: crossings can occur after an episode begins, and false alarms away from crash episodes are not included.

A hand-selected historical review covers 2011, 2015–16, December 2018, COVID-19 in 2020, and the 2022 bear market. The report describes a rise in the 50-day TDA measure in four of five selected periods and in the 100-day measure in all five. These plots are qualitative checks, not unbiased out-of-sample estimates. Event-aligned figures and per-event detail are linked below.

### 2.6 V3 experiments (partial)

V3 prepared 159 price targets and 15 state series on 25,781 dates, with seven label configurations. The BASE configuration contains 1,044,519 valid outcomes, 20,445 positive outcomes, and 1.96% pooled prevalence. Feature sets contain F0=42, F1=61, F2=114, F3=1,027, and F4=1,109 features. A point-in-time audit tested 11,884 numeric comparisons (11,440 TDA); maximum absolute difference was 2.9e-6. The audit is a consistency check, not proof against every possible leak.

**EXP-V3-A:** only F0 completed all five walk-forward folds for GBM, logistic regression, GRU, and TCN. All-assets mean AP was GBM **0.0429**, logistic regression **0.0603**, GRU **0.0416**, and TCN **0.0431**. At the R3 b=1 rule, mean event recall was 0.734/0.999/1.000/1.000 respectively, but mean alarm-day rates were **64.2% / 99.95% / 99.97% / 99.99%**. Near-everyday alarms make those recall values operationally weak. F1–F4 were not run; there is no complete architecture or feature-set comparison.

**EXP-V3-D:** five folds completed for US4, approx30_index_etf, all_nonstock, and all_assets. At R3 b=1, mean event recall / event precision / test alarm-day rate were:

| Universe | Event recall | Event precision | Alarm-day rate | Random baseline mean events caught |
|---|---:|---:|---:|---:|
| US4 | 0.600 | 0.203 | 57.7% | 9.49 |
| Approx. 30 indices and ETFs | 0.873 | 0.445 | 77.2% | 80.55 |
| All non-stock assets | 0.739 | 0.368 | 63.5% | 257.33 |
| All assets | 0.849 | 0.465 | 79.3% | 318.56 |

The universes have different event counts and should be compared against their own random baselines. The 1997-history universe has only the 2000–2004 fold; it is not enough to support a history-length conclusion. **EXP-B, EXP-F, final model export, EXP-C, and EXP-E were not run. No final V3 model is trained or frozen.**

## 3. Completion status and next interpretation

| Area | Recorded state | Interpretation |
|---|---|---|
| Original CSD/TDA and cross-asset analyses | Results and reports exist | Completed for their stated protocols; qualitative/full-sample diagnostics have limitations |
| Peer-index experiments | Results and model artifacts exist | Completed exploratory runs; single-seed/regime-sensitive |
| V1 benchmark | 406/420 cells done | Partial grid |
| V2 benchmark | 696/800 done; 40 skipped; 64 not run | Partial grid |
| V3 feature pipeline and audit | Feature files and audit exist | Preparation completed; audit scope is finite |
| V3-A | F0 only across five folds and four models | Incomplete feature comparison; alarm rates are too high for a practical warning claim |
| V3-D | Four universes across five folds; one history-cut fold | Partial universe analysis |
| V3 remaining phases | Not run | No final V3 model or complete V3 conclusion |

The safest overall conclusion is that cross-index and peer information is promising in some experiments, combined CSD+TDA beats either feature family alone in the original four-index sequence run, and broader TDA gains depend on the dataset and feature representation. No single approach wins consistently across periods and datasets. The V3 experiment currently demonstrates pipeline execution and diagnostics, not a validated production-quality predictor.

## 4. Exhaustive result records and artifacts

The following files retain the detailed fold, model, dataset, or event outputs. They are the source of truth for results beyond the summary tables above.

- [EXPERIMENT_RESULTS.md](./EXPERIMENT_RESULTS.md) — earlier consolidated report for the four-index, peer, and cross-index studies.
- [EXPERIMENTS_FULL_REPORT.md](./EXPERIMENTS_FULL_REPORT.md) — detailed V1/V2 methods and experiment account.
- [RUN_RECORD.md](./RUN_RECORD.md) — per-cell AP and completion status for V1/V2.
- [GLM_EXPERIMENT_LOG.md](./GLM_EXPERIMENT_LOG.md) — chronological experiment log and engineering notes, including V3.
- [COMPLETE_WORK_REPORT.md](./COMPLETE_WORK_REPORT.md) and [FINAL_PROJECT_REPORT.md](./FINAL_PROJECT_REPORT.md) — broader project writeups; consult alongside the run records where status summaries differ in age.
- [data/benchmark/run_record_v1.csv](./data/benchmark/run_record_v1.csv) and [data/benchmark/run_record_v2.csv](./data/benchmark/run_record_v2.csv) — machine-readable planned-run accounting and metrics.
- [data/benchmark/all_results_v1.csv](./data/benchmark/all_results_v1.csv) and [data/benchmark/all_results.csv](./data/benchmark/all_results.csv) — benchmark prediction/result rows and per-ticker breakdowns.
- [data/forecasting/](./data/forecasting/) — original CSD/TDA, cross-asset, pooled-peer, and target-specific peer outputs and model artifacts.
- [data/cross_index_forecasting/](./data/cross_index_forecasting/) — cross-index attention metrics and saved model/scaler artifacts.
- [data/v3/exp_a_results.csv](./data/v3/exp_a_results.csv), [data/v3/exp_d_results.csv](./data/v3/exp_d_results.csv), [data/v3/leak_test.csv](./data/v3/leak_test.csv) — V3 experiment rows and leakage-audit comparisons.
- [data/group_a_original_indices/](./data/group_a_original_indices/) and [data/plots/](./data/plots/) — objective-label, indicator, and qualitative output reports/figures.

For future continuation, resume only the incomplete work that is still part of the approved plan, and retain these raw CSVs and snapshots so results can be reproduced. Recheck the exact run records before presenting updated completion counts.
