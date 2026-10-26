# Complete Project Work and Results Report

**Project:** Comparing topological early warning signals with classical volatility indicators for financial crash prediction, extended with cross-index deep learning  
**Report date:** 25 October 2026  
**Scope:** This document consolidates the project files and results available in this workspace, including the existing TDA/CSD work and the more recent forecasting experiments. It distinguishes historical results already present in the project from the later experiments and records the limitations needed to interpret all scores correctly.

---

## 1. Executive summary

The project now contains three related lines of work:

1. **TDA and classical-indicator replication/diagnostics.** Existing scripts implement persistent-homology features and classical indicators, evaluate their behavior around objective future drawdowns, and create paper-style figures and qualitative event comparisons.
2. **Leakage-controlled sequence-model comparison.** A new 20-session LSTM experiment compares classical features, joint TDA features, and their combination for predicting a large decline within 15 sessions across the S&P 500, Dow Jones, Nasdaq Composite, and Russell 2000.
3. **Cross-index deep learning.** A new temporal-attention model compares each target index using only its own history, only the other three indices, or all four indices. In the three test blocks, the peer-only model has higher mean average precision (AP) than the self-only model; the all-index model has the highest mean AP. These are promising preliminary results, not conclusive proof of a dependable crash-warning advantage.

The most important quantitative result from the newer cross-index experiment is mean AP of **0.0366 using only peer indices**, **0.0285 using only the target index**, and **0.0436 using all indices**. The mean test prevalence is low and varies by period, so AP must be read against each test block's prevalence. The results come from one model run per fold and mode, and the test blocks contain few positive days. Repeated seeds, uncertainty intervals, and episode-level analysis remain necessary.

The earlier TDA-versus-classical LSTM comparison produced mean AP of **0.0307 for combined features**, **0.0144 for classical features**, and **0.0124 for TDA features**. TDA alone did not trigger alerts at the selected validation threshold in any of the three fold-level aggregate rows. Combined features did better on average, but their performance varied markedly by period.

These results do **not** establish that a neural network is itself a topological method. Neural networks can learn nonlinear representations, but persistent homology explicitly measures topological properties such as connected components and loops. The cross-index model is a multivariate temporal representation and attention model; it is an additional approach that complements TDA rather than replacing or reproducing it.

## 2. Project question and scope

The original project asks whether TDA-based early-warning indicators can give useful advance warning of market crashes compared with classical critical-slowing-down (CSD) indicators. The extended question is whether information in other market indices can improve a crash forecast for a target index, and whether a sequence model can learn that relationship.

The current experiments focus on four US equity indices, represented in the data by:

| Symbol | Index |
|---|---|
| `^GSPC` | S&P 500 |
| `^DJI` | Dow Jones Industrial Average |
| `^IXIC` | Nasdaq Composite |
| `^RUT` | Russell 2000 |

The core prediction task is a binary forecast: as of a trading date, predict whether the index's minimum simple return over the next 15 trading sessions crosses a fixed downside threshold. This is a measurable operational definition of a crash-like event. It does not claim to identify every financial crisis, and the resulting positive label dates can cluster around the same market episode.

## 3. Data and label construction

### 3.1 Price data

The existing Group A files contain daily prices for the four indices. The existing objective-label report describes a sample from 1997-01-03 through 2026-09-21, with 7,475 price observations and 7,460 dates for which the full 15-session forward outcome is available. Exact available dates can differ slightly by source-file version and index holidays. The modeling scripts align the indices by date and use only complete observations needed for each feature and label.

The source files and extraction code are retained in the workspace, including `download_market_data.py` and the data folders. The project contains historical data and generated panels rather than a data-download step in every model run. Re-running a live download later may produce revised data, so a final submission should archive the exact inputs used.

### 3.2 Existing full-sample objective diagnostic

An existing Group A analysis labels a date positive when the S&P 500's minimum simple return over the next 15 sessions is at or below the **full-sample second percentile**, -11.4133%. This produces 150 positive dates grouped into 31 contiguous runs. Its report is `data/group_a_original_indices/objective_crash_detection_report.txt`.

This is a useful retrospective diagnostic, but the threshold is estimated from the full sample, which includes the later evaluation period. It is therefore not the label protocol used for the newer out-of-sample forecasting experiments. It should not be described as a fully leakage-free forecast benchmark.

### 3.3 Frozen calibration-only thresholds for the new forecasts

For the forecasting experiments, each index receives its own threshold: the second percentile of its forward 15-session minimum simple return, calculated only from origins whose complete outcome was observable by the end of the calibration period in 2006. The last eligible label origin is 2006-12-07; its complete 15-session outcome is available by 2006-12-31. Those per-index cutoffs are then frozen for later dates.

| Index | Frozen threshold |
|---|---:|
| `^GSPC` | -10.4492% |
| `^DJI` | -11.1293% |
| `^IXIC` | -18.3202% |
| `^RUT` | -14.0087% |

For an index with price (P_t), the label is positive when:

```text
min((P[t+k] - P[t]) / P[t] for k = 1,...,15) <= frozen_threshold[index]
```

The label is assigned to the forecast origin date (t), because the following 15 sessions contain a decline beyond the threshold. This is a forward outcome label; it is not information available at date (t). The forecasting scripts exclude rows where the full outcome is unavailable.

Across the 29,500 rows in the first modeling panel, the positive share was 1.97% for `^DJI`, 2.75% for `^GSPC`, 1.15% for `^IXIC`, and 2.13% for `^RUT`. Exact fold-level prevalence differs from these full-panel proportions.

## 4. Existing TDA and CSD work

### 4.1 Methods represented in the project

The project already includes persistent-homology/TDA feature calculations, classical CSD indicators, paper replication and extension scripts, crash-label utilities, and plotting scripts. Relevant source files include:

- `tda_replication.py`
- `csd_baseline.py`
- `crash_labels.py`
- `paper_reproduction_figures.py`
- `paper_exact_detection.py`
- `paper_extension.py`
- `episode_extension.py`
- `qualitative_leadtime.py`
- `generate_final_assets.py`

The feature work includes persistent-homology landscape/norm summaries at point-cloud windows of 30, 50, and 100 observations, as well as classical rolling variance, lag-1 autocorrelation, skewness, realized volatility, and GARCH volatility in the relevant analyses. The exact set depends on the experiment; see each script and its accompanying output report before comparing runs.

### 4.2 Existing episode detector result

The existing objective episode diagnostic checks whether an indicator exceeds its preceding-100-observation mean plus 1.5 standard deviations in a window around each of 31 positive episodes. It reports whether a signal crossed the baseline in at least one of the 30-, 50-, or 100-observation configurations:

| Indicator | Episodes with a crossing | Share |
|---|---:|---:|
| TDA L1 | 8 / 31 | 25.8% |
| Rolling variance | 24 / 31 | 77.4% |
| GARCH volatility | 23 / 31 | 74.2% |

These are episode detection counts, **not** classification accuracy, precision, recall, or a clean forecast lead-time score. The crossing search includes observations after the objective episode start (up to five trading observations), and false alarms outside positive episodes are not part of this count. The report itself correctly cautions against interpreting these counts as a complete prediction evaluation.

### 4.3 Existing qualitative TDA event comparison

The existing `future_period_performance_report.txt` compares TDA behavior around selected periods: 2011, 2015–16, December 2018, COVID-19 in 2020, and the 2022 bear market. Its summary describes an increase in the 50-observation TDA measure in four of five selected periods and in the 100-observation measure in all five. These are hand-selected retrospective periods; they are useful for visual/qualitative validation, not an unbiased estimate of out-of-sample predictive performance. The December 2018 behavior varies by window, and some signal peaks occur during or after the decline. The historical sample starts in 1997, so the requested pre-event context for the dot-com episode is incomplete.

The workspace contains event and project figures under `assets/` and `data/plots/`, including market overview, pipeline, objective labels, indicator examples, correlations, and event-aligned views. `forecasting_qualitative.py` also creates five event panels for the dot-com period, Lehman/global financial crisis, 2011 US debt-ceiling stress, COVID-19 selloff, and 2022 bear market. These plots help inspect timing and index differences; they do not establish predictive skill by themselves.

## 5. New experiment A: TDA/CSD sequence-model comparison

### 5.1 Design

The newer `crash_forecasting.py` study turns daily indicators into supervised sequence inputs and compares three feature sets using the same basic LSTM architecture:

1. **Classical:** per-index CSD/volatility features, including rolling autocorrelation, variance, skewness, and realized volatility.
2. **TDA:** joint four-index TDA landscape/norm features at 50- and 100-observation windows.
3. **Combined:** classical and TDA features together.

The network consumes a 20-session history and predicts the target index's binary 15-session event label. The LSTM hidden size is 24. Features are standardized using training data only. The study fits per-index models and reports daily test predictions.

Evaluation is chronological. Test blocks are 2011–2015, 2016–2020, and 2021 through the latest available observations in the panel (September 2026 for the current data). Validation precedes each test block. A 25-calendar-day gap is used before validation and test blocks to reduce overlap from the 15-session forward labels. A warning cutoff is selected from the validation-score 95th percentile, targeting roughly 5% validation alert days. Scores are model outputs and **are not calibrated probabilities**.

### 5.2 Aggregate results

Average precision is the primary metric because positive labels are rare. ROC AUC measures ranking, Brier score measures squared probability error (although model scores are not calibrated probabilities), and precision/recall/alarm rate are reported at the validation-selected cutoff. A dash means no positive test examples for the relevant aggregate metric or a metric that cannot be computed.

| Test block | Model | Prevalence | AP | ROC AUC | Brier | Precision | Recall | Alarm rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2011–2015 | Classical LSTM | 0.0095 | 0.0082 | 0.3008 | 0.0783 | 0.0267 | 0.0417 | 0.0149 |
| 2011–2015 | TDA LSTM | 0.0095 | 0.0094 | 0.4953 | 0.0243 | 0.0000 | 0.0000 | 0.0000 |
| 2011–2015 | Combined LSTM | 0.0095 | 0.0074 | 0.3024 | 0.0626 | 0.0163 | 0.0417 | 0.0244 |
| 2016–2020 | Classical LSTM | 0.0218 | 0.0199 | 0.4483 | 0.1191 | 0.0218 | 0.3273 | 0.3284 |
| 2016–2020 | TDA LSTM | 0.0218 | 0.0165 | 0.3742 | 0.0491 | 0.0000 | 0.0000 | 0.1851 |
| 2016–2020 | Combined LSTM | 0.0218 | 0.0340 | 0.7030 | 0.0871 | 0.0214 | 0.1818 | 0.1855 |
| 2021 onward | Classical LSTM | 0.0070 | 0.0150 | 0.7123 | 0.0663 | 0.0000 | 0.0000 | 0.0005 |
| 2021 onward | TDA LSTM | 0.0070 | 0.0113 | 0.7012 | 0.1482 | 0.0000 | 0.0000 | 0.0293 |
| 2021 onward | Combined LSTM | 0.0070 | 0.0508 | 0.8470 | 0.0737 | 0.0650 | 0.3250 | 0.0349 |
| Mean across blocks | Classical LSTM | — | 0.0144 | 0.4872 | 0.0879 | 0.0161 | 0.1230 | 0.1146 |
| Mean across blocks | TDA LSTM | — | 0.0124 | 0.5236 | 0.0739 | 0.0000 | 0.0000 | 0.0715 |
| Mean across blocks | Combined LSTM | — | **0.0307** | **0.6174** | 0.0745 | **0.0342** | **0.1828** | 0.0816 |

The three blocks contain respectively 48, 110, and 40 positive rows among their pooled four-index evaluation rows. There are no positive `^IXIC` examples in the 2011–2015 and 2021+ blocks, so ticker-level ROC AUC for those cases is undefined.

### 5.3 Interpretation

The combined feature model has the strongest mean AP and mean recall in this run, but the result is not uniform: it performs poorly in 2011–2015, improves in 2016–2020, and is strongest in the 2021+ block. TDA-only features achieve mean AP of 0.0124 and no true positive alerts at the chosen cutoffs. This does not mean TDA contains no useful information in every setting: the model, label, alert rule, and limited positive counts constrain the conclusion. It does mean this experiment does not show that TDA-only LSTM forecasts outperform classical-only forecasts.

The 2016–2020 classical model's alarm rate is about 32.8%, much higher than the nominal 5% target. A validation-selected threshold does not guarantee the same alarm rate in a shifted test distribution. In 2011–2015 the test prevalence is under 1%, and AP values close to that baseline indicate weak ranking. Interpret AP against each block's own prevalence rather than treating a single value as universal.

## 6. New experiment B: cross-index temporal attention

### 6.1 Research question and comparison

For every target index, can a model predict the target's future crash label using the other indices' histories? The experiment compares three information settings with the same model family and test dates:

- **`self_only`:** target index's own trailing features only.
- **`other_only`:** the other three indices' trailing features only; the target's complete input history is masked before its temporal encoding and the target index is excluded as an attention source. A learned target-index query is retained so the model knows which index it is predicting.
- **`all_indices`:** all four index histories are available.

For every index, the per-index input features are one-day return; 5-, 20-, and 60-day momentum; 20- and 60-day realized volatility; 20-day downside volatility; 60- and 252-day drawdowns; a 20/60-day volatility ratio; and 20-day rolling skewness. The model encodes each index's 30-session feature history with an LSTM, then uses a learned target query and temporal attention across the index embeddings to produce a forecast.

This is a multivariate temporal attention architecture. It tests cross-index information transfer. It does not compute persistent homology and should not be labeled a TDA model.

### 6.2 Validation and test protocol

The experiment reuses the frozen calibration thresholds and label dates from Experiment A. It uses the same chronological test blocks (2011–2015, 2016–2020, 2021 onward) and the 25-calendar-day purging gap. Standardization is fit on the training block. Each fold/mode has a deterministic seed and one training run; the saved results are not averages over random seeds. Training uses at most 10 epochs with validation AP for early stopping and a positive-class weight capped at 10. The cutoff is set to the validation 95th percentile, so test precision/recall describe an operating point chosen on validation, not an optimized test cutoff.

### 6.3 Aggregate cross-index results

| Test block | Prevalence | Mode | AP | ROC AUC |
|---|---:|---|---:|---:|
| 2011–2015 | 0.0095 | Self only | 0.0102 | 0.5235 |
| 2011–2015 | 0.0095 | Other indices only | 0.0117 | 0.6129 |
| 2011–2015 | 0.0095 | All indices | 0.0106 | 0.5652 |
| 2016–2020 | 0.0218 | Self only | 0.0554 | 0.8071 |
| 2016–2020 | 0.0218 | Other indices only | 0.0632 | 0.7497 |
| 2016–2020 | 0.0218 | All indices | **0.0863** | **0.8252** |
| 2021 onward | 0.0070 | Self only | 0.0199 | 0.8232 |
| 2021 onward | 0.0070 | Other indices only | **0.0349** | 0.8986 |
| 2021 onward | 0.0070 | All indices | 0.0337 | **0.9008** |
| Mean across blocks | — | Self only | 0.0285 | 0.7179 |
| Mean across blocks | — | Other indices only | 0.0366 | 0.7538 |
| Mean across blocks | — | All indices | **0.0436** | **0.7637** |

Other summary metrics at the validation-selected cutoff, averaged across blocks:

| Mode | Mean Brier | Mean precision | Mean recall | Mean alarm rate |
|---|---:|---:|---:|---:|
| Self only | 0.0426 | 0.0279 | 0.1083 | 0.0447 |
| Other indices only | **0.0377** | 0.0255 | 0.1455 | 0.0451 |
| All indices | 0.0381 | **0.0345** | **0.1745** | 0.0547 |

Peer-only AP exceeds self-only AP in all three blocks. The all-index model is best by AP in 2016–2020 and on the unweighted mean across the three blocks. The other-only model is best by AP and ROC AUC in the 2021+ block, although at its selected operating threshold it records no true positive alerts in that block. Ranking and thresholded alert performance answer different questions.

The early block is particularly weak: its prevalence is 0.95%, and all three AP scores are close to that value. The more favorable 2016–2020 and 2021+ ranking scores should be treated cautiously because there are few independent crash episodes and one seed per experiment. The mean is an unweighted average of three time blocks, not a pooled daily estimate.

### 6.4 Per-index context

The following AP values show the main per-target pattern. Ticker order in each cell is `^GSPC`, `^DJI`, `^IXIC`, `^RUT`.

| Test block | Positive rows (same ticker order) | Self-only AP | Other-only AP | All-index AP |
|---|---|---|---|---|
| 2011–2015 | 11, 18, 0, 19 | .0174, .0064, n/a, .0228 | .0214, .0109, n/a, .0200 | .0151, .0062, n/a, .0153 |
| 2016–2020 | 37, 35, 13, 25 | .0933, .0792, .0274, .0390 | .0845, .0759, .0291, .0653 | .1067, .1089, .0523, .0879 |
| 2021 onward | 26, 4, 0, 10 | .0625, .0098, n/a, .0347 | .0713, .0114, n/a, .0288 | .0810, .0140, n/a, .0338 |

Per-index AP can be unstable with very few positives; for example, there are only four Dow Jones positive rows in the final block, and no Nasdaq positive rows. Do not overinterpret small differences in these cells.

## 7. Qualitative validation and user interfaces

### 7.1 Event plots

The project includes plots for selected historical drawdowns and indicator histories. They help answer whether an indicator's level changed before, during, or after a visible event and whether indices behaved differently. They are retrospective visual checks. They are subject to event selection and should be paired with the objective-label and walk-forward results rather than treated as separate performance proof.

The forecasting event panels are located at `data/forecasting/event_*.png`. Indicator traces are log-transformed/z-scored and clipped for readable plotting because raw TDA/variance scale differences otherwise obscure the event history. Any visual standardization is a plotting choice and is not an input transformation in the forecast model unless the modeling code separately specifies it.

### 7.2 Existing feature-panel interface

`forecast_app.py` is a Streamlit interface for the earlier LSTM models. It accepts a feature-panel CSV with the required date/ticker/features, displays available benchmark performance and forecasts, and supports a result-table download. It uses the trained classical, TDA, and combined LSTM artifacts under `data/forecasting/`.

### 7.3 Raw-price cross-index interface

`cross_index_app.py` is a separate Streamlit interface for the cross-index attention models. It accepts a CSV of date plus the four index price columns, builds trailing features, and displays `self_only`, `other_only`, and `all_indices` risk scores for each target along with benchmark metrics. The uploaded historical price rows are processed as input to produce current scores. These scores are model risk scores, not calibrated event probabilities.

Both interfaces were exercised with Streamlit's app testing utilities. The earlier interface rendered without exceptions and displayed its benchmark rows. The cross-index interface was tested by uploading the existing Group A price CSV; it rendered without exceptions and returned score rows for all four target indices. This verifies basic interface execution, not long-term operational robustness or predictive quality.

Run from the project directory using the Python launcher (`py`):

```powershell
py -3.12 -m streamlit run forecast_app.py
py -3.12 -m streamlit run cross_index_app.py
```

Dependencies are listed in `requirements.txt`. The report and code use `py` commands for Windows.

## 8. Model files, outputs, and reproducibility map

### 8.1 New TDA/CSD forecasting outputs

`data/forecasting/` contains:

- `classical_lstm.pt`, `tda_lstm.pt`, `combined_lstm.pt`: trained model weights.
- Corresponding `*_scaler.joblib` files: feature scalers.
- `label_thresholds.csv`: per-index frozen calibration thresholds.
- `modeling_panel.csv`: date/ticker labels and model features.
- `walk_forward_metrics.csv`: fold and per-index evaluation metrics.
- `report.txt`: experiment protocol and concise summary.
- `event_*.png`: event-aligned qualitative plots.
- Additional `cross_asset_*` and `peer_index_*` files described under preserved work below.

### 8.2 New cross-index outputs

`data/cross_index_forecasting/` contains:

- `self_only.pt`, `other_only.pt`, `all_indices.pt`: trained weights.
- Matching `*_scaler.joblib` files.
- `metadata.json`: model/feature metadata.
- `cross_market_feature_panel.csv`: generated daily panel.
- `cross_index_walk_forward_metrics.csv`: fold-level and per-index results.
- `report.txt`: concise run summary.

### 8.3 Main source and documentation files

- `crash_forecasting.py`: TDA/classical/combined LSTM training and evaluation.
- `forecasting_qualitative.py`: event visualization generation.
- `forecast_app.py`: feature-panel inference interface.
- `cross_index_forecasting.py`: self-only/other-only/all-index temporal attention training and evaluation.
- `cross_index_app.py`: raw-price upload and cross-index inference interface.
- `README.md`: project entry point and commands.
- `requirements.txt`: dependency list.
- `FINAL_PROJECT_REPORT.md`: an earlier detailed report. Some result tables there predate the final reruns; this document records the latest metric CSV values and should be used when they differ.

Reproduction commands for the new experiments are:

```powershell
py -3.12 crash_forecasting.py
py -3.12 forecasting_qualitative.py
py -3.12 cross_index_forecasting.py
```

Running the scripts retrains models and writes outputs. Preserve a copy of the current artifacts before rerunning if you need to compare the saved weights with the reported results. Random seeds make the run deterministic for the current software and data, but one seed does not quantify training variability.

## 9. Other experiments found and preserved

The workspace contains additional experiments that appear to be separate team work or earlier project branches. They were left in place and are not merged into the metrics above:

- `data/forecasting/cross_asset_*`: cross-asset classical/TDA/combined LSTM artifacts, metrics, and report.
- `data/forecasting/peer_index_deep_learning_metrics.csv` and `peer_index_deep_learning_report.txt`.
- `data/forecasting/peer_index_deep_learning_v2_metrics.csv` and `peer_index_deep_learning_v2_report.txt`.
- `data/forecasting/peer_index_gru.pt` and `peer_index_transformer.pt`, with their scalers.
- `data/forecasting/cross_asset_walk_forward_metrics.csv` and `cross_asset_forecasting_report.txt`.
- `data/benchmark/` run-level benchmark tables, logs, and qualitative lead-time reports.
- `EXPERIMENTS_FULL_REPORT.md`, `EXPERIMENT_RESULTS.md`, and `GLM_EXPERIMENT_LOG.md`, which contain longer historical experiment notes. The full report includes interim/partial run statuses and is not a substitute for checking each current output CSV.
- Other scripts at the repository root, including `cross_asset_forecasting.py`, `peer_index_forecasting.py`, `peer_index_forecasting_v2.py`, `run_cross_asset_extension.py`, `run_benchmark.py`, and `analyze_benchmark.py`.

Their results should be reviewed on their own protocol before comparing them with the two newer experiments in this report. No existing experiment/model files were intentionally deleted as part of this work.

### 9.1 Existing pooled peer-index GRU and Transformer

The earlier `peer_index_forecasting.py` run also tests the peer-only idea, using a 30-session sequence and 10 peer-summary features (peer returns, volatility, mean/dispersion, and correlation). The target index's own returns are excluded. Its saved report gives the following averages across the three test blocks:

| Model | Mean AP | Mean ROC AUC | Mean precision | Mean recall | Mean F1 | Mean Brier |
|---|---:|---:|---:|---:|---:|---:|
| Peer GRU | **0.0357** | **0.6775** | 2.35% | 17.58% | 0.0415 | **0.0661** |
| Peer Transformer | 0.0287 | 0.5642 | **3.94%** | **19.77%** | **0.0603** | 0.0781 |

Approximate pooled counts reconstructed in the existing report are 15,804 observations and 198 positive rows for each model. The GRU generated 840 alarms (6.90% precision, 29.29% recall); the Transformer generated 1,253 alarms (4.23% precision, 26.77% recall). The fold-level GRU AP values were 0.0110, 0.0836, and 0.0125 for 2011–2015, 2016–2020, and 2021 onward. The corresponding Transformer values were 0.0081, 0.0394, and 0.0386. The concentration of GRU performance in 2016–2020 and its lack of thresholded true alerts in the other two periods show regime dependence.

### 9.2 Existing target-specific peer model comparison

`peer_index_forecasting_v2.py` trains one model per target and compares GRU, attention-BiLSTM, TCN, and Transformer, with a validation-selected top-two ensemble. Its existing report gives:

| Model | Mean AP | Mean ROC AUC | Mean precision | Mean recall | Mean F1 |
|---|---:|---:|---:|---:|---:|
| Target-specific GRU | **0.0508** | 0.7022 | 2.97% | 11.12% | 0.0447 |
| Attention-BiLSTM | 0.0400 | **0.7068** | 1.77% | 7.51% | 0.0274 |
| TCN | 0.0401 | 0.6981 | 2.30% | 12.02% | 0.0380 |
| Transformer | 0.0362 | 0.6836 | 2.14% | **15.41%** | 0.0351 |
| Top-two ensemble | 0.0414 | 0.6949 | 2.07% | 11.40% | 0.0349 |

The existing v2 report's approximate pooled statistics are based on 15,804 rows and 198 positive labels: GRU 752 alarms (4.92% precision, 18.69% recall); TCN 859 alarms (5.01%, 21.72%); Transformer 1,148 alarms (4.36%, 25.25%); top-two ensemble 730 alarms (5.89%, 21.72%). The report says the ensemble had the strongest balance among those v2 outputs, while GRU had the best mean AP. The 2016–2020 target-specific ensemble APs were 0.1106 (`^GSPC`), 0.0592 (`^DJI`), 0.0248 (`^IXIC`), and 0.0551 (`^RUT`). In 2021 onward, thresholded recall was zero for all four targets; Nasdaq had no positive labels. Thus strong ranking in parts of the test history did not translate into consistently useful alerts.

### 9.3 Existing cross-asset LSTM

`cross_asset_forecasting.py` trains classical, TDA, and combined LSTMs on a six-group extension panel with 30,838 rows across energy, financials, cross-asset ETFs, crypto, international indices, and technology. Its reported mean results are:

| Model | Mean AP | Mean ROC AUC | Mean precision | Mean recall | Mean F1 | Mean Brier |
|---|---:|---:|---:|---:|---:|---:|
| Classical CSD | **0.0268** | **0.5845** | **3.45%** | **16.07%** | **0.0493** | **0.0799** |
| TDA | 0.0201 | 0.5393 | 0.91% | 3.83% | 0.0146 | 0.1370 |
| Combined CSD + TDA | 0.0198 | 0.5380 | 2.21% | 5.35% | 0.0308 | 0.1236 |

Classical fold AP was 0.0150 (2011–2015), 0.0418 (2016–2020), and 0.0236 (2021 onward). TDA/combined feature performance was weaker on this heterogeneous panel. This suggests that pooling asset groups without group-aware adaptation can hurt; it should not be taken as evidence that more asset classes are intrinsically unhelpful.

### 9.4 Existing broader 39-ticker model benchmark and lead-time analysis

The workspace also contains a unified feature/model pipeline and broader benchmark results. `EXPERIMENTS_FULL_REPORT.md`, `EXPERIMENT_RESULTS.md`, `GLM_EXPERIMENT_LOG.md`, `data/benchmark/all_results.csv`, `data/benchmark/all_results_v1.csv`, and `data/benchmark/all_results_v2_partial.csv` hold the detailed run-level records. These are separate from the latest cross-index experiment and should be treated according to their own protocols.

The broader pipeline covers 39 tickers across seven data groups, with a 212,816-row feature/label panel described in the existing consolidated experiment report. Its feature families include CSD and volatility statistics, cross-sectional correlation/eigenvalue summaries, joint and per-series TDA features (including landscape summaries), momentum/price features, and peer summaries. The model zoo includes LSTM, BiLSTM, attention-BiLSTM, GRU, CNN, TCN, Transformer, and logistic-regression sequence baselines.

The completed v1 grid is recorded as 490 of 504 runs. Its mean AP by feature mode was:

| Evaluation level | Classical | TDA | Combined |
|---|---:|---:|---:|
| Group level | **0.0419** | 0.0267 | 0.0379 |
| Pooled ticker level | 0.0417 | 0.0253 | **0.0424** |

The v1 report found classical best on five of six group datasets, while TDA-only led on energy. Logistic regression was close to deep models at group level, which is an important simple-baseline result. The later v2 grid was recorded as partial in `EXPERIMENTS_FULL_REPORT.md` (280/800 cells at the time of that snapshot), with interim group-level AP highest for combined at 0.0525 and landscape at 0.0466. Because the record calls it partial, those interim values must not be represented as a final benchmark. Current `all_results.csv` and log files may contain later runs; consult their run status before updating this report with a final v2 result.

The existing qualitative lead-time analysis reports pre-onset warning behavior across 2,034 label onsets and 39 tickers. It calculates past-only standardized signals and scans up to 60 sessions before each onset. Its overall AUC / mean lead sessions / warning-fire share were: CSD **0.543 / 41 / 0.40**, TDA-index 0.525 / 38 / 0.68, TDA-joint 0.505 / 35 / 0.58, and cross-sectional features 0.478 / 40 / 0.30. This suggests TDA-index signals fire before a larger share of onsets under that specific rule, while CSD ranks the pre-onset rows better. These are exploratory diagnostic statistics; event overlap, the selected threshold, and the definition of a “fire” affect the result. The detailed breakdown is in `data/benchmark/qualitative/leadtime_report.txt`.

## 10. How to read the performance numbers

- **Average precision (AP):** primary ranking metric for rare events. Its no-skill reference is roughly the positive prevalence in the evaluated set, which changes by period.
- **ROC AUC:** probability that a randomly selected positive receives a higher score than a randomly selected negative. It can look strong even when alerts are too sparse or too noisy for practical use.
- **Brier score:** mean squared error between a score and a binary label. It is most interpretable for calibrated probabilities; these model outputs have not been calibrated, so Brier comparisons are secondary.
- **Precision and recall:** computed after choosing a score cutoff on validation. They change with the chosen alert budget and can fall sharply under distribution shift.
- **Alarm rate:** fraction of test rows above that validation cutoff. A 95th-percentile validation cutoff does not force a 5% test alarm rate.
- **Event detection count:** the legacy TDA/CSD diagnostic count is a different measure entirely; it is neither AP nor a confusion-matrix metric.

Positive labels are rare and overlapping in time. Adjacent positive origins can be consequences of one underlying market move, so daily rows are not independent observations. A sound final comparison should add episode-based confidence intervals or block bootstrap intervals and make alert persistence/deduplication explicit.

## 11. Limitations and open risks

1. **Small number of independent crises.** Thousands of daily rows do not provide thousands of independent crash events. Fold-level performance can hinge on a small number of episodes.
2. **One seed in cross-index experiment.** Each fold/mode has one model fit. The observed differences could partly reflect initialization and optimization variability.
3. **Threshold portability.** A second-percentile threshold frozen using 1997–2006 is objective and avoids fitting the cutoff on test outcomes, but market volatility regimes can change. Results depend on this choice and the 15-session horizon.
4. **Calibration and decision utility.** Scores are not calibrated probabilities, and the validation percentile is a simple alert rule. Costs, false-alert burden, missed-event cost, and lead-time utility are not modeled.
5. **Overlapping labels.** Purging uses a 25-calendar-day separation around boundaries to reduce target overlap; sensitivity to other purge lengths should be assessed.
6. **Feature and source limitations.** The current study uses price-derived index features. It does not yet incorporate rates, credit spreads, volatility indices, breadth, futures, or macro releases. Price data can contain source revisions and holidays; archive provenance and exact input snapshots for the final submission.
7. **TDA comparison scope.** TDA features are joint across the four indices in the older LSTM comparison, while classical indicators are calculated per target. That is a useful ablation, but not a perfectly matched representation-size or information-set comparison.
8. **Retrospective qualitative review.** Named-event plots and hand-selected periods are valuable for interpretation but are not independent validation.
9. **Novelty claims.** The peer-only question is a defensible project extension, but cross-market prediction and attention are established research directions. Describe this as an experimental project contribution and testable hypothesis, not as a globally unprecedented architecture.
10. **Interface scope.** The Streamlit tools demonstrate CSV upload and inference locally. They do not include user accounts, production data feeds, deployment, or validated live trading controls.

## 12. Conclusions and recommended next experiments

The current evidence supports a cautious conclusion: **peer-index histories contain useful ranking information in these experiments**, since `other_only` exceeds `self_only` in mean AP and in each of the three temporal blocks. Combining target and peer data yields the best average AP in the current run. The earlier sequence experiment finds its strongest average results when classical and TDA features are combined, while TDA-only forecasts do not beat the classical model on average. The legacy episode diagnostic shows more frequent threshold crossings from rolling variance and GARCH than from TDA L1 under its retrospective rule.

These findings are not yet enough to claim a reliable early-warning system. Before treating them as final scientific conclusions, the highest-value next steps are:

1. Repeat every cross-index fold/mode with multiple seeds and report mean, spread, and paired differences.
2. Add event-level evaluation: collapse positive dates into distinct drawdown episodes, report lead-time distributions and missed episodes, and estimate uncertainty with time-block resampling.
3. Compare against simple baselines (historical event rate, logistic regression, gradient-boosted trees, and volatility-only rules) under the same splits and label protocol.
4. Calibrate scores on validation only and report reliability plots, Brier score, and precision/recall at clearly stated alert budgets.
5. Run sensitivity analysis over crash threshold percentile and forecast horizon, chosen before examining each test block.
6. Test ablations for each peer index and feature family to identify which external markets add information.
7. Archive source data versions, software versions, training configuration, and exact commands for the final report package.

Those experiments would turn the current encouraging cross-index result into a more defensible assessment of whether peer-market signals help, when they help, and whether that help arrives early enough to matter.

---

**Primary latest metric files:** `data/forecasting/walk_forward_metrics.csv` and `data/cross_index_forecasting/cross_index_walk_forward_metrics.csv`.  
**Existing objective episode report:** `data/group_a_original_indices/objective_crash_detection_report.txt`.  
**This consolidated report:** `COMPLETE_WORK_REPORT.md`.
