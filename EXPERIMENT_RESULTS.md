# Crash Forecasting Experiments and Results

This document consolidates the crash-forecasting experiments performed in this
repository. It is intended as a decision document for choosing the next
research direction.

All experiments use chronological evaluation rather than random train/test
splits. Existing experiments were preserved; the newer experiments were added
as separate scripts and output files.

## Executive summary

The strongest current result is the **target-specific peer-index GRU** from the
v2 experiment:

- Mean average precision: **0.0508**
- Mean ROC AUC: **0.7022**
- Approximate pooled precision: **4.92%**
- Approximate pooled recall: **18.69%**

The strongest operational precision/recall balance came from the v2
validation-selected top-two ensemble:

- Approximate pooled precision: **5.89%**
- Approximate pooled recall: **21.72%**
- Approximate pooled accuracy: **94.67%**

The earlier pooled peer GRU achieved a higher approximate pooled recall:

- Approximate pooled precision: **6.90%**
- Approximate pooled recall: **29.29%**
- Mean average precision: **0.0357**
- Mean ROC AUC: **0.6775**

The results are encouraging but regime-dependent. The models were much stronger
in the 2016-2020 test block than in 2011-2015 or 2021-2026. They should
currently be treated as ranking and research models, not as reliable trading
signals.

## Important metric interpretation

Crash labels are rare. The original and peer-index test sets contain roughly
1-2% positive observations. Therefore:

- Accuracy is not a useful primary metric.
- A model that always predicts "no crash" can have very high accuracy.
- Average precision is more informative than accuracy for this imbalanced task.
- ROC AUC measures ranking quality across thresholds.
- Precision measures how many warnings were correct.
- Recall measures how many crash-label observations were detected.
- F1 summarizes precision and recall at one selected threshold.
- Brier score measures squared probability error; lower is better.

The warning threshold in these experiments is selected from validation scores,
usually as the 95th percentile. The test period is not used to choose the
threshold.

## Evaluation protocol

The common protocol is:

1. Build features using data available on or before the forecast date.
2. Define the target as whether the target index reaches its ticker-specific
   2nd-percentile minimum return during the next 15 trading sessions.
3. Use chronological train, validation, and test periods.
4. Purge approximately 25 calendar days before validation and test periods.
   This protects against overlapping 15-session forward targets.
5. Fit scalers using training data only.
6. Select model checkpoints using validation average precision.
7. Select operational warning thresholds using validation scores.
8. Report untouched chronological test results.

The main test blocks are:

- `2011-2015`
- `2016-2020`
- `2021-2026`

## Experiment inventory

| Experiment | Script | Main output |
|---|---|---|
| Original four-index CSD/TDA forecast | [`crash_forecasting.py`](./crash_forecasting.py) | [`walk_forward_metrics.csv`](./data/forecasting/walk_forward_metrics.csv) |
| Full cross-asset CSD/TDA forecast | [`cross_asset_forecasting.py`](./cross_asset_forecasting.py) | [`cross_asset_walk_forward_metrics.csv`](./data/forecasting/cross_asset_walk_forward_metrics.csv) |
| Pooled peer-index GRU/Transformer | [`peer_index_forecasting.py`](./peer_index_forecasting.py) | [`peer_index_deep_learning_metrics.csv`](./data/forecasting/peer_index_deep_learning_metrics.csv) |
| Target-specific peer deep learning v2 | [`peer_index_forecasting_v2.py`](./peer_index_forecasting_v2.py) | [`peer_index_deep_learning_v2_metrics.csv`](./data/forecasting/peer_index_deep_learning_v2_metrics.csv) |
| Existing broader model zoo | [`pipeline/benchmark.py`](./pipeline/benchmark.py) and [`pipeline/models.py`](./pipeline/models.py) | [`all_results.csv`](./data/benchmark/all_results.csv) |

The model artifacts and experiment reports are under
[`data/forecasting/`](./data/forecasting/).

## 1. Original four-index forecast

### Purpose

This was the original benchmark comparing:

- Classical CSD features
- TDA features
- Combined CSD + TDA features

It uses:

- S&P 500: `^GSPC`
- Dow Jones: `^DJI`
- Nasdaq: `^IXIC`
- Russell 2000: `^RUT`

The features include rolling autocorrelation, variance, skewness, realized
volatility, and joint TDA landscape features.

### Mean results across test blocks

| Model | Average precision | ROC AUC | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Classical CSD | 0.0144 | 0.4872 | 1.61% | 12.30% | 0.0244 | 0.0879 |
| TDA | 0.0124 | 0.5235 | 0.00% | 0.00% | 0.0000 | 0.0739 |
| **Combined CSD + TDA** | **0.0307** | **0.6174** | **3.42%** | **18.28%** | **0.0567** | **0.0745** |

### Per-period combined model

| Test block | Average precision | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| 2011-2015 | 0.0073 | 0.3024 | 1.63% | 4.17% | 0.0234 |
| 2016-2020 | 0.0340 | 0.7030 | 2.14% | 18.18% | 0.0383 |
| 2021-2026 | **0.0508** | **0.8470** | **6.50%** | **32.50%** | **0.1083** |

### Interpretation

The combined model was clearly better than CSD-only and TDA-only in the
original four-index benchmark. TDA alone generated no successful warnings at
the selected operating thresholds, but adding TDA to CSD improved the combined
model, especially in the latest test period.

## 2. Full cross-asset CSD/TDA forecast

### Purpose

This experiment used the expanded extension panel rather than only the four
original indices. It contains:

- 30,838 rows
- 6 market groups
- Energy
- Financials
- Technology
- Cross-asset ETFs
- Crypto
- International indices

The groups and row counts were:

| Group | Rows |
|---|---:|
| Energy | 7,426 |
| Financials | 6,838 |
| Cross-asset ETFs | 4,843 |
| Crypto | 2,308 |
| International | 5,868 |
| Technology | 3,555 |

### Mean results across test blocks

| Model | Average precision | ROC AUC | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| **Classical CSD** | **0.0268** | **0.5845** | **3.45%** | **16.07%** | **0.0493** | **0.0799** |
| TDA | 0.0201 | 0.5393 | 0.91% | 3.83% | 0.0146 | 0.1370 |
| Combined CSD + TDA | 0.0198 | 0.5380 | 2.21% | 5.35% | 0.0308 | 0.1236 |

### Per-period classical CSD model

| Test block | Average precision | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| 2011-2015 | 0.0150 | 0.5360 | 2.27% | 1.43% | 0.0175 |
| 2016-2020 | **0.0418** | 0.6149 | **4.74%** | **39.63%** | **0.0847** |
| 2021-2026 | 0.0236 | 0.6027 | 3.35% | 7.14% | 0.0456 |

### Interpretation

Adding more rows and more asset classes improved the expanded **CSD-only**
model compared with the original CSD-only model:

- Average precision: `0.0144 -> 0.0268`
- ROC AUC: `0.4872 -> 0.5845`
- Recall: `12.30% -> 16.07%`

However, adding heterogeneous groups made the combined TDA+CSD model worse than
the original four-index combined model. This suggests that the asset groups
have different regimes, volatility scales, and crash dynamics. A single global
model does not automatically benefit from more data.

## 3. Pooled peer-index deep learning

### Purpose

This was the first experiment aimed at learning a new signal beyond directly
feeding CSD or TDA features.

For every target index, the target's own returns were excluded. The model used
only the other three indices:

- Peer returns
- Peer rolling volatility
- Mean peer return
- Peer dispersion
- Mean peer volatility
- Rolling peer correlation

The sequence length was 30 sessions. The models were:

- GRU
- Transformer

### Mean results across test blocks

| Model | Average precision | ROC AUC | Mean precision | Mean recall | Mean F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| **Peer GRU** | **0.0357** | **0.6775** | 2.35% | 17.58% | 0.0415 | **0.0661** |
| Peer Transformer | 0.0287 | 0.5642 | 3.94% | **19.77%** | **0.0603** | 0.0781 |

### Approximate pooled statistics

| Model | Observations | Crash labels | Warnings | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Peer GRU | 15,804 | 198 | 840 | 94.17% | **6.90%** | **29.29%** |
| Peer Transformer | 15,804 | 198 | 1,253 | 91.49% | 4.23% | 26.77% |

### Per-period results

#### Peer GRU

| Test block | Average precision | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| 2011-2015 | 0.0110 | 0.5846 | 0.00% | 0.00% | 0.0000 |
| 2016-2020 | **0.0836** | 0.7120 | 7.06% | **52.73%** | 0.1246 |
| 2021-2026 | 0.0125 | **0.7359** | 0.00% | 0.00% | 0.0000 |

#### Peer Transformer

| Test block | Average precision | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| 2011-2015 | 0.0081 | 0.2251 | 0.00% | 0.00% | 0.0000 |
| 2016-2020 | 0.0394 | 0.6641 | 3.96% | 41.82% | 0.0724 |
| 2021-2026 | **0.0386** | **0.8035** | **7.87%** | 17.50% | **0.1085** |

### Interpretation

This experiment showed that the other indices contain predictive information.
The peer GRU outperformed the earlier CSD/TDA expanded combined model in
average precision and ROC AUC:

- Average precision: `0.0198 -> 0.0357`
- ROC AUC: `0.5380 -> 0.6775`

The signal is not stable across time. It is strong in 2016-2020 but weak at
the selected warning threshold in the other two periods.

## 4. Target-specific peer deep learning v2

### Purpose

The v2 experiment attempted to improve the pooled peer model by:

1. Training a separate model for each target index.
2. Keeping the target index excluded from the input.
3. Adding richer peer-only features:
   - Current peer returns
   - Lagged returns
   - Five-session momentum
   - Rolling volatility
   - Volatility shocks
   - Peer drawdowns
   - Peer mean and median returns
   - Negative-return breadth
   - Peer dispersion
   - Average peer volatility
   - Rolling peer correlation
4. Comparing four architectures:
   - GRU
   - Attention-BiLSTM
   - TCN
   - Transformer
5. Selecting the top two architectures by validation average precision and
   averaging their probabilities.

### Mean results

| Model | Mean average precision | Mean ROC AUC | Mean precision | Mean recall | Mean F1 |
|---|---:|---:|---:|---:|---:|
| **Target-specific GRU** | **0.0508** | 0.7022 | 2.97% | 11.12% | 0.0447 |
| Attention-BiLSTM | 0.0400 | **0.7068** | 1.77% | 7.51% | 0.0274 |
| TCN | 0.0401 | 0.6981 | 2.30% | **12.02%** | 0.0380 |
| Transformer | 0.0362 | 0.6836 | 2.14% | **15.41%** | 0.0351 |
| Top-two ensemble | 0.0414 | 0.6949 | 2.07% | 11.40% | 0.0349 |

### Approximate pooled statistics

| Model | Observations | Crash labels | Warnings | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Target-specific GRU | 15,804 | 198 | 752 | 94.46% | 4.92% | 18.69% |
| TCN | 15,804 | 198 | 859 | 93.86% | 5.01% | 21.72% |
| Transformer | 15,804 | 198 | 1,148 | 92.12% | 4.36% | **25.25%** |
| **Top-two ensemble** | 15,804 | 198 | 730 | **94.67%** | **5.89%** | 21.72% |

### 2016-2020 target-specific ensemble

| Target | Average precision | ROC AUC | Precision | Recall |
|---|---:|---:|---:|---:|
| S&P 500 | **0.1106** | 0.7870 | 11.18% | **48.57%** |
| Dow Jones | 0.0592 | 0.7686 | 6.61% | 40.54% |
| Nasdaq | 0.0248 | 0.7972 | 0.99% | 7.69% |
| Russell 2000 | 0.0551 | **0.8099** | 6.06% | 40.00% |

### 2021-2026 target-specific ensemble

| Target | Average precision | ROC AUC | Precision | Recall |
|---|---:|---:|---:|---:|
| S&P 500 | 0.0038 | 0.5152 | 0.00% | 0.00% |
| Dow Jones | **0.0822** | **0.8841** | 0.00% | 0.00% |
| Nasdaq | Not defined; no positive labels | Not defined | 0.00% | 0.00% |
| Russell 2000 | 0.0335 | 0.8485 | 0.00% | 0.00% |

### Interpretation

Target specialization improved ranking quality:

- Pooled peer GRU average precision: `0.0357`
- Target-specific GRU average precision: `0.0508`
- Improvement: approximately **42%**

The v2 models produced better target-specific rankings but did not improve
recall consistently at the fixed validation-derived threshold. This difference
between ranking quality and operational warnings is important.

## Overall comparison

| Model family | Average precision | ROC AUC | Main strength |
|---|---:|---:|---|
| Original combined CSD + TDA | 0.0307 | 0.6174 | Best original TDA/CSD combination |
| Expanded CSD | 0.0268 | 0.5845 | More data helped classical indicators |
| Expanded combined CSD + TDA | 0.0198 | 0.5380 | Heterogeneous groups hurt one global model |
| Pooled peer GRU | 0.0357 | 0.6775 | Strong pooled recall and cross-index signal |
| Target-specific peer GRU | **0.0508** | 0.7022 | Best average precision |
| Target-specific top-two ensemble | 0.0414 | **0.6949** | Best precision/recall balance among v2 outputs |

## What the experiments suggest

### 1. Other-index information is useful

The peer-only models performed better than the expanded CSD/TDA model in
average precision and ROC AUC. This supports the idea that cross-index lead/lag
relationships contain predictive information.

### 2. More data is not automatically better

The expanded cross-asset panel increased the number of rows but mixed markets
with very different behavior. Without group normalization, embeddings,
hierarchical modeling, or separate heads, the extra data hurt the combined
TDA/CSD model.

### 3. TDA and deep learning are not the same

TDA explicitly calculates topological summaries such as persistence
landscapes and persistence entropy. Deep learning learns nonlinear temporal
representations. A neural network may learn geometry-like regime patterns, but
it does not automatically perform persistent homology.

The peer-index experiments are therefore a genuinely different signal source:
they learn cross-index temporal dependencies directly.

### 4. Market regime matters

Most models performed best in 2016-2020. Performance was weaker in 2011-2015
and operational threshold recall was often weak in 2021-2026. This indicates
non-stationarity rather than a single stable crash mechanism.

### 5. Ranking and warning generation are different problems

Some models had high ROC AUC in the latest period but produced few or no true
warnings at the selected threshold. The next stage should separately optimize:

- Ranking crash risk
- Choosing alert thresholds
- Grouping daily warnings into crash episodes
- Measuring lead time before an episode

## Limitations and cautions

1. **Rare labels:** A 2nd-percentile crash definition creates very few positive
   observations.
2. **Episode dependence:** Multiple consecutive positive days can represent one
   crash episode, so daily metrics are not independent.
3. **Threshold sensitivity:** Precision and recall change substantially with
   the operational threshold.
4. **Regime shift:** The models do not perform consistently across historical
   periods.
5. **Data revisions:** Market-data providers can revise or replace historical
   observations.
6. **Approximate pooled confusion statistics:** The pooled TP/FP/TN/FN values
   are reconstructed from saved rounded precision and alarm counts. The
   per-fold metrics in the CSV files are authoritative.
7. **No trading backtest:** These are forecast classification experiments, not
   tested investment strategies.
8. **No claim of causality:** Cross-index predictive power does not prove that
   one index causes another index to crash.

## Suggested next experiments

### Highest priority: event-level evaluation

Convert consecutive positive labels into crash episodes and evaluate:

- Was there at least one warning before the episode?
- How many trading days of lead time were available?
- How many warnings occurred per episode?
- How many false-warning episodes occurred?
- What was the maximum warning score before each episode?

This is likely more meaningful than treating every positive day independently.

### Target-specific threshold calibration

The current validation 95th-percentile threshold can be too conservative or
too aggressive for a particular target. Compare:

- Fixed alert-rate thresholds
- Per-target thresholds
- Precision-constrained thresholds
- Recall-constrained thresholds
- Thresholds selected by episode-level utility

Threshold selection must remain validation-only.

### Regime-aware modeling

Add regime variables or separate experts for:

- Volatility regime
- Correlation regime
- Bull versus bear trend
- Liquidity or drawdown regime

A mixture-of-experts model could select different peer relationships in
different regimes.

### Cross-asset target-specific models

Apply the peer-only architecture to the larger cross-asset panel, but use:

- Market-group embeddings
- Per-group normalization
- Separate output heads
- Group-aware train/validation splits
- Shared representation plus target-specific heads

This is safer than pooling all crypto, commodity, equity, and international
series into one undifferentiated model.

### Multitask learning

Train one shared network with one output head per target index. Compare it
against fully separate models. This can share common market-wide information
while preserving target-specific behavior.

### Calibrated probability modeling

The model scores are not calibrated probabilities. Test validation-only:

- Isotonic calibration
- Platt scaling
- Beta calibration

Then evaluate calibration curves, expected calibration error, and Brier score
on the held-out test blocks.

### Robustness and ablations

Run controlled ablations for:

- Returns only
- Returns plus volatility
- Returns plus correlation
- Momentum and drawdown only
- No breadth
- No correlation
- Different sequence lengths
- Different label quantiles
- Different forecast horizons

This will show whether the performance comes from peer relationships or from a
small number of engineered volatility variables.

## Reproducibility commands

From the repository root:

```text
py -3.12 crash_forecasting.py
py -3.12 run_cross_asset_extension.py
py -3.12 cross_asset_forecasting.py
py -3.12 peer_index_forecasting.py
py -3.12 peer_index_forecasting_v2.py
```

The existing broader model zoo can be run through its own benchmark workflow;
it was not deleted or replaced by the newer peer-index experiments.

