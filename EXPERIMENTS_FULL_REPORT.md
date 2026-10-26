# Complete Experiment Report — every run, every result (as of Oct 25, 2026)

Companion to `GLM_EXPERIMENT_LOG.md` (append-only log). This document is the consolidated snapshot requested after EXP-4. All numbers are mean test metrics across walk-forward folds unless a single fold is named. Metric of record: **average precision (AP)** — predicting crashes within a 15-trading-day horizon, positive rate ≈ 2%.

---

## 0. Environment & infrastructure built

- Discovered two Python environments: `.venv` (ripser/yfinance/arch, NO torch) and `py -3.12` global (torch 2.8.0+cpu, sklearn, pandas; ripser and arch 8.0.0 installed during this session). All prediction work runs in `py -3.12`.
- Built a unified pipeline (`pipeline/` package + entry scripts), 39 tickers × 7 datasets, 212,816-row feature/label panel.
- **Files created:** `pipeline/{__init__,datasets,labels,features,models,benchmark}.py`, `compute_features.py`, `run_benchmark.py`, `analyze_benchmark.py`, `qualitative_leadtime.py`, `augment_panel.py`, `dump_all_results.py`, `GLM_EXPERIMENT_LOG.md`, this file.
- **Bugs fixed along the way:** ticker-level task duplication (1520→800), zero-positive test windows skipped, log-file closing order, landscape vector padding (windows with <3 H1 bars), cross-sectional index offset, `--resume` capability for interrupted runs.

## 1. Objective crash labels (deliverable 1)

- Per-ticker label: forward **15-session minimum simple return ≤ per-ticker 2nd percentile**.
- Threshold estimated **calibration-only**: 1997→2006-12-31 anchor where ≥1,000 fully-observed origins exist (group_a, international, financials, energy), else first 40% of origins (group_b, crypto, tech). No full-sample quantiles (the older `cross_asset_panel.csv` used those — replaced).
- Example: S&P 500 threshold −10.4%, 39 episodes, 2,501 calibration origins.

**Datasets (7):** group_a US indices ^GSPC,^DJI,^IXIC,^RUT (1997–) · group_b ETFs SPY,TLT,GLD,USO,UUP,HYG (2007–) · crypto BTC,ETH,XRP,SOL,BNB (2020–) · international ^N225,^GDAXI,^FTSE,000001.SS,^HSI,^BVSP,^BSESN (1997–) · financials JPM,BAC,GS,C,MS,WFC (1999–) · energy XOM,CVX,COP,SLB,EOG (1997–) · tech AAPL,MSFT,NVDA,GOOGL,META,AMD (2012–).

**Feature blocks (v2 panel, 119 cols):**
CSD/classical: rolling 50/100d lag-1 autocorrelation, variance, skew, excess kurtosis; 20d realized vol; 20d EWMA vol; GARCH(1,1) conditional vol (stride-5 refit on trailing 250d, forward-fill only).
Cross-sectional: mean pairwise correlation + max eigenvalue of the correlation matrix, w50/w100.
TDA joint (per dataset, Gidea–Katz): H1 landscape L1 (sum p²/4), L2 (√(Σp³/12)), persistence entropy, total persistence, finite-bar count + 36-dim discretized landscape vector (3 levels × 12 grid pts) — z-scored 50-day multivariate window.
TDA per-ticker (Takens): same summaries on z-scored 50-day 3-D delay-embedded point cloud (dim 3, delay 1).
Momentum/price ("price"): 1d return, 5/20/60d momentum, 20/60d vol, 20d downside vol, 60/252d drawdowns, vol ratio, 20d skew.
Peer (per ticker, own series excluded): peer return mean/median, breadth, dispersion, mean peer 20d vol, mean 50d pairwise corr among peers.

**Feature modes benchmarked:** `classical`(+cross), `tda` (5 scalars), `landscape` (41 TDA cols), `combined` (classical+tda scalars), `full` (all CSD+cross+TDA), and after EXP-4: `price` (11), `peer` (6), `everything` (all families).

**Models (8):** LSTM(24) · BiLSTM(24) · **attn-BiLSTM** (additive attention pooling) · GRU(24) · CNN1D(2×24 ch) · TCN(3 dilated residual blocks) · Transformer (2 layers, d=32, 4 heads) · Logistic regression on flattened sequences (floor baseline). All: LayerNorm+Dropout head, train-only scaler, pos_weight min(neg/pos,20), AdamW lr 8e-4 wd 1e-3, early stop on val AP (patience 4), batch 512, ≤12 epochs, seq len 20.

**Evaluation protocol:** 3 chronological test folds (2011–15, 2016–20, 2021+) or proportional 70/15/15 for short series (crypto, tech); 25-day purge before val/test; val window 2 years; alarm threshold = val 95th-percentile score (~5% alarm days) — never tuned on test.

## 2. Team-original baselines (before the unified benchmark)

| Source script | Model | AP | AUC |
|---|---|---|---|
| `crash_forecasting.py` (group A pooled, 3 folds) | LSTM classical | 0.0144 | 0.4872 |
| | LSTM TDA | 0.0124 | 0.5235 |
| | LSTM combined | 0.0307 | 0.6174 |
| `cross_asset_forecasting.py` (6-group panel) | LSTM classical | 0.0268 | 0.5845 |
| | LSTM TDA | 0.0201 | 0.5393 |
| | LSTM combined | 0.0198 | 0.5380 |

Per-ticker from the original run: ^GSPC combined 0.0549, ^RUT combined 0.0702, ^DJI 0.0240, ^IXIC 0.0170. TDA-only model emitted **zero alarms** below the threshold in all folds (precision/recall = 0).

## 3. EXP-1 — v1 unified grid (COMPLETE, 490/504 runs; 7 models × 3 modes × 2 levels; 3,494 result rows in `all_results_v1.csv`)

Mode × level (mean AP / mean AUC):

| level | classical | tda | combined |
|---|---|---|---|
| group | **0.0419** / 0.574 | 0.0267 / 0.524 | 0.0379 / 0.543 |
| ticker (pooled 39 tickers) | 0.0417 / 0.653 | 0.0253 / 0.518 | **0.0424** / 0.649 |

Model ranking (mean AP ± std across all cells): bilstm **0.0413**±.037 · gru .0393±.036 · lstm .0362±.027 · logreg .0349±.029 · cnn .0328±.02 · tcn .0323±.02 · transformer .0316±.03.
(Ticker-level only: transformer .0384 top, all 7 models in .033–.038 — tight.)

Group level, dataset × mode (AP):

| dataset | classical | combined | tda |
|---|---|---|---|
| energy | .0489 | .0522 | **.0552** |
| financials | **.0547** | .0460 | .0262 |
| tech | **.0708** | .0591 | .0431 |
| international | .0300 | .0287 | .0173 |
| group_b | .0330 | .0294 | .0110 |
| group_a | .0239 | .0192 | .0131 |

Findings: classical best on 5/6 datasets; TDA-only best on energy; transformer strong pooled but worst on group-level small datasets; LogReg within 0.006 of deep models on group level.

## 4. EXP-2 — v2 grid (RUNNING at ~319/800 runs; 5 modes incl. landscape/full, 8 models incl. attn-BiLSTM; v2 features: GARCH, total persistence, landscape vectors, cross-corr)

Interim aggregate (280/800 overall cells done: energy + financials complete, group_a partial):

| level | mode | mean AP (interim) |
|---|---|---|
| group | classical | 0.0402 |
| group | combined | **0.0525** |
| group | full | 0.0467 |
| group | landscape | 0.0466 |
| group | tda | 0.0370 |

Interim model ranking: transformer .0498 · cnn .0481 · attn_bilstm .0475 · tcn .0459 · lstm .0429 · gru .0421 · bilstm .0401 · logreg .0352.

Interim dataset × mode (AP): energy: landscape **.0694** > full .061 > combined .0571 · financials: combined .0478 > classical .0467 (TDA variants ~.02-.03) · group_a (so far): classical .0236, tda .0152 (TDA/landscape/full shows NaN for the partial rows - fold coverage incomplete).

Early v2 signal: `combined` and TDA-rich modes improved vs v1 (combined .038→.053); the **landscape-vector mode is now competitive** (best on energy) — richer topological summaries help the deep models.

## 5. EXP-3 — lead-time / early-warning robustness (COMPLETE; `data/benchmark/qualitative/`)

2,034 label onsets across 39 tickers; per-family past-only z-scores (250d); pre-onset window 30 sessions; lead = first pre-onset crossing of the family's quiet-day 90th percentile (scan −60..−1).

Dataset × family (AUC / mean lead-sessions / fire-share):

| dataset | CSD | TDA-index | TDA-joint | CROSS |
|---|---|---|---|---|
| crypto | .461 / 27 / .23 | .569 / 37 / .83 | .575 / 26 / .57 | .414 / 24 / .27 |
| energy | .529 / 42 / .35 | .500 / 37 / .64 | .482 / 37 / .60 | .535 / 44 / .27 |
| financials | .543 / 43 / .42 | .492 / 40 / .67 | .492 / 39 / .59 | .489 / 47 / .31 |
| group_a | .591 / 44 / .43 | .572 / 36 / .79 | .505 / 49 / .53 | .476 / 49 / .23 |
| group_b | .660 / 48 / .51 | .479 / 38 / .67 | .488 / 26 / .55 | .549 / 42 / .47 |
| international | .502 / 38 / .42 | .522 / 40 / .61 | .497 / 34 / .60 | .462 / 22 / .30 |
| tech | .520 / 40 / .40 | .558 / 39 / .63 | .508 / 40 / .57 | .424 / 53 / .19 |

Overall (across datasets): CSD .543/.41 lead_mean, fire .40 · TDA-index .525/.38, fire .68 · TDA-joint .505/.35, fire .58 · CROSS .478/.40, fire .30.

Findings: (1) CSD discriminates harder pre-onset on the four-index/two-ETF groups; (2) TDA-index **fires early before 58–83% of onsets** vs 23–51% for CSD — the "earlier warning" edge belongs to topology on consistency, CSD on strength; (3) crypto is a TDA domain (CSD below noise, AUC .46; TDA-joint .58); (4) only 1 of 4 landmark events matched cleanly for aligned figures (tolerance too tight) — flagged for improvement.

## 6. EXP-4 — the two independent agents' approaches (COMPLETE, both on the identical group-A protocol)

Agent A — temporal cross-attention across the 4 US indices (`cross_index_forecasting.py`, seq 30, momentum/drawdown/vol-shock features; own labels impl. same calibration rule):

| model | AP | AUC | alarm rate |
|---|---|---|---|
| all_indices | **0.0436** | 0.7637 | 5.5% |
| other_only (peer-only) | 0.0366 | 0.7538 | 4.5% |
| self_only | 0.0285 | 0.7179 | 4.5% |

Per-ticker (all_indices): ^GSPC .0676 · ^IXIC .0523 · ^RUT .0457 · ^DJI .0430.

Agent B — peer-index deep models (`peer_index_forecasting.py` pooled; `..._v2.py` per-target + val-selected top-2 ensemble; target's returns excluded from its features):

| model | v1 pooled AP | v2 per-target AP |
|---|---|---|
| GRU | 0.0436 | **0.0508** |
| ensemble_top2 | — | 0.0414 |
| TCN | — | 0.0401 |
| attn_bilstm | — | 0.0400 |
| transformer | 0.0395 | 0.0362 |

Per-target ensemble across folds (AP): ^DJI .016/.059/.082 · ^GSPC .010/**.111**/.004 · ^IXIC —/.025/— · ^RUT .019/.055/.034 (2011-15 / 2016-20 / 2021+). Fold-2011-15 is essentially signal-free for every approach; 2016-20 carries most signal.

**Cross-family comparison on group A (same labels & folds):**

| Feature family / model family | best AP |
|---|---|
| CSD classical (mine, v1) | 0.024 |
| TDA-only (mine) | 0.015 |
| peer/momentum + cross-attention (Agent A) | **0.0436** |
| peer/momentum GRU (Agent B v2) | **0.0508** |
| team-original combined LSTM | 0.031 |

Verdict: the peer/momentum feature family ≈ 2× stronger than CSD-only or TDA-only on US indices; my TDA/CSD modes are behind. Integration: both ideas absorbed into the unified panel (`augment_panel.py`, +17 cols) and new modes `price`, `peer`, `everything` will test them across all 39 tickers (agent runs only covered 4 indices).

## 7. Not yet done / queued

- Main v2 grid (~480 remaining runs) → then supplement run (modes price/peer/everything) → consolidated final ranking incl. `everything`.
- `analyze_benchmark.py` final tables + figures; event-figure tolerance loosening in `qualitative_leadtime.py`.
- Host-interface (`forecast_app.py`, `cross_index_app.py`) refresh with new models/modes.
- README + final report refresh with the v2 numbers.
