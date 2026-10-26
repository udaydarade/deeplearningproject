# MASTER DATA AND METHODS BIBLE (COMPACT)

This compact reference is generated from files currently present in the repository. Large raw tables remain in CSV/Parquet form and are linked below; they are not duplicated into a hundreds-of-megabytes Markdown file.

## 1. Environment

- OS: Windows-11-10.0.26200-SP0
- Python: 3.12.0
- Processor: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel
- Logical CPUs: 20
- GPU probe: none reported (nvidia-smi unavailable)

| Package | Version |
|---|---|
| torch | 2.8.0 |
| ripser | 0.6.15 |
| arch | 8.0.0 |
| lightgbm | 4.7.0 |
| scikit-learn | 1.9.1 |
| pandas | 2.3.3 |
| numpy | 2.4.6 |
| streamlit | 1.66.1 |

## 2. Codebase inventory

Every Python file is listed with line count. Runtime is experiment-specific and is recorded in run logs/status files where available.

| script                                  |   lines |
|:----------------------------------------|--------:|
| analyze_benchmark.py                    |     181 |
| augment_panel.py                        |      83 |
| build_compact_master_bible.py           |     253 |
| build_comparative_performance_report.py |     191 |
| build_crash_model_ledger.py             |     188 |
| build_master_data_methods_bible.py      |     299 |
| build_precrash_model_ledger.py          |     242 |
| build_run_record.py                     |     171 |
| compute_features.py                     |     144 |
| crash_forecasting.py                    |     254 |
| crash_labels.py                         |     435 |
| cross_asset_forecasting.py              |     205 |
| cross_index_app.py                      |     115 |
| cross_index_forecasting.py              |     283 |
| csd_baseline.py                         |     119 |
| download_market_data.py                 |     193 |
| dump_all_results.py                     |     108 |
| episode_extension.py                    |     266 |
| final_analysis.py                       |      43 |
| forecast_app.py                         |     177 |
| forecasting_qualitative.py              |      70 |
| generate_final_assets.py                |     335 |
| ismail_2022_reproduction.py             |     218 |
| paper_exact_detection.py                |     150 |
| paper_extension.py                      |     199 |
| paper_reproduction_figures.py           |     164 |
| paper_style_figures.py                  |     134 |
| peer_index_forecasting.py               |     238 |
| peer_index_forecasting_v2.py            |     233 |
| pipeline/__init__.py                    |      14 |
| pipeline/benchmark.py                   |     352 |
| pipeline/datasets.py                    |      43 |
| pipeline/features.py                    |     319 |
| pipeline/labels.py                      |      80 |
| pipeline/models.py                      |     242 |
| pipeline_v3/__init__.py                 |       0 |
| pipeline_v3/eval_events.py              |     167 |
| pipeline_v3/operating_rules.py          |     122 |
| pipeline_v3/test_eval_events.py         |      89 |
| pipeline_v5/__init__.py                 |       1 |
| pipeline_v5/cluster.py                  |      46 |
| pipeline_v5/dynamics.py                 |      38 |
| pipeline_v5/price1d.py                  |      73 |
| pipeline_v5/sequence.py                 |      84 |
| pipeline_v5/topology.py                 |     145 |
| pipeline_v6/labels.py                   |      66 |
| pipeline_v6/v5_eval_copy.py             |     217 |
| presentation_figures.py                 |     199 |
| qualitative_leadtime.py                 |     227 |
| run_benchmark.py                        |      58 |
| run_cross_asset_extension.py            |     255 |
| tda_replication.py                      |     152 |
| two_period_paper_figures.py             |     147 |
| v3_1_blocks.py                          |      43 |
| v3_1_bootstrap.py                       |      75 |
| v3_1_eval.py                            |     139 |
| v3_1_final.py                           |     104 |
| v3_1_inventory.py                       |      36 |
| v3_1_log_step.py                        |      32 |
| v3_1_more.py                            |     144 |
| v3_1_pack_scores.py                     |      19 |
| v3_1_run_steps.py                       |     149 |
| v3_1_select_classical.py                |      19 |
| v3_1_summarize.py                       |     130 |
| v3_1_test_eval.py                       |      48 |
| v3_1_train.py                           |     188 |
| v3_build_features.py                    |     515 |
| v3_download_data.py                     |     210 |
| v3_download_stocks.py                   |      73 |
| v3_finalize_features.py                 |      50 |
| v3_leak_test.py                         |      97 |
| v3_prepare_data.py                      |     277 |
| v3_run_exp_d.py                         |      99 |
| v3_run_experiments.py                   |     445 |
| v4_1_audit.py                           |     128 |
| v4_2_test_eval.py                       |      45 |
| v4_3_build_raw.py                       |     162 |
| v4_4_build_dyn.py                       |      70 |
| v4_5_price1d.py                         |     140 |
| v4_6_checks.py                          |      64 |
| v4_7_fair_models.py                     |     120 |
| v4_eval.py                              |     106 |
| v5_0_inventory.py                       |      89 |
| v5_10_e4_labels.py                      |     140 |
| v5_11_episode_table.py                  |      89 |
| v5_12_takens_vol.py                     |      69 |
| v5_1_diagnose_offbudget.py              |      43 |
| v5_1_e1.py                              |     197 |
| v5_2_build_tda.py                       |     158 |
| v5_2_enforce_universe.py                |      56 |
| v5_3_build_cluster.py                   |      72 |
| v5_4_build_price1d.py                   |      60 |
| v5_5_build_dynamics.py                  |     149 |
| v5_6_diagnostics.py                     |     105 |
| v5_6_quality_gates.py                   |     100 |
| v5_7_summarize_e1.py                    |      72 |
| v5_8_e3_deep.py                         |     203 |
| v5_9_e2.py                              |     213 |
| v5_9_summarize_e2.py                    |     112 |
| v5_eval.py                              |     217 |
| v5_test_cluster.py                      |      25 |
| v5_test_dynamics.py                     |      29 |
| v5_test_eval.py                         |      78 |
| v5_test_price1d.py                      |      67 |
| v5_test_sequence.py                     |      30 |
| v5_test_topology.py                     |      31 |
| v6_0_inventory.py                       |      56 |
| v6_10_transfer.py                       |      61 |
| v6_11_current_status.py                 |      22 |
| v6_12_label_audit.py                    |      54 |
| v6_1_labels.py                          |      90 |
| v6_2_semivar.py                         |      61 |
| v6_3_eval_tests.py                      |      53 |
| v6_4_overlap.py                         |      44 |
| v6_5_sweep.py                           |      85 |
| v6_6_realtime.py                        |      89 |
| v6_8_finalize.py                        |      35 |
| v6_9_analysis.py                        |      38 |
| v7_crash_statistics.py                  |     219 |
| v7_final_synthesis.py                   |     382 |

## 3. Dataset and event population

### V6 asset classes

| asset_class   |   assets |
|:--------------|---------:|
| bond          |       11 |
| commodity     |       19 |
| country_etf   |       29 |
| crypto        |       11 |
| equity_index  |       32 |
| fx            |       14 |
| stock         |       17 |
| us_etf        |       26 |

### V6 label prevalence

|   N |   p | direction   |       n |   positive |   prevalence |
|----:|----:|:------------|--------:|-----------:|-------------:|
|   3 |   2 | down        | 1047504 |      19199 |    0.0183283 |
|   3 |   2 | up          | 1047504 |      18719 |    0.0178701 |
|   3 |   2 | move        | 1047504 |      37729 |    0.036018  |
|   3 |   3 | down        | 1047504 |      28302 |    0.0270185 |
|   3 |   3 | up          | 1047504 |      27813 |    0.0265517 |
|   3 |   3 | move        | 1047504 |      55788 |    0.053258  |
|   3 |   4 | down        | 1047504 |      37500 |    0.0357994 |
|   3 |   4 | up          | 1047504 |      36921 |    0.0352466 |
|   3 |   4 | move        | 1047504 |      73939 |    0.0705859 |
|   3 |   5 | down        | 1047504 |      46714 |    0.0445955 |
|   3 |   5 | up          | 1047504 |      46167 |    0.0440733 |
|   3 |   5 | move        | 1047504 |      92267 |    0.0880827 |
|   3 |   6 | down        | 1047504 |      56021 |    0.0534805 |
|   3 |   6 | up          | 1047504 |      55393 |    0.0528809 |
|   3 |   6 | move        | 1047504 |     110623 |    0.105606  |
|   3 |   8 | down        | 1047504 |      75057 |    0.0716532 |
|   3 |   8 | up          | 1047504 |      73999 |    0.0706432 |
|   3 |   8 | move        | 1047504 |     147902 |    0.141195  |
|   3 |  10 | down        | 1047504 |      94181 |    0.0899099 |
|   3 |  10 | up          | 1047504 |      92979 |    0.0887624 |
|   3 |  10 | move        | 1047504 |     185525 |    0.177111  |
|   5 |   2 | down        | 1046329 |      19568 |    0.0187016 |
|   5 |   2 | up          | 1046329 |      18698 |    0.0178701 |
|   5 |   2 | move        | 1046329 |      38019 |    0.0363356 |
|   5 |   3 | down        | 1046329 |      28605 |    0.0273384 |
|   5 |   3 | up          | 1046329 |      27770 |    0.0265404 |
|   5 |   3 | move        | 1046329 |      55952 |    0.0534746 |
|   5 |   4 | down        | 1046329 |      37599 |    0.0359342 |
|   5 |   4 | up          | 1046329 |      36864 |    0.0352317 |
|   5 |   4 | move        | 1046329 |      73838 |    0.0705686 |
|   5 |   5 | down        | 1046329 |      46807 |    0.0447345 |
|   5 |   5 | up          | 1046329 |      45684 |    0.0436612 |
|   5 |   5 | move        | 1046329 |      91672 |    0.087613  |
|   5 |   6 | down        | 1046329 |      55996 |    0.0535166 |
|   5 |   6 | up          | 1046329 |      54859 |    0.05243   |
|   5 |   6 | move        | 1046329 |     109850 |    0.104986  |
|   5 |   8 | down        | 1046329 |      74678 |    0.0713714 |
|   5 |   8 | up          | 1046329 |      73200 |    0.0699589 |
|   5 |   8 | move        | 1046329 |     146315 |    0.139837  |
|   5 |  10 | down        | 1046329 |      93456 |    0.089318  |
|   5 |  10 | up          | 1046329 |      91857 |    0.0877898 |
|   5 |  10 | move        | 1046329 |     183124 |    0.175016  |
|  10 |   2 | down        | 1043448 |      20049 |    0.0192142 |
|  10 |   2 | up          | 1043448 |      18928 |    0.0181399 |
|  10 |   2 | move        | 1043448 |      38683 |    0.0370723 |
|  10 |   3 | down        | 1043448 |      29049 |    0.0278394 |
|  10 |   3 | up          | 1043448 |      27633 |    0.0264824 |
|  10 |   3 | move        | 1043448 |      56150 |    0.053812  |
|  10 |   4 | down        | 1043448 |      37954 |    0.0363736 |
|  10 |   4 | up          | 1043448 |      36639 |    0.0351134 |
|  10 |   4 | move        | 1043448 |      73838 |    0.0707635 |
|  10 |   5 | down        | 1043448 |      47057 |    0.0450976 |
|  10 |   5 | up          | 1043448 |      45574 |    0.0436764 |
|  10 |   5 | move        | 1043448 |      91641 |    0.0878252 |
|  10 |   6 | down        | 1043448 |      56305 |    0.0539605 |
|  10 |   6 | up          | 1043448 |      54539 |    0.0522681 |
|  10 |   6 | move        | 1043448 |     109586 |    0.105023  |
|  10 |   8 | down        | 1043448 |      74710 |    0.0715992 |
|  10 |   8 | up          | 1043448 |      72514 |    0.0694946 |
|  10 |   8 | move        | 1043448 |     145337 |    0.139285  |
|  10 |  10 | down        | 1043448 |      93288 |    0.0894036 |
|  10 |  10 | up          | 1043448 |      90986 |    0.0871974 |
|  10 |  10 | move        | 1043448 |     181703 |    0.174137  |
|  15 |   2 | down        | 1040578 |      20280 |    0.0194892 |
|  15 |   2 | up          | 1040578 |      19241 |    0.0184907 |
|  15 |   2 | move        | 1040578 |      39242 |    0.0377117 |
|  15 |   3 | down        | 1040578 |      29197 |    0.0280584 |
|  15 |   3 | up          | 1040578 |      28131 |    0.027034  |
|  15 |   3 | move        | 1040578 |      56845 |    0.0546283 |
|  15 |   4 | down        | 1040578 |      38332 |    0.0368372 |
|  15 |   4 | up          | 1040578 |      36904 |    0.0354649 |
|  15 |   4 | move        | 1040578 |      74541 |    0.0716342 |
|  15 |   5 | down        | 1040578 |      47354 |    0.0455074 |
|  15 |   5 | up          | 1040578 |      45551 |    0.0437747 |
|  15 |   5 | move        | 1040578 |      91994 |    0.0884066 |
|  15 |   6 | down        | 1040578 |      56337 |    0.0541401 |
|  15 |   6 | up          | 1040578 |      54241 |    0.0521258 |
|  15 |   6 | move        | 1040578 |     109418 |    0.105151  |
|  15 |   8 | down        | 1040578 |      74591 |    0.0716823 |
|  15 |   8 | up          | 1040578 |      71960 |    0.0691539 |
|  15 |   8 | move        | 1040578 |     144785 |    0.139139  |
|  15 |  10 | down        | 1040578 |      92958 |    0.089333  |
|  15 |  10 | up          | 1040578 |      90042 |    0.0865308 |
|  15 |  10 | move        | 1040578 |     180500 |    0.173461  |
|  30 |   2 | down        | 1032093 |      20462 |    0.0198257 |
|  30 |   2 | up          | 1032093 |      20186 |    0.0195583 |
|  30 |   2 | move        | 1032093 |      40461 |    0.0392029 |
|  30 |   3 | down        | 1032093 |      28995 |    0.0280934 |
|  30 |   3 | up          | 1032093 |      28581 |    0.0276923 |
|  30 |   3 | move        | 1032093 |      57237 |    0.0554572 |
|  30 |   4 | down        | 1032093 |      37945 |    0.0367651 |
|  30 |   4 | up          | 1032093 |      37164 |    0.0360084 |
|  30 |   4 | move        | 1032093 |      74562 |    0.0722435 |
|  30 |   5 | down        | 1032093 |      47022 |    0.0455598 |
|  30 |   5 | up          | 1032093 |      45782 |    0.0443584 |
|  30 |   5 | move        | 1032093 |      92090 |    0.0892265 |
|  30 |   6 | down        | 1032093 |      56106 |    0.0543614 |
|  30 |   6 | up          | 1032093 |      53975 |    0.0522966 |
|  30 |   6 | move        | 1032093 |     109137 |    0.105743  |
|  30 |   8 | down        | 1032093 |      74264 |    0.0719548 |

_Showing 100 of 105 rows; full table is in the linked source file._

### Full-history observed crash events

- Observed event rows: 8,099.
- Distinct assets: 159.
- Labels: y_N15_p2, y_N5_p2.
- Source: `data/v7_final/crash_stats/observed_crash_events.csv`.

## 4. Methods and implementation evidence

The exact formulas and settings must be read in context from the source files below. This index prevents unsupported claims while keeping the reference compact.

### `tda_replication.py`

```text
1: """Replicate the Gidea-Katz sliding-window H1 persistence-landscape signal.
11: Landscape norms are computed analytically from finite H1 bars.
43: def landscape_norms(bars: np.ndarray) -> tuple[float, float, float, float]:
44: """Return exact landscape L1/L2 and retained bar-length diagnostics."""
62: l1, l2, proxy_l1, proxy_l2 = landscape_norms(bars)
68: "landscape_l1": l1,
69: "landscape_l2": l2,
106: ax.plot(subset.index, subset["landscape_l1"], label="numerical L1", color="tab:blue")
107: ax.plot(subset.index, subset["landscape_l2"], label="numerical L2", color="tab:orange")
115: ax.set_title("Gidea-Katz replication: H1 persistence-landscape norms and S&P 500")
130: "Landscape norms: exact integrals from finite H1 bars (L1=sum persistence^2/4; L2=sqrt(sum persistence^3/12)).",
138: for metric in ("landscape_l1", "landscape_l2", "bar_length_l1_proxy"):
```

### `csd_baseline.py`

```text
4: TDA replication. GARCH is refit on each window so its conditional volatility
31: def fit_garch_window(window_returns: np.ndarray) -> float:
32: """Fit GARCH(1,1) on one window and return sigma for its final day."""
34: model = arch_model(scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
40: # fail. Preserve the observed rolling indicators and mark GARCH NaN.
53: garch_sigma = fit_garch_window(block)
62: "garch_conditional_volatility": garch_sigma,
81: ("garch_conditional_volatility", "GARCH(1,1) conditional volatility"),
107: "GARCH is refit on every rolling window, with a constant mean and normal GARCH(1,1).",
108: "GARCH optimization uses returns scaled by 100 and volatility is reported in raw daily-return units.",
109: f"GARCH missing estimates: {int(indicators['garch_conditional_volatility'].isna().sum())}",
110: "This is a descriptive baseline, not a tuned crisis detector; no thresholds or parameter selection used crisis dates.",
```

### `augment_panel.py`

```text
2: existing unified panel - without recomputing the expensive TDA/GARCH blocks.
```

### `peer_index_forecasting_v2.py`

```text
7: GRU, attention-BiLSTM, TCN, and Transformer scores.
35: MODELS = ("gru", "attn_bilstm", "tcn", "transformer")
48: forward_min, thresholds = {}, {}
56: thresholds[target] = float(fit.quantile(0.02))
84: frame["label"] = (forward_min[target] <= thresholds[target]).astype(float)
91: return result, thresholds, features
112: "brier": float(brier_score_loss(y, p)), "threshold": float(cutoff),
127: optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=2e-3)
128: loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(neg / pos, 20.0)))
169: frame, thresholds, features = build_frame()
199: result = score(y[te], test_p, float(np.quantile(val_p, 0.95, method="higher")))
209: cutoff = float(np.quantile(ensemble_val, 0.95, method="higher"))
224: f"Label thresholds: {json.dumps(thresholds, sort_keys=True)}",
```

### `pipeline/features.py`

```text
9: (RiskMetrics-style span 20) and GARCH(1,1) conditional volatility. Fitting a
10: GARCH model on every trailing window is prohibitive for ~39 tickers, so the
11: model is refit on a trailing 250-day window every ``GARCH_STRIDE`` sessions
20: feature space alongside the joint TDA landscape.
27: Euclidean Vietoris-Rips, and summarised via H1 persistence-landscape norms
28: (closed-form integrals), persistence entropy, total persistence and the
29: finite-bar count, plus a discretised landscape curve vector
30: (``LANDSCAPE_LEVELS`` x ``LANDSCAPE_GRID``) usable directly as deep-model
33: * Per-ticker (Takens) - the same scheme applied to a scalar series via
46: except ImportError:  # GARCH block degrades to NaN, never blocks the pipeline
53: TAKENS_DIM = 3        # embedding dimension for the per-ticker point cloud
54: TAKENS_DELAY = 1      # embedding delay
55: LANDSCAPE_GRID = 12   # grid points per landscape level
56: LANDSCAPE_LEVELS = 3  # top-k landscape levels kept as a vector
57: LANDSCAPE_UPPER = 2.5 # grid upper bound (z-scored log-return units)
58: GARCH_WINDOW = 250    # trailing fit window for GARCH(1,1)
59: GARCH_STRIDE = 5      # refit cadence, sessions
76: out["csd_garch_vol"] = garch_vol(returns)
80: def garch_vol(returns: pd.Series, window: int = GARCH_WINDOW,
81: stride: int = GARCH_STRIDE) -> pd.Series:
82: """GARCH(1,1) conditional volatility, point-in-time with infrequent refits.
95: fit = arch_model(block * 100.0, mean="Constant", vol="GARCH", p=1, q=1,
142: def landscape_stats(bars: np.ndarray) -> tuple[float, float, float, int]:
143: """Closed-form landscape L1/L2, persistence entropy and bar count.
145: L1 = sum(p^2)/4 and L2 = sqrt(sum(p^3)/12) integrate each bar's landscape
151: persistence = bars[:, 1] - bars[:, 0]
152: l1 = float(np.square(persistence).sum() / 4.0)
153: l2 = float(np.sqrt(np.power(persistence, 3).sum() / 12.0))
154: probs = persistence / persistence.sum()
159: LANDSCAPE_GRID_POINTS = np.linspace(0.0, LANDSCAPE_UPPER, LANDSCAPE_GRID)
162: def landscape_vector(bars: np.ndarray) -> np.ndarray:
163: """Top-k landscape curves sampled on a fixed grid -> flat channel vector.
167: input (persistences computed on standardised windows are O(1)).
170: return np.zeros(LANDSCAPE_LEVELS * LANDSCAPE_GRID)
171: grid = LANDSCAPE_GRID_POINTS
176: fixed = np.zeros((LANDSCAPE_LEVELS, LANDSCAPE_GRID))
177: kept = min(LANDSCAPE_LEVELS, tents.shape[0])
182: def extended_landscape_stats(bars: np.ndarray) -> dict:
183: """All scalar summaries plus the discretised landscape vector."""
184: l1, l2, entropy, betti = landscape_stats(bars)
185: persistence = (bars[:, 1] - bars[:, 0]) if bars.size else np.zeros(0)
188: "total_persistence": float(persistence.sum()),
191: vector = landscape_vector(bars)
192: for level in range(LANDSCAPE_LEVELS):
193: for j in range(LANDSCAPE_GRID):
194: values[prefix.format(level + 1, j)] = float(vector[level * LANDSCAPE_GRID + j])
199: """Nominal landscape-summary names for an empty bars set, stable order."""
202: for level in range(1, LANDSCAPE_LEVELS + 1) for j in range(LANDSCAPE_GRID)
208: keys = ["l1", "l2", "entropy", "betti", "total_persistence"]
211: for level in range(1, LANDSCAPE_LEVELS + 1) for j in range(LANDSCAPE_GRID)
217: def takens_point_cloud(block: np.ndarray) -> np.ndarray:
224: [z[i : i + TAKENS_DIM] for i in range(len(z) - TAKENS_DIM + 1)], axis=0
228: def takens_features(returns: pd.Series, window: int = TDA_WINDOW) -> pd.DataFrame:
234: stats = extended_landscape_stats(finite_h1_bars(takens_point_cloud(block)))
315: stats = extended_landscape_stats(
```

### `pipeline/models.py`

```text
11: class imbalance is handled with a capped pos_weight, early stopping tracks
12: validation average precision with patience 4, and thresholds are never tuned
74: class GRUModel(nn.Module):
77: self.rnn = nn.GRU(n_features, hidden, batch_first=True)
104: class TCNModel(nn.Module):
112: blocks.append(TCNBlock(hidden, hidden, dilation))
123: class TCNBlock(nn.Module):
135: class TransformerModel(nn.Module):
140: layer = nn.TransformerEncoderLayer(
144: self.encoder = nn.TransformerEncoder(layer, layers)
157: "gru": GRUModel,
159: "tcn": TCNModel,
160: "transformer": TransformerModel,
192: optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
193: loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(negatives / positives, 20.0)))
```

### `v5_eval.py`

```text
128: """Label-free within-era per-asset threshold; skip assets with <60 rows."""
136: threshold=np.quantile(s[finite],1-a)
137: out[ix]=finite & (s>threshold)
149: if len(hist): out[ix]=np.isfinite(scores)&(scores>np.quantile(hist,1-a))
165: if len(hist)>=60 and np.isfinite(score): alarm[j]=score>np.quantile(hist[-window:],1-a)
```

### `v6_2_semivar.py`

```text
1: """V6 past-only SEMIVAR block and extended panel."""
15: out[f"semivar_up_{w}"] = x.clip(lower=0).pow(2).rolling(w, min_periods=w).mean().pow(.5)
16: out[f"semivar_down_{w}"] = x.clip(upper=0).pow(2).rolling(w, min_periods=w).mean().pow(.5)
17: out[f"semivar_logratio_{w}"] = np.log((out[f"semivar_up_{w}"] + 1e-8) /
18: (out[f"semivar_down_{w}"] + 1e-8))
19: out[f"semivar_skew_{w}"] = x.rolling(w, min_periods=w).skew()
49: print(f"[V6-SEMIVAR] assets={j}/{len(returns.columns)}", flush=True)
51: features.to_parquet(OUT / "semivar.parquet", index=False)
57: print(f"[V6-SEMIVAR] rows={len(features):,} columns={len(features.columns)}")
```

### `v7_final_synthesis.py`

```text
39: SEMIVAR = ["semivar_up_20", "semivar_down_20", "semivar_logratio_20",
40: "semivar_skew_20", "semivar_up_60", "semivar_down_60",
41: "semivar_logratio_60", "semivar_skew_60", "largest_gain_20",
61: tda_path = V5 / "blocks/V5_landscape_sampled.parquet"
70: "SEMIVAR_MOM": [c for c in CLASSICAL + SEMIVAR if c in f],
71: "PEER_CROSS": [c for c in CLASSICAL + SEMIVAR + PEER if c in f],
112: scale_pos_weight=min(neg / pos, 10), n_estimators=600,
130: def lead_time(df, alarm):
157: "mean_lead_time": lead_time(te, alarm), "alarm": alarm, "test": te,
232: "ci_low": float(np.quantile(means, .025)),
233: "ci_high": float(np.quantile(means, .975)),
238: for model in ("lightgbm", "gru", "tcn"):
249: for deep in ("gru", "tcn"):
265: "A causal head-to-head ablation of classical CSD, SEMIVAR momentum, "
270: "- 25-session purge gap and causal rolling-250-session RQ threshold.",
273: "- GRU and TCN use hidden size 24, sequence length 20, and seeds 42, 43, 44.",
275: "V6 supplied 17 SEMIVAR columns (not 18). V5 supplied 36 sampled "
276: "date-level landscape coordinates; these are merged causally by date.",
290: mean_lead_time=("mean_lead_time", "mean"),
294: "Event recall | Random recall | Excess recall | Lead time | Folds |",
300: "{mean_lead_time:.2f} | {folds} |".format(**r))
334: for model in ("logreg", "lightgbm", "gru", "tcn")
335: for seed in (SEEDS if model in {"gru", "tcn"} else (42,))]
359: "panel": "GroupA" if model in {"gru", "tcn"} else "unified_159",
374: "tda_source": "V5_landscape_sampled, first 36 date-level coordinates",
375: "semivar_source": "V6 panel; 17 available columns",
376: "rq": "rolling 250-session causal quantile, 5 percent",
```

### `pipeline_v5/sequence.py`

```text
10: from pipeline.models import (AttnBiLSTMModel,BiLSTMModel,GRUModel,LSTMModel,
11: TCNModel,TransformerModel)
29: "gru":lambda:GRUModel(n_features,hidden),
32: "tcn":lambda:TCNModel(n_features,hidden),
33: "transformer":lambda:TransformerModel(n_features,d_model=hidden,heads=4,layers=2)}
56: pos_weight=min(neg_used/max(len(pos),1),20.)
58: opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=1e-3)
59: loss_fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,dtype=torch.float32))
81: return model,best_ap,valp,testp,{"epochs":epoch+1,"pos_weight":pos_weight,
```

## 5. Experiment status and quantitative outputs

### V7 final synthesis

- Status file: `{
  "completed_rows": 240,
  "planned_cells": 240,
  "remaining_cells": 0,
  "feature_counts": {
    "CLASSICAL_CSD": 9,
    "SEMIVAR_MOM": 26,
    "PEER_CROSS": 35,
    "TDA_ONLY_NORM": 36,
    "GRAND_COMBINED": 71
  },
  "tda_source": "V5_landscape_sampled, first 36 date-level coordinates",
  "semivar_source": "V6 panel; 17 available columns",
  "rq": "rolling 250-session causal quantile, 5 percent",
  "deep_group": "four core US indices"
}`
- Saved result rows: 240.
- Labels: y_N5_p2, y_down_N15_p2.
- Feature sets: CLASSICAL_CSD, GRAND_COMBINED, PEER_CROSS, SEMIVAR_MOM, TDA_ONLY_NORM.
- Models: gru, lightgbm, logreg, tcn.
- Folds: 2011_2015, 2016_2020, 2021_plus.

| label         | feature_set    | model    |   cells |   mean_ap |   mean_auc |   mean_alarm_rate |   mean_event_recall |   mean_lead |
|:--------------|:---------------|:---------|--------:|----------:|-----------:|------------------:|--------------------:|------------:|
| y_N5_p2       | CLASSICAL_CSD  | gru      |       9 | 0.0365596 |   0.705565 |         0.100477  |           0.304029  |    11.1663  |
| y_N5_p2       | CLASSICAL_CSD  | lightgbm |       3 | 0.0270351 |   0.655179 |         0.0387543 |           0.158089  |    15.1602  |
| y_N5_p2       | CLASSICAL_CSD  | logreg   |       3 | 0.0292497 |   0.669759 |         0.0691024 |           0.234431  |    13.5856  |
| y_N5_p2       | CLASSICAL_CSD  | tcn      |       9 | 0.051893  |   0.765031 |         0.0812291 |           0.345706  |    11.2334  |
| y_N5_p2       | GRAND_COMBINED | gru      |       9 | 0.0628344 |   0.803608 |         0.075209  |           0.432804  |     9.19348 |
| y_N5_p2       | GRAND_COMBINED | lightgbm |       3 | 0.0413928 |   0.719582 |         0.0479585 |           0.176262  |    14.6626  |
| y_N5_p2       | GRAND_COMBINED | logreg   |       3 | 0.0377772 |   0.741394 |         0.0763809 |           0.264366  |    12.3029  |
| y_N5_p2       | GRAND_COMBINED | tcn      |       9 | 0.0512202 |   0.767261 |         0.0742943 |           0.323891  |     8.5117  |
| y_N5_p2       | PEER_CROSS     | gru      |       9 | 0.0657377 |   0.809533 |         0.07114   |           0.404233  |    10.0611  |
| y_N5_p2       | PEER_CROSS     | lightgbm |       3 | 0.0358147 |   0.727507 |         0.0460709 |           0.189407  |    14.9987  |
| y_N5_p2       | PEER_CROSS     | logreg   |       3 | 0.0400162 |   0.755886 |         0.0784668 |           0.268973  |    12.7878  |
| y_N5_p2       | PEER_CROSS     | tcn      |       9 | 0.0571055 |   0.794388 |         0.0774022 |           0.398087  |     9.13128 |
| y_N5_p2       | SEMIVAR_MOM    | gru      |       9 | 0.0512265 |   0.773106 |         0.0781505 |           0.382011  |     9.88749 |
| y_N5_p2       | SEMIVAR_MOM    | lightgbm |       3 | 0.0378468 |   0.718877 |         0.0457145 |           0.176697  |    14.7638  |
| y_N5_p2       | SEMIVAR_MOM    | logreg   |       3 | 0.0408221 |   0.735773 |         0.0806698 |           0.26229   |    12.8204  |
| y_N5_p2       | SEMIVAR_MOM    | tcn      |       9 | 0.0552124 |   0.78704  |         0.0799816 |           0.37733   |     9.23871 |
| y_N5_p2       | TDA_ONLY_NORM  | gru      |       9 | 0.0203444 |   0.534259 |         0.057379  |           0.0976801 |    16.1815  |
| y_N5_p2       | TDA_ONLY_NORM  | lightgbm |       3 | 0.0125672 |   0.490932 |         0.0139563 |           0.0341336 |    17.6125  |
| y_N5_p2       | TDA_ONLY_NORM  | logreg   |       3 | 0.0127853 |   0.499177 |         0.03254   |           0.126852  |    14.0565  |
| y_N5_p2       | TDA_ONLY_NORM  | tcn      |       9 | 0.0178978 |   0.491336 |         0.0600125 |           0.134676  |    13.2894  |
| y_down_N15_p2 | CLASSICAL_CSD  | gru      |       9 | 0.0207601 |   0.602455 |         0.0992272 |           0.265432  |    13.5421  |
| y_down_N15_p2 | CLASSICAL_CSD  | lightgbm |       3 | 0.017393  |   0.609642 |         0.034699  |           0.176856  |    15.9755  |
| y_down_N15_p2 | CLASSICAL_CSD  | logreg   |       3 | 0.0205022 |   0.637739 |         0.0711415 |           0.303158  |    13.7979  |
| y_down_N15_p2 | CLASSICAL_CSD  | tcn      |       9 | 0.0232705 |   0.641855 |         0.0786493 |           0.313272  |    13.8797  |
| y_down_N15_p2 | GRAND_COMBINED | gru      |       9 | 0.0348821 |   0.712507 |         0.0804514 |           0.384259  |    11.3648  |
| y_down_N15_p2 | GRAND_COMBINED | lightgbm |       3 | 0.0232764 |   0.669894 |         0.0449949 |           0.222169  |    15.3274  |
| y_down_N15_p2 | GRAND_COMBINED | logreg   |       3 | 0.0259847 |   0.714309 |         0.0818492 |           0.335824  |    12.2108  |
| y_down_N15_p2 | GRAND_COMBINED | tcn      |       9 | 0.0258235 |   0.66144  |         0.0747556 |           0.351852  |    10.6877  |
| y_down_N15_p2 | PEER_CROSS     | gru      |       9 | 0.0314743 |   0.699085 |         0.0868224 |           0.438272  |    11.7296  |
| y_down_N15_p2 | PEER_CROSS     | lightgbm |       3 | 0.0266692 |   0.693101 |         0.0453743 |           0.222785  |    15.7723  |
| y_down_N15_p2 | PEER_CROSS     | logreg   |       3 | 0.0270508 |   0.725708 |         0.0828178 |           0.318341  |    13.2489  |
| y_down_N15_p2 | PEER_CROSS     | tcn      |       9 | 0.0390985 |   0.727392 |         0.0842862 |           0.412809  |    11.0833  |
| y_down_N15_p2 | SEMIVAR_MOM    | gru      |       9 | 0.0276645 |   0.718059 |         0.0862127 |           0.338735  |    11.189   |
| y_down_N15_p2 | SEMIVAR_MOM    | lightgbm |       3 | 0.0265635 |   0.68827  |         0.0434945 |           0.201463  |    15.9274  |
| y_down_N15_p2 | SEMIVAR_MOM    | logreg   |       3 | 0.0267095 |   0.707619 |         0.0844987 |           0.321049  |    13.0469  |
| y_down_N15_p2 | SEMIVAR_MOM    | tcn      |       9 | 0.0296567 |   0.710039 |         0.082866  |           0.349537  |    11.4477  |
| y_down_N15_p2 | TDA_ONLY_NORM  | gru      |       9 | 0.0201693 |   0.507876 |         0.0547309 |           0.199846  |    13.9684  |
| y_down_N15_p2 | TDA_ONLY_NORM  | lightgbm |       3 | 0.0124441 |   0.499151 |         0.005425  |           0.0530775 |    18.1744  |
| y_down_N15_p2 | TDA_ONLY_NORM  | logreg   |       3 | 0.0125241 |   0.503999 |         0.025513  |           0.255786  |    15.4094  |
| y_down_N15_p2 | TDA_ONLY_NORM  | tcn      |       9 | 0.0160341 |   0.485697 |         0.0566807 |           0.329475  |    13.3825  |

Full V7 results: `data/v7_final/v7_ablation_results.csv`.

### Best verified V7 aggregate cells

| label         |      fold | feature_set    | model   |   seed |        ap |      auc |   alarm_day_rate |   event_recall |   mean_lead_time |
|:--------------|----------:|:---------------|:--------|-------:|----------:|---------:|-----------------:|---------------:|-----------------:|
| y_down_N15_p2 | 2016_2020 | PEER_CROSS     | tcn     |     42 | 0.0555194 | 0.789291 |        0.0975096 |       1        |          6.53333 |
| y_down_N15_p2 | 2016_2020 | PEER_CROSS     | gru     |     42 | 0.0549005 | 0.776782 |        0.103257  |       1        |          7.46667 |
| y_down_N15_p2 | 2016_2020 | GRAND_COMBINED | gru     |     44 | 0.102442  | 0.748325 |        0.0833333 |       1        |          8.46667 |
| y_down_N15_p2 | 2016_2020 | CLASSICAL_CSD  | gru     |     42 | 0.0399703 | 0.702588 |        0.0971264 |       1        |          7.93333 |
| y_down_N15_p2 | 2016_2020 | CLASSICAL_CSD  | tcn     |     42 | 0.0362532 | 0.649411 |        0.0875479 |       1        |          6.86667 |
| y_down_N15_p2 | 2016_2020 | PEER_CROSS     | tcn     |     43 | 0.0769865 | 0.833939 |        0.0977011 |       0.888889 |          5.73333 |
| y_down_N15_p2 | 2016_2020 | PEER_CROSS     | tcn     |     44 | 0.0721491 | 0.807449 |        0.0873563 |       0.888889 |          9.06667 |
| y_down_N15_p2 | 2016_2020 | PEER_CROSS     | gru     |     44 | 0.0662098 | 0.78174  |        0.097318  |       0.888889 |          8.8     |
| y_down_N15_p2 | 2016_2020 | SEMIVAR_MOM    | tcn     |     42 | 0.0445973 | 0.715358 |        0.0911877 |       0.888889 |          8       |
| y_down_N15_p2 | 2016_2020 | SEMIVAR_MOM    | gru     |     43 | 0.0427889 | 0.697695 |        0.101533  |       0.888889 |         11.4667  |
| y_down_N15_p2 | 2016_2020 | TDA_ONLY_NORM  | tcn     |     43 | 0.0305822 | 0.52968  |        0.0536398 |       0.888889 |         13.0667  |
| y_N5_p2       | 2011_2015 | GRAND_COMBINED | gru     |     43 | 0.0440603 | 0.858527 |        0.0810966 |       0.785714 |          7.07692 |
| y_down_N15_p2 | 2016_2020 | SEMIVAR_MOM    | gru     |     44 | 0.0452115 | 0.743025 |        0.103065  |       0.777778 |          6.93333 |
| y_down_N15_p2 | 2016_2020 | GRAND_COMBINED | gru     |     42 | 0.0498821 | 0.741241 |        0.0795019 |       0.777778 |         11.2667  |
| y_down_N15_p2 | 2016_2020 | SEMIVAR_MOM    | tcn     |     43 | 0.0413694 | 0.723046 |        0.0967433 |       0.777778 |          7.26667 |
| y_down_N15_p2 | 2016_2020 | CLASSICAL_CSD  | tcn     |     43 | 0.0413532 | 0.676192 |        0.0975096 |       0.777778 |          7.46667 |
| y_N5_p2       | 2011_2015 | PEER_CROSS     | tcn     |     44 | 0.0383138 | 0.814542 |        0.0697853 |       0.714286 |          7       |
| y_N5_p2       | 2011_2015 | SEMIVAR_MOM    | gru     |     42 | 0.0384398 | 0.790369 |        0.0780291 |       0.714286 |          6.28571 |
| y_down_N15_p2 | 2016_2020 | SEMIVAR_MOM    | tcn     |     44 | 0.0570514 | 0.763653 |        0.107471  |       0.666667 |          7.6     |
| y_down_N15_p2 | 2016_2020 | GRAND_COMBINED | tcn     |     42 | 0.0473626 | 0.738459 |        0.0710728 |       0.666667 |          7.33333 |
| y_down_N15_p2 | 2016_2020 | CLASSICAL_CSD  | tcn     |     44 | 0.0360093 | 0.677258 |        0.0965517 |       0.666667 |         10.8667  |
| y_down_N15_p2 | 2016_2020 | CLASSICAL_CSD  | gru     |     43 | 0.0277211 | 0.515536 |        0.111686  |       0.666667 |         10.0667  |
| y_N5_p2       | 2011_2015 | PEER_CROSS     | gru     |     44 | 0.0440935 | 0.826118 |        0.0630752 |       0.642857 |         12.0769  |
| y_N5_p2       | 2011_2015 | PEER_CROSS     | gru     |     42 | 0.0354315 | 0.791082 |        0.0628834 |       0.642857 |          9.53846 |
| y_N5_p2       | 2011_2015 | PEER_CROSS     | tcn     |     42 | 0.045694  | 0.857709 |        0.0845475 |       0.571429 |          5.53846 |

_Showing 25 of 240 rows; full table is in the linked source file._

### V7 strict pre-crash correction

Only alarms strictly earlier than `crash_start` are valid detections. The legacy in-crash ledger is not used as forecasting evidence.
- Pre-crash event/model rows: 58,330.
- Pre-crash summary rows: 120.
- Windows: 1, 5, 10, 20, and 60 prior trading sessions.
- Full event ledger: `data/v7_final/precrash_ledger/precrash_model_event_ledger.csv`.
- Event view: `data/v7_final/precrash_ledger/precrash_by_event.csv`.
- Summary: `data/v7_final/precrash_ledger/precrash_model_summary.csv`.

### V6 Scheme W headline table

|   N | direction   | feature_set      | engine   |   mean_ap |   mean_auc |   mean_excess |   off_budget |
|----:|:------------|:-----------------|:---------|----------:|-----------:|--------------:|-------------:|
|   5 | down        | BEST_SET_SEMIVAR | E-bin    |  0.3395   |   0.891722 |     0.301037  |            0 |
|   5 | down        | BEST_SET_SEMIVAR | E-sev    |  0.335257 |   0.885917 |     0.317499  |            0 |
|   5 | down        | B3               | E-bin    |  0.247285 |   0.855665 |     0.353397  |            0 |
|   5 | down        | B3               | E-sev    |  0.239403 |   0.851355 |     0.357841  |            0 |
|  15 | down        | BEST_SET_SEMIVAR | E-bin    |  0.455946 |   0.935436 |     0.228793  |            0 |
|  15 | down        | BEST_SET_SEMIVAR | E-sev    |  0.461594 |   0.935398 |     0.236308  |            0 |
|  15 | down        | B3               | E-bin    |  0.278999 |   0.886161 |     0.20384   |            0 |
|  15 | down        | B3               | E-sev    |  0.276186 |   0.883965 |     0.187015  |            0 |
|  30 | down        | BEST_SET_SEMIVAR | E-sev    |  0.434497 |   0.928009 |     0.109704  |            0 |
|  30 | down        | BEST_SET_SEMIVAR | E-bin    |  0.42424  |   0.925008 |     0.0972684 |            0 |
|  30 | down        | B3               | E-sev    |  0.326452 |   0.903241 |     0.121118  |            2 |
|  30 | down        | B3               | E-bin    |  0.323416 |   0.901649 |     0.120718  |            1 |
|   5 | up          | BEST_SET_SEMIVAR | E-bin    |  0.19514  |   0.831539 |     0.231693  |            0 |
|   5 | up          | BEST_SET_SEMIVAR | E-sev    |  0.183512 |   0.812823 |     0.229871  |            0 |
|   5 | up          | B3               | E-bin    |  0.156888 |   0.781812 |     0.242489  |            0 |
|   5 | up          | B3               | E-sev    |  0.144251 |   0.762765 |     0.208966  |            0 |
|  15 | up          | BEST_SET_SEMIVAR | E-bin    |  0.262851 |   0.88795  |     0.208463  |            0 |
|  15 | up          | BEST_SET_SEMIVAR | E-sev    |  0.259362 |   0.880939 |     0.203981  |            0 |
|  15 | up          | B3               | E-bin    |  0.157672 |   0.790541 |     0.153168  |            0 |
|  15 | up          | B3               | E-sev    |  0.153387 |   0.779914 |     0.14943   |            0 |
|  30 | up          | BEST_SET_SEMIVAR | E-bin    |  0.215894 |   0.852962 |     0.126087  |            0 |
|  30 | up          | BEST_SET_SEMIVAR | E-sev    |  0.217017 |   0.850002 |     0.137889  |            0 |
|  30 | up          | B3               | E-sev    |  0.144483 |   0.76626  |     0.0704846 |            0 |
|  30 | up          | B3               | E-bin    |  0.144381 |   0.763741 |     0.0441495 |            0 |

### V5/V6 pending and completed state

- V5 E2 checkpoint rows: 1; checkpoint file: `data/v5/e2_v7_final_checkpoint.csv`.
- V5/V6 outputs already present are preserved; no new long training job was started by this compact-report step.

## 6. Canonical result files

| Area | Files |
|---|---|
| V1/V2 benchmark | `data/benchmark/all_results.csv`, `data/benchmark/all_results_v1.csv` |
| V3/V4 | `data/v3/`, `data/v4/` |
| V5 | `data/v5/e1_results.csv`, `data/v5/e2_results.csv`, diagnostics |
| V6 | `data/v6/` and archived V6 result tables |
| V7 | `data/v7_final/v7_ablation_results.csv` and `data/v7_final/scores/` |
| Crash population | `data/v7_final/crash_stats/observed_crash_events.csv` |

## 7. Scientific limitations

- V7 currently has complete saved cells for its configured folds, but its configured folds are not the same as testing every historical date from 1927 onward.
- The compact document does not convert missing score coverage into false model misses.
- A model detection is a forecast only when the alarm precedes onset; alarms on or after onset are excluded.
