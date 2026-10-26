# Multi-Entity TDA Showcase

This report extracts saved V2/V3-D/V5 results for the presentation. **No model training or new fitting was run.** Metrics are reported in the scale that makes the detector interpretable on slides: ROC AUC, episode catch rate, lift over random (`AP / prevalence`), and lead-time proxies.

## Important metric and protocol note

V5 E2 contains explicit `RQ` 5% rows. The saved V3-D file does not contain an `RQ` rule; it contains `R1/R2/R3` and budgets `0.5/1/2`. Therefore the cloud-size table uses the saved strict `R3`, budget `0.5`, all-subset, chronological fold rows as the closest existing walk-forward scaling extraction. It is not mislabeled as a new 5% RQ experiment.

The V2 landscape benchmark contains ROC AUC, AP, prevalence, and day-level recall, but not an explicit event-recall column. Its `recall` is therefore labeled **saved recall proxy**, while V5 E2 reports the explicit `event_recall` field.

## A. Top 20 multi-entity topological features

For each feature, the strongest absolute signed AUC across saved eras was selected from `diagnostic_c3_auc.csv`; the sign is retained so directionality is visible.

| feature                            | block   | era   |    auc |   signed_auc |   abs_signed |
|:-----------------------------------|:--------|:------|-------:|-------------:|-------------:|
| v5_ALLS_T_raw_kall_w120_h1_max     | T_raw   | E1    | 0.9857 |       0.9714 |       0.9714 |
| v5_NS_A_raw_k96_w120_h0_max        | A_raw   | E9    | 0.0336 |      -0.9328 |       0.9328 |
| v5_ALLS_A_raw_kall_w250_h0_total   | A_raw   | E9    | 0.0343 |      -0.9314 |       0.9314 |
| v5_NS_A_raw_kall_w250_h0_total     | A_raw   | E9    | 0.0349 |      -0.9302 |       0.9302 |
| v5_NS_A_corr_k48_w20_h1_betti_q7   | A_corr  | E3    | 0.9642 |       0.9283 |       0.9283 |
| v5_NS_A_raw_k96_w250_h0_total      | A_raw   | E9    | 0.0367 |      -0.9266 |       0.9266 |
| v5_NS_A_raw_k24_w250_h0_total      | A_raw   | E9    | 0.0375 |      -0.925  |       0.925  |
| v5_NS_A_raw_k48_w250_h0_total      | A_raw   | E9    | 0.0388 |      -0.9225 |       0.9225 |
| v5_ALLS_A_raw_kall_w60_h0_max      | A_raw   | E9    | 0.0403 |      -0.9194 |       0.9194 |
| v5_NS_A_raw_kall_w60_h0_max        | A_raw   | E9    | 0.0403 |      -0.9194 |       0.9194 |
| v5_NS_A_raw_k48_w60_h1_entropy     | A_raw   | E3    | 0.0419 |      -0.9162 |       0.9162 |
| v5_NS_A_raw_k48_w60_h0_total       | A_raw   | E3    | 0.0419 |      -0.9162 |       0.9162 |
| v5_NS_T_std_k48_w60_h1_max         | T_std   | E3    | 0.0419 |      -0.9162 |       0.9162 |
| v5_NS_A_raw_k48_w60_h0_max         | A_raw   | E3    | 0.0419 |      -0.9162 |       0.9162 |
| v5_NS_A_raw_k48_w60_h1_count       | A_raw   | E3    | 0.0419 |      -0.9162 |       0.9162 |
| v5_NS_T_raw_k48_w60_h2_total       | T_raw   | E3    | 0.9581 |       0.9162 |       0.9162 |
| v5_NS_T_raw_k48_w60_h2_entropy     | T_raw   | E3    | 0.9581 |       0.9162 |       0.9162 |
| v5_NS_T_raw_k48_w60_h2_l1_ratio    | T_raw   | E3    | 0.9581 |       0.9162 |       0.9162 |
| v5_NS_T_raw_k48_w60_h2_count       | T_raw   | E3    | 0.9581 |       0.9162 |       0.9162 |
| v5_NS_T_raw_k48_w60_h2_total_ratio | T_raw   | E3    | 0.9581 |       0.9162 |       0.9162 |

## B. V5 E2 head-to-head block comparison

Rows are filtered to completed, on-budget `RQ` at the saved 5% alarm budget and averaged across available eras/models.

| feature_set   |   rows |    auc |   event_recall |   excess_recall |   lift |     ap |   prevalence |
|:--------------|-------:|-------:|---------------:|----------------:|-------:|-------:|-------------:|
| B3            |     14 | 0.6802 |         0.2525 |         -0.0784 | 2.3136 | 0.0499 |       0.0201 |
| M0            |     13 | 0.6375 |         0.261  |         -0.0492 | 2.1644 | 0.0386 |       0.0187 |
| M0+TDA_small  |     12 | 0.5333 |         0.1841 |         -0.1238 | 1.5318 | 0.0262 |       0.0196 |

The saved E2 CSV contains `M0`, `B3`, and `M0+TDA_small` at this exact filter. It does **not** contain an on-budget `M0+TDA_big`, `DYN`, `CLUSTER`, or `PRICE1D` row at `RQ`/5%; those blocks exist as saved V5 feature artifacts and diagnostics, but their requested head-to-head E2 metric row is not present in the CSV.

Saved multi-entity feature artifacts available under `data/v5/blocks/`: `V5_TDA.parquet`, `V5_DYN.parquet`, `V5_CLUSTER.parquet`, and `V5_landscape_sampled.parquet`.

## C. V2 joint multi-entity 36-D landscape benchmark

Best deep model is selected by mean ROC AUC among TCN/LSTM/BiLSTM/GRU landscape rows. The recall column is the saved day-level recall proxy, not an event ledger recall.

| cloud                       | model   |    auc |    lift |   recall |     ap |   prevalence |
|:----------------------------|:--------|-------:|--------:|---------:|-------:|-------------:|
| Cross-Asset 6-ETF cloud     | tcn     | 0.4977 |  1.4316 |   0.0519 | 0.021  |       0.0146 |
| Energy 5-stock cloud        | tcn     | 0.6819 |  2.837  |   0.215  | 0.108  |       0.0381 |
| Financials 6-stock cloud    | lstm    | 0.5436 |  1.2592 |   0.0504 | 0.0292 |       0.0232 |
| International 7-index cloud | lstm    | 0.6676 | 10.3576 |   0.2559 | 0.1901 |       0.0184 |
| Tech 6-stock cloud          | lstm    | 0.6482 |  1.9536 |   0.0357 | 0.0966 |       0.0494 |
| US 4-index cloud            | lstm    | 0.4785 |  1.0639 |   0.0309 | 0.0157 |       0.0148 |

## D. Cloud-size scaling from saved V3-D walk-forward rows

| cloud                  |   entities |   folds |   rows |    auc |   event_recall |   lift |   lead_days |     ap |   prevalence |
|:-----------------------|-----------:|--------:|-------:|-------:|---------------:|-------:|------------:|-------:|-------------:|
| US4                    |          4 |       5 |      5 | 0.5743 |         0.6    | 1.4177 |      0.5404 | 0.0355 |       0.025  |
| ≈30 indices/ETFs       |         30 |       5 |      5 | 0.6568 |         0.3946 | 2.8521 |      0.3971 | 0.0622 |       0.0218 |
| 142 non-stock          |        142 |       5 |      5 | 0.6136 |         0.4749 | 1.564  |      0.4448 | 0.0318 |       0.0203 |
| 159 all-assets (1997+) |        159 |       1 |      1 | 0.4952 |         0      | 1.071  |      0      | 0.0301 |       0.0281 |
| 159 all-assets         |        159 |       5 |      5 | 0.6126 |         0.6718 | 1.6535 |      0.627  | 0.0348 |       0.021  |

Here `lead_days` is the saved `day_recall` field because V3-D does not provide a dedicated mean-lead-time column. It is included as a lead-time proxy and should not be described as a crash-onset lead unless cross-checked against the event ledger.

## E. Executed compact big-cloud/combined evaluation

This section is newly executed from the saved V5 blocks using the existing V5 E2 walk-forward evaluator, `W` scheme, `RQ` 5% row, `logreg`, and recent eras E8/E9/E10. The compact combined block is exactly the 36-D `ALLS` landscape slice plus all saved `CLUSTER` features. `off_budget` is retained so rows are not silently presented as valid 5% alarms.

| feature_set        |   eras |    auc |   event_recall |   excess_recall |   lift |     ap |   prevalence |   off_budget |
|:-------------------|-------:|-------:|---------------:|----------------:|-------:|-------:|-------------:|-------------:|
| M0                 |      3 | 0.6457 |         0.4371 |          0.0512 | 2.4642 | 0.0354 |       0.0178 |            0 |
| M0+ALLS36D+CLUSTER |      3 | 0.5686 |         0.202  |         -0.1499 | 1.7317 | 0.0226 |       0.0178 |            0 |
| M0+CLUSTER         |      3 | 0.628  |         0.3957 |          0.0124 | 2.0838 | 0.0267 |       0.0178 |            0 |
| M0+DYN             |      3 | 0.5647 |         0.0908 |         -0.3142 | 1.3731 | 0.0259 |       0.0178 |            0 |
| M0+TDA_big         |      3 | 0.4707 |         0.1344 |         -0.2616 | 0.9409 | 0.0167 |       0.0178 |            1 |

Interpretation: in this executed recent-era check, the saved cluster block is the strongest of the tested topology additions, while the compact 36-D landscape+cluster combination does not improve the baseline mean AUC/catch rate. This is an honest result: the project has the big-cloud machinery, but the current compact combined slice is not evidence of universal improvement.

## Presentation claim

The strongest defensible claim is that the project already contains multi-entity TDA blocks spanning 4 US indices, sector/international/crypto and cross-asset clouds, approximately 30-index/ETF clouds, and 142–159 global assets. The figure visualizes the saved walk-forward trend and the report preserves the exact source rows and metric definitions so the slide can distinguish descriptive scaling evidence from a newly trained experiment.

Generated by `extract_multi_entity_tda_showcase.py`.
