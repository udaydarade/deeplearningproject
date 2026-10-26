# Presentation figure manifest

All charts were generated from saved repository artifacts by `generate_presentation_figures.py`.

## fig1_tda_intuition_worked_example.png
- Source: `data/group_a_original_indices/returns.csv; Ripser on 50 trading-day windows`
- Exact values plotted: two date-specific z-scored clouds, finite H1 pairs, 12-point landscape vectors, and derived H1/max-persistence/L1/L2/entropy table

## fig2_v2_sector_landscape_breakthrough.png
- Source: `data/benchmark/run_record_v2.csv plus reported top model cells`
- Exact values plotted: means=[[0.049583, 0.060277, 0.069368, 0.057129, 0.060976], [0.021589, 0.016853, 0.047061, 0.026644, 0.028859], [0.056421, 0.062411, 0.066195, 0.066815, 0.052933], [0.04668, 0.027244, 0.023818, 0.047793, 0.030283], [0.021001, 0.013335, 0.010771, 0.022105, 0.026652], [0.023327, 0.013834, 0.015582, 0.019011, 0.01194]] for datasets=['energy', 'international', 'tech', 'financials', 'group_b_cross_asset', 'group_a_us_indices'], modes=['classical', 'tda', 'landscape', 'combined', 'full']; annotated cells={'international landscape + LSTM': 0.1102, 'tech tda + BiLSTM': 0.1038, 'energy landscape + TCN': 0.0971, 'group_b full + GRU': 0.0604}

## fig3_leadtime_crypto_and_sectors.png
- Source: `data/benchmark/qualitative/family_stats_dataset.csv`
- Exact values plotted: shares={'crypto': {'CROSS': 0.267, 'CSD': 0.233, 'TDA-index': 0.833, 'TDA-joint': 0.567}, 'energy': {'CROSS': 0.272, 'CSD': 0.353, 'TDA-index': 0.642, 'TDA-joint': 0.604}, 'financials': {'CROSS': 0.313, 'CSD': 0.417, 'TDA-index': 0.667, 'TDA-joint': 0.585}, 'group_a': {'CROSS': 0.232, 'CSD': 0.426, 'TDA-index': 0.789, 'TDA-joint': 0.532}, 'group_b': {'CROSS': 0.471, 'CSD': 0.513, 'TDA-index': 0.673, 'TDA-joint': 0.553}, 'international': {'CROSS': 0.298, 'CSD': 0.422, 'TDA-index': 0.606, 'TDA-joint': 0.597}, 'tech': {'CROSS': 0.193, 'CSD': 0.396, 'TDA-index': 0.632, 'TDA-joint': 0.573}}; auc={'crypto': {'CROSS': 0.414, 'CSD': 0.461, 'TDA-index': 0.569, 'TDA-joint': 0.575}, 'energy': {'CROSS': 0.535, 'CSD': 0.529, 'TDA-index': 0.5, 'TDA-joint': 0.482}, 'financials': {'CROSS': 0.489, 'CSD': 0.543, 'TDA-index': 0.492, 'TDA-joint': 0.492}, 'group_a': {'CROSS': 0.476, 'CSD': 0.591, 'TDA-index': 0.572, 'TDA-joint': 0.505}, 'group_b': {'CROSS': 0.549, 'CSD': 0.66, 'TDA-index': 0.479, 'TDA-joint': 0.488}, 'international': {'CROSS': 0.462, 'CSD': 0.502, 'TDA-index': 0.522, 'TDA-joint': 0.497}, 'tech': {'CROSS': 0.424, 'CSD': 0.52, 'TDA-index': 0.558, 'TDA-joint': 0.508}}; overall reference shares TDA-index=68%, TDA-joint=58%, CSD=40%, CROSS=30%

## fig4_v5_scale_law_and_redundancy.png
- Source: `data/v5/diagnostic_c1_raw_scaling.csv and data/v5/diagnostic_c2_proxy_r2.csv`
- Exact values plotted: median slopes={('A_raw', 20): 0.484397, ('A_raw', 60): 0.524641, ('A_raw', 120): 0.510324, ('A_raw', 250): 0.49147, ('T_raw', 60): 0.476979, ('T_raw', 120): 0.543584}; proxy_r2 histogram uses 1902 finite values from 2044 rows, count above .8=1

## fig5_v6_semivar_and_transfer.png
- Source: `data/v6/.../scheme_w_summary.csv and t2_cross_transfer_n5.csv`
- Exact values plotted: scheme-W mean AP by N/direction/feature; transfer AUC={'Native down': 0.886, 'Native up': 0.813, 'Down→up': 0.1877172703641264, 'Up→down': 0.1716653269486704}

## fig6_v7_final_synthesis_ablation.png
- Source: `data/v7_final/v7_ablation_results.csv`
- Exact values plotted: mean AP N5={'gru': {'CLASSICAL_CSD': 0.03656, 'SEMIVAR_MOM': 0.051226, 'PEER_CROSS': 0.065738, 'TDA_ONLY_NORM': 0.020344, 'GRAND_COMBINED': 0.062834}, 'lightgbm': {'CLASSICAL_CSD': 0.027035, 'SEMIVAR_MOM': 0.037847, 'PEER_CROSS': 0.035815, 'TDA_ONLY_NORM': 0.012567, 'GRAND_COMBINED': 0.041393}, 'logreg': {'CLASSICAL_CSD': 0.02925, 'SEMIVAR_MOM': 0.040822, 'PEER_CROSS': 0.040016, 'TDA_ONLY_NORM': 0.012785, 'GRAND_COMBINED': 0.037777}, 'tcn': {'CLASSICAL_CSD': 0.051893, 'SEMIVAR_MOM': 0.055212, 'PEER_CROSS': 0.057105, 'TDA_ONLY_NORM': 0.017898, 'GRAND_COMBINED': 0.05122}}, N15={'gru': {'CLASSICAL_CSD': 0.02076, 'SEMIVAR_MOM': 0.027665, 'PEER_CROSS': 0.031474, 'TDA_ONLY_NORM': 0.020169, 'GRAND_COMBINED': 0.034882}, 'lightgbm': {'CLASSICAL_CSD': 0.017393, 'SEMIVAR_MOM': 0.026563, 'PEER_CROSS': 0.026669, 'TDA_ONLY_NORM': 0.012444, 'GRAND_COMBINED': 0.023276}, 'logreg': {'CLASSICAL_CSD': 0.020502, 'SEMIVAR_MOM': 0.026709, 'PEER_CROSS': 0.027051, 'TDA_ONLY_NORM': 0.012524, 'GRAND_COMBINED': 0.025985}, 'tcn': {'CLASSICAL_CSD': 0.023271, 'SEMIVAR_MOM': 0.029657, 'PEER_CROSS': 0.039099, 'TDA_ONLY_NORM': 0.016034, 'GRAND_COMBINED': 0.025824}}; event recall N5 annotation

## fig7_cross_index_and_peer_models.png
- Source: `data/cross_index_forecasting/cross_index_walk_forward_metrics.csv and data/forecasting/peer_index_deep_learning_v2_metrics.csv`
- Exact values plotted: cross-index model means={'all_indices': 0.0501, 'other_only': 0.0407, 'self_only': 0.0368}; peer model means={'gru': 0.0508, 'ensemble_top2': 0.0414, 'tcn': 0.0401, 'attn_bilstm': 0.04, 'transformer': 0.0362}

## fig8_strict_precrash_lead_windows.png
- Source: `data/v7_final/precrash_ledger/precrash_model_summary.csv`
- Exact values plotted: mean strict event recall across labels/folds/seeds for four feature sets at K=1,5,10,20,60

## fig9_qualitative_events_composite.png
- Source: `data\forecasting\event_2022_bear_market.png, data\forecasting\event_covid-19_selloff.png, data\forecasting\event_dot-com_bear_market.png, data\forecasting\event_lehman___global_financial_crisis.png`
- Exact values plotted: pixel-preserving 2x2 composite of the first four historical event PNGs.

## Dataset-information figures

The supporting dataset description is in [DATASET_INFORMATION.md](./DATASET_INFORMATION.md), generated from the same saved CSV artifacts.

| Figure | Exact values plotted |
|---|---|
| `dataset_fig1_history_coverage.png` | Calendar start/end dates for Group A US indices, Group B cross-asset, Group C crypto, Group C international, and Group C sectors; red markers show the first and last source-panel crash episode dates. |
| `dataset_fig2_observed_crash_counts.png` | Source-panel episode totals versus `crash == 1` counts: Group A `422 / 12`, Group B `214 / 7`, crypto `167 / 9`, international `396 / 22`, sectors `423 / 12`. |
| `dataset_fig3_v7_crash_onsets_by_year.png` | Unique V7 observed event counts by onset year for `y_N5_p2` (5,604 events, 159 assets) and `y_N15_p2` (2,495 events, 155 assets). |

## Multi-entity TDA showcase

- Report: [MULTI_ENTITY_TDA_SHOWCASE.md](../MULTI_ENTITY_TDA_SHOWCASE.md)
- Figure: `fig10_multi_entity_tda_showcase.png` (also copied to `multi_entity_tda_showcase/fig10_multi_entity_tda_showcase.png`)
- Exact values plotted: saved V3-D R3/budget-0.5 walk-forward cloud-size rows for US4, approximately 30 indices/ETFs, 142 non-stock assets, and 159 all-assets, with ROC AUC, event-recall field, AP/prevalence lift, and the saved lead-time proxy; V5 E2 block means are overlaid as reference points.
- The report also includes a newly executed V5 E2 `W`/`RQ`/5%/logreg check across E8–E10 for `M0`, `M0+TDA_big`, `M0+DYN`, `M0+CLUSTER`, and compact `M0+ALLS36D+CLUSTER`; those exact rows are saved in `data/v5/multi_entity_big_cloud_results.csv`.
