# V7_FINAL_SYNTHESIS

A causal head-to-head ablation of classical CSD, SEMIVAR momentum, peer/cross-index features, scale-normalized TDA, and their combination.

## Design

- Scheme W chronological folds: 2011-2015, 2016-2020, 2021+.
- 25-session purge gap and causal rolling-250-session RQ threshold.
- Alarm-day budget: 5%.
- Horizons: N=5 and N=15, p=2% downside labels.
- GRU and TCN use hidden size 24, sequence length 20, and seeds 42, 43, 44.

V6 supplied 17 SEMIVAR columns (not 18). V5 supplied 36 sampled date-level landscape coordinates; these are merged causally by date.
The four-index Group A restriction applies to sequence models.

## Feature counts

| Set | Features |
|---|---:|
| CLASSICAL_CSD | 9 |
| SEMIVAR_MOM | 26 |
| PEER_CROSS | 35 |
| TDA_ONLY_NORM | 36 |
| GRAND_COMBINED | 71 |

## Ablation results

| Label | Feature set | Model | AP | AUC | Alarm rate | Event recall | Random recall | Excess recall | Lead time | Folds |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| y_N5_p2 | CLASSICAL_CSD | gru | 0.0366 | 0.7056 | 0.1005 | 0.3040 | 0.2738 | 0.0303 | 11.17 | 9 |
| y_N5_p2 | CLASSICAL_CSD | lightgbm | 0.0270 | 0.6552 | 0.0388 | 0.1581 | 0.1131 | 0.0450 | 15.16 | 3 |
| y_N5_p2 | CLASSICAL_CSD | logreg | 0.0292 | 0.6698 | 0.0691 | 0.2344 | 0.1896 | 0.0449 | 13.59 | 3 |
| y_N5_p2 | CLASSICAL_CSD | tcn | 0.0519 | 0.7650 | 0.0812 | 0.3457 | 0.2374 | 0.1083 | 11.23 | 9 |
| y_N5_p2 | GRAND_COMBINED | gru | 0.0628 | 0.8036 | 0.0752 | 0.4328 | 0.2221 | 0.2107 | 9.19 | 9 |
| y_N5_p2 | GRAND_COMBINED | lightgbm | 0.0414 | 0.7196 | 0.0480 | 0.1763 | 0.1335 | 0.0428 | 14.66 | 3 |
| y_N5_p2 | GRAND_COMBINED | logreg | 0.0378 | 0.7414 | 0.0764 | 0.2644 | 0.2065 | 0.0579 | 12.30 | 3 |
| y_N5_p2 | GRAND_COMBINED | tcn | 0.0512 | 0.7673 | 0.0743 | 0.3239 | 0.2197 | 0.1042 | 8.51 | 9 |
| y_N5_p2 | PEER_CROSS | gru | 0.0657 | 0.8095 | 0.0711 | 0.4042 | 0.2122 | 0.1920 | 10.06 | 9 |
| y_N5_p2 | PEER_CROSS | lightgbm | 0.0358 | 0.7275 | 0.0461 | 0.1894 | 0.1275 | 0.0620 | 15.00 | 3 |
| y_N5_p2 | PEER_CROSS | logreg | 0.0400 | 0.7559 | 0.0785 | 0.2690 | 0.2127 | 0.0563 | 12.79 | 3 |
| y_N5_p2 | PEER_CROSS | tcn | 0.0571 | 0.7944 | 0.0774 | 0.3981 | 0.2279 | 0.1702 | 9.13 | 9 |
| y_N5_p2 | SEMIVAR_MOM | gru | 0.0512 | 0.7731 | 0.0782 | 0.3820 | 0.2290 | 0.1530 | 9.89 | 9 |
| y_N5_p2 | SEMIVAR_MOM | lightgbm | 0.0378 | 0.7189 | 0.0457 | 0.1767 | 0.1303 | 0.0464 | 14.76 | 3 |
| y_N5_p2 | SEMIVAR_MOM | logreg | 0.0408 | 0.7358 | 0.0807 | 0.2623 | 0.2160 | 0.0463 | 12.82 | 3 |
| y_N5_p2 | SEMIVAR_MOM | tcn | 0.0552 | 0.7870 | 0.0800 | 0.3773 | 0.2348 | 0.1425 | 9.24 | 9 |
| y_N5_p2 | TDA_ONLY_NORM | gru | 0.0203 | 0.5343 | 0.0574 | 0.0977 | 0.1814 | -0.0837 | 16.18 | 9 |
| y_N5_p2 | TDA_ONLY_NORM | lightgbm | 0.0126 | 0.4909 | 0.0140 | 0.0341 | 0.0424 | -0.0083 | 17.61 | 3 |
| y_N5_p2 | TDA_ONLY_NORM | logreg | 0.0128 | 0.4992 | 0.0325 | 0.1269 | 0.0975 | 0.0293 | 14.06 | 3 |
| y_N5_p2 | TDA_ONLY_NORM | tcn | 0.0179 | 0.4913 | 0.0600 | 0.1347 | 0.1865 | -0.0519 | 13.29 | 9 |
| y_down_N15_p2 | CLASSICAL_CSD | gru | 0.0208 | 0.6025 | 0.0992 | 0.2654 | 0.4724 | -0.2069 | 13.54 | 9 |
| y_down_N15_p2 | CLASSICAL_CSD | lightgbm | 0.0174 | 0.6096 | 0.0347 | 0.1769 | 0.2097 | -0.0328 | 15.98 | 3 |
| y_down_N15_p2 | CLASSICAL_CSD | logreg | 0.0205 | 0.6377 | 0.0711 | 0.3032 | 0.3647 | -0.0615 | 13.80 | 3 |
| y_down_N15_p2 | CLASSICAL_CSD | tcn | 0.0233 | 0.6419 | 0.0786 | 0.3133 | 0.4105 | -0.0972 | 13.88 | 9 |
| y_down_N15_p2 | GRAND_COMBINED | gru | 0.0349 | 0.7125 | 0.0805 | 0.3843 | 0.4263 | -0.0420 | 11.36 | 9 |
| y_down_N15_p2 | GRAND_COMBINED | lightgbm | 0.0233 | 0.6699 | 0.0450 | 0.2222 | 0.2484 | -0.0263 | 15.33 | 3 |
| y_down_N15_p2 | GRAND_COMBINED | logreg | 0.0260 | 0.7143 | 0.0818 | 0.3358 | 0.3985 | -0.0627 | 12.21 | 3 |
| y_down_N15_p2 | GRAND_COMBINED | tcn | 0.0258 | 0.6614 | 0.0748 | 0.3519 | 0.4014 | -0.0495 | 10.69 | 9 |
| y_down_N15_p2 | PEER_CROSS | gru | 0.0315 | 0.6991 | 0.0868 | 0.4383 | 0.4395 | -0.0012 | 11.73 | 9 |
| y_down_N15_p2 | PEER_CROSS | lightgbm | 0.0267 | 0.6931 | 0.0454 | 0.2228 | 0.2502 | -0.0274 | 15.77 | 3 |
| y_down_N15_p2 | PEER_CROSS | logreg | 0.0271 | 0.7257 | 0.0828 | 0.3183 | 0.4031 | -0.0848 | 13.25 | 3 |
| y_down_N15_p2 | PEER_CROSS | tcn | 0.0391 | 0.7274 | 0.0843 | 0.4128 | 0.4293 | -0.0165 | 11.08 | 9 |
| y_down_N15_p2 | SEMIVAR_MOM | gru | 0.0277 | 0.7181 | 0.0862 | 0.3387 | 0.4408 | -0.1020 | 11.19 | 9 |
| y_down_N15_p2 | SEMIVAR_MOM | lightgbm | 0.0266 | 0.6883 | 0.0435 | 0.2015 | 0.2478 | -0.0463 | 15.93 | 3 |
| y_down_N15_p2 | SEMIVAR_MOM | logreg | 0.0267 | 0.7076 | 0.0845 | 0.3210 | 0.4055 | -0.0845 | 13.05 | 3 |
| y_down_N15_p2 | SEMIVAR_MOM | tcn | 0.0297 | 0.7100 | 0.0829 | 0.3495 | 0.4239 | -0.0743 | 11.45 | 9 |
| y_down_N15_p2 | TDA_ONLY_NORM | gru | 0.0202 | 0.5079 | 0.0547 | 0.1998 | 0.3221 | -0.1223 | 13.97 | 9 |
| y_down_N15_p2 | TDA_ONLY_NORM | lightgbm | 0.0124 | 0.4992 | 0.0054 | 0.0531 | 0.0358 | 0.0173 | 18.17 | 3 |
| y_down_N15_p2 | TDA_ONLY_NORM | logreg | 0.0125 | 0.5040 | 0.0255 | 0.2558 | 0.1683 | 0.0875 | 15.41 | 3 |
| y_down_N15_p2 | TDA_ONLY_NORM | tcn | 0.0160 | 0.4857 | 0.0567 | 0.3295 | 0.3352 | -0.0057 | 13.38 | 9 |

## Paired episode bootstrap

| Comparison | Label | Fold | Events | Estimate | 95% CI | p-value |
|---|---|---|---:|---:|---|---:|
| GRAND_COMBINED_vs_PEER_CROSS | y_N5_p2 | 2011_2015 | 743 | -0.0242 | [-0.0485, 0.0054] | 0.1160 |
| GRAND_COMBINED_vs_PEER_CROSS | y_N5_p2 | 2011_2015 | 17 | -0.1176 | [-0.3529, 0.1176] | 0.4480 |
| GRAND_COMBINED_vs_PEER_CROSS | y_N5_p2 | 2011_2015 | 17 | -0.0588 | [-0.3529, 0.1765] | 0.8280 |
| gru_vs_logreg | y_N5_p2 | 2011_2015 | 17 | -0.1765 | [-0.5294, 0.1765] | 0.4500 |
| tcn_vs_logreg | y_N5_p2 | 2011_2015 | 17 | -0.1176 | [-0.2941, 0.0000] | 0.2340 |
| GRAND_COMBINED_vs_PEER_CROSS | y_N5_p2 | 2016_2020 | 1095 | 0.0174 | [0.0000, 0.0338] | 0.0560 |
| GRAND_COMBINED_vs_PEER_CROSS | y_N5_p2 | 2016_2020 | 44 | -0.0227 | [-0.1364, 0.0682] | 0.8440 |
| GRAND_COMBINED_vs_PEER_CROSS | y_N5_p2 | 2016_2020 | 44 | -0.0455 | [-0.1591, 0.0682] | 0.5800 |
| gru_vs_logreg | y_N5_p2 | 2016_2020 | 44 | 0.0455 | [-0.0909, 0.1818] | 0.5960 |
| tcn_vs_logreg | y_N5_p2 | 2016_2020 | 44 | 0.0000 | [-0.1364, 0.1364] | 1.0000 |
| GRAND_COMBINED_vs_PEER_CROSS | y_N5_p2 | 2021_plus | 1297 | -0.0108 | [-0.0308, 0.0085] | 0.2960 |
| GRAND_COMBINED_vs_PEER_CROSS | y_N5_p2 | 2021_plus | 26 | 0.0385 | [-0.1154, 0.1923] | 0.7940 |
| GRAND_COMBINED_vs_PEER_CROSS | y_N5_p2 | 2021_plus | 26 | -0.0769 | [-0.1923, 0.0000] | 0.2140 |
| gru_vs_logreg | y_N5_p2 | 2021_plus | 26 | 0.1923 | [0.0385, 0.3846] | 0.0400 |
| tcn_vs_logreg | y_N5_p2 | 2021_plus | 26 | -0.0385 | [-0.1923, 0.1154] | 0.8400 |
| GRAND_COMBINED_vs_PEER_CROSS | y_down_N15_p2 | 2011_2015 | 320 | -0.0031 | [-0.0406, 0.0344] | 0.8900 |
| GRAND_COMBINED_vs_PEER_CROSS | y_down_N15_p2 | 2011_2015 | 8 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| GRAND_COMBINED_vs_PEER_CROSS | y_down_N15_p2 | 2011_2015 | 8 | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| gru_vs_logreg | y_down_N15_p2 | 2011_2015 | 8 | -0.2500 | [-0.6250, 0.0000] | 0.2000 |
| tcn_vs_logreg | y_down_N15_p2 | 2011_2015 | 8 | -0.2500 | [-0.6250, 0.0000] | 0.2300 |
| GRAND_COMBINED_vs_PEER_CROSS | y_down_N15_p2 | 2016_2020 | 491 | -0.0122 | [-0.0407, 0.0143] | 0.4020 |
| GRAND_COMBINED_vs_PEER_CROSS | y_down_N15_p2 | 2016_2020 | 15 | -0.0667 | [-0.2667, 0.1333] | 0.7960 |
| GRAND_COMBINED_vs_PEER_CROSS | y_down_N15_p2 | 2016_2020 | 15 | -0.2667 | [-0.5333, 0.0000] | 0.1080 |
| gru_vs_logreg | y_down_N15_p2 | 2016_2020 | 15 | 0.0000 | [-0.2000, 0.2000] | 1.0000 |
| tcn_vs_logreg | y_down_N15_p2 | 2016_2020 | 15 | -0.1333 | [-0.4000, 0.1333] | 0.4820 |
| GRAND_COMBINED_vs_PEER_CROSS | y_down_N15_p2 | 2021_plus | 648 | 0.0093 | [-0.0201, 0.0402] | 0.5720 |
| GRAND_COMBINED_vs_PEER_CROSS | y_down_N15_p2 | 2021_plus | 22 | -0.0909 | [-0.3182, 0.1364] | 0.5400 |
| GRAND_COMBINED_vs_PEER_CROSS | y_down_N15_p2 | 2021_plus | 22 | -0.0909 | [-0.2727, 0.0909] | 0.4560 |
| gru_vs_logreg | y_down_N15_p2 | 2021_plus | 22 | 0.0455 | [-0.1364, 0.2273] | 0.8320 |
| tcn_vs_logreg | y_down_N15_p2 | 2021_plus | 22 | 0.0000 | [-0.1364, 0.1364] | 1.0000 |

## Interpretation

Positive excess recall means the model caught more crash episodes than the same-budget random baseline. A bootstrap interval crossing zero is inconclusive; it is not evidence of no effect.

Outputs: `data/v7_final/v7_ablation_results.csv`, `data/v7_final/v7_bootstrap_significance.csv`, and `data/v7_final/scores/`.