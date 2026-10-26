# Classical vs topological vs peer/SEMIVAR performance

## Scope and denominator

The underlying panel spans 1927-12-30 through 2026-10-23 and contains 159 assets. The observed-crash tables below use all available asset label files over that full period.

Model results are reported with their actual coverage. V5 completed cells use the full 159-asset panel and V5 era definitions. V7 is still in progress: its saved model scores currently cover only the completed modern folds, not the entire historical period. Therefore V7 detection counts must not be interpreted as full-history model detection rates.

## Observed crash population: full entities and full period

| Label | Events | Assets |
|---|---:|---:|
| y_N15_p2 | 2,495 | 155 |
| y_N5_p2 | 5,604 | 159 |

These are observed label episodes, not model detections. An episode is a maximal positive run after merging gaps of up to five sessions.

## V5 full-era comparative results

The table uses saved 5% operating points. Scheme B rows use the V5 within-era RE rule; Scheme W rows use the causal RQ rule when present. Results are means over completed era cells only.

| Source | Scheme | Feature set | Model | Cells | Eras | AP | AUC | Alarm rate | Event recall | Random recall | Excess recall |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V5_E1 | W | B1 | gbm | 7 | 7 | 0.0414 | 0.6459 | 0.0364 | 0.1753 | 0.2250 | -0.0498 |
| V5_E1 | W | B1 | logreg | 7 | 7 | 0.0501 | 0.6406 | 0.0750 | 0.3014 | 0.3893 | -0.0879 |
| V5_E1 | W | B2 | gbm | 7 | 7 | 0.0417 | 0.6577 | 0.0478 | 0.1977 | 0.2688 | -0.0710 |
| V5_E1 | W | B2 | logreg | 5 | 5 | 0.0502 | 0.6628 | 0.0877 | 0.2390 | 0.4056 | -0.1665 |
| V5_E2 | W | B3 | gbm | 7 | 7 | 0.0493 | 0.6816 | 0.0470 | 0.2129 | 0.2697 | -0.0568 |
| V5_E2 | W | B3 | logreg | 7 | 7 | 0.0505 | 0.6787 | 0.0767 | 0.2922 | 0.3922 | -0.1000 |
| V5_E2 | W | M0 | gbm | 7 | 7 | 0.0308 | 0.6186 | 0.0387 | 0.1791 | 0.2336 | -0.0545 |
| V5_E2 | W | M0 | logreg | 7 | 7 | 0.0507 | 0.6516 | 0.0690 | 0.3184 | 0.3696 | -0.0512 |
| V5_E2 | W | M0+TDA_small | logreg | 1 | 1 | 0.0466 | 0.5943 | 0.0005 | 0.0000 | 0.0017 | -0.0017 |

V5 feature interpretation:

- `B1`/`F0` are classical/CSD-oriented baselines.
- `B2`/`F1` add price momentum, volatility, relative strength, and peer-style cross-asset features.
- `B3`/`F2` add broad market-state, breadth, dispersion, correlation, eigenvalue, cluster, and macro-state variables.
- `M0+TDA_small` is the V5 topology augmentation; its E2 result is incomplete and must not be compared as a finished full-era model.

## V7 ablation results currently saved

| Label | Feature set | Model | Cells | Folds | AP | AUC | Alarm rate | Event recall | Random recall | Excess recall | Lead time |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| y_N5_p2 | CLASSICAL_CSD | gru | 9 | 3 | 0.0366 | 0.7056 | 0.1005 | 0.3040 | 0.2738 | 0.0303 | 11.17 |
| y_N5_p2 | CLASSICAL_CSD | lightgbm | 3 | 3 | 0.0270 | 0.6552 | 0.0388 | 0.1581 | 0.1131 | 0.0450 | 15.16 |
| y_N5_p2 | CLASSICAL_CSD | logreg | 3 | 3 | 0.0292 | 0.6698 | 0.0691 | 0.2344 | 0.1896 | 0.0449 | 13.59 |
| y_N5_p2 | CLASSICAL_CSD | tcn | 9 | 3 | 0.0519 | 0.7650 | 0.0812 | 0.3457 | 0.2374 | 0.1083 | 11.23 |
| y_N5_p2 | GRAND_COMBINED | gru | 9 | 3 | 0.0628 | 0.8036 | 0.0752 | 0.4328 | 0.2221 | 0.2107 | 9.19 |
| y_N5_p2 | GRAND_COMBINED | lightgbm | 3 | 3 | 0.0414 | 0.7196 | 0.0480 | 0.1763 | 0.1335 | 0.0428 | 14.66 |
| y_N5_p2 | GRAND_COMBINED | logreg | 3 | 3 | 0.0378 | 0.7414 | 0.0764 | 0.2644 | 0.2065 | 0.0579 | 12.30 |
| y_N5_p2 | GRAND_COMBINED | tcn | 9 | 3 | 0.0512 | 0.7673 | 0.0743 | 0.3239 | 0.2197 | 0.1042 | 8.51 |
| y_N5_p2 | PEER_CROSS | gru | 9 | 3 | 0.0657 | 0.8095 | 0.0711 | 0.4042 | 0.2122 | 0.1920 | 10.06 |
| y_N5_p2 | PEER_CROSS | lightgbm | 3 | 3 | 0.0358 | 0.7275 | 0.0461 | 0.1894 | 0.1275 | 0.0620 | 15.00 |
| y_N5_p2 | PEER_CROSS | logreg | 3 | 3 | 0.0400 | 0.7559 | 0.0785 | 0.2690 | 0.2127 | 0.0563 | 12.79 |
| y_N5_p2 | PEER_CROSS | tcn | 9 | 3 | 0.0571 | 0.7944 | 0.0774 | 0.3981 | 0.2279 | 0.1702 | 9.13 |
| y_N5_p2 | SEMIVAR_MOM | gru | 9 | 3 | 0.0512 | 0.7731 | 0.0782 | 0.3820 | 0.2290 | 0.1530 | 9.89 |
| y_N5_p2 | SEMIVAR_MOM | lightgbm | 3 | 3 | 0.0378 | 0.7189 | 0.0457 | 0.1767 | 0.1303 | 0.0464 | 14.76 |
| y_N5_p2 | SEMIVAR_MOM | logreg | 3 | 3 | 0.0408 | 0.7358 | 0.0807 | 0.2623 | 0.2160 | 0.0463 | 12.82 |
| y_N5_p2 | SEMIVAR_MOM | tcn | 9 | 3 | 0.0552 | 0.7870 | 0.0800 | 0.3773 | 0.2348 | 0.1425 | 9.24 |
| y_N5_p2 | TDA_ONLY_NORM | gru | 9 | 3 | 0.0203 | 0.5343 | 0.0574 | 0.0977 | 0.1814 | -0.0837 | 16.18 |
| y_N5_p2 | TDA_ONLY_NORM | lightgbm | 3 | 3 | 0.0126 | 0.4909 | 0.0140 | 0.0341 | 0.0424 | -0.0083 | 17.61 |
| y_N5_p2 | TDA_ONLY_NORM | logreg | 3 | 3 | 0.0128 | 0.4992 | 0.0325 | 0.1269 | 0.0975 | 0.0293 | 14.06 |
| y_N5_p2 | TDA_ONLY_NORM | tcn | 9 | 3 | 0.0179 | 0.4913 | 0.0600 | 0.1347 | 0.1865 | -0.0519 | 13.29 |
| y_down_N15_p2 | CLASSICAL_CSD | gru | 3 | 1 | 0.0187 | 0.6292 | 0.0998 | 0.0417 | 0.5013 | -0.4596 | 14.86 |
| y_down_N15_p2 | CLASSICAL_CSD | lightgbm | 1 | 1 | 0.0127 | 0.6108 | 0.0344 | 0.0659 | 0.2005 | -0.1346 | 16.37 |
| y_down_N15_p2 | CLASSICAL_CSD | logreg | 1 | 1 | 0.0150 | 0.6494 | 0.0734 | 0.1744 | 0.3430 | -0.1686 | 14.60 |
| y_down_N15_p2 | CLASSICAL_CSD | tcn | 3 | 1 | 0.0193 | 0.6931 | 0.0726 | 0.0833 | 0.4096 | -0.3262 | 16.19 |

V7 feature sets are progressive ablations:
`CLASSICAL_CSD`, `SEMIVAR_MOM`, `PEER_CROSS`, `TDA_ONLY_NORM`, and `GRAND_COMBINED`. The V7 TDA features are the saved 36-coordinate scale-normalized landscape block merged by date. V7 deep models use Group A only; tabular models use the unified panel.

## Which observed crashes were detected?

The complete event-level model detection records are in `data/v7_final/crash_stats/model_event_detections.csv`. The summary below counts only saved score files.

| Label | Feature set | Model | Fold | Events covered | Detected | Detection rate |
|---|---|---|---|---:|---:|---:|
| y_N5_p2 | CLASSICAL_CSD | gru | 2011_2015 | 15 | 5 | 0.3333 |
| y_N5_p2 | CLASSICAL_CSD | gru | 2011_2015 | 15 | 5 | 0.3333 |
| y_N5_p2 | CLASSICAL_CSD | gru | 2011_2015 | 15 | 4 | 0.2667 |
| y_N5_p2 | CLASSICAL_CSD | gru | 2016_2020 | 30 | 11 | 0.3667 |
| y_N5_p2 | CLASSICAL_CSD | gru | 2016_2020 | 30 | 10 | 0.3333 |
| y_N5_p2 | CLASSICAL_CSD | gru | 2016_2020 | 30 | 9 | 0.3000 |
| y_N5_p2 | CLASSICAL_CSD | gru | 2021_plus | 27 | 8 | 0.2963 |
| y_N5_p2 | CLASSICAL_CSD | gru | 2021_plus | 27 | 6 | 0.2222 |
| y_N5_p2 | CLASSICAL_CSD | gru | 2021_plus | 27 | 7 | 0.2593 |
| y_N5_p2 | CLASSICAL_CSD | lightgbm | 2011_2015 | 612 | 74 | 0.1209 |
| y_N5_p2 | CLASSICAL_CSD | lightgbm | 2016_2020 | 795 | 176 | 0.2214 |
| y_N5_p2 | CLASSICAL_CSD | lightgbm | 2021_plus | 1080 | 139 | 0.1287 |
| y_N5_p2 | CLASSICAL_CSD | logreg | 2011_2015 | 612 | 122 | 0.1993 |
| y_N5_p2 | CLASSICAL_CSD | logreg | 2016_2020 | 795 | 247 | 0.3107 |
| y_N5_p2 | CLASSICAL_CSD | logreg | 2021_plus | 1080 | 210 | 0.1944 |
| y_N5_p2 | CLASSICAL_CSD | tcn | 2011_2015 | 15 | 6 | 0.4000 |
| y_N5_p2 | CLASSICAL_CSD | tcn | 2011_2015 | 15 | 6 | 0.4000 |
| y_N5_p2 | CLASSICAL_CSD | tcn | 2011_2015 | 15 | 6 | 0.4000 |
| y_N5_p2 | CLASSICAL_CSD | tcn | 2016_2020 | 30 | 11 | 0.3667 |
| y_N5_p2 | CLASSICAL_CSD | tcn | 2016_2020 | 30 | 12 | 0.4000 |
| y_N5_p2 | CLASSICAL_CSD | tcn | 2016_2020 | 30 | 11 | 0.3667 |
| y_N5_p2 | CLASSICAL_CSD | tcn | 2021_plus | 27 | 9 | 0.3333 |
| y_N5_p2 | CLASSICAL_CSD | tcn | 2021_plus | 27 | 6 | 0.2222 |
| y_N5_p2 | CLASSICAL_CSD | tcn | 2021_plus | 27 | 3 | 0.1111 |
| y_N5_p2 | GRAND_COMBINED | gru | 2011_2015 | 15 | 8 | 0.5333 |
| y_N5_p2 | GRAND_COMBINED | gru | 2011_2015 | 15 | 12 | 0.8000 |
| y_N5_p2 | GRAND_COMBINED | gru | 2011_2015 | 15 | 8 | 0.5333 |
| y_N5_p2 | GRAND_COMBINED | gru | 2016_2020 | 30 | 10 | 0.3333 |
| y_N5_p2 | GRAND_COMBINED | gru | 2016_2020 | 30 | 8 | 0.2667 |
| y_N5_p2 | GRAND_COMBINED | gru | 2016_2020 | 30 | 11 | 0.3667 |
| y_N5_p2 | GRAND_COMBINED | lightgbm | 2011_2015 | 612 | 95 | 0.1552 |
| y_N5_p2 | GRAND_COMBINED | lightgbm | 2016_2020 | 795 | 177 | 0.2226 |
| y_N5_p2 | GRAND_COMBINED | logreg | 2011_2015 | 612 | 170 | 0.2778 |
| y_N5_p2 | GRAND_COMBINED | logreg | 2016_2020 | 795 | 257 | 0.3233 |
| y_N5_p2 | GRAND_COMBINED | tcn | 2011_2015 | 15 | 7 | 0.4667 |
| y_N5_p2 | GRAND_COMBINED | tcn | 2011_2015 | 15 | 8 | 0.5333 |
| y_N5_p2 | GRAND_COMBINED | tcn | 2011_2015 | 15 | 8 | 0.5333 |
| y_N5_p2 | GRAND_COMBINED | tcn | 2016_2020 | 30 | 8 | 0.2667 |
| y_N5_p2 | GRAND_COMBINED | tcn | 2016_2020 | 30 | 8 | 0.2667 |
| y_N5_p2 | GRAND_COMBINED | tcn | 2016_2020 | 30 | 7 | 0.2333 |
| y_N5_p2 | PEER_CROSS | gru | 2011_2015 | 15 | 9 | 0.6000 |
| y_N5_p2 | PEER_CROSS | gru | 2011_2015 | 15 | 5 | 0.3333 |
| y_N5_p2 | PEER_CROSS | gru | 2011_2015 | 15 | 9 | 0.6000 |
| y_N5_p2 | PEER_CROSS | gru | 2016_2020 | 30 | 11 | 0.3667 |
| y_N5_p2 | PEER_CROSS | gru | 2016_2020 | 30 | 10 | 0.3333 |
| y_N5_p2 | PEER_CROSS | gru | 2016_2020 | 30 | 11 | 0.3667 |
| y_N5_p2 | PEER_CROSS | gru | 2021_plus | 27 | 9 | 0.3333 |
| y_N5_p2 | PEER_CROSS | gru | 2021_plus | 27 | 9 | 0.3333 |
| y_N5_p2 | PEER_CROSS | gru | 2021_plus | 27 | 8 | 0.2963 |
| y_N5_p2 | PEER_CROSS | lightgbm | 2011_2015 | 612 | 116 | 0.1895 |
| y_N5_p2 | PEER_CROSS | lightgbm | 2016_2020 | 795 | 169 | 0.2126 |
| y_N5_p2 | PEER_CROSS | lightgbm | 2021_plus | 1080 | 200 | 0.1852 |
| y_N5_p2 | PEER_CROSS | logreg | 2011_2015 | 612 | 173 | 0.2827 |
| y_N5_p2 | PEER_CROSS | logreg | 2016_2020 | 795 | 256 | 0.3220 |
| y_N5_p2 | PEER_CROSS | logreg | 2021_plus | 1080 | 246 | 0.2278 |
| y_N5_p2 | PEER_CROSS | tcn | 2011_2015 | 15 | 8 | 0.5333 |
| y_N5_p2 | PEER_CROSS | tcn | 2011_2015 | 15 | 9 | 0.6000 |
| y_N5_p2 | PEER_CROSS | tcn | 2011_2015 | 15 | 10 | 0.6667 |
| y_N5_p2 | PEER_CROSS | tcn | 2016_2020 | 30 | 9 | 0.3000 |
| y_N5_p2 | PEER_CROSS | tcn | 2016_2020 | 30 | 10 | 0.3333 |
| y_N5_p2 | PEER_CROSS | tcn | 2016_2020 | 30 | 12 | 0.4000 |
| y_N5_p2 | PEER_CROSS | tcn | 2021_plus | 27 | 6 | 0.2222 |
| y_N5_p2 | PEER_CROSS | tcn | 2021_plus | 27 | 8 | 0.2963 |
| y_N5_p2 | PEER_CROSS | tcn | 2021_plus | 27 | 4 | 0.1481 |
| y_N5_p2 | SEMIVAR_MOM | gru | 2011_2015 | 15 | 10 | 0.6667 |
| y_N5_p2 | SEMIVAR_MOM | gru | 2011_2015 | 15 | 6 | 0.4000 |
| y_N5_p2 | SEMIVAR_MOM | gru | 2011_2015 | 15 | 6 | 0.4000 |
| y_N5_p2 | SEMIVAR_MOM | gru | 2016_2020 | 30 | 5 | 0.1667 |
| y_N5_p2 | SEMIVAR_MOM | gru | 2016_2020 | 30 | 10 | 0.3333 |
| y_N5_p2 | SEMIVAR_MOM | gru | 2016_2020 | 30 | 11 | 0.3667 |
| y_N5_p2 | SEMIVAR_MOM | gru | 2021_plus | 27 | 8 | 0.2963 |
| y_N5_p2 | SEMIVAR_MOM | gru | 2021_plus | 27 | 10 | 0.3704 |
| y_N5_p2 | SEMIVAR_MOM | gru | 2021_plus | 27 | 9 | 0.3333 |
| y_N5_p2 | SEMIVAR_MOM | lightgbm | 2011_2015 | 612 | 98 | 0.1601 |
| y_N5_p2 | SEMIVAR_MOM | lightgbm | 2016_2020 | 795 | 178 | 0.2239 |
| y_N5_p2 | SEMIVAR_MOM | lightgbm | 2021_plus | 1080 | 165 | 0.1528 |
| y_N5_p2 | SEMIVAR_MOM | logreg | 2011_2015 | 612 | 155 | 0.2533 |
| y_N5_p2 | SEMIVAR_MOM | logreg | 2016_2020 | 795 | 259 | 0.3258 |
| y_N5_p2 | SEMIVAR_MOM | logreg | 2021_plus | 1080 | 246 | 0.2278 |
| y_N5_p2 | SEMIVAR_MOM | tcn | 2011_2015 | 15 | 8 | 0.5333 |
| y_N5_p2 | SEMIVAR_MOM | tcn | 2011_2015 | 15 | 4 | 0.2667 |
| y_N5_p2 | SEMIVAR_MOM | tcn | 2011_2015 | 15 | 7 | 0.4667 |
| y_N5_p2 | SEMIVAR_MOM | tcn | 2016_2020 | 30 | 10 | 0.3333 |
| y_N5_p2 | SEMIVAR_MOM | tcn | 2016_2020 | 30 | 10 | 0.3333 |
| y_N5_p2 | SEMIVAR_MOM | tcn | 2016_2020 | 30 | 11 | 0.3667 |
| y_N5_p2 | SEMIVAR_MOM | tcn | 2021_plus | 27 | 11 | 0.4074 |
| y_N5_p2 | SEMIVAR_MOM | tcn | 2021_plus | 27 | 6 | 0.2222 |
| y_N5_p2 | SEMIVAR_MOM | tcn | 2021_plus | 27 | 11 | 0.4074 |
| y_N5_p2 | TDA_ONLY_NORM | gru | 2011_2015 | 15 | 2 | 0.1333 |
| y_N5_p2 | TDA_ONLY_NORM | gru | 2011_2015 | 15 | 0 | 0.0000 |
| y_N5_p2 | TDA_ONLY_NORM | gru | 2011_2015 | 15 | 0 | 0.0000 |
| y_N5_p2 | TDA_ONLY_NORM | gru | 2016_2020 | 30 | 2 | 0.0667 |
| y_N5_p2 | TDA_ONLY_NORM | gru | 2016_2020 | 30 | 7 | 0.2333 |
| y_N5_p2 | TDA_ONLY_NORM | gru | 2016_2020 | 30 | 6 | 0.2000 |
| y_N5_p2 | TDA_ONLY_NORM | lightgbm | 2011_2015 | 612 | 16 | 0.0261 |
| y_N5_p2 | TDA_ONLY_NORM | lightgbm | 2016_2020 | 795 | 39 | 0.0491 |
| y_N5_p2 | TDA_ONLY_NORM | logreg | 2011_2015 | 612 | 40 | 0.0654 |
| y_N5_p2 | TDA_ONLY_NORM | logreg | 2016_2020 | 795 | 203 | 0.2553 |
| y_N5_p2 | TDA_ONLY_NORM | tcn | 2011_2015 | 15 | 0 | 0.0000 |
| y_N5_p2 | TDA_ONLY_NORM | tcn | 2011_2015 | 15 | 0 | 0.0000 |
| y_N5_p2 | TDA_ONLY_NORM | tcn | 2011_2015 | 15 | 2 | 0.1333 |
| y_N5_p2 | TDA_ONLY_NORM | tcn | 2016_2020 | 30 | 6 | 0.2000 |
| y_N5_p2 | TDA_ONLY_NORM | tcn | 2016_2020 | 30 | 7 | 0.2333 |
| y_N5_p2 | TDA_ONLY_NORM | tcn | 2016_2020 | 30 | 11 | 0.3667 |

## Scientific conclusion

The full-period observed-crash population is large and is measured over all 159 assets. The completed V5 results show that classical baselines, peer/market-state features, and topology are not interchangeable feature families. The completed V7 cells currently provide a focused modern-fold ablation, but they do not yet provide full-period model detection counts.

A final claim that one family detects crashes across the entire 1927-2026 record requires completing the historical V7 folds and then regenerating the detection tables. Until then, this report intentionally separates full-history crash counts from partial saved-model detection coverage.

Machine-readable tables:

- `data/v7_final/comparative/v5_comparative_metrics.csv`
- `data/v7_final/comparative/v7_comparative_metrics.csv`
- `data/v7_final/crash_stats/observed_crash_events.csv`
- `data/v7_final/crash_stats/model_event_detections.csv`