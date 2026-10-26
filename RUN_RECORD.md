# Run record — every benchmark run, individually

One row per (level × dataset × mode × model × fold). Machine-readable,
full-metric versions: `data/benchmark/run_record_v2.csv` (800 rows),
`data/benchmark/run_record_v1.csv` (420 rows). Per-ticker breakdown rows
(5,160 v2 + 3,494 v1 raw rows) stay in `all_results.csv` /
`all_results_v1.csv`. Metric = test average precision (15-day horizon),
means are over fold columns of the same row.

## v2 accounting (800 planned runs)

| status | runs |
|---|---:|
| done | 696 |
| not_run | 64 |
| skipped_insufficient_data | 40 |

## v1 accounting (420 planned runs)

| status | runs |
|---|---:|
| done | 406 |
| not_run | 14 |
## v2: status detail (runs per dataset)
status                      done  not_run  skipped_insufficient_data
level  dataset                                                      
group  crypto                  0        0                         40
       energy                120        0                          0
       financials            120        0                          0
       group_a_us_indices    120        0                          0
       group_b_cross_asset   120        0                          0
       international         120        0                          0
       tech                   40        0                          0
ticker ALL                    56       64                          0
## v1: status detail (runs per dataset)
status                      done  not_run
level  dataset                           
group  energy                 63        0
       financials             63        0
       group_a_us_indices     63        0
       group_b_cross_asset    63        0
       international          63        0
       tech                   42        0
ticker ALL                    49       14
## v2 — every run (AP by fold)

**crypto · group-level · mode `classical`**

| model | fold_proportional | mean AP | status |
|---|---|---|---|
| attn_bilstm | - | - | skipped_insufficient_data |
| bilstm | - | - | skipped_insufficient_data |
| cnn | - | - | skipped_insufficient_data |
| gru | - | - | skipped_insufficient_data |
| logreg | - | - | skipped_insufficient_data |
| lstm | - | - | skipped_insufficient_data |
| tcn | - | - | skipped_insufficient_data |
| transformer | - | - | skipped_insufficient_data |

**crypto · group-level · mode `combined`**

| model | fold_proportional | mean AP | status |
|---|---|---|---|
| attn_bilstm | - | - | skipped_insufficient_data |
| bilstm | - | - | skipped_insufficient_data |
| cnn | - | - | skipped_insufficient_data |
| gru | - | - | skipped_insufficient_data |
| logreg | - | - | skipped_insufficient_data |
| lstm | - | - | skipped_insufficient_data |
| tcn | - | - | skipped_insufficient_data |
| transformer | - | - | skipped_insufficient_data |

**crypto · group-level · mode `full`**

| model | fold_proportional | mean AP | status |
|---|---|---|---|
| attn_bilstm | - | - | skipped_insufficient_data |
| bilstm | - | - | skipped_insufficient_data |
| cnn | - | - | skipped_insufficient_data |
| gru | - | - | skipped_insufficient_data |
| logreg | - | - | skipped_insufficient_data |
| lstm | - | - | skipped_insufficient_data |
| tcn | - | - | skipped_insufficient_data |
| transformer | - | - | skipped_insufficient_data |

**crypto · group-level · mode `landscape`**

| model | fold_proportional | mean AP | status |
|---|---|---|---|
| attn_bilstm | - | - | skipped_insufficient_data |
| bilstm | - | - | skipped_insufficient_data |
| cnn | - | - | skipped_insufficient_data |
| gru | - | - | skipped_insufficient_data |
| logreg | - | - | skipped_insufficient_data |
| lstm | - | - | skipped_insufficient_data |
| tcn | - | - | skipped_insufficient_data |
| transformer | - | - | skipped_insufficient_data |

**crypto · group-level · mode `tda`**

| model | fold_proportional | mean AP | status |
|---|---|---|---|
| attn_bilstm | - | - | skipped_insufficient_data |
| bilstm | - | - | skipped_insufficient_data |
| cnn | - | - | skipped_insufficient_data |
| gru | - | - | skipped_insufficient_data |
| logreg | - | - | skipped_insufficient_data |
| lstm | - | - | skipped_insufficient_data |
| tcn | - | - | skipped_insufficient_data |
| transformer | - | - | skipped_insufficient_data |

**energy · group-level · mode `classical`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0555 | 0.0588 | 0.0407 | 0.0517 | done |
| bilstm | 0.0532 | 0.0555 | 0.0257 | 0.0448 | done |
| cnn | 0.0790 | 0.0539 | 0.0432 | 0.0587 | done |
| gru | 0.0360 | 0.0618 | 0.0372 | 0.0450 | done |
| logreg | 0.0374 | 0.0679 | 0.0474 | 0.0509 | done |
| lstm | 0.0310 | 0.0546 | 0.0343 | 0.0400 | done |
| tcn | 0.0290 | 0.0622 | 0.0379 | 0.0430 | done |
| transformer | 0.0815 | 0.0586 | 0.0476 | 0.0626 | done |

**energy · group-level · mode `combined`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.1107 | 0.0680 | 0.0793 | 0.0860 | done |
| bilstm | 0.0425 | 0.0553 | 0.0541 | 0.0507 | done |
| cnn | 0.0638 | 0.0749 | 0.0747 | 0.0711 | done |
| gru | 0.0514 | 0.0596 | 0.0424 | 0.0511 | done |
| logreg | 0.0369 | 0.0830 | 0.0347 | 0.0515 | done |
| lstm | 0.0362 | 0.0572 | 0.0475 | 0.0470 | done |
| tcn | 0.0519 | 0.0578 | 0.0432 | 0.0510 | done |
| transformer | 0.0489 | 0.0586 | 0.0386 | 0.0487 | done |

**energy · group-level · mode `full`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0363 | 0.0656 | 0.0312 | 0.0443 | done |
| bilstm | 0.0407 | 0.0557 | 0.0538 | 0.0501 | done |
| cnn | 0.0860 | 0.0680 | 0.0425 | 0.0655 | done |
| gru | 0.0610 | 0.0523 | 0.0581 | 0.0571 | done |
| logreg | 0.0292 | 0.0605 | 0.0347 | 0.0415 | done |
| lstm | 0.0970 | 0.0572 | 0.0980 | 0.0841 | done |
| tcn | 0.0466 | 0.0495 | 0.0646 | 0.0535 | done |
| transformer | 0.1579 | 0.0734 | 0.0438 | 0.0917 | done |

**energy · group-level · mode `landscape`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.1130 | 0.1390 | 0.0337 | 0.0952 | done |
| bilstm | 0.0563 | 0.0608 | 0.0413 | 0.0528 | done |
| cnn | 0.0754 | 0.1267 | 0.0428 | 0.0816 | done |
| gru | 0.0557 | 0.0655 | 0.0437 | 0.0549 | done |
| logreg | 0.0655 | 0.0562 | 0.0394 | 0.0537 | done |
| lstm | 0.0553 | 0.0620 | 0.0369 | 0.0514 | done |
| tcn | 0.1374 | 0.1023 | 0.0516 | 0.0971 | done |
| transformer | 0.0842 | 0.0789 | 0.0415 | 0.0682 | done |

**energy · group-level · mode `tda`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0817 | 0.0920 | 0.0292 | 0.0676 | done |
| bilstm | 0.0456 | 0.0852 | 0.0303 | 0.0537 | done |
| cnn | 0.1030 | 0.0850 | 0.0313 | 0.0731 | done |
| gru | 0.0614 | 0.0801 | 0.0385 | 0.0600 | done |
| logreg | 0.0397 | 0.0734 | 0.0326 | 0.0486 | done |
| lstm | 0.0578 | 0.0726 | 0.0302 | 0.0536 | done |
| tcn | 0.0699 | 0.0817 | 0.0397 | 0.0637 | done |
| transformer | 0.0692 | 0.0893 | 0.0273 | 0.0619 | done |

**financials · group-level · mode `classical`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.1026 | 0.0269 | 0.0437 | 0.0577 | done |
| bilstm | 0.0978 | 0.0315 | 0.0252 | 0.0515 | done |
| cnn | 0.0548 | 0.0302 | 0.0410 | 0.0420 | done |
| gru | 0.0954 | 0.0254 | 0.0154 | 0.0454 | done |
| logreg | 0.0702 | 0.0308 | 0.0135 | 0.0382 | done |
| lstm | 0.1098 | 0.0232 | 0.0456 | 0.0595 | done |
| tcn | 0.0997 | 0.0260 | 0.0114 | 0.0457 | done |
| transformer | 0.0531 | 0.0221 | 0.0250 | 0.0334 | done |

**financials · group-level · mode `combined`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0820 | 0.0328 | 0.0159 | 0.0436 | done |
| bilstm | 0.0904 | 0.0378 | 0.0367 | 0.0550 | done |
| cnn | 0.1192 | 0.0326 | 0.0127 | 0.0548 | done |
| gru | 0.0721 | 0.0331 | 0.0252 | 0.0435 | done |
| logreg | 0.0391 | 0.0477 | 0.0134 | 0.0334 | done |
| lstm | 0.1055 | 0.0294 | 0.0315 | 0.0555 | done |
| tcn | 0.0786 | 0.0338 | 0.0256 | 0.0460 | done |
| transformer | 0.0929 | 0.0314 | 0.0273 | 0.0506 | done |

**financials · group-level · mode `full`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0216 | 0.0260 | 0.0178 | 0.0218 | done |
| bilstm | 0.0390 | 0.0367 | 0.0424 | 0.0393 | done |
| cnn | 0.0210 | 0.0355 | 0.0231 | 0.0265 | done |
| gru | 0.0259 | 0.0332 | 0.0214 | 0.0269 | done |
| logreg | 0.0169 | 0.0491 | 0.0133 | 0.0264 | done |
| lstm | 0.0377 | 0.0249 | 0.0319 | 0.0315 | done |
| tcn | 0.0336 | 0.0393 | 0.0374 | 0.0368 | done |
| transformer | 0.0553 | 0.0283 | 0.0155 | 0.0330 | done |

**financials · group-level · mode `landscape`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0229 | 0.0310 | 0.0199 | 0.0246 | done |
| bilstm | 0.0161 | 0.0263 | 0.0184 | 0.0203 | done |
| cnn | 0.0148 | 0.0644 | 0.0129 | 0.0307 | done |
| gru | 0.0150 | 0.0465 | 0.0144 | 0.0253 | done |
| logreg | 0.0137 | 0.0372 | 0.0200 | 0.0237 | done |
| lstm | 0.0199 | 0.0359 | 0.0185 | 0.0248 | done |
| tcn | 0.0206 | 0.0262 | 0.0120 | 0.0196 | done |
| transformer | 0.0166 | 0.0290 | 0.0193 | 0.0217 | done |

**financials · group-level · mode `tda`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0329 | 0.0350 | 0.0180 | 0.0286 | done |
| bilstm | 0.0335 | 0.0266 | 0.0174 | 0.0258 | done |
| cnn | 0.0347 | 0.0280 | 0.0124 | 0.0250 | done |
| gru | 0.0315 | 0.0284 | 0.0184 | 0.0261 | done |
| logreg | 0.0349 | 0.0353 | 0.0132 | 0.0278 | done |
| lstm | 0.0346 | 0.0280 | 0.0178 | 0.0268 | done |
| tcn | 0.0382 | 0.0278 | 0.0191 | 0.0284 | done |
| transformer | 0.0353 | 0.0367 | 0.0163 | 0.0294 | done |

**group_a_us_indices · group-level · mode `classical`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0135 | 0.0167 | 0.0217 | 0.0173 | done |
| bilstm | 0.0071 | 0.0170 | 0.0483 | 0.0241 | done |
| cnn | 0.0190 | 0.0208 | 0.0161 | 0.0186 | done |
| gru | 0.0075 | 0.0268 | 0.1001 | 0.0448 | done |
| logreg | 0.0097 | 0.0278 | 0.0069 | 0.0148 | done |
| lstm | 0.0068 | 0.0199 | 0.0611 | 0.0292 | done |
| tcn | 0.0076 | 0.0275 | 0.0259 | 0.0204 | done |
| transformer | 0.0123 | 0.0232 | 0.0164 | 0.0173 | done |

**group_a_us_indices · group-level · mode `combined`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0102 | 0.0169 | 0.0112 | 0.0128 | done |
| bilstm | 0.0072 | 0.0503 | 0.0107 | 0.0227 | done |
| cnn | 0.0210 | 0.0204 | 0.0092 | 0.0169 | done |
| gru | 0.0061 | 0.0424 | 0.0237 | 0.0241 | done |
| logreg | 0.0082 | 0.0324 | 0.0073 | 0.0160 | done |
| lstm | 0.0060 | 0.0503 | 0.0357 | 0.0307 | done |
| tcn | 0.0080 | 0.0326 | 0.0078 | 0.0161 | done |
| transformer | 0.0120 | 0.0193 | 0.0072 | 0.0128 | done |

**group_a_us_indices · group-level · mode `full`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0086 | 0.0151 | 0.0061 | 0.0099 | done |
| bilstm | 0.0069 | 0.0225 | 0.0106 | 0.0133 | done |
| cnn | 0.0100 | 0.0178 | 0.0042 | 0.0107 | done |
| gru | 0.0072 | 0.0179 | 0.0160 | 0.0137 | done |
| logreg | 0.0084 | 0.0252 | 0.0070 | 0.0135 | done |
| lstm | 0.0062 | 0.0186 | 0.0090 | 0.0113 | done |
| tcn | 0.0072 | 0.0206 | 0.0073 | 0.0117 | done |
| transformer | 0.0128 | 0.0163 | 0.0050 | 0.0114 | done |

**group_a_us_indices · group-level · mode `landscape`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0145 | 0.0177 | 0.0060 | 0.0127 | done |
| bilstm | 0.0162 | 0.0151 | 0.0093 | 0.0135 | done |
| cnn | 0.0631 | 0.0188 | 0.0064 | 0.0294 | done |
| gru | 0.0194 | 0.0138 | 0.0057 | 0.0130 | done |
| logreg | 0.0094 | 0.0225 | 0.0073 | 0.0130 | done |
| lstm | 0.0131 | 0.0149 | 0.0098 | 0.0126 | done |
| tcn | 0.0148 | 0.0169 | 0.0256 | 0.0191 | done |
| transformer | 0.0108 | 0.0131 | 0.0099 | 0.0112 | done |

**group_a_us_indices · group-level · mode `tda`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0173 | 0.0293 | 0.0051 | 0.0172 | done |
| bilstm | 0.0090 | 0.0245 | 0.0058 | 0.0131 | done |
| cnn | 0.0124 | 0.0273 | 0.0053 | 0.0150 | done |
| gru | 0.0082 | 0.0234 | 0.0042 | 0.0119 | done |
| logreg | 0.0082 | 0.0241 | 0.0049 | 0.0124 | done |
| lstm | 0.0068 | 0.0216 | 0.0060 | 0.0115 | done |
| tcn | 0.0085 | 0.0418 | 0.0046 | 0.0183 | done |
| transformer | 0.0081 | 0.0212 | 0.0045 | 0.0113 | done |

**group_b_cross_asset · group-level · mode `classical`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0078 | 0.0239 | 0.0125 | 0.0148 | done |
| bilstm | 0.0094 | 0.0291 | 0.0102 | 0.0162 | done |
| cnn | 0.0126 | 0.0277 | 0.0109 | 0.0171 | done |
| gru | 0.0073 | 0.0120 | 0.0114 | 0.0102 | done |
| logreg | 0.0209 | 0.1028 | 0.0130 | 0.0456 | done |
| lstm | 0.0055 | 0.0344 | 0.0139 | 0.0179 | done |
| tcn | 0.0111 | 0.0608 | 0.0166 | 0.0295 | done |
| transformer | 0.0089 | 0.0256 | 0.0158 | 0.0167 | done |

**group_b_cross_asset · group-level · mode `combined`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0095 | 0.0266 | 0.0145 | 0.0169 | done |
| bilstm | 0.0131 | 0.0274 | 0.0132 | 0.0179 | done |
| cnn | 0.0088 | 0.0183 | 0.0190 | 0.0154 | done |
| gru | 0.0089 | 0.0583 | 0.0112 | 0.0261 | done |
| logreg | 0.0176 | 0.1115 | 0.0128 | 0.0473 | done |
| lstm | 0.0123 | 0.0217 | 0.0177 | 0.0172 | done |
| tcn | 0.0047 | 0.0460 | 0.0096 | 0.0201 | done |
| transformer | 0.0096 | 0.0233 | 0.0150 | 0.0159 | done |

**group_b_cross_asset · group-level · mode `full`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0148 | 0.0147 | 0.0118 | 0.0138 | done |
| bilstm | 0.0071 | 0.0174 | 0.0108 | 0.0118 | done |
| cnn | 0.0127 | 0.0213 | 0.0139 | 0.0160 | done |
| gru | 0.0073 | 0.1621 | 0.0118 | 0.0604 | done |
| logreg | 0.0076 | 0.0775 | 0.0107 | 0.0319 | done |
| lstm | 0.0070 | 0.0260 | 0.0122 | 0.0151 | done |
| tcn | 0.0154 | 0.1164 | 0.0100 | 0.0473 | done |
| transformer | 0.0074 | 0.0286 | 0.0153 | 0.0171 | done |

**group_b_cross_asset · group-level · mode `landscape`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0130 | 0.0100 | 0.0092 | 0.0108 | done |
| bilstm | 0.0092 | 0.0098 | 0.0085 | 0.0092 | done |
| cnn | 0.0098 | 0.0108 | 0.0100 | 0.0102 | done |
| gru | 0.0081 | 0.0106 | 0.0094 | 0.0094 | done |
| logreg | 0.0077 | 0.0165 | 0.0110 | 0.0117 | done |
| lstm | 0.0069 | 0.0166 | 0.0088 | 0.0108 | done |
| tcn | 0.0060 | 0.0229 | 0.0091 | 0.0127 | done |
| transformer | 0.0114 | 0.0126 | 0.0104 | 0.0115 | done |

**group_b_cross_asset · group-level · mode `tda`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0137 | 0.0353 | 0.0105 | 0.0199 | done |
| bilstm | 0.0073 | 0.0246 | 0.0098 | 0.0139 | done |
| cnn | 0.0118 | 0.0129 | 0.0114 | 0.0121 | done |
| gru | 0.0092 | 0.0336 | 0.0100 | 0.0176 | done |
| logreg | 0.0104 | 0.0124 | 0.0102 | 0.0110 | done |
| lstm | 0.0057 | 0.0118 | 0.0094 | 0.0090 | done |
| tcn | 0.0109 | 0.0179 | 0.0097 | 0.0128 | done |
| transformer | 0.0086 | 0.0127 | 0.0104 | 0.0105 | done |

**international · group-level · mode `classical`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0232 | 0.0125 | 0.0135 | 0.0164 | done |
| bilstm | 0.0235 | 0.0293 | 0.0066 | 0.0198 | done |
| cnn | 0.0178 | 0.0111 | 0.0086 | 0.0125 | done |
| gru | 0.0287 | 0.0264 | 0.0062 | 0.0205 | done |
| logreg | 0.0142 | 0.0876 | 0.0060 | 0.0359 | done |
| lstm | 0.0395 | 0.0336 | 0.0044 | 0.0258 | done |
| tcn | 0.0470 | 0.0311 | 0.0050 | 0.0277 | done |
| transformer | 0.0188 | 0.0130 | 0.0105 | 0.0141 | done |

**international · group-level · mode `combined`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0956 | 0.0113 | 0.0065 | 0.0378 | done |
| bilstm | 0.0460 | 0.0123 | 0.0100 | 0.0228 | done |
| cnn | 0.0623 | 0.0099 | 0.0056 | 0.0259 | done |
| gru | 0.0467 | 0.0196 | 0.0161 | 0.0275 | done |
| logreg | 0.0251 | 0.0205 | 0.0060 | 0.0172 | done |
| lstm | 0.0847 | 0.0183 | 0.0067 | 0.0366 | done |
| tcn | 0.0318 | 0.0140 | 0.0140 | 0.0199 | done |
| transformer | 0.0587 | 0.0121 | 0.0057 | 0.0255 | done |

**international · group-level · mode `full`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0310 | 0.0177 | 0.0048 | 0.0178 | done |
| bilstm | 0.0286 | 0.0759 | 0.0054 | 0.0367 | done |
| cnn | 0.0692 | 0.0269 | 0.0106 | 0.0355 | done |
| gru | 0.0234 | 0.0534 | 0.0044 | 0.0271 | done |
| logreg | 0.0175 | 0.0648 | 0.0051 | 0.0291 | done |
| lstm | 0.0340 | 0.0283 | 0.0037 | 0.0220 | done |
| tcn | 0.0472 | 0.0686 | 0.0036 | 0.0398 | done |
| transformer | 0.0249 | 0.0391 | 0.0045 | 0.0228 | done |

**international · group-level · mode `landscape`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0245 | 0.0303 | 0.0041 | 0.0197 | done |
| bilstm | 0.0211 | 0.0308 | 0.0047 | 0.0188 | done |
| cnn | 0.0160 | 0.0539 | 0.0073 | 0.0257 | done |
| gru | 0.0154 | 0.0618 | 0.0036 | 0.0269 | done |
| logreg | 0.0174 | 0.0674 | 0.0032 | 0.0293 | done |
| lstm | 0.0147 | 0.3114 | 0.0046 | 0.1102 | done |
| tcn | 0.0147 | 0.2352 | 0.0031 | 0.0843 | done |
| transformer | 0.0298 | 0.1515 | 0.0029 | 0.0614 | done |

**international · group-level · mode `tda`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0204 | 0.0148 | 0.0078 | 0.0143 | done |
| bilstm | 0.0165 | 0.0523 | 0.0075 | 0.0254 | done |
| cnn | 0.0177 | 0.0437 | 0.0040 | 0.0218 | done |
| gru | 0.0209 | 0.0225 | 0.0057 | 0.0163 | done |
| logreg | 0.0304 | 0.0100 | 0.0055 | 0.0153 | done |
| lstm | 0.0178 | 0.0119 | 0.0036 | 0.0111 | done |
| tcn | 0.0216 | 0.0177 | 0.0036 | 0.0143 | done |
| transformer | 0.0291 | 0.0105 | 0.0091 | 0.0162 | done |

**tech · group-level · mode `classical`**

| model | fold_proportional | mean AP | status |
|---|---|---|---|
| attn_bilstm | 0.0530 | 0.0530 | done |
| bilstm | 0.0547 | 0.0547 | done |
| cnn | 0.0522 | 0.0522 | done |
| gru | 0.0920 | 0.0920 | done |
| logreg | 0.0554 | 0.0554 | done |
| lstm | 0.0488 | 0.0488 | done |
| tcn | 0.0446 | 0.0446 | done |
| transformer | 0.0508 | 0.0508 | done |

**tech · group-level · mode `combined`**

| model | fold_proportional | mean AP | status |
|---|---|---|---|
| attn_bilstm | 0.0624 | 0.0624 | done |
| bilstm | 0.0851 | 0.0851 | done |
| cnn | 0.0508 | 0.0508 | done |
| gru | 0.0686 | 0.0686 | done |
| logreg | 0.0746 | 0.0746 | done |
| lstm | 0.0592 | 0.0592 | done |
| tcn | 0.0679 | 0.0679 | done |
| transformer | 0.0660 | 0.0660 | done |

**tech · group-level · mode `full`**

| model | fold_proportional | mean AP | status |
|---|---|---|---|
| attn_bilstm | 0.0491 | 0.0491 | done |
| bilstm | 0.0484 | 0.0484 | done |
| cnn | 0.0435 | 0.0435 | done |
| gru | 0.0588 | 0.0588 | done |
| logreg | 0.0484 | 0.0484 | done |
| lstm | 0.0572 | 0.0572 | done |
| tcn | 0.0543 | 0.0543 | done |
| transformer | 0.0638 | 0.0638 | done |

**tech · group-level · mode `landscape`**

| model | fold_proportional | mean AP | status |
|---|---|---|---|
| attn_bilstm | 0.0566 | 0.0566 | done |
| bilstm | 0.0695 | 0.0695 | done |
| cnn | 0.0707 | 0.0707 | done |
| gru | 0.0557 | 0.0557 | done |
| logreg | 0.0434 | 0.0434 | done |
| lstm | 0.0843 | 0.0843 | done |
| tcn | 0.0784 | 0.0784 | done |
| transformer | 0.0709 | 0.0709 | done |

**tech · group-level · mode `tda`**

| model | fold_proportional | mean AP | status |
|---|---|---|---|
| attn_bilstm | 0.0510 | 0.0510 | done |
| bilstm | 0.1038 | 0.1038 | done |
| cnn | 0.0508 | 0.0508 | done |
| gru | 0.0624 | 0.0624 | done |
| logreg | 0.0504 | 0.0504 | done |
| lstm | 0.0461 | 0.0461 | done |
| tcn | 0.0688 | 0.0688 | done |
| transformer | 0.0660 | 0.0660 | done |

**ALL · ticker-level · mode `classical`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0565 | 0.0392 | 0.0292 | 0.0416 | done |
| bilstm | 0.0553 | 0.0499 | 0.0284 | 0.0445 | done |
| cnn | 0.0398 | 0.0468 | 0.0247 | 0.0371 | done |
| gru | 0.0505 | 0.0497 | 0.0282 | 0.0428 | done |
| logreg | 0.0349 | 0.0698 | 0.0274 | 0.0440 | done |
| lstm | 0.0503 | 0.0483 | 0.0276 | 0.0421 | done |
| tcn | 0.0517 | 0.0507 | 0.0238 | 0.0421 | done |
| transformer | 0.0469 | 0.0411 | 0.0264 | 0.0381 | done |

**ALL · ticker-level · mode `combined`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0507 | - | - | 0.0507 | partial |
| bilstm | 0.0434 | 0.0524 | - | 0.0479 | partial |
| cnn | - | - | - | - | not_run |
| gru | 0.0556 | - | - | 0.0556 | partial |
| logreg | 0.0345 | 0.0692 | 0.0275 | 0.0437 | done |
| lstm | 0.0427 | 0.0455 | 0.0269 | 0.0384 | done |
| tcn | - | - | - | - | not_run |
| transformer | - | - | - | - | not_run |

**ALL · ticker-level · mode `full`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | - | - | - | - | not_run |
| bilstm | - | - | - | - | not_run |
| cnn | - | - | - | - | not_run |
| gru | - | - | - | - | not_run |
| logreg | - | - | - | - | not_run |
| lstm | - | - | - | - | not_run |
| tcn | - | - | - | - | not_run |
| transformer | - | - | - | - | not_run |

**ALL · ticker-level · mode `landscape`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | - | - | - | - | not_run |
| bilstm | - | - | - | - | not_run |
| cnn | - | - | - | - | not_run |
| gru | - | - | - | - | not_run |
| logreg | - | - | - | - | not_run |
| lstm | - | - | - | - | not_run |
| tcn | - | - | - | - | not_run |
| transformer | - | - | - | - | not_run |

**ALL · ticker-level · mode `tda`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| attn_bilstm | 0.0177 | 0.0405 | 0.0165 | 0.0249 | done |
| bilstm | 0.0164 | 0.0436 | 0.0166 | 0.0256 | done |
| cnn | 0.0157 | 0.0339 | 0.0170 | 0.0222 | done |
| gru | 0.0183 | 0.0425 | 0.0173 | 0.0260 | done |
| logreg | 0.0189 | 0.0392 | 0.0177 | 0.0253 | done |
| lstm | 0.0168 | 0.0419 | 0.0181 | 0.0256 | done |
| tcn | 0.0163 | 0.0402 | 0.0165 | 0.0243 | done |
| transformer | 0.0167 | - | - | 0.0167 | partial |
## v1 — every run (AP by fold)

**energy · group-level · mode `classical`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| bilstm | 0.0374 | 0.0618 | 0.0506 | 0.0499 | done |
| cnn | 0.0429 | 0.0563 | 0.0609 | 0.0533 | done |
| gru | 0.0405 | 0.0606 | 0.0465 | 0.0492 | done |
| logreg | 0.0372 | 0.0792 | 0.0330 | 0.0498 | done |
| lstm | 0.0488 | 0.0583 | 0.0385 | 0.0485 | done |
| tcn | 0.0310 | 0.0609 | 0.0550 | 0.0490 | done |
| transformer | 0.0239 | 0.0589 | 0.0446 | 0.0425 | done |

**energy · group-level · mode `combined`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| bilstm | 0.0369 | 0.0627 | 0.0418 | 0.0471 | done |
| cnn | 0.0895 | 0.0646 | 0.0488 | 0.0676 | done |
| gru | 0.0375 | 0.0633 | 0.0327 | 0.0445 | done |
| logreg | 0.0334 | 0.0873 | 0.0302 | 0.0503 | done |
| lstm | 0.0487 | 0.0609 | 0.0625 | 0.0574 | done |
| tcn | 0.0337 | 0.0668 | 0.0540 | 0.0515 | done |
| transformer | 0.0478 | 0.0681 | 0.0239 | 0.0466 | done |

**energy · group-level · mode `tda`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| bilstm | 0.0547 | 0.0787 | 0.0400 | 0.0578 | done |
| cnn | 0.0681 | 0.0843 | 0.0295 | 0.0607 | done |
| gru | 0.0680 | 0.0775 | 0.0339 | 0.0598 | done |
| logreg | 0.0360 | 0.0731 | 0.0338 | 0.0477 | done |
| lstm | 0.0359 | 0.0789 | 0.0303 | 0.0484 | done |
| tcn | 0.0609 | 0.0725 | 0.0369 | 0.0568 | done |
| transformer | 0.0470 | 0.0914 | 0.0269 | 0.0551 | done |

**financials · group-level · mode `classical`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| bilstm | 0.1521 | 0.0343 | 0.0251 | 0.0705 | done |
| cnn | 0.0924 | 0.0292 | 0.0151 | 0.0456 | done |
| gru | 0.0796 | 0.0436 | 0.0201 | 0.0477 | done |
| logreg | 0.0573 | 0.0411 | 0.0217 | 0.0400 | done |
| lstm | 0.1113 | 0.0355 | 0.0836 | 0.0768 | done |
| tcn | 0.1051 | 0.0387 | 0.0141 | 0.0527 | done |
| transformer | 0.0924 | 0.0267 | 0.0289 | 0.0493 | done |

**financials · group-level · mode `combined`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| bilstm | 0.1014 | 0.0344 | 0.0284 | 0.0547 | done |
| cnn | 0.0684 | 0.0311 | 0.0107 | 0.0368 | done |
| gru | 0.1614 | 0.0303 | 0.0202 | 0.0706 | done |
| logreg | 0.0434 | 0.0417 | 0.0159 | 0.0337 | done |
| lstm | 0.0851 | 0.0298 | 0.0161 | 0.0436 | done |
| tcn | 0.0791 | 0.0403 | 0.0174 | 0.0456 | done |
| transformer | 0.0735 | 0.0279 | 0.0106 | 0.0373 | done |

**financials · group-level · mode `tda`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| bilstm | 0.0319 | 0.0287 | 0.0206 | 0.0271 | done |
| cnn | 0.0327 | 0.0269 | 0.0123 | 0.0239 | done |
| gru | 0.0219 | 0.0275 | 0.0202 | 0.0232 | done |
| logreg | 0.0300 | 0.0401 | 0.0153 | 0.0285 | done |
| lstm | 0.0336 | 0.0284 | 0.0189 | 0.0270 | done |
| tcn | 0.0300 | 0.0251 | 0.0162 | 0.0237 | done |
| transformer | 0.0379 | 0.0359 | 0.0168 | 0.0302 | done |

**group_a_us_indices · group-level · mode `classical`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| bilstm | 0.0087 | 0.0396 | 0.0513 | 0.0332 | done |
| cnn | 0.0121 | 0.0351 | 0.0084 | 0.0185 | done |
| gru | 0.0078 | 0.0362 | 0.0192 | 0.0211 | done |
| logreg | 0.0123 | 0.0574 | 0.0097 | 0.0264 | done |
| lstm | 0.0116 | 0.0240 | 0.0298 | 0.0218 | done |
| tcn | 0.0095 | 0.0373 | 0.0200 | 0.0223 | done |
| transformer | 0.0147 | 0.0252 | 0.0312 | 0.0237 | done |

**group_a_us_indices · group-level · mode `combined`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| bilstm | 0.0085 | 0.0343 | 0.0140 | 0.0189 | done |
| cnn | 0.0101 | 0.0189 | 0.0143 | 0.0144 | done |
| gru | 0.0090 | 0.0320 | 0.0145 | 0.0185 | done |
| logreg | 0.0074 | 0.0747 | 0.0066 | 0.0296 | done |
| lstm | 0.0068 | 0.0278 | 0.0474 | 0.0273 | done |
| tcn | 0.0069 | 0.0270 | 0.0079 | 0.0139 | done |
| transformer | 0.0151 | 0.0139 | 0.0058 | 0.0116 | done |

**group_a_us_indices · group-level · mode `tda`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| bilstm | 0.0101 | 0.0239 | 0.0057 | 0.0132 | done |
| cnn | 0.0087 | 0.0232 | 0.0070 | 0.0130 | done |
| gru | 0.0096 | 0.0220 | 0.0047 | 0.0121 | done |
| logreg | 0.0081 | 0.0257 | 0.0052 | 0.0130 | done |
| lstm | 0.0093 | 0.0278 | 0.0056 | 0.0142 | done |
| tcn | 0.0077 | 0.0386 | 0.0055 | 0.0173 | done |
| transformer | 0.0080 | 0.0134 | 0.0046 | 0.0087 | done |

**group_b_cross_asset · group-level · mode `classical`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| bilstm | 0.0085 | 0.0630 | 0.0126 | 0.0280 | done |
| cnn | 0.0090 | 0.0509 | 0.0099 | 0.0233 | done |
| gru | 0.0091 | 0.1882 | 0.0116 | 0.0696 | done |
| logreg | 0.0070 | 0.1457 | 0.0132 | 0.0553 | done |
| lstm | 0.0083 | 0.0513 | 0.0110 | 0.0235 | done |
| tcn | 0.0064 | 0.0326 | 0.0102 | 0.0164 | done |
| transformer | 0.0092 | 0.0238 | 0.0122 | 0.0151 | done |

**group_b_cross_asset · group-level · mode `combined`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| bilstm | 0.0104 | 0.0458 | 0.0131 | 0.0231 | done |
| cnn | 0.0094 | 0.0830 | 0.0139 | 0.0354 | done |
| gru | 0.0061 | 0.0726 | 0.0126 | 0.0304 | done |
| logreg | 0.0081 | 0.1361 | 0.0134 | 0.0525 | done |
| lstm | 0.0093 | 0.0249 | 0.0114 | 0.0152 | done |
| tcn | 0.0062 | 0.0635 | 0.0100 | 0.0266 | done |
| transformer | 0.0086 | 0.0441 | 0.0146 | 0.0224 | done |

**group_b_cross_asset · group-level · mode `tda`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| bilstm | 0.0073 | 0.0112 | 0.0099 | 0.0095 | done |
| cnn | 0.0136 | 0.0111 | 0.0103 | 0.0117 | done |
| gru | 0.0071 | 0.0185 | 0.0095 | 0.0117 | done |
| logreg | 0.0114 | 0.0126 | 0.0098 | 0.0112 | done |
| lstm | 0.0078 | 0.0120 | 0.0101 | 0.0100 | done |
| tcn | 0.0107 | 0.0135 | 0.0098 | 0.0113 | done |
| transformer | 0.0108 | 0.0130 | 0.0112 | 0.0117 | done |

**international · group-level · mode `classical`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| bilstm | 0.1439 | 0.0249 | 0.0042 | 0.0577 | done |
| cnn | 0.0450 | 0.0136 | 0.0055 | 0.0214 | done |
| gru | 0.0516 | 0.0256 | 0.0030 | 0.0267 | done |
| logreg | 0.0234 | 0.0238 | 0.0059 | 0.0177 | done |
| lstm | 0.1296 | 0.0121 | 0.0043 | 0.0487 | done |
| tcn | 0.0400 | 0.0334 | 0.0073 | 0.0269 | done |
| transformer | 0.0169 | 0.0102 | 0.0051 | 0.0108 | done |

**international · group-level · mode `combined`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| bilstm | 0.0480 | 0.0168 | 0.0489 | 0.0379 | done |
| cnn | 0.0482 | 0.0097 | 0.0074 | 0.0218 | done |
| gru | 0.0874 | 0.0174 | 0.0048 | 0.0365 | done |
| logreg | 0.0228 | 0.0175 | 0.0055 | 0.0153 | done |
| lstm | 0.0382 | 0.0105 | 0.0095 | 0.0194 | done |
| tcn | 0.0406 | 0.0189 | 0.0055 | 0.0217 | done |
| transformer | 0.1269 | 0.0115 | 0.0057 | 0.0480 | done |

**international · group-level · mode `tda`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| bilstm | 0.0167 | 0.0339 | 0.0081 | 0.0196 | done |
| cnn | 0.0183 | 0.0152 | 0.0038 | 0.0124 | done |
| gru | 0.0218 | 0.0388 | 0.0038 | 0.0215 | done |
| logreg | 0.0317 | 0.0143 | 0.0055 | 0.0172 | done |
| lstm | 0.0168 | 0.0343 | 0.0063 | 0.0191 | done |
| tcn | 0.0156 | 0.0358 | 0.0030 | 0.0181 | done |
| transformer | 0.0221 | 0.0138 | 0.0036 | 0.0132 | done |

**tech · group-level · mode `classical`**

| model | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|
| bilstm | 0.2068 | 0.0640 | 0.1354 | done |
| cnn | 0.0577 | 0.0575 | 0.0576 | done |
| gru | 0.0476 | 0.0676 | 0.0576 | done |
| logreg | 0.0623 | 0.0538 | 0.0581 | done |
| lstm | 0.0454 | 0.0744 | 0.0599 | done |
| tcn | 0.0347 | 0.0592 | 0.0470 | done |
| transformer | 0.0976 | 0.0628 | 0.0802 | done |

**tech · group-level · mode `combined`**

| model | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|
| bilstm | 0.0620 | 0.0616 | 0.0618 | done |
| cnn | 0.0599 | 0.0396 | 0.0498 | done |
| gru | 0.1312 | 0.0492 | 0.0902 | done |
| logreg | 0.0630 | 0.0461 | 0.0546 | done |
| lstm | 0.0499 | 0.0896 | 0.0697 | done |
| tcn | 0.0468 | 0.0453 | 0.0460 | done |
| transformer | 0.0368 | 0.0469 | 0.0418 | done |

**tech · group-level · mode `tda`**

| model | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|
| bilstm | 0.0489 | 0.0523 | 0.0506 | done |
| cnn | 0.0383 | 0.0416 | 0.0400 | done |
| gru | 0.0436 | 0.0496 | 0.0466 | done |
| logreg | 0.0434 | 0.0400 | 0.0417 | done |
| lstm | 0.0369 | 0.0466 | 0.0418 | done |
| tcn | 0.0445 | 0.0449 | 0.0447 | done |
| transformer | 0.0350 | 0.0382 | 0.0366 | done |

**ALL · ticker-level · mode `classical`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| bilstm | 0.0566 | 0.0470 | 0.0277 | 0.0438 | done |
| cnn | 0.0425 | 0.0472 | 0.0257 | 0.0385 | done |
| gru | 0.0475 | 0.0514 | 0.0276 | 0.0422 | done |
| logreg | 0.0358 | 0.0607 | 0.0266 | 0.0410 | done |
| lstm | 0.0536 | 0.0460 | 0.0282 | 0.0426 | done |
| tcn | 0.0516 | 0.0527 | 0.0290 | 0.0444 | done |
| transformer | 0.0388 | 0.0380 | - | 0.0384 | partial |

**ALL · ticker-level · mode `combined`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| bilstm | 0.0459 | 0.0491 | - | 0.0475 | partial |
| cnn | 0.0367 | - | - | 0.0367 | partial |
| gru | 0.0387 | 0.0561 | - | 0.0474 | partial |
| logreg | 0.0355 | 0.0603 | 0.0263 | 0.0407 | done |
| lstm | 0.0451 | 0.0458 | 0.0266 | 0.0392 | done |
| tcn | - | - | - | - | not_run |
| transformer | - | - | - | - | not_run |

**ALL · ticker-level · mode `tda`**

| model | fold_2011_2015 | fold_2016_2020 | fold_2021_plus | mean AP | status |
|---|---|---|---|---|---|
| bilstm | 0.0192 | 0.0316 | 0.0224 | 0.0244 | done |
| cnn | 0.0210 | 0.0359 | 0.0194 | 0.0254 | done |
| gru | 0.0180 | 0.0367 | 0.0221 | 0.0256 | done |
| logreg | 0.0169 | 0.0365 | 0.0226 | 0.0253 | done |
| lstm | 0.0167 | 0.0455 | 0.0195 | 0.0272 | done |
| tcn | 0.0198 | 0.0342 | 0.0177 | 0.0239 | done |
| transformer | - | - | - | - | not_run |