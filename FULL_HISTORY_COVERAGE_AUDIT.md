# Full-history coverage audit

Audit date: 2026-10-25

## Bottom line

The project has generally used the full available source panel
(1927-12-30 through 2026-10-23, 159 assets), but “full source history” and
“full-history test coverage” are different claims.

The source panel is long. Some experiments trained on all eligible past data
but tested only on selected recent folds. Other experiments tested every V5 era.
Therefore the results cannot be described uniformly as full-history testing.

## Verified coverage

| Round/output | Source history | Test coverage verified | Assessment |
|---|---|---|---|
| V4 `d1_paired.csv` | Reused V3 source data | Four folds, identifiers `1`–`4`; date mapping requires the V4 fold manifest | Not enough evidence here to call it full-history testing |
| V5 E1 | V5 inventory: 1927-12-30 to 2026-10-23 | Scheme B: E0–E10; Scheme W: E4–E10 | Full era coverage for the completed E1 result table |
| V5 E2 | Same V5 source panel | Completed rows include E0–E10, but only 396 rows of a planned 2,448-job grid | Full-era design, incomplete execution |
| V6 blocked sweep | V6 source panel and labels exist | Expected result filename is not currently present under `data/v6/`; coverage cannot be re-verified from that file | Do not claim until reconstructed |
| V6 Scheme W | V6 source panel and labels exist | Expected result filename is not currently present under `data/v6/`; coverage cannot be re-verified from that file | Do not claim until reconstructed |
| V7 current run | V6 panel, 1927-2026 | Three requested modern folds: 2011-2015, 2016-2020, 2021+ | Not full-history test coverage |

## What V7 is doing now

V7 uses the long history for feature rows and training before each fold, but its
current test schedule is only:

- 2011-2015;
- 2016-2020;
- 2021 onward.

As of this audit, V7 has 90 saved result rows, all for the N=5 label. N=15 has
not yet been reached. The run is resumable and its active process was left
running.

The V7 ladder is an ablation tool, not a replacement for historical coverage:

1. `CLASSICAL_CSD`;
2. `SEMIVAR_MOM`;
3. `PEER_CROSS`;
4. `TDA_ONLY_NORM`;
5. `GRAND_COMBINED`.

The ladder determines which feature family adds value. It should be evaluated
over every historical test fold, not only recent folds, if the report claims
full-history robustness.

## What must be rerun versus what must not be rerun

### Does not automatically require rerunning

- V5 E1 already has completed Scheme-B results across E0-E10.
- Existing V5 feature construction does not need rebuilding merely because a
  broader evaluation schedule is desired.
- V6 labels and SEMIVAR construction do not need rebuilding if their files are
  intact and checksums/row counts verify them.

### Requires additional evaluation or reconstruction

- V5 E2 must finish its incomplete 2,448-job grid if that grid is needed.
- V6 result coverage must be reconstructed from available archives/status files
  because the expected summary CSVs are not currently visible.
- V7 needs an added full-era walk-forward schedule if the course report claims
  full-history testing.
- V4 needs its fold manifest mapped to calendar dates before making a
  full-history claim.

## Recommended corrected V7 evaluation schedule

Keep the modern headline folds, but add historical chronological folds:

`1950-1959`, `1960-1969`, `1970-1979`, `1980-1989`, `1990-1999`,
`2000-2009`, `2010-2015`, `2016-2020`, and `2021+`.

For every fold:

- train only before the fold;
- preserve a 25-session purge gap;
- use the causal rolling-250-session RQ threshold;
- evaluate all five feature sets;
- evaluate Logistic Regression and LightGBM on the unified panel;
- evaluate GRU and TCN with seeds 42, 43, 44 on Group A;
- save per-cell scores and episode identifiers;
- run the paired episode bootstrap.

This produces a defensible distinction between:

- historical robustness across the full available record; and
- the modern real-time headline comparison.

## Conclusion

The prior work was not uniformly “only ten years.” V5 E1 explicitly tested all
V5 eras, while V5 E2 was designed for all eras but remains incomplete. V7,
however, currently tests only the recent three-fold schedule. The scientific
correction is to add the historical fold schedule rather than blindly rebuild
all feature files and labels.
