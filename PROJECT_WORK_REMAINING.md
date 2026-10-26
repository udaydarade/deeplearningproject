# Project work remaining

Status date: 2026-10-25

This is the honest remaining-work list. Items below are either incomplete,
not run, or require broader historical folds before they can be described as
full-history results.

## 1. V5 E2 paired ablation grid

Status: **incomplete and stopped**.

- Planned grid: 2,448 jobs.
- Previously saved result rows: 396.
- Last checkpoint recorded approximately 70 unique completed jobs.
- A resumable continuation was started and then stopped at the user's
  request.
- Existing rows and score files remain preserved in:
  - `data/v5/e2_results.csv`
  - `data/v5/e2_validation.csv`
  - `data/v5/scores/`
  - `data/v5/e2_v7_final_checkpoint.csv`

V5 E2 must not be described as complete.

## 2. V5 E3 deep experiment

Status: **partial / completion not verified**.

- `data/v5/e3_results.csv` contains 116 rows.
- `data/v5/e3_validation.csv` exists.
- No authoritative completion marker confirms that all intended E3 cells were
  run.

## 3. V6 optional analyses not run

The corrected V6 remaining-status audit records the following:

- External universe extension: not run.
- T3 conditional-direction fit: not run.
- S1–S4 strategy suite: not run because component score histories were not
  saved.
- Detailed per-row cross-direction transfer: not run/persisted in the
  corrected time-boxed runner.

The presence of an aggregate or archived transfer table must not be confused
with a complete persisted per-row transfer experiment.

## 4. V7 full historical folds

Status: **not complete**.

V7 currently tests three configured modern folds:

- 2011–2015.
- 2016–2020.
- 2021 onward.

The source panel reaches approximately 1927, but that does not mean V7 tested
every historical date. A true full-history V7 extension would need additional
chronological folds, for example:

- 1950–1959.
- 1960–1969.
- 1970–1979.
- 1980–1989.
- 1990–1999.
- 2000–2009.
- 2010–2015.
- 2016–2020.
- 2021 onward.

That extension was not started because the user asked to stop running jobs.

## 5. Full-history event-level model ledger

Status: **not complete**.

The observed event table covers 8,099 events, but the event/model ledgers only
cover events intersecting saved V7 score files. A full-history event-by-model
ledger requires score files for every historical test fold and every requested
model.

## 6. Bootstrap and significance completeness

The V7 bootstrap output exists, but a final scientific sign-off should verify
that every requested comparison uses matched event IDs and common score
coverage:

- `GRAND_COMBINED` versus `PEER_CROSS`.
- GRU/TCN versus Logistic Regression/LightGBM.

This verification was not expanded after the user requested stopping.

## 7. Status-file caveat

The root `data/v6/current_status.csv` is stale and still contains
`running`/`pending` values. The corrected archived status is the authoritative
one:

`data/v6/archive_pre_forward_label_fix_20261025/current_status.csv`

The root file was not rewritten because it is part of the historical audit
trail.

## 8. Process state after stop request

- The V5 E2 process started during the latest continuation was stopped.
- No V5/V6/V7 training process should be considered active.
- Remaining Python processes observed afterward were notebook kernels, not
  identifiable experiment runners; they were left untouched.

## 9. What is most worth trying later

If the project resumes, the highest-value next experiment is not another broad
model grid. It is:

1. Finish or explicitly close V5 E2 only if its ablation comparison is needed.
2. Add the historical V7 folds with the same strict pre-onset definition.
3. Persist per-row cross-direction transfer scores for V6.
4. Produce paired event-level confidence intervals using only common coverage.

The saved V6 Scheme W SEMIVAR results are currently the strongest broad
operational evidence and should be prioritized in the course report.
