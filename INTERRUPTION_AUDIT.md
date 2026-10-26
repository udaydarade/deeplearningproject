# Experiment interruption audit

Audit date: 2026-10-25

This is a read-only audit of prior V1-V6 execution evidence. No process was
stopped while producing this file. Existing result files were not deleted or
rewritten.

## Current processes at audit time

The following processes were observed running:

| Process | Command | Interpretation |
|---:|---|---|
| 26584 | `v5_8_e3_deep.py --smoke` | V5 E3 smoke |
| 15160 | `v5_9_e2.py --timebox-min 90` | Resumed V5 E2 sweep |

No V6 process was observed in the current process list. V6 result files are
present and their latest timestamps are earlier than the active V5 work.

## Directly attributable stop

### V5 E2 full sweep

At approximately 16:17 on 2026-10-25, the original V5 E2 process was stopped
by the assistant while responding to the V7 request. The queue log records:

```text
END e2_full exit_code=-1
```

The process command was:

```text
py -3.12 v5_9_e2.py --timebox-min 90
```

At the time of interruption:

- planned E2 jobs: 2,448;
- completed result rows: 380;
- completed unique `(scheme, feature_set, model, era)` jobs: 67;
- completed validation rows: 67;
- saved score Parquet files: 170;
- latest completed job: `B/M0+TDA_small/logreg/E8`.

### Effect on results

The interruption did **not** erase the 380 result rows or score files. The
runner reads `data/v5/e2_results.csv`, reconstructs the completed-job set, and
skips those jobs on restart. A checkpoint file was also written:

- `data/v5/e2_v7_checkpoint.csv`

The resumed E2 command is currently active and has preserved the prior rows.
The sweep is still incomplete; 2,448 is the full planned grid, not the number
already completed.

## Earlier queue events and their effects

These events are supported by `data/v5/continuation_queue.log` and individual
queue logs.

| Event | Evidence | Effect | Final status |
|---|---|---|---|
| Existing process 9744 exited | continuation log, 14:29:54 | Previous queued process ended before corrected V5 rebuild | Corrected downstream work ran |
| First V5 dynamics build failed | `dynamics_build.queue.log`, exit code 1 | Initial dynamics attempt did not produce the final block | Manual guarded rerun completed 2,578/2,578 landscapes; 3/3 tests passed |
| E1 summary exited 1 | `e1_summary.queue.log` | That summary attempt failed | E1 result and validation files exist; summary should be treated as separately verified |
| E2 smoke exited 1 | `e2_smoke.queue.log` | Smoke wrapper failed | Queue log says the E2 smoke had already produced 240,490 score rows; the full runner later produced resumable outputs |
| V5 E2 full sweep stopped | `e2_full.queue.log`, exit code -1 | Remaining E2 jobs were not run in that pass | Resumed; existing 380 rows retained |

The first dynamics failure was a code defect in the early PCA slice, not lost
completed results. The guarded rerun is explicitly recorded in the continuation
log as complete.

## V6 interruption audit

The available V6 records show completed outputs for:

- inventory;
- labels and prevalence;
- SEMIVAR;
- blocked E-severity sweep;
- Scheme-W evaluation;
- native-direction comparison;
- N=5 cross-direction transfer.

No V6 termination event, partial-result marker, or active V6 process was found
in the current process list or V6 output directory. V6 omissions were planned
scope omissions, not evidence of accidental process loss:

- external universe extension;
- T3 conditional-direction model;
- S1 ensemble;
- S2 two-stage model;
- S3 per-class models;
- S4 multitask GRU.

## What was not run or remains incomplete

### V5

- The full E2 grid remains incomplete and is now being resumed.
- E3 was observed running in smoke mode; its completion should be checked from
  `data/v5/e3_results.csv` and `data/v5/e3_validation.csv`.
- Any downstream E2 summary, E3 full run, E4 labels, and episode tables should
  not be assumed complete solely because a queue entry exists.
- Existing V5 result files must be inspected after the active jobs finish.

### V7_FINAL_SYNTHESIS

No V7 experiment has been executed. The code scaffold is being written only.
The following are therefore not yet results:

- clean five-set ablation;
- N=5/N=15 head-to-head model matrix;
- three-seed GRU and TCN/attention comparison;
- paired episode bootstrap;
- lead-time table;
- V7 CSV deliverables.

## Recovery and scheduling decision

No additional V5/V6 process should be stopped. V7 is currently configured as a
dry-run/planning scaffold and must be explicitly approved before execution.
When approved, V7 should use separate `data/v7_final/` outputs and never write
to V5 or V6 files.

