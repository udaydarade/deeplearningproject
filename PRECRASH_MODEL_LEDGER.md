# Strict pre-crash model ledger

This is the corrected forecasting ledger. A model is counted as detecting a crash only when its RQ alarm occurs strictly before the crash onset. Alarms on or after `crash_start` are never detections.

## Window definitions

For each event, `wK` is the last K available trading sessions before the onset date. The onset session itself is excluded. Results are reported for K = 1, 5, 10, 20, and 60; this makes the operational lead-time assumption explicit.

## Coverage

- Observed events in the full label population: 8,099.
- Events with at least one saved-model pre-onset coverage row: 2,479.
- Event/model rows with pre-onset coverage: 58,330.
- Events without saved score coverage are excluded from the model denominator, not silently labeled as misses.
- These results cover the saved V7 score files only; they are not full-history model results until historical V7 score folds are run.

## Key interpretation

- `wK_detected=True` means at least one alarm occurred in the K trading-session pre-onset window.
- `lead_sessions` is the number of trading sessions between the alarm and onset within that window.
- Alarms inside the crash interval are retained only in the legacy audit ledger and do not count here.

## Complete files

- `data/v7_final/precrash_ledger/precrash_model_event_ledger.csv`: one row per event/model with all strict pre-onset windows.
- `data/v7_final/precrash_ledger/precrash_by_event.csv`: one row per event with the models that forecast it before onset.
- `data/v7_final/precrash_ledger/precrash_model_summary.csv`: grouped event-recall and lead-time summary.
