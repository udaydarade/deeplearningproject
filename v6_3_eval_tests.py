"""V6 evaluator smoke tests; the production evaluator is copied from V5."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from pipeline_v6.v5_eval_copy import (
    ERAS, assign_eras, blocked_split, evaluate, off_budget, _alarm_runs,
)


def main():
    dates = pd.date_range("1928-01-01", "2050-12-31", freq="D")
    frame = pd.DataFrame({"date": dates, "asset": "A", "y": 0})
    assert assign_eras(frame.date).notna().all()
    train, val, test = blocked_split(frame, "E1", 15)
    assert len(test) > 0
    dates_unique = pd.DatetimeIndex(frame.date.drop_duplicates().sort_values())
    positions = {d: i for i, d in enumerate(dates_unique)}
    test_positions = [positions[d] for d in frame.iloc[test].date.unique()]
    lo, hi = min(test_positions), max(test_positions)
    for ix in (train, val):
        train_positions = [positions[d] for d in frame.iloc[ix].date.unique()]
        assert all(p < lo - 15 or p > hi + 15 for p in train_positions)

    # The held-out era assignment covers every labeled input row exactly once.
    era_ids = assign_eras(frame.date)
    assert len(era_ids) == len(frame) and era_ids.notna().all()
    assert sum(int((era_ids == name).sum()) for name, _, _ in ERAS) == len(frame)

    # Oracle ranking is perfect and random ranking remains non-degenerate.
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9]); random_score = np.random.default_rng(3).permutation(score)
    assert roc_auc_score(y, score) == 1.0
    assert average_precision_score(y, score) == 1.0
    assert 0.0 <= roc_auc_score(y, random_score) <= 1.0

    # Hand-counted 30-row toy: 2 crashes, 2 caught, 3 alarm episodes, 1 wrong.
    toy_y = np.zeros(30, dtype=int); toy_y[5:8] = 1; toy_y[20:22] = 1
    toy_alarm = np.zeros(30, dtype=bool); toy_alarm[[4, 5, 10, 11, 20, 29]] = True
    toy = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=30),
                        "asset": "A", "y": toy_y, "score": np.arange(30, dtype=float)})
    metrics = evaluate(toy, toy_alarm, sims=200)
    assert (metrics["crashes_total"], metrics["crashes_caught"]) == (2, 2)
    assert (metrics["alarms_total"], metrics["alarms_correct"], metrics["alarms_wrong"]) == (3, 2, 1)
    assert _alarm_runs(np.ones(30, dtype=bool)) == [(0, 14), (15, 29)]
    assert off_budget(1.0, .05) and not off_budget(.05, .05)
    print(f"era coverage={len(assign_eras(frame.date).unique())} "
          f"blocked train={len(train)} val={len(val)} test={len(test)}")
    print("V6 evaluator tests passed: embargo, full era coverage, oracle, hand-counted events, alarm split, off-budget")


if __name__ == "__main__":
    main()
