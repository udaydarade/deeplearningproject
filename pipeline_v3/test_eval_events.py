"""Small deterministic tests for the V3 event evaluator."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline_v3.eval_events import (  # noqa: E402
    _event_metrics_per_asset, _random_baseline_per_asset, event_metrics,
    merge_runs,
)
from pipeline_v3.operating_rules import rolling_r2  # noqa: E402


class EventMetricTests(unittest.TestCase):
    def test_gap_merging_boundary(self):
        # Five zero rows between positive runs merge; six do not.
        self.assertEqual(merge_runs([(0, 2), (8, 9)]), [(0, 9)])
        self.assertEqual(merge_runs([(0, 2), (9, 9)]), [(0, 2), (9, 9)])

    def test_counts_events_and_day_overlap(self):
        y = np.zeros(24, dtype=bool)
        y[0:4] = True
        y[10:12] = True  # six-row gap: separate crash event
        alarm = np.zeros(24, dtype=bool)
        alarm[4] = True  # outside positive days, own alarm event
        alarm[11:13] = True  # overlaps second crash event
        self.assertEqual(_event_metrics_per_asset(y, alarm), (2, 1, 2, 1))

    def test_merged_alarm_gap_does_not_count_as_a_correct_alarm_day(self):
        y = np.zeros(10, dtype=bool)
        y[3] = True
        alarm = np.zeros(10, dtype=bool)
        alarm[[0, 6]] = True  # one merged alarm event; neither alarm day is positive
        self.assertEqual(_event_metrics_per_asset(y, alarm), (1, 0, 1, 0))

    def test_random_baseline_is_reproducible_counts(self):
        y = np.zeros(40, dtype=bool)
        y[5:8] = True
        y[25:28] = True
        first = _random_baseline_per_asset(y, 4, seed=17, sims=200)
        second = _random_baseline_per_asset(y, 4, seed=17, sims=200)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first[0], 0.0)
        self.assertLessEqual(first[0], 2.0)  # expected caught-event count
        self.assertGreaterEqual(first[1], 0.0)  # expected correct alarm-event count

    def test_panel_output_and_no_crash_asset(self):
        frame = pd.DataFrame({
            "asset": ["A"] * 12 + ["B"] * 12,
            "date": list(pd.bdate_range("2020-01-01", periods=12)) * 2,
            "y": [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0] + [0] * 12,
            "alarm": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] + [1] + [0] * 11,
        })
        result = event_metrics(frame, alarm_col="alarm", sims=50, seed=3)
        self.assertEqual(result["crashes_total"], 1)
        self.assertEqual(result["crashes_caught"], 1)
        self.assertEqual(result["alarms_total"], 2)
        self.assertEqual(result["alarms_correct"], 1)
        self.assertEqual(result["alarms_wrong"], 1)
        self.assertEqual(result["day_precision"], 0.5)
        self.assertEqual(result["day_recall"], 0.5)
        self.assertEqual(result["rand_caught"], result["rand_caught"])

    def test_rolling_rule_uses_previous_days_only(self):
        dates = pd.date_range("2020-01-01", periods=3, freq="B")
        f = pd.DataFrame({"asset": "A", "date": dates, "score": [.1, .2, .9]})
        out = rolling_r2(f, window=2, quantile=.95)
        self.assertFalse(bool(out.iloc[0].alarm))  # no prior scores
        self.assertTrue(bool(out.iloc[1].alarm))   # threshold is prior .1
        self.assertTrue(bool(out.iloc[2].alarm))   # threshold uses prior .1,.2

    def test_missing_price_dates_count_toward_event_gap(self):
        f = pd.DataFrame({
            "asset": ["A", "A"],
            "date": [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-13")],
            "y": [1, 1], "alarm": [0, 0],
        })
        m = event_metrics(f, alarm_col="alarm", sims=0)
        self.assertEqual(m["crashes_total"], 2)
        self.assertEqual(m["asset_years"], 2 / 252)


if __name__ == "__main__":
    unittest.main(verbosity=2)
