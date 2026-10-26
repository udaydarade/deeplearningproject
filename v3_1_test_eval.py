import unittest

import numpy as np
import pandas as pd

from v3_1_eval import _counts_one, alarm_days, evaluate, off_budget


class MetricTests(unittest.TestCase):
    def frame(self, y, score=None):
        n = len(y)
        return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n),
                             "asset": "A", "y": y,
                             "score": np.asarray(score if score is not None else y, float)})

    def test_oracle(self):
        f = self.frame([0] * 90 + [1] * 10)
        a = alarm_days(f, "RV", .05, self.frame([0] * 100, [0] * 95 + [1] * 5))
        m = evaluate(f, a, sims=20)
        self.assertAlmostEqual(m["event_recall"], 1.0)
        self.assertAlmostEqual(m["realized_alarm_day_rate"], .1)

    def test_random_score_matches_baseline(self):
        rng = np.random.default_rng(5)
        pieces = []
        for i in range(50):
            y = np.tile([0] * 95 + [1] * 5, 5)
            g = self.frame(y, rng.random(len(y)))
            g["asset"] = f"A{i}"
            pieces.append(g)
        f = pd.concat(pieces, ignore_index=True)
        a = rng.random(len(f)) < .05
        m = evaluate(f, a, sims=500, seed=11)
        self.assertLess(abs(m["event_recall"] - m["rand_event_recall"]), .03)

    def test_always_alarm_is_off_budget(self):
        self.assertTrue(off_budget(1.0, .05))

    def test_hand_made_capped_runs(self):
        y = np.zeros(30, bool); y[2:4] = 1; y[20:23] = 1
        a = np.zeros(30, bool); a[3] = 1; a[8:10] = 1; a[22] = 1; a[29] = 1
        self.assertEqual(_counts_one(y, a), (2, 2, 3, 2, 1))
        y2 = np.zeros(40, bool); a2 = np.ones(40, bool)
        self.assertEqual(_counts_one(y2, a2)[2], 3)  # 15 + 15 + 10


if __name__ == "__main__":
    unittest.main(verbosity=2)
