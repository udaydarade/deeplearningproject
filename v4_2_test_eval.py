import unittest
import numpy as np
import pandas as pd

from v4_eval import evaluate, off_budget, rolling_alarm


class V4MetricTests(unittest.TestCase):
    def frame(self, y, score=None):
        n = len(y)
        return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n),
                             "asset": "A", "y": np.asarray(y, int),
                             "score": np.asarray(score if score is not None else y, float)})

    def test_oracle_score(self):
        f = self.frame([0] * 90 + [1] * 10)
        out = evaluate(rolling_alarm(f, .05), simulations=20)
        self.assertGreater(out["ap"], .99)

    def test_random_scores(self):
        rng = np.random.default_rng(5)
        y = np.tile([0] * 95 + [1] * 5, 5)
        f = self.frame(y, rng.random(len(y)))
        out = evaluate(f.assign(alarm=rng.random(len(f)) < .05), simulations=200)
        self.assertLess(abs(out["event_recall"] - out["rand_event_recall"]), .4)

    def test_always_alarm_off_budget(self):
        self.assertTrue(off_budget(1.0, .05))

    def test_hand_counted_toy(self):
        y = [0] * 30
        y[2:4] = [1, 1]
        y[20:23] = [1, 1, 1]
        alarms = [False] * 30
        alarms[3] = alarms[8] = alarms[9] = alarms[22] = alarms[29] = True
        f = self.frame(y, np.linspace(0, 1, 30)).assign(alarm=alarms)
        out = evaluate(f, simulations=20)
        self.assertEqual(out["crashes_total"], 2)
        self.assertEqual(out["crashes_caught"], 2)
        self.assertEqual(out["alarms_total"], 3)
        self.assertEqual(out["alarms_correct"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
