import unittest

import numpy as np

from pipeline_v5.dynamics import dynamic_summary,landscape_velocity,past_250


class DynamicsTests(unittest.TestCase):
    def test_past_250_has_no_future_dependence(self):
        x=np.arange(400,dtype=float);y=x.copy();y[300:]=-1e6
        zx,dx=past_250(x);zy,dy=past_250(y)
        np.testing.assert_allclose(zx[:300],zy[:300],equal_nan=True)
        np.testing.assert_allclose(dx[:300],dy[:300],equal_nan=True)

    def test_summary_uses_trailing_values(self):
        x=np.arange(10,dtype=float);m=dynamic_summary(x)
        self.assertAlmostEqual(m["slope4"],1.)
        self.assertAlmostEqual(m["slope8"],1.)
        changed=np.r_[np.arange(9,dtype=float),1000.]
        self.assertNotEqual(dynamic_summary(changed)["slope4"],m["slope4"])

    def test_landscape_velocity_lags(self):
        x=np.arange(100,dtype=float)[:,None]
        v=landscape_velocity(x,5)
        self.assertTrue(np.isnan(v[:5]).all())
        np.testing.assert_allclose(v[5:],5.)


if __name__=="__main__":unittest.main(verbosity=2)
