import unittest

import numpy as np

from pipeline_v5.topology import (correlation_distance,diagram_summary,
                                   raw_asset_distance)


class TopologySummaryTests(unittest.TestCase):
    def test_scale_exponents_for_raw_summaries(self):
        d=np.array([[0.,2.],[0.,1.],[0.,np.inf]])
        a=diagram_summary(d,1,"x")
        c=3.; b=diagram_summary(d*c,1,"x")
        self.assertAlmostEqual(b["x_total"],a["x_total"]*c)
        self.assertAlmostEqual(b["x_max"],a["x_max"]*c)
        self.assertAlmostEqual(b["x_l1"],a["x_l1"]*c**2)
        self.assertAlmostEqual(b["x_l2"],a["x_l2"]*c**1.5)
        self.assertAlmostEqual(b["x_entropy"],a["x_entropy"])

    def test_correlation_distance_scale_invariant(self):
        x=np.random.default_rng(11).normal(size=(35,12))
        a=correlation_distance(x); b=correlation_distance(3*x)
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)

    def test_raw_distance_scales_linearly(self):
        x=np.random.default_rng(23).normal(size=(60,9))
        a=raw_asset_distance(x); b=raw_asset_distance(3*x)
        np.testing.assert_allclose(b,3*a,atol=1e-12,rtol=1e-12)


if __name__=="__main__":unittest.main(verbosity=2)
