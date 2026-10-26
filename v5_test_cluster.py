import unittest

import numpy as np

from pipeline_v5.cluster import distance_features,network_features


class ClusterBlockTests(unittest.TestCase):
    def test_triangle_density_and_components(self):
        corr=np.full((3,3),.5); np.fill_diagonal(corr,1.)
        x=network_features(corr,"x")
        self.assertAlmostEqual(x["x_edge_density_r0.5"],1.)
        self.assertAlmostEqual(x["x_triangle_density_r0.5"],1.)
        self.assertAlmostEqual(x["x_giant_share_r0.5"],1.)
        self.assertAlmostEqual(x["x_giant_share_r0.7"],1/3)

    def test_mst_and_cluster_summary(self):
        d=np.array([[0.,.2,.9],[.2,0.,.8],[.9,.8,0.]])
        x=distance_features(d,"x")
        self.assertAlmostEqual(x["x_mst_total"],1.)
        self.assertEqual(x["x_cluster_count_d0.8"],2)
        self.assertEqual(x["x_cluster_count_d1.0"],1)


if __name__=="__main__":unittest.main(verbosity=2)
