"""Cross-asset MST, average-linkage, and correlation-network summaries."""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster,linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components,minimum_spanning_tree
from scipy.spatial.distance import squareform


def _entropy(counts: np.ndarray) -> float:
    if not len(counts):return 0.0
    p=counts/counts.sum(); return float(-(p[p>0]*np.log(p[p>0])).sum())


def distance_features(distance: np.ndarray, prefix: str) -> dict[str,float]:
    n=len(distance); out={}
    if n<2:return out
    out[f"{prefix}_mst_total"]=float(minimum_spanning_tree(distance).sum())
    z=linkage(squareform(distance,checks=False),method="average")
    for cut in (.8,1.0):
        labels=fcluster(z,t=cut,criterion="distance")
        _,sizes=np.unique(labels,return_counts=True)
        out[f"{prefix}_cluster_count_d{cut:.1f}"]=int(len(sizes))
        out[f"{prefix}_largest_share_d{cut:.1f}"]=float(sizes.max()/n)
        out[f"{prefix}_cluster_entropy_d{cut:.1f}"]=_entropy(sizes.astype(float))
    return out


def network_features(corr: np.ndarray, prefix: str) -> dict[str,float]:
    n=len(corr); out={}
    if n<2:return out
    possible=n*(n-1)/2
    for rho in (.3,.5,.7,.9):
        adj=(corr>=rho); np.fill_diagonal(adj,False)
        upper=np.triu(adj,1); edges=int(upper.sum())
        ncomp,labels=connected_components(csr_matrix(adj),directed=False,return_labels=True)
        sizes=np.bincount(labels,minlength=ncomp)
        out[f"{prefix}_edge_density_r{rho:.1f}"]=float(edges/possible)
        out[f"{prefix}_giant_share_r{rho:.1f}"]=float(sizes.max()/n)
        if n>=3:
            tri=int(round(float(np.trace(adj.astype(np.int64)@adj.astype(np.int64)@adj.astype(np.int64)))/6))
            out[f"{prefix}_triangle_density_r{rho:.1f}"]=float(tri/(n*(n-1)*(n-2)/6))
        else:out[f"{prefix}_triangle_density_r{rho:.1f}"]=0.0
    return out

