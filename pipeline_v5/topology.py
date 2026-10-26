"""V5 large-cloud persistent-homology summaries and scale diagnostics."""
from __future__ import annotations

import numpy as np
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

K_VALUES=(6,12,24,48,96,"ALL")
A_WINDOWS=(20,60,120,250)
T_WINDOWS=(60,120)


def finite_bars(diagram: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    if diagram.size==0:return np.empty((0,2)),np.empty(0)
    bars=diagram[np.isfinite(diagram).all(axis=1)]
    return bars,(bars[:,1]-bars[:,0]) if len(bars) else np.empty(0)


def diagram_summary(diagram: np.ndarray, homology: int, prefix: str) -> dict[str,float]:
    bars,p=finite_bars(diagram)
    total=float(p.sum()) if len(p) else 0.0
    q=p/total if total>0 else np.empty(0)
    out={f"{prefix}_total":total,f"{prefix}_max":float(p.max()) if len(p) else 0.0,
         f"{prefix}_entropy":float(-(q[q>0]*np.log(q[q>0])).sum()) if total>0 and len(p) else 0.0,
         f"{prefix}_count":int(len(p))}
    if homology in (1,2):
        out[f"{prefix}_l1"]=float(np.sum(p*p)/4.0) if len(p) else 0.0
        out[f"{prefix}_l2"]=float(np.sqrt(np.sum(p*p*p)/12.0)) if len(p) else 0.0
    return out


def correlation_distance(block: np.ndarray) -> np.ndarray:
    corr=np.corrcoef(block.T)
    corr=np.clip(np.nan_to_num(corr,nan=0.0,posinf=0.0,neginf=0.0),-1,1)
    np.fill_diagonal(corr,1.0)
    return np.sqrt(np.maximum(0,2*(1-corr)))


def raw_asset_distance(block: np.ndarray) -> np.ndarray:
    return squareform(pdist(block.T,metric="euclidean"))


def day_points(block: np.ndarray, kind: str) -> np.ndarray:
    """Rows are days; columns are selected assets."""
    if kind=="T_raw":
        scale=float(np.sqrt(np.mean(np.var(block,axis=0,ddof=0))))
        scale=scale if np.isfinite(scale) and scale>1e-12 else 1.0
        points=block/scale
        if points.shape[1]>24:
            points=PCA(n_components=min(10,points.shape[0],points.shape[1]),
                       svd_solver="full").fit_transform(points)
        return points*scale
    if kind!="T_std":raise ValueError(kind)
    # Exact V3 legacy construction: z-score each asset in the trailing window,
    # PCA to at most five components, then normalize by median nonzero distance.
    mean=block.mean(axis=0,keepdims=True); std=block.std(axis=0,keepdims=True)
    std[~np.isfinite(std)|(std<1e-12)]=1.0
    z=(block-mean)/std
    points=PCA(n_components=min(5,z.shape[0]-1,z.shape[1]),svd_solver="full").fit_transform(z)
    dist=pdist(points)
    med=float(np.median(dist[dist>0])) if np.any(dist>0) else 1.0
    return points/max(med,1e-12)


def cloud_matrix(block: np.ndarray, cloud: str) -> tuple[np.ndarray,bool]:
    if cloud=="A_corr":return correlation_distance(block),True
    if cloud=="A_raw":return raw_asset_distance(block),True
    points=day_points(block,cloud)
    return squareform(pdist(points)),True


def landscape_vector(diagram: np.ndarray, levels: int=3, grid_size: int=20) -> np.ndarray:
    bars,_=finite_bars(diagram)
    if not len(bars):return np.zeros(levels*grid_size,dtype=float)
    grid=np.linspace(float(bars[:,0].min()),float(bars[:,1].max()),grid_size)
    tents=np.maximum(0.,np.minimum(grid[None,:]-bars[:,0,None],bars[:,1,None]-grid[None,:]))
    ordered=np.sort(tents,axis=0)[::-1]
    out=np.zeros((levels,grid_size),float)
    out[:min(levels,len(ordered))]=ordered[:levels]
    return out.ravel()


def _cover_radius(result: dict, n: int) -> float:
    if "r_cover" in result:return float(result["r_cover"])
    if "idx_perm" not in result or n<=len(result["idx_perm"]):return 0.0
    idx=np.asarray(result["idx_perm"],int)
    # Fallback recomputation is only used if the installed ripser omits r_cover.
    return float("nan") if not len(idx) else 0.0


def persistence_features(block: np.ndarray, cloud: str, prefix: str,
                         maxdim: int, h2: bool, betti: bool=False,
                         landscape: bool=False) -> dict[str,float]:
    dist,_=cloud_matrix(block,cloud); n=len(dist); out={}
    # Normalize raw distance matrices before ripser. This makes greedy landmark
    # selection invariant to a change of return units; diagrams are restored to
    # the original distance scale immediately after computation.
    distance_scale=1.0
    if cloud in ("A_raw","T_raw"):
        variance=float(np.mean(np.var(block,axis=0,ddof=0)))
        if np.isfinite(variance) and variance>0:
            distance_scale=float(np.sqrt(variance))
    ripser_dist=dist/distance_scale
    landmark_h01=100 if n>100 else None
    result=ripser(ripser_dist,distance_matrix=True,maxdim=1,n_perm=landmark_h01)
    diagrams=[d*distance_scale for d in result["dgms"]]
    for h in (0,1):out.update(diagram_summary(diagrams[h],h,f"{prefix}_h{h}"))
    if landscape:
        vec=landscape_vector(diagrams[1])
        for i,value in enumerate(vec):out[f"{prefix}_h1_landscape_{i}"]=float(value)
    if landmark_h01 is not None:out[f"{prefix}_cover_radius_h01"]=_cover_radius(result,n)*distance_scale
    if h2:
        perm=60 if n>60 else None
        result2=ripser(ripser_dist,distance_matrix=True,maxdim=2,n_perm=perm)
        d2=result2["dgms"][2]*distance_scale
        out.update(diagram_summary(d2,2,f"{prefix}_h2"))
        if perm is not None:out[f"{prefix}_cover_radius_h2"]=_cover_radius(result2,n)*distance_scale
    if cloud in ("A_raw","T_raw"):
        variance=float(np.mean(np.var(block,axis=0,ddof=0)))
        if variance>0:
            for h in (1,2):
                key=f"{prefix}_h{h}"
                if f"{key}_l1" in out:out[f"{key}_l1_ratio"]=out[f"{key}_l1"]/variance
                # L2 persistence has units of distance^(3/2), so normalize by
                # standard-deviation^(3/2) = variance^(3/4) for scale invariance.
                if f"{key}_l2" in out:out[f"{key}_l2_ratio"]=out[f"{key}_l2"]/variance**0.75
                if f"{key}_total" in out:out[f"{key}_total_ratio"]=out[f"{key}_total"]/np.sqrt(variance)
    if betti:
        upper=dist[np.triu_indices(n,1)]
        bars,_=finite_bars(diagrams[1])
        for q,eps in enumerate(np.quantile(upper,np.arange(.1,.9,.1))):
            out[f"{prefix}_h1_betti_q{q+1}"]=float(((bars[:,0]<=eps)&(bars[:,1]>eps)).sum()) if len(bars) else 0.0
    return out


def bag_columns(available: np.ndarray, k: int|str, bags: int, seed: int) -> list[np.ndarray]:
    if k=="ALL":return [available.copy()] if len(available) else []
    k=int(k)
    if len(available)<k:return []
    if len(available)==k:return [available.copy()]
    rng=np.random.default_rng(seed+int(k))
    return [np.sort(rng.choice(available,k,replace=False)) for _ in range(bags)]


