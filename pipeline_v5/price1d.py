"""Stride-1 one-dimensional sublevel persistence with a path union-find."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _find(parent, x):
    root=x
    while parent[root]!=root:root=parent[root]
    while parent[x]!=x:
        nxt=parent[x]; parent[x]=root; x=nxt
    return root


@njit(cache=True)
def _union_find_bars(values: np.ndarray) -> np.ndarray:
    n=len(values)
    order=np.argsort(values)
    active=np.zeros(n,np.uint8); parent=np.arange(n)
    birth=values.copy(); bars=np.empty((max(0,n-1),2),np.float64); count=0
    for oi in range(n):
        i=order[oi]; active[i]=1; parent[i]=i; birth[i]=values[i]
        for j in (i-1,i+1):
            if j<0 or j>=n or active[j]==0:continue
            ri=_find(parent,i); rj=_find(parent,j)
            if ri==rj:continue
            if birth[ri]<=birth[rj]:
                bars[count,0]=birth[rj]; bars[count,1]=values[i]; count+=1
                parent[rj]=ri
            else:
                bars[count,0]=birth[ri]; bars[count,1]=values[i]; count+=1
                parent[ri]=rj
    return bars[:count]


def persistence_bars(values: np.ndarray) -> np.ndarray:
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError("persistence input must be a finite 1-D window of length >=2")
    bars=_union_find_bars(x)
    return bars[bars[:,1]>bars[:,0]]


def _summarize(bars: np.ndarray, scale: float, prefix: str) -> dict[str,float]:
    lengths=np.maximum(0.,bars[:,1]-bars[:,0]) if len(bars) else np.empty(0)
    total=float(lengths.sum())
    p=lengths[lengths>0]
    q=p/p.sum() if len(p) and p.sum()>0 else np.empty(0)
    mx=float(lengths.max()) if len(lengths) else 0.0
    entropy=float(-(q*np.log(q)).sum()) if len(q) else 0.0
    scale=float(scale) if np.isfinite(scale) and scale>0 else 0.0
    return {f"{prefix}_total":total,f"{prefix}_max":mx,
            f"{prefix}_entropy":entropy,
            f"{prefix}_bars_over_scale":int((lengths>scale).sum()),
            f"{prefix}_max_over_scale":mx/scale if scale>0 else 0.0}


def price1d_window(log_price: np.ndarray, log_returns: np.ndarray, window: int,
                   prefix: str) -> dict[str,float]:
    p=np.asarray(log_price,float); r=np.asarray(log_returns,float)
    if len(p)!=window or len(r)<window-1:raise ValueError("window length mismatch")
    scale=float(np.nanstd(r[-(window-1):],ddof=0)*np.sqrt(window))
    # Translation does not change persistence; centering reduces numerical scale.
    centered=p-p[0]
    valleys=persistence_bars(centered)
    peaks=persistence_bars(-centered)
    out={}
    out.update(_summarize(valleys,scale,f"{prefix}_valley"))
    out.update(_summarize(peaks,scale,f"{prefix}_peak"))
    return out

