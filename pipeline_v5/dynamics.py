"""Past-only temporal summaries and landscape velocity for V5 topology."""
from __future__ import annotations

import numpy as np
import pandas as pd


def dynamic_summary(values: np.ndarray, windows=(4,8)) -> dict[str,float]:
    x=np.asarray(values,float); out={}
    for width in windows:
        tail=x[-width:]; valid=np.isfinite(tail)
        if valid.sum()>=3:
            t=np.arange(len(tail))[valid]
            out[f"slope{width}"]=float(np.polyfit(t,tail[valid],1)[0])
        else:out[f"slope{width}"]=np.nan
    tail=x[-8:]; valid=np.isfinite(tail); z=tail[valid]
    out["std8"]=float(np.std(z)) if len(z)>=3 else np.nan
    out["ac1_8"]=float(pd.Series(z).autocorr(1)) if len(z)>=3 else np.nan
    return out


def past_250(values: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    s=pd.Series(np.asarray(values,float))
    mean=s.rolling(250,min_periods=20).mean()
    std=s.rolling(250,min_periods=20).std(ddof=0)
    mx=s.rolling(250,min_periods=20).max()
    z=(s-mean)/std.replace(0,np.nan)
    z=z.mask(std.eq(0)&s.notna(),0.)
    return z.to_numpy(),(s-mx).to_numpy()


def landscape_velocity(vectors: np.ndarray, lag: int) -> np.ndarray:
    x=np.asarray(vectors,float); out=np.full(len(x),np.nan)
    for i in range(lag,len(x)):
        if np.isfinite(x[i]).all() and np.isfinite(x[i-lag]).all():
            out[i]=float(np.linalg.norm(x[i]-x[i-lag]))
    return out

