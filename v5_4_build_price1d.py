"""Stride-1 per-asset PRICE1D persistence features, checkpointed by asset."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_v5.price1d import price1d_window

ROOT=Path(__file__).resolve().parent; V3=ROOT/"data"/"v3"; OUT=ROOT/"data"/"v5"/"blocks"/"V5_PRICE1D"
WINDOWS=(20,60,120)


def build_asset(ticker:str, prices:pd.Series)->pd.DataFrame:
    p=prices.to_numpy(np.float64); dates=prices.index
    good=np.isfinite(p)&(p>0); logp=np.full(len(p),np.nan); logp[good]=np.log(p[good])
    rows=[]
    for i in range(len(p)):
        if not good[i]:continue
        row={"date":dates[i],"asset":ticker}
        for w in WINDOWS:
            if i+1<w:continue
            lp=logp[i-w+1:i+1]
            if not np.isfinite(lp).all():continue
            ret=np.diff(lp)
            row.update(price1d_window(lp,ret,w,f"v5_price1d_w{w}"))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--timebox-min",type=float,default=100)
    ap.add_argument("--smoke",action="store_true"); args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    prices=pd.read_parquet(V3/"prices.parquet").astype("float64")
    if args.smoke:
        ticker=max(prices.columns,key=lambda c:prices[c].notna().sum())
        s=prices[ticker].dropna().tail(250)
        started=time.perf_counter(); _=build_asset(ticker,s)
        elapsed=time.perf_counter()-started
        print(f"[V5-PRICE1D smoke] asset={ticker}, dates={len(s)}, elapsed={elapsed:.2f}s; no output saved")
        print(f"[V5-PRICE1D smoke] full estimate scales by {prices.shape[1]} assets and {len(prices):,} dates; workers=1")
        return
    start=time.perf_counter(); deadline=start+args.timebox_min*60
    todo=[c for c in prices.columns if not (OUT/f"asset_{c.replace('^','').replace('/','_')}.parquet").exists()]
    print(f"[V5-PRICE1D] total assets={prices.shape[1]}, remaining={len(todo)}, dates={len(prices):,}, workers=1",flush=True)
    for j,ticker in enumerate(todo,1):
        frame=build_asset(ticker,prices[ticker])
        name=f"asset_{ticker.replace('^','').replace('/','_')}.parquet"
        frame.to_parquet(OUT/name,index=False)
        if j%10==0:print(f"[V5-PRICE1D] checkpoint assets={j}/{len(todo)} elapsed={(time.perf_counter()-start)/60:.1f}min",flush=True)
        if time.perf_counter()>=deadline:break
    done=len(list(OUT.glob("asset_*.parquet")))
    print(f"[V5-PRICE1D] saved assets={done}/{prices.shape[1]}, elapsed={(time.perf_counter()-start)/60:.1f}min",flush=True)


if __name__=="__main__":main()
