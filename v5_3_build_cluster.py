"""Stride-1 full-history cross-asset clustering/network blocks, checkpointed."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_v5.cluster import distance_features,network_features

ROOT=Path(__file__).resolve().parent; V3=ROOT/"data"/"v3"; OUT=ROOT/"data"/"v5"
PATH=OUT/"blocks"/"V5_CLUSTER.parquet"


def one_date(returns: pd.DataFrame,index: int)->dict:
    date=returns.index[index]; out={"date":date}
    for w in (60,120):
        if index+1<w:continue
        block=returns.iloc[index-w+1:index+1].to_numpy(np.float64)
        avail=np.flatnonzero(np.isfinite(block).all(axis=0))
        if len(avail)<8:continue
        corr=np.corrcoef(block[:,avail].T)
        corr=np.clip(np.nan_to_num(corr,nan=0.,posinf=0.,neginf=0.),-1.,1.)
        np.fill_diagonal(corr,1.)
        dist=np.sqrt(np.maximum(0.,2*(1-corr)))
        prefix=f"v5_cluster_w{w}"
        out.update(distance_features(dist,prefix))
        out.update(network_features(corr,prefix))
    return out


def save(rows:list[dict]):
    if not rows:return
    new=pd.DataFrame(rows)
    if PATH.exists():
        old=pd.read_parquet(PATH); new=pd.concat([old,new],ignore_index=True).drop_duplicates("date",keep="last")
    new.sort_values("date",inplace=True,ignore_index=True); new.to_parquet(PATH,index=False)
    meta={"V5_CLUSTER":[c for c in new if c!="date"],"config":{"stride":1,"universe":"all complete assets","checkpointed":True}}
    (OUT/"feature_blocks_cluster.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--timebox-min",type=float,default=100)
    ap.add_argument("--smoke",action="store_true"); args=ap.parse_args()
    (OUT/"blocks").mkdir(parents=True,exist_ok=True)
    returns=pd.read_parquet(V3/"returns.parquet").astype("float64")
    dates=returns.index; last=pd.Timestamp("1900-01-01")
    if PATH.exists():
        saved=pd.read_parquet(PATH)
        if len(saved):last=pd.Timestamp(saved.date.max())
    ix=np.flatnonzero((dates>last)&(dates>=pd.Timestamp("1928-01-01")))
    if args.smoke:ix=ix[-3:]
    deadline=time.perf_counter()+args.timebox_min*60; start=time.perf_counter(); rows=[]
    print(f"[V5-CLUSTER] stride=1, dates={len(ix)}, resumes after={last.date()}, workers=1",flush=True)
    for j,i in enumerate(ix,1):
        rows.append(one_date(returns,int(i)))
        if j%100==0:
            save(rows); rows=[]
            print(f"[V5-CLUSTER] checkpoint {j}/{len(ix)} elapsed={(time.perf_counter()-start)/60:.1f}min",flush=True)
        if time.perf_counter()>=deadline:break
    if args.smoke:
        print(f"[V5-CLUSTER smoke] computed {len(rows)} dates in {(time.perf_counter()-start):.2f}s; outputs not saved",flush=True)
        return
    save(rows)
    current=pd.read_parquet(PATH) if PATH.exists() else pd.DataFrame()
    print(f"[V5-CLUSTER] saved {len(current):,} dates; elapsed={(time.perf_counter()-start)/60:.1f}min",flush=True)


if __name__=="__main__":main()
