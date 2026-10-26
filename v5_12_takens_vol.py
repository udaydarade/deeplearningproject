"""P2 per-asset Takens embeddings of past five-day realized volatility."""
from __future__ import annotations
import argparse,time
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ripser import ripser
from scipy.spatial.distance import pdist,squareform

ROOT=Path(__file__).resolve().parent;V3=ROOT/"data"/"v3";OUT=ROOT/"data"/"v5"/"blocks"/"V5_TAKENS_VOL"

def build_asset(ticker,returns,stride=5):
    r=returns.to_numpy(float);rv5=pd.Series(r).rolling(5,min_periods=5).std().to_numpy()
    s=np.log(rv5+1e-8);delay=np.full(len(s),10,np.int8)
    for i in range(249,len(s)):
        x=s[i-249:i+1];chosen=10
        for lag in range(1,11):
            a=x[:-lag];b=x[lag:];good=np.isfinite(a)&np.isfinite(b);a=a[good];b=b[good]
            if len(a)<3 or np.std(a)==0 or np.std(b)==0:continue
            ac=float(np.corrcoef(a,b)[0,1])
            if np.isfinite(ac) and ac<1/np.e:chosen=lag;break
        delay[i]=chosen
    rows=[]
    for i in range(249,len(s),stride):
        tau=int(delay[i]);end=np.arange(i-59,i+1);idx=np.column_stack((end,end-tau,end-2*tau))
        if idx.min()<0:continue
        points=s[idx]
        if not np.isfinite(points).all():continue
        dgms=ripser(squareform(pdist(points)),distance_matrix=True,maxdim=1)["dgms"]
        bars=dgms[1];bars=bars[np.isfinite(bars).all(axis=1)]
        p=bars[:,1]-bars[:,0] if len(bars) else np.empty(0)
        total=float(p.sum());q=p/total if total>0 else np.empty(0)
        rows.append({"date":returns.index[i],"asset":ticker,"v5_takens_vol_h1_total":total,
            "v5_takens_vol_h1_max":float(p.max()) if len(p) else 0.,
            "v5_takens_vol_h1_l2":float(np.sqrt(np.sum(p**3)/12)) if len(p) else 0.,
            "v5_takens_vol_h1_entropy":float(-(q[q>0]*np.log(q[q>0])).sum()) if len(q) else 0.,
            "v5_takens_vol_h1_count":int(len(p)),"delay":tau})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--timebox-min",type=float,default=90)
    ap.add_argument("--smoke",action="store_true");ap.add_argument("--stride",type=int,default=5);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);r=pd.read_parquet(V3/"returns.parquet").astype(float)
    if args.smoke:
        a=max(r.columns,key=lambda c:r[c].notna().sum());sample=r[a].dropna().tail(400)
        start=time.perf_counter();result=build_asset(a,sample,args.stride);elapsed=time.perf_counter()-start
        estimate=elapsed*len(r.columns)*len(r)/(max(len(sample),1)*60)
        print(f"[V5-TAKENS smoke] {a} sample={len(sample)} stride={args.stride}, elapsed={elapsed:.2f}s; projected full={estimate:.1f}m; no files saved")
        return
    start=time.perf_counter();deadline=start+args.timebox_min*60
    assets=list(r.columns)
    todo=[a for a in assets if not (OUT/f"{a.replace('^','').replace('/','_')}.parquet").exists()]
    print(f"[V5-TAKENS] assets={len(todo)}/{len(assets)} stride={args.stride} estimate={len(todo)*.5:.0f}m; box={args.timebox_min:g}m; one worker",flush=True)
    for j,a in enumerate(todo,1):
        x=build_asset(a,r[a],args.stride)
        x.to_parquet(OUT/f"{a.replace('^','').replace('/','_')}.parquet",index=False)
        print(f"[V5-TAKENS] {j}/{len(todo)} {a} rows={len(x)} elapsed={(time.perf_counter()-start)/60:.1f}m",flush=True)
        if time.perf_counter()>=deadline:break
    done=list(OUT.glob("*.parquet"))
    if done:
        sample=pd.read_parquet(done[0]);features=[c for c in sample if c not in {"date","asset","delay"}]
        (OUT.parent.parent/"feature_blocks_takens.json").write_text(
            json.dumps({"V5_TAKENS_VOL":features,"config":{"delay_max":10,"embedding_dimension":3,
                "window":60,"stride":args.stride,"assets_completed":len(done),"assets_total":len(r.columns)}},indent=2),encoding="utf-8")
    print(f"[V5-TAKENS] saved={len(list(OUT.glob('*.parquet')))} assets; elapsed={(time.perf_counter()-start)/60:.1f}m",flush=True)

if __name__=="__main__":main()
