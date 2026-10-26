"""Recompute selected H1 landscapes, then build V5 past-only dynamics."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ripser import ripser

from pipeline_v5.dynamics import dynamic_summary,landscape_velocity,past_250
from pipeline_v5.topology import K_VALUES,bag_columns,cloud_matrix,landscape_vector

ROOT=Path(__file__).resolve().parent; V3=ROOT/"data"/"v3"; V5=ROOT/"data"/"v5"
BLOCKS=V5/"blocks"; LAND_PATH=BLOCKS/"V5_landscape_sampled.parquet"
LAND_CLOUDS=("A_corr","A_raw","T_raw","T_std")


def landscapes_at_date(date,index,returns,nonstock,stocks,hist,bags):
    row={"date":date}; seed=int(pd.Timestamp(date).strftime("%Y%m%d")); colpos={c:i for i,c in enumerate(returns.columns)}
    # No 60-session topology cloud exists before this point. Python's negative
    # slice semantics otherwise turn early windows into empty PCA inputs.
    if index < 59:return row
    stock_order=sorted(stocks,key=lambda c:(-int(hist[index,colpos[c]]),c))[:250]
    for universe in ("NS","ALLS"):
        pool=nonstock if universe=="NS" else nonstock+stock_order
        blockdf=returns.iloc[index-59:index+1][pool]
        avail=np.flatnonzero(blockdf.notna().all(axis=0).to_numpy())
        if len(avail)<8:continue
        block=blockdf.to_numpy(float)
        for cloud in LAND_CLOUDS:
            for k in (24,"ALL"):
                if universe=="ALLS" and k!= "ALL":continue
                cols=bag_columns(avail,k,bags,seed)
                if not cols:continue
                prefix=f"v5_{universe}_{cloud}_k{str(k).lower()}_w60"
                vecs=[]
                for ci in cols:
                    sample=block[:,ci]
                    if not np.isfinite(sample).all():continue
                    dist,_=cloud_matrix(sample,cloud)
                    scale=float(np.sqrt(np.mean(np.var(sample,axis=0,ddof=0)))) if cloud in ("A_raw","T_raw") else 1.0
                    scale=scale if np.isfinite(scale) and scale>1e-12 else 1.0
                    perm=100 if len(dist)>100 else None
                    result=ripser(dist/scale,distance_matrix=True,maxdim=1,n_perm=perm)
                    vecs.append(landscape_vector(result["dgms"][1]*scale))
                if vecs:
                    avg=np.mean(vecs,axis=0)
                    for j,v in enumerate(avg):row[f"{prefix}_h1_landscape_{j}"]=float(v)
    return row


def _save_landscape(rows):
    if not rows:return
    new=pd.DataFrame(rows)
    if LAND_PATH.exists():new=pd.concat([pd.read_parquet(LAND_PATH),new],ignore_index=True).drop_duplicates("date",keep="last")
    new.sort_values("date",inplace=True,ignore_index=True); new.to_parquet(LAND_PATH,index=False)


def build_landscapes(returns,nonstock,stocks,hist,bags,timebox_min):
    raw=pd.read_parquet(BLOCKS/"V5_TDA_sampled.parquet").sort_values("date")
    dates=pd.DatetimeIndex(pd.to_datetime(raw.date)); last=pd.Timestamp("1900-01-01")
    if LAND_PATH.exists():
        old=pd.read_parquet(LAND_PATH)
        if len(old):last=pd.Timestamp(old.date.max())
    todo=dates[dates>last]; start=time.perf_counter(); deadline=start+timebox_min*60; rows=[]
    print(f"[V5-DYN] landscape backfill dates={len(todo)} after={last.date()}; workers=1",flush=True)
    for i,date in enumerate(todo,1):
        rows.append(landscapes_at_date(date,returns.index.get_loc(date),returns,nonstock,stocks,hist,bags))
        if i%20==0:
            _save_landscape(rows); rows=[]
            print(f"[V5-DYN] landscape checkpoint {i}/{len(todo)} elapsed={(time.perf_counter()-start)/60:.1f}m",flush=True)
        if time.perf_counter()>=deadline:break
    _save_landscape(rows)
    saved=pd.read_parquet(LAND_PATH)
    complete=(saved.date.max()>=raw.date.max())
    print(f"[V5-DYN] landscapes={len(saved):,}/{len(raw):,} complete={complete}; elapsed={(time.perf_counter()-start)/60:.1f}m",flush=True)
    return complete


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--landscape-timebox-min",type=float,default=45)
    ap.add_argument("--skip-landscape-build",action="store_true")
    ap.add_argument("--pilot",action="store_true");ap.add_argument("--smoke",action="store_true");args=ap.parse_args()
    raw=pd.read_parquet(BLOCKS/"V5_TDA_sampled.parquet").sort_values("date").reset_index(drop=True)
    returns=pd.read_parquet(V3/"returns.parquet").astype(float)
    reg=pd.read_csv(V3/"registry.csv").set_index("ticker")
    stocks=[c for c in returns if reg.loc[c,"asset_class"]=="stock"]
    nonstock=[c for c in returns if reg.loc[c,"asset_class"]!="stock"]
    hist=returns.notna().cumsum(axis=0).to_numpy(dtype=np.uint16)
    config=json.loads((V5/"feature_blocks_tda.json").read_text()).get("config",{})
    bags=int(config.get("bags",2)); stride=int(config.get("stride",10))
    if args.pilot or args.smoke:
        sample_n=20 if args.pilot else 1; dates=pd.DatetimeIndex(raw.date.iloc[-sample_n:])
        start=time.perf_counter();
        for j,date in enumerate(dates,1):
            landscapes_at_date(date,returns.index.get_loc(date),returns,nonstock,stocks,hist,bags)
        elapsed=time.perf_counter()-start
        estimate=elapsed*len(raw)/max(len(dates),1)/60
        print(f"[V5-DYN {'pilot' if args.pilot else 'smoke'}] windows={len(dates)} elapsed={elapsed/60:.2f}m; full landscape projection={estimate:.1f}m; bags={bags}, stride={stride}, workers=1")
        print(f"[V5-DYN {'pilot' if args.pilot else 'smoke'}] no feature outputs changed")
        return
    complete=True
    if not args.skip_landscape_build:
        complete=build_landscapes(returns,nonstock,stocks,hist,bags,args.landscape_timebox_min)
    land=pd.read_parquet(LAND_PATH).sort_values("date").reset_index(drop=True)
    landscape_cols=[c for c in land if "h1_landscape_" in c]
    merged=raw.merge(land,on="date",how="left",validate="one_to_one",suffixes=("","_land"))
    market=pd.read_parquet(V3/"market_features.parquet").reset_index()
    if "date" not in market:market.rename(columns={market.columns[0]:"date"},inplace=True)
    for src in ("market_mean_corr_w60","market_lambda1_share_w60"):
        if src in market:merged=merged.merge(market[["date",src]],on="date",how="left",validate="one_to_one")
    # Calculate sampled-point dynamics, then carry them forward over session dates.
    selected=[c for c in merged if c.startswith("v5_") and "_w60_h" in c and
              ("_h1_l1" in c or "_h1_l2" in c or "_h1_total" in c or "_h0_total" in c)]
    for src in ["market_mean_corr_w60","market_lambda1_share_w60"]:
        if src in merged:selected.append(src)
    sample=merged.sort_values("date").reset_index(drop=True); dyn=pd.DataFrame({"date":returns.index})
    dense=sample.set_index("date").reindex(returns.index).ffill()
    for src in dict.fromkeys(selected):
        vals=sample[src].to_numpy(float)
        sums=[dynamic_summary(vals[max(0,i-7):i+1]) for i in range(len(vals))]
        for metric in ("slope4","slope8","std8","ac1_8"):
            series=pd.Series([x[metric] for x in sums],index=pd.DatetimeIndex(sample.date))
            dyn[f"{src}_{metric}"]=series.reindex(returns.index).ffill().to_numpy()
        z,dm=past_250(dense[src].to_numpy(float))
        dyn[f"{src}_z250"]=z; dyn[f"{src}_distmax250"]=dm
    for prefix in sorted({c.rsplit("_",1)[0] for c in landscape_cols}):
        group=[x for x in landscape_cols if x.rsplit("_",1)[0]==prefix]
        vectors=land.set_index("date")[group].reindex(returns.index).ffill().to_numpy(float)
        for lag in (5,20):dyn[f"{prefix}_velocity{lag}"]=landscape_velocity(vectors,lag)
    for src in ("market_mean_corr_w60","market_lambda1_share_w60"):
        if src in dense:dyn[src]=dense[src].to_numpy(float)
    # Cross-asset dynamics inherit the TDA block's minimum universe size.
    for universe,assets in (("NS",nonstock),("ALLS",nonstock+stocks)):
        enough=returns[assets].notna().rolling(60,min_periods=60).sum().ge(8).sum(axis=1).ge(8).to_numpy()
        for col in dyn.columns:
            if col.startswith(f"v5_{universe}_"):
                dyn.loc[~enough,col]=np.nan
    out=BLOCKS/"V5_DYN.parquet"; dyn.to_parquet(out,index=False)
    (V5/"feature_blocks_dyn.json").write_text(json.dumps({"V5_DYN":[c for c in dyn if c!="date"],
        "config":{"stride":stride,"slope_points":[4,8],"landscape_lags_sessions":[5,20],
                  "landscapes_complete":complete}},indent=2),encoding="utf-8")
    print(f"[V5-DYN] rows={len(dyn):,} columns={len(dyn.columns)-1}; landscapes_complete={complete}; saved={out}")


if __name__=="__main__":main()
