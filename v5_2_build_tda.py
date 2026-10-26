"""Build V5 full-history multi-size topology blocks; resumable time-boxed checkpoints."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_v5.topology import (A_WINDOWS,K_VALUES,T_WINDOWS,bag_columns,
                                  persistence_features)

ROOT=Path(__file__).resolve().parent; V3=ROOT/"data"/"v3"; OUT=ROOT/"data"/"v5"
BLOCKS=OUT/"blocks"; SAMPLED=BLOCKS/"V5_TDA_sampled.parquet"


def _key_k(k: int|str)->str:return str(k).lower() if k=="ALL" else str(k)


def _bag_average(block: np.ndarray, cloud: str, prefix: str, h2: bool,
                 betti: bool, landscape: bool)->dict:
    return persistence_features(block,cloud,prefix,maxdim=2 if h2 else 1,h2=h2,
                                betti=betti,landscape=landscape)


def features_at_date(date: pd.Timestamp, index: int, returns: pd.DataFrame,
                     nonstock: list[str], stocks: list[str], history_counts: np.ndarray,
                     bags: int) -> dict[str,float]:
    row={"date":date}; date_seed=int(pd.Timestamp(date).strftime("%Y%m%d"))
    maxwin=max(A_WINDOWS)
    start=max(0,index-maxwin+1)
    colpos={c:i for i,c in enumerate(returns.columns)}
    recent_stocks=sorted(stocks,key=lambda c:(-int(history_counts[index,colpos[c]]),c))[:250]
    for universe in ("NS","ALLS"):
        asset_pool=nonstock if universe=="NS" else nonstock+recent_stocks
        for w in sorted(set(A_WINDOWS+T_WINDOWS)):
            if index+1<w: continue
            block_df=returns.iloc[index-w+1:index+1][asset_pool]
            complete=block_df.notna().all(axis=0).to_numpy()
            avail=np.flatnonzero(complete)
            if universe=="ALLS" and len(avail):
                # Stocks are already ordered by trailing available-history length;
                # the project universe currently has fewer than the 250 cap.
                ns_idx=np.flatnonzero(np.isin(np.asarray(asset_pool),np.asarray(nonstock)))
                st_idx=np.flatnonzero(np.isin(np.asarray(asset_pool),np.asarray(stocks)))
                avail=np.r_[np.intersect1d(avail,ns_idx),np.intersect1d(avail,st_idx[:250])]
                avail=np.unique(avail)
            block_all=block_df.to_numpy(np.float64)
            # The V5 protocol treats small cross-sections as unavailable, even
            # when an individual k would technically fit.
            if len(avail)<8: continue
            clouds=[]
            if w in A_WINDOWS:clouds.extend(("A_corr","A_raw"))
            if w in T_WINDOWS:clouds.extend(("T_raw","T_std"))
            for cloud in clouds:
                ks=K_VALUES
                for k in ks:
                    if universe=="ALLS" and k!="ALL": continue
                    if cloud.startswith("A") and w not in A_WINDOWS: continue
                    if cloud.startswith("T") and w not in T_WINDOWS: continue
                    prefix=f"v5_{universe}_{cloud}_k{_key_k(k)}_w{w}"
                    h2=(w==60 and (k in (12,24,48) or (k=="ALL" and cloud=="A_corr")))
                    cols=bag_columns(avail,k,bags,date_seed)
                    if not cols: continue
                    bag_rows=[]
                    for col_idx in cols:
                        sample=block_all[:,col_idx]
                        if not np.isfinite(sample).all(): continue
                        try:
                            bag_rows.append(_bag_average(sample,cloud,prefix,h2,
                                betti=(cloud=="A_corr"),
                                landscape=(w==60 and k in (24,"ALL"))))
                        except (ValueError,np.linalg.LinAlgError,FloatingPointError):
                            continue
                    if bag_rows:
                        keys=sorted(set().union(*(x.keys() for x in bag_rows)))
                        for key in keys:
                            vals=np.asarray([x.get(key,np.nan) for x in bag_rows],float)
                            row[key]=float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
    return row


def save_checkpoint(rows: list[dict], returns_index: pd.DatetimeIndex,
                    stride: int, bags: int) -> None:
    if not rows:return
    new=pd.DataFrame(rows)
    if SAMPLED.exists():
        old=pd.read_parquet(SAMPLED)
        all_rows=pd.concat([old,new],ignore_index=True).drop_duplicates("date",keep="last")
    else: all_rows=new
    all_rows.sort_values("date",inplace=True,ignore_index=True)
    all_rows.to_parquet(SAMPLED,index=False)
    dense=(all_rows.set_index("date").reindex(returns_index).ffill())
    dense=dense.loc[dense.index>=all_rows.date.min()].reset_index(names="date")
    dense.to_parquet(BLOCKS/"V5_TDA.parquet",index=False)
    meta={"V5_TDA":[c for c in dense if c!="date"],
          "config":{"stride":stride,"bags":bags,"stock_cap":250,
                    "date_first":str(all_rows.date.min().date()),"date_last":str(all_rows.date.max().date()),
                    "min_assets":8,"l2_ratio_scale":"variance^0.75","checkpointed":True}}
    (OUT/"feature_blocks_tda.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pilot",action="store_true")
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--stride",type=int,default=5)
    ap.add_argument("--bags",type=int,default=3)
    ap.add_argument("--timebox-min",type=float,default=100)
    ap.add_argument("--limit-dates",type=int,default=0)
    args=ap.parse_args()
    BLOCKS.mkdir(parents=True,exist_ok=True)
    returns=pd.read_parquet(V3/"returns.parquet").astype("float64")
    reg=pd.read_csv(V3/"registry.csv").set_index("ticker")
    stocks=[c for c in returns if reg.loc[c,"asset_class"]=="stock"]
    nonstock=[c for c in returns if reg.loc[c,"asset_class"]!="stock"]
    # deterministic longest-history ordering for the stock cap
    counts=returns[stocks].notna().sum(axis=0).sort_values(ascending=False,kind="stable")
    stocks=list(counts.index[:250])
    eligible=returns.index[returns.index>=pd.Timestamp("1928-01-01")][::args.stride]
    if args.limit_dates: eligible=eligible[:args.limit_dates]
    if args.pilot or args.smoke:
        sample_n=20 if args.pilot else 1
        dates=eligible[-sample_n:]
        rows=[]; started=time.perf_counter()
        hist=returns.notna().cumsum(axis=0).to_numpy(dtype=np.uint16)
        for j,date in enumerate(dates,1):
            rows.append(features_at_date(date,returns.index.get_loc(date),returns,nonstock,stocks,hist,args.bags))
            if j%5==0 or args.smoke: print(f"[V5-TDA {'smoke' if args.smoke else 'pilot'}] {j}/{sample_n} windows elapsed={(time.perf_counter()-started)/60:.1f}min",flush=True)
        elapsed=time.perf_counter()-started
        dates_total=len(eligible); est=elapsed*dates_total/max(len(dates),1)/60
        print(f"[V5-TDA {'smoke' if args.smoke else 'pilot'}] stride={args.stride} bags={args.bags}; projected full run={est:.1f}min for {dates_total} sampled dates")
        if args.pilot:print(f"[V5-TDA pilot] bags=2 projection={est*2/args.bags:.1f}min; then stride=10 projection={est*2/args.bags*5/10:.1f}min")
        print(f"[V5-TDA {'smoke' if args.smoke else 'pilot'}] computed feature columns={len(set().union(*(r.keys() for r in rows)))-1}; no output features saved")
        return
    hist=returns.notna().cumsum(axis=0).to_numpy(dtype=np.uint16)
    last=pd.Timestamp("1900-01-01")
    if SAMPLED.exists():
        old=pd.read_parquet(SAMPLED)
        if len(old):last=pd.Timestamp(old.date.max())
    dates=eligible[eligible>last]
    started=time.perf_counter(); deadline=started+args.timebox_min*60; rows=[]
    print(f"[V5-TDA] start stride={args.stride} bags={args.bags}, sampled dates={len(eligible)}, resumes after={last.date()}, workers=1",flush=True)
    for j,date in enumerate(dates,1):
        rows.append(features_at_date(date,returns.index.get_loc(date),returns,nonstock,stocks,hist,args.bags))
        if j%20==0:
            save_checkpoint(rows,returns.index,args.stride,args.bags); rows=[]
            elapsed=(time.perf_counter()-started)/60
            print(f"[V5-TDA] checkpoint {j}/{len(dates)} elapsed={elapsed:.1f}min",flush=True)
        if time.perf_counter()>=deadline:break
    save_checkpoint(rows,returns.index,args.stride,args.bags)
    done=pd.read_parquet(SAMPLED) if SAMPLED.exists() else pd.DataFrame()
    print(f"[V5-TDA] timebox end; saved sampled dates={len(done):,}; elapsed={(time.perf_counter()-started)/60:.1f}min",flush=True)


if __name__=="__main__":main()
