"""Descriptive V5 raw-scale, proxy-R2, and single-feature AUC diagnostics."""
from __future__ import annotations
import re,time,json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from v5_eval import ERAS

ROOT=Path(__file__).resolve().parent;V3=ROOT/"data"/"v3";V5=ROOT/"data"/"v5";B=V5/"blocks"
RAW=re.compile(r"^v5_(NS|ALLS)_(A_raw|T_raw)_k([^_]+)_w(\d+)_h([012])_(total|max|l1|l2)$")

def r2(x,y):
    z=np.column_stack([np.ones(len(x)),x]);coef=np.linalg.lstsq(z,y,rcond=None)[0]
    pred=z@coef;den=np.sum((y-y.mean())**2)
    return 1-float(np.sum((y-pred)**2)/den) if den>0 else np.nan

def auc(y,x):
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<20 or np.unique(y[ok]).size<2:return np.nan
    try:return float(roc_auc_score(y[ok],x[ok]))
    except ValueError:return np.nan

def blockname(col):
    m=re.match(r"^v5_(?:NS|ALLS)_(A_corr|A_raw|T_raw|T_std)_",col)
    return m.group(1) if m else ("reference" if not col.startswith("v5_") else col.split("_")[1])

def main():
    timebox=30.;start=time.perf_counter();deadline=start+timebox*60
    tda=pd.read_parquet(B/"V5_TDA_sampled.parquet").sort_values("date").reset_index(drop=True)
    ret=pd.read_parquet(V3/"returns.parquet").astype(float);reg=pd.read_csv(V3/"registry.csv").set_index("ticker")
    ns=[c for c in ret if reg.loc[c,"asset_class"]!="stock"];assets=ns+[c for c in ret if reg.loc[c,"asset_class"]=="stock"]
    market=pd.read_parquet(V3/"market_features.parquet").reset_index()
    if "date" not in market:market.rename(columns={market.columns[0]:"date"},inplace=True)
    eq=ret[ns].mean(axis=1,skipna=True);market_vol=eq.rolling(20,min_periods=20).std(ddof=0).rename("v20_market")
    proxies=pd.DataFrame({"date":ret.index,"v20_market":market_vol.to_numpy()}).merge(
        market[["date","market_mean_corr_w60","market_lambda1_share_w60"]],on="date",how="left")
    # C1: expected volatility-only exponents for unnormalized raw persistence scalars.
    var_rows=[];variance={}
    for u,pool in (("NS",ns),("ALLS",assets)):
        for w in (20,60,120,250):
            vv=ret[pool].rolling(w,min_periods=w).var(ddof=0).mean(axis=1)
            count=ret[pool].notna().rolling(w,min_periods=w).sum().eq(w).sum(axis=1)
            vv[count<8]=np.nan;variance[(u,w)]=vv
    raw_rows=[]
    for col in [c for c in tda if c!="date"]:
        m=RAW.match(col)
        if not m:continue
        u,cloud,k,w,h,metric=m.groups();x=variance[(u,int(w))].reindex(pd.DatetimeIndex(tda.date)).to_numpy(float)
        y=tda[col].to_numpy(float);ok=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
        if ok.sum()<20:continue
        slope=float(np.polyfit(np.log(x[ok]),np.log(y[ok]),1)[0]);expected={"l1":1.,"l2":.75,"total":.5,"max":.5}[metric]
        raw_rows.append({"feature":col,"universe":u,"cloud":cloud,"k":k,"window":int(w),"homology":int(h),
                         "metric":metric,"n":int(ok.sum()),"loglog_slope":slope,"volatility_only_expected":expected,
                         "slope_gap":slope-expected,"r2_loglog":r2(np.log(x[ok]),np.log(y[ok]))})
    pd.DataFrame(raw_rows).to_csv(V5/"diagnostic_c1_raw_scaling.csv",index=False)
    # C2: feature R2 against three common-market state proxies.
    c2=[];proxy=proxies.set_index("date").reindex(pd.DatetimeIndex(tda.date))
    xx=proxy[["v20_market","market_mean_corr_w60","market_lambda1_share_w60"]].to_numpy(float)
    for j,col in enumerate(c for c in tda if c!="date"):
        y=tda[col].to_numpy(float);ok=np.isfinite(y)&np.isfinite(xx).all(axis=1)
        score=r2(xx[ok],y[ok]) if ok.sum()>20 else np.nan
        c2.append({"feature":col,"block":blockname(col),
                   "n":int(ok.sum()),"proxy_r2":score,"flag_r2_gt_0_8":bool(score>.8) if np.isfinite(score) else False})
        if j and j%250==0 and time.perf_counter()>=deadline:break
    c2f=pd.DataFrame(c2);c2f.to_csv(V5/"diagnostic_c2_proxy_r2.csv",index=False)
    c2f.groupby("block",as_index=False).agg(features=("feature","count"),flagged=("flag_r2_gt_0_8","sum")).assign(
        flagged_share=lambda z:z.flagged/z.features.replace(0,np.nan)).to_csv(V5/"diagnostic_c2_block_share.csv",index=False)
    # C3: signed single-feature AUC by era and market-vol tercile; train cutoffs exclude test era.
    panel=pd.read_parquet(V3/"panel.parquet",columns=["date","asset","y_BASE","realized_vol_w20","price_drawdown_60d"])
    panel.date=pd.to_datetime(panel.date);panel=panel[panel.y_BASE.notna()].sort_values(["date","asset"]).reset_index(drop=True)
    vol=proxies.set_index("date").v20_market
    panel["market_vol"]=panel.date.map(vol)
    panel["market_mean_corr_w60"]=panel.date.map(proxies.set_index("date").market_mean_corr_w60)
    features=[c for c in tda if c!="date"]
    auc_rows=[];refs=["realized_vol_w20","price_drawdown_60d","market_mean_corr_w60"]
    all_features=[*refs,*features]
    tda_idx=tda.set_index("date")
    for j,col in enumerate(all_features):
        if time.perf_counter()>=deadline:break
        x=(panel[col].to_numpy(float) if col in panel else
           tda_idx[col].reindex(pd.DatetimeIndex(panel.date)).to_numpy(float))
        y=panel.y_BASE.to_numpy(np.int8);v=panel.market_vol.to_numpy(float)
        for era,lo,hi in ERAS:
            in_test=panel.date.dt.year.between(lo,hi).to_numpy();train=~in_test
            overall=auc(y[in_test],x[in_test]);trainv=v[train&np.isfinite(v)]
            if len(trainv)<3:continue
            q1,q2=np.quantile(trainv,[1/3,2/3])
            for tercile,mask in (("all",in_test),("low",in_test&(v<=q1)),("mid",in_test&(v>q1)&(v<=q2)),("high",in_test&(v>q2))):
                val=overall if tercile=="all" else auc(y[mask],x[mask])
                auc_rows.append({"feature":col,"block":blockname(col),
                    "era":era,"vol_tercile":tercile,"n":int(mask.sum()),"auc":val,
                    "signed_auc":2*val-1 if np.isfinite(val) else np.nan,
                    "reference":"yes" if col in refs else "no"})
        if j and j%100==0:
            print(f"[V5-DIAG] C3 features={j}/{len(all_features)} elapsed={(time.perf_counter()-start)/60:.1f}m",flush=True)
    aucf=pd.DataFrame(auc_rows);aucf.to_csv(V5/"diagnostic_c3_auc.csv",index=False)
    top=(aucf[(aucf.vol_tercile=="all")&(aucf.reference=="no")].assign(abs_signed=lambda z:z.signed_auc.abs())
         .sort_values(["block","era","abs_signed"],ascending=[True,True,False]).groupby(["block","era"]).head(15))
    top.to_csv(V5/"diagnostic_c3_top15.csv",index=False)
    print(f"[V5-DIAG] C1 rows={len(raw_rows)} C2 rows={len(c2f)} C3 rows={len(aucf)} elapsed={(time.perf_counter()-start)/60:.1f}m; saved data/v5/diagnostic_c*.csv")

if __name__=="__main__":main()
