"""Set TDA features to missing whenever fewer than eight assets are available."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
V3=ROOT/"data"/"v3"; V5=ROOT/"data"/"v5"; BLOCKS=V5/"blocks"
PAT=re.compile(r"^v5_(NS|ALLS)_(A_corr|A_raw|T_raw|T_std)_k.+_w(\d+)_")


def main():
    ret=pd.read_parquet(V3/"returns.parquet")
    reg=pd.read_csv(V3/"registry.csv").set_index("ticker")
    ns=[c for c in ret if reg.loc[c,"asset_class"]!="stock"]
    stocks=[c for c in ret if reg.loc[c,"asset_class"]=="stock"]
    sampled=pd.read_parquet(BLOCKS/"V5_TDA_sampled.parquet").sort_values("date").reset_index(drop=True)
    daily=pd.read_parquet(BLOCKS/"V5_TDA.parquet").sort_values("date").reset_index(drop=True)
    meta=json.loads((V5/"feature_blocks_tda.json").read_text(encoding="utf-8"))
    if meta.get("config",{}).get("l2_ratio_scale")=="variance^1.5":
        # The scale audit exposed that variance^1.5 is dimensionally wrong for
        # L2 persistence (distance^(3/2)). Recover variance from the old L2 and
        # ratio columns and rewrite as L2 / variance^(3/4).
        for frame in (sampled,daily):
            for col in [c for c in frame if c.endswith("_l2_ratio")]:
                raw=col[:-6]
                l2=frame[raw].to_numpy(float); old=frame[col].to_numpy(float)
                frame[col]=np.where((l2>=0)&(old>=0),np.sqrt(l2*old),np.nan)
    allowed={}
    for universe,assets in (("NS",ns),("ALLS",ns+stocks)):
        for w in (20,60,120,250):
            count=ret[assets].notna().rolling(w,min_periods=w).sum().eq(w).sum(axis=1)
            allowed[(universe,w)]=count.ge(8).to_numpy()
    for frame in (sampled,daily):
        ix=pd.DatetimeIndex(pd.to_datetime(frame.date))
        loc=ret.index.get_indexer(ix)
        for col in frame.columns:
            m=PAT.match(col)
            if not m:continue
            universe,cloud,w=m.group(1),m.group(2),int(m.group(3))
            valid=allowed[(universe,w)][loc]
            frame.loc[~valid,col]=np.nan
    sampled.to_parquet(BLOCKS/"V5_TDA_sampled.parquet",index=False)
    daily.to_parquet(BLOCKS/"V5_TDA.parquet",index=False)
    meta["config"]["min_assets"] = 8
    meta["config"]["checkpointed"] = False
    meta["config"]["l2_ratio_scale"] = "variance^0.75"
    (V5/"feature_blocks_tda.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(f"[V5-TDA universe] masked {len(meta.get('V5_TDA',[]))} feature columns below eight complete-window assets")


if __name__=="__main__":main()
