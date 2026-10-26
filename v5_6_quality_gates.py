"""V5 point-in-time and scale checks plus per-column topology diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v5_2_build_tda import features_at_date

ROOT=Path(__file__).resolve().parent; V3=ROOT/"data"/"v3"; V5=ROOT/"data"/"v5"; BLOCKS=V5/"blocks"


def _max_abs_diff(a:pd.Series,b:pd.Series)->float:
    x=a.to_numpy(float); y=b.to_numpy(float); ok=np.isfinite(x)&np.isfinite(y)
    return float(np.max(np.abs(x[ok]-y[ok]))) if ok.any() else 0.


def main():
    sample=pd.read_parquet(BLOCKS/"V5_TDA_sampled.parquet").sort_values("date").reset_index(drop=True)
    dense=pd.read_parquet(BLOCKS/"V5_TDA.parquet").sort_values("date").reset_index(drop=True)
    returns=pd.read_parquet(V3/"returns.parquet").astype(float)
    reg=pd.read_csv(V3/"registry.csv").set_index("ticker")
    stocks=[c for c in returns if reg.loc[c,"asset_class"]=="stock"]
    nonstock=[c for c in returns if reg.loc[c,"asset_class"]!="stock"]
    config=json.loads((V5/"feature_blocks_tda.json").read_text()).get("config",{})
    bags=int(config.get("bags",2)); cols=[c for c in sample if c!="date"]
    rng=np.random.default_rng(5521)
    eligible=sample.loc[pd.to_datetime(sample.date).dt.year>2000,"date"]
    leak_dates=rng.choice(eligible.to_numpy(),size=min(10,len(eligible)),replace=False)
    leak_rows=[]
    for date in sorted(pd.Timestamp(x) for x in leak_dates):
        full=sample.loc[pd.to_datetime(sample.date).eq(date)].iloc[0]
        past=returns.loc[:date]; hist=past.notna().cumsum(axis=0).to_numpy(dtype=np.uint16)
        rebuilt=features_at_date(date,past.index.get_loc(date),past,nonstock,stocks,hist,bags)
        common=sorted(set(cols)&set(rebuilt)-{"date"})
        diffs=[abs(float(full[c])-float(rebuilt[c])) for c in common
               if pd.notna(full[c]) and pd.notna(rebuilt[c])]
        mx=max(diffs,default=0.)
        leak_rows.append({"date":date,"columns_compared":len(diffs),"max_abs_diff":mx,"pass":mx<1e-5})
    pd.DataFrame(leak_rows).to_csv(V5/"quality_leak_test.csv",index=False)

    eligible_dates=sample.loc[pd.to_datetime(sample.date).dt.year>2000,"date"]
    scale_dates=sorted(pd.Timestamp(x) for x in rng.choice(eligible_dates.to_numpy(),size=min(20,len(eligible_dates)),replace=False))
    scale_rows=[]; c=3.0
    for date in scale_dates:
        ix=returns.index.get_loc(date); block=returns.iloc[:ix+1]
        hist=block.notna().cumsum(axis=0).to_numpy(dtype=np.uint16)
        original=features_at_date(date,ix,block,nonstock,stocks,hist,bags)
        scaled=features_at_date(date,ix,block*c,nonstock,stocks,hist,bags)
        for name,x in original.items():
            if name=="date" or name not in scaled or not np.isfinite(x) or not np.isfinite(scaled[name]):continue
            target=None
            if "_A_corr_" in name or "_T_std_" in name:target=float(x)
            elif "_ratio" in name:target=float(x)
            elif "_l1" in name and "_ratio" not in name:target=float(x)*c**2
            elif "_l2" in name and "_ratio" not in name:target=float(x)*c**1.5
            elif any(t in name for t in ("_total","_max","cover_radius")) and "_ratio" not in name:
                target=float(x) if "_entropy" in name or "_count" in name else float(x)*c
            elif "_entropy" in name or "_count" in name or "_betti_" in name:target=float(x)
            elif "_landscape_" in name:target=float(x)*c
            if target is None:continue
            error=abs(float(scaled[name])-target)/max(abs(target),1e-10)
            scale_rows.append({"date":date,"column":name,"relative_error":error,"pass":error<1e-6})
    pd.DataFrame(scale_rows).to_csv(V5/"quality_scale_test.csv",index=False)

    quality=[]
    sampled_dates=set(pd.to_datetime(sample.date))
    is_forward_filled=~pd.to_datetime(dense.date).isin(sampled_dates)
    for col in cols:
        x=sample[col]; full=dense[col] if col in dense else x
        quality.append({"column":col,"share_nan":float(x.isna().mean()),
                        "share_zero":float(x.eq(0).mean()),
                        "share_forward_filled":float((full.notna()&is_forward_filled).mean()),
                        "finite_count":int(x.notna().sum())})
    pd.DataFrame(quality).to_csv(V5/"tda_quality_by_column.csv",index=False)
    group_rows=[]
    import re
    rgx=re.compile(r"^v5_(NS|ALLS)_(A_corr|A_raw|T_raw|T_std)_k([^_]+)_w(\d+)_h([12])_(count)$")
    for col in cols:
        m=rgx.match(col)
        if not m:continue
        universe,cloud,k,w,h,_=m.groups(); val=sample[col]
        group_rows.append({"universe":universe,"cloud":cloud,"k":k,"window":int(w),
                           "homology":int(h),"n_windows":int(val.notna().sum()),
                           "share_ge1_bar":float(val.gt(0).mean()),
                           "share_nan":float(val.isna().mean()),"share_zero":float(val.eq(0).mean())})
    pd.DataFrame(group_rows).to_csv(V5/"tda_quality_by_group.csv",index=False)
    gates=[{"gate":"leak","n":len(leak_rows),"max_error":max(x["max_abs_diff"] for x in leak_rows),
            "pass":all(x["pass"] for x in leak_rows)},
           {"gate":"scale","n":len(scale_rows),"max_error":max((x["relative_error"] for x in scale_rows),default=np.nan),
            "pass":all(x["pass"] for x in scale_rows)}]
    pd.DataFrame(gates).to_csv(V5/"quality_gate_results.csv",index=False)
    print(f"[V5-QG] leak dates={len(leak_rows)}, max diff={max(x['max_abs_diff'] for x in leak_rows):.3g}, pass={all(x['pass'] for x in leak_rows)}")
    print(f"[V5-QG] scale checks={len(scale_rows)}, max relative error={max((x['relative_error'] for x in scale_rows),default=np.nan):.3g}, pass={all(x['pass'] for x in scale_rows)}")
    print(f"[V5-QG] quality columns={len(quality)}, H1/H2 group rows={len(group_rows)}; saved under data/v5/")


if __name__=="__main__":main()
