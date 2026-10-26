"""Required historical episode validation for completed V5 model scores."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from v5_eval import alarms_re,alarms_rq

ROOT=Path(__file__).resolve().parent;V5=ROOT/"data"/"v5";SCORES=V5/"scores"
WINDOWS=[("1929-09","1929-09-01","1932-07-31","^GSPC"),
 ("1937-08","1937-08-01","1938-04-30",None),("1973-74","1973-01-01","1974-12-31",None),
 ("1987","1987-09-01","1987-12-31",None),("1990","1990-07-01","1990-12-31",None),
 ("1998","1998-07-01","1998-10-31",None),("2000-02","2000-03-01","2002-10-31",None),
 ("2008","2008-08-01","2009-03-31",None),("2011","2011-07-01","2011-10-31",None),
 ("2015-16","2015-08-01","2016-02-29",None),("2018","2018-10-01","2018-12-31",None),
 ("2020","2020-02-01","2020-04-30",None),("2022","2022-01-01","2022-10-31",None)]

def event_runs(y):
    ix=np.flatnonzero(np.asarray(y,bool));out=[]
    for i in ix:
        if not out or i-out[-1][-1]>6:out.append([int(i)])
        else:out[-1].append(int(i))
    return [(x[0],x[-1]) for x in out]

def alarm_runs(a):
    ix=np.flatnonzero(np.asarray(a,bool));merged=[]
    for i in ix:
        if not merged or i-merged[-1][-1]>6:merged.append([int(i)])
        else:merged[-1].append(int(i))
    out=[]
    for run in merged:
        s,e=run[0],run[-1]
        while s<=e:out.append((s,min(e,s+14)));s+=15
    return out

def evaluate_window(d,alarm,start,end,target):
    d=d.copy();d.date=pd.to_datetime(d.date)
    lo=pd.Timestamp(start);hi=pd.Timestamp(end)
    if target:d=d[d.asset.eq(target)]
    d=d[d.date.between(lo,hi)].copy();caught=events=assets=0;wrong=alarm_count=0
    for asset,g in d.groupby("asset",sort=False):
        g=g.sort_values("date");y=g.y.to_numpy(bool);a=alarm[g.index.to_numpy()]
        cr=event_runs(y);ar=alarm_runs(a);assets+=int(y.any())
        events+=len(cr);caught+=sum(bool(a[s:e+1].any()) for s,e in cr)
        wrong+=sum(not bool((a[s:e+1]&y[s:e+1]).any()) for s,e in ar);alarm_count+=len(ar)
    n=len(d);asset_years=n/252
    return {"crashes":events,"crashes_caught":caught,"share_crashes_caught":caught/events if events else np.nan,
            "assets_with_crashes":assets,"assets_in_window":d.asset.nunique(),"wrong_alarms":wrong,
            "alarms_total":alarm_count,"wrong_per_asset_year":wrong/asset_years if asset_years else np.nan,
            "alarm_day_rate":float(alarm[d.index].mean()) if n else np.nan}

def main():
    rows=[]
    e1=V5/"e1_headline.csv"
    if e1.exists():
        h=pd.read_csv(e1);cfg=json.loads((V5/"base_set.json").read_text());fs=cfg["feature_set"]
        models=h[(h.scheme=="B")&(h.feature_set==fs)].model.unique().tolist()
        for model in models:
          for scheme in ("B","W"):
            eras=[x[0] for x in __import__("v5_eval").ERAS] if scheme=="B" else __import__("v5_eval").W_ERAS
            for era in eras:
                path=SCORES/f"{scheme}_{fs}_{model}_{era}.parquet"
                if not path.exists():continue
                d=pd.read_parquet(path);v=d[d.split.eq("validation")];t=d[d.split.eq("test")].copy()
                alarm=alarms_re(t,.05)[0] if scheme=="B" else alarms_rq(t,v,.05);t=t.reset_index(drop=True)
                for label,start,end,target in WINDOWS:
                    m=evaluate_window(t,alarm,start,end,target)
                    rows.append({"exp":"E1","scheme":scheme,"feature_set":fs,"model":model,"era_or_fold":era,
                                 "episode":label,"start":start,"end":end,**m})
    e3=V5/"e3_validation.csv"
    if e3.exists():
        val=pd.read_csv(e3);sub=val[(val.scheme=="B")&val.feature_set.eq("BEST_ALL_TOP80")]
        model=sub.groupby("model").validation_excess_event_recall.mean().idxmax() if len(sub) else None
        if model:
            for scheme in ("B","W"):
              eras=[x[0] for x in __import__("v5_eval").ERAS if x[0] in {"E3","E4","E5","E6","E7","E8","E9","E10"}] if scheme=="B" else __import__("v5_eval").W_ERAS
              for era in eras:
                path=SCORES/f"{scheme}_BEST_ALL_TOP80_{model}_{era}.parquet"
                if not path.exists():continue
                d=pd.read_parquet(path);v=d[d.split.eq("validation")];t=d[d.split.eq("test")].copy()
                alarm=alarms_re(t,.05)[0] if scheme=="B" else alarms_rq(t,v,.05);t=t.reset_index(drop=True)
                for label,start,end,target in WINDOWS:
                    m=evaluate_window(t,alarm,start,end,target)
                    rows.append({"exp":"E3","scheme":scheme,"feature_set":"BEST_ALL_TOP80","model":model,
                                 "era_or_fold":era,"episode":label,"start":start,"end":end,**m})
    out=pd.DataFrame(rows);out.to_csv(V5/"episode_results.csv",index=False)
    print(f"[V5 episodes] rows={len(out)}; windows={len(WINDOWS)}; saved data/v5/episode_results.csv")

if __name__=="__main__":main()
