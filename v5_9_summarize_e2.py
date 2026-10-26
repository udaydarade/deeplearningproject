"""Validation-selected E2 block verdicts and paired crash-event bootstrap CIs."""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from scipy.stats import hypergeom

from v5_eval import alarms_re,alarms_rq

ROOT=Path(__file__).resolve().parent;V5=ROOT/"data"/"v5";SCORES=V5/"scores"


def _safe(name):return name.replace("+","_")

def _events(frame,alarm):
    f=frame.reset_index(drop=True).copy();f.date=pd.to_datetime(f.date);f["alarm"]=np.asarray(alarm,bool)
    out={}
    for asset,g in f.groupby("asset",sort=False):
        g=g.sort_values("date");y=g.y.to_numpy(bool);a=g.alarm.to_numpy(bool)
        pos=np.flatnonzero(y);runs=[]
        for ix in pos:
            if not runs or ix-runs[-1][-1]>6:runs.append([int(ix)])
            else:runs[-1].append(int(ix))
        for run in runs:
            s,e=run[0],run[-1];key=(asset,str(g.date.iloc[s].date()))
            out[key]={"caught":bool(a[s:e+1].any()),"start":s,"end":e,"n":len(y),"alarm_days":int(a.sum())}
    return out

def _rule_scores(path,scheme):
    d=pd.read_parquet(path);v=d[d.split.eq("validation")].copy();t=d[d.split.eq("test")].copy()
    v.sort_values(["date","asset"],inplace=True);t.sort_values(["date","asset"],inplace=True)
    alarm=alarms_re(t,.05)[0] if scheme=="B" else alarms_rq(t,v,.05)
    return t,v,_events(t,alarm),float(average_precision_score(t.y,t.score)) if t.y.sum() else np.nan

def _bootstrap(deltas_by_era,seed=7841,sims=1000):
    rng=np.random.default_rng(seed);eras=list(deltas_by_era)
    observed=np.mean([np.mean(deltas_by_era[e]) for e in eras if len(deltas_by_era[e])]) if eras else np.nan
    draws=[]
    for _ in range(sims):
        per=[]
        for e in eras:
            x=np.asarray(deltas_by_era[e],float)
            if len(x):per.append(float(np.mean(rng.choice(x,size=len(x),replace=True))))
        if per:draws.append(float(np.mean(per)))
    return observed,np.quantile(draws,.025),np.quantile(draws,.975)

def main():
    res=pd.read_csv(V5/"e2_results.csv");val=pd.read_csv(V5/"e2_validation.csv")
    # BEST_ALL is chosen only from Scheme-B validation excess event recall.
    pref=["M0+TDA_small","M0+TDA_big","M0+CLUSTER","M0+PRICE1D","M0+DYN"]
    v=val[(val.scheme=="B")&val.feature_set.isin(pref)]
    means=v.groupby("feature_set").validation_excess_event_recall.mean().dropna()
    best=str(means.idxmax()) if len(means) else "M0"
    (V5/"best_all.json").write_text(pd.Series({"feature_set":best,"selection":"highest Scheme-B mean validation excess_event_recall among primary add-on blocks"}).to_json(indent=2),encoding="utf-8")
    primary_rows=res[(res.budget=="5%")&(((res.scheme=="B")&(res.rule=="RE"))|((res.scheme=="W")&(res.rule=="RQ")))]
    eligible=primary_rows[~primary_rows.off_budget.astype(bool)]
    sets=[(x,"M0") for x in eligible.feature_set.unique() if x.startswith("M0+")]
    if "TDA_only_class" in set(eligible.feature_set):sets.append(("TDA_only_class","B3"))
    long=[];summary=[]
    for scheme in ("B","W"):
      models=eligible.loc[eligible.scheme.eq(scheme),"model"].unique()
      for model in models:
       for fs,baseline in sets:
        dmap={};aps=[];wins=0;used=0
        base_eras=set(eligible.loc[(eligible.scheme==scheme)&(eligible.model==model)&(eligible.feature_set==baseline),"era_or_fold"])
        cand_eras=set(eligible.loc[(eligible.scheme==scheme)&(eligible.model==model)&(eligible.feature_set==fs),"era_or_fold"])
        attempted_base=set(primary_rows.loc[(primary_rows.scheme==scheme)&(primary_rows.model==model)&(primary_rows.feature_set==baseline),"era_or_fold"])
        attempted_cand=set(primary_rows.loc[(primary_rows.scheme==scheme)&(primary_rows.model==model)&(primary_rows.feature_set==fs),"era_or_fold"])
        attempted_common=len(attempted_base&attempted_cand)
        eras=sorted(base_eras&cand_eras)
        for era in eras:
            try:
                p0=SCORES/f"{scheme}_{_safe(baseline)}_{model}_{era}.parquet"
                p1=SCORES/f"{scheme}_{_safe(fs)}_{model}_{era}.parquet"
                t0,v0,e0,ap0=_rule_scores(p0,scheme);t1,v1,e1,ap1=_rule_scores(p1,scheme)
            except FileNotFoundError:continue
            common=sorted(set(e0)&set(e1))
            delta=[]
            for k in common:
                a,b=e0[k],e1[k];length=a["end"]-a["start"]+1;n=a["n"]
                r0=float(hypergeom.sf(0,n,length,a["alarm_days"])) if a["alarm_days"] else 0.
                r1=float(hypergeom.sf(0,n,length,b["alarm_days"])) if b["alarm_days"] else 0.
                delta.append((int(b["caught"])-r1)-(int(a["caught"])-r0))
            delta=np.asarray(delta,float)
            if not len(delta):continue
            dmap[era]=delta;diff=float(delta.mean());wins+=diff>0;used+=1
            aps.append(ap1-ap0)
            long.append({"scheme":scheme,"feature_set":fs,"baseline":baseline,"model":model,"era_or_fold":era,
                         "crashes":len(delta),"baseline_event_recall":float(np.mean([e0[k]["caught"] for k in common])),
                         "candidate_event_recall":float(np.mean([e1[k]["caught"] for k in common])),
                         "paired_difference":diff,"AP_difference_diagnostic":ap1-ap0})
        if not dmap:continue
        mean,lo,hi=_bootstrap(dmap,seed=21000+len(summary))
        category=("cloud_size" if fs.startswith("M0+curve_") else "cloud_type" if fs.startswith("M0+type_") else
                  "homology" if fs.startswith("M0+homology_") else "window" if fs.startswith("M0+window_") else "block")
        complete=(attempted_common==(11 if scheme=="B" else 7))
        supported=bool(scheme=="B" and complete and wins/used>=.6 and lo>0)
        verdict="adds value" if supported else "incomplete" if not complete else "no supported gain"
        summary.append({"scheme":scheme,"category":category,"feature_set":fs,"baseline":baseline,"model":model,"eras":used,"attempted_eras":attempted_common,"complete_test_eras":complete,
            "eras_won":int(wins),"win_share":wins/used,"mean_paired_excess_event_recall_diff":mean,
            "bootstrap95_low":lo,"bootstrap95_high":hi,"mean_AP_difference_diagnostic":float(np.mean(aps)),
            "adds_value_preregistered":supported,"verdict":verdict})
    pd.DataFrame(long).to_csv(V5/"e2_paired_eras.csv",index=False)
    table=pd.DataFrame(summary);table.to_csv(V5/"e2_paired_summary.csv",index=False)
    print(f"[V5-E2 summary] BEST_ALL={best} by Scheme-B validation only; paired tests={len(table)}")
    print(f"[V5-E2 summary] paired event bootstrap=1000/era; tables=e2_paired_summary.csv,e2_paired_eras.csv")
    if len(table):print(table[table.scheme.eq("B")][["feature_set","model","eras_won","eras","bootstrap95_low","bootstrap95_high","verdict"]].to_string(index=False))

if __name__=="__main__":main()
