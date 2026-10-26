"""V5 minimal label sweep using the frozen V3 past-only label variants."""
from __future__ import annotations
import argparse,gc,json,time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from pipeline_v5.sequence import fit_lazy_sequence
from v5_1_e1 import append_rows,fit_score,make_data
from v5_8_e3_deep import DEEP,lazy_index,preprocess,select_top80
from v5_9_e2 import assemble,feature_groups
from v5_eval import ERAS,W_ERAS,alarms_re,alarms_rq,alarms_rv,blocked_split,evaluate,off_budget,walk_forward_split

ROOT=Path(__file__).resolve().parent;V3=ROOT/"data"/"v3";V5=ROOT/"data"/"v5";SCORES=V5/"scores"
LABELS=[("BASE","y_BASE",15,.02),("N5_p2","y_N5_p2",5,.02),("N30_p2","y_N30_p2",30,.02),
        ("N15_p5","y_N15_p5",15,.05),("N15_p10","y_N15_p10",15,.10)]


def best_deep()->str|None:
    p=V5/"e3_validation.csv"
    if not p.exists():return None
    d=pd.read_csv(p);d=d[(d.scheme=="B")&d.model.isin(DEEP)]
    if "BEST_ALL_TOP80" in set(d.feature_set):d=d[d.feature_set.eq("BEST_ALL_TOP80")]
    m=d.groupby("model").validation_excess_event_recall.mean().dropna()
    return str(m.idxmax()) if len(m) else None


def eval_pair(frame,features,scheme,era,n,p,model,seed):
    purge=n+(40 if scheme=="B" else 25)
    tr,va,te=(blocked_split(frame,era,purge,deep=(model in DEEP),validation_purge_sessions=n+25)
              if scheme=="B" else walk_forward_split(frame,era,purge))
    if min(len(tr),len(va),len(te))==0:return None
    selected=select_top80(frame,features,tr,seed) if model in DEEP else features
    if model in DEEP:
        z=preprocess(frame,selected,tr);arrays,assets,pos,valid=lazy_index(frame,z)
        tr=tr[np.isin(tr,valid)];va=va[np.isin(va,valid)];te=te[np.isin(te,valid)]
        if min(len(tr),len(va),len(te))==0:return None
        results=[]
        for h in (24,64):
            net,apv,vp,tp,detail=fit_lazy_sequence(model,arrays,assets,pos,z,frame.y.to_numpy(np.int8),tr,va,te,h,seed+h)
            results.append((apv,h,vp,tp,detail))
        _,hidden,vp,tp,detail=max(results,key=lambda x:x[0])
    else:
        if model=="gbm" and len(features)>400:
            sample=np.asarray(tr,dtype=np.int64);rng=np.random.default_rng(seed)
            if len(sample)>200_000:sample=np.sort(rng.choice(sample,200_000,replace=False))
            x=frame.iloc[sample][features].to_numpy(np.float32);y=frame.y.to_numpy(np.int8)
            pos=max(int(y[sample].sum()),1);neg=max(len(sample)-pos,1)
            g=lgb.LGBMClassifier(objective="binary",learning_rate=.03,num_leaves=15,min_child_samples=500,
                feature_fraction=.5,bagging_fraction=.7,bagging_freq=1,reg_lambda=5,
                scale_pos_weight=min(neg/pos,10),n_estimators=100,max_bin=63,n_jobs=1,random_state=seed,verbosity=-1)
            g.fit(x,y[sample]);gain=g.booster_.feature_importance(importance_type="gain")
            selected=[features[i] for i in np.argsort(-gain,kind="stable")[:400]]
        else:selected=features
        vp,tp=fit_score(frame,selected,tr,va,te,model,seed);hidden=None;detail={}
    vf=frame.iloc[va][["date","asset","y"]].copy().reset_index(drop=True);vf["score"]=vp
    tf=frame.iloc[te][["date","asset","y"]].copy().reset_index(drop=True);tf["score"]=tp
    val_alarm,_=alarms_re(vf,.05);vm=evaluate(vf,val_alarm,sims=200,seed=seed+1)
    vrow={"scheme":scheme,"feature_set":f"E4_{n}_{p:g}","model":model,"era_or_fold":era,
          "validation_excess_event_recall":vm["excess_event_recall"],"validation_n":len(vf),
          "validation_n_pos":int(vf.y.sum()),"validation_crashes_total":vm["crashes_total"],"hidden_size":hidden,**detail}
    rows=[]
    for rule in (("RE",) if scheme=="B" else ("RQ","RV")):
        for budget,a in (("2%",.02),("5%",.05),("10%",.10),("a_rel",2.5*float(vf.y.mean()))):
            if rule=="RE":alarm,skip=alarms_re(tf,a)
            elif rule=="RQ":alarm=alarms_rq(tf,vf,a);skip=0
            else:alarm=alarms_rv(tf,vf,a);skip=0
            m=evaluate(tf,alarm,sims=200,seed=seed)
            rows.append({"exp":"E4","scheme":scheme,"label_N":n,"label_p":p,
              "feature_set":f"E4_{n}_{p:g}","model":model,"rule":rule,"budget":budget,"budget_value":a,
              "era_or_fold":era,"seed":seed,"subset":"all","off_budget":off_budget(m["alarm_day_rate"],a),
              "train_n":len(tr),"validation_n":len(va),"hidden_size":hidden,**m})
    scores=pd.concat([vf.assign(split="validation"),tf.assign(split="test")],ignore_index=True)[["date","asset","split","y","score"]]
    return rows,vrow,scores


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--timebox-min",type=float,default=90)
    ap.add_argument("--smoke",action="store_true");args=ap.parse_args()
    panel=pd.read_parquet(V3/"panel.parquet");shared=pd.read_parquet(V3/"market_features.parquet")
    labels=pd.read_parquet(V3/"labels.parquet")
    manifest=json.loads((V3/"feature_sets.json").read_text(encoding="utf-8"));base_set=json.loads((V5/"base_set.json").read_text())["feature_set"]
    base,base_features=make_data(base_set,panel,shared,manifest);groups=feature_groups(panel,manifest)
    e2v=pd.read_csv(V5/"e2_validation.csv") if (V5/"e2_validation.csv").exists() else pd.DataFrame()
    best_all="M0";candidate=e2v[(e2v.scheme=="B")&e2v.feature_set.str.startswith("M0+")] if len(e2v) else pd.DataFrame()
    if len(candidate):best_all=str(candidate.groupby("feature_set").validation_excess_event_recall.mean().idxmax())
    deep=best_deep()
    if deep is None:print("[V5-E4] no completed E3 validation model; running GBM label sweep only",flush=True)
    jobs=[]
    schemes=("B","W")
    for tag,label,n,p in LABELS:
        for model in (["gbm",deep] if deep else ["gbm"]):
            for scheme in schemes:
                eras=[e[0] for e in ERAS if (model not in DEEP or e[0] in {"E3","E4","E5","E6","E7","E8","E9","E10"})] if scheme=="B" else W_ERAS
                jobs += [(tag,label,n,p,model,scheme,era) for era in eras]
    planned=len(jobs)
    if args.smoke:jobs=jobs[:1]
    print(f"[V5-E4] configs=5 models={['gbm',deep] if deep else ['gbm']} BEST_ALL={best_all}; jobs={len(jobs)}/{planned}; estimate={planned*.5:.0f}m; workers=1",flush=True)
    SCORES.mkdir(parents=True,exist_ok=True);start=time.perf_counter();deadline=start+args.timebox_min*60
    out=V5/"e4_results.csv";valout=V5/"e4_validation.csv";completed=set();off={};stopped=set()
    if out.exists():
        old=pd.read_csv(out)
        completed=set(tuple(x) for x in old[["label_N","label_p","scheme","model","era_or_fold"]].drop_duplicates().itertuples(index=False,name=None))
        old=old[(old.budget=="5%")&(((old.scheme=="B")&(old.rule=="RE"))|((old.scheme=="W")&(old.rule=="RQ")))]
        for (n,p,scheme,model),g in old.groupby(["label_N","label_p","scheme","model"]):
            k=(n,p,scheme,model);off[k]=int(g.off_budget.astype(bool).sum())
            planned_folds=(8 if model in DEEP and scheme=="B" else 11 if scheme=="B" else 7)
            if off[k]>.2*planned_folds:stopped.add(k)
    for i,(tag,label,n,p,model,scheme,era) in enumerate(jobs,1):
        if time.perf_counter()>=deadline:print(f"[V5-E4] hard box reached after {i-1}/{len(jobs)}",flush=True);break
        run_key=(n,p,scheme,model)
        if run_key in stopped or (n,p,scheme,model,era) in completed:continue
        label_panel=panel.drop(columns=["y_BASE"]).merge(
            labels[["date","asset",label]].rename(columns={label:"y_BASE"}),
            on=["date","asset"],how="left",validate="one_to_one")
        fs=base_set if model=="gbm" else best_all
        base_label,base_label_features=make_data(base_set,label_panel,shared,manifest)
        if fs=="M0":f,feat=base_label,base_label_features
        else:f,feat=assemble(fs,base_label,base_label_features,groups)
        result=eval_pair(f,feat,scheme,era,n,p,model,9500+i)
        if result is None:continue
        rows,vrow,scores=result
        for r in rows:r["feature_set"]=f"E4_{tag}_{fs}"
        vrow["feature_set"]=f"E4_{tag}_{fs}"
        append_rows(out,rows,["label_N","label_p","scheme","feature_set","model","rule","budget","era_or_fold"])
        append_rows(valout,[vrow],["label_N","label_p","scheme","feature_set","model","era_or_fold"])
        stem=f"{scheme}_E4_{tag}_{model}_{era}"
        scores.to_parquet(SCORES/f"{stem}.parquet",index=False)
        primary=[r for r in rows if r["budget"]=="5%" and r["rule"]==("RE" if scheme=="B" else "RQ")]
        off[run_key]=off.get(run_key,0)+sum(bool(r["off_budget"]) for r in primary)
        planned_folds=(8 if model in DEEP and scheme=="B" else 11 if scheme=="B" else 7)
        if off[run_key]>.2*planned_folds:
            stopped.add(run_key);print(f"[V5-E4] STOP {tag}/{model}/{scheme}: off-budget >20%",flush=True)
        print(f"[V5-E4] {i}/{len(jobs)} {tag}/{model}/{scheme}/{era} n={len(scores)} positives={int(scores.y.sum())} events={rows[0]['crashes_total']} elapsed={(time.perf_counter()-start)/60:.1f}m",flush=True)
        del f,rows,scores;gc.collect()
    print(f"[V5-E4] end elapsed={(time.perf_counter()-start)/60:.1f}m",flush=True)

if __name__=="__main__":main()
