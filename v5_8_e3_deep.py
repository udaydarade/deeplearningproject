"""V5 E3 ML/sequence models using lazy 20-session windows."""
from __future__ import annotations

import argparse,gc,json,time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from pipeline_v5.sequence import fit_lazy_sequence
from v5_1_e1 import append_rows,make_data
from v5_9_e2 import assemble,feature_groups
from v5_eval import (ERAS,W_ERAS,alarms_re,alarms_rq,alarms_rv,blocked_split,
                     evaluate,off_budget,walk_forward_split)

ROOT=Path(__file__).resolve().parent; V3=ROOT/"data"/"v3"; V5=ROOT/"data"/"v5"; SCORES=V5/"scores"
DEEP=("lstm","gru","attn_bilstm","tcn","transformer")


def choose_base_set()->str:
    p=V5/"base_set.json"
    if p.exists():return str(json.loads(p.read_text()).get("feature_set","F0"))
    return "F0"


def choose_best_all(validation:pd.DataFrame)->str:
    preferred=["M0+TDA_small","M0+TDA_big","M0+CLUSTER","M0+PRICE1D","M0+DYN"]
    if validation.empty:return "M0"
    sub=validation[(validation.scheme=="B")&validation.feature_set.isin(preferred)]
    means=sub.groupby("feature_set").validation_excess_event_recall.mean().dropna()
    return str(means.idxmax()) if len(means) else "M0"


def select_top80(frame,features,train,seed):
    sample=np.asarray(train,dtype=np.int64);rng=np.random.default_rng(seed)
    if len(sample)>200_000:sample=np.sort(rng.choice(sample,200_000,replace=False))
    x=frame.iloc[sample][features].to_numpy(np.float32); y=frame.y.to_numpy(np.int8)
    npos=max(int(y[sample].sum()),1); nneg=max(len(sample)-npos,1)
    model=lgb.LGBMClassifier(objective="binary",learning_rate=.03,num_leaves=15,
        min_child_samples=500,feature_fraction=.5,bagging_fraction=.7,bagging_freq=1,
        reg_lambda=5,scale_pos_weight=min(nneg/npos,10),n_estimators=100,max_bin=63,
        n_jobs=1,random_state=seed,verbosity=-1)
    model.fit(x,y[sample])
    gain=model.booster_.feature_importance(importance_type="gain")
    order=np.argsort(-gain,kind="stable")[:min(80,len(features))]
    return [features[i] for i in order]


def preprocess(frame,features,train):
    x=frame[features].to_numpy(np.float32,copy=True)
    med=np.nanmedian(x[train],axis=0); med[~np.isfinite(med)]=0.
    x=np.where(np.isfinite(x),x,med)
    scaler=StandardScaler().fit(x[train]); z=scaler.transform(x).astype(np.float32)
    return z


def lazy_index(frame,z):
    n=len(frame); assets=np.empty(n,dtype=object); positions=np.zeros(n,np.int32); arrays={}
    for asset,ix in frame.groupby("asset",sort=False).groups.items():
        ix=np.asarray(list(ix),np.int64); ix=ix[np.argsort(frame.iloc[ix].date.to_numpy(),kind="stable")]
        arrays[asset]=z[ix].copy(); assets[ix]=asset; positions[ix]=np.arange(len(ix),dtype=np.int32)
    good=np.flatnonzero(positions>=19)
    return arrays,assets,positions,good


def fit_ml(frame,z,train,val,test,model_name,seed):
    y=frame.y.to_numpy(np.int8)
    if model_name=="logreg":
        model=LogisticRegression(class_weight="balanced",max_iter=400,solver="lbfgs",random_state=seed)
        model.fit(z[train],y[train]); return model.predict_proba(z[val])[:,1],model.predict_proba(z[test])[:,1],{}
    npos=max(int(y[train].sum()),1);nneg=max(len(train)-npos,1)
    model=lgb.LGBMClassifier(objective="binary",learning_rate=.03,num_leaves=15,
        min_child_samples=500,feature_fraction=.5,bagging_fraction=.7,bagging_freq=1,
        reg_lambda=5,scale_pos_weight=min(nneg/npos,10),n_estimators=600,max_bin=63,
        n_jobs=1,random_state=seed,verbosity=-1)
    model.fit(z[train],y[train],eval_set=[(z[val],y[val])],eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(0)])
    return model.predict_proba(z[val])[:,1],model.predict_proba(z[test])[:,1],{"best_iteration":model.best_iteration_}


def eval_rows(frame,val_idx,test_idx,vp,tp,scheme,set_name,model_name,era,train_n,seed):
    val=frame.iloc[val_idx][["date","asset","y"]].copy().reset_index(drop=True);val["score"]=vp
    test=frame.iloc[test_idx][["date","asset","y"]].copy().reset_index(drop=True);test["score"]=tp
    val_ap=float(average_precision_score(val.y,vp)) if val.y.sum() else 0.
    val_alarm,skipped=alarms_re(val,.05); val_metrics=evaluate(val,val_alarm,sims=200,seed=seed+1)
    vrow={"scheme":scheme,"era_or_fold":era,"feature_set":set_name,"model":model_name,
          "validation_ap":val_ap,"validation_excess_event_recall":val_metrics["excess_event_recall"],
          "validation_alarm_day_rate":val_metrics["alarm_day_rate"],"validation_n":len(val),
          "validation_n_pos":int(val.y.sum()),"validation_crashes_total":val_metrics["crashes_total"],
          "skipped_assets_re":skipped}
    budgets=[("2%",.02),("5%",.05),("10%",.10),("a_rel",2.5*float(val.y.mean()))]
    rows=[]
    for rule in (("RE",) if scheme=="B" else ("RQ","RV")):
        for budget,a in budgets:
            if rule=="RE":alarm,skipped=alarms_re(test,a)
            elif rule=="RQ":alarm=alarms_rq(test,val,a);skipped=0
            else:alarm=alarms_rv(test,val,a);skipped=0
            m=evaluate(test,alarm,sims=200,seed=seed)
            rows.append({"exp":"E3","scheme":scheme,"label_N":15,"label_p":.02,
                "feature_set":set_name,"model":model_name,"rule":rule,"budget":budget,
                "budget_value":a,"era_or_fold":era,"seed":seed,"subset":"all",
                "off_budget":off_budget(m["alarm_day_rate"],a),"skipped_assets_re":skipped,
                "train_n":train_n,"validation_n":len(val),**m})
    return rows,vrow,val,test


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--timebox-min",type=float,default=150)
    ap.add_argument("--sets",default="BASE_SET,BEST_ALL");ap.add_argument("--models",default=",".join((*DEEP,"logreg","gbm")))
    ap.add_argument("--schemes",default="B,W");ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()
    torch.set_num_threads(1);SCORES.mkdir(parents=True,exist_ok=True);V5.mkdir(parents=True,exist_ok=True)
    panel=pd.read_parquet(V3/"panel.parquet");shared=pd.read_parquet(V3/"market_features.parquet")
    manifest=json.loads((V3/"feature_sets.json").read_text(encoding="utf-8"))
    base_name=choose_base_set()
    base,base_features=make_data(base_name,panel,shared,manifest)
    groups=feature_groups(panel,manifest)
    e2v=pd.read_csv(V5/"e2_validation.csv") if (V5/"e2_validation.csv").exists() else pd.DataFrame()
    best_all=choose_best_all(e2v)
    requested=args.sets.split(",")
    sets=[(x,base_name if x=="BASE_SET" else best_all if x=="BEST_ALL" else x) for x in requested]
    models=args.models.split(",");schemes=args.schemes.split(",")
    jobs=[]
    for scheme in schemes:
        eras=[x[0] for x in ERAS if x[0] in {"E3","E4","E5","E6","E7","E8","E9","E10"}] if scheme=="B" else W_ERAS
        for alias,fs in sets:
            for model in models:
                for era in eras:jobs.append((scheme,alias,fs,model,era))
    if args.smoke:jobs=jobs[:1]
    estimate= len(jobs)*1.0
    print(f"[V5-E3] tasks={len(jobs)} smoke={args.smoke}; rough estimate={estimate:.0f}+ min, hard box={args.timebox_min:g}m; workers=1",flush=True)
    started=time.perf_counter();deadline=started+args.timebox_min*60
    result_path=V5/"e3_results.csv";val_path=V5/"e3_validation.csv"
    completed=set();off={};stopped=set()
    if result_path.exists():
        old=pd.read_csv(result_path)
        completed=set(tuple(x) for x in old[["scheme","feature_set","model","era_or_fold"]].drop_duplicates().itertuples(index=False,name=None))
        for scheme,rule,planned in (("B","RE",8),("W","RQ",7)):
            sub=old[(old.scheme==scheme)&(old.rule==rule)&(old.budget=="5%")]
            for (fs,model),g in sub.groupby(["feature_set","model"]):
                key=(scheme,fs,model);off[key]=int(g.off_budget.astype(bool).sum())
                if off[key]>.2*planned:stopped.add(key)
    current=None;frame=None;features=None
    for ji,(scheme,alias,fs,model_name,era) in enumerate(jobs,1):
        outname=f"{alias}_TOP80";model_label=model_name if model_name in DEEP else f"e3_{model_name}"
        key=(scheme,outname,model_label)
        if key in stopped or (scheme,outname,model_label,era) in completed:continue
        if time.perf_counter()>=deadline:
            print(f"[V5-E3] timebox reached after {ji-1} tasks",flush=True);break
        if fs!=current:
            if frame is not None:del frame;gc.collect()
            if fs in (base_name,"BASE_SET"):
                frame,features=base,base_features
            else:
                frame,features=assemble(fs,base,base_features,groups)
            current=fs
        deep=model_name in DEEP;purge=15+(40 if scheme=="B" else 25)
        tr,va,te=(blocked_split(frame,era,purge,deep=deep,validation_purge_sessions=40)
                  if scheme=="B" else walk_forward_split(frame,era,purge))
        if min(map(len,(tr,va,te)))==0:print(f"[V5-E3] skip empty {scheme}/{fs}/{model_name}/{era}",flush=True);continue
        selected=select_top80(frame,features,tr,ji);y=frame.y.to_numpy(np.int8)
        z=(frame[selected].to_numpy(np.float32,copy=True) if model_name=="gbm"
           else preprocess(frame,selected,tr))
        sequence_assets=sequence_pos=None;arrays=None;valid=None
        if model_name in DEEP:
            arrays,sequence_assets,sequence_pos,valid=lazy_index(frame,z)
            tr=tr[np.isin(tr,valid)];va=va[np.isin(va,valid)];te=te[np.isin(te,valid)]
            if not len(tr) or not len(va) or not len(te):print(f"[V5-E3] skip <20-session sequence history {era}",flush=True);continue
            candidates=[]
            for hidden in (24,64):
                _,apv,vp,tp,detail=fit_lazy_sequence(model_name,arrays,sequence_assets,sequence_pos,
                    z,y,tr,va,te,hidden,9000+ji+hidden)
                candidates.append((apv,hidden,vp,tp,detail))
                print(f"[V5-E3] {scheme}/{fs}/{model_name}/{era}/h{hidden} val_AP={apv:.4f}",flush=True)
            apv,hidden,vp,tp,detail=max(candidates,key=lambda q:q[0])
            model_label=model_name
        else:
            vp,tp,detail=fit_ml(frame,z,tr,va,te,model_name,9000+ji)
            hidden=None;model_label=f"e3_{model_name}"
        output_set=outname
        rows,vrow,val_frame,test_frame=eval_rows(frame,va,te,vp,tp,scheme,output_set,model_label,era,len(tr),9000+ji)
        vrow.update(selected_columns=len(selected),hidden_size=hidden,**detail)
        append_rows(result_path,rows,["scheme","feature_set","model","rule","budget","era_or_fold"])
        append_rows(val_path,[vrow],["scheme","feature_set","model","era_or_fold"])
        score=pd.concat([val_frame.assign(split="validation"),test_frame.assign(split="test")],ignore_index=True)
        stem=f"{scheme}_{output_set}_{model_label}_{era}.parquet"
        score[["date","asset","split","y","score"]].to_parquet(SCORES/stem,index=False)
        primary=[r for r in rows if r["budget"]=="5%" and r["rule"]==("RE" if scheme=="B" else "RQ")]
        off[key]=off.get(key,0)+sum(bool(r["off_budget"]) for r in primary)
        planned=8 if scheme=="B" else 7
        if off[key]>.2*planned:
            stopped.add(key);print(f"[V5-E3] STOP {outname}/{model_label}/{scheme}: off-budget >20%",flush=True)
        print(f"[V5-E3] {ji}/{len(jobs)} {scheme}/{fs}/{model_label}/{era} selected={len(selected)} n={len(te)} elapsed={(time.perf_counter()-started)/60:.1f}m",flush=True)
        del z,arrays
    print(f"[V5-E3] end elapsed={(time.perf_counter()-started)/60:.1f}m",flush=True)


if __name__=="__main__":main()
