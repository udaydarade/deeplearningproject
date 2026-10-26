"""V5 E1 full-history BASE_SET ladder; resumable, time-boxed, <=1 worker/model."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from v5_eval import (ERAS, W_ERAS, alarms_re, alarms_rq, alarms_rv,
                     blocked_split, evaluate, off_budget, walk_forward_split)

ROOT=Path(__file__).resolve().parent
V3=ROOT/"data"/"v3"; OUT=ROOT/"data"/"v5"; SCORES=OUT/"scores"
BASESETS={"B1":["realized_vol_w20"],
          "B2":["price_drawdown_60d","price_momentum_20d"],
          "B3":["realized_vol_w20","price_drawdown_60d","price_momentum_20d"]}


def make_data(name: str, panel: pd.DataFrame, shared: pd.DataFrame,
              manifest: dict[str,list[str]], label_column: str="y_BASE"
              ) -> tuple[pd.DataFrame,list[str]]:
    class_cols=[c for c in panel if c.startswith("class_")]
    if name in BASESETS:
        features=BASESETS[name]+(class_cols if name=="B3" else [])
    else: features=list(manifest[name])
    local=[c for c in features if c in panel.columns]
    external=[c for c in features if c in shared.columns]
    missing=set(features)-set(local)-set(external)
    if missing: raise ValueError(f"{name}: missing columns {sorted(missing)[:8]}")
    keep=["date","asset",label_column]+local
    frame=panel[keep].copy()
    if external:
        frame=frame.merge(shared[external].reset_index(),on="date",how="left",validate="many_to_one")
    frame=frame.loc[frame[label_column].notna()].copy()
    frame.rename(columns={label_column:"y"},inplace=True)
    frame.date=pd.to_datetime(frame.date)
    frame.sort_values(["date","asset"],kind="stable",inplace=True,ignore_index=True)
    return frame,features


def fit_score(frame: pd.DataFrame, features: list[str], train: np.ndarray,
              val: np.ndarray, test: np.ndarray, model_name: str, seed: int):
    x=frame[features].to_numpy(np.float32,copy=True)
    y=frame.y.to_numpy(np.int8)
    if model_name=="logreg":
        med=np.nanmedian(x[train],axis=0); med[~np.isfinite(med)]=0
        x=np.where(np.isfinite(x),x,med)
        scaler=StandardScaler().fit(x[train]); z=scaler.transform(x).astype(np.float32)
        model=LogisticRegression(class_weight="balanced",max_iter=400,solver="lbfgs",random_state=seed)
        model.fit(z[train],y[train])
        return model.predict_proba(z[val])[:,1].astype(np.float32),model.predict_proba(z[test])[:,1].astype(np.float32)
    npos=max(int(y[train].sum()),1); nneg=max(len(train)-npos,1)
    model=lgb.LGBMClassifier(objective="binary",learning_rate=.03,num_leaves=15,
        min_child_samples=500,feature_fraction=.5,bagging_fraction=.7,bagging_freq=1,
        reg_lambda=5,scale_pos_weight=min(nneg/npos,10),n_estimators=600,max_bin=63,
        n_jobs=1,random_state=seed,verbosity=-1)
    model.fit(x[train],y[train],eval_set=[(x[val],y[val])],eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(0)])
    return model.predict_proba(x[val])[:,1].astype(np.float32),model.predict_proba(x[test])[:,1].astype(np.float32)


def validation_metrics(frame: pd.DataFrame, score: np.ndarray, seed: int) -> dict:
    val=frame.copy(); val["score"]=score
    alarm,skipped=alarms_re(val,.05)
    m=evaluate(val,alarm,sims=200,seed=seed)
    return {**m,"skipped_assets_re":skipped}


def score_era(frame: pd.DataFrame, features: list[str], set_name: str,
              model_name: str, scheme: str, era: str, deep: bool,
              label_n: int, seed: int):
    purge=label_n+40 if scheme=="B" else label_n+25
    train,val,test=(blocked_split(frame,era,purge,deep=deep,
                                  validation_purge_sessions=label_n+25) if scheme=="B"
                    else walk_forward_split(frame,era,purge))
    if min(map(len,(train,val,test)))==0:
        return [],[],{"train":len(train),"val":len(val),"test":len(test),"error":"empty split"}
    stem=f"{scheme}_{set_name}_{model_name}_{era}"
    score_path=SCORES/f"{stem}.parquet"
    if score_path.exists():
        saved=pd.read_parquet(score_path)
        va=saved.loc[saved.split.eq("validation")].sort_values(["date","asset"])
        te=saved.loc[saved.split.eq("test")].sort_values(["date","asset"])
        vp=va.score.to_numpy(np.float32); tp=te.score.to_numpy(np.float32)
        val_frame=va[["date","asset","y"]].reset_index(drop=True)
        test_frame=te[["date","asset","y"]].reset_index(drop=True)
    else:
        vp,tp=fit_score(frame,features,train,val,test,model_name,seed)
        val_frame=frame.iloc[val][["date","asset","y"]].copy().reset_index(drop=True)
        test_frame=frame.iloc[test][["date","asset","y"]].copy().reset_index(drop=True)
        val_frame["score"]=vp; test_frame["score"]=tp
        val_save=val_frame.copy(); val_save["split"]="validation"
        test_save=test_frame.copy(); test_save["split"]="test"
        SCORES.mkdir(parents=True,exist_ok=True)
        pd.concat([val_save,test_save],ignore_index=True).assign(score=lambda z:z.score.astype("float32"))\
          .to_parquet(score_path,index=False)
    val_frame["score"]=vp; test_frame["score"]=tp
    valm=validation_metrics(val_frame,vp,seed+13)
    rows=[]
    if scheme=="B": rules=("RE",)
    else: rules=("RQ","RV")
    # a_rel is set from validation prevalence so the test label distribution
    # never determines an alarm threshold or alarm budget.
    prevalence=float(val_frame.y.mean())
    budgets=[("2%",.02),("5%",.05),("10%",.10),("a_rel",2.5*prevalence)]
    for rule in rules:
        for budget,a in budgets:
            alarms,skipped=(alarms_re(test_frame.assign(score=tp),a) if rule=="RE"
                            else (alarms_rq(test_frame.assign(score=tp),val_frame.assign(score=vp),a),0)
                            if rule=="RQ" else (alarms_rv(test_frame.assign(score=tp),val_frame.assign(score=vp),a),0))
            metrics=evaluate(test_frame.assign(score=tp),alarms,sims=200,seed=seed)
            rows.append({"exp":"E1","scheme":scheme,"label_N":label_n,"label_p":.02,
                "feature_set":set_name,"model":model_name,"rule":rule,"budget":budget,
                "budget_value":a,"era_or_fold":era,"seed":seed,"subset":"all",
                "off_budget":off_budget(metrics["alarm_day_rate"],a),"skipped_assets_re":skipped,
                "train_n":len(train),"validation_n":len(val),**metrics})
    vrow={"scheme":scheme,"era_or_fold":era,"feature_set":set_name,"model":model_name,
          **valm}
    return rows,[vrow],{"train":len(train),"val":len(val),"test":len(test)}


def append_rows(path: Path, rows: list[dict], keys: list[str]):
    if not rows:return
    new=pd.DataFrame(rows)
    old=pd.read_csv(path) if path.exists() else pd.DataFrame()
    all_rows=pd.concat([old,new],ignore_index=True).drop_duplicates(keys,keep="last")
    all_rows.to_csv(path,index=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--timebox-min",type=float,default=60)
    ap.add_argument("--sets",default="B1,B2,B3,F0,F1,F2")
    ap.add_argument("--models",default="logreg,gbm")
    ap.add_argument("--schemes",default="B,W")
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); SCORES.mkdir(parents=True,exist_ok=True)
    panel=pd.read_parquet(V3/"panel.parquet"); shared=pd.read_parquet(V3/"market_features.parquet")
    manifest=json.loads((V3/"feature_sets.json").read_text(encoding="utf-8"))
    set_names=args.sets.split(","); models=args.models.split(","); schemes=args.schemes.split(",")
    jobs=[]
    for s in schemes:
        eras=[e[0] for e in ERAS] if s=="B" else W_ERAS
        for fs in set_names:
            for model in models:
                for era in eras: jobs.append((s,fs,model,era))
    if args.smoke: jobs=jobs[:1]
    started=time.monotonic(); deadline=started+args.timebox_min*60
    print(f"[V5-E1] estimated tasks={len(jobs)}; smoke={args.smoke}; timebox={args.timebox_min:g}min; workers=1",flush=True)
    result_path=OUT/"e1_results.csv"; validation_path=OUT/"e1_validation.csv"
    completed=set();off_counts={};stopped=set()
    if args.resume and result_path.exists():
        old=pd.read_csv(result_path)
        completed=set(tuple(x) for x in old[["scheme","feature_set","model","era_or_fold"]].drop_duplicates().itertuples(index=False,name=None))
        for scheme,primary_rule,planned in (("B","RE",11),("W","RQ",7)):
            sub=old[(old.scheme==scheme)&(old.rule==primary_rule)&(old.budget=="5%")]
            for (fs,model),g in sub.groupby(["feature_set","model"]):
                key=(scheme,fs,model);off_counts[key]=int(g.off_budget.astype(bool).sum())
                if off_counts[key]>.2*planned:stopped.add(key)
    current_fs=None; frame=None; feature_list=None
    for i,(scheme,fs,model,era) in enumerate(jobs,1):
        run_key=(scheme,fs,model)
        if run_key in stopped:continue
        if (scheme,fs,model,era) in completed:continue
        now=time.monotonic()
        if now>=deadline:
            print(f"[V5-E1] timebox reached after {i-1} tasks; results checkpointed",flush=True); break
        if fs!=current_fs:
            if frame is not None: del frame
            gc.collect()
            frame,feature_list=make_data(fs,panel,shared,manifest)
            current_fs=fs
        rows,vrows,sizes=score_era(frame,feature_list,fs,model,scheme,era,False,15,501+i)
        if rows: append_rows(result_path,rows,["scheme","feature_set","model","rule","budget","era_or_fold"])
        if vrows: append_rows(validation_path,vrows,["scheme","feature_set","model","era_or_fold"])
        print(f"[V5-E1] {i}/{len(jobs)} {scheme}/{fs}/{model}/{era} n={sizes} elapsed={(time.monotonic()-started)/60:.1f}m",flush=True)
        primary_rule="RE" if scheme=="B" else "RQ"
        primary=[r for r in rows if r["rule"]==primary_rule and r["budget"]=="5%"]
        off_counts[run_key]=off_counts.get(run_key,0)+sum(bool(r["off_budget"]) for r in primary)
        planned=11 if scheme=="B" else 7
        if off_counts[run_key]>.2*planned:
            stopped.add(run_key)
            print(f"[V5-E1] STOP {scheme}/{fs}/{model}: {off_counts[run_key]}/{planned} primary runs off-budget (>20%); diagnose this model/set",flush=True)
        if args.smoke:break
    print(f"[V5-E1] ended; elapsed={(time.monotonic()-started)/60:.1f}m; completed rows={len(pd.read_csv(result_path)) if result_path.exists() else 0}",flush=True)


if __name__=="__main__": main()
