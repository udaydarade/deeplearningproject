"""V3.1 Steps 3–7 score-generation runner. Uses only frozen V3 data/features."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

import numpy as np
import pandas as pd

from v3_download_data import OUT
from v3_run_experiments import FOLDS
from v3_1_train import joined, fit_scores, write_scores, eval_pair, append_results
from v3_1_eval import alarm_days, evaluate, off_budget
from v3_1_blocks import build_blocks


def append(path: str, rows: list[dict]):
    append_results(rows, path)

def append_validation(rows: list[dict], filename: str):
    path=OUT/filename; new=pd.DataFrame(rows)
    old=pd.read_csv(path) if path.exists() else pd.DataFrame()
    frame=pd.concat([old,new],ignore_index=True)
    if len(frame): frame=frame.drop_duplicates(['fold','feature_set','model'],keep='last')
    frame.to_csv(path,index=False)


def validation_rows(train: pd.DataFrame, val: pd.DataFrame, fi: int, fs: str, model: str) -> dict:
    alarms = alarm_days(val, "RQ", .05, validation=train)
    m = evaluate(val, alarms, sims=200, seed=7800+fi)
    return {"fold":fi,"feature_set":fs,"model":model,**m,
            "off_budget":off_budget(m["realized_alarm_day_rate"],.05)}


def run_classical(sets: list[str], models: list[str], step: str, resume: bool):
    panel = pd.read_parquet(OUT/"panel.parquet")
    shared = pd.read_parquet(OUT/"market_features.parquet")
    manifest = json.loads((OUT/"feature_sets.json").read_text(encoding="utf-8"))
    all_results=[]; validation=[]; off_counts={}; stopped=set()
    for fs in sets:
        data=joined(panel,shared,manifest[fs])
        for fi,(first,last) in enumerate(FOLDS):
            fold=f"{first}-{last if last!=9999 else 'latest'}"
            for model in models:
                if (fs,model) in stopped: continue
                stem=f"step{step}_{fs}_{model}_fold{fi}"
                pv=OUT/"scores"/f"{stem}_val.parquet"; pt=OUT/"scores"/f"{stem}_test.parquet"
                if resume and pv.exists() and pt.exists():
                    val=pd.read_parquet(pv); test=pd.read_parquet(pt)
                    # Training history must be re-derived only for validation RQ selection.
                    train,_,_=fit_scores(data,manifest[fs],first,last,model,1701+fi)
                else:
                    train,val,test=fit_scores(data,manifest[fs],first,last,model,1701+fi)
                    write_scores(f"step{step}",fs,model,fi,val,test)
                vrow=validation_rows(train,val,fi,fs,model)
                validation.append(vrow)
                append_validation([vrow],f'v31_step{step}_validation.csv')
                rows=eval_pair(val,test,step=step,feature_set=fs,model=model,
                    fold=fold,fold_i=fi,seed=1701,subset="all")
                all_results.extend(rows)
                append_results(rows,f"v31_step{step}.csv")
                primary=[r for r in rows if r['rule']=='RQ' and r['alarm_budget']==.05]
                off_counts[(fs,model)]=off_counts.get((fs,model),0)+sum(r['off_budget'] for r in primary)
                if off_counts[(fs,model)]>1:
                    stopped.add((fs,model)); print(f"STOP {fs}/{model}: >20% RQ@5% folds are off-budget",flush=True)
                print(f"step{step}: {fs}/{model}/{fold} completed",flush=True)
    if validation:
        append_validation(validation,f'v31_step{step}_validation.csv')
    return pd.DataFrame(validation)


def choose_best_classical(df: pd.DataFrame) -> str:
    eligible=df.loc[~df.off_budget]
    means=eligible.groupby("feature_set").excess_event_recall.mean()
    if means.empty: raise RuntimeError("no on-budget validation cells; cannot select classical baseline")
    best=str(means.idxmax())
    (OUT/"v31_best_classical.json").write_text(json.dumps({"feature_set":best,"validation_mean_excess_event_recall":float(means[best])},indent=2))
    return best


def tda_blocks():
    manifest=json.loads((OUT/"feature_sets.json").read_text(encoding="utf-8"))
    return build_blocks(manifest)


def run_step4(best: str, models: list[str], resume: bool):
    # This fits the registered baseline-plus-one-block models; F4c selection is
    # performed after event-paired inference from the saved fold predictions.
    panel=pd.read_parquet(OUT/"panel.parquet"); shared=pd.read_parquet(OUT/"market_features.parquet")
    manifest=json.loads((OUT/"feature_sets.json").read_text(encoding="utf-8"))
    blocks=tda_blocks(); base=manifest[best]
    configs={}
    for b,cols in blocks.items():
        if not cols:
            print(f"step4 unavailable block {b}: no columns in frozen F4 manifest")
            continue
        if len(cols)>60: raise ValueError(f"{b} has {len(cols)} columns >60")
        configs[f"{best}+{b}"]=list(dict.fromkeys(base+cols))
    # TDA-only diagnostic: all bounded blocks plus class one-hot.
    cls=[c for c in panel.columns if c.startswith("class_")]
    configs["TDA_only"] = list(dict.fromkeys([c for block in blocks.values() for c in block]+cls))
    block_counts={k:len(v) for k,v in blocks.items()}
    (OUT/"v31_tda_blocks.json").write_text(json.dumps({"counts":block_counts,"blocks":blocks},indent=2))
    all_rows=[]; val_rows=[]; off_counts={}; stopped=set()
    for fs,features in configs.items():
        data=joined(panel,shared,features)
        for fi,(first,last) in enumerate(FOLDS):
            fold=f"{first}-{last if last!=9999 else 'latest'}"
            for model in models:
                if (fs,model) in stopped: continue
                stem=f"step4_{fs}_{model}_fold{fi}".replace("+","_")
                pv=OUT/"scores"/f"{stem}_val.parquet"; pt=OUT/"scores"/f"{stem}_test.parquet"
                if resume and pv.exists() and pt.exists():
                    print(f"step4 reuse available validation/test predictions {fs}/{model}/{fold}",flush=True)
                    val=pd.read_parquet(pv); test=pd.read_parquet(pt)
                    # Refit is needed only to derive the train-history scores
                    # used for validation-only rule selection.
                    train,_,_=fit_scores(data,features,first,last,model,2718+fi)
                else:
                    train,val,test=fit_scores(data,features,first,last,model,2718+fi)
                write_scores("step4",fs.replace("+","_"),model,fi,val,test)
                vrow=validation_rows(train,val,fi,fs,model)
                val_rows.append(vrow)
                append_validation([vrow],'v31_step4_validation.csv')
                rows=eval_pair(val,test,step="4",feature_set=fs,model=model,
                    fold=fold,fold_i=fi,seed=2718)
                all_rows.extend(rows); append_results(rows,"v31_step4.csv")
                primary=[r for r in rows if r['rule']=='RQ' and r['alarm_budget']==.05]
                off_counts[(fs,model)]=off_counts.get((fs,model),0)+sum(r['off_budget'] for r in primary)
                if off_counts[(fs,model)]>1:
                    stopped.add((fs,model)); print(f"STOP {fs}/{model}: >20% RQ@5% folds are off-budget",flush=True)
                print(f"step4: {fs}/{model}/{fold} scored ({len(features)} cols)",flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("step",choices=["3","4"])
    ap.add_argument("--models",default="logreg,gbm")
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args()
    if args.step=="3":
        validation=run_classical(["F0","F1","F2"],args.models.split(","),"3",args.resume)
        best=choose_best_classical(validation)
        print(f"BEST_CLASSICAL selected on validation-only RQ excess event recall@5%: {best}")
    else:
        best=json.loads((OUT/"v31_best_classical.json").read_text())["feature_set"]
        run_step4(best,args.models.split(","),args.resume)


if __name__=="__main__": main()
