"""V5 E2 paired feature-block ablations; test metrics never choose a block."""
from __future__ import annotations

import argparse,gc,json,re,time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from v5_1_e1 import append_rows,fit_score,make_data
from v5_eval import ERAS,W_ERAS,alarms_re,alarms_rq,alarms_rv,blocked_split,evaluate,off_budget,walk_forward_split

ROOT=Path(__file__).resolve().parent;V3=ROOT/"data"/"v3";V5=ROOT/"data"/"v5";B=V5/"blocks";SCORES=V5/"scores"


def feature_groups(panel,manifest):
    tda=pd.read_parquet(B/"V5_TDA.parquet");tda_cols=[c for c in tda if c!="date"]
    groups={"TDA_small":[c for c in tda_cols if re.search(r"_k(6|12|24)_",c)],
            "TDA_big":[c for c in tda_cols if re.search(r"_k(48|96|all)_",c)]}
    cl=pd.read_parquet(B/"V5_CLUSTER.parquet");groups["CLUSTER"]=[c for c in cl if c!="date"]
    pr=pd.concat([pd.read_parquet(p) for p in sorted((B/"V5_PRICE1D").glob("*.parquet"))],ignore_index=True)
    groups["PRICE1D"]=[c for c in pr if c not in {"date","asset"}]
    groups["TAKENS_VOL"]=[]
    takens_dir=B/"V5_TAKENS_VOL"
    if takens_dir.exists() and list(takens_dir.glob("*.parquet")):
        tak=pd.concat([pd.read_parquet(p) for p in sorted(takens_dir.glob("*.parquet"))],ignore_index=True)
        groups["TAKENS_VOL"]=[c for c in tak if c not in {"date","asset","delay"}]
    else:tak=pd.DataFrame(columns=["date","asset"])
    groups["DYN"]=[]
    if (B/"V5_DYN.parquet").exists():
        d=pd.read_parquet(B/"V5_DYN.parquet");groups["DYN"]=[c for c in d if c!="date"]
    f4=manifest.get("F4",[])
    groups["M_old"]=[c for c in f4 if c.startswith("tda_") and
        any(c.endswith(s) for s in ("_totalpers","_maxpers","_entropy","_bar_count"))]
    groups["M_allTDA"]=tda_cols
    groups["TDA_only_class"]=[*tda_cols,*[c for c in panel if c.startswith("class_")]]
    oldcols=[c for c in groups["M_old"] if c in panel.columns]
    groups["__date_frames"]={"TDA_small":tda,"TDA_big":tda,"M_allTDA":tda,"CLUSTER":cl,
                              "TDA_only_class":tda,"M_old":panel[["date",*oldcols]].drop_duplicates("date")}
    groups["__date_frames"]["DYN"]=pd.read_parquet(B/"V5_DYN.parquet") if (B/"V5_DYN.parquet").exists() else pd.DataFrame(columns=["date"])
    groups["__asset_frame"]=pr
    groups["__takens_frame"]=tak
    # Registered curve ablations (subset labels come from feature names only).
    for k in ("6","12","24","48","96"):
        for cloud in ("A_corr","A_raw"):
            cols=[c for c in tda_cols if re.match(rf"^v5_NS_{cloud}_k{k}_.*_h1_",c)]
            if cols:groups[f"curve_{cloud}_k{k}"]=cols
    for cloud in ("A_corr","A_raw"):
        for universe,suffix in (("NS",""),("ALLS","_stocks")):
            cols=[c for c in tda_cols if re.match(rf"^v5_{universe}_{cloud}_kall_.*_h1_",c)]
            if cols:groups[f"curve_{cloud}_kall{suffix}"]=cols
    for cloud in ("T_std","T_raw","A_corr","A_raw"):
        cols=[c for c in tda_cols if re.match(rf"^v5_NS_{cloud}_k24_.*_h1_",c)]
        if cols:groups[f"type_{cloud}_k24"]=cols
    for k in ("12","24","48"):
        for cloud in ("A_corr","A_raw","T_raw","T_std"):
            for hs,tag in (((1,),"H1"),((0,1),"H01"),((0,1,2),"H012")):
                cols=[c for c in tda_cols if re.match(rf"^v5_NS_{cloud}_k{k}_w60_h([012])_",c)
                      and int(re.search(r"_h([012])_",c).group(1)) in hs]
                if cols:groups[f"homology_{cloud}_k{k}_{tag}"]=cols
    for w in (20,60,120,250):
        cols=[c for c in tda_cols if re.match(rf"^v5_NS_A_(corr|raw)_k(6|12|24)_w{w}_h1_",c)]
        if cols:groups[f"window_A_w{w}"]=cols
    return groups


def assemble(name,base,base_features,groups):
    if name=="M0":return base.copy(),list(base_features)
    only_tda=name=="TDA_only_class"
    parts=["TDA_only_class"] if only_tda else name.split("+")
    extra=[]
    for part in (parts if only_tda else parts[1:]):extra.extend(groups.get(part,[]))
    extra=list(dict.fromkeys(extra))
    if only_tda:
        frame=base[["date","asset","y"]].copy();base_features=[]
    else:frame=base.copy()
    date_frames=groups["__date_frames"];asset_frame=groups["__asset_frame"]
    date_cols=[];asset_cols=[]
    for part in parts:
        cols=groups.get(part,[])
        if part in date_frames:date_cols.extend(c for c in cols if c not in date_cols)
        elif part in ("PRICE1D","TAKENS_VOL"):asset_cols.extend(cols)
        elif part=="M_old":date_cols.extend(c for c in cols if c not in date_cols)
        elif part=="TDA_only_class":date_cols.extend(c for c in cols if c.startswith("v5_"))
        else:date_cols.extend(c for c in cols if c.startswith("v5_"))
    if date_cols:
        for part in parts:
            if part not in date_frames:continue
            src=date_frames.get(part)
            if src is None:continue
            cols=[c for c in groups.get(part,[]) if c in src.columns]
            if cols:frame=frame.merge(src[["date",*cols]].drop_duplicates("date"),on="date",how="left",validate="many_to_one")
    if asset_cols:
        for part in ("PRICE1D","TAKENS_VOL"):
            target=groups["__asset_frame"] if part=="PRICE1D" else groups["__takens_frame"]
            cols=[c for c in dict.fromkeys(asset_cols) if c in target.columns and c not in frame.columns]
            if cols:frame=frame.merge(target[["date","asset",*cols]],on=["date","asset"],how="left",validate="one_to_one")
    if only_tda:
        cls=[c for c in groups["TDA_only_class"] if c.startswith("class_") and c in base.columns]
        for c in cls:
            if c not in frame:frame[c]=base[c].to_numpy()
    features=list(dict.fromkeys([*base_features,*extra]))
    features=[c for c in features if c in frame.columns]
    return frame,features


def prune(frame,features,train,seed):
    if len(features)<=400:return features
    rng=np.random.default_rng(seed);sample=np.asarray(train,dtype=np.int64)
    if len(sample)>200_000:sample=np.sort(rng.choice(sample,200_000,replace=False))
    x=frame.iloc[sample][features].to_numpy(np.float32);y=frame.y.to_numpy(np.int8)
    pos=max(int(y[sample].sum()),1);neg=max(len(sample)-pos,1)
    m=lgb.LGBMClassifier(objective="binary",learning_rate=.03,num_leaves=15,
        min_child_samples=500,feature_fraction=.5,bagging_fraction=.7,bagging_freq=1,
        reg_lambda=5,scale_pos_weight=min(neg/pos,10),n_estimators=100,max_bin=63,
        n_jobs=1,random_state=seed,verbosity=-1)
    m.fit(x,y[sample]);gain=m.booster_.feature_importance(importance_type="gain")
    ix=np.argsort(-gain,kind="stable")[:400]
    return [features[i] for i in ix]


def score_one(frame,features,scheme,era,model,seed):
    deep=False;purge=15+(40 if scheme=="B" else 25)
    tr,va,te=(blocked_split(frame,era,purge,deep=False,validation_purge_sessions=40)
              if scheme=="B" else walk_forward_split(frame,era,purge))
    if min(len(tr),len(va),len(te))==0:return None
    selected=prune(frame,features,tr,seed)
    vp,tp=fit_score(frame,selected,tr,va,te,model,seed)
    vf=frame.iloc[va][["date","asset","y"]].copy().reset_index(drop=True);vf["score"]=vp
    tf=frame.iloc[te][["date","asset","y"]].copy().reset_index(drop=True);tf["score"]=tp
    valalarm,skip=alarms_re(vf,.05);vm=evaluate(vf,valalarm,sims=200,seed=seed+1)
    vrow={"scheme":scheme,"feature_set":"","model":model,"era_or_fold":era,
        "validation_excess_event_recall":vm["excess_event_recall"],"validation_ap":vm["ap"],
        "validation_alarm_day_rate":vm["alarm_day_rate"],"validation_n":len(vf),
        "validation_n_pos":int(vf.y.sum()),"validation_crashes_total":vm["crashes_total"],
        "features_used":len(selected),"skipped_assets_re":skip}
    primary=[]
    for rule in (("RE",) if scheme=="B" else ("RQ","RV")):
        for budget,a0 in (("2%",.02),("5%",.05),("10%",.10),("a_rel",2.5*float(vf.y.mean()))):
            if rule=="RE":alarm,skip=alarms_re(tf,a0)
            elif rule=="RQ":alarm=alarms_rq(tf,vf,a0);skip=0
            else:alarm=alarms_rv(tf,vf,a0);skip=0
            m=evaluate(tf,alarm,sims=200,seed=seed)
            row={"exp":"E2","scheme":scheme,"label_N":15,"label_p":.02,
                "feature_set":"","model":model,"rule":rule,"budget":budget,"budget_value":a0,
                "era_or_fold":era,"seed":seed,"subset":"all","off_budget":off_budget(m["alarm_day_rate"],a0),
                "train_n":len(tr),"validation_n":len(va),"features_used":len(selected),**m}
            primary.append(row)
    vf["split"]="validation";tf["split"]="test"
    scores=pd.concat([vf,tf],ignore_index=True)[["date","asset","split","y","score"]]
    return primary,vrow,scores


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--timebox-min",type=float,default=90)
    ap.add_argument("--models",default="logreg,gbm");ap.add_argument("--schemes",default="B,W")
    ap.add_argument("--smoke",action="store_true");args=ap.parse_args()
    panel=pd.read_parquet(V3/"panel.parquet");shared=pd.read_parquet(V3/"market_features.parquet")
    manifest=json.loads((V3/"feature_sets.json").read_text(encoding="utf-8"))
    base_set=json.loads((V5/"base_set.json").read_text(encoding="utf-8"))["feature_set"]
    base,base_features=make_data(base_set,panel,shared,manifest)
    groups=feature_groups(panel,manifest)
    # P1 model/block tests first; registered cloud curves follow if the hard box remains.
    b3,b3_features=make_data("B3",panel,shared,manifest)
    block_names=["TDA_small","TDA_big","CLUSTER","PRICE1D"]
    block_names.extend(x for x in ("TAKENS_VOL","DYN","M_old","M_allTDA") if groups[x])
    names=["M0","B3",*[f"M0+{x}" for x in block_names],"TDA_only_class"]
    names += [f"M0+{x}" for x in groups if x.startswith(("curve_","type_","homology_","window_"))]
    jobs=[(s,n,m,e[0]) for n in names for m in args.models.split(",") for s in args.schemes.split(",")
          for e in (ERAS if s=="B" else [next(x for x in ERAS if x[0]==w) for w in W_ERAS])]
    planned=len(jobs)
    if args.smoke:jobs=jobs[:1]
    SCORES.mkdir(parents=True,exist_ok=True);start=time.perf_counter();deadline=start+args.timebox_min*60
    print(f"[V5-E2] jobs={len(jobs)}/{planned} sets={len(names)} smoke={args.smoke}; rough full-grid estimate={planned*.5:.0f}min; box={args.timebox_min:g}m, one worker",flush=True)
    result=V5/"e2_results.csv";validation=V5/"e2_validation.csv";completed=set();off={};stopped=set();cached={};current_name=None
    if result.exists():
        old=pd.read_csv(result);completed=set(tuple(x) for x in old[["scheme","feature_set","model","era_or_fold"]].drop_duplicates().itertuples(index=False,name=None))
        for scheme,rule,planned in (("B","RE",11),("W","RQ",7)):
            x=old[(old.scheme==scheme)&(old.rule==rule)&(old.budget=="5%")]
            for (fs,model),g in x.groupby(["feature_set","model"]):
                key=(scheme,fs,model);off[key]=int(g.off_budget.astype(bool).sum())
                if off[key]>.2*planned:stopped.add(key)
    for i,(scheme,name,model,era) in enumerate(jobs,1):
        key=(scheme,name,model)
        if key in stopped or (scheme,name,model,era) in completed:continue
        if time.perf_counter()>=deadline:print(f"[V5-E2] hard box reached at {i-1}/{len(jobs)} jobs",flush=True);break
        if name!=current_name:
            cached.clear();gc.collect();current_name=name
        if name not in cached:
            if name=="M0":cached[name]=(base,base_features)
            elif name=="B3":cached[name]=(b3,b3_features)
            else:cached[name]=assemble(name,base,base_features,groups)
        frame,features=cached[name]
        z=score_one(frame,features,scheme,era,model,7300+i)
        if z is None:continue
        rows,vrow,scores=z
        for r in rows:r["feature_set"]=name
        vrow["feature_set"]=name
        append_rows(result,rows,["scheme","feature_set","model","rule","budget","era_or_fold"])
        append_rows(validation,[vrow],["scheme","feature_set","model","era_or_fold"])
        stem=f"{scheme}_{name}_{model}_{era}".replace("+","_")
        scores.to_parquet(SCORES/f"{stem}.parquet",index=False)
        primary=[r for r in rows if r["budget"]=="5%" and r["rule"]==("RE" if scheme=="B" else "RQ")]
        off[key]=off.get(key,0)+sum(bool(r["off_budget"]) for r in primary)
        if off[key]>.2*(11 if scheme=="B" else 7):
            stopped.add(key);print(f"[V5-E2] STOP {name}/{model}/{scheme}: off-budget >20%; inspect score ties/rates",flush=True)
        print(f"[V5-E2] {i}/{len(jobs)} {scheme}/{name}/{model}/{era} rows={len(scores)} p={int(scores.y.sum())} events={primary[0]['crashes_total'] if primary else 'NA'} elapsed={(time.perf_counter()-start)/60:.1f}m",flush=True)
        del rows,scores;gc.collect()
    print(f"[V5-E2] end elapsed={(time.perf_counter()-start)/60:.1f}m",flush=True)


if __name__=="__main__":main()
