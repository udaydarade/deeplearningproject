"""V3.1 score generation for corrected offline metrics; outputs only data/v3/."""
from __future__ import annotations

import argparse, json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from v3_download_data import OUT
from v3_run_experiments import FOLDS, splits, _tabular_view
from v3_1_eval import alarm_days, evaluate, off_budget

SCORES = OUT / "scores"
RESULTS = OUT


def load_panel() -> tuple[pd.DataFrame, dict]:
    panel = pd.read_parquet(OUT / "panel.parquet")
    shared = pd.read_parquet(OUT / "market_features.parquet")
    sets = json.loads((OUT / "feature_sets.json").read_text(encoding="utf-8"))
    return panel, {"shared": shared, "sets": sets}


def joined(panel: pd.DataFrame, shared: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    local = [c for c in features if c in panel.columns]
    external = [c for c in features if c in shared.columns]
    if len(local) + len(external) != len(set(features)):
        raise ValueError(f"feature columns missing: {set(features)-set(local)-set(external)}")
    meta = ["asset", "date", "source_dataset", "asset_class", "y_BASE"]
    f = panel[meta + local].merge(shared[external].reset_index(), on="date", how="left", validate="many_to_one")
    return f.sort_values(["asset", "date"], kind="stable").reset_index(drop=True)


def fit_scores(data: pd.DataFrame, features: list[str], first: int, last: int,
               model_name: str, seed: int = 1701) -> tuple[pd.DataFrame, pd.DataFrame]:
    end_label = last if last != 9999 else "latest"
    fold = f"{first}-{end_label}"
    test_start, test_end, val_start, val_end, train_end = splits(data, first, last)
    tr = data.date.le(train_end) & data.y_BASE.notna()
    va = data.date.between(val_start, val_end) & data.y_BASE.notna()
    te = data.date.between(test_start, test_end) & data.y_BASE.notna()
    tri, vai, tei = [np.flatnonzero(x.to_numpy()) for x in (tr, va, te)]
    if min(map(len, (tri, vai, tei))) == 0:
        raise ValueError(f"empty train/validation/test in {fold}: {len(tri)}/{len(vai)}/{len(tei)}")
    y = data.y_BASE.fillna(0).to_numpy(np.int8)
    if model_name == "logreg":
        x = data[features].to_numpy(np.float32)
        med = np.nanmedian(x[tri], axis=0); med[~np.isfinite(med)] = 0
        x = np.where(np.isfinite(x), x, med)
        scaler = StandardScaler().fit(x[tri])
        x = scaler.transform(x).astype(np.float32)
        model = LogisticRegression(class_weight="balanced", max_iter=400, n_jobs=8,
                                   solver="lbfgs", random_state=seed)
        model.fit(x[tri], y[tri])
        all_score = model.predict_proba(x)[:, 1].astype(np.float32)
    elif model_name == "gbm":
        x, _ = _tabular_view(data, features)
        npos = max(int(y[tri].sum()), 1); nneg = max(len(tri) - npos, 1)
        model = lgb.LGBMClassifier(objective="binary", learning_rate=.03,
            num_leaves=15, min_child_samples=500, feature_fraction=.5,
            bagging_fraction=.7, bagging_freq=1, reg_lambda=5,
            scale_pos_weight=min(nneg/npos,10), n_estimators=600,
            max_bin=63, n_jobs=8, random_state=seed, verbosity=-1)
        model.fit(x[tri], y[tri], eval_set=[(x[vai], y[vai])],
                  eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        all_score = model.predict_proba(x)[:,1].astype(np.float32)
        del x
    else:
        raise ValueError(model_name)
    train = data.loc[tr, ["date","asset","y_BASE","asset_class","source_dataset"]].copy()
    train["y"] = train.pop("y_BASE").astype(np.int8); train["score"] = all_score[tri]
    val = data.loc[va, ["date","asset","y_BASE","asset_class","source_dataset"]].copy()
    test = data.loc[te, ["date","asset","y_BASE","asset_class","source_dataset"]].copy()
    val["y"] = val.pop("y_BASE").astype(np.int8); test["y"] = test.pop("y_BASE").astype(np.int8)
    val["score"] = all_score[vai]; test["score"] = all_score[tei]
    for frame in (val, test):
        frame["fold"] = fold
    return train, val, test


def write_scores(step: str, feature_set: str, model: str, fold_i: int,
                 val: pd.DataFrame, test: pd.DataFrame) -> None:
    SCORES.mkdir(parents=True, exist_ok=True)
    packed=[]
    for split, f in (("val", val), ("test", test)):
        out = f[["date","asset","fold","y","score"]].copy()
        out["split"] = split
        out["score"] = out.score.astype("float32")
        packed.append(out)
        path = SCORES / f"{step}_{feature_set}_{model}_fold{fold_i}_{split}.parquet"
        out.to_parquet(path, index=False)
    combined = pd.concat(packed, ignore_index=True).sort_values(["split","date","asset"], kind="stable")
    (SCORES / f"{step}_{feature_set}_{model}_fold{fold_i}.parquet").unlink(missing_ok=True)
    combined.to_parquet(SCORES / f"{step}_{feature_set}_{model}_fold{fold_i}.parquet", index=False)


def eval_pair(val: pd.DataFrame, test: pd.DataFrame, *, step: str,
              feature_set: str, model: str, fold: str, fold_i: int,
              seed: int, subset: str = "all", label_n: int = 15,
              label_p: float = .02) -> list[dict]:
    if subset != "all":
        val = val.loc[subset_mask(val, subset)].copy()
        test = test.loc[subset_mask(test, subset)].copy()
    rows = []
    for rule in ("RQ", "RV"):
        for a in (.02, .05, .10):
            alarms = alarm_days(test, rule, a, validation=val)
            m = evaluate(test, alarms, sims=200, seed=seed + fold_i)
            off = off_budget(m["realized_alarm_day_rate"], a)
            rows.append({"step":step,"scheme":"W","label_N":label_n,"label_p":label_p,
                "feature_set":feature_set,"model":model,"rule":rule,"alarm_budget":a,
                "fold":fold,"seed":seed+fold_i,"subset":subset,"off_budget":off,**m})
    return rows


def subset_mask(frame: pd.DataFrame, subset: str) -> np.ndarray:
    if subset == "all": return np.ones(len(frame), bool)
    if subset == "us4": return frame.asset.isin({"^GSPC","^DJI","^IXIC","^RUT"}).to_numpy()
    if subset == "all_nonstock": return frame.asset_class.ne("stock").to_numpy()
    if subset == "approx30_index_etf": return frame.asset_class.isin({"equity_index","us_etf"}).to_numpy()
    return frame.source_dataset.eq(subset).to_numpy()


def append_results(rows: list[dict], fname: str) -> None:
    path = RESULTS / fname
    frame = pd.DataFrame(rows)
    if not len(frame): return
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True)
        keys=[c for c in ("step","scheme","label_N","label_p","feature_set","model","rule","alarm_budget","fold","subset","universe","test_view") if c in frame]
        frame=frame.drop_duplicates(keys,keep="last")
    frame.to_csv(path,index=False)


def run() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="2")
    ap.add_argument("--sets", default="F0")
    ap.add_argument("--models", default="logreg,gbm")
    ap.add_argument("--folds", default="all")
    ap.add_argument("--labels", default="BASE")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    panel, aux = load_panel(); shared, sets = aux["shared"], aux["sets"]
    class_cols = [c for c in panel.columns if c.startswith("class_")]
    sets = dict(sets)
    sets.update({
        "B1": ["realized_vol_w20"],
        "B2": ["price_drawdown_60d", "price_momentum_20d"],
        "B3": ["realized_vol_w20", "price_drawdown_60d", "price_momentum_20d", *class_cols],
    })
    requested_folds = range(5) if args.folds == "all" else [int(x) for x in args.folds.split(",")]
    all_rows = []
    out_name=f"v31_step{args.step}.csv"
    off_counts={}; stopped=set()
    for fs in args.sets.split(","):
        data = joined(panel, shared, sets[fs])
        for fi in requested_folds:
            first,last = FOLDS[fi]
            fold = f"{first}-{last if last != 9999 else 'latest'}"
            for model in args.models.split(","):
                if (fs,model) in stopped: continue
                score_stem = f"step{args.step}_{fs}_{model}_fold{fi}"
                val_path, test_path = SCORES / f"{score_stem}_val.parquet", SCORES / f"{score_stem}_test.parquet"
                if args.resume and val_path.exists() and test_path.exists():
                    val=pd.read_parquet(val_path); test=pd.read_parquet(test_path)
                    print(f"resume: reuse scores for {fs}/{model}/{fold}", flush=True)
                else:
                    _,val,test = fit_scores(data, sets[fs], first,last,model,1701)
                    write_scores(f"step{args.step}",fs,model,fi,val,test)
                rows=eval_pair(val,test,step=f"{args.step}",feature_set=fs,
                    model=model,fold=fold,fold_i=fi,seed=1701)
                append_results(rows,out_name)
                all_rows.extend(rows)
                primary=[r for r in rows if r['rule']=='RQ' and r['alarm_budget']==.05]
                off_counts[(fs,model)]=off_counts.get((fs,model),0)+sum(r['off_budget'] for r in primary)
                if off_counts[(fs,model)]>1:
                    stopped.add((fs,model)); print(f"STOP {fs}/{model}: >20% RQ@5% folds are off-budget",flush=True)
                print(f"scored step={args.step} {fs}/{model}/{fold}: val={len(val)} test={len(test)}",flush=True)


if __name__ == "__main__": run()
