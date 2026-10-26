"""Fair V4 paired model comparison on the frozen V3 folds.

This runner reuses V3 prices/labels/folds and adds only the completed V4
market-wide blocks. It writes every validation/test score to data/v4/scores/.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier, early_stopping

from v4_eval import evaluate, rolling_alarm, off_budget

ROOT = Path(__file__).resolve().parent
V3, V4 = ROOT / "data" / "v3", ROOT / "data" / "v4"
FOLDS = [(2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 9999)]


def split_dates(max_date, first, last):
    test_start = pd.Timestamp(f"{first}-01-01")
    test_end = max_date if last == 9999 else min(max_date, pd.Timestamp(f"{last}-12-31"))
    val_end = test_start - pd.offsets.BDay(25)
    val_start = test_start - pd.offsets.BDay(2 * 252 + 25)
    train_end = val_start - pd.offsets.BDay(25)
    return train_end, val_start, val_end, test_start, test_end


def prepare() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    panel = pd.read_parquet(V3 / "panel.parquet")
    manifest = json.loads((V3 / "feature_sets.json").read_text())
    base_cols = [c for c in manifest.get("F2", []) if c in panel.columns]
    raw = pd.read_parquet(V4 / "blocks" / "V4_RAW.parquet")
    dyn = pd.read_parquet(V4 / "blocks" / "V4_DYN.parquet")
    old = pd.read_parquet(V3 / "market_features.parquet")
    old.index.name = "date"
    old = old.reset_index()
    old_cols = [c for c in old.columns if c.startswith("tda_") and
                any(token in c for token in ("_h0_totalpers", "_h1_totalpers", "_h1_maxpers",
                                             "_h1_entropy", "_h1_bar_count", "_h1_landscape_l1",
                                             "_h1_landscape_l2")) and
                any(token in c for token in ("_w60_", "_w120_")) and
                any(token in c for token in ("_k12_", "_k24_", "_kfull_"))]
    frame = panel[["date", "asset", "y_BASE", *base_cols]].copy()
    frame = frame.merge(raw, on="date", how="inner").merge(dyn, on="date", how="left")
    frame = frame.merge(old[["date", *old_cols]], on="date", how="left")
    raw_cols = [c for c in raw.columns if c != "date"]
    dyn_cols = [c for c in dyn.columns if c != "date"]
    sets = {"M0": base_cols, "OLD_TDA": old_cols, "V4_RAW": raw_cols,
            "V4_DYN": dyn_cols, "ALL_V4": sorted(set(raw_cols + dyn_cols))}
    return frame.dropna(subset=["y_BASE"]).sort_values(["date", "asset"]), sets


def fit_scores(train, validation, test, features, model_name):
    med = train[features].median(numeric_only=True).fillna(0)
    x_train = train[features].fillna(med).to_numpy(float)
    x_val = validation[features].fillna(med).to_numpy(float)
    x_test = test[features].fillna(med).to_numpy(float)
    y_train = train.y_BASE.to_numpy(int)
    if model_name == "logreg":
        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(max_iter=300, class_weight="balanced", C=.2, n_jobs=-1)
        model.fit(scaler.transform(x_train), y_train)
        return (model.predict_proba(scaler.transform(x_val))[:, 1],
                model.predict_proba(scaler.transform(x_test))[:, 1])
    model = LGBMClassifier(
        learning_rate=.03, num_leaves=15, min_child_samples=500,
        feature_fraction=.5, bagging_fraction=.7, reg_lambda=5,
        scale_pos_weight=min((len(y_train) - y_train.sum()) / max(y_train.sum(), 1), 10),
        n_estimators=600, verbosity=-1, random_state=1701,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, validation.y_BASE.to_numpy(int))],
              callbacks=[early_stopping(30, verbose=False)])
    return model.predict_proba(x_val)[:, 1], model.predict_proba(x_test)[:, 1]


def main():
    V4.joinpath("scores").mkdir(parents=True, exist_ok=True)
    frame, sets = prepare()
    rows = []
    for fold, (first, last) in enumerate(FOLDS):
        train_end, val_start, val_end, test_start, test_end = split_dates(frame.date.max(), first, last)
        train = frame[frame.date <= train_end]
        validation = frame[frame.date.between(val_start, val_end)]
        test = frame[frame.date.between(test_start, test_end)]
        for set_name, features in sets.items():
            usable = [c for c in features if c in frame.columns]
            if not usable or min(len(train), len(validation), len(test)) == 0:
                continue
            for model_name in ("logreg", "gbm"):
                val_p, test_p = fit_scores(train, validation, test, usable, model_name)
                val_score = validation[["date", "asset", "y_BASE"]].rename(columns={"y_BASE": "y"}).copy()
                test_score = test[["date", "asset", "y_BASE"]].rename(columns={"y_BASE": "y"}).copy()
                val_score["score"] = val_p.astype("float32")
                test_score["score"] = test_p.astype("float32")
                val_score["fold"] = fold; val_score["split"] = "val"
                test_score["fold"] = fold; test_score["split"] = "test"
                score_frame = pd.concat([val_score, test_score], ignore_index=True)
                score_frame.to_parquet(V4 / "scores" / f"{set_name}_{model_name}_fold{fold}.parquet", index=False)
                seeded = rolling_alarm(score_frame, .05)
                test_alarm = seeded[seeded.split == "test"]
                metric = evaluate(test_alarm, simulations=200, seed=fold + 100)
                metric.update(fold=fold, feature_set=set_name, model=model_name,
                              n=len(test), positives=int(test.y_BASE.sum()),
                              off_budget=off_budget(metric["realized_alarm_day_rate"], .05))
                rows.append(metric)
                print(f"fold={fold} set={set_name} model={model_name} n={metric['n']} "
                      f"pos={metric['n_pos']} crashes={metric['crashes_total']} "
                      f"alarms={metric['alarms_total']} rate={metric['realized_alarm_day_rate']:.3f} "
                      f"excess={metric['excess_event_recall']:.3f}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(V4 / "d1_paired.csv", index=False)
    print(result.groupby(["feature_set", "model"])[["excess_event_recall", "ap", "auc",
                                                     "realized_alarm_day_rate"]].mean().round(4))


if __name__ == "__main__":
    main()
