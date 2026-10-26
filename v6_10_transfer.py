"""V6 score-persisting N=5 Scheme-W transfer and conditional-direction run."""
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pipeline_v6.v5_eval_copy import W_ERAS, walk_forward_split
from v6_6_realtime import load_frame

ROOT = Path(__file__).resolve().parent
V6 = ROOT / "data" / "v6"


def fit(tr, va, te, features, target):
    med = tr[features].median().fillna(0)
    model = lgb.LGBMRegressor(learning_rate=.08, num_leaves=15,
        min_child_samples=500, feature_fraction=.5, bagging_fraction=.7,
        reg_lambda=5, n_estimators=200, verbosity=-1, random_state=1701)
    model.fit(tr[features].fillna(med), target,
              eval_set=[(va[features].fillna(med), -np.log(np.maximum(va.u, .002)))],
              callbacks=[lgb.early_stopping(20, verbose=False)])
    return model.predict(te[features].fillna(med))


def main():
    n = 5
    all_scores = []
    features = None
    for direction in ("down", "up"):
        frame = load_frame(n, direction).dropna(subset=["u", "y"]).sort_values(["date", "asset"])
        features = [c for c in frame.columns if c.startswith(("semivar_", "realized_", "price_"))]
        for era in W_ERAS:
            train, val, test = walk_forward_split(frame, era, n + 25)
            if not len(train) or not len(val) or not len(test):
                continue
            tr, va, te = frame.iloc[train], frame.iloc[val], frame.iloc[test]
            score = fit(tr, va, te, features, -np.log(np.maximum(tr.u, .002)))
            out = te[["date", "asset", "asset_class", "y"]].copy()
            out["direction"] = direction; out["era"] = era; out["score"] = score
            all_scores.append(out)
    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_parquet(V6 / "scheme_w_n5_scores.parquet", index=False)
    down = scores[scores.direction == "down"].merge(
        scores[scores.direction == "up"][["date", "asset", "score"]].rename(columns={"score": "up_score"}),
        on=["date", "asset"], how="inner")
    down["down_y"] = down.y
    up_labels = load_frame(5, "up")[["date", "asset", "y"]].rename(columns={"y": "up_y"})
    down = down.merge(up_labels, on=["date", "asset"], how="left")
    rows = []
    for subset, d in [("all", down)] + [(k, v) for k, v in down.groupby("asset_class")]:
        for source, target in (("score", "up_y"), ("up_score", "down_y")):
            z = d[[source, target]].dropna()
            rows.append({"subset": subset, "source": source, "target": target,
                         "n": len(z), "auc": roc_auc_score(z[target], z[source])
                         if z[target].nunique() == 2 else np.nan})
    pd.DataFrame(rows).to_csv(V6 / "t2_cross_transfer_n5.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
