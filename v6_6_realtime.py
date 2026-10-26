"""V6 real-time Scheme-W validation and E-severity versus binary checks."""
from __future__ import annotations
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from pipeline_v6.v5_eval_copy import W_ERAS, walk_forward_split, alarms_rq, evaluate, off_budget

ROOT = Path(__file__).resolve().parent
V6 = ROOT / "data" / "v6"
HORIZONS = (5, 15, 30)


def load_frame(n, direction):
    panel = pd.read_parquet(V6 / "panel.parquet")
    base = ["date", "asset", "asset_class", "realized_vol_w20",
            "price_drawdown_60d", "price_momentum_1d", "price_momentum_5d"]
    semivar = [c for c in panel.columns if c.startswith("semivar_")]
    panel = panel[[c for c in base + semivar if c in panel.columns]]
    labels = []
    for f in (V6 / "labels").glob("*.parquet"):
        labels.append(pd.read_parquet(f, columns=["date", "asset", f"u_{direction}_N{n}",
                                                   f"y_{direction}_N{n}_p5"]))
    return panel.merge(pd.concat(labels, ignore_index=True), on=["date", "asset"],
                       how="inner").rename(columns={f"u_{direction}_N{n}": "u",
                                                     f"y_{direction}_N{n}_p5": "y"})


def fit_model(tr, va, te, features, target_kind):
    med = tr[features].median().fillna(0)
    xtr, xva, xte = tr[features].fillna(med), va[features].fillna(med), te[features].fillna(med)
    if target_kind == "sev":
        ytr = -np.log(np.maximum(tr.u.to_numpy(float), .002))
        yva = -np.log(np.maximum(va.u.to_numpy(float), .002))
        model = lgb.LGBMRegressor
    else:
        ytr, yva = tr.y.to_numpy(int), va.y.to_numpy(int)
        model = lgb.LGBMClassifier
    m = model(learning_rate=.08, num_leaves=15, min_child_samples=500,
              feature_fraction=.5, bagging_fraction=.7, reg_lambda=5,
              n_estimators=200, verbosity=-1, random_state=1701)
    m.fit(xtr, ytr, eval_set=[(xva, yva)],
          callbacks=[lgb.early_stopping(20, verbose=False)])
    score = m.predict(xte) if target_kind == "sev" else m.predict_proba(xte)[:, 1]
    return score


def main():
    out_path = V6 / "scheme_w_results.csv"
    rows = []
    started = time.monotonic()
    for n in HORIZONS:
        for direction in ("down", "up"):
            frame = load_frame(n, direction).dropna(subset=["u", "y"]).sort_values(["date", "asset"])
            semivar = [c for c in frame.columns if c.startswith("semivar_")]
            b3 = [c for c in ("realized_vol_w20", "price_drawdown_60d",
                              "price_momentum_1d") if c in frame.columns]
            best = b3 + semivar
            for era in W_ERAS:
                train, val, test = walk_forward_split(frame, era, n + 25)
                if not len(train) or not len(val) or not len(test):
                    continue
                tr, va, te = frame.iloc[train], frame.iloc[val], frame.iloc[test]
                for set_name, features in (("B3", b3), ("BEST_SET_SEMIVAR", best)):
                    for engine in (("E-sev", "sev"), ("E-bin", "bin")):
                        score = fit_model(tr, va, te, features, engine[1])
                        test_score = te[["date", "asset", "y"]].copy()
                        test_score["score"] = score
                        validation = va[["date", "asset"]].copy()
                        validation["score"] = fit_model(tr, va, va, features, engine[1])
                        alarm = alarms_rq(test_score, validation, .05, window=250)
                        metric = evaluate(test_score, alarm)
                        metric.update(scheme="W", N=n, direction=direction, era=era,
                                      engine=engine[0], feature_set=set_name, model="gbm_lite",
                                      rule="RQ", budget=.05,
                                      off_budget=off_budget(metric["alarm_day_rate"], .05))
                        rows.append(metric)
                        print(f"[V6-W] N={n} {direction} {era} {set_name} {engine[0]} "
                              f"excess={metric['excess_event_recall']:.3f}", flush=True)
                pd.DataFrame(rows).to_csv(out_path, index=False)
                if time.monotonic() - started > 115 * 60:
                    print("[V6-W] timebox reached")
                    return
    print(f"[V6-W] complete rows={len(rows)}")


if __name__ == "__main__":
    main()
