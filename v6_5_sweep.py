"""Checkpointed V6 E-severity blocked-era sweep.

This is deliberately separate from V5 and uses no TDA columns.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from pipeline_v6.v5_eval_copy import ERAS, blocked_split, alarms_re, evaluate, off_budget

ROOT = Path(__file__).resolve().parent
V6 = ROOT / "data" / "v6"
HORIZONS, DIRECTIONS = (3, 5, 10, 15, 30), ("down", "up")


def load_frame(n, direction):
    panel = pd.read_parquet(V6 / "panel.parquet")
    cols = ["date", "asset", "asset_class", "realized_vol_w20",
            "price_drawdown_60d", "price_momentum_1d", "price_momentum_5d"]
    cols += [c for c in panel.columns if c.startswith("semivar_")]
    panel = panel[[c for c in cols if c in panel.columns]]
    pieces = []
    for f in (V6 / "labels").glob("*.parquet"):
        d = pd.read_parquet(f, columns=["date", "asset", f"u_{direction}_N{n}",
                                        f"y_{direction}_N{n}_p5"])
        pieces.append(d)
    labels = pd.concat(pieces, ignore_index=True)
    return panel.merge(labels, on=["date", "asset"], how="inner").rename(
        columns={f"u_{direction}_N{n}": "u", f"y_{direction}_N{n}_p5": "y"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timebox-min", type=float, default=45)
    args = ap.parse_args()
    out = V6 / "sweep_e_sev.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    done = (set(zip(old.N, old.direction, old.era))
            if {"N", "direction", "era"}.issubset(old.columns) else set())
    started = time.monotonic()
    rows = []
    for n in HORIZONS:
        for direction in DIRECTIONS:
            frame = load_frame(n, direction)
            features = [c for c in frame.columns if c.startswith(("semivar_", "realized_", "price_"))]
            frame = frame.dropna(subset=["u", "y"]).sort_values(["date", "asset"])
            for era, _, _ in ERAS:
                if (n, direction, era) in done:
                    continue
                if time.monotonic() - started > args.timebox_min * 60:
                    pd.DataFrame(rows).to_csv(out, index=False)
                    print(f"[V6-SWEEP] timebox; rows={len(rows)}")
                    return
                train, val, test = blocked_split(frame, era, n + 40)
                if not len(train) or not len(val) or not len(test):
                    continue
                tr, va, te = frame.iloc[train], frame.iloc[val], frame.iloc[test]
                med = tr[features].median().fillna(0)
                xtr, xva, xte = tr[features].fillna(med), va[features].fillna(med), te[features].fillna(med)
                target = -np.log(np.maximum(tr.u.to_numpy(float), .002))
                model = lgb.LGBMRegressor(learning_rate=.08, num_leaves=15,
                    min_child_samples=500, feature_fraction=.5, bagging_fraction=.7,
                    reg_lambda=5, n_estimators=200, verbosity=-1, random_state=1701)
                model.fit(xtr, target, eval_set=[(xva, -np.log(np.maximum(va.u, .002)))],
                          callbacks=[lgb.early_stopping(20, verbose=False)])
                te = te.copy(); te["score"] = model.predict(xte)
                alarm, skipped = alarms_re(te, .05)
                metric = evaluate(te[["date", "asset", "y", "score"]], alarm)
                metric.update(N=n, direction=direction, era=era, engine="E-sev",
                              feature_set="BEST_SET_SEMIVAR", model="gbm_lite",
                              budget=.05, skipped_assets=skipped,
                              off_budget=off_budget(metric["alarm_day_rate"], .05))
                rows.append(metric)
                print(f"[V6-SWEEP] N={n} {direction} {era} n={metric['n']} "
                      f"events={metric['crashes_total']} excess={metric['excess_event_recall']:.3f}",
                      flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[V6-SWEEP] complete rows={len(rows)}")


if __name__ == "__main__":
    main()
