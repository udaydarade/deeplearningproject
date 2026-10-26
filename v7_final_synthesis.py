"""Complete V7_FINAL_SYNTHESIS experiment.

Outputs are isolated under data/v7_final/. The run is resumable at the cell
level and never modifies V1-V6 artifacts.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from pipeline_v5.sequence import fit_lazy_sequence
from v5_eval import alarms_rq, evaluate

ROOT = Path(__file__).resolve().parent
V5, V6, OUT = ROOT / "data/v5", ROOT / "data/v6", ROOT / "data/v7_final"
SCORE_DIR = OUT / "scores"
FOLDS = (
    ("2011_2015", "2011-01-01", "2015-12-31"),
    ("2016_2020", "2016-01-01", "2020-12-31"),
    ("2021_plus", "2021-01-01", "2100-01-01"),
)
SEEDS = (42, 43, 44)
CORE = {"^GSPC", "^DJI", "^IXIC", "^RUT"}
CLASSICAL = ["csd_ac1_w50", "csd_variance_w50", "csd_skew_w50",
             "csd_kurtosis_w50", "csd_ac1_w100", "csd_variance_w100",
             "csd_skew_w100", "csd_kurtosis_w100", "realized_vol_w20"]
SEMIVAR = ["semivar_up_20", "semivar_down_20", "semivar_logratio_20",
           "semivar_skew_20", "semivar_up_60", "semivar_down_60",
           "semivar_logratio_60", "semivar_skew_60", "largest_gain_20",
           "largest_loss_20", "close_position_60", "close_position_252",
           "days_since_high_60", "days_since_low_60", "vol_compression_20",
           "up_streak", "down_streak"]
PEER = ["price_momentum_20d", "price_momentum_60d", "price_volatility_20d",
        "price_drawdown_60d", "relative_rank_mom20", "relative_rank_vol20",
        "peer_mean_momentum20", "peer_mean_vol20", "relative_corr_60d"]


def load_data():
    f = pd.read_parquet(V6 / "panel.parquet")
    f["date"] = pd.to_datetime(f.date)
    # V6 stores N=15 p=2 in per-asset label checkpoints rather than the
    # compact panel. Load only that required column and merge it once.
    parts = []
    for p in (V6 / "labels").glob("*.parquet"):
        parts.append(pd.read_parquet(p, columns=["date", "asset", "y_down_N15_p2"]))
    labs = pd.concat(parts, ignore_index=True)
    f = f.drop(columns=["y_down_N15_p2"], errors="ignore").merge(
        labs, on=["date", "asset"], how="left", validate="one_to_one")
    tda_path = V5 / "blocks/V5_landscape_sampled.parquet"
    tda_cols = []
    if tda_path.exists():
        t = pd.read_parquet(tda_path)
        tda_cols = [c for c in t.columns if c != "date"][:36]
        f = f.merge(t[["date", *tda_cols]].drop_duplicates("date"),
                    on="date", how="left", validate="many_to_one")
    sets = {
        "CLASSICAL_CSD": [c for c in CLASSICAL if c in f],
        "SEMIVAR_MOM": [c for c in CLASSICAL + SEMIVAR if c in f],
        "PEER_CROSS": [c for c in CLASSICAL + SEMIVAR + PEER if c in f],
        "TDA_ONLY_NORM": tda_cols,
    }
    sets["GRAND_COMBINED"] = sets["PEER_CROSS"] + tda_cols
    return f.sort_values(["asset", "date"]).reset_index(drop=True), sets


def split(f, fold):
    _, start, end = fold
    test = (f.date >= start) & (f.date <= end)
    dates = pd.DatetimeIndex(f.loc[f.date < start, "date"].unique()).sort_values()
    purge = set(dates[-25:])
    td = pd.DatetimeIndex(f.loc[test, "date"].unique()).sort_values()
    if not len(td):
        return np.array([], int), np.array([], int), np.array([], int)
    val_end = td[0]
    val_start = val_end - pd.DateOffset(years=2)
    val = (f.date >= val_start) & (f.date < val_end) & ~f.date.isin(purge)
    train = (f.date < val_start) & ~f.date.isin(purge)
    return np.flatnonzero(train), np.flatnonzero(val), np.flatnonzero(test)


def scale_matrix(f, cols, train):
    x = f[cols].to_numpy(np.float32)
    med = np.nanmedian(x[train], axis=0)
    med[~np.isfinite(med)] = 0
    x = np.where(np.isfinite(x), x, med[None, :])
    sc = StandardScaler().fit(x[train])
    return sc.transform(x).astype(np.float32)


def fit_tabular(x, y, tr, va, te, model, seed):
    if model == "logreg":
        m = LogisticRegression(class_weight="balanced", max_iter=400,
                               solver="lbfgs", random_state=seed)
        m.fit(x[tr], y[tr])
    else:
        pos, neg = max(int(y[tr].sum()), 1), max(len(tr) - int(y[tr].sum()), 1)
        m = lgb.LGBMClassifier(objective="binary", learning_rate=.03,
            num_leaves=15, min_child_samples=300, feature_fraction=.7,
            bagging_fraction=.8, bagging_freq=1, reg_lambda=5,
            scale_pos_weight=min(neg / pos, 10), n_estimators=600,
            n_jobs=1, random_state=seed, verbosity=-1)
        m.fit(x[tr], y[tr], eval_set=[(x[va], y[va])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return m.predict_proba(x[va])[:, 1], m.predict_proba(x[te])[:, 1]


def sequence_arrays(f, z):
    arrays, assets, positions = {}, np.empty(len(f), object), np.zeros(len(f), int)
    for asset, idx in f.groupby("asset", sort=False).groups.items():
        ix = np.asarray(list(idx), int)
        ix = ix[np.argsort(f.iloc[ix].date.to_numpy(), kind="stable")]
        arrays[asset] = z[ix]
        assets[ix] = asset
        positions[ix] = np.arange(len(ix))
    return arrays, assets, positions


def lead_time(df, alarm):
    total, count = 0.0, 0
    for _, g in df.assign(alarm=alarm).sort_values("date").groupby("asset"):
        y, a = g.y.to_numpy(bool), g.alarm.to_numpy(bool)
        starts = np.flatnonzero(y & ~np.r_[False, y[:-1]])
        for s in starts:
            prior = np.flatnonzero(a[:s])
            if len(prior):
                total += min(s - prior[-1], 20)
                count += 1
    return total / count if count else np.nan


def score_metrics(f, va_idx, te_idx, vp, tp, seed):
    va = f.iloc[va_idx][["date", "asset", "y"]].copy()
    te = f.iloc[te_idx][["date", "asset", "y"]].copy()
    va["score"], te["score"] = vp, tp
    alarm = alarms_rq(te, va, .05)
    m = evaluate(te, alarm, sims=200, seed=seed)
    return {
        "n": len(te), "n_pos": int(te.y.sum()),
        "ap": average_precision_score(te.y, tp) if te.y.sum() else np.nan,
        "auc": roc_auc_score(te.y, tp) if te.y.nunique() > 1 else np.nan,
        "alarm_day_rate": m["alarm_day_rate"], "event_recall": m["event_recall"],
        "rand_event_recall": m["rand_event_recall"],
        "excess_event_recall": m["excess_event_recall"],
        "event_precision": m["event_precision"],
        "mean_lead_time": lead_time(te, alarm), "alarm": alarm, "test": te,
    }


def run_cell(f, cols, label, fold, feature_set, model, seed):
    tr, va, te = split(f, fold)
    if min(len(tr), len(va), len(te)) == 0:
        return None
    y = f[label].fillna(0).to_numpy(np.int8)
    f = f.copy()
    f["y"] = y
    z = scale_matrix(f, cols, tr)
    if model in {"logreg", "lightgbm"}:
        vp, tp = fit_tabular(z, y, tr, va, te, model, seed)
    else:
        sub = f.asset.isin(CORE)
        keep = np.flatnonzero(sub.to_numpy())
        pos = {old: i for i, old in enumerate(keep)}
        tr, va, te = [np.asarray([pos[i] for i in x if i in pos], int)
                      for x in (tr, va, te)]
        sf = f.iloc[keep].reset_index(drop=True)
        sy = y[keep]
        sf["y"] = sy
        z = scale_matrix(sf, cols, tr)
        arrays, assets, positions = sequence_arrays(sf, z)
        valid = positions >= 19
        tr = tr[valid[tr]]
        va = va[valid[va]]
        te = te[valid[te]]
        _, _, vp, tp, _ = fit_lazy_sequence(
            model, arrays, assets, positions, z, sy, tr, va, te,
            hidden=24, seed=seed, epochs=12, patience=4, batch_size=512)
        f, y = sf, sy
    mm = score_metrics(f, va, te, vp, tp, seed)
    score = mm.pop("test").copy()
    score["score"] = tp
    score["alarm"] = mm.pop("alarm")
    return mm, score


def bootstrap_significance(result):
    rows = []
    rng = np.random.default_rng(707)

    def events(path):
        d = pd.read_parquet(path)
        out = []
        for asset, g in d.sort_values("date").groupby("asset", sort=False):
            y = g.y.to_numpy(bool)
            a = g.alarm.to_numpy(bool)
            starts = np.flatnonzero(y & ~np.r_[False, y[:-1]])
            ends = np.flatnonzero(y & ~np.r_[y[1:], False])
            for s, e in zip(starts, ends):
                out.append((asset, int(s), int(e),
                            bool(a[s:e + 1].any())))
        return pd.DataFrame(out, columns=["asset", "start", "end", "caught"])

    def paired(path_a, path_b, comparison, label, fold, model):
        if not path_a.exists() or not path_b.exists():
            return None
        a, b = events(path_a), events(path_b)
        if a.empty or b.empty:
            return None
        a = a.rename(columns={"caught": "caught_a"})
        b = b.rename(columns={"caught": "caught_b"})
        j = a.merge(b, on=["asset", "start", "end"], how="inner")
        if j.empty:
            return None
        dif = j.caught_a.astype(float).to_numpy() - j.caught_b.astype(float).to_numpy()
        draws = rng.integers(0, len(dif), size=(1000, len(dif)))
        means = dif[draws].mean(axis=1)
        est = float(dif.mean())
        p = float(2 * min(np.mean(means <= 0), np.mean(means >= 0)))
        return {"comparison": comparison, "label": label, "fold": fold,
                "model": model, "events": len(dif), "estimate": est,
                "ci_low": float(np.quantile(means, .025)),
                "ci_high": float(np.quantile(means, .975)),
                "p_value": min(p, 1.0), "status": "paired episode bootstrap"}

    for label in result.label.unique():
        for fold in result.fold.unique():
            for model in ("lightgbm", "gru", "tcn"):
                a = result[(result.label == label) & (result.fold == fold) &
                           (result.model == model) & (result.feature_set == "GRAND_COMBINED")]
                b = result[(result.label == label) & (result.fold == fold) &
                           (result.model == model) & (result.feature_set == "PEER_CROSS")]
                if len(a) and len(b):
                    x = paired(SCORE_DIR / a.score_file.iloc[0],
                               SCORE_DIR / b.score_file.iloc[0],
                               "GRAND_COMBINED_vs_PEER_CROSS", label, fold, model)
                    if x:
                        rows.append(x)
            for deep in ("gru", "tcn"):
                a = result[(result.label == label) & (result.fold == fold) &
                           (result.model == deep) & (result.feature_set == "GRAND_COMBINED")]
                b = result[(result.label == label) & (result.fold == fold) &
                           (result.model == "logreg") & (result.feature_set == "GRAND_COMBINED")]
                if len(a) and len(b):
                    x = paired(SCORE_DIR / a.score_file.iloc[0],
                               SCORE_DIR / b.score_file.iloc[0],
                               f"{deep}_vs_logreg", label, fold, deep)
                    if x:
                        rows.append(x)
    return pd.DataFrame(rows)


def write_report(result, significance, sets):
    lines = ["# V7_FINAL_SYNTHESIS", "",
             "A causal head-to-head ablation of classical CSD, SEMIVAR momentum, "
             "peer/cross-index features, scale-normalized TDA, and their combination.",
             "",
             "## Design", "",
             "- Scheme W chronological folds: 2011-2015, 2016-2020, 2021+.",
             "- 25-session purge gap and causal rolling-250-session RQ threshold.",
             "- Alarm-day budget: 5%.",
             "- Horizons: N=5 and N=15, p=2% downside labels.",
             "- GRU and TCN use hidden size 24, sequence length 20, and seeds 42, 43, 44.",
             "",
             "V6 supplied 17 SEMIVAR columns (not 18). V5 supplied 36 sampled "
             "date-level landscape coordinates; these are merged causally by date.",
             "The four-index Group A restriction applies to sequence models.",
             "",
             "## Feature counts", "", "| Set | Features |", "|---|---:|"]
    lines += [f"| {k} | {len(v)} |" for k, v in sets.items()]
    if result.empty:
        lines += ["", "No completed cells were recorded."]
    else:
        agg = result.groupby(["label", "feature_set", "model"], as_index=False).agg(
            mean_ap=("ap", "mean"), mean_auc=("auc", "mean"),
            alarm_rate=("alarm_day_rate", "mean"),
            event_recall=("event_recall", "mean"),
            random_recall=("rand_event_recall", "mean"),
            excess_recall=("excess_event_recall", "mean"),
            mean_lead_time=("mean_lead_time", "mean"),
            folds=("fold", "count"))
        lines += ["", "## Ablation results", "",
                  "| Label | Feature set | Model | AP | AUC | Alarm rate | "
                  "Event recall | Random recall | Excess recall | Lead time | Folds |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in agg.iterrows():
            lines.append("| {label} | {feature_set} | {model} | {mean_ap:.4f} | "
                         "{mean_auc:.4f} | {alarm_rate:.4f} | {event_recall:.4f} | "
                         "{random_recall:.4f} | {excess_recall:.4f} | "
                         "{mean_lead_time:.2f} | {folds} |".format(**r))
    lines += ["", "## Paired episode bootstrap", ""]
    if significance.empty:
        lines.append("No paired bootstrap rows were available.")
    else:
        lines += ["| Comparison | Label | Fold | Events | Estimate | 95% CI | p-value |",
                  "|---|---|---|---:|---:|---|---:|"]
        for _, r in significance.iterrows():
            lines.append(f"| {r.comparison} | {r.label} | {r.fold} | {r.events} | "
                         f"{r.estimate:.4f} | [{r.ci_low:.4f}, {r.ci_high:.4f}] | "
                         f"{r.p_value:.4f} |")
    lines += ["", "## Interpretation", "",
              "Positive excess recall means the model caught more crash episodes "
              "than the same-budget random baseline. A bootstrap interval crossing "
              "zero is inconclusive; it is not evidence of no effect.",
              "",
              "Outputs: `data/v7_final/v7_ablation_results.csv`, "
              "`data/v7_final/v7_bootstrap_significance.csv`, and "
              "`data/v7_final/scores/`."]
    (ROOT / "V7_FINAL_SYNTHESIS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--timebox-min", type=float, default=0)
    args = ap.parse_args()
    if not args.run:
        raise SystemExit("Use --run to execute the complete V7 matrix.")
    OUT.mkdir(parents=True, exist_ok=True); SCORE_DIR.mkdir(exist_ok=True)
    f, sets = load_data()
    cells = [(label, fold, fs, model, seed)
             for label in ("y_N5_p2", "y_down_N15_p2")
             for fold in FOLDS for fs in sets
             for model in ("logreg", "lightgbm", "gru", "tcn")
             for seed in (SEEDS if model in {"gru", "tcn"} else (42,))]
    result_path = OUT / "v7_ablation_results.csv"
    done = set()
    if result_path.exists():
        old = pd.read_csv(result_path)
        done = set(zip(old.label, old.fold, old.feature_set, old.model, old.seed))
    start = time.perf_counter()
    for label, fold, fs, model, seed in cells:
        key = (label, fold[0], fs, model, seed)
        if key in done:
            continue
        if args.timebox_min and time.perf_counter() - start > args.timebox_min * 60:
            break
        if label not in f:
            continue
        out = run_cell(f, sets[fs], label, fold, fs, model, seed)
        if out is None:
            continue
        mm, scores = out
        stem = f"{label}_{fold[0]}_{fs}_{model}_{seed}"
        scores.to_parquet(SCORE_DIR / f"{stem}.parquet", index=False)
        row = {k: v for k, v in mm.items() if np.isscalar(v)}
        row.update({"label": label, "fold": fold[0], "feature_set": fs,
                    "model": model, "seed": seed, "feature_count": len(sets[fs]),
                    "panel": "GroupA" if model in {"gru", "tcn"} else "unified_159",
                    "score_file": f"{stem}.parquet"})
        pd.DataFrame([row]).to_csv(result_path, mode="a",
            header=not result_path.exists(), index=False)
        print("completed", key, flush=True)
        gc.collect()
    res = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
    sig = bootstrap_significance(res)
    sig.to_csv(OUT / "v7_bootstrap_significance.csv", index=False)
    write_report(res, sig, sets)
    (OUT / "v7_run_status.json").write_text(json.dumps({
        "completed_rows": len(res), "planned_cells": len(cells),
        "remaining_cells": len(cells) - len(set(zip(
            res.label, res.fold, res.feature_set, res.model, res.seed))),
        "feature_counts": {k: len(v) for k, v in sets.items()},
        "tda_source": "V5_landscape_sampled, first 36 date-level coordinates",
        "semivar_source": "V6 panel; 17 available columns",
        "rq": "rolling 250-session causal quantile, 5 percent",
        "deep_group": "four core US indices",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
