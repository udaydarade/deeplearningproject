"""V3 P1 walk-forward experiments and validation-only alarm rules.

Writes model/results only under data/v3/. No V1/V2 output is read as a target
or overwritten. Run with ``py -3.12 v3_run_experiments.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from pipeline.models import GRUModel, TCNModel
from pipeline_v3.eval_events import event_metrics
from pipeline_v3.operating_rules import rolling_r2, validation_rule_table
from v3_download_data import OUT

torch.set_num_threads(1)
RESULT_PATH = OUT / "exp_a_results.csv"
FOLDS = [(2000, 2004), (2005, 2009), (2010, 2014),
         (2015, 2019), (2020, 9999)]
US4 = {"^GSPC", "^DJI", "^IXIC", "^RUT"}
ORIGINAL_DATASETS = ["group_a", "group_b", "crypto", "international",
                     "financials", "energy", "tech"]


def splits(frame: pd.DataFrame, first_year: int, last_year: int) -> tuple[pd.Timestamp, ...]:
    max_date = pd.Timestamp(frame.date.max())
    test_start = pd.Timestamp(f"{first_year}-01-01")
    test_end = max_date if last_year == 9999 else min(max_date, pd.Timestamp(f"{last_year}-12-31"))
    val_end = test_start - pd.offsets.BDay(25)
    val_start = test_start - pd.offsets.BDay(2 * 252 + 25)
    train_end = val_start - pd.offsets.BDay(25)
    return test_start, test_end, val_start, val_end, train_end


def _nan_fill_scale(x_train: np.ndarray, x_other: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(x_train, axis=0)
    med[~np.isfinite(med)] = 0.0
    a = np.where(np.isfinite(x_train), x_train, med)
    b = np.where(np.isfinite(x_other), x_other, med)
    scaler = StandardScaler().fit(a)
    return scaler.transform(a).astype("float32"), scaler.transform(b).astype("float32"), med, scaler


def _event_f1(m: dict) -> float:
    p, r = m["event_precision"], m["event_recall"]
    return 2 * p * r / (p + r) if np.isfinite(p) and np.isfinite(r) and p + r else 0.0


def _score_metrics(test: pd.DataFrame, alarm: np.ndarray, rule: str,
                   budget: float | None, fold: str, feature_set: str,
                   model: str, val_f1: float | None, seed: int = 1701) -> list[dict]:
    test = test.copy()
    if "y_BASE" in test.columns and "y" not in test.columns:
        test.rename(columns={"y_BASE": "y"}, inplace=True)
    test["alarm"] = np.asarray(alarm, dtype=bool)
    test["scheme"] = "W"
    test["label_N"], test["label_p"] = 15, .02
    subsets: list[tuple[str, pd.Series]] = [("all", pd.Series(True, index=test.index))]
    stock = test.asset_class.eq("stock")
    subsets += [("stock", stock), ("non_stock", ~stock), ("us4", test.asset.isin(US4))]
    for name in ORIGINAL_DATASETS:
        subsets.append((f"dataset:{name}", test.source_dataset.eq(name)))
    for name in sorted(test.asset_class.dropna().unique()):
        subsets.append((f"class:{name}", test.asset_class.eq(name)))
    out = []
    for subset, mask in subsets:
        f = test.loc[mask]
        if f.empty:
            continue
        y = f.y.astype(int).to_numpy()
        pred = f.alarm.to_numpy(bool)
        ev = event_metrics(f[["asset", "date", "y", "score", "alarm"]],
                           alarm_col="alarm", sims=200, seed=seed)
        ap = float(average_precision_score(y, f.score)) if y.sum() else np.nan
        try:
            auc = float(roc_auc_score(y, f.score)) if len(np.unique(y)) == 2 else np.nan
        except ValueError:
            auc = np.nan
        row = {"exp": "EXP-V3-A", "scheme": "W", "label_N": 15, "label_p": .02,
               "feature_set": feature_set, "model": model, "rule": rule,
               "budget": budget, "fold": fold, "seed": seed, "subset": subset,
               "n": len(f), "prevalence": float(y.mean()), "n_pos": int(y.sum()),
               "alarm_rate": float(pred.mean()),
               "precision": float(((y == 1) & pred).sum() / pred.sum()) if pred.sum() else np.nan,
               "recall": float(((y == 1) & pred).sum() / y.sum()) if y.sum() else np.nan,
               "ap": ap, "auc": auc, "val_event_f1": val_f1}
        row.update(ev)
        out.append(row)
    return out


def _tabular_view(frame: pd.DataFrame, features: list[str]
                  ) -> tuple[np.ndarray, list[str]]:
    # Per-asset past-only tabular expansion: current value, five-day change,
    # trailing 20-session mean and standard deviation.
    chunks, names = [], ([f"value::{c}" for c in features] +
                         [f"change5::{c}" for c in features] +
                         [f"mean20::{c}" for c in features] +
                         [f"std20::{c}" for c in features])
    for _, g in frame.groupby("asset", sort=False):
        x = g[features].astype("float32")
        value = x.to_numpy(copy=True)
        change = x - x.shift(5)
        mean = x.rolling(20, min_periods=1).mean()
        std = x.rolling(20, min_periods=2).std()
        chunks.append(np.concatenate([value, change.to_numpy(dtype="float32"),
                                      mean.to_numpy(dtype="float32"),
                                      std.to_numpy(dtype="float32")], axis=1))
    return np.concatenate(chunks, axis=0), names


def _fit_gbm(x: np.ndarray, y: np.ndarray, train: np.ndarray,
             val: np.ndarray, names: list[str], seed: int):
    npos = max(int(y[train].sum()), 1)
    nneg = max(int(len(train) - npos), 1)
    model = lgb.LGBMClassifier(
        objective="binary", learning_rate=.03, num_leaves=15,
        min_child_samples=500, feature_fraction=.5, bagging_fraction=.7,
        bagging_freq=1, reg_lambda=5.0,
        scale_pos_weight=min(nneg / npos, 10), n_estimators=600,
        max_bin=63, n_jobs=8, random_state=seed, verbosity=-1)
    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    model.fit(x[train], y[train], eval_set=[(x[val], y[val])],
              eval_metric="average_precision", callbacks=callbacks)
    val_pred = model.predict_proba(x[val])[:, 1]
    return model, val_pred, model.predict_proba(x)[:, 1], model.booster_.feature_importance(importance_type="gain")


class _SequenceRows(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], records: list[tuple[str, int, int]]):
        self.arrays, self.records = arrays, records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        asset, end, label = self.records[i][:3]
        return torch.from_numpy(self.arrays[asset][end - 19:end + 1]), torch.tensor(label, dtype=torch.float32)


def _sequence_bank(frame: pd.DataFrame, features: list[str], train_end: pd.Timestamp,
                   val_start: pd.Timestamp, val_end: pd.Timestamp,
                   test_start: pd.Timestamp, test_end: pd.Timestamp,
                   seed: int) -> tuple[dict, list, list, list, StandardScaler]:
    train_rows = frame.loc[(frame.date <= train_end) & frame.y_BASE.notna()
                           & frame[features].notna().all(axis=1)]
    if train_rows.empty:
        raise ValueError("no complete training rows for selected deep features")
    scaler = StandardScaler().fit(train_rows[features].to_numpy(dtype=np.float32))
    arrays, records = {}, []
    train, val, test, history = [], [], [], []
    for asset, g in frame.groupby("asset", sort=False):
        g = g.sort_values("date", kind="stable")
        x = g[features].to_numpy(dtype=np.float32)
        finite = np.isfinite(x).all(axis=1)
        x = np.where(np.isfinite(x), x, scaler.mean_)
        x = scaler.transform(x).astype("float32")
        arrays[asset] = x
        dates = g.date.to_numpy()
        labels = g.y_BASE.to_numpy()
        for i in range(19, len(g)):
            if not finite[i - 19:i + 1].all():
                continue
            # Do not build a sequence across a prolonged missing-data break.
            if (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i - 19])).days > 35:
                continue
            date = pd.Timestamp(dates[i])
            j = len(records)
            label = -1 if pd.isna(labels[i]) else int(labels[i])
            records.append((asset, i, label, date))
            if val_start <= date <= test_end:
                history.append(j)
            if label < 0:
                continue
            if date <= train_end:
                train.append(j)
            elif val_start <= date <= val_end:
                val.append(j)
            elif test_start <= date <= test_end:
                test.append(j)
    return arrays, records, train, val, test, history, scaler


def _predict_sequence(model: nn.Module, dataset: _SequenceRows, indices: list[int]) -> np.ndarray:
    model.eval()
    loader = DataLoader(Subset(dataset, indices), batch_size=1024, shuffle=False)
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.extend(torch.sigmoid(model(x)).numpy().tolist())
    return np.asarray(out, dtype=np.float32)


def _train_deep(frame: pd.DataFrame, features: list[str], train_end, val_start,
                val_end, test_start, test_end, arch: str, seed: int):
    arrays, records, tr, va, te, history, scaler = _sequence_bank(
        frame, features, train_end, val_start, val_end, test_start, test_end, seed)
    dataset = _SequenceRows(arrays, records)
    y = np.asarray([r[2] for r in records], dtype=np.float32)
    if not tr or not va:
        raise ValueError(f"not enough sequence rows: train={len(tr)}, val={len(va)}")
    best = None
    for hidden in (24, 64):
        torch.manual_seed(seed + hidden)
        model = GRUModel(len(features), hidden) if arch == "gru" else TCNModel(len(features), hidden)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
        best_ap, best_state, stale = -1.0, None, 0
        rng = np.random.default_rng(seed + hidden)
        pos = np.asarray([j for j in tr if y[j] == 1], dtype=int)
        neg = np.asarray([j for j in tr if y[j] == 0], dtype=int)
        # Build minibatches from all positives plus a fresh 20% negative sample each epoch.
        for epoch in range(12):
            take = rng.choice(neg, size=max(1, int(.2 * len(neg))), replace=False) if len(neg) else neg
            use = np.r_[pos, take]
            rng.shuffle(use)
            subset_loader = DataLoader(Subset(dataset, use.tolist()), batch_size=512, shuffle=True)
            ratio = min(max(len(take) / max(len(pos), 1), 1.0), 20.0)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio))
            model.train()
            for xb, yb in subset_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            vp = _predict_sequence(model, dataset, va)
            ap = average_precision_score(y[va], vp) if y[va].sum() else 0.0
            if ap > best_ap + 1e-5:
                best_ap, stale = ap, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= 4:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        val_p = _predict_sequence(model, dataset, va)
        if best is None or average_precision_score(y[va], val_p) > best[0]:
            test_p = _predict_sequence(model, dataset, te) if te else np.empty(0)
            history_p = _predict_sequence(model, dataset, history) if history else np.empty(0)
            best = (average_precision_score(y[va], val_p), model, hidden, val_p, test_p,
                    np.asarray(va), np.asarray(te), np.asarray(history), history_p,
                    arrays, records, scaler)
    return best


def _subsets(frame: pd.DataFrame):
    yield "all", np.ones(len(frame), dtype=bool)
    yield "non_stock", frame.asset_class.to_numpy() != "stock"
    yield "stock", frame.asset_class.to_numpy() == "stock"
    yield "us4", frame.asset.isin(US4).to_numpy()
    for d in ORIGINAL_DATASETS:
        yield f"dataset:{d}", frame.source_dataset.eq(d).to_numpy()
    for d in sorted(frame.asset_class.dropna().unique()):
        yield f"class:{d}", frame.asset_class.eq(d).to_numpy()


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-sets", default="F0,F1,F2,F3,F4")
    parser.add_argument("--models", default="logreg,gbm,gru,tcn")
    parser.add_argument("--folds", default="all")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    panel = pd.read_parquet(OUT / "panel.parquet")
    shared = pd.read_parquet(OUT / "market_features.parquet")
    sets = json.loads((OUT / "feature_sets.json").read_text(encoding="utf-8"))
    requested_folds = set(range(5)) if args.folds == "all" else {int(x) for x in args.folds.split(",")}
    requested_models = args.models.split(",")
    completed = set()
    if args.resume and RESULT_PATH.exists():
        old = pd.read_csv(RESULT_PATH)
        completed = set(zip(old.feature_set, old.model, old.fold))
    for fs in args.feature_sets.split(","):
        if fs not in sets:
            raise ValueError(f"unknown feature set {fs}")
        features = sets[fs]
        local = [c for c in features if c in panel.columns]
        external = [c for c in features if c in shared.columns]
        if len(local) + len(external) != len(set(features)):
            missing = sorted(set(features) - set(local) - set(external))
            raise ValueError(f"feature manifest columns unavailable: {missing[:10]}")
        metadata = ["asset", "date", "source_dataset", "asset_class", "y_BASE"]
        data = panel[metadata + local].merge(shared[external].reset_index(), on="date", how="left", validate="many_to_one")
        data = data.sort_values(["asset", "date"], kind="stable").reset_index(drop=True)
        if data.asset_class.isna().any():
            raise ValueError("asset_class missing; registry join must be fixed")
        for fold_i, (first, last) in enumerate(FOLDS):
            if fold_i not in requested_folds:
                continue
            test_start, test_end, val_start, val_end, train_end = splits(data, first, last)
            fold = f"{first}-{last if last != 9999 else 'latest'}"
            todo_models = [m for m in requested_models if (fs, m, fold) not in completed]
            if not todo_models:
                print(f"[resume] skip {fs} {fold}: all requested models recorded")
                continue
            train_mask = data.date.le(train_end) & data.y_BASE.notna()
            val_mask = data.date.between(val_start, val_end) & data.y_BASE.notna()
            test_mask = data.date.between(test_start, test_end) & data.y_BASE.notna()
            hist_mask = data.date.between(val_start, test_end)
            train_ix = np.flatnonzero(train_mask.to_numpy())
            val_ix = np.flatnonzero(val_mask.to_numpy())
            test_ix = np.flatnonzero(test_mask.to_numpy())
            if not len(train_ix) or not len(val_ix) or not len(test_ix):
                print(f"[skip] {fs} {fold}: n train/val/test={len(train_ix)}/{len(val_ix)}/{len(test_ix)}")
                continue
            y = data.y_BASE.fillna(0).to_numpy(dtype=np.int8)
            importance = None
            gbm_val = gbm_all = None
            t0 = time.perf_counter()
            print(f"[start] {fs} {fold}: n={len(data)} features={len(features)} train/val/test={len(train_ix)}/{len(val_ix)}/{len(test_ix)} pos={int(y[train_ix].sum())}/{int(y[val_ix].sum())}/{int(y[test_ix].sum())}", flush=True)
            if any(m in todo_models for m in ("gbm", "gru", "tcn")):
                tab, tab_names = _tabular_view(data, features)
                gbm, val_pred, all_pred, gain = _fit_gbm(tab, y, train_ix, val_ix, tab_names, args.seed + fold_i)
                importance = {}
                for name, value in zip(tab_names, gain):
                    base = name.split("::", 1)[1]
                    importance[base] = importance.get(base, 0.0) + float(value)
                gbm_val, gbm_all = val_pred, all_pred
                del tab
                if "gbm" in todo_models:
                    data["score"] = gbm_all
                    val = data.loc[val_mask, ["asset", "date", "y_BASE", "score", "asset_class", "source_dataset"]].rename(columns={"y_BASE": "y"})
                    test = data.loc[test_mask, ["asset", "date", "y_BASE", "score", "asset_class", "source_dataset"]].rename(columns={"y_BASE": "y"})
                    rows = _evaluate_model(data, val, test, fs, "gbm", fold, None, args.seed)
                    _append_results(rows)
                    completed.add((fs, "gbm", fold))
            if "logreg" in todo_models:
                x = data[features].to_numpy(dtype=np.float32)
                xt, xo, med, scaler = _nan_fill_scale(x[train_ix], x)
                model = LogisticRegression(class_weight="balanced", max_iter=400,
                                           n_jobs=8, solver="lbfgs", random_state=args.seed + fold_i)
                model.fit(xt, y[train_ix])
                score = model.predict_proba(xo)[:, 1].astype(np.float32)
                data["score"] = score
                val = data.loc[val_mask, ["asset", "date", "y_BASE", "score", "asset_class", "source_dataset"]].rename(columns={"y_BASE": "y"})
                test = data.loc[test_mask, ["asset", "date", "y_BASE", "score", "asset_class", "source_dataset"]].rename(columns={"y_BASE": "y"})
                rows = _evaluate_model(data, val, test, fs, "logreg", fold, None, args.seed)
                _append_results(rows)
                completed.add((fs, "logreg", fold))
            for arch in ("gru", "tcn"):
                if arch not in todo_models:
                    continue
                if not importance:
                    raise ValueError("deep models require the fold GBM gain ranking")
                top = sorted(importance, key=importance.get, reverse=True)[:80]
                deep = _train_deep(data[metadata + top], top, train_end, val_start, val_end,
                                   test_start, test_end, arch, args.seed + fold_i)
                (_, model, hidden, valp, testp, validx, testidx, historyix,
                 historyp, arrays, recs, _) = deep
                # Align prediction vectors to selected complete-sequence rows.
                seqdata = data[metadata + top].copy()
                # Returned indices reference the filtered per-asset sequence records;
                # reconstruct the matching validation/test frame from the helper.
                valrecords, testrecords = [], []
                for idx in validx:
                    asset, end, label, date = recs[int(idx)]
                    valrecords.append((asset, date, label))
                for idx in testidx:
                    asset, end, label, date = recs[int(idx)]
                    testrecords.append((asset, date, label))
                val_frame = pd.DataFrame(valrecords, columns=["asset", "date", "y"])
                val_frame["score"] = valp
                test_frame = pd.DataFrame(testrecords, columns=["asset", "date", "y"])
                test_frame["score"] = testp
                history_records = [(recs[int(idx)][0], recs[int(idx)][3])
                                   for idx in historyix]
                history_frame = pd.DataFrame(history_records, columns=["asset", "date"])
                history_frame["score"] = historyp
                meta_lookup = data.drop_duplicates("asset").set_index("asset")[["asset_class", "source_dataset"]]
                val_frame = val_frame.join(meta_lookup, on="asset")
                test_frame = test_frame.join(meta_lookup, on="asset")
                rows = _evaluate_model(data, val_frame, test_frame, fs, arch, fold, hidden, args.seed,
                                       history=history_frame)
                _append_results(rows)
                completed.add((fs, arch, fold))
            print(f"[done] {fs} {fold}: models={todo_models} elapsed={time.perf_counter()-t0:.1f}s", flush=True)


def _evaluate_model(data: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                    fs: str, model: str, fold: str, hidden: int | None,
                    seed: int, history: pd.DataFrame | None = None) -> list[dict]:
    val = val.rename(columns={"y_BASE": "y"}).copy()
    test = test.rename(columns={"y_BASE": "y"}).copy()
    rules = validation_rule_table(val[["asset", "date", "y", "score"]])
    out = []
    def emit(rule, alarms, budget, tau=None, val_f1=None):
        metrics_rows = _score_metrics(test[["asset", "date", "y", "score", "asset_class", "source_dataset"]],
                                      alarms, rule, budget, fold, fs, model, val_f1,
                                      seed + len(out))
        for row in metrics_rows:
            row["hidden"] = hidden
            row["threshold"] = tau
        out.extend(metrics_rows)
    tau, val_f1 = rules["R1"]
    emit("R1", test.score.to_numpy() > tau, None, tau, val_f1)
    # Past-only threshold uses validation, purge-gap, and prior test scores as
    # history, while metrics are evaluated only on labeled test rows.
    if history is None:
        history = data.loc[data.date.between(val.date.min(), test.date.max()),
                           ["asset", "date", "score"]].copy()
    else:
        history = history[["asset", "date", "score"]].copy()
    r2 = rolling_r2(history)
    r2_test = r2.loc[r2.date.isin(test.date)].sort_values(["asset", "date"])
    test_sorted = test.sort_values(["asset", "date"])
    alarm_map = r2_test.set_index(["asset", "date"]).alarm
    alarms = pd.MultiIndex.from_frame(test_sorted[["asset", "date"]]).map(alarm_map).fillna(False).to_numpy(bool)
    # _score_metrics expects the same row order as the sorted test frame.
    original_order = test.sort_values(["asset", "date"], kind="stable").index
    sorted_test = test.loc[original_order]
    out.extend(_score_metrics(sorted_test, alarms, "R2", None, fold, fs, model, None, seed + 99))
    for budget in (.5, 1.0, 2.0):
        tau, _ = rules[f"R3_b{budget:g}"]
        vm = event_metrics(val[["asset", "date", "y", "score"]], tau=tau, sims=0)
        emit("R3", test.score.to_numpy() > tau, budget, tau, _event_f1(vm))
    return out


def _append_results(rows: list[dict]) -> None:
    if not rows:
        return
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULT_PATH, mode="a", header=not RESULT_PATH.exists(), index=False)


if __name__ == "__main__":
    run()
