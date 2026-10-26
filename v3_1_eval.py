"""V3.1 alarm-day-budget evaluation and rescoring helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

FOLDS = [(2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 9999)]


def alarm_days(frame: pd.DataFrame, rule: str, a: float,
               validation: pd.DataFrame | None = None, window: int = 250) -> np.ndarray:
    """Return per-row alarms; RQ uses asset history and RV a pooled val quantile."""
    f = frame.copy().reset_index(drop=True)
    f["date"] = pd.to_datetime(f.date)
    out = np.zeros(len(f), dtype=bool)
    if rule == "RV":
        if validation is None or validation.empty:
            return out
        tau = float(validation.score.quantile(1 - a))
        return (f.score.to_numpy(float) > tau) & np.isfinite(f.score.to_numpy(float))
    if rule != "RQ":
        raise ValueError("rule must be RQ or RV")
    # Each test asset starts with its last 250 validation scores, then rolls
    # forward using preceding scored test rows only; current score is excluded.
    for asset, idx in f.groupby("asset", sort=False).groups.items():
        loc = np.asarray(list(idx), dtype=int)
        test = f.iloc[loc].sort_values("date", kind="stable")
        hist = np.array([], dtype=float)
        if validation is not None and len(validation):
            v = validation.loc[validation.asset.eq(asset)].sort_values("date")
            hist = v.score.to_numpy(float)[-window:]
            hist = hist[np.isfinite(hist)]
        scores = test.score.to_numpy(float)
        alarm = np.zeros(len(test), dtype=bool)
        for j, score in enumerate(scores):
            prior = hist[-window:]
            if len(prior) >= 60 and np.isfinite(score):
                alarm[j] = score > np.quantile(prior, 1 - a)
            if np.isfinite(score):
                hist = np.r_[hist, score]
        out[test.index.to_numpy()] = alarm
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8)))
    return [(int(edges[i]), int(edges[i + 1] - 1)) for i in range(0, len(edges), 2)]


def _merge(runs: list[tuple[int, int]], gap: int = 5) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for s, e in runs:
        if out and s - out[-1][1] - 1 <= gap:
            out[-1][1] = e
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _capped_alarm_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out = []
    for s, e in _merge(_runs(mask), 5):
        while s <= e:
            end = min(e, s + 14)
            out.append((s, end))
            s = end + 1
    return out


def _counts_one(y: np.ndarray, alarm: np.ndarray) -> tuple[int, int, int, int, int]:
    crashes = _merge(_runs(y), 5)
    alarms = _capped_alarm_runs(alarm)
    # Merging defines one crash event, but the catch must occur on a y=1 day,
    # never merely in a short zero-label bridge between positive runs.
    caught = sum(bool((alarm[s:e + 1] & y[s:e + 1]).any()) for s, e in crashes)
    correct = sum(bool((y[s:e + 1] & alarm[s:e + 1]).any()) for s, e in alarms)
    return len(crashes), caught, len(alarms), correct, len(alarms) - correct


def _random_event_recall(group: pd.DataFrame, n_days: int, sims: int, rng: np.random.Generator) -> float:
    ys = group.y.to_numpy(bool)
    crashes = _merge(_runs(ys), 5)
    if not crashes or n_days <= 0:
        return 0.0
    caught = np.zeros(sims)
    for sim in range(sims):
        ix = rng.choice(len(ys), size=min(n_days, len(ys)), replace=False)
        a = np.zeros(len(ys), bool); a[ix] = True
        caught[sim] = sum(bool((a[s:e + 1] & ys[s:e + 1]).any()) for s, e in crashes)
    return float(caught.mean() / len(crashes))


def evaluate(frame: pd.DataFrame, alarms: np.ndarray, sims: int = 200,
             seed: int = 1701) -> dict:
    f = frame.copy().reset_index(drop=True)
    f["date"] = pd.to_datetime(f.date)
    f["alarm"] = np.asarray(alarms, bool)
    if f.duplicated(["asset", "date"]).any():
        raise ValueError("duplicate asset/date rows")
    crashes = caught = alarm_events = correct = wrong = 0
    rand_caught = 0.0
    rng = np.random.default_rng(seed)
    for _, g in f.groupby("asset", sort=False):
        g = g.sort_values("date")
        vals = _counts_one(g.y.to_numpy(bool), g.alarm.to_numpy(bool))
        c, cc, ac, co, wr = vals
        crashes += c; caught += cc; alarm_events += ac; correct += co; wrong += wr
        rand_caught += _random_event_recall(g, int(g.alarm.sum()), sims, rng) * c
    n = len(f); npos = int(f.y.sum()); n_alarm = int(f.alarm.sum())
    prevalence = npos / n if n else np.nan
    day_tp = int((f.y.astype(bool) & f.alarm).sum())
    precision = day_tp / n_alarm if n_alarm else np.nan
    try:
        ap = float(average_precision_score(f.y, f.score)) if npos else np.nan
        auc = float(roc_auc_score(f.y, f.score)) if f.y.nunique() == 2 else np.nan
    except ValueError:
        ap = auc = np.nan
    asset_years = f.groupby("asset").size().sum() / 252
    event_recall = caught / crashes if crashes else np.nan
    event_precision = correct / alarm_events if alarm_events else np.nan
    random_recall = rand_caught / crashes if crashes else np.nan
    rate = n_alarm / n if n else np.nan
    return {"n": n, "prevalence": prevalence, "n_pos": npos,
            "crashes_total": crashes, "crashes_caught": caught,
            "alarms_total": alarm_events, "alarms_correct": correct,
            "alarms_wrong": wrong, "realized_alarm_day_rate": rate,
            "event_recall": event_recall, "event_precision": event_precision,
            "wrong_alarms_per_asset_year": wrong / asset_years if asset_years else np.nan,
            "day_precision": precision, "day_recall": day_tp / npos if npos else np.nan,
            "day_lift": precision / prevalence if prevalence and np.isfinite(precision) else np.nan,
            "rand_event_recall": random_recall,
            "excess_event_recall": event_recall - random_recall if np.isfinite(event_recall) else np.nan,
            "ap": ap, "auc": auc}


def off_budget(rate: float, a: float) -> bool:
    return not np.isfinite(rate) or rate < .5 * a or rate > 2 * a

