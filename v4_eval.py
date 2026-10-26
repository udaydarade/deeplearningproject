"""V4 alarm-day budget evaluation utilities.

All thresholds are derived from preceding sessions only. This module is
independent of the V3 evaluation implementation so V4 rules can be audited
and unit-tested in isolation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def rolling_alarm(frame: pd.DataFrame, budget: float, window: int = 250) -> pd.DataFrame:
    out = frame.sort_values(["asset", "date"], kind="stable").copy()
    alarms = []
    thresholds = []
    for _, group in out.groupby("asset", sort=False):
        values = group["score"].to_numpy(float)
        alarm = np.zeros(len(group), dtype=bool)
        threshold = np.full(len(group), np.inf, dtype=float)
        for i in range(len(group)):
            past = values[max(0, i - window):i]
            past = past[np.isfinite(past)]
            if len(past):
                threshold[i] = np.quantile(past, 1.0 - budget)
            alarm[i] = np.isfinite(values[i]) and values[i] > threshold[i]
        alarms.extend(alarm)
        thresholds.extend(threshold)
    out["threshold"] = thresholds
    out["alarm"] = alarms
    return out


def _runs(mask: np.ndarray, gap: int = 0) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    runs = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value - previous > gap + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def _split_alarm_runs(runs: list[tuple[int, int]], size: int = 15) -> list[tuple[int, int]]:
    pieces = []
    for start, end in runs:
        while start <= end:
            stop = min(end, start + size - 1)
            pieces.append((start, stop))
            start = stop + 1
    return pieces


def evaluate(frame: pd.DataFrame, simulations: int = 200, seed: int = 17) -> dict:
    crash_total = crashes_caught = alarms_total = alarms_correct = 0
    by_asset = []
    rng = np.random.default_rng(seed)
    random_caught = []
    for asset, group in frame.sort_values("date").groupby("asset", sort=False):
        group = group.reset_index(drop=True)
        y = group["y"].to_numpy(bool)
        alarm = group["alarm"].to_numpy(bool)
        crashes = _runs(y, gap=5)
        alarm_runs = _split_alarm_runs(_runs(alarm, gap=5), size=15)
        crash_total += len(crashes)
        alarms_total += len(alarm_runs)
        crashes_caught += sum(any(alarm[start:end + 1]) for start, end in crashes)
        alarms_correct += sum(y[start:end + 1].any() for start, end in alarm_runs)
        by_asset.append((asset, len(group), int(alarm.sum()), len(crashes)))
        for _ in range(simulations):
            chosen = rng.choice(len(group), size=int(alarm.sum()), replace=False) if alarm.sum() else []
            random_caught.append(sum(any(start <= int(i) <= end for i in chosen)
                                     for start, end in crashes))
    alarms_wrong = alarms_total - alarms_correct
    n = len(frame)
    prevalence = float(frame["y"].mean()) if n else np.nan
    alarm_rate = float(frame["alarm"].mean()) if n else np.nan
    day_precision = float((frame["alarm"] & frame["y"].astype(bool)).sum() / max(int(frame["alarm"].sum()), 1))
    event_recall = crashes_caught / max(crash_total, 1)
    event_precision = alarms_correct / max(alarms_total, 1)
    random_recall = float(np.mean(random_caught) / max(crash_total, 1)) if random_caught else 0.0
    result = {
        "n": n, "prevalence": prevalence, "n_pos": int(frame["y"].sum()),
        "crashes_total": crash_total, "crashes_caught": crashes_caught,
        "alarms_total": alarms_total, "alarms_correct": alarms_correct,
        "alarms_wrong": alarms_wrong, "realized_alarm_day_rate": alarm_rate,
        "event_recall": event_recall, "event_precision": event_precision,
        "day_precision": day_precision, "day_lift": day_precision / prevalence if prevalence else np.nan,
        "rand_event_recall": random_recall,
        "excess_event_recall": event_recall - random_recall,
        "ap": float(average_precision_score(frame["y"], frame["score"])) if frame["y"].sum() else np.nan,
        "auc": float(roc_auc_score(frame["y"], frame["score"])) if frame["y"].nunique() > 1 else np.nan,
    }
    return result


def off_budget(realized_rate: float, budget: float) -> bool:
    return not (0.5 * budget <= realized_rate <= 2.0 * budget)

