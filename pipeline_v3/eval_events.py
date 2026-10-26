"""Deterministic event-level crash-warning metrics for V3."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

MERGE_GAP = 5
ROWS_PER_YEAR = 252


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index ranges for maximal True runs."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return [(int(edges[i]), int(edges[i + 1] - 1))
            for i in range(0, len(edges), 2)]


def merge_runs(runs: list[tuple[int, int]], gap: int = MERGE_GAP
               ) -> list[tuple[int, int]]:
    """Join positive/alarm runs separated by at most ``gap`` rows."""
    if not runs:
        return []
    out = [list(runs[0])]
    for start, end in runs[1:]:
        if start - out[-1][1] - 1 <= gap:
            out[-1][1] = end
        else:
            out.append([start, end])
    return [(int(start), int(end)) for start, end in out]


def _event_metrics_per_asset(y: np.ndarray, alarm: np.ndarray,
                             merge_gap: int = MERGE_GAP
                             ) -> tuple[int, int, int, int]:
    """Return crashes, caught crashes, alarm events, correct alarm events."""
    y = np.asarray(y, dtype=bool)
    alarm = np.asarray(alarm, dtype=bool)
    y_runs = merge_runs(_runs(y), merge_gap)
    a_runs = merge_runs(_runs(alarm), merge_gap)
    event_id = np.full(y.size, -1, dtype=np.int32)
    for i, (start, end) in enumerate(y_runs):
        local = event_id[start:end + 1]
        local[y[start:end + 1]] = i
    # Count an event as caught only if an alarm was issued on a positive day.
    ids = event_id[np.flatnonzero(alarm)]
    caught = int(np.unique(ids[ids >= 0]).size)
    correct = sum(bool((y[start:end + 1] & alarm[start:end + 1]).any())
                  for start, end in a_runs)
    return len(y_runs), caught, len(a_runs), int(correct)


def _random_baseline_per_asset(y: np.ndarray, n_alarm_days: int, seed: int,
                               sims: int = 200,
                               merge_gap: int = MERGE_GAP,
                               eligible_indices: np.ndarray | None = None
                               ) -> tuple[float, float]:
    """Expected counts caught/correct under uniform distinct alarm days."""
    y = np.asarray(y, dtype=bool)
    n = y.size
    if n == 0 or n_alarm_days <= 0 or sims <= 0 or not y.any():
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    crash_id = np.full(n, -1, dtype=np.int32)
    for j, (start, end) in enumerate(merge_runs(_runs(y), merge_gap)):
        local = crash_id[start:end + 1]
        local[y[start:end + 1]] = j
    caught_counts = np.zeros(sims, dtype=float)
    correct_counts = np.zeros(sims, dtype=float)
    eligible = np.arange(n) if eligible_indices is None else np.asarray(eligible_indices, dtype=int)
    count = min(int(n_alarm_days), len(eligible))
    for sim in range(sims):
        idx = np.sort(rng.choice(eligible, size=count, replace=False))
        ids = crash_id[idx]
        caught_counts[sim] = np.unique(ids[ids >= 0]).size
        alarm_runs = merge_runs(_runs(np.isin(np.arange(n), idx)), merge_gap)
        correct_counts[sim] = sum(bool(y[idx[s:e + 1]].any())
                                  for s, e in alarm_runs)
    return float(caught_counts.mean()), float(correct_counts.mean())


def _stable_asset_seed(asset: object, seed: int) -> int:
    raw = f"{seed}:{asset}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=4).digest(), "little")


def event_metrics(frame: pd.DataFrame, score_col: str = "score",
                  tau: float | None = None, alarm_col: str | None = None,
                  sims: int = 200, seed: int = 0,
                  merge_gap: int = MERGE_GAP) -> dict[str, float]:
    """Evaluate daily alarms and merged events for a date/asset panel.

    Required columns are ``asset``, ``date`` and binary ``y``. Give either
    ``alarm_col`` or both ``score_col`` and ``tau``. Random-baseline fields
    are expected *counts* over the panel at the observed alarm-day budget.
    """
    required = {"asset", "date", "y"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if alarm_col is None and (tau is None or score_col not in frame.columns):
        raise ValueError("give alarm_col, or score_col and tau")
    f = frame.copy()
    if f[list(required)].isna().any().any():
        raise ValueError("asset, date, and y cannot contain missing values")
    if not f.y.isin([0, 1, False, True]).all():
        raise ValueError("y must be binary 0/1")
    f["date"] = pd.to_datetime(f.date)
    f = f.sort_values(["asset", "date"], kind="stable")
    if f.duplicated(["asset", "date"]).any():
        raise ValueError("asset/date rows must be unique")
    if alarm_col is None:
        f["_alarm"] = pd.to_numeric(f[score_col], errors="coerce") > tau
    else:
        if alarm_col not in f:
            raise ValueError(f"missing alarm column: {alarm_col}")
        f["_alarm"] = f[alarm_col].astype(bool)

    totals = np.zeros(5, dtype=np.int64)
    random_caught, random_correct = [], []
    day_tp = int((f.y.astype(bool) & f._alarm).sum())
    for asset, sub in f.groupby("asset", sort=False):
        observed_y = sub.y.to_numpy(dtype=bool)
        observed_alarm = sub._alarm.to_numpy(dtype=bool)
        dates = pd.DatetimeIndex(sub.date)
        full_dates = pd.bdate_range(dates.min(), dates.max())
        eligible = full_dates.get_indexer(dates)
        if len(full_dates) == len(dates) and np.array_equal(eligible, np.arange(len(dates))):
            y, alarm = observed_y, observed_alarm
            eligible = np.arange(len(y))
        else:
            y = np.zeros(len(full_dates), dtype=bool)
            alarm = np.zeros(len(full_dates), dtype=bool)
            y[eligible] = observed_y
            alarm[eligible] = observed_alarm
        values = _event_metrics_per_asset(y, alarm, merge_gap)
        crashes, caught, alarms, correct = values
        totals += (crashes, caught, alarms, correct, alarms - correct)
        rc, rk = _random_baseline_per_asset(
            y, int(observed_alarm.sum()), _stable_asset_seed(asset, seed), sims,
            merge_gap, eligible)
        random_caught.append(rc)
        random_correct.append(rk)

    crashes, caught, alarms, correct, wrong = map(int, totals)
    positives = int(f.y.sum())
    alarm_days = int(f._alarm.sum())
    asset_years = len(f) / ROWS_PER_YEAR
    return {
        "crashes_total": crashes,
        "crashes_caught": caught,
        "alarms_total": alarms,
        "alarms_correct": correct,
        "alarms_wrong": wrong,
        "event_recall": caught / crashes if crashes else np.nan,
        "event_precision": correct / alarms if alarms else np.nan,
        "wrong_per_asset_year": wrong / asset_years if asset_years else np.nan,
        "day_precision": day_tp / alarm_days if alarm_days else np.nan,
        "day_recall": day_tp / positives if positives else np.nan,
        "rand_caught": float(np.sum(random_caught)),
        "rand_correct": float(np.sum(random_correct)),
        "asset_years": asset_years,
    }
