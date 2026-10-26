"""Validation-only operating thresholds and past-only rolling alerts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline_v3.eval_events import event_metrics


def _thresholds(scores: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    clean = scores[np.isfinite(scores)]
    if clean.size == 0:
        return np.array([np.inf])
    vals = np.quantile(clean, quantiles)
    return np.unique(np.r_[vals, np.nextafter(clean.max(), np.inf)])


def choose_r1(validation: pd.DataFrame) -> tuple[float, float]:
    """Choose tau maximizing event F1 subject to 1%-15% validation alarms."""
    candidates = _thresholds(validation.score.to_numpy(float),
                             np.linspace(.85, .99, 29))
    best = (-1.0, float("-inf"), float("-inf"))
    for tau in candidates:
        pred = validation.score.to_numpy(float) > tau
        rate = float(pred.mean()) if len(pred) else 0.0
        if rate < .01 or rate > .15:
            continue
        metrics = event_metrics(validation, tau=float(tau), sims=0)
        precision = metrics["event_precision"]
        recall = metrics["event_recall"]
        f1 = (2 * precision * recall / (precision + recall)
              if np.isfinite(precision) and np.isfinite(recall) and precision + recall else 0.0)
        key = (f1, -rate, float(tau))
        if key > best:
            best = key
    if best[0] < 0:
        # No candidate satisfied the cap (often a constant/degenerate score).
        tau = float(np.quantile(validation.score.dropna(), .95))
        metrics = event_metrics(validation, tau=tau, sims=0)
        p, r = metrics["event_precision"], metrics["event_recall"]
        f1 = 2 * p * r / (p + r) if np.isfinite(p) and np.isfinite(r) and p + r else 0.0
        return tau, float(f1)
    return best[2], float(best[0])


def choose_r3(validation: pd.DataFrame, budget: float) -> tuple[float, float]:
    """Maximize validation event recall under a wrong-alert budget/year."""
    candidates = _thresholds(validation.score.to_numpy(float),
                             np.linspace(0.0, .999, 201))
    feasible = []
    for tau in candidates:
        metrics = event_metrics(validation, tau=float(tau), sims=0)
        rate = metrics["wrong_per_asset_year"]
        if np.isfinite(rate) and rate <= budget:
            recall = metrics["event_recall"]
            precision = metrics["event_precision"]
            feasible.append((float(recall) if np.isfinite(recall) else -1.0,
                             float(precision) if np.isfinite(precision) else -1.0,
                             -float(metrics["alarms_total"]), float(tau)))
    if not feasible:
        return float(np.nextafter(validation.score.max(), np.inf)), np.nan
    best = max(feasible)
    return best[3], best[0]


def rolling_r2(frame: pd.DataFrame, window: int = 250,
               quantile: float = .95) -> pd.DataFrame:
    """Apply score > pooled 95th percentile of previous sessions only.

    The input can contain validation and test rows from one fitted fold model.
    Threshold at date d uses all finite scores from the preceding ``window``
    distinct dates; current-date scores are never part of the threshold.
    """
    f = frame.sort_values(["date", "asset"], kind="stable").copy()
    dates = pd.Index(f.date.drop_duplicates())
    daily = {d: f.loc[f.date.eq(d), "score"].to_numpy(float) for d in dates}
    taus = {}
    for i, date in enumerate(dates):
        past_dates = dates[max(0, i - window):i]
        arrays = [daily[d][np.isfinite(daily[d])] for d in past_dates]
        arrays = [a for a in arrays if a.size]
        taus[date] = float(np.quantile(np.concatenate(arrays), quantile)) if arrays else np.inf
    f["threshold"] = f.date.map(taus)
    f["alarm"] = f.score > f.threshold
    return f


def validation_rule_table(validation: pd.DataFrame) -> dict[str, tuple[float, float]]:
    # Score quantiles give a compact validation-only search grid. Evaluate
    # event metrics once per threshold, then reuse them for R1 and all R3 caps.
    qs = np.unique(np.r_[np.linspace(.0, .999, 51), np.linspace(.85, .99, 15)])
    candidates = _thresholds(validation.score.to_numpy(float), qs)
    rows = []
    for tau in candidates:
        alarm_rate = float((validation.score.to_numpy(float) > tau).mean())
        metrics = event_metrics(validation, tau=float(tau), sims=0)
        p, r = metrics["event_precision"], metrics["event_recall"]
        f1 = 2 * p * r / (p + r) if np.isfinite(p) and np.isfinite(r) and p + r else 0.0
        rows.append({"tau": float(tau), "alarm_rate": alarm_rate,
                     "event_f1": float(f1), "event_recall": metrics["event_recall"],
                     "event_precision": metrics["event_precision"],
                     "wrong_per_asset_year": metrics["wrong_per_asset_year"],
                     "alarms_total": metrics["alarms_total"]})
    r1 = [r for r in rows if .01 <= r["alarm_rate"] <= .15]
    if r1:
        best = max(r1, key=lambda x: (x["event_f1"], -x["alarm_rate"], x["tau"]))
        out = {"R1": (best["tau"], best["event_f1"])}
    else:
        fallback = min(rows, key=lambda x: abs(x["alarm_rate"] - .05))
        out = {"R1": (fallback["tau"], fallback["event_f1"])}
    for budget in (.5, 1.0, 2.0):
        feasible = [r for r in rows if np.isfinite(r["wrong_per_asset_year"])
                    and r["wrong_per_asset_year"] <= budget]
        if not feasible:
            out[f"R3_b{budget:g}"] = (float(np.nextafter(validation.score.max(), np.inf)), 0.0)
        else:
            best = max(feasible, key=lambda x: (
                -1 if not np.isfinite(x["event_recall"]) else x["event_recall"],
                -1 if not np.isfinite(x["event_precision"]) else x["event_precision"],
                -x["alarms_total"], x["tau"]))
            out[f"R3_b{budget:g}"] = (best["tau"], best["event_recall"])
    return out
