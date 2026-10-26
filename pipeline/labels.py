"""Objective crash-label construction.

A crash origin is any date t where the *minimum* simple return available from
the close at t over the next ``HORIZON`` sessions falls at or below a fixed
threshold: the 2nd percentile of that same statistic (see Why, below). Labels
are per ticker - each market's episodes are detected on its own price path, so
a quiet index is never labelled by a loud neighbour.

Threshold estimation is deliberately separate from evaluation:

* Tick-anchored ("golden") rule - if the series extends before
  ``CALIBRATION_END`` (2006-12-31) with at least 1,000 fully-observed forecast
  origins, the threshold is the 2nd percentile of forward-min returns over
  origins through that date only. This matches ``crash_forecasting.py`` and
  keeps the 2011-2026 test folds untouched by threshold information.
* Rolling-anchored fallback - newer series (crypto, tech sectors) lack a long
  pre-2007 history, so the 2nd percentile is estimated on the first 40% of
  their observable origins. Testing folds remain strictly later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 15
LABEL_QUANTILE = 0.02
CALIBRATION_END = pd.Timestamp("2006-12-31")
# Minimum fully-observed origins before trusting the fixed 2006 anchor.
MIN_GOLDEN_ORIGINS = 1000


def forward_min_return(prices: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """Minimum simple return close(t) -> min close(t+1..t+horizon)."""
    values = prices.to_numpy(float)
    out = np.full(values.shape, np.nan)
    for i in range(len(values) - horizon):
        span = values[i + 1 : i + horizon + 1]
        out[i] = float(np.min((span - values[i]) / values[i]))
    return pd.Series(out, index=prices.index)


def calibration_threshold(
    prices: pd.Series, forward_min: pd.Series, horizon: int = HORIZON
) -> tuple[float, str]:
    """Threshold plus a human-readable rule name, using info-free estimation."""
    if len(prices.index) and prices.index[0] <= CALIBRATION_END:
        inside = int((prices.index <= CALIBRATION_END).sum())
        last_origin = inside - 1 - horizon  # outcome fully observed by 2006-12-31
        if last_origin >= MIN_GOLDEN_ORIGINS:
            fit = forward_min.iloc[: last_origin + 1].dropna()
            return float(fit.quantile(LABEL_QUANTILE)), f"golden_2006(origins={len(fit)})"
    observed = forward_min.dropna()
    keep = max(int(0.4 * len(observed)), horizon + 1)
    fit = observed.iloc[:keep]
    return float(fit.quantile(LABEL_QUANTILE)), f"first40pct(origins={len(fit)})"


def label_series(
    prices: pd.Series, horizon: int = HORIZON
) -> tuple[pd.DataFrame, dict]:
    """Crash labels for one price series.

    Returns (frame, info) where frame is indexed like ``prices`` with columns
    ``forward_min_return`` and ``label`` (NaN label where the forward window is
    not fully observable), and info documents the threshold chosen.
    """
    forward_min = forward_min_return(prices, horizon)
    threshold, rule = calibration_threshold(prices, forward_min, horizon)
    label = (forward_min <= threshold).astype(float)
    label[forward_min.isna()] = np.nan
    flags = label.fillna(0).astype(bool)
    info = {
        "threshold": round(threshold, 6),
        "rule": rule,
        "origins": int(forward_min.notna().sum()),
        "positive_days": int(label.sum()),
        "positive_share": round(float(label.mean()), 4),
        "episodes": int((flags & ~flags.shift(1, fill_value=False)).sum()),
    }
    return pd.DataFrame({"forward_min_return": forward_min, "label": label}), info
