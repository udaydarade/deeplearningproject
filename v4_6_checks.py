"""V4 point-in-time and scale checks for the rebuilt raw topology block."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from v4_3_build_raw import date_features

ROOT = Path(__file__).resolve().parent
V3 = ROOT / "data" / "v3"
V4 = ROOT / "data" / "v4"


def main() -> None:
    returns = pd.read_parquet(V3 / "returns.parquet").astype(float)
    registry = pd.read_csv(V3 / "registry.csv")
    classes = registry.set_index("ticker").asset_class.to_dict()
    nonstock = [c for c in returns if classes.get(c) != "stock"]
    eligible_dates = returns.index[returns.index >= pd.Timestamp("2005-01-01")]
    dates = eligible_dates[np.linspace(0, len(eligible_dates) - 1, min(10, len(eligible_dates)), dtype=int)]
    leak_rows = []
    for date in dates:
        full = date_features(date, returns, nonstock, 60, 24, 2, ("A_corr", "A_raw", "T_raw"))
        truncated = date_features(date, returns.loc[:date], nonstock, 60, 24, 2, ("A_corr", "A_raw", "T_raw"))
        for column in full:
            if column == "date":
                continue
            a, b = full.get(column, np.nan), truncated.get(column, np.nan)
            diff = 0.0 if pd.isna(a) and pd.isna(b) else abs(float(a) - float(b))
            leak_rows.append({"date": date, "feature": column, "abs_diff": diff})
    leak = pd.DataFrame(leak_rows)
    leak.to_csv(V4 / "leak_test.csv", index=False)
    print(f"[leak] dates={len(dates)} comparisons={len(leak):,} max_diff={leak.abs_diff.max():.3g}")

    date = dates[-1]
    for candidate in reversed(dates):
        candidate_base = returns.loc[:candidate, nonstock].tail(60).dropna(axis=1)
        if candidate_base.shape[1] >= 24:
            date, base = candidate, candidate_base
            break
    from v4_3_build_raw import one
    checks = []
    for kind in ("A_corr", "A_raw", "T_raw"):
        a = one(base.to_numpy(), kind, "x")
        b = one((base * 3).to_numpy(), kind, "x")
        for field in ("h1_total", "h1_max", "h1_l1", "h1_l2"):
            key = f"x_{field}"
            if key not in a or not np.isfinite(a[key]) or a[key] == 0:
                continue
            expected = {"A_corr": 1.0, "A_raw": {"h1_total": 3.0, "h1_max": 3.0,
                       "h1_l1": 9.0, "h1_l2": 3.0 ** 1.5},
                       "T_raw": {"h1_total": 3.0, "h1_max": 3.0,
                       "h1_l1": 9.0, "h1_l2": 3.0 ** 1.5}}[kind]
            ratio = b[key] / a[key]
            target = expected if isinstance(expected, float) else expected[field]
            checks.append({"kind": kind, "field": field, "ratio": ratio, "expected": target,
                           "error": abs(ratio - target)})
    scale = pd.DataFrame(checks)
    scale.to_csv(V4 / "scale_test.csv", index=False)
    print(f"[scale] max_error={scale.error.max() if len(scale) else np.nan:.3g}")


if __name__ == "__main__":
    main()
