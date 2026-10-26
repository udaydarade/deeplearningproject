"""Audit the frozen V3 TDA feature columns without rebuilding V3 data."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V3 = ROOT / "data" / "v3"
OUT = ROOT / "data" / "v4"


def parse_column(column: str) -> dict:
    match = re.match(r"tda_([AT])_k(\d+|full)_w(\d+)_h(\d+)_(.+)", column)
    if not match:
        return {}
    cloud, k, window, homology, summary = match.groups()
    return {
        "column": column, "cloud": "T_day" if cloud == "T" else "A_asset",
        "standardization": "row_zscore; T additionally PCA+median-distance normalization"
        if cloud == "T" else "row_zscore then correlation distance",
        "k": k, "window": int(window), "homology": int(homology),
        "summary": summary, "stride": 10, "forward_fill": True,
    }


def market_crash_runs(panel: pd.DataFrame, folds: list[tuple[int, int]]) -> pd.DataFrame:
    rows = []
    for fold, (start, end) in enumerate(folds):
        subset = panel[(panel.date >= pd.Timestamp(f"{start}-01-01")) &
                       (panel.date <= (panel.date.max() if end == 9999 else pd.Timestamp(f"{end}-12-31")))]
        for date, group in subset.groupby("date"):
            positive_share = group["y_BASE"].mean()
            rows.append({"fold": fold, "date": date, "positive_share": positive_share,
                         "market_crash": positive_share >= 0.20})
    daily = pd.DataFrame(rows)
    result = []
    for fold, group in daily.groupby("fold"):
        result.append({"fold": fold, "market_crash_runs": len(_runs(group.market_crash.to_numpy(), 0)),
                       "market_crash_days": int(group.market_crash.sum())})
    return pd.DataFrame(result)


def _runs(mask: np.ndarray, gap: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    out, start, previous = [], int(idx[0]), int(idx[0])
    for value in idx[1:]:
        value = int(value)
        if value - previous > gap + 1:
            out.append((start, previous))
            start = value
        previous = value
    out.append((start, previous))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(V3 / "market_features.parquet")
    tda = [c for c in features.columns if c.startswith("tda_")]
    rows = []
    for column in tda:
        item = parse_column(column)
        values = features[column]
        repeated = values.notna() & values.eq(values.shift(1))
        item.update(
            share_nan=float(values.isna().mean()),
            share_zero=float((values.fillna(np.nan) == 0).mean()),
            share_forward_filled=float(repeated.mean()),
            mean_value=float(values.mean()) if values.notna().any() else np.nan,
        )
        rows.append(item)
    audit = pd.DataFrame(rows)
    panel = pd.read_parquet(V3 / "panel.parquet", columns=["asset", "date", "y_BASE"])
    panel.date = pd.to_datetime(panel.date)
    folds = [(2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 9999)]
    market = market_crash_runs(panel, folds)
    distinct_dates = features.index if isinstance(features.index, pd.DatetimeIndex) else len(features)
    groups = []
    for (cloud, k, window, homology), group in audit.groupby(
            ["cloud", "k", "window", "homology"], dropna=False):
        prefix = [c for c in tda if parse_column(c).get("cloud") == cloud
                  and parse_column(c).get("k") == k
                  and parse_column(c).get("window") == window
                  and parse_column(c).get("homology") == homology]
        count_columns = [c for c in prefix if c.endswith("_bar_count")]
        if count_columns:
            counts = features[count_columns].mean(axis=1)
            groups.append({"column": "__GROUP__", "cloud": cloud, "k": k,
                           "window": window, "homology": homology,
                           "summary": "finite_bar_summary",
                           "share_nan": float(features[prefix].isna().mean().mean()),
                           "share_zero": float((features[prefix] == 0).mean().mean()),
                           "share_forward_filled": np.nan,
                           "mean_value": float(counts.mean()),
                           "share_ge1_bar": float((counts >= 1).mean())})
    summary = pd.DataFrame(groups + [
        {"column": "__SUMMARY__", "cloud": "market-wide", "standardization": "see V3 code",
         "k": "distinct_dates", "window": int(len(features)),
         "homology": int(len(tda)), "summary": "asset_date_rows",
         "stride": 10, "forward_fill": True,
         "share_nan": float(features[tda].isna().mean().mean()),
         "share_zero": float((features[tda] == 0).mean().mean()),
         "share_forward_filled": np.nan, "mean_value": float(len(panel) / max(len(features), 1))},
    ])
    pd.concat([audit, summary], ignore_index=True).to_csv(OUT / "audit_old_tda.csv", index=False)
    market.to_csv(OUT / "audit_market_crash_runs.csv", index=False)
    manifest = json.loads((V3 / "feature_sets.json").read_text())
    print(f"[A1] old TDA columns={len(tda)}; market dates={len(features):,}; panel asset-date rows={len(panel):,}")
    print(audit.groupby(["cloud", "k", "window", "homology"], dropna=False).size().to_string())
    print("[A3] market crash runs")
    print(market.to_string(index=False))
    print("[A4] H-a: supported in code: T/A inputs are row-z-scored; raw volatility is removed.")
    print("[A4] H-b: untested by this audit; V3 T cloud is PCA on z-scored returns, not raw-return delay embedding.")
    print("[A4] H-c: supported in code: V3 stores snapshots and no V4-style slope/velocity block.")
    print("[A4] H-d: supported: TDA stride=10 and forward-fill=True.")
    print("[A4] H-e: partly supported: H2 exists only in selected T/k/window combinations; frozen A H1/A H2 counts are absent.")
    print("[A4] H-f: supported: TDA lives in market_features and is shared across panel assets.")
    print(f"[audit] feature sets: { {k: len(v) for k, v in manifest.items()} }")


if __name__ == "__main__":
    main()
