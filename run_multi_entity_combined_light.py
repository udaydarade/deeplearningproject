"""Evaluate the intended compact combined showcase: ALLS 36-D landscape + CLUSTER."""
from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd

from v5_1_e1 import make_data
from v5_9_e2 import score_one
from v5_eval import W_ERAS

ROOT = Path(__file__).resolve().parent
V3 = ROOT / "data" / "v3"
V5 = ROOT / "data" / "v5"
OUT = V5 / "multi_entity_big_cloud_results.csv"


def main() -> None:
    panel = pd.read_parquet(V3 / "panel.parquet")
    shared = pd.read_parquet(V3 / "market_features.parquet")
    import json
    manifest = json.loads((V3 / "feature_sets.json").read_text(encoding="utf-8"))
    base_name = json.loads((V5 / "base_set.json").read_text(encoding="utf-8"))["feature_set"]
    base, base_features = make_data(base_name, panel, shared, manifest)
    landscape = pd.read_parquet(V5 / "blocks" / "V5_TDA.parquet")
    cluster = pd.read_parquet(V5 / "blocks" / "V5_CLUSTER.parquet")
    landscape_cols = sorted([c for c in landscape.columns if "v5_ALLS_A_corr_kall_w60_h1_landscape_" in c],
                            key=lambda c: int(c.rsplit("_", 1)[1]))[:36]
    cluster_cols = [c for c in cluster.columns if c != "date"]
    frame = base[["date", "asset", "y", *base_features]].copy()
    frame = frame.merge(landscape[["date", *landscape_cols]], on="date", how="left", validate="many_to_one")
    frame = frame.merge(cluster[["date", *cluster_cols]], on="date", how="left", validate="many_to_one")
    features = base_features + landscape_cols + cluster_cols
    rows = []
    for era in [x for x in W_ERAS if x in {"E8", "E9", "E10"}]:
        result = score_one(frame, features, "W", era, "logreg", 9300 + len(rows))
        if result is None:
            continue
        primary, validation, scores = result
        for row in primary:
            if row["rule"] == "RQ" and row["budget"] == "5%":
                row.update({"feature_set": "M0+ALLS36D+CLUSTER", "model": "logreg", "era": era,
                            "evaluation_source": "focused V5 E2 reuse; compact 36-D ALLS landscape + CLUSTER"})
                rows.append(row)
        print(f"M0+ALLS36D+CLUSTER logreg {era}: rows={len(primary)} features={validation['features_used']}", flush=True)
        del scores
        gc.collect()
    if rows:
        new = pd.DataFrame(rows)
        old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
        combined = pd.concat([old, new], ignore_index=True) if len(old) else new
        combined.drop_duplicates(["feature_set", "model", "era", "rule", "budget"], keep="last").to_csv(OUT, index=False)
    print(f"Saved combined rows={len(rows)} to {OUT}")


if __name__ == "__main__":
    main()
