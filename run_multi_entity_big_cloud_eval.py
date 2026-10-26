"""Focused V5 walk-forward evaluation for the large multi-entity blocks.

This reuses the project's existing V5 E2 assembly, splitting, model, alarm, and
metric functions. It does not recompute TDA features.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import pandas as pd

from v5_1_e1 import make_data
from v5_9_e2 import assemble, feature_groups, score_one
from v5_eval import W_ERAS

ROOT = Path(__file__).resolve().parent
V3 = ROOT / "data" / "v3"
V5 = ROOT / "data" / "v5"
OUT = V5 / "multi_entity_big_cloud_results.csv"


def main() -> None:
    panel = pd.read_parquet(V3 / "panel.parquet")
    shared = pd.read_parquet(V3 / "market_features.parquet")
    manifest = json.loads((V3 / "feature_sets.json").read_text(encoding="utf-8"))
    base_name = json.loads((V5 / "base_set.json").read_text(encoding="utf-8"))["feature_set"]
    base, base_features = make_data(base_name, panel, shared, manifest)
    groups = feature_groups(panel, manifest)
    names = ["M0", "M0+TDA_big", "M0+DYN", "M0+CLUSTER", "M0+TDA_big+CLUSTER", "M0+TDA_big+DYN+CLUSTER"]
    models = ["logreg"]
    eras_to_run = [era for era in W_ERAS if era in {"E8", "E9", "E10"}]
    completed = set()
    if OUT.exists():
        old = pd.read_csv(OUT)
        completed = set(tuple(x) for x in old[["feature_set", "model", "era"]].drop_duplicates().itertuples(index=False, name=None))
    cache = {}
    rows = []
    for name in names:
        if name not in cache:
            cache[name] = (base.copy(), list(base_features)) if name == "M0" else assemble(name, base, base_features, groups)
        frame, features = cache[name]
        for model in models:
            for era in eras_to_run:
                if (name, model, era) in completed:
                    continue
                result = score_one(frame, features, "W", era, model, 9100 + len(rows))
                if result is None:
                    continue
                primary, validation, scores = result
                for row in primary:
                    if row["rule"] == "RQ" and row["budget"] == "5%":
                        row["feature_set"] = name
                        row["evaluation_source"] = "focused V5 E2 reuse"
                        row["model"] = model
                        row["era"] = era
                        rows.append(row)
                # Checkpoint each completed era so long runs remain resumable.
                new = pd.DataFrame(rows)
                old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
                combined = pd.concat([old, new], ignore_index=True) if len(old) else new
                combined.drop_duplicates(["feature_set", "model", "era", "rule", "budget"], keep="last").to_csv(OUT, index=False)
                rows = []
                print(f"{name} {model} {era}: rows={len(primary)} features={validation['features_used']}", flush=True)
                del scores
                gc.collect()
    if rows:
        new = pd.DataFrame(rows)
        old = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
        combined = pd.concat([old, new], ignore_index=True) if len(old) else new
        combined.drop_duplicates(["feature_set", "model", "era", "rule", "budget"], keep="last").to_csv(OUT, index=False)
    print(f"Saved {OUT}; new primary rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
