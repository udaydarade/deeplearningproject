"""Produce the definitive per-run record for both benchmark grids.

Every planned run (v2: 800, v1: 420) gets exactly one row:
- ``done``  -> full test metrics of that model on that fold
- ``skipped_insufficient_data`` -> crypto group cells (zero positives in the
  2024-26 test window under the calibration-era threshold)
- ``not_run`` -> enqueued but cut when compute was stopped

Outputs:
- ``data/benchmark/run_record_v2.csv``  (800 rows)
- ``data/benchmark/run_record_v1.csv``  (420 rows)
- ``RUN_RECORD.md`` (project root): per-run tables + status accounting

Run with ``py -3.12 build_run_record.py`` (no training involved).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import pipeline.benchmark as B

ROOT = B.ROOT
OUT = B.OUT

V2_GROUP_MODES = ["classical", "tda", "landscape", "combined", "full"]
V1_MODES = ["classical", "tda", "combined"]
METRIC_COLS = ["n", "positives", "prevalence", "average_precision", "roc_auc",
               "brier", "threshold", "precision", "recall", "f1", "alarms",
               "train_start", "train_end", "val_start", "test_start", "test_end",
               "feature_count", "sequence_length"]
KEY = ["grid", "level", "dataset", "mode", "model", "fold"]


def planned_rows(panel: pd.DataFrame, grid: str) -> list[dict]:
    modes = V2_GROUP_MODES if grid == "v2" else V1_MODES
    models = [m for m in B.MODEL_NAMES if not (grid == "v1" and m == "attn_bilstm")]
    calendar = [f for f, *_ in B.FOLDS]
    if grid == "v1":
        # v1 plan: five calendar-fold datasets; tech started 2012 so its first
        # fold had no evaluable test window (excluded); crypto not included.
        group_plan = {ds: calendar for ds in
                      ("energy", "financials", "group_a_us_indices",
                       "group_b_cross_asset", "international")}
        group_plan["tech"] = ["fold_2016_2020", "fold_2021_plus"]
        ticker_plan = {"ALL": calendar}
    else:
        plan = B.fold_plan(panel, "group", sorted(panel.dataset.unique()), B.FOLDS)
        group_plan = {k: [f for f, *_ in v] for k, v in plan.items()}
        ticker_plan = {"ALL": calendar}
    rows = []
    for level in ("group", "ticker"):
        by_dataset = ticker_plan if level == "ticker" else group_plan
        for name, folds in by_dataset.items():
            for mode in modes:
                for model in models:
                    for fold_name in folds:
                        rows.append({
                            "grid": grid, "level": level, "dataset": name,
                            "mode": mode, "model": model, "fold": fold_name,
                            "planned_test_start": pd.NA, "planned_test_end": pd.NA,
                        })
    return rows


def build_record(grid: str) -> pd.DataFrame:
    result_file = "all_results.csv" if grid == "v2" else "all_results_v1.csv"
    results = pd.read_csv(OUT / result_file)
    overall = results[results["ticker"].isna()].copy()
    lookup = {}
    for _, row in overall.iterrows():
        lookup[(row["level"], row["dataset"], row["mode"], row["model"], row["fold"])] = row

    panel = B.load_panel()
    record = pd.DataFrame(planned_rows(panel, grid))
    for col in METRIC_COLS:
        record[col] = pd.NA
    for i, plan_row in record.iterrows():
        key = (plan_row["level"], plan_row["dataset"], plan_row["mode"],
               plan_row["model"], plan_row["fold"])
        hit = lookup.get(key)
        if hit is not None:
            record.loc[i, "status"] = "done"
            for col in METRIC_COLS:
                val = hit.get(col, pd.NA)
                record.loc[i, col] = pd.NA if pd.isna(val) else val
        elif plan_row.dataset == "crypto" and plan_row.level == "group" and grid == "v2":
            record.loc[i, "status"] = "skipped_insufficient_data"
        else:
            record.loc[i, "status"] = "not_run"
    record = record.sort_values(["level", "dataset", "mode", "model", "fold"])
    record.to_csv(OUT / f"run_record_{grid}.csv", index=False)
    return record


def fmt(v) -> str:
    return "-" if pd.isna(v) else f"{float(v):.4f}"


def mode_table(rec: pd.DataFrame, level: str, dataset: str, mode: str) -> str:
    sub = rec[(rec.level == level) & (rec.dataset == dataset) & (rec["mode"] == mode)]
    folds = list(sub.fold.unique())
    lines = [f"**{dataset} · {level}-level · mode `{mode}`**", "",
             "| model | " + " | ".join(str(f) for f in folds)
             + " | mean AP | status |", "|" + "---|" * (len(folds) + 3)]
    for model in sub.model.unique():
        block = sub[sub.model == model]
        vals = []
        status_sc = set(block.status)
        if status_sc == {"done"}:
            status = "done"
        elif "done" in status_sc:
            status = "partial"
        else:
            status = next(iter(status_sc))
        for f in folds:
            row = block[block.fold == f]
            ap = row.average_precision.iloc[0] if "done" in status_sc else pd.NA
            vals.append(fmt(ap))
        mean = pd.to_numeric(block.average_precision, errors="coerce").mean()
        lines.append(f"| {model} | " + " | ".join(vals) + f" | {fmt(mean)} | {status} |")
    return "\n".join(lines)


def main() -> None:
    v2 = build_record("v2")
    v1 = build_record("v1")
    counts_v2 = v2.status.value_counts().to_dict()
    counts_v1 = v1.status.value_counts().to_dict()

    parts = ["# Run record — every benchmark run, individually",
             "",
             "One row per (level × dataset × mode × model × fold). Machine-readable,",
             "full-metric versions: `data/benchmark/run_record_v2.csv` (800 rows),",
             "`data/benchmark/run_record_v1.csv` (420 rows). Per-ticker breakdown rows",
             "(5,160 v2 + 3,494 v1 raw rows) stay in `all_results.csv` /",
             "`all_results_v1.csv`. Metric = test average precision (15-day horizon),",
             "means are over fold columns of the same row.",
             "",
             "## v2 accounting (800 planned runs)",
             "",
             "| status | runs |", "|---|---:|"]
    for status, n in sorted(counts_v2.items(), key=lambda kv: -kv[1]):
        parts.append(f"| {status} | {n} |")
    parts += ["", "## v1 accounting (420 planned runs)", "",
              "| status | runs |", "|---|---:|"]
    for status, n in sorted(counts_v1.items(), key=lambda kv: -kv[1]):
        parts.append(f"| {status} | {n} |")
    for grid, rec in (("v2", v2), ("v1", v1)):
        parts.append(f"## {grid}: status detail (runs per dataset)")
        detail = rec.groupby(["level", "dataset"]).status.value_counts().unstack(fill_value=0)
        parts.append(detail.to_string())

    # per-run tables
    for grid, rec in (("v2", v2), ("v1", v1)):
        parts.append(f"## {grid} — every run (AP by fold)")
        for level in ("group", "ticker"):
            for dataset in rec[rec.level == level].dataset.unique():
                for mode in rec[(rec.level == level) & (rec.dataset == dataset)]["mode"].unique():
                    parts.append("")
                    parts.append(mode_table(rec, level, dataset, mode))

    Path(ROOT / "RUN_RECORD.md").write_text("\n".join(parts), encoding="utf-8")
    print("v2:", counts_v2, "| v1:", counts_v1)
    print("wrote:", OUT / "run_record_v2.csv", OUT / "run_record_v1.csv",
          ROOT / "RUN_RECORD.md")


if __name__ == "__main__":
    main()
