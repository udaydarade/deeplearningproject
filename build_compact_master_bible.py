"""Write a compact evidence-backed project reference document."""
from __future__ import annotations

import importlib.metadata as metadata
import os
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "MASTER_DATA_AND_METHODS_BIBLE.md"


def version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    shown = frame.head(max_rows).copy()
    for column in shown.columns:
        shown[column] = shown[column].map(
            lambda value: "" if pd.isna(value) else str(value).replace("|", "\\|")
        )
    result = shown.to_markdown(index=False)
    if len(frame) > max_rows:
        result += f"\n\n_Showing {max_rows:,} of {len(frame):,} rows; full table is in the linked source file._"
    return result


def read_csv(relative: str) -> pd.DataFrame:
    path = ROOT / relative
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def gpu_probe() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return "none reported (nvidia-smi unavailable)"
    return result.stdout.strip() or "none reported"


def source_evidence(names: list[str]) -> list[str]:
    lines: list[str] = []
    keywords = (
        "BCEWithLogitsLoss", "pos_weight", "AdamW", "GARCH", "semivar",
        "SEMIVAR", "landscape", "Takens", "persistence", "Transformer",
        "GRU", "TCN", "quantile", "threshold", "lead",
    )
    for name in names:
        path = ROOT / name
        if not path.exists():
            continue
        matches = []
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if any(keyword.lower() in line.lower() for keyword in keywords):
                matches.append(f"{number}: {line.strip()}")
        lines += [f"### `{name}`", "", "```text", *matches[:80], "```", ""]
    return lines


def main() -> None:
    scripts = []
    for path in sorted(ROOT.rglob("*.py")):
        if any(part in {".venv", "venv", "__pycache__"} for part in path.parts):
            continue
        scripts.append({
            "script": path.relative_to(ROOT).as_posix(),
            "lines": len(path.read_text(encoding="utf-8", errors="replace").splitlines()),
        })

    status = read_csv("data/v7_final/v7_ablation_results.csv")
    v7_status = ROOT / "data/v7_final/v7_run_status.json"
    status_text = v7_status.read_text(encoding="utf-8") if v7_status.exists() else "missing"
    e2 = read_csv("data/v5/e2_v7_final_checkpoint.csv")
    classes = read_csv("data/v6/assets_by_class.csv")
    prevalence = read_csv("data/v6/label_prevalence.csv")
    observed = read_csv("data/v7_final/crash_stats/observed_crash_events.csv")
    precrash = read_csv("data/v7_final/precrash_ledger/precrash_model_summary.csv")
    v6 = read_csv(
        "data/v6/archive_pre_forward_label_fix_20261025/scheme_w_summary.csv"
    )

    lines = [
        "# MASTER DATA AND METHODS BIBLE (COMPACT)",
        "",
        "This compact reference is generated from files currently present in the "
        "repository. Large raw tables remain in CSV/Parquet form and are linked "
        "below; they are not duplicated into a hundreds-of-megabytes Markdown file.",
        "",
        "## 1. Environment",
        "",
        f"- OS: {platform.platform()}",
        f"- Python: {sys.version.split()[0]}",
        f"- Processor: {platform.processor() or 'not reported'}",
        f"- Logical CPUs: {os.cpu_count()}",
        f"- GPU probe: {gpu_probe()}",
        "",
        "| Package | Version |",
        "|---|---|",
    ]
    for package in ("torch", "ripser", "arch", "lightgbm", "scikit-learn", "pandas", "numpy", "streamlit"):
        lines.append(f"| {package} | {version(package)} |")
    lines += [
        "",
        "## 2. Codebase inventory",
        "",
        "Every Python file is listed with line count. Runtime is experiment-specific "
        "and is recorded in run logs/status files where available.",
        "",
        table(pd.DataFrame(scripts), max_rows=len(scripts)),
        "",
        "## 3. Dataset and event population",
        "",
        "### V6 asset classes",
        "",
        table(classes, max_rows=100),
        "",
        "### V6 label prevalence",
        "",
        table(prevalence, max_rows=100),
        "",
        "### Full-history observed crash events",
        "",
        f"- Observed event rows: {len(observed):,}.",
        f"- Distinct assets: {observed.asset.nunique() if not observed.empty else 0:,}.",
        f"- Labels: {', '.join(sorted(observed.label.unique())) if not observed.empty else 'none'}.",
        "- Source: `data/v7_final/crash_stats/observed_crash_events.csv`.",
        "",
        "## 4. Methods and implementation evidence",
        "",
        "The exact formulas and settings must be read in context from the source "
        "files below. This index prevents unsupported claims while keeping the "
        "reference compact.",
        "",
        *source_evidence([
            "tda_replication.py", "csd_baseline.py", "augment_panel.py",
            "peer_index_forecasting_v2.py", "pipeline/features.py",
            "pipeline/models.py", "v5_eval.py", "v6_2_semivar.py",
            "v7_final_synthesis.py", "pipeline_v5/sequence.py",
        ]),
        "## 5. Experiment status and quantitative outputs",
        "",
        "### V7 final synthesis",
        "",
        f"- Status file: `{status_text.strip()}`",
        f"- Saved result rows: {len(status):,}.",
        f"- Labels: {', '.join(sorted(status.label.unique())) if not status.empty else 'none'}.",
        f"- Feature sets: {', '.join(sorted(status.feature_set.unique())) if not status.empty else 'none'}.",
        f"- Models: {', '.join(sorted(status.model.unique())) if not status.empty else 'none'}.",
        f"- Folds: {', '.join(sorted(status.fold.unique())) if not status.empty else 'none'}.",
        "",
        table(status.groupby(
            ["label", "feature_set", "model"], as_index=False
        ).agg(
            cells=("seed", "count"),
            mean_ap=("ap", "mean"),
            mean_auc=("auc", "mean"),
            mean_alarm_rate=("alarm_day_rate", "mean"),
            mean_event_recall=("event_recall", "mean"),
            mean_lead=("mean_lead_time", "mean"),
        ), max_rows=100),
        "",
        "Full V7 results: `data/v7_final/v7_ablation_results.csv`.",
        "",
        "### Best verified V7 aggregate cells",
        "",
        table(
            status.sort_values(
                ["event_recall", "auc", "ap"],
                ascending=False,
            )[
                [
                    "label", "fold", "feature_set", "model", "seed",
                    "ap", "auc", "alarm_day_rate", "event_recall",
                    "mean_lead_time",
                ]
            ],
            max_rows=25,
        ),
        "",
        "### V7 strict pre-crash correction",
        "",
        "Only alarms strictly earlier than `crash_start` are valid detections. "
        "The legacy in-crash ledger is not used as forecasting evidence.",
        f"- Pre-crash event/model rows: {len(read_csv('data/v7_final/precrash_ledger/precrash_model_event_ledger.csv')):,}.",
        f"- Pre-crash summary rows: {len(precrash):,}.",
        "- Windows: 1, 5, 10, 20, and 60 prior trading sessions.",
        "- Full event ledger: `data/v7_final/precrash_ledger/precrash_model_event_ledger.csv`.",
        "- Event view: `data/v7_final/precrash_ledger/precrash_by_event.csv`.",
        "- Summary: `data/v7_final/precrash_ledger/precrash_model_summary.csv`.",
        "",
        "### V6 Scheme W headline table",
        "",
        table(
            v6[
                [
                    "N", "direction", "feature_set", "engine",
                    "mean_ap", "mean_auc", "mean_excess", "off_budget",
                ]
            ].sort_values(["direction", "N", "mean_auc"], ascending=[True, True, False]),
            max_rows=100,
        ) if not v6.empty else "_V6 Scheme W summary not present._",
        "",
        "### V5/V6 pending and completed state",
        "",
        f"- V5 E2 checkpoint rows: {len(e2):,}; checkpoint file: `data/v5/e2_v7_final_checkpoint.csv`.",
        "- V5/V6 outputs already present are preserved; no new long training job "
        "was started by this compact-report step.",
        "",
        "## 6. Canonical result files",
        "",
        "| Area | Files |",
        "|---|---|",
        "| V1/V2 benchmark | `data/benchmark/all_results.csv`, `data/benchmark/all_results_v1.csv` |",
        "| V3/V4 | `data/v3/`, `data/v4/` |",
        "| V5 | `data/v5/e1_results.csv`, `data/v5/e2_results.csv`, diagnostics |",
        "| V6 | `data/v6/` and archived V6 result tables |",
        "| V7 | `data/v7_final/v7_ablation_results.csv` and `data/v7_final/scores/` |",
        "| Crash population | `data/v7_final/crash_stats/observed_crash_events.csv` |",
        "",
        "## 7. Scientific limitations",
        "",
        "- V7 currently has complete saved cells for its configured folds, but "
        "its configured folds are not the same as testing every historical date "
        "from 1927 onward.",
        "- The compact document does not convert missing score coverage into "
        "false model misses.",
        "- A model detection is a forecast only when the alarm precedes onset; "
        "alarms on or after onset are excluded.",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
