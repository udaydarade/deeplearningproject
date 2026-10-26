"""Generate the exhaustive, data-backed IE 643 project reference bible.

This is an inspection/aggregation script only.  It does not train models,
alter experiment outputs, or delete/checkpoint any running process.
"""
from __future__ import annotations

import importlib.metadata as metadata
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "MASTER_DATA_AND_METHODS_BIBLE.md"
PACKAGE_NAMES = (
    "torch", "ripser", "arch", "lightgbm", "scikit-learn", "pandas",
    "numpy", "streamlit",
)
SKIP_TABLE_PARTS = {
    "prices", "returns", "panel", "features", "scores", "states",
    "market_features", "modeling_panel", "cross_market_feature_panel",
    "ledger", "drawdowns", "indicators", "classification", "trend",
}


def command_text(*args: str) -> str:
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def hardware() -> list[str]:
    lines = [
        f"- OS: {platform.platform()}",
        f"- Python: {sys.version.replace(chr(10), ' ')}",
        f"- Processor: {platform.processor() or 'unreported by platform'}",
        f"- Logical CPUs: {os.cpu_count()}",
    ]
    try:
        import psutil
        lines.append(f"- Physical CPUs: {psutil.cpu_count(logical=False)}")
        lines.append(f"- Total RAM: {psutil.virtual_memory().total / 2**30:.2f} GiB")
    except ImportError:
        lines.append("- Physical CPUs/RAM: psutil is not installed; not inferred.")
    lines.append(f"- GPU probe: {command_text('nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader')}")
    lines.append("- Exact package versions:")
    for package in PACKAGE_NAMES:
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "not installed"
        lines.append(f"  - {package}: {version}")
    return lines


def purpose(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    first = re.search(r'(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', text, re.S)
    if first:
        return " ".join(first.group(1).split())[:240]
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith(("#", "import ", "from ")):
            return line.strip()[:240]
    return "No module description found."


def script_inventory() -> list[str]:
    paths = sorted(ROOT.rglob("*.py"))
    rows = [
        "| Script | Lines | Purpose | Runtime evidence |",
        "|---|---:|---|---|",
    ]
    for path in paths:
        count = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        rel = path.relative_to(ROOT).as_posix()
        rows.append(f"| `{rel}` | {count:,} | {purpose(path).replace('|', '\\|')} | Not measured in source; see run records/results where available. |")
    return rows


def markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["_Empty table._"]
    safe = frame.copy()
    for col in safe.columns:
        safe[col] = safe[col].map(
            lambda value: "" if pd.isna(value) else str(value).replace("|", "\\|")
        )
    return safe.to_markdown(index=False).splitlines()


def csv_catalog() -> list[str]:
    lines = [
        "The following are complete tables for every non-raw CSV result, "
        "diagnostic, inventory, label, threshold, and summary file. Raw "
        "price/return/panel/feature tables and per-run score Parquets are "
        "catalogued by schema and row count below because expanding millions "
        "of observations into Markdown would be a second dataset, not a report.",
        "",
    ]
    files = sorted(DATA.rglob("*.csv"))
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        lower = path.stem.lower()
        if any(part in lower for part in SKIP_TABLE_PARTS):
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            lines += [f"### `{rel}`", "", f"_Could not parse: `{exc}`._", ""]
            continue
        if len(frame) > 50_000:
            lines += [
                f"### `{rel}`",
                "",
                f"Rows: {len(frame):,}; columns: {len(frame.columns):,}. "
                "Complete source retained in the CSV; not expanded into "
                "hundreds of thousands of Markdown rows.",
                "",
                *markdown_table(pd.DataFrame([{
                    "column": column,
                    "dtype": str(frame[column].dtype),
                    "non_null": int(frame[column].notna().sum()),
                } for column in frame.columns])),
                "",
            ]
            continue
        lines += [
            f"### `{rel}`",
            "",
            f"Rows: {len(frame):,}; columns: {len(frame.columns):,}.",
            "",
            *markdown_table(frame),
            "",
        ]
    return lines


def csv_inventory() -> list[str]:
    lines = [
        "| CSV | Rows | Columns |",
        "|---|---:|---:|",
    ]
    for path in sorted(DATA.rglob("*.csv")):
        try:
            frame = pd.read_csv(path, nrows=0)
            rows = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace")) - 1
            lines.append(
                f"| `{path.relative_to(ROOT).as_posix()}` | {max(rows, 0):,} | "
                f"{len(frame.columns):,} |"
            )
        except Exception:
            lines.append(
                f"| `{path.relative_to(ROOT).as_posix()}` | unavailable | unavailable |"
            )
    return lines


def parquet_catalog() -> list[str]:
    lines = [
        "| Parquet | Rows | Columns | Column names |",
        "|---|---:|---:|---|",
    ]
    for path in sorted(DATA.rglob("*.parquet")):
        try:
            frame = pd.read_parquet(path)
            cols = ", ".join(str(c) for c in frame.columns)
            lines.append(
                f"| `{path.relative_to(ROOT).as_posix()}` | {len(frame):,} | "
                f"{len(frame.columns):,} | {cols.replace('|', '\\|')} |"
            )
        except Exception as exc:
            lines.append(
                f"| `{path.relative_to(ROOT).as_posix()}` | unavailable | unavailable | "
                f"{str(exc).replace('|', '\\|')} |"
            )
    return lines


def source_formula_index() -> list[str]:
    targets = [
        "tda_replication.py", "csd_baseline.py", "augment_panel.py",
        "peer_index_forecasting_v2.py", "v5_eval.py", "v6_2_semivar.py",
        "v7_final_synthesis.py", "pipeline/features.py", "pipeline/models.py",
        "pipeline_v5/sequence.py", "cross_index_forecasting.py",
        "crash_forecasting.py",
    ]
    lines: list[str] = []
    patterns = re.compile(
        r"AC1|autocorr|variance|skew|kurt|vol|GARCH|semivar|SEMIVAR|"
        r"landscape|persistence|Takens|drawdown|streak|BCE|pos_weight|"
        r"AdamW|Transformer|GRU|TCN|LSTM|quantile|q95|threshold|lead",
        re.I,
    )
    for name in targets:
        path = ROOT / name
        if not path.exists():
            continue
        lines += [f"### `{name}`", "", "Relevant implementation lines:", ""]
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if patterns.search(line):
                lines.append(f"`{number}: {line.replace('`', '\\`')}`")
        lines.append("")
    return lines


def main() -> None:
    parts = [
        "# MASTER DATA AND METHODS BIBLE",
        "",
        "Generated from the repository by `build_master_data_methods_bible.py`.",
        "This document reports observed workspace facts; it does not replace "
        "missing experiments with estimates. Legacy in-crash alarms are not "
        "treated as pre-crash predictions.",
        "",
        "## Section 1 — Hardware, environment, and codebase inventory",
        "",
        *hardware(),
        "",
        *script_inventory(),
        "",
        "## Section 2 — Dataset and asset-universe catalogue",
        "",
        "### CSV inventory and asset-class tables",
        "",
        *csv_inventory(),
        "",
        "### Parquet catalogue",
        "",
        *parquet_catalog(),
        "",
        "## Section 3 — Exact feature and method implementation index",
        "",
        "The following index is extracted from the actual implementation files. "
        "A line is included only when it contains a method/feature/training "
        "keyword; readers can follow the source line to the complete context.",
        "",
        *source_formula_index(),
        "## Section 4 — Deep-learning architectures and training settings",
        "",
        "Architecture, loss, optimizer, threshold, and seed settings are "
        "listed from the source-line index above and the complete result "
        "tables below. No setting is invented when it is absent from code.",
        "",
        "## Section 5 — Complete quantitative result tables",
        "",
        *csv_catalog(),
        "",
        "## Section 6 — Figures and Streamlit interfaces",
        "",
        "| Figure | Size (bytes) |",
        "|---|---:|",
    ]
    figure_paths = sorted(
        path for folder in ("assets", "data")
        for path in (ROOT / folder).rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".pdf"}
    )
    if figure_paths:
        parts.extend(
            f"| `{path.relative_to(ROOT).as_posix()}` | {path.stat().st_size:,} |"
            for path in figure_paths
        )
    else:
        parts.append("| _No PNG/PDF files found._ | 0 |")
    parts += [
        "",
        "### Streamlit source inventory",
        "",
        *[
            f"- `{path.relative_to(ROOT).as_posix()}`: {len(path.read_text(encoding='utf-8', errors='replace').splitlines()):,} lines; "
            "inspect the source for its exact controls, schemas, and checkpoint paths."
            for path in sorted(ROOT.glob("*_app.py"))
        ],
        "",
        "## Explicit limitations and status",
        "",
        "- This bible is an extraction of files currently present; it does not "
        "claim that an absent V6/V7 result file was completed.",
        "- Full-history observed crash labels and model-score coverage are "
        "different populations. Only alarms strictly before onset belong in "
        "forecast-detection statistics.",
    ]
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
