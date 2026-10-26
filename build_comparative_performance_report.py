"""Build classical/TDA/peer/SEMIVAR comparative report from saved outputs."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
V5 = ROOT / "data/v5"
V7 = ROOT / "data/v7_final"
OUT = V7 / "comparative"


def rq_table(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    d = d[(d.get("rule") == "RQ") & (d.get("budget") == "5%")]
    if d.empty:
        # Scheme-B rows use RE; retain the same 5% budget for historical
        # comparison but label the threshold rule accurately in the report.
        d = pd.read_csv(path)
        d = d[(d.get("rule") == "RE") & (d.get("budget") == "5%")]
    d = d.copy()
    d["source"] = name
    return d


def aggregate(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return d
    keys = [c for c in ["source", "scheme", "feature_set", "model"] if c in d]
    return d.groupby(keys, as_index=False).agg(
        cells=("era_or_fold", "size"),
        eras=("era_or_fold", "nunique"),
        mean_ap=("ap", "mean"),
        mean_auc=("auc", "mean"),
        mean_alarm_rate=("alarm_day_rate", "mean"),
        mean_event_recall=("event_recall", "mean"),
        mean_random_event_recall=("rand_event_recall", "mean"),
        mean_excess_event_recall=("excess_event_recall", "mean"),
        mean_event_precision=("event_precision", "mean"),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = [
        rq_table(V5 / "e1_results.csv", "V5_E1"),
        rq_table(V5 / "e2_results.csv", "V5_E2"),
    ]
    v5 = pd.concat([x for x in parts if not x.empty], ignore_index=True)
    v5_agg = aggregate(v5)
    v5_agg.to_csv(OUT / "v5_comparative_metrics.csv", index=False)

    v7 = pd.read_csv(V7 / "v7_ablation_results.csv") if (V7 / "v7_ablation_results.csv").exists() else pd.DataFrame()
    if not v7.empty:
        v7["source"] = "V7_FINAL"
        v7_agg = v7.groupby(["source", "label", "feature_set", "model"], as_index=False).agg(
            cells=("fold", "size"), folds=("fold", "nunique"), mean_ap=("ap", "mean"),
            mean_auc=("auc", "mean"), mean_alarm_rate=("alarm_day_rate", "mean"),
            mean_event_recall=("event_recall", "mean"),
            mean_random_event_recall=("rand_event_recall", "mean"),
            mean_excess_event_recall=("excess_event_recall", "mean"),
            mean_event_precision=("event_precision", "mean"),
            mean_lead_time=("mean_lead_time", "mean"),
        )
        v7_agg.to_csv(OUT / "v7_comparative_metrics.csv", index=False)
    else:
        v7_agg = pd.DataFrame()

    event_path = V7 / "crash_stats/observed_crash_events.csv"
    events = pd.read_csv(event_path) if event_path.exists() else pd.DataFrame()
    detection_path = V7 / "crash_stats/model_detection_summary.csv"
    detections = pd.read_csv(detection_path) if detection_path.exists() else pd.DataFrame()

    lines = [
        "# Classical vs topological vs peer/SEMIVAR performance",
        "",
        "## Scope and denominator",
        "",
        "The underlying panel spans 1927-12-30 through 2026-10-23 and contains "
        "159 assets. The observed-crash tables below use all available asset "
        "label files over that full period.",
        "",
        "Model results are reported with their actual coverage. V5 completed "
        "cells use the full 159-asset panel and V5 era definitions. V7 is still "
        "in progress: its saved model scores currently cover only the completed "
        "modern folds, not the entire historical period. Therefore V7 detection "
        "counts must not be interpreted as full-history model detection rates.",
        "",
        "## Observed crash population: full entities and full period",
        "",
        "| Label | Events | Assets |",
        "|---|---:|---:|",
    ]
    if not events.empty:
        for label, g in events.groupby("label"):
            lines.append(f"| {label} | {len(g):,} | {g.asset.nunique():,} |")
    lines += [
        "",
        "These are observed label episodes, not model detections. An episode is a "
        "maximal positive run after merging gaps of up to five sessions.",
        "",
        "## V5 full-era comparative results",
        "",
        "The table uses saved 5% operating points. Scheme B rows use the V5 "
        "within-era RE rule; Scheme W rows use the causal RQ rule when present. "
        "Results are means over completed era cells only.",
        "",
        "| Source | Scheme | Feature set | Model | Cells | Eras | AP | AUC | Alarm rate | Event recall | Random recall | Excess recall |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in v5_agg.iterrows():
        lines.append(f"| {r.source} | {r.scheme} | {r.feature_set} | {r.model} | "
                     f"{r.cells} | {r.eras} | {r.mean_ap:.4f} | {r.mean_auc:.4f} | "
                     f"{r.mean_alarm_rate:.4f} | {r.mean_event_recall:.4f} | "
                     f"{r.mean_random_event_recall:.4f} | {r.mean_excess_event_recall:.4f} |")
    lines += [
        "",
        "V5 feature interpretation:",
        "",
        "- `B1`/`F0` are classical/CSD-oriented baselines.",
        "- `B2`/`F1` add price momentum, volatility, relative strength, and peer-style "
        "cross-asset features.",
        "- `B3`/`F2` add broad market-state, breadth, dispersion, correlation, "
        "eigenvalue, cluster, and macro-state variables.",
        "- `M0+TDA_small` is the V5 topology augmentation; its E2 result is "
        "incomplete and must not be compared as a finished full-era model.",
        "",
        "## V7 ablation results currently saved",
        "",
        "| Label | Feature set | Model | Cells | Folds | AP | AUC | Alarm rate | Event recall | Random recall | Excess recall | Lead time |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in v7_agg.iterrows():
        lines.append(f"| {r.label} | {r.feature_set} | {r.model} | {r.cells} | "
                     f"{r.folds} | {r.mean_ap:.4f} | {r.mean_auc:.4f} | "
                     f"{r.mean_alarm_rate:.4f} | {r.mean_event_recall:.4f} | "
                     f"{r.mean_random_event_recall:.4f} | {r.mean_excess_event_recall:.4f} | "
                     f"{r.mean_lead_time:.2f} |")
    lines += [
        "",
        "V7 feature sets are progressive ablations:",
        "`CLASSICAL_CSD`, `SEMIVAR_MOM`, `PEER_CROSS`, `TDA_ONLY_NORM`, and "
        "`GRAND_COMBINED`. The V7 TDA features are the saved 36-coordinate "
        "scale-normalized landscape block merged by date. V7 deep models use "
        "Group A only; tabular models use the unified panel.",
        "",
        "## Which observed crashes were detected?",
        "",
        "The complete event-level model detection records are in "
        "`data/v7_final/crash_stats/model_event_detections.csv`. The summary "
        "below counts only saved score files.",
        "",
        "| Label | Feature set | Model | Fold | Events covered | Detected | Detection rate |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    if not detections.empty:
        for _, r in detections.sort_values(["label", "feature_set", "model", "fold"]).iterrows():
            lines.append(f"| {r.label} | {r.feature_set} | {r.model} | {r.fold} | "
                         f"{r.events_covered} | {r.detected} | {r.detection_rate:.4f} |")
    lines += [
        "",
        "## Scientific conclusion",
        "",
        "The full-period observed-crash population is large and is measured over "
        "all 159 assets. The completed V5 results show that classical baselines, "
        "peer/market-state features, and topology are not interchangeable feature "
        "families. The completed V7 cells currently provide a focused modern-fold "
        "ablation, but they do not yet provide full-period model detection counts.",
        "",
        "A final claim that one family detects crashes across the entire 1927-2026 "
        "record requires completing the historical V7 folds and then regenerating "
        "the detection tables. Until then, this report intentionally separates "
        "full-history crash counts from partial saved-model detection coverage.",
        "",
        "Machine-readable tables:",
        "",
        "- `data/v7_final/comparative/v5_comparative_metrics.csv`",
        "- `data/v7_final/comparative/v7_comparative_metrics.csv`",
        "- `data/v7_final/crash_stats/observed_crash_events.csv`",
        "- `data/v7_final/crash_stats/model_event_detections.csv`",
    ]
    (ROOT / "CLASSICAL_TDA_OTHERS_COMPARATIVE_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print("wrote comparative report", len(v5_agg), len(v7_agg),
          "event rows", len(events), "detection summary rows", len(detections))


if __name__ == "__main__":
    main()
