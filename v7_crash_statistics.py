"""Build detailed observed-crash and model-detection statistics for V7.

This is read-only with respect to V1-V6 and writes only V7 statistics/report
artifacts. It can be rerun while V7 is active; it analyzes the checkpoints
available at invocation time.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V6 = ROOT / "data" / "v6"
V7 = ROOT / "data" / "v7_final"
OUT = V7 / "crash_stats"


def runs(mask: np.ndarray, gap: int = 5) -> list[tuple[int, int]]:
    ix = np.flatnonzero(mask)
    if not len(ix):
        return []
    out: list[list[int]] = [[int(ix[0]), int(ix[0])]]
    for value in ix[1:]:
        value = int(value)
        if value - out[-1][1] - 1 <= gap:
            out[-1][1] = value
        else:
            out.append([value, value])
    return [(a, b) for a, b in out]


def load_events(label_col: str, label_name: str) -> pd.DataFrame:
    rows = []
    for path in sorted((V6 / "labels").glob("*.parquet")):
        d = pd.read_parquet(path, columns=["date", "asset", label_col])
        d["date"] = pd.to_datetime(d["date"])
        d = d.loc[d[label_col].notna()].sort_values("date").reset_index(drop=True)
        y = d[label_col].astype(bool).to_numpy()
        for number, (start, end) in enumerate(runs(y), 1):
            dates = d.date.iloc[start:end + 1]
            rows.append({
                "label": label_name,
                "asset": str(d.asset.iloc[0]),
                "event_number": number,
                "start_date": dates.iloc[0],
                "end_date": dates.iloc[-1],
                "duration_sessions": int(end - start + 1),
                "peak_label_days": int(y[start:end + 1].sum()),
                "event_id": f"{d.asset.iloc[0]}|{dates.iloc[0].date()}|{dates.iloc[-1].date()}",
            })
    return pd.DataFrame(rows)


def event_date_table(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for date, g in events.groupby("start_date"):
        rows.append({
            "label": g.label.iloc[0],
            "onset_date": date,
            "assets_with_crash_onset": g.asset.nunique(),
            "event_count": len(g),
            "assets": ", ".join(sorted(g.asset.astype(str))),
            "marketwide_5_assets": g.asset.nunique() >= 5,
            "marketwide_10_assets": g.asset.nunique() >= 10,
        })
    return pd.DataFrame(rows).sort_values(["label", "onset_date"])


def parse_score_name(path: Path) -> dict[str, str]:
    stem = path.stem
    m = re.match(r"(?P<label>y_N(?:5|15)_p2)_(?P<fold>[^_]+_[^_]+|2021_plus)_(?P<rest>.*)", stem)
    if not m:
        return {"score_file": path.name, "label": "", "fold": "", "feature_set": "",
                "model": "", "seed": ""}
    rest = m.group("rest")
    bits = rest.rsplit("_", 2)
    return {"score_file": path.name, "label": m.group("label"), "fold": m.group("fold"),
            "feature_set": bits[0], "model": bits[1], "seed": bits[2]}


def detection_table(events: pd.DataFrame) -> pd.DataFrame:
    records = []
    event_lookup = {}
    for _, row in events.iterrows():
        event_lookup.setdefault((row.label, row.asset), []).append(row)
    for path in sorted((V7 / "scores").glob("*.parquet")):
        meta = parse_score_name(path)
        d = pd.read_parquet(path, columns=["date", "asset", "y", "alarm"])
        d["date"] = pd.to_datetime(d.date)
        label = meta["label"]
        # Score files are fold subsets; only compare against events from the
        # same label and asset, preserving the file's actual causal coverage.
        for asset, g in d.groupby("asset", sort=False):
            g = g.sort_values("date")
            for event in event_lookup.get((label, str(asset)), []):
                inside = g[(g.date >= event.start_date) & (g.date <= event.end_date)]
                if inside.empty:
                    continue
                records.append({
                    **meta,
                    "label": label,
                    "event_id": event.event_id,
                    "asset": str(asset),
                    "start_date": event.start_date,
                    "end_date": event.end_date,
                    "detected": bool(inside.alarm.astype(bool).any()),
                    "alarm_days_inside_event": int(inside.alarm.astype(bool).sum()),
                    "score_rows_inside_event": len(inside),
                })
    return pd.DataFrame(records)


def report(events, onset, detections, result):
    lines = [
        "# V7 crash statistics and detection report", "",
        "This report is generated from the V6 per-asset severity labels and the "
        "V7 score Parquet checkpoints available at report time. It is additive "
        "and does not modify V1-V6 files.", "",
        "## Coverage caveat", "",
        f"- Label events: {len(events):,}.",
        f"- Score-file detection records: {len(detections):,}.",
        f"- V7 result rows available: {len(result):,}.",
        "- The label statistics cover the full available asset histories.",
        "- Model detection statistics cover only score files already saved; they "
        "are not yet full-history model coverage unless every fold is present.",
        "",
        "## Crash-event totals by horizon", "",
        "| Label | Events | Assets | First onset | Last onset | Median duration |",
        "|---|---:|---:|---|---|---:|",
    ]
    for label, g in events.groupby("label", sort=True):
        lines.append(f"| {label} | {len(g):,} | {g.asset.nunique():,} | "
                     f"{g.start_date.min().date()} | {g.start_date.max().date()} | "
                     f"{g.duration_sessions.median():.1f} |")
    lines += [
        "", "## Cross-asset crash-onset dates", "",
        "| Label | Onset dates | Max simultaneous assets | Dates with >=5 assets | Dates with >=10 assets |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, g in onset.groupby("label", sort=True):
        lines.append(f"| {label} | {len(g):,} | {g.assets_with_crash_onset.max():,} | "
                     f"{int(g.marketwide_5_assets.sum()):,} | "
                     f"{int(g.marketwide_10_assets.sum()):,} |")
    lines += ["", "## Assets with the most observed crash events", "",
              "| Label | Asset | Events | Median duration | Longest event |",
              "|---|---|---:|---:|---:|"]
    by_asset = events.groupby(["label", "asset"]).agg(
        events=("event_id", "count"), median_duration=("duration_sessions", "median"),
        longest_event=("duration_sessions", "max")).reset_index()
    for _, r in by_asset.sort_values(["label", "events"], ascending=[True, False]).head(40).iterrows():
        lines.append(f"| {r.label} | {r.asset} | {r.events} | "
                     f"{r.median_duration:.1f} | {r.longest_event} |")
    lines += ["", "## Detection counts by model", "",
              "| Label | Feature set | Model | Seed | Fold | Events covered | Detected | Detection rate |",
              "|---|---|---|---|---|---:|---:|---:|"]
    if not detections.empty:
        grp = detections.groupby(["label", "feature_set", "model", "seed", "fold"]).agg(
            events_covered=("event_id", "nunique"), detected=("detected", "sum")).reset_index()
        for _, r in grp.sort_values(["label", "feature_set", "model", "fold"]).iterrows():
            rate = r.detected / r.events_covered if r.events_covered else np.nan
            lines.append(f"| {r.label} | {r.feature_set} | {r.model} | {r.seed} | "
                         f"{r.fold} | {r.events_covered} | {r.detected} | {rate:.4f} |")
    lines += ["", "## Crashes detected by multiple models", "",
              "| Label | Event | Asset | Onset | Models covering event | Models detecting | Detection count |",
              "|---|---|---|---|---:|---|---:|"]
    if not detections.empty:
        ev = detections.groupby(["label", "event_id", "asset", "start_date"]).agg(
            models_covering=("score_file", "nunique"),
            models_detecting=("detected", "sum"),
            detecting_files=("score_file", lambda x: ", ".join(x))).reset_index()
        for _, r in ev.sort_values(["label", "models_detecting"], ascending=[True, False]).head(100).iterrows():
            lines.append(f"| {r.label} | {r.event_id} | {r.asset} | "
                         f"{pd.Timestamp(r.start_date).date()} | {r.models_covering} | "
                         f"{r.detecting_files} | {r.models_detecting} |")
    lines += ["", "## Interpretation", "",
              "- An observed crash is a maximal run of label-positive sessions, "
              "merging gaps of up to five sessions, matching the project event rule.",
              "- A model detects an event only when an alarm day falls inside that "
              "event's positive run.",
              "- “Models covering” counts saved score files, so it is a coverage "
              "denominator, not a claim that every planned V7 cell ran.",
              "- The CSV tables in `data/v7_final/crash_stats/` contain the complete "
              "event-level and detection-level records, including events omitted "
              "from the abbreviated Markdown tables.",
    ]
    (ROOT / "V7_CRASH_STATISTICS_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    events = pd.concat([
        load_events("y_down_N5_p2", "y_N5_p2"),
        load_events("y_down_N15_p2", "y_N15_p2"),
    ], ignore_index=True)
    onset = event_date_table(events)
    detections = detection_table(events)
    result_path = V7 / "v7_ablation_results.csv"
    result = pd.read_csv(result_path) if result_path.exists() else pd.DataFrame()
    events.to_csv(OUT / "observed_crash_events.csv", index=False)
    onset.to_csv(OUT / "cross_asset_crash_onsets.csv", index=False)
    detections.to_csv(OUT / "model_event_detections.csv", index=False)
    if not detections.empty:
        detections.groupby(["label", "feature_set", "model", "seed", "fold"]).agg(
            events_covered=("event_id", "nunique"),
            detected=("detected", "sum"),
            detection_rate=("detected", "mean"),
        ).reset_index().to_csv(OUT / "model_detection_summary.csv", index=False)
    report(events, onset, detections, result)
    print({"events": len(events), "onsets": len(onset),
           "detections": len(detections), "result_rows": len(result)})


if __name__ == "__main__":
    main()
