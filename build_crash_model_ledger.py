"""Build an event-by-event crash/model ledger from saved V7 score files."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
V7 = ROOT / "data/v7_final"
STATS = V7 / "crash_stats"
OUT = V7 / "crash_ledger"


def meta(path: Path) -> dict[str, str]:
    m = re.match(
        r"(?P<label>y_(?:N5|down_N15)_p2)_(?P<fold>[^_]+_[^_]+|2021_plus)_(?P<rest>.*)",
        path.stem,
    )
    if not m:
        raise ValueError(f"Unrecognized score filename: {path.name}")
    bits = m.group("rest").rsplit("_", 2)
    return {
        "score_file": path.name,
        "label": m.group("label"),
        "fold": m.group("fold"),
        "feature_set": bits[0],
        "model": bits[1],
        "seed": bits[2],
        "model_key": f"{bits[0]}|{bits[1]}|seed={bits[2]}|fold={m.group('fold')}",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(STATS / "observed_crash_events.csv",
                         parse_dates=["start_date", "end_date"])
    events = events.set_index("event_id", drop=False)
    rows = []
    for path in sorted((V7 / "scores").glob("*.parquet")):
        info = meta(path)
        scores = pd.read_parquet(path, columns=["date", "asset", "y", "score", "alarm"])
        scores["date"] = pd.to_datetime(scores["date"])
        candidates = events[
            (events.label == info["label"])
            & events.asset.isin(scores.asset.unique())
        ]
        for asset, sg in scores.groupby("asset", sort=False):
            eg = candidates[candidates.asset == asset]
            for _, event in eg.iterrows():
                inside = sg[(sg.date >= event.start_date) & (sg.date <= event.end_date)]
                if inside.empty:
                    continue
                alarms = inside.loc[inside.alarm.astype(bool), "date"].sort_values()
                first = alarms.iloc[0] if len(alarms) else pd.NaT
                lead = ((event.start_date - first).days if pd.notna(first) else pd.NA)
                rows.append({
                    "event_id": event.event_id,
                    "label": event.label,
                    "asset": event.asset,
                    "crash_start": event.start_date,
                    "crash_end": event.end_date,
                    "duration_sessions": event.duration_sessions,
                    "model_key": info["model_key"],
                    "score_file": info["score_file"],
                    "feature_set": info["feature_set"],
                    "model": info["model"],
                    "seed": info["seed"],
                    "fold": info["fold"],
                    "score_rows_available": len(inside),
                    "detected": bool(len(alarms)),
                    "alarm_days_inside_crash": len(alarms),
                    "alarm_dates": ", ".join(x.strftime("%Y-%m-%d") for x in alarms),
                    "first_alarm_date": first,
                    "lead_days_calendar": lead,
                    "score_max_inside_crash": inside.score.max(),
                    "score_mean_inside_crash": inside.score.mean(),
                })
    ledger = pd.DataFrame(rows)
    if ledger.empty:
        raise RuntimeError("No saved score file overlapped an observed event")
    ledger.to_csv(OUT / "crash_model_event_ledger.csv", index=False)

    # One row per observed event with compact, human-readable detection details.
    compact = []
    for event_id, g in ledger.groupby("event_id", sort=False):
        first = g.iloc[0]
        detected = g[g.detected]
        missed = g[~g.detected]
        compact.append({
            "event_id": event_id,
            "label": first.label,
            "asset": first.asset,
            "crash_start": first.crash_start,
            "crash_end": first.crash_end,
            "duration_sessions": first.duration_sessions,
            "saved_models_covering_event": g.model_key.nunique(),
            "saved_models_detecting_event": detected.model_key.nunique(),
            "detection_fraction_of_saved_models": (
                detected.model_key.nunique() / g.model_key.nunique()
            ),
            "detecting_models": " || ".join(
                f"{r.model_key} [{r.alarm_dates}]" for _, r in detected.iterrows()
            ),
            "missed_models": " || ".join(missed.model_key.astype(str)),
            "earliest_alarm_date": detected.first_alarm_date.min()
            if len(detected) else pd.NaT,
            "earliest_lead_days": detected.lead_days_calendar.min()
            if len(detected) else pd.NA,
            "total_alarm_days_inside_event": int(detected.alarm_days_inside_crash.sum()),
        })
    compact_df = pd.DataFrame(compact)
    compact_df.to_csv(OUT / "crash_by_event.csv", index=False)

    summary = ledger.groupby(["label", "feature_set", "model", "seed", "fold"],
                             as_index=False).agg(
        events_covered=("event_id", "nunique"),
        events_detected=("detected", "sum"),
        mean_alarm_days_inside_crash=("alarm_days_inside_crash", "mean"),
        mean_lead_days_detected=("lead_days_calendar", "mean"),
    )
    summary["detection_rate"] = summary.events_detected / summary.events_covered
    summary.to_csv(OUT / "crash_model_ledger_summary.csv", index=False)

    lines = [
        "# Crash-by-crash model ledger",
        "",
        "This report answers: for each observed crash episode, what did each saved "
        "model do? It is not a model-average performance report.",
        "",
        "## How to read it",
        "",
        "- One observed crash is one `event_id`.",
        "- `detected=True` means at least one saved RQ alarm occurred inside that "
        "crash episode.",
        "- `alarm_dates` lists the exact alarm sessions inside the episode.",
        "- `lead_days_calendar` is the calendar-day difference between the first "
        "alarm and crash onset; alarms inside the crash have zero or negative "
        "lead, so this field is not a pre-crash lead metric.",
        "- `saved_models_covering_event` is the denominator actually available. "
        "An uncovered historical event is not counted as a model miss.",
        "",
        "## Coverage",
        "",
        f"- Observed event rows in full labels: {len(events):,}.",
        f"- Event/model rows produced from saved scores: {len(ledger):,}.",
        f"- Events with at least one saved model covering them: "
        f"{compact_df.event_id.nunique():,}.",
        "- The model ledger currently covers saved V7 folds only; it is not a "
        "full-history model ledger until all historical score files exist.",
        "",
        "## Per-event results",
        "",
        "The complete event-by-event table is in "
        "`data/v7_final/crash_ledger/crash_by_event.csv`.",
        "",
        "| Event | Label | Asset | Crash interval | Models covering | Models detecting | "
        "Detection fraction | Earliest alarm | Detecting models |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for _, r in compact_df.sort_values(["label", "crash_start"]).iterrows():
        lines.append(
            f"| {r.event_id} | {r.label} | {r.asset} | {r.crash_start} to "
            f"{r.crash_end} | {r.saved_models_covering_event} | "
            f"{r.saved_models_detecting_event} | "
            f"{r.detection_fraction_of_saved_models:.3f} | "
            f"{r.earliest_alarm_date} | {r.detecting_models} |"
        )
    lines += [
        "",
        "## Machine-readable files",
        "",
        "- `crash_model_event_ledger.csv`: one row per event and saved model.",
        "- `crash_by_event.csv`: one row per observed event covered by at least "
        "one saved score file.",
        "- `crash_model_ledger_summary.csv`: grouped counts for convenience.",
        "",
        "Events with no saved score overlap are deliberately absent from the "
        "per-event model table, rather than being incorrectly labeled as misses.",
    ]
    (ROOT / "CRASH_BY_CRASH_MODEL_LEDGER.md").write_text("\n".join(lines),
                                                          encoding="utf-8")
    print({"event_model_rows": len(ledger), "events_covered": compact_df.event_id.nunique(),
           "summary_rows": len(summary)})


if __name__ == "__main__":
    main()
