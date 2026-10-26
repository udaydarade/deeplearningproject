"""Build a strictly pre-onset crash detection ledger from saved V7 scores.

An alarm is a valid prediction only when its trading-session date is strictly
before the observed crash onset.  Results are reported for several explicit
lookback windows so the lead-time claim is reproducible rather than implicit.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
V7 = ROOT / "data/v7_final"
STATS = V7 / "crash_stats"
OUT = V7 / "precrash_ledger"
WINDOWS = (1, 5, 10, 20, 60)


def meta(path: Path) -> dict[str, str]:
    match = re.match(
        r"(?P<label>y_(?:N5|down_N15)_p2)_(?P<fold>[^_]+_[^_]+|2021_plus)_(?P<rest>.*)",
        path.stem,
    )
    if not match:
        raise ValueError(f"Unrecognized score filename: {path.name}")
    bits = match.group("rest").rsplit("_", 2)
    return {
        "score_file": path.name,
        "label": match.group("label"),
        "fold": match.group("fold"),
        "feature_set": bits[0],
        "model": bits[1],
        "seed": bits[2],
        "model_key": (
            f"{bits[0]}|{bits[1]}|seed={bits[2]}|fold={match.group('fold')}"
        ),
    }


def prior_sessions(dates: pd.Series, onset: pd.Timestamp, window: int) -> pd.Series:
    """Return the last ``window`` available sessions strictly before onset."""
    before = dates[dates < onset].drop_duplicates().sort_values()
    return before.tail(window)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(
        STATS / "observed_crash_events.csv",
        parse_dates=["start_date", "end_date"],
    ).rename(columns={"start_date": "crash_start", "end_date": "crash_end"})
    events = events.set_index("event_id", drop=False)
    rows: list[dict[str, object]] = []

    for path in sorted((V7 / "scores").glob("*.parquet")):
        info = meta(path)
        scores = pd.read_parquet(
            path, columns=["date", "asset", "y", "score", "alarm"]
        )
        scores["date"] = pd.to_datetime(scores["date"])
        scores["alarm"] = scores["alarm"].fillna(False).astype(bool)
        candidates = events[
            (events.label == info["label"])
            & events.asset.isin(scores.asset.unique())
        ]
        for asset, asset_scores in scores.groupby("asset", sort=False):
            asset_scores = asset_scores.sort_values("date").drop_duplicates("date")
            asset_events = candidates[candidates.asset == asset]
            for _, event in asset_events.iterrows():
                available_prior = asset_scores[asset_scores.date < event.crash_start]
                if available_prior.empty:
                    continue
                row: dict[str, object] = {
                    "event_id": event.event_id,
                    "label": event.label,
                    "asset": asset,
                    "crash_start": event.crash_start,
                    "crash_end": event.crash_end,
                    "duration_sessions": event.duration_sessions,
                    "model_key": info["model_key"],
                    "score_file": info["score_file"],
                    "feature_set": info["feature_set"],
                    "model": info["model"],
                    "seed": info["seed"],
                    "fold": info["fold"],
                    "prior_score_rows_available": len(available_prior),
                }
                for window in WINDOWS:
                    window_dates = prior_sessions(
                        asset_scores.date, event.crash_start, window
                    )
                    in_window = asset_scores[
                        asset_scores.date.isin(window_dates)
                    ]
                    alarms = in_window.loc[in_window.alarm, "date"].sort_values()
                    prefix = f"w{window}"
                    first = alarms.iloc[0] if len(alarms) else pd.NaT
                    row[f"{prefix}_sessions_available"] = len(in_window)
                    row[f"{prefix}_detected"] = bool(len(alarms))
                    row[f"{prefix}_alarm_count"] = len(alarms)
                    row[f"{prefix}_alarm_dates"] = ", ".join(
                        value.strftime("%Y-%m-%d") for value in alarms
                    )
                    row[f"{prefix}_first_alarm_date"] = first
                    row[f"{prefix}_lead_sessions"] = (
                        len(window_dates)
                        - 1
                        - window_dates.tolist().index(first)
                        if pd.notna(first)
                        else pd.NA
                    )
                    row[f"{prefix}_max_score"] = in_window.score.max()
                rows.append(row)

    ledger = pd.DataFrame(rows)
    if ledger.empty:
        raise RuntimeError("No saved score file had pre-onset score coverage")
    ledger.to_csv(OUT / "precrash_model_event_ledger.csv", index=False)

    compact_rows: list[dict[str, object]] = []
    for event_id, group in ledger.groupby("event_id", sort=False):
        first = group.iloc[0]
        compact: dict[str, object] = {
            "event_id": event_id,
            "label": first.label,
            "asset": first.asset,
            "crash_start": first.crash_start,
            "crash_end": first.crash_end,
            "duration_sessions": first.duration_sessions,
            "saved_models_with_preonset_coverage": group.model_key.nunique(),
        }
        for window in WINDOWS:
            prefix = f"w{window}"
            detected = group[group[f"{prefix}_detected"]]
            compact[f"{prefix}_models_detected"] = detected.model_key.nunique()
            compact[f"{prefix}_detection_fraction"] = (
                detected.model_key.nunique() / group.model_key.nunique()
            )
            compact[f"{prefix}_detecting_models"] = " || ".join(
                f"{r.model_key} [{r[f'{prefix}_alarm_dates']}]"
                for _, r in detected.iterrows()
            )
            compact[f"{prefix}_earliest_alarm"] = (
                detected[f"{prefix}_first_alarm_date"].min()
                if len(detected)
                else pd.NaT
            )
            compact[f"{prefix}_earliest_lead_sessions"] = (
                detected[f"{prefix}_lead_sessions"].max()
                if len(detected)
                else pd.NA
            )
        compact_rows.append(compact)
    compact_df = pd.DataFrame(compact_rows)
    compact_df.to_csv(OUT / "precrash_by_event.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    group_cols = ["label", "feature_set", "model", "seed", "fold"]
    for keys, group in ledger.groupby(group_cols, sort=False):
        base = dict(zip(group_cols, keys))
        base["events_with_preonset_coverage"] = group.event_id.nunique()
        for window in WINDOWS:
            prefix = f"w{window}"
            base[f"{prefix}_events_detected"] = int(
                group.groupby("event_id")[f"{prefix}_detected"].any().sum()
            )
            base[f"{prefix}_event_recall"] = (
                base[f"{prefix}_events_detected"]
                / base["events_with_preonset_coverage"]
            )
            base[f"{prefix}_mean_alarm_count"] = group[
                f"{prefix}_alarm_count"
            ].mean()
            base[f"{prefix}_mean_preonset_alarm_fraction"] = (
                group[f"{prefix}_alarm_count"]
                / group[f"{prefix}_sessions_available"].clip(lower=1)
            ).mean()
            leads = group.loc[
                group[f"{prefix}_detected"], f"{prefix}_lead_sessions"
            ]
            base[f"{prefix}_mean_lead_sessions_when_detected"] = (
                leads.mean() if len(leads) else pd.NA
            )
        summary_rows.append(base)
    pd.DataFrame(summary_rows).to_csv(
        OUT / "precrash_model_summary.csv", index=False
    )

    full_events = len(events)
    covered_events = compact_df.event_id.nunique()
    lines = [
        "# Strict pre-crash model ledger",
        "",
        "This is the corrected forecasting ledger. A model is counted as "
        "detecting a crash only when its RQ alarm occurs strictly before the "
        "crash onset. Alarms on or after `crash_start` are never detections.",
        "",
        "## Window definitions",
        "",
        "For each event, `wK` is the last K available trading sessions before "
        "the onset date. The onset session itself is excluded. Results are "
        "reported for K = 1, 5, 10, 20, and 60; this makes the operational "
        "lead-time assumption explicit.",
        "",
        "## Coverage",
        "",
        f"- Observed events in the full label population: {full_events:,}.",
        f"- Events with at least one saved-model pre-onset coverage row: "
        f"{covered_events:,}.",
        f"- Event/model rows with pre-onset coverage: {len(ledger):,}.",
        "- Events without saved score coverage are excluded from the model "
        "denominator, not silently labeled as misses.",
        "- These results cover the saved V7 score files only; they are not "
        "full-history model results until historical V7 score folds are run.",
        "",
        "## Key interpretation",
        "",
        "- `wK_detected=True` means at least one alarm occurred in the K "
        "trading-session pre-onset window.",
        "- `lead_sessions` is the number of trading sessions between the alarm "
        "and onset within that window.",
        "- Alarms inside the crash interval are retained only in the legacy "
        "audit ledger and do not count here.",
        "",
        "## Complete files",
        "",
        "- `data/v7_final/precrash_ledger/precrash_model_event_ledger.csv`: "
        "one row per event/model with all strict pre-onset windows.",
        "- `data/v7_final/precrash_ledger/precrash_by_event.csv`: one row per "
        "event with the models that forecast it before onset.",
        "- `data/v7_final/precrash_ledger/precrash_model_summary.csv`: grouped "
        "event-recall and lead-time summary.",
    ]
    (ROOT / "PRECRASH_MODEL_LEDGER.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
