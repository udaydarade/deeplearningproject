"""Qualitative + robustness analysis: do TDA signals warn EARLIER than CSD?

For every crash-label onset (0 -> 1 transition with a quiet PRE_ONSET
lead-in), each indicator family is compared on:

1. Discrimination - Mann-Whitney AUC of the family's past-only standardised
   (z) score over the 30 pre-onset sessions versus all quiet days. 0.5 = no
   warning content; 1.0 = the family is elevated before every crash.
2. Lead time - sessions between the first pre-onset crossing of the
   family's quiet-day 90th percentile (scanned over the 60 sessions before
   onset) and the onset date; -1 when the family never fires early.

Outputs under ``data/benchmark/qualitative/``:
  family_stats.csv           per dataset x ticker x family: AUC, lead, events
  family_stats_dataset.csv   per dataset x family aggregate
  event_aligned_<event>.png  family-average z curves anchored at onset
  leadtime_report.txt        narrative summary

Run with ``py -3.12 qualitative_leadtime.py`` after compute_features.py.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline import datasets

ROOT = datasets.ROOT
OUT = ROOT / "data" / "benchmark" / "qualitative"
LOOKBACK = 250      # past-only standardisation window
PRE_ONSET = 30      # pre-onset evaluation span (sessions)
SCAN_WINDOW = 60    # how far back the lead-time scan searches
MATCH_TOLERANCE = 45  # sessions allowed between a landmark date and an onset
FAMILIES = {
    "CSD": ["csd_ac1_w50", "csd_variance_w50", "csd_skew_w50", "csd_kurtosis_w50",
            "realized_vol_w20", "ewma_vol_w20", "csd_garch_vol"],
    "TDA-index": ["tda_idx_l1", "tda_idx_l2", "tda_idx_entropy", "tda_idx_totalpers"],
    "CROSS": ["csd_meanpaircorr_w50", "csd_maxeig_w50"],
    "TDA-joint": ["tda_joint_l1", "tda_joint_l2", "tda_joint_entropy",
                  "tda_joint_totalpers"],
}
EVENTS = {
    "dotcom_2000": "2000-04-01",
    "gfc_2008": "2008-09-15",
    "covid_2020": "2020-02-20",
    "rate_shock_2022": "2022-01-10",
}


def past_z(series: pd.Series) -> pd.Series:
    """Standardise each date using only the trailing LOOKBACK window."""
    rolling = series.rolling(LOOKBACK, min_periods=LOOKBACK // 2)
    return (series - rolling.mean()) / rolling.std().replace(0.0, np.nan)


def onsets(flags: np.ndarray) -> list[int]:
    """Episode starts: 0 -> 1 transitions with a quiet PRE_ONSET lead-in."""
    return [
        i for i in range(len(flags))
        if flags[i] and not flags[max(0, i - PRE_ONSET):i].any()
    ]


def mannwhitney_auc(pre: np.ndarray, quiet: np.ndarray) -> float:
    """U statistic / (n_pre * n_quiet): P(pre-value > quiet-value) + 0.5 P(tie)."""
    from scipy.stats import rankdata

    if len(pre) < 10 or len(quiet) < 50:
        return np.nan
    pooled = np.concatenate([pre, quiet])
    ranks = rankdata(pooled)
    u = ranks[: len(pre)].sum() - len(pre) * (len(pre) + 1) / 2.0
    return float(u / (len(pre) * len(quiet)))


def analyse_pair(rows: pd.DataFrame) -> list[dict]:
    """Per-family stats for one (dataset, ticker)."""
    labels = rows["label"].to_numpy(float)
    flags = np.nan_to_num(labels).astype(bool)
    starts = onsets(flags)
    if not starts:
        return []
    stats = []
    for family, candidates in FAMILIES.items():
        members = [m for m in candidates if m in rows.columns]
        if not members:
            continue
        z_table = rows[members].apply(past_z, axis=0)
        z_curve = z_table.mean(axis=1)
        quiet_z = z_curve[~flags].dropna()
        aucs, leads, fired = [], [], 0
        for start in starts:
            lo = max(0, start - PRE_ONSET)
            pre = z_curve.iloc[lo:start].dropna()
            auc = mannwhitney_auc(pre.to_numpy(), quiet_z.to_numpy())
            if not np.isfinite(auc):
                continue
            aucs.append(auc)
            threshold = float(np.nanquantile(quiet_z.to_numpy(), 0.90))
            scan = z_curve.iloc[max(0, start - SCAN_WINDOW):start].dropna()
            early = scan[scan > threshold]
            leads.append((start - early.index[0]) if len(early) else None)
            fired += int(len(early) > 0)
        if aucs:
            lead_values = [value for value in leads if value is not None]
            stats.append({
                "family": family,
                "auc_mean": float(np.mean(aucs)),
                "lead_mean": float(np.mean(lead_values)) if lead_values else np.nan,
                "lead_share": fired / len(leads),
                "events": len(aucs),
            })
    return stats


def aligned_curves(rows: pd.DataFrame, start: int) -> dict[str, pd.Series]:
    """Relative-session z curves for every family anchored at an onset."""
    flags = np.nan_to_num(rows["label"].to_numpy(float)).astype(bool)
    result = {}
    for family, candidates in FAMILIES.items():
        members = [m for m in candidates if m in rows.columns]
        if not members:
            continue
        z_curve = rows[members].apply(past_z, axis=0).mean(axis=1)
        lo = max(0, start - SCAN_WINDOW)
        segment = z_curve.iloc[lo:min(start + PRE_ONSET, len(z_curve))]
        rel = np.arange(lo, min(start + PRE_ONSET, len(z_curve)) ) - start
        result[family] = pd.Series(segment.to_numpy(), index=rel)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(ROOT / "data" / "benchmark" / "unified_panel.csv",
                        parse_dates=["date"])
    panel = panel.dropna(subset=["label"])
    stat_rows, curve_rows = [], []
    for dataset, group in panel.groupby("dataset"):
        for ticker, rows in group.groupby("ticker"):
            rows = rows.sort_values("date").reset_index(drop=True)
            stats = analyse_pair(rows)
            for record in stats:
                record.update(dataset=dataset, ticker=ticker)
                stat_rows.append(record)
            # Event-aligned curves matched to landmark dates.
            flags = np.nan_to_num(rows["label"].to_numpy(float)).astype(bool)
            starts = onsets(flags)
            dates = rows["date"].to_numpy()
            for event, anchor in EVENTS.items():
                anchor_ts = np.datetime64(anchor)
                if len(dates) == 0:
                    continue
                distances = np.abs(dates - anchor_ts) / np.timedelta64(1, "D")
                closest = int(np.argmin(distances))
                if distances.min() > MATCH_TOLERANCE / 1.5 or closest not in starts:
                    continue
                result = aligned_curves(rows, closest)
                for family, curve in result.items():
                    for rel, value in curve.items():
                        curve_rows.append({"event": event, "dataset": dataset,
                                           "ticker": ticker, "family": family,
                                           "rel": int(rel), "z": float(value)})

    stats_table = pd.DataFrame(stat_rows)
    stats_table.to_csv(OUT / "family_stats.csv", index=False)
    dataset_stats = (stats_table.groupby(["dataset", "family"])
                     [["auc_mean", "lead_mean", "lead_share", "events"]].mean().round(3))
    dataset_stats.to_csv(OUT / "family_stats_dataset.csv")

    # --- Event-aligned figures -------------------------------------------
    curves = pd.DataFrame(curve_rows)
    if len(curves):
        for event, subgroup in curves.groupby("event"):
            families = sorted(subgroup["family"].unique())
            datasets_present = sorted(subgroup["dataset"].unique())
            fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
            for ax, family in zip(axes.ravel(), families):
                for dataset in datasets_present:
                    part = subgroup[(subgroup["dataset"] == dataset)
                                    & (subgroup["family"] == family)]
                    mean = part.groupby("rel")["z"].mean()
                    ax.plot(mean.index, mean.values, label=dataset, lw=1.4)
                ax.axvline(0, color="black", lw=1)
                ax.axhline(0, color="black", lw=0.5, ls="--")
                ax.set_title(family)
                ax.grid(alpha=0.25)
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(f"{event}: family z-scores aligned to crash onset "
                         "(0 = onset; mean across tickers)")
            fig.tight_layout()
            fig.savefig(OUT / f"event_aligned_{event}.png", dpi=140)
            plt.close(fig)

    # --- Narrative --------------------------------------------------------
    overall = stats_table.groupby("family")[["auc_mean", "lead_mean", "lead_share"]].mean().round(3)
    lines = [
        "Lead-time / early-warning robustness report",
        f"Onsets analysed: {int(stats_table['events'].sum())} across "
        f"{stats_table.ticker.nunique()} tickers (family, dataset specific).",
        "Method: past-only rolling z-scores (250d), pre-onset window "
        f"{PRE_ONSET} sessions, quiet-day 90th-percentile alarm scan over "
        f"{SCAN_WINDOW} sessions prior to onset.",
        "",
        "Overall per family:",
        overall.to_string(),
        "",
        "Per dataset x family (mean AUC / mean lead in sessions / share of "
        "onsets fired early):",
        dataset_stats.to_string(),
        "",
        "Reading: AUC >> 0.5 means elevated before crashes; larger positive "
        "lead_mean with high lead_share means earlier and more consistent "
        "warnings. Compare CSD vs TDA-index vs TDA-joint per dataset.",
    ]
    (OUT / "leadtime_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote stats and {len(curves.groupby('event'))} event figures under {OUT}")


if __name__ == "__main__":
    main()
