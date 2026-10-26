"""Create dataset documentation and presentation figures from saved artifacts only."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "presentation_figs"
OUT.mkdir(exist_ok=True)
BLUE, CRIMSON, GOLD, TEAL, GREY = "#1769aa", "#b2182b", "#d98c00", "#16817a", "#4c566a"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 16, "axes.labelsize": 13, "legend.fontsize": 10})


def read(rel: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / rel)


def panel_summary() -> pd.DataFrame:
    specs = [
        ("Group A: US indices", "data/group_a_original_indices/returns.csv", "data/group_a_original_indices/crash_episodes.csv"),
        ("Group B: cross-asset", "data/group_b_cross_asset/returns.csv", "data/group_b_cross_asset/crash_episodes.csv"),
        ("Group C: crypto", "data/group_c_panel/crypto_returns.csv", "data/group_c_panel/crypto_crash_episodes.csv"),
        ("Group C: international", "data/group_c_panel/international_returns.csv", "data/group_c_panel/international_crash_episodes.csv"),
        ("Group C: sectors", "data/group_c_panel/sector_returns.csv", "data/group_c_panel/sector_crash_episodes.csv"),
    ]
    rows = []
    for name, returns_path, episodes_path in specs:
        r = read(returns_path)
        date_col = "date" if "date" in r.columns else r.columns[0]
        dates = pd.to_datetime(r[date_col])
        e = read(episodes_path)
        crash = pd.to_numeric(e.get("crash", 0), errors="coerce").fillna(0).astype(bool)
        rows.append({
            "dataset": name, "assets": len(r.columns) - 1 if date_col in r.columns else len(r.columns) - 1,
            "observations": len(r), "first_date": dates.min(), "last_date": dates.max(),
            "all_episodes": len(e), "crash_episodes": int(crash.sum()),
            "crash_start": pd.to_datetime(e.loc[crash, "start_date"]).min() if crash.any() else pd.NaT,
            "crash_end": pd.to_datetime(e.loc[crash, "start_date"]).max() if crash.any() else pd.NaT,
        })
    return pd.DataFrame(rows)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def coverage_figure(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    y = np.arange(len(summary))
    for i, row in summary.iterrows():
        ax.plot([row.first_date, row.last_date], [i, i], lw=12, solid_capstyle="round", color=BLUE)
        ax.scatter([row.crash_start, row.crash_end], [i, i], s=70, color=CRIMSON, zorder=3)
    ax.set_yticks(y, summary.dataset)
    ax.set_xlabel("Calendar date"); ax.set_title("Dataset history coverage and crash-episode span")
    ax.legend(handles=[
        plt.Line2D([0], [0], color=BLUE, lw=8, label="Available return history"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CRIMSON, label="First/last flagged crash", markersize=9),
    ], loc="lower right")
    ax.grid(axis="x", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "dataset_fig1_history_coverage.png")


def crash_count_figure(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(summary)); w = .38
    ax.bar(x - w / 2, summary.all_episodes, w, label="All detected drawdown episodes", color=GREY)
    ax.bar(x + w / 2, summary.crash_episodes, w, label="Crash episodes (crash = 1)", color=CRIMSON)
    ax.set_xticks(x, summary.dataset, rotation=20, ha="right"); ax.set_ylabel("Episode count")
    ax.set_title("Observed episodes in the source datasets"); ax.legend()
    for i, row in summary.iterrows():
        ax.text(i + w / 2, row.crash_episodes + max(summary.all_episodes) * .015, str(row.crash_episodes), ha="center", weight="bold")
    ax.grid(axis="y", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "dataset_fig2_observed_crash_counts.png")


def v7_figure() -> tuple[pd.DataFrame, pd.DataFrame]:
    e = read("data/v7_final/crash_stats/observed_crash_events.csv")
    e["start_date"] = pd.to_datetime(e["start_date"])
    totals = e.groupby("label").agg(events=("event_id", "nunique"), assets=("asset", "nunique"), first_onset=("start_date", "min"), last_onset=("start_date", "max")).reset_index()
    year = e.assign(year=e.start_date.dt.year).pivot_table(index="year", columns="label", values="event_id", aggfunc="nunique", fill_value=0)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for label, color in [("y_N5_p2", BLUE), ("y_N15_p2", CRIMSON)]:
        if label in year:
            ax.plot(year.index, year[label], marker="o", lw=2, color=color, label=label)
    ax.set_xlabel("Onset year"); ax.set_ylabel("Unique observed crash events"); ax.set_title("V7 full-history crash onsets by year"); ax.legend()
    ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    save(fig, "dataset_fig3_v7_crash_onsets_by_year.png")
    return totals, year


def documentation(summary: pd.DataFrame, totals: pd.DataFrame, year: pd.DataFrame) -> None:
    lines = [
        "# Dataset information for the Beamer presentation and IE 643 report",
        "",
        "## Scope and interpretation",
        "",
        "The project uses daily financial return panels to study crash detection and early warning. The source panels below are descriptive datasets; the saved benchmark/diagnostic files are the authoritative results used in the figures. No model is trained by the documentation script.",
        "",
        "Two crash-count definitions are reported deliberately:",
        "1. **Source-panel crash episodes**: rows in each panel's `crash_episodes.csv` with `crash == 1`. These are the historical episodes detected by the source-panel drawdown/tail rule.",
        "2. **V7 label events**: unique `(asset, start_date, end_date)` events in `data/v7_final/crash_stats/observed_crash_events.csv`. These are the full-history forward-label events used in V7; N=5 and N=15 are different label horizons and must not be added together.",
        "",
        "## Source datasets",
        "",
        "| Dataset | Assets/series | Daily observations | History | All episodes | Crash episodes (`crash=1`) | Crash episode date span |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(f"| {r.dataset} | {r.assets} | {r.observations:,} | {r.first_date:%Y-%m-%d} to {r.last_date:%Y-%m-%d} | {r.all_episodes:,} | {r.crash_episodes:,} | {r.crash_start:%Y-%m-%d} to {r.crash_end:%Y-%m-%d} |")
    lines += [
        "",
        "### What each panel contains",
        "",
        "- **Group A: US indices** — broad US equity indices (S&P 500, Dow Jones, Nasdaq, Russell 2000) used for the original TDA-vs-classical comparison.",
        "- **Group B: cross-asset** — liquid cross-asset instruments such as SPY, GLD, HYG, TLT, USO and UUP; used to test whether topology transfers beyond a single equity index.",
        "- **Group C: crypto** — cryptocurrency return series, including BTC-USD and other saved crypto assets; this panel has a shorter modern history.",
        "- **Group C: international** — international equity indices such as Shanghai, BSE Sensex, Brazil, FTSE, DAX, Hang Seng and Nikkei.",
        "- **Group C: sectors** — US sector ETFs; used to compare sector-level crash behavior.",
        "",
        "## V7 full-history event counts",
        "",
        "| Label horizon | Unique crash events | Assets | First onset | Last onset |",
        "|---|---:|---:|---|---|",
    ]
    for _, r in totals.iterrows():
        lines.append(f"| {r.label} | {r.events:,} | {r.assets:,} | {r.first_onset:%Y-%m-%d} | {r.last_onset:%Y-%m-%d} |")
    lines += [
        "",
        "The V7 report therefore contains **5,604 N=5 events across 159 assets** and **2,495 N=15 events across 155 assets**. The combined row count is 8,099 only as a bookkeeping total across two alternative horizons, not 8,099 independent crashes.",
        "",
        "The earlier EXP-3 lead-time summary reports **2,034 crash onsets across 39 tickers and 7 evaluation datasets** (`crypto`, `energy`, `financials`, `group_a`, `group_b`, `international`, `tech`). That is an evaluation subset/definition, not the full-history V7 count.",
        "",
        "## Major historical periods represented",
        "",
        "The panels cover the dot-com period, the 2007–09 global financial crisis, the 2011 stress period, the 2015–16 selloff, the March 2020 COVID crash, and the 2022 bear market where the underlying asset history permits. The exact first/last dates and counts in the tables above come directly from the saved CSV files.",
        "",
        "## Figure files",
        "",
        "- [dataset_fig1_history_coverage.png](./dataset_fig1_history_coverage.png) — calendar coverage and flagged-episode span.",
        "- [dataset_fig2_observed_crash_counts.png](./dataset_fig2_observed_crash_counts.png) — all source-panel episodes versus crash episodes.",
        "- [dataset_fig3_v7_crash_onsets_by_year.png](./dataset_fig3_v7_crash_onsets_by_year.png) — V7 full-history onset counts by year.",
        "",
        "Generated by `generate_dataset_information.py` from saved CSV artifacts; rerun it whenever the underlying ledgers change.",
    ]
    (OUT / "DATASET_INFORMATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = panel_summary()
    coverage_figure(summary)
    crash_count_figure(summary)
    totals, year = v7_figure()
    documentation(summary, totals, year)
    print(summary[["dataset", "crash_episodes"]].to_string(index=False))
    print(totals.to_string(index=False))


if __name__ == "__main__":
    main()
