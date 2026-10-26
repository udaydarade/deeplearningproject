"""Summarise and visualise the completed benchmark grid.

Reads ``data/benchmark/all_results.csv`` (rows: fold x model x mode, overall
plus per-ticker breakdowns) and writes:

  data/benchmark/summary_overall.csv      level x dataset x mode x model means
  data/benchmark/summary_model.csv        model x mode ranking across datasets
  data/benchmark/ticker_breakdown.csv     per-ticker averages
  data/benchmark/figures/*.png            heatmaps and bar charts
  data/benchmark/analysis_report.txt      narrative summary

Run with ``py -3.12 analyze_benchmark.py`` after any benchmark run.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "benchmark"
FIGS = OUT / "figures"
METRICS = ["average_precision", "roc_auc", "brier", "precision", "recall", "f1"]


def load_results() -> pd.DataFrame:
    path = OUT / "all_results.csv"
    if not path.exists():
        raise SystemExit(f"No results yet: {path}. Run run_benchmark.py first.")
    return pd.read_csv(path)


def overall_table(results: pd.DataFrame) -> pd.DataFrame:
    """Mean test-fold metrics for every level x dataset x mode x model cell."""
    results = results[results["ticker"].isna()].copy()
    means = results.groupby(["level", "dataset", "mode", "model"])[METRICS].mean()
    folds = results.groupby(["level", "dataset", "mode", "model"])[["n", "positives", "prevalence"]].mean()
    return means.round(4).join(folds.round(3)).reset_index()


def heatmap(table: pd.DataFrame, rows, columns, value: str, ax, title: str):
    pivot = table.pivot(index=rows, columns=columns, values=value)
    image = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0,
                      vmax=max(0.02, np.nanmax(pivot.values)))
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value_cell = pivot.values[i, j]
            if np.isfinite(value_cell):
                ax.text(j, i, f"{value_cell:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    plt.colorbar(image, ax=ax, shrink=0.8)
    return pivot


def figures(aggregated: pd.DataFrame, results: pd.DataFrame) -> None:
    FIGS.mkdir(exist_ok=True)
    group_overall = aggregated[aggregated["level"] == "group"]

    # 1. group-level: mean AP across models per dataset x mode.
    means = (group_overall.groupby(["dataset", "mode"])["average_precision"].mean().reset_index())
    fig, ax = plt.subplots(figsize=(9, 6))
    heatmap(means, "dataset", "mode", "average_precision", ax,
            "Group level: mean test AP (avg over models & folds)")
    fig.tight_layout()
    fig.savefig(FIGS / "group_level_ap_heatmap.png", dpi=140)
    plt.close(fig)

    # 2. group-level: best model per dataset x mode.
    best = (group_overall.sort_values("average_precision")
            .groupby(["dataset", "mode"]).tail(1))
    fig, ax = plt.subplots(figsize=(9, 6))
    heatmap(best, "dataset", "mode", "average_precision", ax,
            "Group level: best-model test AP per dataset x mode")
    fig.tight_layout()
    fig.savefig(FIGS / "group_level_ap_best_model.png", dpi=140)
    plt.close(fig)

    # 3. model ranking per level: pooled mean AP by model x mode.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, level in zip(axes, ("group", "ticker")):
        pooled = (aggregated[aggregated["level"] == level]
                  .groupby(["model", "mode"])["average_precision"].mean().unstack())
        x = np.arange(len(pooled.index))
        width = 0.27
        for k, mode in enumerate(pooled.columns):
            ax.bar(x + (k - 1) * width, pooled[mode].values, width, label=mode)
        ax.set_xticks(x, pooled.index, rotation=30, ha="right")
        ax.set_title(f"{level} level: mean test AP by model x mode")
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "model_ranking.png", dpi=140)
    plt.close(fig)

    # 4. TDA-vs-classical lift per dataset (best model each).
    pivot = group_overall.pivot_table(index="dataset", columns="mode",
                                      values="average_precision", aggfunc="mean")
    mode_order = [m for m in ("classical", "tda", "combined", "landscape", "full")
                  if m in pivot.columns]
    fig, ax = plt.subplots(figsize=(10, 5))
    datasets = pivot.index
    width = 0.8 / max(len(mode_order), 1)
    for k, mode in enumerate(mode_order):
        ax.bar(np.arange(len(datasets)) + (k - len(mode_order) / 2) * width,
               pivot[mode].fillna(0).values, width, label=mode)
    ax.set_xticks(np.arange(len(datasets)), datasets, rotation=25, ha="right")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Mean test AP (avg over models & folds)")
    ax.set_title("Group level: TDA vs classical across the datasets")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "tda_vs_classical_per_dataset.png", dpi=140)
    plt.close(fig)

    # 5. ticker level: per-ticker heatmap of pooled-model mean AP.
    ticker_rows = results[(results["level"] == "ticker") & results["ticker"].notna()]
    if len(ticker_rows):
        per_ticker = (ticker_rows.groupby(["dataset", "ticker", "mode"])["average_precision"]
                      .mean().reset_index())
        per_ticker["row"] = per_ticker["dataset"] + " | " + per_ticker["ticker"]
        per_ticker["col"] = per_ticker["mode"]
        fig, ax = plt.subplots(figsize=(8, 14))
        heatmap(per_ticker, "row", "col", "average_precision", ax,
                "Ticker level: pooled-model mean test AP per ticker x mode")
        fig.tight_layout()
        fig.savefig(FIGS / "ticker_level_ap_heatmap.png", dpi=140)
        plt.close(fig)


def narrative(aggregated: pd.DataFrame) -> str:
    lines = ["Benchmark analysis report", ""]
    for level in ("group", "ticker"):
        sub = aggregated[aggregated["level"] == level]
        if not len(sub):
            continue
        prevalence = sub["prevalence"].mean()
        mode_names = list(sub["mode"].unique())
        lines += [f"## Level: {level}", f"Average crash prevalence: {prevalence:.3f}",
                  "", "| dataset | " + " | ".join(mode_names) + " | best model (mode) |",
                  "|---" * (len(mode_names) + 2) + "|"]
        for dataset in sub["dataset"].unique():
            part = sub[sub["dataset"] == dataset]
            means = {m: f"{part.loc[part['mode'] == m, 'average_precision'].mean():.4f}"
                     for m in mode_names}
            top = part.sort_values("average_precision", ascending=False).iloc[0]
            lines.append(f"| {dataset} | " + " | ".join(means[m] for m in mode_names)
                         + f" | {top['model']} ({top['mode']}) AP={top['average_precision']:.4f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    results = load_results()
    aggregated = overall_table(results)
    aggregated.to_csv(OUT / "summary_overall.csv", index=False)

    (aggregated.groupby(["mode", "model"])[METRICS].mean().round(4)
     .to_csv(OUT / "summary_model.csv"))

    ticker_breakdown = (results[results["ticker"].notna()]
                        .groupby(["level", "dataset", "ticker", "mode", "model"])[METRICS]
                        .mean().round(4).reset_index())
    ticker_breakdown.to_csv(OUT / "ticker_breakdown.csv", index=False)

    figures(aggregated, results)
    report = narrative(aggregated)
    (OUT / "analysis_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote summaries, figures, and report under {OUT}")


if __name__ == "__main__":
    main()
