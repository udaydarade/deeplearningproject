"""Generate the evidence-grounded visual assets requested for the final deck."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "assets" / "final"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#17263d"
TEAL = "#208c91"
BLUE = "#2d6a9f"
RED = "#c53b3b"
GRAY = "#5f6b7a"
LIGHT = "#eef3f6"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def prices(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()


def returns(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def asset_1() -> None:
    frame = prices(DATA / "group_a_original_indices" / "prices.csv")["^GSPC"].dropna()
    episodes = pd.read_csv(
        DATA / "group_a_original_indices" / "crash_episodes.csv",
        parse_dates=["start_date", "trough_date", "recovery_date"],
    )
    episodes = episodes[(episodes.ticker == "^GSPC") & episodes.crash.eq(1)]
    fig, ax = plt.subplots(figsize=(15, 6.4))
    ax.plot(frame.index, frame, color=NAVY, lw=1.0)
    events = [
        ("Dot-com", pd.Timestamp("2000-03-10"), pd.Timestamp("2002-10-09")),
        ("2008 financial crisis", pd.Timestamp("2007-10-09"), pd.Timestamp("2009-03-09")),
        ("COVID-19", pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23")),
    ]
    for name, start, end in events:
        present = episodes[(episodes.trough_date >= start) & (episodes.trough_date <= end)]
        if present.empty:
            continue
        ax.axvspan(start, end, color=RED, alpha=0.09)
        y = frame.loc[start:end].max() * 1.08
        ax.annotate(name, xy=(present.trough_date.iloc[0], frame.loc[present.trough_date.iloc[0]]),
                    xytext=(start, y), color=RED, fontsize=11,
                    arrowprops={"arrowstyle": "-", "color": RED, "lw": 0.8})
    ax.set_title("Market crashes are rare, heterogeneous, and regime-dependent", color=NAVY, pad=14)
    ax.set_ylabel("S&P 500 index level")
    ax.set_xlabel("Date")
    ax.grid(axis="y", alpha=0.2)
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "01_market_crashes_overview.png")


def box(ax, xy, width, height, text, fc=LIGHT, ec=NAVY, fs=10):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=fc, edgecolor=ec, linewidth=1.2
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center",
            fontsize=fs, color=NAVY, wrap=True)


def asset_2() -> None:
    fig = plt.figure(figsize=(16, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    labels = [
        "4 US equity indices\nS&P 500 · DJIA · NASDAQ · Russell 2000",
        "Daily log\nreturns",
        "Sliding\nwindow",
        "Point cloud\nin R⁴",
        "Vietoris–Rips\nfiltration",
        "Persistent H1",
        "Persistence\ndiagram",
        "Persistence\nlandscape",
        "Lᵖ norm",
        "Early-warning\nsignal",
    ]
    x0, y, w, h, gap = 0.015, 0.72, 0.082, 0.12, 0.018
    for i, label in enumerate(labels):
        x = x0 + i * (w + gap)
        box(ax, (x, y), w, h, label, fc="#e8f3f4" if i in (0, 9) else LIGHT, fs=9)
        if i < len(labels) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.003, y + h / 2),
                                         (x + w + gap - 0.003, y + h / 2),
                                         arrowstyle="->", mutation_scale=12,
                                         color=TEAL, linewidth=1.3))
    ax.text(0.5, 0.91, "Gidea–Katz (2018): methodological pipeline", ha="center",
            fontsize=18, color=NAVY, weight="bold")
    ax.text(0.5, 0.61, "Conceptual transformation (not an original-paper figure)", ha="center",
            fontsize=12, color=GRAY)
    # Inset: point cloud -> persistence diagram -> landscape.
    centers = [(0.20, 0.26), (0.50, 0.26), (0.80, 0.26)]
    box(ax, (0.10, 0.08), 0.20, 0.35, "Point cloud\n(window in R⁴;\nshown schematically)", fc="white")
    inset = fig.add_axes([0.12, 0.12, 0.16, 0.25])
    rng = np.random.default_rng(7)
    theta = np.linspace(0, 2 * np.pi, 60)
    inset.scatter(np.r_[np.cos(theta), rng.normal(0, 0.08, 35)],
                  np.r_[np.sin(theta), rng.normal(0, 0.08, 35)],
                  s=8, color=TEAL)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_frame_on(False)
    box(ax, (0.40, 0.08), 0.20, 0.35, "Persistence diagram\nbirth × death\nH1 bars highlighted", fc="white")
    inset2 = fig.add_axes([0.42, 0.12, 0.16, 0.25])
    inset2.plot([0, 1], [0, 1], "--", color="#aab4bf", lw=0.8)
    inset2.scatter([0.12, 0.23, 0.35, 0.46], [0.44, 0.63, 0.72, 0.88],
                   color=RED, s=20)
    inset2.set_xlabel("birth", fontsize=8)
    inset2.set_ylabel("death", fontsize=8)
    inset2.tick_params(labelsize=7)
    box(ax, (0.70, 0.08), 0.20, 0.35, "Persistence landscape\npiecewise-linear\nfunctions λ₁, λ₂, …", fc="white")
    inset3 = fig.add_axes([0.72, 0.12, 0.16, 0.25])
    x = np.linspace(0, 1, 300)
    inset3.plot(x, np.maximum(0, 1 - 2 * np.abs(x - 0.32)), color=TEAL, lw=2)
    inset3.plot(x, np.maximum(0, 0.6 - 2 * np.abs(x - 0.68)), color=RED, lw=1.6)
    inset3.set_xticks([])
    inset3.set_yticks([])
    inset3.spines[["top", "right"]].set_visible(False)
    for x in (0.30, 0.60):
        ax.add_patch(FancyArrowPatch((x, 0.255), (x + 0.09, 0.255),
                                     arrowstyle="->", mutation_scale=14, color=TEAL))
    save(fig, "02_gidea_katz_pipeline.pdf")


def asset_3() -> None:
    tda = pd.read_csv(DATA / "group_a_original_indices" / "tda_norms_paper_method.csv",
                      parse_dates=["date"])
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharey=False)
    events = [("Dot-com episode", pd.Timestamp("2000-03-10")),
              ("Lehman episode", pd.Timestamp("2008-09-15"))]
    for ax, (label, event) in zip(axes, events):
        sub = tda[(tda.date >= event - pd.tseries.offsets.BDay(1000)) &
                  (tda.date <= event + pd.tseries.offsets.BDay(30)) &
                  (tda.window == 50)]
        ax.plot(sub.date, sub.landscape_l1, color=TEAL, lw=1.2, label="landscape L1, w=50")
        sub100 = tda[(tda.date >= event - pd.tseries.offsets.BDay(1000)) &
                     (tda.date <= event + pd.tseries.offsets.BDay(30)) &
                     (tda.window == 100)]
        ax.plot(sub100.date, sub100.landscape_l1, color=BLUE, lw=1.0, alpha=0.8, label="landscape L1, w=100")
        ax.axvline(event, color=RED, ls="--", lw=1.2)
        ax.text(event, ax.get_ylim()[1] * 0.90, "event", color=RED, ha="right", fontsize=10)
        ax.set_title(label, loc="left", color=NAVY)
        ax.set_ylabel("L1 norm")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    axes[-1].set_xlabel("Date")
    fig.suptitle("Persistence-landscape norms around historical stress episodes", color=NAVY, y=0.99)
    fig.text(0.99, 0.01, "Source: project reproduction / reconstruction using stated methodology; descriptive, not OOS prediction.",
             ha="right", fontsize=8, color=GRAY)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save(fig, "03_gidea_katz_empirical_result.png")


def asset_4() -> None:
    episodes = pd.read_csv(DATA / "group_a_original_indices" / "crash_episodes.csv",
                           parse_dates=["trough_date"])
    episodes = episodes[episodes.ticker == "^GSPC"].sort_values("drawdown")
    values = episodes.drawdown.to_numpy()
    rate = float(episodes.bulk_rate.iloc[0])
    cutoff = float(episodes.crash_cutoff.iloc[0])
    n = len(values)
    survival = 1 - np.arange(1, n + 1) / (n + 1)
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    bulk = values[values < cutoff]
    tail = values[values >= cutoff]
    ax.semilogy(bulk * 100, survival[:len(bulk)], "o", ms=4, color=BLUE, label="empirical bulk")
    ax.semilogy(tail * 100, survival[len(bulk):], "o", ms=7, color=RED, label="extreme drawdowns / outliers")
    x = np.linspace(max(values.min(), 1e-5), max(values) * 1.03, 400)
    ax.semilogy(x * 100, np.exp(-rate * x), color=TEAL, lw=2, label="fitted tail model")
    ax.axvline(cutoff * 100, color=RED, ls="--", lw=1.4)
    ax.text(cutoff * 100 + 0.5, 0.35, f"project cutoff = {cutoff:.1%}\nα = 0.05 / {n}",
            color=RED, fontsize=10)
    ax.set_xlabel("drawdown size (%)")
    ax.set_ylabel("P(D > d)")
    ax.set_title("Drawdown outliers relative to the fitted exponential bulk", color=NAVY)
    ax.grid(which="both", axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.text(0.99, 0.01, "Johansen–Sornette-inspired diagnostic; project cutoff uses Bonferroni control, not the original criterion.",
             ha="right", fontsize=8, color=GRAY)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save(fig, "04_johansen_drawdown_outliers.png")


def asset_5() -> None:
    group_a = returns(DATA / "group_a_original_indices" / "returns.csv")
    group_b = returns(DATA / "group_b_cross_asset" / "returns.csv")
    a_cols = ["^DJI", "^GSPC", "^IXIC", "^RUT"]
    b_cols = ["SPY", "GLD", "TLT", "HYG", "USO", "UUP"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.25]})
    for ax, frame, cols, title in zip(axes, [group_a, group_b], [a_cols, b_cols],
                                      ["Group A: original US indices", "Cross-asset universe"]):
        corr = frame[cols].corr()
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(cols)), [c.replace("^", "") for c in cols], rotation=45, ha="right")
        ax.set_yticks(range(len(cols)), [c.replace("^", "") for c in cols])
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                        fontsize=9, color="white" if abs(corr.iloc[i, j]) > 0.55 else NAVY)
        ax.set_title(title, color=NAVY)
    cbar = fig.colorbar(im, ax=axes, shrink=0.82)
    cbar.set_label("Pearson correlation")
    fig.suptitle("The original indices are closely related; cross-asset markets add structure", color=NAVY)
    fig.tight_layout(rect=[0, 0, 0.94, 0.95])
    save(fig, "05_asset_correlation_heatmap.png")


def asset_6() -> None:
    p = prices(DATA / "group_a_original_indices" / "prices.csv")
    e = pd.read_csv(DATA / "group_a_original_indices" / "crash_episodes.csv",
                    parse_dates=["start_date", "trough_date", "recovery_date"])
    fig, axes = plt.subplots(3, 2, figsize=(16, 11), sharex="col")
    for col, ticker in enumerate(["^DJI", "^GSPC"]):
        s = p[ticker].dropna()
        ep = e[e.ticker == ticker]
        crashes = ep[ep.crash.eq(1)]
        cutoff = float(ep.crash_cutoff.iloc[0])
        dd = 1 - s / s.cummax()
        axes[0, col].plot(s.index, s / s.iloc[0] * 100, color=NAVY, lw=0.85)
        axes[1, col].plot(dd.index, dd * 100, color=TEAL, lw=0.85)
        axes[1, col].axhline(cutoff * 100, color=RED, ls="--", lw=1.2)
        axes[2, col].scatter(ep.trough_date, ep.drawdown * 100, color=BLUE, s=13)
        axes[2, col].scatter(crashes.trough_date, crashes.drawdown * 100, color=RED, s=42, zorder=3)
        for _, row in crashes.iterrows():
            axes[0, col].axvspan(row.start_date, row.trough_date, color=RED, alpha=0.10)
            axes[1, col].axvspan(row.start_date, row.trough_date, color=RED, alpha=0.10)
        axes[2, col].axhline(cutoff * 100, color=RED, ls="--", lw=1.2)
        axes[0, col].set_title("DJIA" if ticker == "^DJI" else "S&P 500", color=NAVY)
        axes[0, col].set_ylabel("normalized price")
        axes[1, col].set_ylabel("drawdown (%)")
        axes[2, col].set_ylabel("episode drawdown (%)")
        axes[2, col].set_xlabel("Date")
        for row in range(3):
            axes[row, col].grid(axis="y", alpha=0.18)
            axes[row, col].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Objective crash labels: high-water-mark drawdowns and calibrated outliers", color=NAVY, y=0.995)
    fig.text(0.5, 0.01, "Red markers/regions are statistically identified crashes; dashed lines are series-specific exponential cutoffs.",
             ha="center", fontsize=9, color=GRAY)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    save(fig, "06_objective_crash_labels.png")


def asset_7() -> None:
    fig = plt.figure(figsize=(14, 7.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.91, "TDA example: raw intermediate geometry is not saved in this project",
            ha="center", fontsize=18, color=NAVY, weight="bold")
    ax.text(0.5, 0.84, "Only the methodology diagram can be generated without inventing a point cloud or persistence diagram.",
            ha="center", fontsize=11, color=RED)
    labels = ["return window\n(4D point cloud)", "Vietoris–Rips\nfiltration", "persistence\nH1", "landscape\nand Lp norm"]
    for i, label in enumerate(labels):
        x = 0.05 + i * 0.24
        box(ax, (x, 0.49), 0.17, 0.17, label, fc="#e8f3f4", fs=12)
        if i < 3:
            ax.add_patch(FancyArrowPatch((x + 0.18, 0.575), (x + 0.235, 0.575),
                                         arrowstyle="->", mutation_scale=14, color=TEAL))
    ax.text(0.5, 0.33, "Available evidence: tda_norms.csv contains window-level L1/L2 norms,",
            ha="center", fontsize=11, color=GRAY)
    ax.text(0.5, 0.29, "but no saved raw 4D windows, persistence diagrams, barcodes, or landscape functions.",
            ha="center", fontsize=11, color=GRAY)
    save(fig, "07_tda_example.png")


def asset_8_report() -> None:
    path = OUT / "08_event_aligned_indicators_report.txt"
    path.write_text(
        "ASSET 8 NOT GENERATED AS A RESULTS FIGURE\n"
        "=========================================\n\n"
        "The repository does not contain a common-label event-aligned table with\n"
        "TDA L1, TDA L2, AC1, variance, and MPS on a shared -60..+20 crash-day\n"
        "axis. Existing outputs are full-history indicator series and separate\n"
        "episode diagnostics, not a common event-aligned panel.\n\n"
        "No event-aligned figure was fabricated. A valid figure requires a\n"
        "predefined event list, one common trading-day alignment, and a documented\n"
        "normalization/aggregation rule before plotting median traces or bands.\n",
        encoding="utf-8",
    )


def main() -> None:
    asset_1()
    asset_2()
    asset_3()
    asset_4()
    asset_5()
    asset_6()
    asset_7()
    asset_8_report()
    print(f"Generated final assets in {OUT}")


if __name__ == "__main__":
    main()
