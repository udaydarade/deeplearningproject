"""Create presentation-ready Group A figures for the TDA crash detector.

This script uses only the four original Group A indices and the paper-method
TDA output. It makes the decision chain visible:
returns -> Rips filtration (epsilon) -> H1 bars -> landscape norms ->
objective crash label and an out-of-sample-style rolling threshold.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from ripser import ripser


ROOT = Path(__file__).resolve().parent
GROUP = ROOT / "data" / "group_a_original_indices"
OUT = GROUP / "presentation"
RETURNS = GROUP / "returns.csv"
PRICES = GROUP / "prices.csv"
TDA = GROUP / "tda_norms_paper_method.csv"
CSD = GROUP / "csd_indicators_extended.csv"
LABELS = GROUP / "labels_N15.csv"
SUMMARY = GROUP.parent.parent / "crash_labels_summary.csv"
WINDOW = 50
BASELINE = 100
PERIODS = {
    "Dot-com": ("2000-03-10", "2000-10-31"),
    "GFC / Lehman": ("2007-10-01", "2009-03-31"),
    "2011 stress": ("2011-07-01", "2011-10-31"),
    "2015–16 selloff": ("2015-08-01", "2016-02-29"),
    "COVID": ("2020-02-01", "2020-03-31"),
    "2022 bear": ("2022-01-01", "2022-12-31"),
}


def threshold(series: pd.Series) -> pd.Series:
    baseline_mean = series.rolling(BASELINE, min_periods=BASELINE).mean().shift(1)
    baseline_std = series.rolling(BASELINE, min_periods=BASELINE).std(ddof=1).shift(1)
    return baseline_mean + 1.5 * baseline_std


def episodes(label_frame: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    labels = label_frame.set_index("date")["label"].astype(bool).sort_index()
    starts = labels.index[labels & ~labels.shift(1, fill_value=False)]
    ends = labels.index[labels & ~labels.shift(-1, fill_value=False)]
    return list(zip(starts, ends))


def normalized(series: pd.Series) -> pd.Series:
    return series / series.iloc[0] * 100


def dashboard(prices: pd.Series, tda: pd.DataFrame, csd: pd.DataFrame, labels: pd.DataFrame) -> None:
    tda50 = tda[tda.window == WINDOW].set_index("date").sort_index()
    csd50 = csd[csd.window == WINDOW].set_index("date").sort_index()
    joined = pd.DataFrame({
        "price": prices,
        "tda_l1": tda50.landscape_l1,
        "tda_l2": tda50.landscape_l2,
        "h1_bars": tda50.h1_finite_bars,
        "variance": csd50.rolling_variance,
        "garch": csd50.garch_conditional_volatility,
    }).dropna()
    joined["tda_threshold"] = threshold(joined.tda_l1)
    joined["variance_threshold"] = threshold(joined.variance)
    joined["garch_threshold"] = threshold(joined.garch)
    ep = episodes(labels[labels.ticker == "^GSPC"])

    fig, axes = plt.subplots(4, 1, figsize=(18, 15), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.4, 1.2, 1.4]})
    axes[0].plot(joined.index, normalized(joined.price), color="black", lw=1.1, label="S&P 500 (start = 100)")
    axes[0].set_ylabel("Price")
    axes[0].set_title("How the four-index TDA detector changes before objective crash labels", loc="left", weight="bold")
    axes[1].plot(joined.index, joined.tda_l1, color="#1464a5", label="TDA landscape L1")
    axes[1].plot(joined.index, joined.tda_l2, color="#f28e2b", alpha=.8, label="TDA landscape L2")
    axes[1].plot(joined.index, joined.tda_threshold, color="#1464a5", ls="--", label="TDA threshold: trailing 100d mean + 1.5σ")
    axes[1].set_ylabel("Landscape norm")
    axes[2].plot(joined.index, joined.h1_bars, color="#7b2cbf", label="Finite H1 bars (Betti-1 features)")
    axes[2].set_ylabel("H1 bar count")
    axes[3].plot(joined.index, joined.variance, color="#2a9d8f", label="Rolling variance")
    axes[3].plot(joined.index, joined.variance_threshold, color="#2a9d8f", ls="--", alpha=.8, label="Variance threshold")
    axes[3].plot(joined.index, joined.garch, color="#e76f51", label="GARCH(1,1) volatility")
    axes[3].plot(joined.index, joined.garch_threshold, color="#e76f51", ls="--", alpha=.8, label="GARCH threshold")
    axes[3].set_ylabel("CSD signals")
    for ax in axes:
        for start, end in ep:
            ax.axvspan(start, end, color="red", alpha=.08)
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
    axes[-1].set_xlabel("Date; red spans = objective S&P 500 N=15, 2nd-percentile drawdown episodes")
    fig.text(.01, .01, "Detector logic: raw 4D return window → Euclidean Vietoris–Rips filtration → finite H1 bars → exact landscape norms → rolling threshold crossing.",
             fontsize=10, color="#333333")
    fig.tight_layout(rect=(0, .025, 1, 1))
    fig.savefig(OUT / "01_detection_dashboard.png", dpi=180)
    plt.close(fig)


def rips_filter_figure(returns: pd.DataFrame, tda: pd.DataFrame) -> None:
    chosen = [
        ("Pre-dot-com", pd.Timestamp("1999-09-01")),
        ("Dot-com stress", pd.Timestamp("2000-03-10")),
        ("Pre-Lehman", pd.Timestamp("2008-04-16")),
        ("Lehman stress", pd.Timestamp("2008-09-15")),
        ("COVID stress", pd.Timestamp("2020-03-16")),
        ("2022 bear", pd.Timestamp("2022-06-16")),
    ]
    values = returns.to_numpy(float)
    dates = returns.index
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (label, target) in zip(axes.flat, chosen):
        end = int(np.searchsorted(dates.values, target.to_datetime64(), side="right"))
        block = values[end - WINDOW : end]
        diagram = ripser(block, maxdim=1, metric="euclidean")["dgms"][1]
        bars = diagram[np.isfinite(diagram).all(axis=1)]
        if len(bars):
            for row, (birth, death) in enumerate(bars):
                ax.plot([birth, death], [row, row], lw=3, color="#1464a5")
                ax.scatter([birth, death], [row, row], s=18, color="#111111", zorder=3)
            ax.set_ylim(-1, len(bars))
        ax.set_title(f"{label}\n{target.date()} | finite H1 bars = {len(bars)}", fontsize=10)
        ax.set_xlabel("Rips scale ε (birth → death)")
        ax.set_ylabel("H1 feature rank")
        ax.grid(alpha=.2)
    fig.suptitle("The Rips filter made visible: loops are born and die as ε increases", weight="bold", y=.99)
    fig.text(.5, .01, "Each horizontal segment is one H1 loop. Longer segments have greater persistence and contribute more to the landscape norms.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.savefig(OUT / "02_rips_filter_evolution.png", dpi=180)
    plt.close(fig)


def scorecard(tda: pd.DataFrame, csd: pd.DataFrame, labels: pd.DataFrame) -> None:
    eps = episodes(labels[labels.ticker == "^GSPC"])
    t = tda[tda.window == WINDOW].set_index("date").landscape_l1
    c = csd[csd.window == WINDOW].set_index("date")
    rows = []
    for i, (start, end) in enumerate(eps, 1):
        search_start = start - pd.tseries.offsets.BDay(5)
        search_end = end + pd.tseries.offsets.BDay(5)
        row = {"episode": f"E{i} {start.date()}", "start": start}
        for name, series in (("TDA L1", t), ("CSD variance", c.rolling_variance), ("CSD GARCH", c.garch_conditional_volatility)):
            s = series.dropna()
            prior = s[s.index < search_start].tail(BASELINE)
            search = s[(s.index >= prior.index.max() if len(prior) else search_start) & (s.index <= search_end)]
            if len(prior) < BASELINE:
                row[name] = np.nan
            else:
                limit = prior.mean() + 1.5 * prior.std(ddof=1)
                hits = search[search > limit]
                row[name] = (start - hits.index[0]).days if len(hits) else np.nan
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("episode")
    fig, ax = plt.subplots(figsize=(14, max(5, len(frame) * .42)))
    y = np.arange(len(frame))
    for col, color in (("TDA L1", "#1464a5"), ("CSD variance", "#2a9d8f"), ("CSD GARCH", "#e76f51")):
        vals = frame[col].to_numpy(float)
        mask = np.isfinite(vals)
        ax.scatter(vals[mask], y[mask], s=70, label=col, color=color)
        ax.scatter(np.full((~mask).sum(), 0), y[~mask], marker="x", s=45, color=color, alpha=.7)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, frame.index)
    ax.set_xlabel("Trading-day lead time (positive = detected before episode start; × = no detection)")
    ax.set_title("Crash detection scorecard: same Group A data, same 50-day method", loc="left", weight="bold")
    ax.grid(axis="x", alpha=.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "03_detection_scorecard.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    old = GROUP / "paper_extension_plots"
    if old.exists():
        shutil.rmtree(old)
    returns = pd.read_csv(RETURNS, parse_dates=["date"]).set_index("date").sort_index()[["^GSPC", "^DJI", "^IXIC", "^RUT"]].dropna()
    prices = pd.read_csv(PRICES, parse_dates=["date"]).set_index("date").sort_index()["^GSPC"].dropna()
    tda = pd.read_csv(TDA, parse_dates=["date"])
    csd = pd.read_csv(CSD, parse_dates=["date"])
    labels = pd.read_csv(LABELS, parse_dates=["date"])
    dashboard(prices, tda, csd, labels)
    rips_filter_figure(returns, tda)
    scorecard(tda, csd, labels)
    (OUT / "README.txt").write_text(
        "Presentation figures use only Group A. Red spans are objective N=15, 2nd-percentile S&P 500 drawdown episodes. "
        "Dashed lines are rolling 100-observation mean + 1.5 standard deviations, used only as a transparent visualization "
        "of a prospective alert rule. The paper itself additionally quantified rising low-frequency spectral density over 250 days.",
        encoding="utf-8",
    )
    print(f"Wrote presentation figures to {OUT}")


if __name__ == "__main__":
    main()
