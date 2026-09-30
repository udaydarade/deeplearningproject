"""Paper-style Group A figures using the verified Gidea-Katz method.

Each figure uses raw daily log-return point clouds, Euclidean Vietoris-Rips
filtrations, finite H1 bars, and exact persistence-landscape norms for w=50
and w=100. No crash window is shaded or otherwise highlighted. The four
indices are shown together so the reader can see the market inputs and the
joint topological signal change over the same period.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
GROUP = ROOT / "data" / "group_a_original_indices"
OUT = GROUP / "paper_style_figures"
PRICES = GROUP / "prices.csv"
NORMS = GROUP / "tda_norms_paper_method.csv"
INDEXES = ["^GSPC", "^DJI", "^IXIC", "^RUT"]
NAMES = {"^GSPC": "S&P 500", "^DJI": "DJIA", "^IXIC": "NASDAQ", "^RUT": "Russell 2000"}

# Original paper dates and the five post-paper periods requested previously.
PERIODS = {
    "original_two_cases": [
        ("Dot-com crash (2000-03-10)", "2000-03-10"),
        ("Lehman bankruptcy (2008-09-15)", "2008-09-15"),
    ],
    "2011_downgrade": [("2011 stress", "2011-08-01")],
    "2015_2016_selloff": [("2015–16 selloff", "2015-08-24")],
    "2018_selloff": [("December 2018 selloff", "2018-12-24")],
    "2020_covid": [("COVID crash (2020)", "2020-03-16")],
    "2022_bear": [("2022 bear market", "2022-06-16")],
}


def plot_case(ax_price, ax_norm, prices, norms, title, event_date):
    event = pd.Timestamp(event_date)
    end_pos = prices.index.searchsorted(event, side="right")
    start_pos = max(0, end_pos - 1000)
    start = prices.index[start_pos]
    finish = min(prices.index[-1], event + pd.Timedelta(days=120))
    price_slice = prices.loc[start:finish]
    norm_slice = norms[(norms.date >= start) & (norms.date <= finish)]

    normalized = price_slice[INDEXES].div(price_slice[INDEXES].iloc[0]).mul(100)
    for ticker in INDEXES:
        ax_price.plot(normalized.index, normalized[ticker], lw=1.0, label=NAMES[ticker])
    ax_price.set_title(title, fontsize=11)
    ax_price.set_ylabel("Index level (start = 100)")
    ax_price.grid(alpha=0.2)

    for window, linestyle in ((50, "-"), (100, "--")):
        subset = norm_slice[norm_slice.window == window]
        ax_norm.plot(subset.date, subset.landscape_l1, lw=1.0, linestyle=linestyle,
                     label=f"L1, w={window}")
        ax_norm.plot(subset.date, subset.landscape_l2, lw=1.0, linestyle=linestyle,
                     alpha=0.55, label=f"L2, w={window}")
    ax_norm.set_ylabel("Landscape norm")
    ax_norm.set_xlabel("Date")
    ax_norm.grid(alpha=0.2)
    ax_price.set_xlim(start, finish)
    ax_norm.set_xlim(start, finish)


def make_original_figure(prices, norms):
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), squeeze=False)
    for column, (title, event) in enumerate(PERIODS["original_two_cases"]):
        plot_case(axes[0, column], axes[1, column], prices, norms, title, event)
        axes[0, column].legend(fontsize=8, loc="best")
        axes[1, column].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Gidea–Katz method: four US indices and persistence-landscape norms",
        fontsize=16, weight="bold",
    )
    fig.text(
        0.5, 0.01,
        "Each column spans the 1,000 trading days before the named paper event and approximately 120 days after. "
        "Solid/dashed lines distinguish w=50/w=100; no event shading or event marker is used.",
        ha="center", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUT / "original_two_cases.png", dpi=180)
    plt.close(fig)


def make_single_figure(prices, norms, filename, title, event):
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    plot_case(axes[0], axes[1], prices, norms, title, event)
    axes[0].legend(ncol=4, fontsize=9, loc="best")
    axes[1].legend(ncol=4, fontsize=9, loc="best")
    fig.suptitle(
        "Gidea–Katz method: four US indices and persistence-landscape norms",
        fontsize=15, weight="bold",
    )
    fig.text(
        0.5, 0.01,
        "Same 1,000-trading-day lookback and approximately 120-day continuation used for every period; no crash highlighting.",
        ha="center", fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    prices = pd.read_csv(PRICES, parse_dates=["date"]).set_index("date").sort_index()
    prices = prices[INDEXES].dropna()
    norms = pd.read_csv(NORMS, parse_dates=["date"])
    make_original_figure(prices, norms)
    for key, entries in PERIODS.items():
        if key == "original_two_cases":
            continue
        title, event = entries[0]
        make_single_figure(prices, norms, f"{key}.png", title, event)
    (OUT / "README.txt").write_text(
        "These figures use only Group A and the verified paper method: raw daily log returns, "
        "four-index R^4 point clouds, Euclidean Vietoris-Rips H1, and exact landscape L1/L2 norms "
        "for w=50 and w=100. Every figure shows all four normalized indices and the joint norms. "
        "No crash period is shaded or colored specially.",
        encoding="utf-8",
    )
    print(f"Wrote {len(list(OUT.glob('*.png')))} paper-style figures to {OUT}")


if __name__ == "__main__":
    main()
