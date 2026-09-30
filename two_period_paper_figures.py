"""Two longitudinal, presentation-ready figures for the Group A paper method.

The paper's empirical claim has two layers:
1) landscape L1/L2 norms visibly grow into a crisis-era primary peak;
2) the quantified early-warning analysis is a rising low-frequency spectral
   density of the norm series over the 250 trading days before the event.

This script deliberately does not create one image per named event. It creates
only two continuous periods: before 2011 and from 2011 onward. Neutral dashed
date references identify historical stress dates without shading or changing
the plotted data.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
GROUP = ROOT / "data" / "group_a_original_indices"
OUT = GROUP / "two_period_figures"
PRICES = GROUP / "prices.csv"
NORMS = GROUP / "tda_norms_paper_method.csv"
TICKERS = ["^GSPC", "^DJI", "^IXIC", "^RUT"]
NAMES = {"^GSPC": "S&P 500", "^DJI": "DJIA", "^IXIC": "NASDAQ", "^RUT": "Russell 2000"}
LOW_FREQ_WINDOW = 250
LOW_FREQ_BINS = 5


EVENTS = {
    "Pre-2011": [
        ("LTCM / Russia stress", "1998-08-17"),
        ("Dot-com crash date", "2000-03-10"),
        ("Lehman date", "2008-09-15"),
        ("2011 stress", "2011-08-01"),
    ],
    "2011 onward": [
        ("2011 stress", "2011-08-01"),
        ("2015 selloff", "2015-08-24"),
        ("2018 selloff", "2018-12-24"),
        ("COVID crash", "2020-03-16"),
        ("2022 bear", "2022-06-16"),
    ],
}


def low_frequency_power(series: pd.Series) -> pd.Series:
    """Rolling average of the first non-zero FFT power bins.

    This is a transparent reproduction of the paper's low-frequency spectral
    diagnostic. The paper reports the statistic but does not define a unique
    bin cutoff in the text, so the bin count is fixed and documented here.
    """
    values = series.to_numpy(float)
    output = np.full(len(values), np.nan)
    for end in range(LOW_FREQ_WINDOW, len(values) + 1):
        block = values[end - LOW_FREQ_WINDOW : end]
        block = block - block.mean()
        power = np.abs(np.fft.rfft(block)) ** 2 / len(block)
        output[end - 1] = float(power[1 : LOW_FREQ_BINS + 1].mean())
    return pd.Series(output, index=series.index, name="low_frequency_power")


def add_event_references(axes, period: str) -> None:
    for label, date in EVENTS[period]:
        ts = pd.Timestamp(date)
        for ax in axes:
            ax.axvline(ts, color="#666666", linestyle=":", linewidth=0.8, alpha=0.7)
        axes[0].text(
            ts, 1.02, label, rotation=90, va="bottom", ha="right",
            transform=axes[0].get_xaxis_transform(), fontsize=8, color="#444444",
        )


def make_period(period: str, prices: pd.DataFrame, norms: pd.DataFrame, spectral: pd.Series, start: str, end: str) -> None:
    price_slice = prices.loc[start:end]
    norm_slice = norms[(norms.date >= start) & (norms.date <= end)]
    spectral_slice = spectral.loc[start:end]
    normalized = price_slice[TICKERS].div(price_slice[TICKERS].iloc[0]).mul(100)

    fig, axes = plt.subplots(4, 1, figsize=(19, 14), sharex=True, gridspec_kw={"height_ratios": [1.25, 1.35, 1.0, 1.15]})
    for ticker in TICKERS:
        axes[0].plot(normalized.index, normalized[ticker], linewidth=0.9, label=NAMES[ticker])
    axes[0].set_ylabel("Indices\n(start = 100)")
    axes[0].set_title(f"{period}: four-index market state and the paper's TDA early-warning diagnostics", loc="left", weight="bold", fontsize=15)
    axes[0].legend(ncol=4, fontsize=9, loc="upper left")

    for window, linestyle in ((50, "-"), (100, "--")):
        subset = norm_slice[norm_slice.window == window]
        axes[1].plot(subset.date, subset.landscape_l1, linestyle=linestyle, linewidth=0.9, label=f"L1, w={window}")
        axes[1].plot(subset.date, subset.landscape_l2, linestyle=linestyle, linewidth=0.9, alpha=0.55, label=f"L2, w={window}")
    axes[1].set_ylabel("Landscape\nL1 / L2")
    axes[1].legend(ncol=4, fontsize=9, loc="upper left")

    subset50 = norm_slice[norm_slice.window == 50].set_index("date")
    axes[2].plot(subset50.index, subset50.h1_finite_bars, color="#7b2cbf", linewidth=0.8, label="Finite H1 bars (Betti-1 count), w=50")
    axes[2].set_ylabel("H1 loops")
    axes[2].legend(fontsize=9, loc="upper left")

    axes[3].plot(spectral_slice.index, spectral_slice, color="#d95f02", linewidth=0.9, label="250-day low-frequency spectral power of TDA L1")
    axes[3].set_ylabel("Low-frequency\npower")
    axes[3].set_xlabel("Date")
    axes[3].legend(fontsize=9, loc="upper left")

    add_event_references(axes, period)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.text(
        0.01, 0.012,
        "Paper detection logic: a visible rise in landscape norms is descriptive; the quantified early-warning test is a rising "
        "low-frequency spectral-density statistic over the 250 trading days before an event (the paper reports Kendall τ=0.89 and 1.00 "
        "for its two cases). Dotted lines are neutral date references only—no crash region is shaded.",
        fontsize=10, color="#333333",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.98))
    fig.savefig(OUT / ("pre_2011.png" if period == "Pre-2011" else "post_2011.png"), dpi=180)
    plt.close(fig)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    prices = pd.read_csv(PRICES, parse_dates=["date"]).set_index("date").sort_index()[TICKERS].dropna()
    norms = pd.read_csv(NORMS, parse_dates=["date"])
    l1 = norms[norms.window == 50].set_index("date")["landscape_l1"].reindex(prices.index).dropna()
    spectral = low_frequency_power(l1)
    make_period("Pre-2011", prices, norms, spectral, "1997-01-01", "2011-12-31")
    make_period("2011 onward", prices, norms, spectral, "2011-01-01", prices.index.max().date().isoformat())
    (OUT / "README.txt").write_text(
        "Exactly two figures are provided: pre_2011.png and post_2011.png. "
        "Both use only the four Group A indices and the verified paper method. "
        "Panels show normalized index levels, landscape L1/L2 for w=50 and w=100, "
        "finite H1 count, and the transparent 250-day low-frequency spectral-power diagnostic. "
        "Dotted lines are neutral historical date references; no regions or events are highlighted.",
        encoding="utf-8",
    )
    print(f"Wrote {OUT / 'pre_2011.png'} and {OUT / 'post_2011.png'}")


if __name__ == "__main__":
    main()
