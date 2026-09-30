"""Presentation figures matching Gidea & Katz Figures 10 and 11.

Only Group A is used. The paper's actual design:
  * 4 index daily log returns -> Rips H1 persistence landscapes;
  * point-cloud windows w=50 (and robustness w=100);
  * Figure-10-style 1000-trading-day normalized series;
  * Figure-11-style 500-trading-day rolling variance, average low-frequency
    spectral density, and lag-1 ACF evaluated over 250 pre-event days;
  * Mann-Kendall trend is the reported quantitative test, not an L1 alarm.

The normal-day comparison is added explicitly for presentation: shaded bands
are the 5th-95th percentile of the same statistic on dates outside ±250
trading days around the named stress dates. They are reference distributions,
not tuned detection thresholds.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent
GROUP = ROOT / "data" / "group_a_original_indices"
OUT = GROUP / "paper_reproduction_figures"
PRICES = GROUP / "prices.csv"
NORMS = GROUP / "tda_norms_paper_method.csv"
WINDOW = 50
STAT_WINDOW = 500
PRE_EVENT = 250
LOW_FREQ_BINS = 5
INDEXES = ["^GSPC", "^DJI", "^IXIC", "^RUT"]
NAMES = {"^GSPC": "S&P 500", "^DJI": "DJIA", "^IXIC": "NASDAQ", "^RUT": "Russell 2000"}
EVENTS = [
    pd.Timestamp("1998-08-17"), pd.Timestamp("2000-03-10"),
    pd.Timestamp("2008-09-15"), pd.Timestamp("2011-08-01"),
    pd.Timestamp("2015-08-24"), pd.Timestamp("2018-12-24"),
    pd.Timestamp("2020-03-16"), pd.Timestamp("2022-06-16"),
]


def rolling_stats(series: pd.Series) -> pd.DataFrame:
    values, dates = series.to_numpy(float), series.index
    rows = []
    for end in range(STAT_WINDOW, len(values) + 1):
        block = values[end - STAT_WINDOW : end]
        freq, power = periodogram(block - block.mean(), scaling="density")
        rows.append({
            "date": dates[end - 1],
            "variance": np.var(block, ddof=1),
            "low_frequency_power": np.mean(power[1 : LOW_FREQ_BINS + 1]),
            "acf1": pd.Series(block).autocorr(lag=1),
        })
    return pd.DataFrame(rows).set_index("date")


def normal_band(frame: pd.DataFrame, column: str, excluded: pd.DatetimeIndex) -> tuple[float, float]:
    values = frame.loc[~frame.index.isin(excluded), column].dropna()
    return float(values.quantile(.05)), float(values.quantile(.95))


def plot_period(name: str, start: str, end: str, prices: pd.DataFrame, norms: pd.DataFrame, stats: pd.DataFrame, normal_ranges: dict[str, tuple[float, float]]) -> None:
    px = prices.loc[start:end]
    nm = norms[(norms.date >= start) & (norms.date <= end)]
    st = stats.loc[start:end]
    normalized = px[INDEXES].div(px[INDEXES].iloc[0]).mul(100)
    fig, axes = plt.subplots(6, 1, figsize=(19, 18), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.15, 1, 1, 1, 1]})
    for ticker in INDEXES:
        axes[0].plot(normalized.index, normalized[ticker], lw=.85, label=NAMES[ticker])
    axes[0].set_ylabel("Indices\n(start=100)")
    axes[0].legend(ncol=4, fontsize=9, loc="upper left")
    axes[0].set_title(f"{name}: exact paper-style early-warning analysis vs normal-day reference", loc="left", weight="bold", fontsize=15)

    for window, linestyle in ((50, "-"), (100, "--")):
        q = nm[nm.window == window]
        axes[1].plot(q.date, q.landscape_l1, linestyle=linestyle, lw=.85, label=f"TDA L1 w={window}")
        axes[1].plot(q.date, q.landscape_l2, linestyle=linestyle, lw=.85, alpha=.5, label=f"TDA L2 w={window}")
    axes[1].set_ylabel("Landscape norms")
    axes[1].legend(ncol=4, fontsize=8, loc="upper left")

    # Each diagnostic gets its own axis: raw magnitudes differ by orders of
    # magnitude, so a single shared axis would hide the changes.
    for axis, column, label, color in zip(
        axes[2:5],
        ("variance", "low_frequency_power", "acf1"),
        ("500d rolling variance", "500d average low-frequency spectral density", "500d lag-1 ACF"),
        ("#2a9d8f", "#e76f51", "#7b2cbf"),
    ):
        lo, hi = normal_ranges[column]
        axis.fill_between(st.index, lo, hi, color=color, alpha=.12, label="ordinary days: 5–95%")
        axis.plot(st.index, st[column], color=color, lw=.9, label=label)
        axis.set_ylabel(label.replace("500d ", ""))
        axis.legend(fontsize=8, loc="upper left")
    # The bottom panel puts all three diagnostics in their own normal-band
    # units, useful for seeing simultaneous departures without arbitrary alarm
    # thresholds.
    for column, label, color in (
        ("variance", "Variance", "#2a9d8f"),
        ("low_frequency_power", "Low-frequency spectral density", "#e76f51"),
        ("acf1", "Lag-1 ACF", "#7b2cbf"),
    ):
        lo, hi = normal_ranges[column]
        denom = max(hi - lo, np.finfo(float).eps)
        axes[5].plot(st.index, (st[column] - lo) / denom, color=color, lw=.9, label=label)
    axes[5].axhline(0, color="black", lw=.7)
    axes[5].axhline(1, color="#777777", ls="--", lw=.7, label="upper ordinary-day band")
    axes[5].set_ylabel("Normal-band\nunits")
    axes[5].set_xlabel("Date; dotted lines are reference event dates, not fitted alarm boundaries")
    axes[5].legend(ncol=4, fontsize=8, loc="upper left")
    for ax in axes:
        ax.grid(alpha=.2)
        for event in EVENTS:
            if pd.Timestamp(start) <= event <= pd.Timestamp(end):
                ax.axvline(event, color="#777777", ls=":", lw=.7, alpha=.65)
    fig.text(.01, .01,
             "Paper interpretation: rising TDA norms are descriptive; the quantitative test is a rising low-frequency "
             "spectral-density trend over the final 250 trading days, assessed by Mann–Kendall. "
             "Bands show ordinary-day variation and are not fitted alert thresholds.",
             fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .98))
    fig.savefig(OUT / f"{name}.png", dpi=180)
    plt.close(fig)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    prices = pd.read_csv(PRICES, parse_dates=["date"]).set_index("date").sort_index()[INDEXES].dropna()
    norms = pd.read_csv(NORMS, parse_dates=["date"])
    l1 = norms[norms.window == WINDOW].set_index("date")["landscape_l1"].sort_index()
    stats = rolling_stats(l1)
    excluded = pd.DatetimeIndex([])
    for event in EVENTS:
        excluded = excluded.union(stats.index[(stats.index >= event - pd.Timedelta(days=250)) & (stats.index <= event + pd.Timedelta(days=250))])
    bands = {column: normal_band(stats, column, excluded) for column in ["variance", "low_frequency_power", "acf1"]}
    plot_period("pre_2011", "1997-01-01", "2011-12-31", prices, norms, stats, bands)
    plot_period("post_2011", "2011-01-01", prices.index[-1].date().isoformat(), prices, norms, stats, bands)

    report = [
        "Exact paper-style reproduction with normal-day comparison",
        "The paper did not predict a crash with a binary TDA threshold.",
        "It used w=50 (and w=100 robustness) TDA point-cloud windows, then a separate 500-trading-day rolling analysis of the norm series.",
        "The three paper diagnostics were variance, average low-frequency spectral density, and lag-1 ACF, evaluated over the 250 trading days before fixed event dates.",
        "The paper's quantitative claim was a rising low-frequency spectral-density trend, tested with Mann-Kendall; it reported tau=0.89 and 1.00 for the two original cases.",
        "The plots compare these curves against 5th–95th percentile normal-day bands computed outside ±250 trading days around all reference stress dates.",
        "The paper does not state the exact low-frequency cutoff, so this implementation uses the first five nonzero periodogram bins and documents that limitation.",
        "",
    ]
    for event in [pd.Timestamp("2000-03-10"), pd.Timestamp("2008-09-15")]:
        window = stats.loc[:event].tail(PRE_EVENT)
        tau = kendalltau(np.arange(len(window)), window.low_frequency_power).statistic
        report.append(f"{event.date()}: 250-day low-frequency-power Kendall tau={tau:.3f}; variance tau={kendalltau(np.arange(len(window)), window.variance).statistic:.3f}; ACF tau={kendalltau(np.arange(len(window)), window.acf1).statistic:.3f}")
    (OUT / "README.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {OUT / 'pre_2011.png'}, {OUT / 'post_2011.png'}")


if __name__ == "__main__":
    main()
