"""Replicate the Gidea-Katz sliding-window H1 persistence-landscape signal.

Input: data/group_a_original_indices/{prices,returns}.csv
Outputs:
  data/group_a_original_indices/tda_norms.csv
  data/group_a_original_indices/tda_norms.png
  data/group_a_original_indices/tda_replication_report.txt

The empirical paper uses raw daily log returns (not rolling standardization),
w=50 and w=100 trading-day windows, one-day steps, and Euclidean Rips.
Landscape norms are computed analytically from finite H1 bars.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ripser import ripser


ROOT = Path(__file__).resolve().parent
GROUP_DIR = ROOT / "data" / "group_a_original_indices"
RETURN_FILE = GROUP_DIR / "returns.csv"
PRICE_FILE = GROUP_DIR / "prices.csv"
OUTPUT_FILE = GROUP_DIR / "tda_norms.csv"
PLOT_FILE = GROUP_DIR / "tda_norms.png"
REPORT_FILE = GROUP_DIR / "tda_replication_report.txt"
TICKERS = ["^GSPC", "^DJI", "^IXIC", "^RUT"]
WINDOWS = (50, 100)


def finite_h1_bars(points: np.ndarray) -> np.ndarray:
    diagrams = ripser(points, maxdim=1, metric="euclidean")["dgms"]
    if len(diagrams) < 2:
        return np.empty((0, 2))
    h1 = diagrams[1]
    return h1[np.isfinite(h1).all(axis=1)]


def landscape_norms(bars: np.ndarray) -> tuple[float, float, float, float]:
    """Return exact landscape L1/L2 and retained bar-length diagnostics."""
    if bars.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    lengths = bars[:, 1] - bars[:, 0]
    l1 = float(np.square(lengths).sum() / 4.0)
    l2 = float(np.sqrt(np.power(lengths, 3).sum() / 12.0))
    proxy_l1 = float(lengths.sum())
    proxy_l2 = float(np.sqrt(np.square(lengths).sum()))
    return l1, l2, proxy_l1, proxy_l2


def compute_window_series(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    rows = []
    values = returns[TICKERS].to_numpy(dtype=float)
    dates = returns.index
    for end in range(window, len(returns) + 1):
        block = values[end - window : end]
        bars = finite_h1_bars(block)
        l1, l2, proxy_l1, proxy_l2 = landscape_norms(bars)
        rows.append(
            {
                "date": dates[end - 1],
                "window": window,
                "h1_finite_bars": len(bars),
                "landscape_l1": l1,
                "landscape_l2": l2,
                "bar_length_l1_proxy": proxy_l1,
                "bar_length_l2_proxy": proxy_l2,
            }
        )
    return pd.DataFrame(rows)


def peak_and_period_stats(frame: pd.DataFrame, column: str) -> dict[str, float | str]:
    pre_crash = frame[(frame["date"] >= "1998-01-01") & (frame["date"] <= "2000-02-29")]
    dotcom = frame[(frame["date"] >= "2000-03-01") & (frame["date"] <= "2002-10-31")]
    financial = frame[(frame["date"] >= "2007-10-01") & (frame["date"] <= "2009-03-31")]
    peak = frame.loc[frame[column].idxmax()]
    return {
        "peak_date": str(peak["date"].date()),
        "peak_value": float(peak[column]),
        "dotcom_max": float(dotcom[column].max()),
        "pre_dotcom_max": float(pre_crash[column].max()),
        "financial_max": float(financial[column].max()),
    }


def main() -> None:
    returns = pd.read_csv(RETURN_FILE, parse_dates=["date"]).set_index("date").sort_index()
    prices = pd.read_csv(PRICE_FILE, parse_dates=["date"]).set_index("date").sort_index()
    missing = [ticker for ticker in TICKERS if ticker not in returns or ticker not in prices]
    if missing:
        raise ValueError(f"Missing required Group A columns: {missing}")
    returns = returns[TICKERS].dropna()
    results = pd.concat([compute_window_series(returns, window) for window in WINDOWS], ignore_index=True)
    results = results.merge(prices[["^GSPC"]].rename(columns={"^GSPC": "sp500_close"}), left_on="date", right_index=True, how="left")
    results.to_csv(OUTPUT_FILE, index=False, date_format="%Y-%m-%d")

    fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)
    for ax, window in zip(axes, WINDOWS):
        subset = results[results["window"] == window].set_index("date")
        ax2 = ax.twinx()
        ax.plot(subset.index, subset["landscape_l1"], label="numerical L1", color="tab:blue")
        ax.plot(subset.index, subset["landscape_l2"], label="numerical L2", color="tab:orange")
        ax.plot(subset.index, subset["bar_length_l1_proxy"], label="bar-length L1 proxy", color="tab:green", alpha=0.65)
        ax2.plot(subset.index, subset["sp500_close"], label="S&P 500 Close", color="black", alpha=0.25)
        for start, end in (("2000-03-01", "2002-10-31"), ("2007-10-01", "2009-03-31")):
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="red", alpha=0.12)
        ax.set_ylabel(f"n={window} norms")
        ax.grid(alpha=0.2)
        if window == WINDOWS[0]:
            ax.set_title("Gidea-Katz replication: H1 persistence-landscape norms and S&P 500")
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, fontsize=8, loc="upper left")
    axes[-1].set_xlabel("Date; shaded: dot-com (2000-03 to 2002-10), financial crisis (2007-10 to 2009-03)")
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=160)
    plt.close(fig)

    report_lines = [
        "TDA replication report",
        f"Input returns: {RETURN_FILE}",
        f"Input date range: {returns.index.min().date()} to {returns.index.max().date()} ({len(returns)} rows)",
        "Method: raw daily log-returns from the four indices; no rolling standardization or smoothing.",
        "Filtration: Ripser Vietoris-Rips, Euclidean metric, maxdim=1; infinite H1 bars excluded.",
        "Landscape norms: exact integrals from finite H1 bars (L1=sum persistence^2/4; L2=sqrt(sum persistence^3/12)).",
        "Plots show numerical L1/L2 and the bar-length L1 proxy; the CSV also contains a bar-length L2 proxy.",
        "",
        "Peak and period diagnostics (descriptive, not a causal test):",
    ]
    for window in WINDOWS:
        subset = results[results["window"] == window]
        report_lines.append(f"\nwindow={window}")
        for metric in ("landscape_l1", "landscape_l2", "bar_length_l1_proxy"):
            report_lines.append(f"{metric}: {peak_and_period_stats(subset, metric)}")
    report_lines += [
        "",
        "Interpretation: assess qualitative consistency by checking whether norms rise before or during the shaded crash periods.",
        "No parameter tuning was performed to force agreement. Remaining differences can arise from data vendor history,",
        "window anchoring, exact software implementation, and the fact that the paper's quantitative test also uses low-frequency spectral density.",
    ]
    REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")
    print("\n".join(report_lines))
    print(f"\nWrote {OUTPUT_FILE}, {PLOT_FILE}, and {REPORT_FILE}")


if __name__ == "__main__":
    main()
