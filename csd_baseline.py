"""Classical critical-slowing-down baselines for Group A.

The rolling indicators use the same primary n=50 trading-day window as the
TDA replication. GARCH is refit on each window so its conditional volatility
is estimated without using future observations. Returns are scaled by 100 for
the numerical stability of the arch optimizer; volatility is converted back
to daily-return units.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model


ROOT = Path(__file__).resolve().parent
GROUP_DIR = ROOT / "data" / "group_a_original_indices"
RETURNS_FILE = GROUP_DIR / "returns.csv"
PRICES_FILE = GROUP_DIR / "prices.csv"
OUTPUT_FILE = GROUP_DIR / "csd_indicators.csv"
PLOT_FILE = GROUP_DIR / "csd_indicators.png"
REPORT_FILE = GROUP_DIR / "csd_baseline_report.txt"
WINDOW = 50
REFERENCE = "^GSPC"


def fit_garch_window(window_returns: np.ndarray) -> float:
    """Fit GARCH(1,1) on one window and return sigma for its final day."""
    scaled = pd.Series(window_returns * 100.0)
    model = arch_model(scaled, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    try:
        result = model.fit(disp="off", show_warning=False, options={"maxiter": 200})
        return float(result.conditional_volatility.iloc[-1] / 100.0)
    except Exception:
        # A short/highly tranquil window can occasionally make the optimizer
        # fail. Preserve the observed rolling indicators and mark GARCH NaN.
        return np.nan


def compute_indicators(returns: pd.Series, prices: pd.Series) -> pd.DataFrame:
    rows = []
    values = returns.to_numpy(dtype=float)
    dates = returns.index
    for end in range(WINDOW, len(values) + 1):
        block = values[end - WINDOW : end]
        ac1 = float(pd.Series(block).autocorr(lag=1))
        variance = float(np.var(block, ddof=1))
        skewness = float(pd.Series(block).skew())
        garch_sigma = fit_garch_window(block)
        rows.append(
            {
                "date": dates[end - 1],
                "window": WINDOW,
                "sp500_close": float(prices.loc[dates[end - 1]]),
                "lag1_autocorrelation": ac1,
                "rolling_variance": variance,
                "rolling_skewness": skewness,
                "garch_conditional_volatility": garch_sigma,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    returns = pd.read_csv(RETURNS_FILE, parse_dates=["date"]).set_index("date").sort_index()[REFERENCE].dropna()
    prices = pd.read_csv(PRICES_FILE, parse_dates=["date"]).set_index("date").sort_index()[REFERENCE]
    common = returns.index.intersection(prices.index)
    returns, prices = returns.loc[common], prices.loc[common]
    indicators = compute_indicators(returns, prices)
    indicators.to_csv(OUTPUT_FILE, index=False, date_format="%Y-%m-%d")

    fig, axes = plt.subplots(4, 1, figsize=(15, 15), sharex=True)
    columns = [
        ("lag1_autocorrelation", "Lag-1 AC"),
        ("rolling_variance", "Rolling variance"),
        ("rolling_skewness", "Rolling skewness"),
        ("garch_conditional_volatility", "GARCH(1,1) conditional volatility"),
    ]
    indexed = indicators.set_index("date")
    for ax, (column, label) in zip(axes, columns):
        ax2 = ax.twinx()
        ax.plot(indexed.index, indexed[column], color="tab:blue", label=label)
        ax2.plot(indexed.index, indexed["sp500_close"], color="black", alpha=0.25, label="S&P 500 Close")
        for start, end in (("2000-03-01", "2002-10-31"), ("2007-10-01", "2009-03-31")):
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="red", alpha=0.12)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, fontsize=8, loc="upper left")
    axes[0].set_title("Classical CSD baselines vs S&P 500 (50-trading-day rolling window)")
    axes[-1].set_xlabel("Date; shaded: dot-com (2000-03 to 2002-10), financial crisis (2007-10 to 2009-03)")
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=160)
    plt.close(fig)

    report = [
        "Classical CSD baseline report",
        f"Input: {RETURNS_FILE}",
        f"Reference: {REFERENCE}; date range {returns.index.min().date()} to {returns.index.max().date()}",
        f"Rolling window: {WINDOW} trading days",
        "Variance uses sample variance (ddof=1); skewness is pandas unbiased sample skewness.",
        "GARCH is refit on every rolling window, with a constant mean and normal GARCH(1,1).",
        "GARCH optimization uses returns scaled by 100 and volatility is reported in raw daily-return units.",
        f"GARCH missing estimates: {int(indicators['garch_conditional_volatility'].isna().sum())}",
        "This is a descriptive baseline, not a tuned crisis detector; no thresholds or parameter selection used crisis dates.",
        "The plot uses the same shaded periods and S&P 500 reference style as the TDA plot.",
    ]
    REPORT_FILE.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"Wrote {OUTPUT_FILE}, {PLOT_FILE}, and {REPORT_FILE}")


if __name__ == "__main__":
    main()
