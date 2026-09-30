"""Reproduce the paper's stated early-warning analysis on Group A.

Important: Gidea & Katz did not train a classifier or set an L1 alarm
threshold. They formed TDA norms with w=50 (and checked w=100), then used a
separate 500-trading-day rolling window to calculate variance, low-frequency
spectral density, and lag-1 ACF of the norm series. Figure 11 evaluates those
statistics for the 250 trading days before 2000-03-10 and 2008-09-15.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
GROUP = ROOT / "data" / "group_a_original_indices"
NORMS = GROUP / "tda_norms_paper_method.csv"
PRICES = GROUP / "prices.csv"
OUT_CSV = GROUP / "paper_exact_250day_diagnostics.csv"
OUT_PNG = GROUP / "paper_exact_250day_diagnostics.png"
OUT_REPORT = GROUP / "paper_exact_detection_report.txt"
EVENTS = {"dotcom_2000_03_10": "2000-03-10", "lehman_2008_09_15": "2008-09-15"}
STAT_WINDOW = 500
PRE_EVENT_DAYS = 250
LOW_FREQ_BINS = 5


def diagnostics(series: pd.Series) -> pd.DataFrame:
    values = series.to_numpy(float)
    rows = []
    for end in range(STAT_WINDOW, len(values) + 1):
        block = values[end - STAT_WINDOW : end]
        centered = block - block.mean()
        power = np.abs(np.fft.rfft(centered)) ** 2 / len(block)
        rows.append(
            {
                "date": series.index[end - 1],
                "variance_500d": float(np.var(block, ddof=1)),
                "low_frequency_power_500d": float(power[1 : LOW_FREQ_BINS + 1].mean()),
                "acf1_500d": float(pd.Series(block).autocorr(lag=1)),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def main() -> None:
    norms = pd.read_csv(NORMS, parse_dates=["date"])
    prices = pd.read_csv(PRICES, parse_dates=["date"]).set_index("date")["^GSPC"].sort_index()
    norm_w50 = norms[norms.window == 50].set_index("date")["landscape_l1"].sort_index()
    norm_w100 = norms[norms.window == 100].set_index("date")["landscape_l1"].sort_index()
    diag50 = diagnostics(norm_w50)
    diag100 = diagnostics(norm_w100)
    all_rows = []
    for event_name, event_date in EVENTS.items():
        event = pd.Timestamp(event_date)
        common_diag_dates = diag50.index.intersection(diag100.index)
        dates = common_diag_dates[common_diag_dates < event][-PRE_EVENT_DAYS:]
        for rank, date in enumerate(dates, 1):
            all_rows.append(
                {
                    "event": event_name,
                    "event_date": event.date().isoformat(),
                    "trading_days_before_event": PRE_EVENT_DAYS - rank,
                    "date": date,
                    "tda_l1_w50": norm_w50.get(date, np.nan),
                    "tda_l1_w100": norm_w100.get(date, np.nan),
                    **diag50.loc[date].add_prefix("w50_").to_dict(),
                    **diag100.loc[date].add_prefix("w100_").to_dict(),
                    "sp500_close": prices.get(date, np.nan),
                }
            )
    output = pd.DataFrame(all_rows)
    output.to_csv(OUT_CSV, index=False, date_format="%Y-%m-%d")

    fig, axes = plt.subplots(3, 2, figsize=(17, 12), sharex="col")
    for col, (event_name, event_date) in enumerate(EVENTS.items()):
        frame = output[output.event == event_name].sort_values("trading_days_before_event")
        x = -frame["trading_days_before_event"].to_numpy()
        def norm(col_name):
            value = frame[col_name].to_numpy(float)
            return value / value[0]
        axes[0, col].plot(x, norm("sp500_close"), color="black", label="S&P 500")
        axes[0, col].plot(x, norm("tda_l1_w50"), color="tab:blue", label="TDA L1, w=50")
        axes[0, col].plot(x, norm("tda_l1_w100"), color="tab:orange", label="TDA L1, w=100")
        axes[0, col].set_title(f"{event_name.replace('_', ' ')}\n0 = {event_date}")
        axes[0, col].set_ylabel("Normalized level")
        axes[0, col].legend(fontsize=8)
        axes[1, col].plot(x, norm("w50_variance_500d"), color="tab:green", label="500d variance")
        axes[1, col].plot(x, norm("w50_low_frequency_power_500d"), color="tab:red", label="500d low-frequency spectral power")
        axes[1, col].set_ylabel("Normalized statistic")
        axes[1, col].legend(fontsize=8)
        axes[2, col].plot(x, frame.w50_acf1_500d, color="tab:purple", label="500d lag-1 ACF")
        axes[2, col].axhline(0, color="black", lw=.7)
        axes[2, col].set_ylabel("ACF(1)")
        axes[2, col].set_xlabel("Trading days relative to paper event (0 = event date)")
        axes[2, col].legend(fontsize=8)
        for row in range(3):
            axes[row, col].grid(alpha=.2)
            axes[row, col].axvline(0, color="#555555", ls=":", lw=.8)
    fig.suptitle(
        "Gidea–Katz early-warning analysis: what changed before the two original events?",
        fontsize=16, weight="bold",
    )
    fig.text(
        .5, .01,
        "Exact stated horizons: TDA point-cloud windows w=50/100; separate 500-trading-day statistics evaluated over the 250 days before each fixed event. "
        "No binary prediction threshold was used. Low-frequency power uses the first five nonzero FFT bins because the paper does not state a numeric cutoff.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, .045, 1, .95))
    fig.savefig(OUT_PNG, dpi=180)
    plt.close(fig)
    report = [
        "Exact Gidea-Katz early-warning replication",
        "",
        "What the paper actually did:",
        "- Daily log returns of four indices formed the 4D point cloud.",
        "- TDA point-cloud windows: w=50 and w=100 trading days; one-day slide.",
        "- The paper displayed 1,000 trading days before each event (Figure 10).",
        "- It then used a separate 500-trading-day rolling window on the resulting norm series.",
        "- It evaluated variance, average low-frequency spectral density, and lag-1 ACF over the 250 trading days before 2000-03-10 and 2008-09-15 (Figure 11).",
        "- It did not use an L1 threshold, binary label, or operational forecast date. The claim is an early-warning trend, not a point prediction.",
        "",
        "Reproduction detail:",
        f"- Low-frequency power here is the mean of FFT bins 1..{LOW_FREQ_BINS}; the paper does not state a numeric frequency cutoff in the text, so this implementation cannot claim an exact spectral-bin reproduction.",
    ]
    for event, frame in output.groupby("event"):
        from scipy.stats import kendalltau
        x = frame["trading_days_before_event"]
        report.append(
            f"- {event}: Kendall tau toward event: variance={kendalltau(x, frame['w50_variance_500d']).statistic:.3f}, "
            f"low-frequency power={kendalltau(x, frame['w50_low_frequency_power_500d']).statistic:.3f}, "
            f"ACF(1)={kendalltau(x, frame['w50_acf1_500d']).statistic:.3f}."
        )
    report += [
        "",
        "Interpretation: the figure shows exactly what could count as advance evidence: a sustained upward movement in the 500-day low-frequency spectral-density curve during the final 250 trading days. "
        "This rerun is not a perfect numerical reproduction of the paper's reported spectral result because its frequency-bin cutoff is not specified and the supplied CSV is Close-derived rather than a separately retrieved adjusted-close series.",
        "No parameter was changed to force agreement.",
    ]
    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {OUT_CSV}, {OUT_PNG}, and {OUT_REPORT}")


if __name__ == "__main__":
    main()
