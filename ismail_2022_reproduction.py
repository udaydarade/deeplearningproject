"""Reproduce the Ismail et al. (2022) PH+CSD experiment on Group A.

The paper uses the four US indices, raw daily log returns, a 50-day sliding
point cloud, the H1 persistence-landscape L1 norm, then 500-value rolling
AC1/variance/MPS indicators. Trend tests are calculated on sliding windows
of 250 indicator values and significant positive trends are used to derive
an event-trained threshold. This script reproduces that architecture on the
available Group A files and reports deviations explicitly.

The paper does not state a numerical low-frequency cutoff in the accessible
text. LOW_FREQ_BINS=5 is therefore a documented, reproducible assumption.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
GROUP = ROOT / "data" / "group_a_original_indices"
OUT = GROUP / "ismail_2022_reproduction"
PRICES_FILE = GROUP / "prices.csv"
NORMS_FILE = GROUP / "tda_norms_paper_method.csv"
EVENTS = {"dotcom": pd.Timestamp("2000-03-10"), "lehman": pd.Timestamp("2008-09-15")}
STAT_WINDOW = 500
TREND_WINDOW = 250
LOW_FREQ_BINS = 5
ALPHA = 0.05


def indicator_series(l1: pd.Series) -> pd.DataFrame:
    rows = []
    values, dates = l1.to_numpy(float), l1.index
    for end in range(STAT_WINDOW, len(l1) + 1):
        block = values[end - STAT_WINDOW:end]
        centered = block - block.mean()
        fft = np.fft.rfft(centered)
        power = np.abs(fft[1:]) ** 2 / len(block)
        rows.append({
            "date": dates[end - 1],
            "ac1": pd.Series(block).autocorr(lag=1),
            "variance": np.var(block, ddof=1),
            "mps_low_frequency": np.mean(power[:LOW_FREQ_BINS]),
        })
    return pd.DataFrame(rows).set_index("date")


def trend_tests(series: pd.Series, name: str) -> pd.DataFrame:
    rows = []
    values, dates = series.to_numpy(float), series.index
    for end in range(TREND_WINDOW, len(values) + 1):
        block = values[end - TREND_WINDOW:end]
        x = np.arange(len(block))
        kt, kp = kendalltau(x, block)
        pt, pp = pearsonr(x, block)
        st, sp = spearmanr(x, block)
        rows.append({
            "date": dates[end - 1], "indicator": name,
            "kendall_r": kt, "kendall_p": kp,
            "pearson_r": pt, "pearson_p": pp,
            "spearman_r": st, "spearman_p": sp,
        })
    return pd.DataFrame(rows)


def threshold_for(frame: pd.DataFrame, method: str, event_dates: list[pd.Timestamp]) -> float:
    r, p = f"{method}_r", f"{method}_p"
    candidates = frame[(frame.date.isin(event_dates)) & (frame[r] > 0) & (frame[p] < ALPHA)][r]
    if len(candidates):
        return float(candidates.min())
    # The paper's exact event-date row can be unavailable because the
    # indicator date is the end of a rolling window. Use the final available
    # row strictly before each event as a transparent fallback.
    rows = []
    for event in event_dates:
        prior = frame[frame.date < event].tail(1)
        if not prior.empty:
            rows.append(prior.iloc[0])
    candidates = pd.DataFrame(rows)
    candidates = candidates[(candidates[r] > 0) & (candidates[p] < ALPHA)]
    return float(candidates[r].min()) if not candidates.empty else np.nan


def main() -> None:
    OUT.mkdir(exist_ok=True)
    prices = pd.read_csv(PRICES_FILE, parse_dates=["date"]).set_index("date").sort_index()
    norms = pd.read_csv(NORMS_FILE, parse_dates=["date"])
    l1 = norms[norms.window == 50].set_index("date")["landscape_l1"].sort_index().dropna()
    indicators = indicator_series(l1)
    test_frames = pd.concat(
        [trend_tests(indicators[column], column) for column in indicators.columns],
        ignore_index=True,
    )
    event_rows = []
    event_dates = list(EVENTS.values())
    for event_name, event in EVENTS.items():
        for indicator in indicators.columns:
            row = indicators[indicators.index < event].tail(1)
            if row.empty:
                continue
            stats = test_frames[(test_frames.indicator == indicator) & (test_frames.date == row.index[0])]
            if not stats.empty:
                event_rows.append({"event": event_name, "event_date": event, "indicator": indicator, **stats.iloc[0].drop("indicator").to_dict()})
    event_tests = pd.DataFrame(event_rows)
    threshold_rows = []
    for indicator in indicators.columns:
        f = test_frames[test_frames.indicator == indicator]
        for method in ("kendall", "pearson", "spearman"):
            threshold_rows.append({"indicator": indicator, "method": method, "threshold": threshold_for(f, method, event_dates)})
    thresholds = pd.DataFrame(threshold_rows)

    classified = []
    for _, row in test_frames.iterrows():
        for _, threshold in thresholds[thresholds.indicator == row.indicator].iterrows():
            method = threshold.method
            r, p = row[f"{method}_r"], row[f"{method}_p"]
            signal = bool(np.isfinite(threshold.threshold) and r >= threshold.threshold and p < ALPHA and r > 0)
            classified.append({
                "date": row.date, "indicator": row.indicator, "method": method,
                "correlation": r, "p_value": p, "threshold": threshold.threshold,
                "signal": signal,
            })
    classified = pd.DataFrame(classified)
    indicators.to_csv(OUT / "indicators.csv", date_format="%Y-%m-%d")
    test_frames.to_csv(OUT / "trend_tests.csv", index=False, date_format="%Y-%m-%d")
    event_tests.to_csv(OUT / "event_day_tests.csv", index=False, date_format="%Y-%m-%d")
    thresholds.to_csv(OUT / "thresholds.csv", index=False)
    classified.to_csv(OUT / "classification_timeline.csv", index=False, date_format="%Y-%m-%d")

    fig, axes = plt.subplots(4, 1, figsize=(18, 15), sharex=True)
    px = prices.loc[: "2011-12-31", ["^GSPC", "^DJI", "^IXIC", "^RUT"]]
    for ticker in px:
        axes[0].plot(px.index, px[ticker] / px[ticker].iloc[0] * 100, lw=.8, label=ticker)
    axes[0].set_ylabel("Prices\n(start=100)")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].set_title("Ismail et al. (2022) reproduction: PH + CSD trend classification", loc="left", weight="bold")
    axes[1].plot(l1.index, l1, color="tab:blue", lw=.8, label="TDA L1, w=50")
    axes[1].set_ylabel("L1 norm")
    axes[1].legend()
    for col, color in zip(indicators.columns, ("tab:purple", "tab:green", "tab:red")):
        axes[2].plot(indicators.index, indicators[col], color=color, lw=.8, label=col)
    axes[2].set_ylabel("500-value CSD")
    axes[2].legend(ncol=3, fontsize=8)
    for method, color in zip(("kendall", "pearson", "spearman"), ("tab:blue", "tab:orange", "tab:green")):
        subset = classified[(classified.indicator == "mps_low_frequency") & (classified.method == method)]
        axes[3].plot(subset.date, subset.correlation, color=color, lw=.8, label=f"{method} rho/tau")
        threshold = thresholds[(thresholds.indicator == "mps_low_frequency") & (thresholds.method == method)].threshold.iloc[0]
        if np.isfinite(threshold):
            axes[3].axhline(threshold, color=color, ls="--", lw=.7, alpha=.7)
    axes[3].set_ylabel("MPS trend correlation")
    axes[3].set_xlabel("Date; dashed lines are event-trained thresholds, not out-of-sample thresholds")
    axes[3].legend(ncol=3, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
        for event in EVENTS.values():
            ax.axvline(event, color="#666", ls=":", lw=.7)
    fig.tight_layout()
    fig.savefig(OUT / "ismail_2022_reproduction.png", dpi=180)
    plt.close(fig)

    report = [
        "Ismail et al. (2022) reproduction on Group A",
        "",
        "Paper procedure implemented:",
        "- Four US indices and raw daily log returns.",
        "- 50-trading-day sliding point clouds and H1 persistence-landscape L1 norm.",
        "- 500-value rolling AC1, variance, and mean low-frequency power spectrum (MPS).",
        "- 250-value sliding correlation tests using Kendall, Pearson, and Spearman.",
        "- Significant positive trend requires p < 0.05; event-trained threshold is the smaller positive significant correlation at the two event dates, then applied across the available timeline.",
        "",
        "Reproduction limitations:",
        "- The supplied data begins in 1997, while the paper begins in 1987. Therefore the dot-com pre-crisis sample cannot contain the paper's full 1,000 prices.",
        "- The paper's accessible text does not specify a numeric low-frequency cutoff. This script uses the first five nonzero FFT bins.",
        "- Event-trained thresholds are retrospective by design, matching the paper's evaluation concept; they are not valid out-of-sample thresholds.",
        "- The paper's reported PSA/PEA percentages cannot be claimed as reproduced without its exact original data, exact frequency cutoff, and exact breakpoint implementation.",
        "",
        "Event-day trend results:",
    ]
    for _, r in event_tests.iterrows():
        report.append(f"{r.event} {r.indicator}: Kendall r={r.kendall_r:.4f}, p={r.kendall_p:.4g}; Pearson r={r.pearson_r:.4f}, p={r.pearson_p:.4g}; Spearman r={r.spearman_r:.4f}, p={r.spearman_p:.4g}")
    report += ["", "Event-trained thresholds and full-history signal prevalence:"]
    for _, r in thresholds.iterrows():
        subset = classified[
            (classified.indicator == r.indicator)
            & (classified.method == r.method)
        ]
        if np.isfinite(r.threshold):
            signals = int(subset.signal.sum())
            report.append(
                f"{r.indicator} / {r.method}: threshold={r.threshold:.6f}; "
                f"signal dates={signals}/{len(subset)} "
                f"({100 * signals / len(subset):.2f}%)"
            )
        else:
            report.append(
                f"{r.indicator} / {r.method}: no finite event-trained threshold; "
                f"signal dates=0/{len(subset)}"
            )
    report += [
        "",
        "Comparison with the published 2022 paper:",
        "- The paper reports whole-history PSA/PEA of 42.31%/57.69% for Kendall, 40%/60% for Pearson, and 46.15%/53.85% for Spearman when using L1 with MPS.",
        "- Those percentages are not reproduced here as exact values because the available project data starts in 1997 rather than 1987, the source paper does not state the numeric MPS frequency cutoff, and its breakpoint/counting implementation is not fully specified in the accessible text.",
        "- In this data rerun, the dot-com event-day MPS trend is negative for all three tests, while the Lehman event-day MPS trend is positive and highly significant for all three. This is a material mismatch with the paper's claimed positive MPS trend before both events.",
        "- The classification timeline is saved so the event-trained threshold procedure can be inspected directly.",
        "",
        "Generated files: indicators.csv, trend_tests.csv, event_day_tests.csv, thresholds.csv, classification_timeline.csv, ismail_2022_reproduction.png.",
    ]
    (OUT / "ismail_2022_reproduction_report.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
