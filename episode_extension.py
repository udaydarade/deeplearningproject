"""Episode-level extension of the Group A S&P 500 analysis.

This is deliberately descriptive: the N=15 threshold is the full-sample
second percentile of forward minimum drawdowns, and the resulting positive
days are collapsed into consecutive episodes.  Five trading days are added
on each side only to make the episode plots and the *diagnostic search
interval* readable.  No threshold or date is tuned to detect a crash.

TDA L1 is read from the existing ``tda_norms.csv`` (the original
standardisation/Ripser/grid methodology is therefore unchanged).  CSD
AC1, sample variance, unbiased skewness, and refit normal GARCH(1,1) are
recomputed using the methodology in ``csd_baseline.py`` for n=30, 50, 100.
For each episode and window, a metric's first value above
``mean + 1.5 * std`` of the preceding 100 available observations is
reported.  These are lead-time diagnostics, not a detector or a forecast.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arch import arch_model

ROOT = Path(__file__).resolve().parent
GROUP = ROOT / "data" / "group_a_original_indices"
PRICES = GROUP / "prices.csv"
RETURNS = GROUP / "returns.csv"
TDA = GROUP / "tda_norms.csv"
OUT_CSV = GROUP / "episode_diagnostics.csv"
OUT_FULL = GROUP / "full_history_extended.png"
OUT_GRID = GROUP / "episode_grid.png"
OUT_LEAD = GROUP / "lead_time_comparison.png"
OUT_REPORT = GROUP / "episode_extension_report.txt"
WINDOWS = (30, 50, 100)
HORIZON = 15
PADDING = 5
BASELINE = 100
QUANTILE = 0.02


def forward_min_returns(prices: pd.Series) -> pd.Series:
    values = prices.to_numpy(float)
    out = np.full(len(values), np.nan)
    for i in range(len(values) - HORIZON):
        out[i] = np.min((values[i + 1 : i + HORIZON + 1] - values[i]) / values[i])
    return pd.Series(out, index=prices.index).dropna()


def fit_garch(block: np.ndarray) -> float:
    scaled = pd.Series(block * 100.0)
    model = arch_model(scaled, mean="Constant", vol="GARCH", p=1, q=1,
                       dist="normal", rescale=False)
    try:
        result = model.fit(disp="off", show_warning=False, options={"maxiter": 200})
        return float(result.conditional_volatility.iloc[-1] / 100.0)
    except Exception:
        return np.nan


def csd_series(returns: pd.Series, prices: pd.Series, window: int) -> pd.DataFrame:
    values = returns.to_numpy(float)
    dates = returns.index
    rows = []
    for end in range(window, len(values) + 1):
        block = values[end - window : end]
        rows.append({
            "date": dates[end - 1], "window": window,
            "sp500_close": float(prices.loc[dates[end - 1]]),
            "lag1_autocorrelation": float(pd.Series(block).autocorr(lag=1)),
            "rolling_variance": float(np.var(block, ddof=1)),
            "rolling_skewness": float(pd.Series(block).skew()),
            "garch_conditional_volatility": fit_garch(block),
        })
    return pd.DataFrame(rows)


def episodes(labels: pd.Series, prices: pd.Series) -> list[dict]:
    positive = labels.astype(bool).to_numpy()
    starts = np.flatnonzero(positive & ~np.r_[False, positive[:-1]])
    ends = np.flatnonzero(positive & ~np.r_[positive[1:], False])
    result = []
    for number, (start, end) in enumerate(zip(starts, ends), 1):
        pad_start = max(0, start - PADDING)
        pad_end = min(len(labels) - 1, end + PADDING)
        episode_prices = prices.reindex(labels.index[start : end + 1]).dropna()
        peak_date = episode_prices.idxmax()
        trough_date = episode_prices.loc[peak_date:].idxmin()
        decline = float(episode_prices.loc[trough_date] / episode_prices.loc[peak_date] - 1.0)
        result.append({
            "episode_id": number,
            "start": labels.index[start], "end": labels.index[end],
            "padded_start": labels.index[pad_start], "padded_end": labels.index[pad_end],
            "positive_days": int(end - start + 1),
            "peak_date": peak_date, "trough_date": trough_date,
            "peak_to_trough_decline": decline,
            "_start_i": start, "_end_i": end,
        })
    return result


def metric_diagnostic(series: pd.Series, search_start: pd.Timestamp,
                      search_end: pd.Timestamp) -> dict[str, float | str | int | bool]:
    values = series.dropna()
    search = values[(values.index >= search_start) & (values.index <= search_end)]
    prior = values[values.index < search_start].tail(BASELINE)
    if len(prior) < BASELINE or search.empty:
        return {"baseline_n": int(len(prior)), "baseline_mean": np.nan,
                "baseline_std": np.nan, "threshold": np.nan,
                "first_exceedance": "", "exceedance_n": 0,
                "detected": False, "lead_days": np.nan}
    mean, std = float(prior.mean()), float(prior.std(ddof=1))
    threshold = mean + 1.5 * std
    exceed = search[search > threshold]
    first = exceed.index[0] if not exceed.empty else None
    return {
        "baseline_n": int(len(prior)), "baseline_mean": mean,
        "baseline_std": std, "threshold": threshold,
        "first_exceedance": "" if first is None else first.date().isoformat(),
        "exceedance_n": int(len(exceed)), "detected": first is not None,
        "lead_days": np.nan if first is None else int((search_start - first).days),
    }


def main() -> None:
    prices = pd.read_csv(PRICES, parse_dates=["date"]).set_index("date").sort_index()["^GSPC"].dropna()
    raw_returns = pd.read_csv(RETURNS, parse_dates=["date"]).set_index("date").sort_index()["^GSPC"].dropna()
    common = raw_returns.index.intersection(prices.index)
    returns, prices = raw_returns.loc[common], prices.loc[common]
    forward = forward_min_returns(prices)
    theta = float(forward.quantile(QUANTILE))
    labels = (forward <= theta)
    eps = episodes(labels, prices)
    tda = pd.read_csv(TDA, parse_dates=["date"])
    tda = tda[tda["window"].isin(WINDOWS)].set_index("date").sort_index()

    csd = pd.concat([csd_series(returns, prices, n) for n in WINDOWS], ignore_index=True)
    csd.to_csv(GROUP / "csd_indicators_extended.csv", index=False, date_format="%Y-%m-%d")
    rows = []
    for ep in eps:
        for window in WINDOWS:
            search_start, search_end = ep["padded_start"], ep["padded_end"]
            tda_metric = tda.loc[tda["window"] == window, "landscape_l1"]
            cframe = csd[csd["window"] == window].set_index("date")
            diagnostics = {
                "tda_l1": metric_diagnostic(tda_metric, search_start, search_end),
                "csd_variance": metric_diagnostic(cframe["rolling_variance"], search_start, search_end),
                "csd_garch": metric_diagnostic(cframe["garch_conditional_volatility"], search_start, search_end),
            }
            row = {
                "episode_id": ep["episode_id"], "window": window,
                "episode_start": ep["start"].date().isoformat(),
                "episode_end": ep["end"].date().isoformat(),
                "padded_search_start": search_start.date().isoformat(),
                "padded_search_end": search_end.date().isoformat(),
                "positive_days": ep["positive_days"],
                "peak_date": ep["peak_date"].date().isoformat(),
                "trough_date": ep["trough_date"].date().isoformat(),
                "peak_to_trough_decline": ep["peak_to_trough_decline"],
            }
            for prefix, values in diagnostics.items():
                for key, value in values.items():
                    row[f"{prefix}_{key}"] = value
            rows.append(row)
    diagnostic = pd.DataFrame(rows)
    diagnostic.to_csv(OUT_CSV, index=False)

    # Full-history view, preserving the three window comparisons and episode spans.
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    for ax, window in zip(axes, WINDOWS):
        subset = tda[tda["window"] == window]
        ax.plot(subset.index, subset["landscape_l1"], label=f"TDA L1 n={window}", color="tab:blue")
        cv = csd[csd["window"] == window].set_index("date")
        ax2 = ax.twinx()
        ax2.plot(cv.index, cv["rolling_variance"], label="CSD variance", color="tab:orange", alpha=.8)
        ax2.plot(cv.index, cv["garch_conditional_volatility"], label="CSD GARCH", color="tab:green", alpha=.8)
        for ep in eps:
            ax.axvspan(ep["padded_start"], ep["padded_end"], color="red", alpha=.05)
        ax.set_ylabel(f"n={window}")
        ax.grid(alpha=.2)
    axes[0].set_title("Group A extended history: TDA L1 and recomputed CSD indicators")
    axes[-1].set_xlabel("Date; faint red spans are padded N=15 positive episodes")
    fig.tight_layout(); fig.savefig(OUT_FULL, dpi=160); plt.close(fig)

    # Largest episodes by absolute peak-to-trough decline, at most twelve.
    selected = sorted(eps, key=lambda e: e["peak_to_trough_decline"])[:12]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, ep in zip(axes.flat, selected):
        ax.set_visible(True)
        lo, hi = ep["padded_start"], ep["padded_end"]
        p = prices.loc[lo:hi]
        ax.plot(p.index, p.values, color="black")
        ax.axvspan(ep["start"], ep["end"], color="red", alpha=.18)
        ax.set_title(f"E{ep['episode_id']}: {ep['start'].date()}–{ep['end'].date()}")
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.grid(alpha=.2)
    fig.suptitle("Largest N=15 positive episodes (red = unpadded episode)")
    fig.tight_layout(); fig.savefig(OUT_GRID, dpi=160); plt.close(fig)

    # Lead-time comparison: first exceedance relative to episode start (calendar days).
    plot_rows = []
    for _, row in diagnostic.iterrows():
        for metric, label in (("tda_l1", "TDA L1"), ("csd_variance", "CSD variance"), ("csd_garch", "CSD GARCH")):
            lead = row[f"{metric}_lead_days"]
            plot_rows.append({"episode_id": row["episode_id"], "window": row["window"],
                              "metric": label, "lead_days": lead})
    plot_frame = pd.DataFrame(plot_rows)
    fig, ax = plt.subplots(figsize=(14, 7))
    for label, sub in plot_frame.groupby("metric"):
        ax.scatter(sub["episode_id"] + (sub["window"] - 50) / 300, sub["lead_days"],
                   label=label, alpha=.7)
    ax.axhline(0, color="black", lw=.8); ax.set_xlabel("Episode ID (small horizontal offsets indicate window)")
    ax.set_ylabel("Lead days before padded search start (NaN = no exceedance)")
    ax.set_title("Lead-time diagnostics; threshold is baseline mean + 1.5 SD")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT_LEAD, dpi=160); plt.close(fig)

    report = [
        "Group A episode extension report",
        f"Inputs: {PRICES}, {RETURNS}, {TDA}",
        f"Date range: {prices.index.min().date()} to {prices.index.max().date()} ({len(prices)} prices)",
        f"N={HORIZON} forward-min drawdown theta (full-sample {QUANTILE:.0%} quantile): {theta:.10f} ({theta:.4%})",
        f"Valid forward-label days: {len(labels)}; positive days: {int(labels.sum())}; episodes: {len(eps)}",
        f"Episode padding: {PADDING} trading observations on each side where available; padding is reporting/search context only.",
        "CSD methodology: sample variance (ddof=1), pandas unbiased skew, lag-1 autocorrelation, and refit normal GARCH(1,1) with returns scaled by 100.",
        "TDA methodology is not rerun or altered: existing landscape_l1 values are read from tda_norms.csv.",
        "Diagnostics use the preceding 100 available indicator observations and threshold mean + 1.5 sample SD; no detection decision is made.",
        "",
        "Episodes (start, end, peak-to-trough decline):",
    ]
    for ep in eps:
        report.append(f"E{ep['episode_id']}: {ep['start'].date()} to {ep['end'].date()}, "
                      f"peak {ep['peak_date'].date()}, trough {ep['trough_date'].date()}, "
                      f"decline {ep['peak_to_trough_decline']:.2%}")
    d = diagnostic
    wins = {"tda": 0, "csd": 0, "neither": 0, "ties": 0}
    for episode_id, group in d.groupby("episode_id"):
        firsts = {m: group[f"{m}_first_exceedance"].replace("", np.nan).dropna()
                  for m in ("tda_l1", "csd_variance", "csd_garch")}
        dates = {m: pd.to_datetime(v).min() for m, v in firsts.items() if not v.empty}
        if not dates: wins["neither"] += 1; continue
        earliest = min(dates.values())
        best = [m for m, value in dates.items() if value == earliest]
        if len(best) > 1: wins["ties"] += 1
        elif best[0] == "tda_l1" and all(dates.get(x, pd.Timestamp.max) > earliest for x in ("csd_variance", "csd_garch")):
            wins["tda"] += 1
        elif best[0] in ("csd_variance", "csd_garch") and dates.get("tda_l1", pd.Timestamp.max) > earliest:
            wins["csd"] += 1
        else: wins["ties"] += 1
    report += [
        "",
        f"Episode-level outcome counts (earliest first exceedance across windows): TDA earlier than both CSD metrics={wins['tda']}; CSD wins={wins['csd']}; neither={wins['neither']}; ties/ambiguous={wins['ties']}.",
        "Crash-type interpretation: these empirical episodes are threshold-defined drawdown clusters, not named-crash truth labels.",
        "Any lead-time differences are descriptive and can reflect metric scale, missing early windows, padding, and the chosen baseline; they do not establish predictive power or causality.",
        f"Outputs: {OUT_CSV.name}, csd_indicators_extended.csv, {OUT_FULL.name}, {OUT_GRID.name}, {OUT_LEAD.name}, {OUT_REPORT.name}.",
    ]
    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))
    print(f"Wrote {OUT_CSV}, {OUT_FULL}, {OUT_GRID}, {OUT_LEAD}, {OUT_REPORT}")


if __name__ == "__main__":
    main()
