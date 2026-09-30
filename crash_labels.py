"""Build statistically calibrated crash episodes and forward labels.

The detector follows an epsilon-drawdown/outlier construction:

* the full running-maximum drawdown is retained for every price series;
* local peak-to-trough moves are extracted with an epsilon zigzag filter;
* the exponential bulk is selected by the smallest KS distance over candidate
  bulk fractions (rather than by hand-picking a crash percentage);
* the outlier probability is family-wise calibrated at 5% with a Bonferroni
  correction over the fitted drawdown sample.

Outputs are written beside each input price file:
``drawdowns.csv``, ``crash_episodes.csv``, ``labels_N10.csv``,
``labels_N20.csv``, ``labels_N40.csv``, and ``crash_detection_report.txt``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
EPSILON = 0.01
FAMILYWISE_ALPHA = 0.05
HORIZONS = (10, 20, 40)

PRICE_FILES = {
    "group_a_original_indices": DATA / "group_a_original_indices" / "prices.csv",
    "group_b_cross_asset": DATA / "group_b_cross_asset" / "prices.csv",
    "group_c_sector": DATA / "group_c_panel" / "sector_prices.csv",
    "group_c_crypto": DATA / "group_c_panel" / "crypto_prices.csv",
    "group_c_international": DATA / "group_c_panel" / "international_prices.csv",
}


def running_drawdown(prices: pd.Series) -> pd.Series:
    """Return (running maximum - price) / running maximum."""
    running_max = prices.cummax()
    return (running_max - prices) / running_max


def epsilon_episodes(prices: pd.Series, epsilon: float) -> list[dict]:
    """Extract independent high-water-mark peak-to-trough moves.

    A drawdown starts only after an epsilon decline from the running peak and
    ends when that peak is recovered. This prevents small rebounds inside a
    prolonged selloff from creating artificial independent episodes.
    """
    values = prices.to_numpy(float)
    dates = prices.index
    if len(values) < 2:
        return []

    episodes: list[dict] = []
    peak_i = 0
    trough_i = 0
    state = "peak"
    for i in range(1, len(values)):
        if state == "peak":
            if values[i] >= values[peak_i]:
                peak_i = i
                trough_i = i
            elif values[i] <= values[peak_i] * (1.0 - epsilon):
                trough_i = i
                state = "trough"
        else:
            if values[i] < values[trough_i]:
                trough_i = i
            elif values[i] >= values[peak_i]:
                drawdown = 1.0 - values[trough_i] / values[peak_i]
                if drawdown >= epsilon:
                    episodes.append(
                        {
                            "start_date": dates[peak_i],
                            "trough_date": dates[trough_i],
                            "recovery_date": dates[i],
                            "drawdown": float(drawdown),
                            "peak_index": peak_i,
                            "trough_index": trough_i,
                        }
                    )
                peak_i = i
                trough_i = i
                state = "peak"

    if state == "trough":
        drawdown = 1.0 - values[trough_i] / values[peak_i]
        if drawdown >= epsilon:
            episodes.append(
                {
                    "start_date": dates[peak_i],
                    "trough_date": dates[trough_i],
                    "recovery_date": pd.NaT,
                    "drawdown": float(drawdown),
                    "peak_index": peak_i,
                    "trough_index": trough_i,
                }
            )
    return episodes


def fit_exponential(drawdowns: np.ndarray) -> dict[str, float | int]:
    """Select the best-fitting lower bulk and fit its exponential rate."""
    values = np.sort(np.asarray(drawdowns, dtype=float))
    if len(values) < 2:
        raise ValueError("At least 2 epsilon drawdowns are required.")

    candidates = sorted(
        {
            max(2, min(len(values), int(round(len(values) * fraction))))
            for fraction in np.arange(0.50, 0.96, 0.01)
        }
    )
    best: tuple[float, int, float] | None = None
    for bulk_n in candidates:
        bulk = values[:bulk_n]
        rate = 1.0 / float(bulk.mean())
        empirical = np.arange(1, bulk_n + 1) / bulk_n
        fitted = 1.0 - np.exp(-rate * bulk)
        ks = float(np.max(np.abs(empirical - fitted)))
        candidate = (ks, bulk_n, rate)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    ks, bulk_n, rate = best
    alpha = FAMILYWISE_ALPHA / len(values)
    cutoff = -np.log(alpha) / rate
    return {
        "n_drawdowns": int(len(values)),
        "bulk_n": int(bulk_n),
        "bulk_fraction": float(bulk_n / len(values)),
        "rate": float(rate),
        "mean_bulk_drawdown": float(1.0 / rate),
        "ks_distance": float(ks),
        "familywise_alpha": FAMILYWISE_ALPHA,
        "per_drawdown_alpha": float(alpha),
        "cutoff": float(cutoff),
    }


def labels_for_episodes(
    dates: pd.DatetimeIndex, episodes: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """Label each date by trading days until the next crash trough."""
    trough_positions = np.flatnonzero(
        dates.isin(pd.to_datetime(episodes["trough_date"]))
    )
    next_position = np.full(len(dates), np.nan)
    for position in range(len(dates)):
        following = trough_positions[trough_positions > position]
        if len(following):
            next_position[position] = following[0] - position
    result = pd.DataFrame({"date": dates, "days_to_next_crash": next_position})
    result["crash_within_N_days"] = (
        result["days_to_next_crash"].le(horizon).fillna(False).astype("int8")
    )
    return result


def plot_group(group: str, frames: dict[str, dict], path: Path) -> None:
    tickers = list(frames)
    columns = 2
    rows = int(np.ceil(len(tickers) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(13, 4 * rows), squeeze=False)
    for ax, ticker in zip(axes.flat, tickers):
        info = frames[ticker]
        values = info["drawdowns"]
        fit = info["fit"]
        bins = np.linspace(0, max(values.max(), fit["cutoff"]) * 1.05, 35)
        ax.hist(values, bins=bins, density=True, alpha=0.55, color="steelblue")
        x = np.linspace(0, bins[-1], 300)
        ax.plot(x, fit["rate"] * np.exp(-fit["rate"] * x), color="black", lw=1.5)
        ax.axvline(fit["cutoff"], color="crimson", ls="--", lw=1.5)
        ax.set_title(
            f"{ticker}: n={fit['n_drawdowns']}, bulk={fit['bulk_fraction']:.2f}, "
            f"cutoff={fit['cutoff']:.2%}"
        )
        ax.set_xlabel("epsilon-filtered drawdown")
        ax.set_ylabel("density")
    for ax in axes.flat[len(tickers) :]:
        ax.remove()
    fig.suptitle(f"{group}: drawdown bulk and calibrated outlier cutoff")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def output_prefix(group: str) -> str:
    return f"{group.removeprefix('group_c_')}_" if group.startswith("group_c_") else ""


def plot_series_diagnostics(
    group: str,
    prices: pd.Series,
    episodes: pd.DataFrame,
    fit: dict[str, float | int],
    path: Path,
) -> None:
    """Plot normalized price, full drawdown, and calibrated crash markers."""
    crash = episodes[episodes["crash"].eq(1)]
    drawdown = running_drawdown(prices)
    normalized = prices / prices.iloc[0] * 100.0
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axes[0].plot(normalized.index, normalized, color="navy", lw=0.9)
    axes[0].set_ylabel("normalized price")
    axes[0].set_title(
        f"{group} / {prices.name}: price, drawdown, and objective crashes"
    )
    axes[1].plot(drawdown.index, drawdown * 100.0, color="darkorange", lw=0.9)
    axes[1].axhline(fit["cutoff"] * 100.0, color="crimson", ls="--", lw=1.2)
    axes[1].set_ylabel("drawdown (%)")
    axes[1].legend(["running drawdown", f"cutoff {fit['cutoff']:.2%}"], loc="upper left")
    for axis in axes[:2]:
        for _, episode in crash.iterrows():
            start = pd.Timestamp(episode["start_date"])
            trough = pd.Timestamp(episode["trough_date"])
            axis.axvspan(start, trough, color="crimson", alpha=0.12)
            axis.axvline(trough, color="crimson", alpha=0.55, lw=0.8)
    axes[2].scatter(
        pd.to_datetime(episodes["trough_date"]),
        episodes["drawdown"] * 100.0,
        c=np.where(episodes["crash"].eq(1), "crimson", "steelblue"),
        s=np.where(episodes["crash"].eq(1), 34, 15),
        alpha=0.8,
    )
    axes[2].axhline(fit["cutoff"] * 100.0, color="crimson", ls="--", lw=1.2)
    axes[2].set_ylabel("episode drawdown (%)")
    axes[2].set_xlabel("date")
    axes[2].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_group_summary(summary: pd.DataFrame, path: Path) -> None:
    """Plot crash counts and calibrated cutoffs for every reference series."""
    ordered = summary.sort_values(["group", "crash_episodes", "ticker"])
    labels = ordered["group"].str.replace("group_", "", regex=False) + " / " + ordered["ticker"]
    fig, axes = plt.subplots(2, 1, figsize=(16, 14), sharex=True)
    colors = plt.get_cmap("tab20")(np.arange(len(ordered)) % 20)
    axes[0].bar(np.arange(len(ordered)), ordered["crash_episodes"], color=colors)
    axes[0].set_ylabel("independent crash episodes")
    axes[0].set_title("Objective crash episodes by reference series")
    axes[1].bar(np.arange(len(ordered)), ordered["cutoff"] * 100.0, color=colors)
    axes[1].set_ylabel("outlier cutoff (%)")
    axes[1].set_title("Bonferroni-calibrated exponential outlier cutoffs")
    axes[1].set_xticks(np.arange(len(ordered)), labels, rotation=75, ha="right", fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_label_heatmaps(label_frames: dict[int, pd.DataFrame], group: str, path: Path) -> None:
    """Plot binary crash-within-horizon labels for all series in a group."""
    fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(16, 2.5 * len(HORIZONS)), sharex=True)
    if len(HORIZONS) == 1:
        axes = [axes]
    for axis, horizon in zip(axes, HORIZONS):
        labels = label_frames[horizon].copy()
        pivot = labels.pivot(index="ticker", columns="date", values="crash_within_N_days")
        axis.imshow(
            pivot.to_numpy(),
            aspect="auto",
            interpolation="none",
            cmap="Reds",
            vmin=0,
            vmax=1,
            extent=[
                mdates.date2num(pivot.columns.min().to_pydatetime()),
                mdates.date2num(pivot.columns.max().to_pydatetime()),
                -0.5,
                len(pivot.index) - 0.5,
            ],
        )
        axis.set_yticks(np.arange(len(pivot.index)), pivot.index)
        axis.set_ylabel(f"N={horizon}")
        axis.grid(False)
        axis.xaxis_date()
        axis.xaxis.set_major_locator(mdates.AutoDateLocator())
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
    axes[-1].set_xlabel("date")
    fig.suptitle(f"{group}: crash-within-N-days labels (red = 1)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def process_group(group: str, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    group_dir = path.parent
    prefix = ""
    if group.startswith("group_c_"):
        prefix = group.removeprefix("group_c_") + "_"
    all_drawdowns: list[pd.DataFrame] = []
    all_episodes: list[pd.DataFrame] = []
    plot_data: dict[str, dict] = {}
    label_frames = {horizon: [] for horizon in HORIZONS}
    plot_dir = DATA / "plots" / "objective_crash"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for ticker in frame.columns:
        prices = pd.to_numeric(frame[ticker], errors="coerce").dropna()
        full = running_drawdown(prices)
        episodes = pd.DataFrame(epsilon_episodes(prices, EPSILON))
        if episodes.empty:
            continue
        fit = fit_exponential(episodes["drawdown"].to_numpy())
        episodes["group"] = group
        episodes["ticker"] = ticker
        episodes["epsilon"] = EPSILON
        episodes["bulk_rate"] = fit["rate"]
        episodes["bulk_n"] = fit["bulk_n"]
        episodes["bulk_fraction"] = fit["bulk_fraction"]
        episodes["ks_distance"] = fit["ks_distance"]
        episodes["per_drawdown_alpha"] = fit["per_drawdown_alpha"]
        episodes["tail_probability"] = np.exp(-fit["rate"] * episodes["drawdown"])
        episodes["crash"] = (episodes["tail_probability"] < fit["per_drawdown_alpha"]).astype("int8")
        episodes["crash_cutoff"] = fit["cutoff"]
        episodes["episode_id"] = np.arange(1, len(episodes) + 1)
        all_episodes.append(episodes)

        dd_frame = pd.DataFrame(
            {
                "date": prices.index,
                "group": group,
                "ticker": ticker,
                "price": prices.to_numpy(),
                "running_drawdown": full.to_numpy(),
            }
        )
        all_drawdowns.append(dd_frame)
        plot_data[ticker] = {"drawdowns": episodes["drawdown"].to_numpy(), "fit": fit}
        plot_series_diagnostics(
            group,
            prices,
            episodes,
            fit,
            plot_dir / f"{group}_{ticker.replace('^', '')}.png",
        )

        crash_episodes = episodes[episodes["crash"].eq(1)]
        for horizon in HORIZONS:
            labels = labels_for_episodes(prices.index, crash_episodes, horizon)
            labels.insert(1, "group", group)
            labels.insert(2, "ticker", ticker)
            label_frames[horizon].append(labels)

    drawdowns = pd.concat(all_drawdowns, ignore_index=True)
    episodes = pd.concat(all_episodes, ignore_index=True)
    drawdowns.to_csv(group_dir / f"{prefix}drawdowns.csv", index=False, date_format="%Y-%m-%d")
    episodes.to_csv(group_dir / f"{prefix}crash_episodes.csv", index=False, date_format="%Y-%m-%d")
    for horizon, pieces in label_frames.items():
        pd.concat(pieces, ignore_index=True).to_csv(
            group_dir / f"{prefix}labels_N{horizon}.csv",
            index=False,
            date_format="%Y-%m-%d",
        )
    plot_group(group, plot_data, group_dir / f"{prefix}drawdown_exponential_fit.png")
    label_outputs = {
        horizon: pd.concat(pieces, ignore_index=True) for horizon, pieces in label_frames.items()
    }
    plot_label_heatmaps(label_outputs, group, plot_dir / f"{group}_label_heatmaps.png")
    return episodes, drawdowns


def write_report(group: str, episodes: pd.DataFrame, path: Path) -> None:
    lines = [
        f"OBJECTIVE CRASH DETECTION — {group}",
        "=" * (31 + len(group)),
        "",
        f"Epsilon reversal filter: {EPSILON:.2%}.",
        "The exponential bulk fraction was selected by minimizing the one-sample "
        "KS distance across 50%–95% candidate lower-tail fractions.",
        "Crash cutoff: per-drawdown tail alpha = 0.05 / n_drawdowns "
        "(Bonferroni family-wise error control at 5%).",
        "",
        "Fitted parameters and independent episodes by reference series",
        "-------------------------------------------------------------",
    ]
    for ticker, subset in episodes.groupby("ticker", sort=True):
        first = subset.iloc[0]
        crashes = subset[subset["crash"].eq(1)]
        lines.append(
            f"{ticker}: n={len(subset)}, bulk_n={int(first['bulk_n'])}, "
            f"lambda={first['bulk_rate']:.6g}, cutoff={first['crash_cutoff']:.4%}, "
            f"per_drawdown_alpha={first['per_drawdown_alpha']:.3g}, "
            f"crashes={len(crashes)}, KS={first['ks_distance']:.4f}"
        )
    lines.extend(
        [
            "",
            "A crash is an epsilon-filtered local peak-to-trough drawdown whose "
            "fitted exponential survival probability is below the calibrated "
            "per-drawdown alpha. Recovery dates are the first subsequent close "
            "at or above the episode's peak; blank means not recovered in-sample.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary_rows = []
    for group, path in PRICE_FILES.items():
        episodes, drawdowns = process_group(group, path)
        prefix = ""
        if group.startswith("group_c_"):
            prefix = group.removeprefix("group_c_") + "_"
        write_report(group, episodes, path.parent / f"{prefix}crash_detection_report.txt")
        for ticker, subset in episodes.groupby("ticker"):
            summary_rows.append(
                {
                    "group": group,
                    "ticker": ticker,
                    "epsilon": EPSILON,
                    "n_drawdowns": len(subset),
                    "crash_episodes": int(subset["crash"].sum()),
                    "lambda": float(subset.iloc[0]["bulk_rate"]),
                    "cutoff": float(subset.iloc[0]["crash_cutoff"]),
                    "min_tail_probability": float(subset["tail_probability"].min()),
                }
            )
    summary = pd.DataFrame(summary_rows).sort_values(["group", "ticker"])
    summary.to_csv(DATA / "crash_labels_summary.csv", index=False)
    plot_group_summary(summary, DATA / "plots" / "objective_crash" / "all_series_summary.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
