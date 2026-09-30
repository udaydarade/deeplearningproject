"""Run the requested n=50 TDA/CSD extension and write reproducible artifacts.

The script uses only observed prices: yfinance is tried first for the three
stock sectors, followed by Stooq. Failed tickers are reported and are never
filled or fabricated. Existing Group B, crypto, and international files are
read as-is. For each market group, H1 persistence-landscape features are
computed from the complete multivariate return window, while CSD features use
the group's declared reference return (SPY for Group B). N15 labels use the
reference price's own empirical 2nd-percentile forward minimum return.

Outputs under ``data/extension``:
``cross_asset_panel.csv`` (the exact requested long-panel schema),
``data_download_summary.csv``, ``group_summary.csv``, ``report.txt``, and
one diagnostic PNG per market group.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from arch import arch_model
from ripser import ripser
import yfinance as yf

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = DATA / "extension"
N = 50
LABEL_QUANTILE = 0.02

SECTORS = {
    "financials": ["JPM", "BAC", "GS", "C", "MS", "WFC"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PXD"],
    "tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD"],
}
SECTOR_START = "1997-01-01"


def close_yfinance(ticker: str, start: str) -> pd.Series:
    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frame = yf.download(ticker, start=start, end=end, auto_adjust=False,
                        progress=False, threads=False)
    if frame.empty:
        raise ValueError("yfinance returned no rows")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.rename(ticker)


def close_stooq(ticker: str, start: str) -> pd.Series:
    symbol = ticker.lower()
    url = f"https://stooq.com/q/d/l/?s={symbol}&d1={start.replace('-', '')}&i=d"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text), parse_dates=["Date"])
    if frame.empty or "Close" not in frame:
        raise ValueError("Stooq returned no rows")
    return pd.Series(pd.to_numeric(frame["Close"], errors="coerce").to_numpy(),
                     index=frame["Date"], name=ticker).dropna()


def download_sector_data() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Download sector prices and return (successful frames, per-ticker report)."""
    frames: dict[str, pd.DataFrame] = {}
    report: list[dict] = []
    for group, tickers in SECTORS.items():
        raw: dict[str, pd.Series] = {}
        for ticker in tickers:
            error_messages = []
            for fetch in (close_yfinance, close_stooq):
                try:
                    series = fetch(ticker, SECTOR_START)
                    if not series.empty:
                        raw[ticker] = series
                        break
                except Exception as exc:  # retain the factual failure in the report
                    error_messages.append(f"{fetch.__name__}: {exc}")
                time.sleep(0.2)
            if ticker in raw:
                s = raw[ticker]
                report.append({"market_group": group, "ticker": ticker,
                               "actual_start": s.index.min().date().isoformat(),
                               "actual_end": s.index.max().date().isoformat(),
                               "trading_days": len(s), "status": "ok", "failure": ""})
            else:
                report.append({"market_group": group, "ticker": ticker,
                               "actual_start": "", "actual_end": "", "trading_days": 0,
                               "status": "FAILED", "failure": " | ".join(error_messages)})
        if raw:
            frames[group] = pd.concat(raw.values(), axis=1, join="inner").dropna()
            frames[group].columns = list(raw)
    OUT.mkdir(parents=True, exist_ok=True)
    for group, prices in frames.items():
        prices.to_csv(OUT / f"{group}_prices.csv", index_label="date")
        np.log(prices / prices.shift(1)).dropna().to_csv(
            OUT / f"{group}_returns.csv", index_label="date")
    return frames, pd.DataFrame(report)


def finite_h1_bars(points: np.ndarray) -> np.ndarray:
    diagrams = ripser(points, maxdim=1, metric="euclidean")["dgms"]
    if len(diagrams) < 2:
        return np.empty((0, 2))
    bars = diagrams[1]
    return bars[np.isfinite(bars).all(axis=1)]


def tda_features(bars: np.ndarray) -> tuple[float, float, float, int]:
    if bars.size == 0:
        return 0.0, 0.0, 0.0, 0
    births, deaths = bars[:, 0], bars[:, 1]
    grid = np.linspace(float(births.min()), float(deaths.max()), 401)
    tents = np.maximum(0.0, np.minimum(grid[None, :] - births[:, None],
                                        deaths[:, None] - grid[None, :]))
    landscapes = np.sort(tents, axis=0)[::-1]
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    l1 = float(trapz(landscapes.sum(axis=0), grid))
    l2 = float(np.sqrt(trapz((landscapes ** 2).sum(axis=0), grid)))
    lengths = deaths - births
    probs = lengths / lengths.sum() if lengths.sum() > 0 else lengths
    entropy = float(-(probs * np.log(probs)).sum()) if len(probs) else 0.0
    return l1, l2, entropy, int(len(bars))


def garch_sigma(values: np.ndarray) -> float:
    try:
        result = arch_model(values * 100, mean="Constant", vol="GARCH",
                            p=1, q=1, dist="normal", rescale=False).fit(
                                disp="off", show_warning=False, options={"maxiter": 200})
        volatility = np.asarray(result.conditional_volatility, dtype=float)
        return float(volatility[-1] / 100)
    except Exception:
        return np.nan


def crash_labels(prices: pd.Series, horizon: int = 15) -> tuple[pd.Series, float, int]:
    values = prices.to_numpy(float)
    forward = np.full(len(values), np.nan)
    for i in range(len(values) - horizon):
        forward[i] = np.min((values[i + 1:i + horizon + 1] - values[i]) / values[i])
    forward = pd.Series(forward, index=prices.index).dropna()
    threshold = float(forward.quantile(LABEL_QUANTILE))
    labels = (forward <= threshold).astype("int8")
    episodes = int((labels.astype(bool) & ~labels.astype(bool).shift(
        1, fill_value=False)).sum())
    return labels, threshold, episodes


def run_group(group: str, prices: pd.DataFrame, reference: str) -> tuple[pd.DataFrame, dict]:
    returns = np.log(prices / prices.shift(1)).dropna(how="any")
    ref = returns[reference]
    labels, threshold, episodes = crash_labels(prices[reference])
    rows = []
    values = returns.to_numpy(float)
    for end in range(N, len(returns) + 1):
        block = values[end - N:end]
        std = block.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        l1, l2, entropy, betti = tda_features(finite_h1_bars((block - block.mean(0)) / std))
        ref_block = ref.iloc[end - N:end].to_numpy(float)
        rows.append({
            "market_group": group, "date": returns.index[end - 1],
            "TDA_L1_norm": l1, "TDA_L2_norm": l2, "TDA_entropy": entropy,
            "TDA_betti1_count": betti,
            "CSD_AC1": float(pd.Series(ref_block).autocorr(1)),
            "CSD_variance": float(np.var(ref_block, ddof=1)),
            "CSD_skew": float(pd.Series(ref_block).skew()),
            "CSD_garch_vol": garch_sigma(ref_block),
            "crash_label_N15": int(labels.get(returns.index[end - 1], 0)),
        })
    panel = pd.DataFrame(rows, columns=[
        "market_group", "date", "TDA_L1_norm", "TDA_L2_norm", "TDA_entropy",
        "TDA_betti1_count", "CSD_AC1", "CSD_variance", "CSD_skew",
        "CSD_garch_vol", "crash_label_N15"])
    summary = {"market_group": group, "reference": reference,
               "actual_start": prices.index.min().date().isoformat(),
               "actual_end": prices.index.max().date().isoformat(),
               "price_rows": len(prices), "return_rows": len(returns),
               "panel_rows": len(panel), "N15_threshold": threshold,
               "N15_positive_days": int(labels.sum()), "N15_episodes": episodes,
               "garch_missing": int(panel["CSD_garch_vol"].isna().sum())}
    return panel, summary


def load_existing() -> dict[str, tuple[pd.DataFrame, str]]:
    specs = {
        "group_b_cross_asset": (DATA / "group_b_cross_asset" / "prices.csv", "SPY"),
        "group_c_crypto": (DATA / "group_c_panel" / "crypto_prices.csv", "BTC-USD"),
        "group_c_international": (DATA / "group_c_panel" / "international_prices.csv", "^N225"),
    }
    loaded = {}
    for group, (path, reference) in specs.items():
        prices = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        prices = prices.apply(pd.to_numeric, errors="coerce").dropna(how="any")
        if reference in prices and len(prices.columns) >= 3:
            loaded[group] = (prices, reference)
    return loaded


def main() -> None:
    sector_frames, download_report = download_sector_data()
    groups = load_existing()
    for group, prices in sector_frames.items():
        if len(prices.columns) >= 3:
            groups[group] = (prices, SECTORS[group][0])
    panels, summaries = [], []
    for group, (prices, reference) in groups.items():
        panel, summary = run_group(group, prices, reference)
        panels.append(panel)
        summaries.append(summary)
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        indexed = panel.set_index("date")
        axes[0].plot(indexed.index, indexed["TDA_L1_norm"], label="TDA L1")
        axes[0].plot(indexed.index, indexed["TDA_L2_norm"], label="TDA L2")
        axes[0].legend(); axes[0].grid(alpha=.2)
        axes[1].plot(indexed.index, indexed["CSD_AC1"], label="CSD AC1")
        axes[1].plot(indexed.index, indexed["CSD_garch_vol"], label="CSD GARCH vol")
        axes[1].legend(); axes[1].grid(alpha=.2)
        fig.suptitle(f"{group} (n={N}, reference={reference})")
        fig.tight_layout()
        fig.savefig(OUT / f"{group}_diagnostics.png", dpi=140)
        plt.close(fig)
    panel = pd.concat(panels, ignore_index=True)
    panel.to_csv(OUT / "cross_asset_panel.csv", index=False, date_format="%Y-%m-%d")
    download_report.to_csv(OUT / "data_download_summary.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "group_summary.csv", index=False)
    report = [
        "Cross-asset TDA/CSD extension report",
        f"Window n={N}; TDA uses z-scored complete multivariate return windows and finite H1 bars.",
        "CSD uses the declared reference return; GARCH is refit on every window (returns scaled by 100).",
        "N15 labels are reference-price forward 15-trading-day minimum returns <= that series' empirical 2nd percentile.",
        "No forward filling, synthetic prices, or cross-series label thresholds were used.",
        "",
        summary.to_string(index=False),
        "",
        "Download details:",
        download_report.to_string(index=False) if not download_report.empty else "No sector download rows.",
    ]
    (OUT / "report.txt").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Wrote {len(panel):,} rows to {OUT / 'cross_asset_panel.csv'}")


if __name__ == "__main__":
    main()
