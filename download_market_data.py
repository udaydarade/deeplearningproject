from __future__ import annotations

import io
import time
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import requests
import yfinance as yf


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TODAY = date.today().isoformat()

GROUPS = {
    "group_a_original_indices": {
        "tickers": ["^GSPC", "^DJI", "^IXIC", "^RUT"],
        "start": "1997-01-01",
        "files": ("prices.csv", "returns.csv"),
    },
    "group_b_cross_asset": {
        "tickers": ["SPY", "TLT", "GLD", "USO", "UUP", "HYG"],
        "start": "2007-01-01",
        "files": ("prices.csv", "returns.csv"),
    },
}

GROUP_C = {
    "sector": {
        "tickers": ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"],
        "start": "1997-01-01",
        "files": ("sector_prices.csv", "sector_returns.csv"),
    },
    "crypto": {
        "tickers": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"],
        "start": "2010-01-01",
        "files": ("crypto_prices.csv", "crypto_returns.csv"),
    },
    "international": {
        "tickers": ["^N225", "^GDAXI", "^FTSE", "000001.SS", "^HSI", "^BVSP", "^BSESN"],
        "start": "1997-01-01",
        "files": ("international_prices.csv", "international_returns.csv"),
    },
}


def fetch_yfinance(ticker: str, start: str) -> pd.Series:
    end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    frame = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False)
    if frame.empty:
        raise ValueError("yfinance returned no rows")
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.rename(ticker)


def fetch_stooq(ticker: str, start: str) -> pd.Series:
    symbol = ticker.replace("^", "").lower()
    if ticker.endswith(".SS"):
        symbol = ticker[:-3].lower() + ".cn"
    frame = web.DataReader(symbol, "stooq", start=start, end=TODAY)
    if frame.empty:
        raise ValueError("Stooq returned no rows")
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna().sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.rename(ticker)


def fetch_fred(ticker: str, start: str) -> pd.Series:
    # FRED CSV is a useful last resort for US index series where a matching
    # series exists; the ticker is mapped explicitly to avoid guessing.
    fred_map = {"^GSPC": "SP500", "^DJI": "DJIA", "^IXIC": "NASDAQCOM"}
    series = fred_map.get(ticker)
    if not series:
        raise ValueError("no explicit FRED mapping for this ticker")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text), parse_dates=["observation_date"])
    values = pd.to_numeric(frame[series], errors="coerce")
    result = pd.Series(values.to_numpy(), index=frame["observation_date"], name=ticker)
    return result.loc[start:TODAY].dropna()


def fetch_ticker(ticker: str, start: str) -> tuple[pd.Series | None, str]:
    errors = []
    for method in (fetch_yfinance, fetch_stooq, fetch_fred):
        try:
            result = method(ticker, start)
            if result.empty:
                raise ValueError("returned no usable Close values")
            return result, ""
        except Exception as exc:  # each fallback is reported below
            errors.append(f"{method.__name__}: {exc}")
            time.sleep(0.5)
    return None, " | ".join(errors)


def summarize_raw(raw: dict[str, pd.Series], failed: dict[str, str], group_start: str, group_name: str) -> list[dict]:
    if not raw:
        return []
    union = pd.DatetimeIndex(sorted(set().union(*(set(s.index) for s in raw.values()))))
    rows = []
    for ticker, series in raw.items():
        aligned_to_union = series.reindex(union)
        rows.append(
            {
                "group": group_name,
                "ticker": ticker,
                "start": series.index.min().date().isoformat(),
                "end": series.index.max().date().isoformat(),
                "trading_days": int(series.size),
                "missing_before_alignment_pct": round(float(aligned_to_union.isna().mean() * 100), 4),
                "status": "ok",
                "failure": "",
            }
        )
    for ticker, error in failed.items():
        rows.append(
            {
                "group": group_name,
                "ticker": ticker,
                "start": "",
                "end": "",
                "trading_days": 0,
                "missing_before_alignment_pct": 100.0,
                "status": "FAILED",
                "failure": error,
            }
        )
    return rows


def process_group(group_name: str, config: dict, out_dir: Path, plot_dir: Path) -> tuple[list[dict], pd.DataFrame]:
    raw, failed = {}, {}
    for ticker in config["tickers"]:
        series, error = fetch_ticker(ticker, config["start"])
        if series is None:
            failed[ticker] = error
        else:
            raw[ticker] = series
    summary = summarize_raw(raw, failed, config["start"], group_name)
    if not raw:
        return summary, pd.DataFrame()
    prices = pd.concat(raw.values(), axis=1, join="inner").sort_index().dropna(how="any")
    prices.columns = list(raw)
    returns = np.log(prices / prices.shift(1)).dropna(how="any")
    out_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out_dir / config["files"][0], index_label="date")
    returns.to_csv(out_dir / config["files"][1], index_label="date")
    normalized = prices.div(prices.iloc[0]).mul(100)
    ax = normalized.plot(figsize=(14, 7), linewidth=1)
    ax.set(title=f"{group_name}: normalized Close prices (common start = {prices.index.min().date()})",
           xlabel="Date", ylabel="Price (start = 100)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{group_name}.png", dpi=150)
    plt.close(fig)
    print(f"{group_name}: common aligned start={prices.index.min().date()}, end={prices.index.max().date()}, rows={len(prices)}")
    if failed:
        print(f"FAILED {group_name}: {', '.join(failed)}")
    return summary, prices


def main() -> None:
    all_summary = []
    for name, config in GROUPS.items():
        summary, _ = process_group(name, config, DATA / name, DATA / "plots")
        all_summary.extend(summary)
    for name, config in GROUP_C.items():
        summary, _ = process_group("group_c_" + name, config, DATA / "group_c_panel", DATA / "plots")
        all_summary.extend(summary)
    summary_frame = pd.DataFrame(all_summary)
    summary_frame.to_csv(DATA / "summary.csv", index=False)
    print("\nRaw-data summary (missing percentage is relative to the union of dates returned by tickers in that subgroup):")
    print(summary_frame.to_string(index=False))
    print("\nAlignment: strict inner join on dates within each subgroup, then drop remaining NaNs.")
    print("No synthetic values or forward-filling were used. See data/summary.csv for failures and details.")


if __name__ == "__main__":
    main()
