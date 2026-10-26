"""Try current S&P 500 constituents, with existing project stocks as fallback."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "v3"
OUT.mkdir(parents=True, exist_ok=True)
EXISTING = ["JPM", "BAC", "GS", "C", "MS", "WFC", "XOM", "CVX", "COP",
            "SLB", "EOG", "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD"]


def constituents() -> list[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        frame = next(t for t in tables if "Symbol" in t.columns)
        symbols = frame.Symbol.astype(str).str.replace(".", "-", regex=False).tolist()
        (OUT / "sp500_constituents.csv").write_text(
            pd.DataFrame({"ticker": symbols}).to_csv(index=False), encoding="utf-8")
        return sorted(set(symbols + EXISTING))
    except Exception as exc:  # network or upstream table failure
        print(f"[constituents] unavailable ({exc}); using existing stocks only", flush=True)
        pd.DataFrame({"ticker": EXISTING}).to_csv(OUT / "sp500_constituents.csv", index=False)
        return EXISTING


def main() -> None:
    tickers = constituents()
    frames, failed = {}, []
    for start in range(0, len(tickers), 50):
        chunk = tickers[start:start + 50]
        data = None
        for attempt in range(3):
            try:
                data = yf.download(chunk, period="max", auto_adjust=True,
                                   group_by="ticker", threads=True, progress=False)
                break
            except Exception as exc:
                print(f"[retry {start} attempt {attempt + 1}] {exc}", flush=True)
                time.sleep(4 * (attempt + 1))
        for ticker in chunk:
            try:
                if data is not None and isinstance(data.columns, pd.MultiIndex) and ticker in data.columns.get_level_values(0):
                    s = data[ticker]["Close"].dropna()
                elif data is not None and len(chunk) == 1:
                    s = data["Close"].dropna()
                else:
                    s = pd.Series(dtype=float)
                if len(s) < 250:
                    one = yf.download(ticker, period="max", auto_adjust=True, progress=False)
                    if "Close" in one:
                        s = one["Close"].dropna()
                if len(s) < 250:
                    failed.append(ticker)
                else:
                    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
                    frames[ticker] = s.rename(ticker)
            except Exception:
                failed.append(ticker)
        print(f"[stocks] {start + len(chunk)}/{len(tickers)}; downloaded={len(frames)}; failed={len(failed)}", flush=True)
    prices = pd.concat(frames.values(), axis=1).sort_index() if frames else pd.DataFrame()
    prices.to_parquet(OUT / "prices_sp500_stocks.parquet")
    (OUT / "stocks_download_failed.json").write_text(json.dumps(sorted(set(failed)), indent=2))
    print(f"[stocks] output={prices.shape}; failed={sorted(set(failed))}", flush=True)


if __name__ == "__main__":
    main()
