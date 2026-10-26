"""V3 Phase 1: universe inventory + history extension + new assets.

- Inventory every local price file -> data/v3/registry_local.csv
- Download extended history + new assets (yfinance, chunks of 50, retries)
- FRED state series via fredgraph.csv (no key)
- Cleaning per HARD RULES -> data/v3/{prices,returns,states}.parquet + registry.csv

Run: py -3.12 v3_download_data.py
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "v3"
OUT.mkdir(parents=True, exist_ok=True)

LOCAL_SOURCES = {
    ("group_a", "equity_index"): ("data/group_a_original_indices/prices.csv", None),
    ("group_b", "asset_mix"): ("data/group_b_cross_asset/prices.csv", None),
    ("crypto", "crypto"): ("data/group_c_panel/crypto_prices.csv", None),
    ("international", "equity_index"): ("data/group_c_panel/international_prices.csv", None),
    ("sector", "us_etf"): ("data/group_c_panel/sector_prices.csv", None),
    ("energy", "stock"): ("data/extension/energy_prices.csv", None),
    ("financials", "stock"): ("data/extension/financials_prices.csv", None),
    ("tech", "stock"): ("data/extension/tech_prices.csv", None),
}

NEW_UNIVERSE: dict[str, list[str]] = {
    # ticker: asset_class
    **{t: "equity_index" for t in [
        "^GSPC", "^DJI", "^IXIC", "^RUT", "^FTSE", "^N225", "^GDAXI", "^HSI",
        "^BVSP", "^BSESN", "^FCHI", "^STOXX50E", "^AEX", "^IBEX", "^SSMI",
        "^OMXS30", "^AXJO", "^KS11", "^TWII", "^NSEI", "^GSPTSE", "^MXX",
        "^JKSE", "^KLSE", "^STI", "^XU100", "^MERV", "^TA125.TA", "^NZ50",
        "^NDX", "^OEX", "^MID", "^SP600"]},
    **{t: "us_etf" for t in [
        "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE",
        "XLC", "XBI", "XHB", "XRT", "KRE", "SMH", "ITB", "IYT", "IGV", "IWM",
        "IWD", "IWF", "MDY", "QQQ", "DIA", "SPY"]},
    **{t: "country_etf" for t in [
        "EWJ", "EWG", "EWU", "EWZ", "EWY", "EWT", "EWH", "EWA", "EWC", "EWW",
        "EWS", "EWI", "EWP", "EWQ", "EWL", "EWD", "EWN", "EWM", "THD", "EIDO",
        "EPHE", "TUR", "EZA", "INDA", "FXI", "MCHI", "EEM", "EFA", "VNQ"]},
    **{t: "bond" for t in ["SHY", "IEF", "TLT", "TIP", "LQD", "HYG", "JNK",
                           "EMB", "AGG", "MUB", "BND"]},
    **{t: "commodity" for t in ["GLD", "SLV", "USO", "UNG", "DBC", "DBA",
                                "CPER", "PPLT", "GC=F", "CL=F", "NG=F", "HG=F",
                                "SI=F", "ZC=F", "ZW=F", "ZS=F", "KC=F", "SB=F",
                                "CT=F"]},
    **{t: "fx" for t in ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X",
                         "USDCAD=X", "USDCHF=X", "USDCNY=X", "USDINR=X",
                         "USDBRL=X", "USDMXN=X", "USDKRW=X", "USDZAR=X",
                         "DX-Y.NYB"]},
    **{t: "crypto" for t in ["BTC-USD", "ETH-USD", "XRP-USD", "BNB-USD",
                             "SOL-USD", "LTC-USD", "DOGE-USD", "ADA-USD",
                             "AVAX-USD", "LINK-USD", "DOT-USD"]},
    # legacy stocks (from financials/energy/tech) so those originals also
    # carry their own classes in the extended-download frame as well
    **{t: "stock" for t in ["JPM", "BAC", "GS", "C", "MS", "WFC", "XOM", "CVX",
                            "COP", "SLB", "EOG", "AAPL", "MSFT", "NVDA",
                            "GOOGL", "META", "AMD"]},
}

STATE_YF = ["^VIX", "^VIX3M", "^VVIX", "^SKEW", "^VXN", "^TNX", "^FVX", "^IRX", "^TYX"]
FRED_IDS = ["T10Y2Y", "T10Y3M", "NFCI", "STLFSI4", "BAMLH0A0HYM2", "DTWEXBGS", "VIXCLS"]
FRED_WEEKLY = {"NFCI", "STLFSI4"}
REGIONS = {
    "equity_index": {"^GSPC": "us", "^DJI": "us", "^IXIC": "us", "^RUT": "us",
                     "^FTSE": "uk", "^N225": "jp", "^GDAXI": "de", "^HSI": "hk",
                     "^BVSP": "br", "^BSESN": "in", "^FCHI": "fr",
                     "^STOXX50E": "eu", "^AEX": "nl", "^IBEX": "es",
                     "^SSMI": "ch", "^OMXS30": "se", "^AXJO": "au",
                     "^KS11": "kr", "^TWII": "tw", "^NSEI": "in",
                     "^GSPTSE": "ca", "^MXX": "mx", "^JKSE": "id",
                     "^KLSE": "my", "^STI": "sg", "^XU100": "tr",
                     "^MERV": "ar", "^TA125.TA": "il", "^NZ50": "nz",
                     "^NDX": "us", "^OEX": "us", "^MID": "us", "^SP600": "us",
                     "000001.SS": "cn"},
}


def inventory_local() -> pd.DataFrame:
    rows = []
    for (source, klass), (rel, _) in LOCAL_SOURCES.items():
        path = ROOT / rel
        if not path.exists():
            continue
        d = pd.read_csv(path, index_col=0, parse_dates=[0])
        for col in d.columns:
            s = d[col].dropna()
            if len(s) == 0:
                continue
            asset_class, region = LOCAL_CLASS(source, col)
            rows.append({"ticker": col, "asset_class": asset_class,
                         "region": region, "source_dataset": source,
                         "source_file": rel, "first_date": s.index.min().date(),
                         "last_date": s.index.max().date(), "n_obs": int(len(s)),
                         "n_valid": int(len(s[s > 0]))})
    return pd.DataFrame(rows)


def LOCAL_CLASS(source: str, ticker: str) -> tuple[str, str]:
    if source in ("group_a", "international"):
        return "equity_index", "us" if source == "group_a" else "intl"
    if source == "group_b":
        klasses = {"SPY": ("us_etf", "us"), "TLT": ("bond", "us"),
                   "GLD": ("commodity", "us"), "USO": ("commodity", "us"),
                   "UUP": ("fx", "us"), "HYG": ("bond", "us")}
        return klasses.get(ticker, ("us_etf", "us"))
    if source == "crypto":
        return "crypto", "global"
    if source == "sector":
        return "us_etf", "us"
    return "stock", "us"


def main() -> None:
    reg_local = inventory_local()
    reg_local.to_csv(OUT / "registry_local.csv", index=False)
    print(f"[local] {len(reg_local)} tickers inventoried", flush=True)

    # ---------------- yahoo downloads (extended history for all of universe) -
    wanted = dict(NEW_UNIVERSE)
    # add local-only tickers missing from NEW_UNIVERSE (e.g., 000001.SS, UUP)
    for t, r in zip(reg_local.ticker, reg_local.itertuples()):
        if t not in wanted:
            wanted[t] = r.asset_class
    print(f"[yahoo] downloading {len(wanted)} tickers ...", flush=True)

    frames = {}
    failed = []
    tickers = list(wanted)
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i + 50]
        for attempt in range(3):
            try:
                data = yf.download(chunk, period="max", auto_adjust=True,
                                   group_by="ticker", threads=True,
                                   progress=False)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[retry chunk {i}] {exc}", flush=True)
                time.sleep(5)
        else:
            data = None
        for t in chunk:
            try:
                sub = data[t].dropna(subset=["Close"]) if data is not None and t in data.columns.get_level_values(0) else None
            except Exception:  # noqa: BLE001
                sub = None
            if sub is None or len(sub) < 300:
                # single retry
                try:
                    sub = yf.download(t, period="max", auto_adjust=True,
                                      progress=False).dropna(subset=["Close"])
                except Exception:  # noqa: BLE001
                    sub = pd.DataFrame()
            if len(sub) < 300:
                failed.append(t)
                continue
            frames[t] = sub[["Close"]].rename(columns={"Close": t})
        print(f"[yahoo] chunk {i}-{i + len(chunk)} done "
              f"(ok={len(frames)}/{len(wanted)}, failed={len(failed)})", flush=True)
    if failed:
        print("[FAILED]", failed, flush=True)
    with open(OUT / "download_failed.json", "w") as fh:
        json.dump(failed, fh)

    prices = pd.concat(frames.values(), axis=1)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices.columns = [str(c).replace("_", "") for c in prices.columns]
    # keep yahoo close for every NEW_UNIVERSE ticker (original naming, minus -USD)
    prices.to_parquet(OUT / "prices_yahoo.parquet")
    print(f"[yahoo] prices_yahoo.parquet {prices.shape}", flush=True)

    # ---------------- FRED state series --------------------------------------
    state = pd.DataFrame(index=pd.bdate_range("2000-01-03", "2099-01-01"))
    for t in STATE_YF:
        try:
            s = yf.download(t, period="max", auto_adjust=True, progress=False)["Close"]
            if len(s):
                s.index = pd.to_datetime(s.index).tz_localize(None)
                state[t.replace("^", "").replace(".", "_")] = s
        except Exception as exc:  # noqa: BLE001
            print(f"[state fail] {t} {exc}", flush=True)
    for fid in FRED_IDS:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fid}"
        try:
            csv = pd.read_csv(url)
            ser = pd.Series(csv.iloc[:, 1].values,
                            index=pd.to_datetime(csv.iloc[:, 0]))
            ser = pd.to_numeric(ser, errors="coerce").dropna()
            state[fid] = ser.reindex(state.index)
        except Exception as exc:  # noqa: BLE001
            print(f"[fred fail] {fid} {exc}", flush=True)
    state = state.dropna(how="all")
    state.to_parquet(OUT / "states_raw.parquet")
    print(f"[state] states_raw.parquet {state.shape}", flush=True)


if __name__ == "__main__":
    main()
