"""V3 Phase 1 cleaning and Phase 2 past-only expanding crash labels.

Reads the V3 download snapshot plus every local price CSV and writes only to
data/v3/. Run with ``py -3.12 v3_prepare_data.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v3_download_data import FRED_WEEKLY, NEW_UNIVERSE, OUT, REGIONS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LOCAL_REGISTRY = OUT / "registry_local.csv"
REGION_OVERRIDES = {
    "^FCHI": "fr", "^STOXX50E": "eu", "^AEX": "nl", "^IBEX": "es",
    "^SSMI": "ch", "^AXJO": "au", "^KS11": "kr", "^TWII": "tw",
    "^NSEI": "in", "^GSPTSE": "ca", "^MXX": "mx", "^JKSE": "id",
    "^KLSE": "my", "^STI": "sg", "^MERV": "ar", "^TA125.TA": "il",
    "^NZ50": "nz", "^NDX": "us", "^OEX": "us", "^MID": "us", "^SP600": "us",
    "000001.SS": "cn", "EWJ": "jp", "EWG": "de", "EWU": "uk", "EWZ": "br",
    "EWY": "kr", "EWT": "tw", "EWH": "hk", "EWA": "au", "EWC": "ca",
    "EWW": "mx", "EWS": "sg", "EWI": "it", "EWP": "es", "EWQ": "fr",
    "EWL": "ch", "EWD": "se", "EWN": "nl", "EWM": "my", "THD": "th",
    "EIDO": "id", "EPHE": "ph", "TUR": "tr", "EZA": "za", "INDA": "in",
    "FXI": "cn", "MCHI": "cn", "EEM": "emerging", "EFA": "developed", "VNQ": "us",
}


def _local_price_files() -> list[Path]:
    return sorted(p for p in DATA.rglob("*.csv")
                  if "price" in p.name.lower() and "data/v3" not in p.as_posix())


def _read_local_prices(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=[0])
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    return frame.apply(pd.to_numeric, errors="coerce")


def load_price_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    sources: dict[str, pd.Series] = {}
    metadata: dict[str, dict] = {}
    registry_local = pd.read_csv(LOCAL_REGISTRY) if LOCAL_REGISTRY.exists() else pd.DataFrame()
    local_meta = {str(r.ticker): r._asdict()
                  for r in registry_local.itertuples(index=False)}
    yahoo_path = OUT / "prices_yahoo.parquet"
    for yahoo_path in (yahoo_path, OUT / "prices_sp500_stocks.parquet"):
      if yahoo_path.exists():
        yahoo = pd.read_parquet(yahoo_path)
        yahoo.index = pd.to_datetime(yahoo.index).tz_localize(None).normalize()
        for ticker in yahoo.columns:
            ticker = str(ticker)
            s = pd.to_numeric(yahoo[ticker], errors="coerce")
            sources[ticker] = sources[ticker].combine_first(s) if ticker in sources else s
            klass = NEW_UNIVERSE.get(ticker, local_meta.get(ticker, {}).get("asset_class", "stock"))
            metadata[ticker] = {
                "ticker": ticker, "asset_class": klass,
                "region": REGION_OVERRIDES.get(ticker, REGIONS.get(klass, {}).get(ticker, "global" if klass in {"crypto", "fx"} else "us")),
                "source_dataset": local_meta.get(ticker, {}).get("source_dataset", "v3_universe"),
                "source_file": str(yahoo_path.relative_to(ROOT)),
                "source_type": "yahoo_auto_adjusted",
            }

    local_seen: dict[str, list[tuple[Path, pd.Series]]] = {}
    for path in _local_price_files():
        rel = path.relative_to(ROOT).as_posix()
        frame = _read_local_prices(path)
        for ticker in frame:
            ticker = str(ticker)
            local_seen.setdefault(ticker, []).append((path, frame[ticker]))
            if ticker not in metadata:
                record = local_meta.get(ticker, {})
                metadata[ticker] = {
                    "ticker": ticker,
                    "asset_class": record.get("asset_class", "stock"),
                    "region": record.get("region", "us"),
                    "source_dataset": record.get("source_dataset", "local"),
                    "source_file": rel,
                    "source_type": "local_price_csv",
                }

    # Prefer auto-adjusted downloaded series, filling only dates absent from
    # them with original local observations. Preserve duplicate local symbols
    # under their original source in the registry, but use one canonical series.
    for ticker, candidates in local_seen.items():
        local_series = candidates[0][1].sort_index()
        for _, candidate in candidates[1:]:
            local_series = local_series.combine_first(candidate.sort_index())
        if ticker in sources:
            sources[ticker] = sources[ticker].combine_first(local_series)
        else:
            sources[ticker] = local_series

    prices = pd.concat(sources, axis=1).sort_index()
    prices.columns = [str(c) for c in prices.columns]
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    return prices, pd.DataFrame(metadata.values())


def clean_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start, end = prices.index.min(), prices.index.max()
    calendar = pd.bdate_range(start, end, name="date")
    prices = prices.loc[prices.index.dayofweek < 5].reindex(calendar)
    raw_missing = prices.isna().sum()
    nonpositive = (prices <= 0).sum()
    prices = prices.where(prices > 0)
    before_fill_missing = prices.isna()
    prices = prices.ffill(limit=3)
    filled_missing = prices.isna().sum()
    filled_count = before_fill_missing.sum()
    returns = np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan).astype("float32")
    prices = prices.astype("float32")
    simple_abs = np.expm1(returns.astype("float64")).abs()
    flags = (simple_abs > 0.40)
    tickers = []
    for ticker in flags.columns:
        if str(ticker).endswith("-USD"):
            continue
        dates = flags.index[flags[ticker].fillna(False)]
        for date in dates:
            tickers.append({"ticker": ticker, "date": date,
                            "simple_return": float(np.expm1(returns.at[date, ticker]))})
    cleaning = pd.DataFrame({
        "ticker": prices.columns,
        "missing_before_fill": raw_missing.reindex(prices.columns).to_numpy(),
        "nonpositive_prices": nonpositive.reindex(prices.columns).to_numpy(),
        "filled_at_most_3_sessions": filled_count.reindex(prices.columns).to_numpy(),
        "missing_after_fill": filled_missing.reindex(prices.columns).to_numpy(),
        "zero_return_share_from_fill": [
            float((filled_count[c] / max(prices[c].notna().sum(), 1)))
            for c in prices.columns],
        "large_abs_return_gt_40pct_count_ex_crypto": [
            int((flags[c] & (not str(c).endswith("-USD"))).sum()) for c in prices.columns],
    })
    large = pd.DataFrame(tickers, columns=["ticker", "date", "simple_return"])
    return prices, returns, cleaning, large


def clean_states() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = OUT / "states_raw.parquet"
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    raw = pd.read_parquet(path)
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    calendar = pd.bdate_range(raw.index.min(), raw.index.max(), name="date")
    states = raw.reindex(calendar)
    meta = []
    for col in states:
        observed = states[col].dropna()
        if observed.empty or (observed.index.max() - observed.index.min()).days < 365 * 15:
            meta.append({"ticker": col, "asset_class": "state", "region": "us",
                         "source_dataset": "state", "source_file": "data/v3/states_raw.parquet",
                         "first_date": None if observed.empty else observed.index.min().date(),
                         "last_date": None if observed.empty else observed.index.max().date(),
                         "n_obs": int(len(observed)), "n_valid": int(len(observed)),
                         "kept": False, "drop_reason": "under_15_years"})
            continue
        delay = 5 if col in FRED_WEEKLY else (1 if col in {
            "T10Y2Y", "T10Y3M", "NFCI", "STLFSI4", "BAMLH0A0HYM2", "DTWEXBGS", "VIXCLS"} else 0)
        if delay:
            states[col] = states[col].shift(delay)
        states[col] = states[col].ffill(limit=3)
        meta.append({"ticker": col, "asset_class": "state", "region": "us",
                     "source_dataset": "state", "source_file": "data/v3/states_raw.parquet",
                     "first_date": observed.index.min().date(),
                     "last_date": observed.index.max().date(), "n_obs": int(len(observed)),
                     "n_valid": int(len(observed)), "kept": True,
                     "publication_lag_sessions": delay})
    kept = [r["ticker"] for r in meta if r.get("kept")]
    return states[kept].astype("float32"), pd.DataFrame(meta)


def build_registry(prices: pd.DataFrame, meta: pd.DataFrame,
                   states_meta: pd.DataFrame) -> pd.DataFrame:
    price_meta = meta.set_index("ticker").to_dict(orient="index")
    rows = []
    for ticker in prices:
        obs = prices[ticker].dropna()
        info = price_meta.get(ticker, {})
        rows.append({"ticker": ticker, "asset_class": info.get("asset_class", "stock"),
                     "region": info.get("region", "unknown"),
                     "source_dataset": info.get("source_dataset", "v3_universe"),
                     "source_file": info.get("source_file", "unknown"),
                     "first_date": obs.index.min().date() if len(obs) else None,
                     "last_date": obs.index.max().date() if len(obs) else None,
                     "n_obs": int(len(obs)), "n_valid": int(len(obs)), "is_state": False})
    for rec in states_meta.to_dict(orient="records"):
        if rec.get("kept"):
            rec["is_state"] = True
            rows.append(rec)
    return pd.DataFrame(rows)


def forward_min(prices: pd.Series, horizon: int) -> pd.Series:
    values = prices.to_numpy(dtype=np.float64)
    out = np.full(values.size, np.nan, dtype=np.float32)
    if values.size <= horizon:
        return pd.Series(out, index=prices.index)
    future = np.lib.stride_tricks.sliding_window_view(values[1:], horizon)
    out[:values.size - horizon] = np.min(
        future / values[:values.size - horizon, None] - 1.0, axis=1).astype(np.float32)
    out[-horizon:] = np.nan
    return pd.Series(out, index=prices.index)


def make_labels(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    configs = [(15, .02, "BASE"), (5, .02, "N5_p2"), (10, .02, "N10_p2"),
               (30, .02, "N30_p2"), (15, .01, "N15_p1"), (15, .05, "N15_p5"),
               (15, .10, "N15_p10")]
    blocks, summary = [], []
    for ticker in prices:
        p = prices[ticker].dropna()
        if len(p) < 2:
            continue
        part = pd.DataFrame({"asset": ticker, "date": p.index})
        for horizon, quantile, name in configs:
            m = forward_min(p, horizon)
            # At forecast origin t, only origins whose full N-session outcome
            # ended on or before t are available: shift m by N rows first.
            prior = m.shift(horizon)
            threshold = prior.expanding(min_periods=750).quantile(quantile)
            label = (m <= threshold).astype("float32")
            label[m.isna() | threshold.isna()] = np.nan
            part[f"m_{name}"] = m.to_numpy()
            part[f"q_{name}"] = threshold.to_numpy()
            part[f"y_{name}"] = label.to_numpy()
            valid = label.notna()
            summary.append({"asset": ticker, "label_config": name,
                            "horizon": horizon, "quantile": quantile,
                            "threshold_last": float(threshold.dropna().iloc[-1]) if threshold.notna().any() else np.nan,
                            "n_rows": int(len(p)), "n_valid": int(valid.sum()),
                            "n_missing_label": int((~valid).sum()),
                            "n_missing_future_outcome": int(m.isna().sum()),
                            "n_warmup_threshold": int((m.notna() & threshold.isna()).sum()),
                            "n_positive": int(label.sum()),
                            "prevalence": float(label.mean()) if valid.any() else np.nan})
        blocks.append(part)
    # Outer concatenation preserves every asset's own history without a
    # cross-universe inner join.
    labels = pd.concat(blocks, ignore_index=True).sort_values(
        ["asset", "date"]).reset_index(drop=True)
    return labels, pd.DataFrame(summary)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw, meta = load_price_sources()
    prices, returns, cleaning, large = clean_prices(raw)
    states, states_meta = clean_states()
    registry = build_registry(prices, meta, states_meta)
    prices.to_parquet(OUT / "prices.parquet")
    returns.to_parquet(OUT / "returns.parquet")
    states.to_parquet(OUT / "states.parquet")
    registry.to_csv(OUT / "registry.csv", index=False)
    cleaning.to_csv(OUT / "cleaning_log.csv", index=False)
    large.to_csv(OUT / "large_moves_gt_40pct.csv", index=False)
    labels, label_summary = make_labels(prices)
    labels.to_parquet(OUT / "labels.parquet", index=False)
    label_summary.to_csv(OUT / "label_prevalence.csv", index=False)
    print(f"[data] price panel={prices.shape}; returns={returns.shape}; states={states.shape}; registry={len(registry)}")
    print(f"[clean] weekdays={len(prices)}; >40% non-crypto rows={len(large)}; ffill/clean log={OUT / 'cleaning_log.csv'}")
    print(f"[labels] rows={len(labels)}; assets={labels.asset.nunique()}; configs={label_summary.label_config.nunique()}; prevalence={OUT / 'label_prevalence.csv'}")
    pooled = label_summary.groupby("label_config", as_index=False).agg(
        n_valid=("n_valid", "sum"), n_positive=("n_positive", "sum"))
    pooled["prevalence"] = pooled.n_positive / pooled.n_valid.replace(0, np.nan)
    print("[label prevalence]\n" + pooled.to_string(index=False))
    print(f"[dropped rows] price gaps remain NaN in cleaning_log.csv; label warm-up and incomplete horizons are counted per asset/config in label_prevalence.csv")


if __name__ == "__main__":
    main()
