"""V3 Phase 3: leakage-safe multi-asset features and panel construction.

All written artifacts stay under data/v3/. Expensive persistence features are
sampled at a logged stride and carried forward only (never backward-filled).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ripser import ripser
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from pipeline.features import landscape_vector
from v3_download_data import OUT

ROOT = Path(__file__).resolve().parent
FEATURE_OUT = OUT
CLOUD_SIZES = (6, 12, 24, 48)
WINDOWS = (60, 120)
EPSILONS = np.linspace(0.25, 2.0, 8)


def _finite_diagram_stats(diagram: np.ndarray) -> dict[str, float]:
    bars = diagram[np.isfinite(diagram).all(axis=1)] if diagram.size else np.empty((0, 2))
    persistence = bars[:, 1] - bars[:, 0] if len(bars) else np.empty(0)
    if len(persistence):
        total = float(persistence.sum())
        prob = persistence / total if total > 0 else np.zeros_like(persistence)
        entropy = float(-(prob[prob > 0] * np.log(prob[prob > 0])).sum())
        l1 = float((persistence ** 2).sum() / 4)
        l2 = float(np.sqrt((persistence ** 3).sum() / 12))
        maximum = float(persistence.max())
    else:
        total = entropy = l1 = l2 = maximum = 0.0
    out = {"totalpers": total, "maxpers": maximum, "entropy": entropy,
           "bar_count": int(len(persistence))}
    if len(persistence):
        out["landscape_l1"] = l1
        out["landscape_l2"] = l2
    else:
        out["landscape_l1"] = 0.0
        out["landscape_l2"] = 0.0
    return out


def _diagram_features(points: np.ndarray, prefix: str, maxdim: int,
                      n_perm: int | None = None,
                      distance: bool = False) -> dict[str, float]:
    result = ripser(points, maxdim=maxdim, distance_matrix=distance,
                    n_perm=n_perm, thresh=np.inf)["dgms"]
    out: dict[str, float] = {}
    for dim in range(maxdim + 1):
        bars = result[dim] if dim < len(result) else np.empty((0, 2))
        vals = _finite_diagram_stats(bars)
        for name, value in vals.items():
            out[f"{prefix}_h{dim}_{name}"] = value
        if dim in (1, 2):
            out[f"{prefix}_h{dim}_landscape_l1"] = vals["landscape_l1"]
            out[f"{prefix}_h{dim}_landscape_l2"] = vals["landscape_l2"]
        if dim == 1:
            finite = bars[np.isfinite(bars).all(axis=1)] if bars.size else np.empty((0, 2))
            for i, eps in enumerate(EPSILONS):
                out[f"{prefix}_h1_betti_eps{i}"] = float(
                    ((finite[:, 0] <= eps) & (finite[:, 1] > eps)).sum()) if len(finite) else 0.0
            vec = landscape_vector(finite)
            for i, value in enumerate(vec):
                out[f"{prefix}_h1_landscape_grid{i}"] = float(value)
    return out


def _normalise_rows(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[~np.isfinite(std) | (std < 1e-12)] = 1.0
    return (x - mean) / std


def _corr_distance(z: np.ndarray) -> np.ndarray:
    # z: assets x dates, each row standardised within the trailing window
    corr = np.corrcoef(z)
    corr = np.clip(np.nan_to_num(corr, nan=0.0), -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return np.sqrt(np.maximum(0.0, 2.0 * (1.0 - corr)))


def _one_cloud_features(block: np.ndarray, prefix: str, maxdim: int,
                        permute: bool) -> tuple[dict[str, float], bool]:
    """Feature extraction for one sampled bag and one cloud type/window."""
    # block is (window, asset) with complete returns.
    z = _normalise_rows(block)
    if prefix.startswith("T"):
        z = _normalise_rows(block)
        pca = PCA(n_components=min(5, z.shape[0] - 1, z.shape[1]), svd_solver="full")
        points = pca.fit_transform(z)
        if len(points) > 1:
            dist = pdist(points)
            med = np.median(dist[dist > 0]) if np.any(dist > 0) else 1.0
            points = points / max(float(med), 1e-12)
        result = _diagram_features(points, prefix, maxdim,
                                   n_perm=60 if permute and len(points) > 60 else None)
    else:
        # A cloud: assets are points; use correlation-derived distances.
        points = _corr_distance(z.T)
        result = _diagram_features(points, prefix, maxdim,
                                   n_perm=60 if permute and len(points) > 60 else None,
                                   distance=True)
    has_h2 = bool(result.get(f"{prefix}_h2_bar_count", 0) > 0)
    return result, has_h2


def _bag_indices(available: np.ndarray, k: int, bags: int,
                 seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed + int(k))
    if len(available) < k:
        return []
    return [np.sort(rng.choice(available, k, replace=False)) for _ in range(bags)]


def _tda_one_date(payload: tuple) -> tuple[pd.Timestamp, dict[str, float], dict[str, float]]:
    date, date_seed, hist, history_count, bags = payload
    nrow, ncol = hist.shape
    out: dict[str, float] = {}
    h2_share: dict[str, float] = {}
    # “full” is the longest-history subset, capped at 64 assets.
    nobs = history_count
    for window in WINDOWS:
        block_all = hist[-window:]
        available = np.flatnonzero(np.isfinite(block_all).all(axis=0))
        if len(available) < 8:
            continue
        full = available[np.argsort(-nobs[available], kind="stable")[:64]]
        for k in CLOUD_SIZES:
            if len(available) < k:
                continue
            idx_bags = _bag_indices(available, k, bags, date_seed)
            for cloud in ("T", "A"):
                collected: dict[str, list[float]] = {}
                for bag_idx, cols in enumerate(idx_bags):
                    maxdim = 2 if (window == 60 and k in (12, 24)) else 1
                    features, has_h2 = _one_cloud_features(
                        block_all[:, cols], f"tda_{cloud}_k{k}_w{window}", maxdim,
                        permute=(maxdim == 2))
                    for name, val in features.items():
                        collected.setdefault(name, []).append(float(val))
                    if maxdim == 2:
                        h2_share[f"{cloud}_k{k}_w{window}"] = (
                            h2_share.get(f"{cloud}_k{k}_w{window}", 0.0) + float(has_h2) / len(idx_bags))
                for name, values in collected.items():
                    out[name] = float(np.mean(values))
        # Full universe bag, no random subsampling.
        if len(full) >= 8:
            for cloud in ("T", "A"):
                maxdim = 2 if window == 60 and len(full) in (12, 24) else 1
                # H2 is only prescribed for nominal sizes 12 and 24.
                maxdim = 1
                prefix = f"tda_{cloud}_kfull_w{window}"
                features, _ = _one_cloud_features(block_all[:, full], prefix,
                                                   maxdim, permute=False)
                out.update(features)
    return date, out, h2_share


def _market_row(ret: pd.DataFrame, prices: pd.DataFrame, states: pd.DataFrame,
                registry: pd.DataFrame, i: int) -> dict[str, float]:
    """Cross-asset indicators for one date, using only observations <= date."""
    date = ret.index[i]
    meta = registry.set_index("ticker").asset_class.to_dict()
    classes = {k: np.array([meta.get(c, "unknown") == k for c in ret.columns])
               for k in ("equity_index", "us_etf", "country_etf", "bond", "commodity", "fx")}
    out: dict[str, float] = {}
    rnow = ret.iloc[i]
    for window in (60, 120):
        start = max(0, i - window + 1)
        block = ret.iloc[start:i + 1]
        valid = block.notna().all(axis=0).to_numpy()
        # Keep full universe for market features. Covariance operations are
        # restricted to non-stock histories, following the V3 cloud rule.
        nonstock = np.array([meta.get(c) != "stock" for c in ret.columns])
        cols = np.flatnonzero(valid & nonstock)
        if len(cols) < 8:
            continue
        x = block.iloc[:, cols].to_numpy(dtype=np.float64)
        x -= x.mean(axis=0, keepdims=True)
        sd = x.std(axis=0, ddof=1)
        keep = np.isfinite(sd) & (sd > 1e-10)
        x = x[:, keep]
        cols = cols[keep]
        if x.shape[1] < 8:
            continue
        z = x / x.std(axis=0, ddof=1, keepdims=True)
        corr = np.clip(z.T @ z / max(len(z) - 1, 1), -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        eig = np.linalg.eigvalsh(corr)
        off = corr[np.triu_indices_from(corr, 1)]
        out[f"market_mean_corr_w{window}"] = float(off.mean())
        out[f"market_lambda1_share_w{window}"] = float(eig[-1] / len(eig))
        out[f"market_top3_eig_share_w{window}"] = float(eig[-3:].sum() / eig.sum())
        dist = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - corr)))
        out[f"cluster_mst_length_w{window}"] = float(minimum_spanning_tree(dist).sum())
        condensed = squareform(dist, checks=False)
        tree = linkage(condensed, method="average")
        for cutoff in (.8, 1.0):
            ids = fcluster(tree, t=cutoff, criterion="distance")
            sizes = np.bincount(ids)[1:]
            out[f"cluster_count_d{cutoff}_w{window}"] = float(len(sizes))
            out[f"cluster_largest_share_d{cutoff}_w{window}"] = float(sizes.max() / len(ids))
            probs = sizes / sizes.sum()
            out[f"cluster_entropy_d{cutoff}_w{window}"] = float(-(probs * np.log(probs)).sum())

    # Short-horizon breadth, dispersion and drawdown share.
    for period in (5, 20, 60):
        start = max(0, i - period + 1)
        rs = ret.iloc[start:i + 1].sum(axis=0, min_count=period)
        valid = rs.notna() & pd.Series(
            [meta.get(c) != "stock" for c in ret.columns], index=ret.columns)
        vals = rs[valid]
        if len(vals) >= 8:
            out[f"market_breadth_negative_{period}d"] = float((vals < 0).mean())
            out[f"market_dispersion_{period}d"] = float(vals.std(ddof=0))
    if i >= 59:
        hist_p = prices.iloc[i - 59:i + 1]
        dd = hist_p.iloc[-1] / hist_p.max() - 1
        nonstock_cols = [c for c in prices.columns if meta.get(c) != "stock"]
        dd = dd[nonstock_cols].dropna()
        if len(dd) >= 8:
            out["market_share_drawdown_gt10pct"] = float((dd < -.10).mean())
    if i >= 199:
        av = prices.iloc[i - 199:i + 1].mean()
        below = prices.iloc[i] / av - 1
        nonstock_cols = [c for c in prices.columns if meta.get(c) != "stock"]
        below = below[nonstock_cols].dropna()
        if len(below) >= 8:
            out["market_share_below_sma200"] = float((below < 0).mean())
    if i >= 250:
        block = ret.iloc[i - 249:i + 1]
        cols = block.columns[block.notna().all(axis=0)]
        cols = [c for c in cols if meta.get(c) != "stock"]
        if len(cols) >= 8:
            x = block[cols].to_numpy(float)
            model = LedoitWolf().fit(x[:-1])
            delta = x[-1] - model.location_
            out["market_turbulence"] = float(delta @ np.linalg.solve(model.covariance_, delta))
    # Cross-asset class links use class-average returns (lagged only).
    class_returns = {}
    for label, mask in classes.items():
        idxs = np.flatnonzero(mask)
        if idxs.size:
            class_returns[label] = ret.iloc[:, idxs].mean(axis=1)
    pairs = [("equity_index", "bond", "equity_bond"),
             ("equity_index", "commodity", "equity_commodity"),
             ("equity_index", "fx", "equity_fx")]
    for a, b, name in pairs:
        if a in class_returns and b in class_returns:
            window = ret.iloc[max(0, i - 59):i + 1].index
            out[f"market_corr_{name}_60d"] = float(
                class_returns[a].loc[window].corr(class_returns[b].loc[window]))
    if "HYG" in ret and "LQD" in ret and i >= 20:
        out["market_hyg_lqd_rel20"] = float(ret["HYG"].iloc[i - 19:i + 1].sum()
                                               - ret["LQD"].iloc[i - 19:i + 1].sum())
    if date in states.index:
        for col in states.columns:
            out[f"state_{col}"] = float(states.at[date, col]) if pd.notna(states.at[date, col]) else np.nan
        for col in ("VIX", "VIXCLS"):
            if col in states:
                out[f"state_{col}_change5"] = float(states[col].diff(5).at[date])
                realized = ret.iloc[max(0, i - 19):i + 1].std(axis=1).iloc[-1]
                out[f"state_{col}_over_realized"] = float(states.at[date, col] / (100 * realized)) if realized and pd.notna(states.at[date, col]) else np.nan
    if "TNX" in states and "IRX" in states and date in states.index:
        slope = states["TNX"] - states["IRX"]
        out["state_tnx_irx_slope"] = float(slope.at[date])
        out["state_tnx_irx_change20"] = float(slope.diff(20).at[date])
    return out


def _asset_features(prices: pd.Series, returns: pd.Series,
                    market_ret: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=returns.index)
    for window in (50, 100):
        roll = returns.rolling(window, min_periods=window)
        out[f"csd_ac1_w{window}"] = returns.rolling(window - 1, min_periods=window - 1).corr(returns.shift(1))
        out[f"csd_variance_w{window}"] = roll.var()
        out[f"csd_skew_w{window}"] = roll.skew()
        out[f"csd_kurtosis_w{window}"] = roll.kurt()
    out["realized_vol_w20"] = returns.rolling(20, min_periods=20).std()
    out["ewma_vol_w20"] = returns.ewm(span=20, min_periods=20).std()
    # GARCH would require about 42k fits from the measured 20-fit timing; the
    # >25-minute estimate is logged and the entire GARCH block is omitted.
    for lag in (1, 5, 20, 60):
        out[f"price_momentum_{lag}d"] = returns.rolling(lag, min_periods=lag).sum()
    out["price_volatility_20d"] = returns.rolling(20, min_periods=20).std()
    out["price_volatility_60d"] = returns.rolling(60, min_periods=60).std()
    out["price_downside_vol_20d"] = np.sqrt(returns.clip(upper=0).pow(2).rolling(20, min_periods=20).mean())
    out["price_drawdown_60d"] = prices / prices.rolling(60, min_periods=60).max() - 1
    out["price_drawdown_252d"] = prices / prices.rolling(252, min_periods=252).max() - 1
    out["price_vol_ratio_20_60"] = out.price_volatility_20d / out.price_volatility_60d
    out["price_skew_20d"] = returns.rolling(20, min_periods=20).skew()
    beta = returns.rolling(60, min_periods=60).cov(market_ret) / market_ret.rolling(60, min_periods=60).var()
    out["relative_beta_60d"] = beta
    out["relative_corr_60d"] = returns.rolling(60, min_periods=60).corr(market_ret)
    for lag in (20, 60):
        out[f"relative_strength_{lag}d"] = returns.rolling(lag, min_periods=lag).sum() - market_ret.rolling(lag, min_periods=lag).sum()
    resid = returns - out.relative_beta_60d * market_ret
    out["relative_idiosyncratic_vol_60d"] = resid.rolling(60, min_periods=60).std()
    return out


def _build_tda(returns: pd.DataFrame, stride: int, bags: int,
               dates: pd.DatetimeIndex | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    nonstock = pd.read_csv(OUT / "registry.csv").set_index("ticker").asset_class
    cols = np.array([c for c in returns.columns if nonstock.get(c) != "stock"])
    history_counts = returns[cols].notna().cumsum().to_numpy(dtype=np.int32)
    selected_dates = dates if dates is not None else returns.index[stride - 1::stride]
    rows, h2rows = [], []
    t0 = time.perf_counter()
    for n, date in enumerate(selected_dates):
        i = returns.index.get_loc(date)
        start = max(0, i - max(WINDOWS) + 1)
        hist = returns.iloc[start:i + 1][cols].to_numpy(dtype=np.float32)
        if hist.shape[0] < max(WINDOWS):
            continue
        seed = int(date.strftime("%Y%m%d"))
        _, features, h2 = _tda_one_date((date, seed, hist, history_counts[i], bags))
        rows.append({"date": date, **features})
        h2rows.append({"date": date, **h2})
        if (n + 1) % 20 == 0:
            print(f"[tda] {n + 1}/{len(selected_dates)} dates; elapsed={time.perf_counter()-t0:.1f}s", flush=True)
    return pd.DataFrame(rows).set_index("date"), pd.DataFrame(h2rows).set_index("date")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true", help="time 20 representative dates only")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--bags", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    returns = pd.read_parquet(OUT / "returns.parquet").astype("float32")
    prices = pd.read_parquet(OUT / "prices.parquet").astype("float32")
    states = pd.read_parquet(OUT / "states.parquet")
    registry = pd.read_csv(OUT / "registry.csv")
    labels = pd.read_parquet(OUT / "labels.parquet")
    stock_tickers = set(registry.loc[registry.asset_class.eq("stock"), "ticker"])
    nonstock = [c for c in returns if c not in stock_tickers]
    market_returns = returns[nonstock].mean(axis=1)
    t0 = time.perf_counter()
    # Expensive cross-sectional correlation/clustering is computed once per
    # date and shared across all targets.
    date_iter = returns.index
    if args.pilot:
        date_iter = returns.index[np.linspace(200, len(returns) - 1, 20, dtype=int)]
    market_rows = []
    market_t0 = time.perf_counter()
    for j, date in enumerate(date_iter):
        i = returns.index.get_loc(date)
        market_rows.append({"date": date, **_market_row(returns, prices, states, registry, i)})
        if (j + 1) % 1000 == 0:
            print(f"[market] {j+1}/{len(date_iter)}", flush=True)
    market = pd.DataFrame(market_rows).set_index("date")
    market = market.reindex(returns.index).ffill()
    if args.pilot:
        # Only a representative TDA subset is timed; the production run will
        # use a compact B/stride selected from this projected duration.
        tdates = returns.index[np.linspace(max(120, 500), len(returns) - 1, 20, dtype=int)]
    else:
        tdates = returns.index[args.stride - 1::args.stride]
    tda_t0 = time.perf_counter()
    tda, h2 = _build_tda(returns, args.stride, args.bags, tdates)
    elapsed_tda = time.perf_counter() - tda_t0
    h2_summary = []
    for col in h2:
        values = h2[col].dropna()
        h2_summary.append({"configuration": col,
                           "share_windows_h2_bar": float(values.mean()) if len(values) else np.nan,
                           "n_windows": int(len(values))})
    h2report = pd.DataFrame(h2_summary)
    h2report.to_csv(OUT / "h2_window_shares.csv", index=False)
    h2_empty_configs = set(h2report.loc[
        h2report.share_windows_h2_bar.fillna(0).eq(0), "configuration"].astype(str))
    empty_h2_cols = [c for c in tda.columns
                     if "_h2_" in c and any(
                         c.startswith(f"tda_{cfg.split('_k')[0]}_k{cfg.split('_k')[1].split('_w')[0]}_w60_h2_")
                         for cfg in h2_empty_configs)]
    if empty_h2_cols:
        tda = tda.drop(columns=empty_h2_cols)
    pd.DataFrame({"configuration": sorted(h2_empty_configs),
                  "dropped_columns": [
                      ",".join(c for c in empty_h2_cols if c.startswith(
                          f"tda_{cfg.split('_k')[0]}_k{cfg.split('_k')[1].split('_w')[0]}_w60_h2_"))
                      for cfg in sorted(h2_empty_configs)]}).to_csv(
                          OUT / "h2_all_empty_dropped.csv", index=False)
    if args.pilot:
        elapsed_market = time.perf_counter() - market_t0
        n_tda_dates = len(returns.index[args.stride - 1::args.stride])
        print(f"[pilot] market_20_dates={elapsed_market:.2f}s; projected_daily_market={elapsed_market * len(returns) / 20 / 60:.1f}min")
        print(f"[pilot] tda_20_dates={elapsed_tda:.2f}s; projected_stride_{args.stride}={elapsed_tda * n_tda_dates / 20 / 60:.1f}min; bags={args.bags}")
        print("[pilot] not building/writing the full panel")
        return
    # Past-only topology dynamics. No backward fill or centered window.
    raw_signal_names = [c for c in tda if any(s in c for s in (
        "_h1_landscape_l2", "_h1_totalpers", "_h0_totalpers", "_h2_totalpers"))]
    for col in raw_signal_names:
        series = tda[col]
        for win in (5,):  # five stride-10 observations ~= 50 sessions
            tda[f"{col}_slope{win}"] = series.diff(win) / win
            tda[f"{col}_std{win}"] = series.rolling(win, min_periods=win).std()
            tda[f"{col}_ac1{win}"] = series.rolling(win, min_periods=win).corr(series.shift(1))
            mu = series.rolling(25, min_periods=10).mean()
            sd = series.rolling(25, min_periods=10).std()
            tda[f"{col}_z250"] = (series - mu) / sd.replace(0, np.nan)
    for cloud_size_window in sorted({c.split("_h1_landscape_grid")[0]
                                     for c in tda if "_h1_landscape_grid" in c}):
        grid_cols = [f"{cloud_size_window}_h1_landscape_grid{i}" for i in range(36)]
        if all(c in tda for c in grid_cols):
            vec = tda[grid_cols].to_numpy(dtype=np.float32)
            velocity = np.sqrt(np.square(vec - np.roll(vec, 1, axis=0)).sum(axis=1))
            velocity[0] = np.nan
            tda[f"{cloud_size_window}_h1_landscape_velocity10"] = velocity
    if "VIX" in states and "VIX3M" in states:
        tda["state_VIX_over_VIX3M"] = states.VIX / states.VIX3M.replace(0, np.nan)
    tda = tda.reindex(returns.index).ffill()
    # No per-asset GARCH: pilot measured ~0.14s/refit, projecting beyond 25m.
    asset_frames = []
    asset_group = registry.loc[~registry.is_state.fillna(False)].set_index("ticker")
    for ticker in returns.columns:
        p = prices[ticker].dropna()
        r = returns[ticker].reindex(p.index)
        f = _asset_features(p, r, market_returns.reindex(p.index)).reset_index().rename(columns={"index": "date"})
        f["asset"] = ticker
        cls = asset_group.at[ticker, "asset_class"] if ticker in asset_group.index else "unknown"
        region = asset_group.at[ticker, "region"] if ticker in asset_group.index else "unknown"
        f["source_dataset"] = asset_group.at[ticker, "source_dataset"] if ticker in asset_group.index else "v3_universe"
        for name in ["equity_index", "us_etf", "country_etf", "bond", "commodity", "fx", "crypto", "stock"]:
            f[f"class_{name}"] = np.float32(cls == name)
        regions = sorted(set(asset_group.region.dropna().astype(str)))
        for name in regions:
            f[f"region_{name}"] = np.float32(region == name)
        asset_frames.append(f)
    panel = pd.concat(asset_frames, ignore_index=True)
    # Relative peer ranks/means by date and class, excluding stocks from the
    # common market return. Target's own series is excluded from class means.
    # For same-class peer aggregates, subtract this asset from the class sum.
    panel["asset_class"] = panel.asset.map(asset_group.asset_class.to_dict())
    panel["relative_rank_mom20"] = panel.groupby("date").price_momentum_20d.rank(pct=True)
    panel["relative_rank_vol20"] = panel.groupby("date").price_volatility_20d.rank(pct=True)
    panel["_class_ret_sum"] = panel.groupby(["date", "asset_class"]).price_momentum_20d.transform("sum")
    panel["_class_vol_sum"] = panel.groupby(["date", "asset_class"]).price_volatility_20d.transform("sum")
    panel["_class_count"] = panel.groupby(["date", "asset_class"]).price_momentum_20d.transform("count")
    panel["peer_mean_momentum20"] = (panel._class_ret_sum - panel.price_momentum_20d) / (panel._class_count - 1).replace(0, np.nan)
    panel["peer_mean_vol20"] = (panel._class_vol_sum - panel.price_volatility_20d) / (panel._class_count - 1).replace(0, np.nan)
    panel.drop(columns=["_class_ret_sum", "_class_vol_sum", "_class_count", "asset_class"], inplace=True)
    # Store shared market/TDA signals once by date. The trainer joins them by
    # date, which avoids a multi-gigabyte repeated copy for every asset row.
    shared = market.join(tda, how="outer").reindex(returns.index).ffill().astype("float32")
    shared.to_parquet(OUT / "market_features.parquet", compression="zstd")
    panel = panel.merge(labels, on=["asset", "date"], how="left", validate="one_to_one")
    # NaN audit outside early history; shared features are audited once per
    # date, asset features once per observed target row.
    label_cols = [c for c in panel if c.startswith(("m_", "q_", "y_"))]
    local_cols = [c for c in panel.select_dtypes(include=[np.number]).columns if c not in label_cols]
    shared_cols = shared.columns.tolist()
    after = pd.Timestamp("2005-01-01")
    local_audit = panel.loc[panel.date >= after, local_cols].isna().mean()
    shared_audit = shared.loc[shared.index >= after].isna().mean()
    audit = pd.concat([local_audit, shared_audit])
    dropped = set(audit[audit > .30].index)
    pd.DataFrame({"feature": audit.index, "nan_share_after_2005": audit.values,
                  "dropped_gt30pct": audit.index.isin(dropped)}).to_csv(OUT / "feature_quality.csv", index=False)
    # Correlation pruning is fitted only on pre-2000 rows. Blocks remain
    # separate to avoid expanding shared date features to every asset.
    def duplicate_drop(frame: pd.DataFrame, columns: list[str], dates: pd.Series | pd.Index) -> set[str]:
        sample = frame.loc[dates <= pd.Timestamp("1999-12-31"), columns].select_dtypes(include=[np.number])
        if sample.shape[1] < 2 or sample.shape[0] < 100:
            return set()
        corr = sample.corr(min_periods=100).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        return {c for c in upper.columns if (upper[c] > .98).any()}
    local_dupes = duplicate_drop(panel, local_cols, panel.date)
    shared_dupes = duplicate_drop(shared, shared_cols, shared.index)
    dropped_corr = local_dupes | shared_dupes
    all_cols = local_cols + shared_cols
    pd.DataFrame({"feature": all_cols,
                  "dropped_nan_gt30pct": [c in dropped for c in all_cols],
                  "dropped_corr_gt98_pre2000": [c in dropped_corr for c in all_cols]}).to_csv(OUT / "feature_pruning.csv", index=False)
    keep = [c for c in all_cols if c not in dropped and c not in dropped_corr]
    categorical = [c for c in keep if c.startswith(("class_", "region_"))]
    feature_sets = {
        "F0": [c for c in keep if c.startswith(("csd_", "realized_vol", "ewma_vol"))] + categorical,
        "F1": [c for c in keep if c.startswith(("csd_", "realized_vol", "ewma_vol", "price_", "relative_", "peer_"))] + categorical,
        "F2": [c for c in keep if c.startswith(("csd_", "realized_vol", "ewma_vol", "price_", "relative_", "peer_", "market_", "cluster_", "state_"))] + categorical,
        "F3": [c for c in keep if c.startswith("tda_")] + categorical,
    }
    feature_sets["F4"] = sorted(set(feature_sets["F2"] + [c for c in keep if c.startswith("tda_")]))
    # Remove high-NaN features from training sets, retaining complete row
    # eligibility logic in the experiment script; GBMs may keep state NaNs.
    for key in feature_sets:
        feature_sets[key] = [c for c in feature_sets[key] if c not in dropped]
    (OUT / "feature_sets.json").write_text(json.dumps(feature_sets, indent=2), encoding="utf-8")
    for col in panel.select_dtypes(include=[np.number]).columns:
        panel[col] = panel[col].astype("float32")
    panel.to_parquet(OUT / "panel.parquet", index=False, compression="zstd")
    print(f"[features] panel={panel.shape}; shared={shared.shape}; kept={len(keep)}; dropped_nan={len(dropped)}; tda dates={len(tda)} elapsed={time.perf_counter()-t0:.1f}s")
    print(f"[quality] F0={len(feature_sets['F0'])}, F1={len(feature_sets['F1'])}, F2={len(feature_sets['F2'])}, F3={len(feature_sets['F3'])}, F4={len(feature_sets['F4'])}; GARCH skipped (>25m projected)")


if __name__ == "__main__":
    main()
