"""Build the V4 P1 raw/correlation TDA blocks from the frozen V3 returns.

This script never rebuilds prices, labels, folds, or the V3 universe. It uses
the V3 non-stock complete-window universe and writes only under data/v4/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ripser import ripser
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
V3 = ROOT / "data" / "v3"
OUT = ROOT / "data" / "v4"


def stats(diagram: np.ndarray, prefix: str, h: int) -> dict[str, float]:
    bars = diagram[np.isfinite(diagram).all(axis=1)] if diagram.size else np.empty((0, 2))
    persistence = bars[:, 1] - bars[:, 0] if len(bars) else np.empty(0)
    if len(persistence):
        total = float(persistence.sum())
        maximum = float(persistence.max())
        p = persistence / total if total else persistence
        entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        out = {f"{prefix}_h{h}_total": total, f"{prefix}_h{h}_max": maximum,
               f"{prefix}_h{h}_entropy": entropy, f"{prefix}_h{h}_count": int(len(persistence))}
        if h == 1:
            out[f"{prefix}_h1_l1"] = float((persistence ** 2).sum() / 4)
            out[f"{prefix}_h1_l2"] = float(np.sqrt((persistence ** 3).sum() / 12))
        return out
    out = {f"{prefix}_h{h}_total": 0.0, f"{prefix}_h{h}_max": 0.0,
           f"{prefix}_h{h}_entropy": 0.0, f"{prefix}_h{h}_count": 0}
    if h == 1:
        out[f"{prefix}_h1_l1"] = 0.0
        out[f"{prefix}_h1_l2"] = 0.0
    return out


def cloud(block: np.ndarray, kind: str) -> np.ndarray:
    # block is time x assets.
    if kind == "A_corr":
        corr = np.corrcoef(block.T)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)
        return np.sqrt(np.maximum(0.0, 2.0 * (1.0 - np.clip(corr, -1, 1))))
    if kind == "A_raw":
        return squareform(pdist(block.T, metric="euclidean"))
    points = block
    if points.shape[1] > 24:
        points = PCA(n_components=min(8, points.shape[0], points.shape[1]),
                     svd_solver="full").fit_transform(points)
    return points


def one(block: np.ndarray, kind: str, prefix: str) -> dict[str, float]:
    distance = kind.startswith("A_")
    points = cloud(block, kind)
    scale = 1.0
    if kind != "A_corr":
        scale = float(np.sqrt(np.mean(points ** 2)))
        scale = scale if np.isfinite(scale) and scale > 0 else 1.0
        points = points / scale
    diagrams = ripser(points, distance_matrix=distance, maxdim=1, thresh=np.inf)["dgms"]
    out = {}
    out.update(stats(diagrams[0] * scale, prefix, 0))
    out.update(stats(diagrams[1] * scale, prefix, 1))
    if kind == "A_raw":
        variance = float(np.mean(np.var(block, axis=0, ddof=0)))
        out[f"{prefix}_h1_l1_ratio"] = out[f"{prefix}_h1_l1"] / variance if variance > 0 else np.nan
        out[f"{prefix}_h1_l2_ratio"] = out[f"{prefix}_h1_l2"] / variance ** 1.5 if variance > 0 else np.nan
        out[f"{prefix}_h1_total_ratio"] = out[f"{prefix}_h1_total"] / np.sqrt(variance) if variance > 0 else np.nan
    return out


def date_features(date, returns: pd.DataFrame, nonstock: list[str], w: int, k: int,
                   bags: int, kinds=("A_corr", "A_raw", "T_raw")) -> dict:
    loc = returns.index.get_loc(date)
    block = returns.iloc[loc - w + 1:loc + 1][nonstock]
    available = np.flatnonzero(block.notna().all(axis=0).to_numpy())
    out = {"date": date, f"mean_var_w{w}": float(np.nanmean(block.var(axis=0, ddof=0)))}
    if len(available) < 8:
        return out
    seed = int(pd.Timestamp(date).strftime("%Y%m%d"))
    for kind in kinds:
        for nominal in (k, "full"):
            if nominal == "full":
                chosen = available
                if len(chosen) == 0:
                    continue
                chosen = chosen[np.argsort(-block.iloc[:, chosen].notna().sum(axis=0).to_numpy(), kind="stable")]
                chosen = chosen[:64]
                bag_indices = [chosen]
            else:
                if len(available) < nominal:
                    continue
                rng = np.random.default_rng(seed + int(nominal))
                bag_indices = [np.sort(rng.choice(available, nominal, replace=False)) for _ in range(bags)]
            bag_rows = []
            for indices in bag_indices:
                sub = block.iloc[:, indices].to_numpy(float)
                if not np.isfinite(sub).all():
                    continue
                bag_rows.append(one(sub, kind, f"v4_{kind}_k{nominal}_w{w}"))
            if bag_rows:
                keys = sorted(set().union(*(row.keys() for row in bag_rows)))
                out.update({key: float(np.mean([row.get(key, np.nan) for row in bag_rows])) for key in keys})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--bags", type=int, default=2)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--limit-dates", type=int, default=0)
    parser.add_argument("--start-date", default="1997-01-01")
    args = parser.parse_args()
    OUT.joinpath("blocks").mkdir(parents=True, exist_ok=True)
    returns = pd.read_parquet(V3 / "returns.parquet").astype("float64")
    registry = pd.read_csv(V3 / "registry.csv")
    classes = registry.set_index("ticker").asset_class.to_dict()
    nonstock = [column for column in returns.columns if classes.get(column) != "stock"]
    dates = returns.index[(returns.index >= pd.Timestamp(args.start_date)) & (returns.index >= returns.index[119])][::args.stride]
    if args.limit_dates:
        dates = dates[:args.limit_dates]
    pilot_dates = dates[:20]
    start = time.perf_counter()
    for date in pilot_dates:
        date_features(date, returns, nonstock, 60, 24, args.bags, ("A_raw", "T_raw"))
    elapsed = time.perf_counter() - start
    projected = elapsed * len(dates) / max(len(pilot_dates), 1) / 60
    print(f"[pilot] dates={len(pilot_dates)} elapsed={elapsed:.1f}s projected={projected:.1f}min stride={args.stride} bags={args.bags}")
    if args.pilot:
        return
    rows = []
    started = time.perf_counter()
    for number, date in enumerate(dates, start=1):
        row = {"date": date}
        for window, kinds in ((20, ("A_raw",)), (60, ("A_corr", "A_raw", "T_raw")),
                              (120, ("A_corr", "A_raw", "T_raw"))):
            row.update(date_features(date, returns, nonstock, window, 24, args.bags, kinds))
        rows.append(row)
        if number % 100 == 0:
            print(f"[build] {number}/{len(dates)} elapsed={(time.perf_counter()-started)/60:.1f}min", flush=True)
    frame = pd.DataFrame(rows).sort_values("date").drop_duplicates("date", keep="last")
    frame.to_parquet(OUT / "blocks" / "V4_RAW.parquet", index=False)
    (OUT / "feature_blocks.json").write_text(json.dumps({
        "V4_RAW": [column for column in frame.columns if column != "date"],
        "config": {"stride": args.stride, "bags": args.bags, "nonstock_count": len(nonstock)},
    }, indent=2), encoding="utf-8")
    print(f"[complete] V4_RAW rows={len(frame):,} columns={len(frame.columns)-1}")


if __name__ == "__main__":
    main()
