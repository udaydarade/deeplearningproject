"""V4 PRICE1D H0 persistence with an O(n log n) union-find algorithm."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V3 = ROOT / "data" / "v3"
V4 = ROOT / "data" / "v4"


def finite_h0(values: np.ndarray) -> list[tuple[float, float]]:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    parent = np.arange(len(values))
    birth = values.copy()
    active = np.zeros(len(values), dtype=bool)
    bars = []

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(left, right, level):
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if birth[root_left] <= birth[root_right]:
            parent[root_right] = root_left
            bars.append((float(birth[root_right]), float(level)))
        else:
            parent[root_left] = root_right
            bars.append((float(birth[root_left]), float(level)))

    for index in order:
        active[index] = True
        if index > 0 and active[index - 1]:
            union(index, index - 1, values[index])
        if index + 1 < len(values) and active[index + 1]:
            union(index, index + 1, values[index])
    # The globally oldest component is the essential H0 bar and is omitted.
    return [(birth_value, death) for birth_value, death in bars if death > birth_value]


def brute_h0(values: np.ndarray) -> list[tuple[float, float]]:
    # Reference implementation by thresholded connected components.
    values = np.asarray(values, dtype=float)
    levels = sorted(set(values.tolist()))
    active = np.zeros(len(values), dtype=bool)
    roots = {}
    bars = []
    for level in levels:
        active |= values == level
        components = []
        seen = set()
        for index in np.flatnonzero(active):
            if index in seen:
                continue
            stack, component = [int(index)], []
            while stack:
                current = stack.pop()
                if current < 0 or current >= len(values) or current in seen or not active[current]:
                    continue
                seen.add(current)
                component.append(current)
                stack.extend([current - 1, current + 1])
            components.append(component)
        new_roots = {}
        for component in components:
            root = min(component, key=lambda i: values[i])
            component_birth = float(values[root])
            old_births = [roots.get(i, float(values[i])) for i in component if i in roots]
            births = old_births + [component_birth]
            survivor = min(births)
            for birth in births:
                if birth > survivor:
                    bars.append((birth, float(level)))
            new_roots[root] = survivor
        roots = new_roots
    return [(birth, death) for birth, death in bars if death > birth]


def summary(values: np.ndarray, bars: list[tuple[float, float]], scale: float, prefix: str) -> dict:
    lengths = np.asarray([death - birth for birth, death in bars], dtype=float)
    p = lengths / lengths.sum() if len(lengths) and lengths.sum() else lengths
    return {
        f"{prefix}_total": float(lengths.sum()) if len(lengths) else 0.0,
        f"{prefix}_max": float(lengths.max()) if len(lengths) else 0.0,
        f"{prefix}_entropy": float(-(p[p > 0] * np.log(p[p > 0])).sum()) if len(p) else 0.0,
        f"{prefix}_count": int(len(lengths)),
        f"{prefix}_long_count": int((lengths > scale).sum()) if len(lengths) else 0,
        f"{prefix}_max_scale_free": float(lengths.max() / scale) if len(lengths) and scale > 0 else np.nan,
    }


def run_tests() -> None:
    assert finite_h0(np.array([0, 2, 1, 3])) == [(1.0, 2.0)]
    assert finite_h0(np.array([1, 0, 2, 0.5, 3])) == [(0.5, 2.0)]
    rng = np.random.default_rng(17)
    for _ in range(200):
        values = rng.normal(size=12)
        assert finite_h0(values) == brute_h0(values)
    print("[PRICE1D tests] 2 hand cases + 200 brute-force random windows passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", action="store_true")
    parser.add_argument("--limit-assets", type=int, default=0)
    args = parser.parse_args()
    run_tests()
    if args.tests:
        return
    prices = pd.read_parquet(V3 / "prices.parquet").astype(float)
    returns = pd.read_parquet(V3 / "returns.parquet").astype(float)
    assets = list(prices.columns)[:args.limit_assets or len(prices.columns)]
    rows = []
    for asset in assets:
        log_price = np.log(prices[asset])
        for i in range(119, len(log_price)):
            if not np.isfinite(log_price.iloc[i - 119:i + 1]).all():
                continue
            row = {"date": log_price.index[i], "asset": asset}
            for window in (20, 60, 120):
                values = log_price.iloc[i - window + 1:i + 1].to_numpy()
                scale = float(returns[asset].iloc[i - window + 1:i + 1].std() * np.sqrt(window))
                row.update(summary(values, finite_h0(values), scale, f"price1d_valley_w{window}"))
                peaks = -values
                row.update(summary(peaks, finite_h0(peaks), scale, f"price1d_peak_w{window}"))
            rows.append(row)
    pd.DataFrame(rows).to_parquet(V4 / "blocks" / "V4_PRICE1D.parquet", index=False)
    print(f"[complete] V4_PRICE1D rows={len(rows):,} assets={len(assets)}")


if __name__ == "__main__":
    main()
