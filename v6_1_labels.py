"""V6 severity-rank label sweep using the frozen V3 price panel."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from pipeline_v6.labels import severity_ranks

ROOT = Path(__file__).resolve().parent
V3, OUT = ROOT / "data" / "v3", ROOT / "data" / "v6"
HORIZONS = (3, 5, 10, 15, 30)
PCTS = (2, 3, 4, 5, 6, 8, 10)


def hand_and_symmetry(prices):
    toy = np.array([100, 98, 101, 95, 110, 90, 105], float)
    d, u = severity_ranks(toy, 2, min_history=1)
    assert np.isfinite(d[-1]) and np.isfinite(u[-1])
    rng = np.random.default_rng(1701)
    for _ in range(5):
        x = np.exp(np.cumsum(rng.normal(0, .02, 1000)))
        a, b = severity_ranks(x, 5, 10)
        c, e = severity_ranks(1 / x, 5, 10)
        assert np.allclose(a, e, equal_nan=True, atol=1e-14)
        assert np.allclose(b, c, equal_nan=True, atol=1e-14)
    return {"toy_finite": int(np.isfinite(d).sum()), "symmetry_assets": 5}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(V3 / "prices.parquet")
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    assets = [c for c in prices.columns if c]
    label_dir = OUT / "labels"
    label_dir.mkdir(exist_ok=True)
    meta = pd.read_parquet(V3 / "panel.parquet",
                           columns=["date", "asset", "asset_class"]).drop_duplicates()
    overlap = []
    for j, asset in enumerate(assets, 1):
        target = label_dir / f"{asset.replace('/', '_')}.parquet"
        if target.exists():
            print(f"[V6-LABELS] reuse {j}/{len(assets)} {asset}", flush=True)
            continue
        p = prices[asset].to_numpy(float)
        observed = np.isfinite(p)
        result = {"date": prices.index, "asset": asset}
        for n in HORIZONS:
            # Match the V3 label contract: N counts observed asset sessions,
            # so exchange holidays and missing panel dates are skipped.
            d_obs, u_obs = severity_ranks(p[observed], n)
            d = np.full(len(p), np.nan); u = np.full(len(p), np.nan)
            d[observed] = d_obs; u[observed] = u_obs
            result[f"u_down_N{n}"] = d.astype("float32")
            result[f"u_up_N{n}"] = u.astype("float32")
            for pct in PCTS:
                q = pct / 100
                down_y = np.where(np.isfinite(d), d <= q, np.nan)
                up_y = np.where(np.isfinite(u), u <= q, np.nan)
                result[f"y_down_N{n}_p{pct}"] = down_y.astype("float32")
                result[f"y_up_N{n}_p{pct}"] = up_y.astype("float32")
                result[f"y_move_N{n}_p{pct}"] = np.where(
                    np.isfinite(d) & np.isfinite(u),
                    (d <= q) | (u <= q), np.nan).astype("float32")
        pd.DataFrame(result).to_parquet(target, index=False)
        if j % 20 == 0:
            print(f"[V6-LABELS] assets={j}/{len(assets)}", flush=True)
    # Long summary is compact and directly usable for prevalence tables.
    per_asset = {asset: pd.read_parquet(label_dir / f"{asset.replace('/', '_')}.parquet")
                 for asset in assets}
    records = []
    for n in HORIZONS:
        for pct in PCTS:
            for direction in ("down", "up", "move"):
                vals = [per_asset[asset][f"y_{direction}_N{n}_p{pct}"].to_numpy()
                        for asset in assets]
                vals = np.concatenate(vals)
                records.append({"N": n, "p": pct, "direction": direction,
                                "n": int(np.isfinite(vals).sum()),
                                "positive": int(np.nansum(vals)),
                                "prevalence": float(np.nanmean(vals))})
    summary = pd.DataFrame(records)
    summary.to_csv(OUT / "label_prevalence.csv", index=False)
    tests = hand_and_symmetry(prices.iloc[:, :5].dropna().iloc[:, 0].to_numpy())
    pd.DataFrame([tests]).to_csv(OUT / "label_unit_tests.csv", index=False)
    print(summary[summary.direction != "move"].to_string(index=False))
    print(f"[V6-LABELS] wrote assets={len(assets)} dates={len(prices)}")


if __name__ == "__main__":
    main()
