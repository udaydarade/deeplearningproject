"""V6 past-only SEMIVAR block and extended panel."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V3, OUT = ROOT / "data" / "v3", ROOT / "data" / "v6"


def one_asset(r: pd.Series) -> pd.DataFrame:
    x = r.astype("float64")
    out = pd.DataFrame(index=x.index)
    for w in (20, 60):
        out[f"semivar_up_{w}"] = x.clip(lower=0).pow(2).rolling(w, min_periods=w).mean().pow(.5)
        out[f"semivar_down_{w}"] = x.clip(upper=0).pow(2).rolling(w, min_periods=w).mean().pow(.5)
        out[f"semivar_logratio_{w}"] = np.log((out[f"semivar_up_{w}"] + 1e-8) /
                                               (out[f"semivar_down_{w}"] + 1e-8))
        out[f"semivar_skew_{w}"] = x.rolling(w, min_periods=w).skew()
    out["largest_gain_20"] = x.rolling(20, min_periods=20).max()
    out["largest_loss_20"] = x.rolling(20, min_periods=20).min()
    for w in (60, 252):
        lo, hi = x.rolling(w, min_periods=w).min(), x.rolling(w, min_periods=w).max()
        out[f"close_position_{w}"] = (x - lo) / (hi - lo).replace(0, np.nan)
    high60, low60 = x.rolling(60, min_periods=60).max(), x.rolling(60, min_periods=60).min()
    out["days_since_high_60"] = x.groupby((x == high60).cumsum()).cumcount()
    out["days_since_low_60"] = x.groupby((x == low60).cumsum()).cumcount()
    rolling_std = x.rolling(20, min_periods=20).std()
    out["vol_compression_20"] = rolling_std.rolling(250, min_periods=60).rank(pct=True)
    pos = x.gt(0)
    neg = x.lt(0)
    out["up_streak"] = pos.groupby((pos != pos.shift()).cumsum()).cumcount() + 1
    out.loc[~pos, "up_streak"] = 0
    out["down_streak"] = neg.groupby((neg != neg.shift()).cumsum()).cumcount() + 1
    out.loc[~neg, "down_streak"] = 0
    return out.astype("float32")


def main():
    OUT.mkdir(exist_ok=True)
    returns = pd.read_parquet(V3 / "returns.parquet")
    blocks = []
    for j, asset in enumerate(returns.columns, 1):
        f = one_asset(returns[asset])
        f["date"] = f.index
        f["asset"] = asset
        blocks.append(f.reset_index(drop=True))
        if j % 25 == 0:
            print(f"[V6-SEMIVAR] assets={j}/{len(returns.columns)}", flush=True)
    features = pd.concat(blocks, ignore_index=True)
    features.to_parquet(OUT / "semivar.parquet", index=False)
    panel = pd.read_parquet(V3 / "panel.parquet")
    panel = panel.merge(features, on=["date", "asset"], how="left")
    panel.to_parquet(OUT / "panel.parquet", index=False)
    registry = panel[["asset", "asset_class"]].drop_duplicates()
    registry.to_csv(OUT / "registry_v6.csv", index=False)
    print(f"[V6-SEMIVAR] rows={len(features):,} columns={len(features.columns)}")


if __name__ == "__main__":
    main()
