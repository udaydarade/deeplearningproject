"""Post-generation V3 feature metadata and all-empty H2 cleanup."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from v3_download_data import OUT


def main() -> None:
    registry = pd.read_csv(OUT / "registry.csv")
    assets = registry.loc[~registry.is_state.fillna(False)].set_index("ticker")
    panel = pd.read_parquet(OUT / "panel.parquet")
    panel["asset_class"] = panel.asset.map(assets.asset_class.to_dict()).fillna("unknown")
    panel["source_dataset"] = panel.asset.map(assets.source_dataset.to_dict()).fillna("v3_universe")
    panel.drop(columns=[c for c in panel if c.startswith("region_")], inplace=True)
    region = panel.asset.map(assets.region.to_dict()).fillna("unknown")
    region_names = sorted(region.unique())
    for name in region_names:
        panel[f"region_{name}"] = (region == name).astype("float32")
    panel.to_parquet(OUT / "panel.parquet", index=False, compression="zstd")

    h2 = pd.read_csv(OUT / "h2_window_shares.csv")
    empty = set(h2.loc[h2.share_windows_h2_bar.fillna(0).eq(0), "configuration"].astype(str))
    shared = pd.read_parquet(OUT / "market_features.parquet")
    remove = []
    for cfg in empty:
        cloud, k, _ = cfg.split("_")
        prefix = f"tda_{cloud}_k{k[1:]}_w60_h2_"
        remove.extend(c for c in shared if c.startswith(prefix))
    remove = sorted(set(remove))
    if remove:
        shared.drop(columns=remove, inplace=True)
        shared.to_parquet(OUT / "market_features.parquet", compression="zstd")
    path = OUT / "feature_sets.json"
    sets = json.loads(path.read_text(encoding="utf-8"))
    for key in sets:
        sets[key] = list(dict.fromkeys([c for c in sets[key] if c not in remove] +
                                       [f"region_{r}" for r in region_names]))
    path.write_text(json.dumps(sets, indent=2), encoding="utf-8")
    pd.DataFrame({"configuration": sorted(empty), "dropped_columns": [
        ",".join(c for c in remove if c.startswith(f"tda_{cfg.split('_')[0]}_k{cfg.split('_')[1][1:]}_w60_h2_"))
        for cfg in sorted(empty)]}).to_csv(OUT / "h2_all_empty_dropped.csv", index=False)
    print(f"[finalize] metadata rows={len(panel)}; regions={region_names}; dropped all-empty H2 columns={len(remove)}; shared={shared.shape}")


if __name__ == "__main__":
    main()
