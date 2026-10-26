"""V6 Phase 0 inventory; reads prior outputs without modifying them."""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
V3 = ROOT / "data" / "v3"
V4 = ROOT / "data" / "v4"
V5 = ROOT / "data" / "v5"
OUT = ROOT / "data" / "v6"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(V3 / "panel.parquet", columns=["date", "asset", "asset_class", "y_BASE"])
    panel["date"] = pd.to_datetime(panel["date"])
    registry = panel[["asset", "asset_class"]].drop_duplicates()
    registry.to_csv(OUT / "registry_inventory.csv", index=False)
    by_class = registry.groupby("asset_class").size().rename("assets").reset_index()
    by_class.to_csv(OUT / "assets_by_class.csv", index=False)
    first_year = panel.groupby("asset")["date"].min().dt.year.rename("first_year").reset_index()
    first_year = first_year.merge(registry, on="asset", how="left")
    first_year.to_csv(OUT / "asset_first_year.csv", index=False)
    alive = (panel.assign(year=panel.date.dt.year)
             .groupby(["year", "asset_class"])["asset"].nunique()
             .rename("assets_alive").reset_index())
    alive.to_csv(OUT / "assets_alive_by_year.csv", index=False)
    label = panel.dropna(subset=["y_BASE"])
    labels = pd.DataFrame([{
        "rows": len(panel), "assets": panel.asset.nunique(),
        "date_min": panel.date.min(), "date_max": panel.date.max(),
        "label_rows": len(label), "positive_rows": int(label.y_BASE.sum()),
        "prevalence": float(label.y_BASE.mean()),
    }])
    labels.to_csv(OUT / "label_inventory.csv", index=False)
    files = []
    for name, path in (("v3", V3), ("v4", V4), ("v5", V5)):
        if path.exists():
            for f in path.rglob("*"):
                if f.is_file():
                    files.append({"area": name, "path": str(f.relative_to(ROOT)),
                                  "bytes": f.stat().st_size})
    pd.DataFrame(files).to_csv(OUT / "prior_outputs_inventory.csv", index=False)
    manifest = json.loads((V3 / "feature_sets.json").read_text())
    pd.DataFrame([{"feature_set": k, "count": len(v)} for k, v in manifest.items()]
                 ).to_csv(OUT / "v3_feature_sets.csv", index=False)
    print(f"panel rows={len(panel):,} assets={panel.asset.nunique()} "
          f"dates={panel.date.min().date()}..{panel.date.max().date()}")
    print(by_class.to_string(index=False))
    print(labels.to_string(index=False))


if __name__ == "__main__":
    main()
