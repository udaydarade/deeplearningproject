"""Inventory frozen V3/V4 inputs for V5 without modifying them."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V3 = ROOT / "data" / "v3"
V4 = ROOT / "data" / "v4"
OUT = ROOT / "data" / "v5"


def runs(mask: np.ndarray, gap: int = 5) -> list[tuple[int, int]]:
    pos = np.flatnonzero(mask)
    if not len(pos):
        return []
    out, start, prev = [], int(pos[0]), int(pos[0])
    for i in pos[1:]:
        i = int(i)
        if i - prev > gap + 1:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(V3 / "panel.parquet", columns=["asset", "date", "y_BASE"])
    panel.date = pd.to_datetime(panel.date)
    panel = panel.sort_values(["asset", "date"])
    panel["year"] = panel.date.dt.year
    annual = []
    for year, group in panel.groupby("year", sort=True):
        valid = group.y_BASE.notna()
        positive = group.y_BASE.eq(1)
        events = 0
        for _, g in group.loc[valid].groupby("asset", sort=False):
            events += len(runs(g.y_BASE.to_numpy() == 1))
        annual.append({"year": int(year), "first_date": group.date.min(), "last_date": group.date.max(),
                       "assets_observed": int(group.asset.nunique()), "valid_asset_days": int(valid.sum()),
                       "positive_asset_days": int(positive.sum()),
                       "prevalence": float(positive.sum() / valid.sum()) if valid.any() else np.nan,
                       "crash_events_starting_in_year": events})
    pd.DataFrame(annual).to_csv(OUT / "inventory_by_year.csv", index=False)

    eras = [("E0",1928,1949),("E1",1950,1979),("E2",1980,1993),("E3",1994,1999),
            ("E4",2000,2003),("E5",2004,2009),("E6",2010,2014),("E7",2015,2017),
            ("E8",2018,2019),("E9",2020,2021),("E10",2022,9999)]
    era_rows=[]
    for name, first, last in eras:
        g=panel[(panel.year>=first)&(panel.year<=last)]
        valid=g.y_BASE.notna(); pos=g.y_BASE.eq(1); event_count=0
        for _, a in g.loc[valid].groupby("asset",sort=False):
            event_count += len(runs(a.y_BASE.to_numpy()==1))
        era_rows.append({"era":name,"first_year":first,"last_year":last,
                         "first_date":g.date.min() if len(g) else pd.NaT,
                         "last_date":g.date.max() if len(g) else pd.NaT,
                         "assets":int(g.asset.nunique()),"valid_rows":int(valid.sum()),
                         "positive_rows":int(pos.sum()),
                         "prevalence":float(pos.sum()/valid.sum()) if valid.any() else np.nan,
                         "crash_events":event_count})
    pd.DataFrame(era_rows).to_csv(OUT / "inventory_by_era.csv",index=False)

    manifest=json.loads((V3/"feature_sets.json").read_text(encoding="utf-8"))
    scores=list((V3/"scores").glob("*.parquet")) if (V3/"scores").exists() else []
    v4=list(V4.rglob("*.parquet")) if V4.exists() else []
    feature_blocks=json.loads((V4/"feature_blocks.json").read_text(encoding="utf-8")) if (V4/"feature_blocks.json").exists() else {}
    audit={"date_first":str(panel.date.min().date()),"date_last":str(panel.date.max().date()),
           "assets":int(panel.asset.nunique()),"panel_rows":int(len(panel)),
           "v3_feature_counts":{k:len(v) for k,v in manifest.items()},
           "v3_score_parquet_count":len(scores),"v3_score_names":sorted(p.name for p in scores),
           "v4_parquet_paths":[str(p.relative_to(ROOT)) for p in v4],
           "v4_raw_config":feature_blocks.get("config",{}),
           "past_only_threshold":"per-asset expanding quantile; 750 prior fully observed forecast outcomes minimum",
           "cpu_logical":int(__import__("os").cpu_count() or 1)}
    (OUT/"inventory.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(f"[V5-P0] panel={len(panel):,} rows, {panel.asset.nunique()} assets, {panel.date.min().date()}..{panel.date.max().date()}")
    print(f"[V5-P0] BASE is past-only expanding q2%, >=750 prior outcomes; annual and era prevalence/events saved")
    print(f"[V5-P0] V3 features={audit['v3_feature_counts']}; score parquets={len(scores)}; V4 parquet files={len(v4)}")
    print(f"[V5-P0] workers cap=8; machine logical CPUs={audit['cpu_logical']}; V5 output=data/v5/")


if __name__ == "__main__":
    main()
