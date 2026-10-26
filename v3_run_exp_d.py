"""EXP-V3-D pooled GBM universe/history ablations (Scheme W)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from v3_download_data import OUT
from v3_run_experiments import FOLDS, US4, _evaluate_model, _fit_gbm, _tabular_view, splits

RESULT = OUT / "exp_d_results.csv"
FEATURE_SET = "F0"  # only complete feature set across all five folds at EXP-A time limit


def universes(panel: pd.DataFrame, registry: pd.DataFrame) -> dict[str, set[str]]:
    assets = registry.loc[~registry.is_state.fillna(False)].copy()
    equity = assets.loc[assets.asset_class.eq("equity_index")].sort_values("n_obs", ascending=False)
    us_etfs = assets.loc[assets.asset_class.eq("us_etf")].sort_values("n_obs", ascending=False)
    approx30 = set(equity.head(20).ticker) | set(us_etfs.head(10).ticker)
    nonstock = set(assets.loc[assets.asset_class.ne("stock"), "ticker"])
    all_assets = set(assets.ticker)
    return {
        "us4": set(US4),
        "approx30_index_etf": approx30,
        "all_nonstock": nonstock,
        "all_assets": all_assets,
        "all_assets_since_1997": all_assets,
    }


def append(rows: list[dict]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    exists = RESULT.exists()
    frame.to_csv(RESULT, mode="a" if exists else "w", header=not exists,
                 index=False, float_format="%.10g")


def main() -> None:
    panel = pd.read_parquet(OUT / "panel.parquet")
    shared = pd.read_parquet(OUT / "market_features.parquet")
    registry = pd.read_csv(OUT / "registry.csv")
    feature_sets = json.loads((OUT / "feature_sets.json").read_text(encoding="utf-8"))
    features = feature_sets[FEATURE_SET]
    local = [c for c in features if c in panel.columns]
    external = [c for c in features if c in shared.columns]
    if len(local) + len(external) != len(set(features)):
        raise ValueError("F0 manifest columns are unavailable")
    metadata = ["asset", "date", "source_dataset", "asset_class", "y_BASE"]
    universes_by_name = universes(panel, registry)
    existing = set()
    if RESULT.exists():
        old = pd.read_csv(RESULT)
        existing = set(zip(old.universe, old.fold))
    seed = 2718
    for name, assets in universes_by_name.items():
        part = panel.loc[panel.asset.isin(assets), metadata + local]
        if name == "all_assets_since_1997":
            part = part.loc[part.date >= pd.Timestamp("1997-01-01")]
        data = part.merge(shared[external].reset_index(), on="date", how="left", validate="many_to_one")
        data = data.sort_values(["asset", "date"], kind="stable").reset_index(drop=True)
        print(f"[EXP-D] universe={name} assets={data.asset.nunique()} rows={len(data)}", flush=True)
        for fold_i, (first, last) in enumerate(FOLDS):
            fold = f"{first}-{last if last != 9999 else 'latest'}"
            if (name, fold) in existing:
                print(f"[resume] {name} {fold}", flush=True)
                continue
            test_start, test_end, val_start, val_end, train_end = splits(data, first, last)
            train_mask = data.date.le(train_end) & data.y_BASE.notna()
            val_mask = data.date.between(val_start, val_end) & data.y_BASE.notna()
            test_mask = data.date.between(test_start, test_end) & data.y_BASE.notna()
            train_ix = np.flatnonzero(train_mask.to_numpy())
            val_ix = np.flatnonzero(val_mask.to_numpy())
            test_ix = np.flatnonzero(test_mask.to_numpy())
            if not len(train_ix) or not len(val_ix) or not len(test_ix):
                print(f"[skip] {name} {fold}: n={len(train_ix)}/{len(val_ix)}/{len(test_ix)}", flush=True)
                continue
            y = data.y_BASE.fillna(0).to_numpy(dtype=np.int8)
            print(f"[start] {name} {fold}: train/val/test={len(train_ix)}/{len(val_ix)}/{len(test_ix)}; positives={int(y[train_ix].sum())}/{int(y[val_ix].sum())}/{int(y[test_ix].sum())}", flush=True)
            tab, names = _tabular_view(data, features)
            model, val_pred, all_pred, _ = _fit_gbm(tab, y, train_ix, val_ix, names, seed + fold_i)
            del tab, model
            data["score"] = all_pred
            val = data.loc[val_mask, ["asset", "date", "y_BASE", "score", "asset_class", "source_dataset"]].rename(columns={"y_BASE": "y"})
            test = data.loc[test_mask, ["asset", "date", "y_BASE", "score", "asset_class", "source_dataset"]].rename(columns={"y_BASE": "y"})
            rows = _evaluate_model(data, val, test, FEATURE_SET, "gbm", fold, None, seed)
            for row in rows:
                row["exp"] = "EXP-V3-D"
                row["universe"] = name
            append(rows)
            existing.add((name, fold))
            print(f"[done] {name} {fold}: test crash events={rows[0].get('crashes_total')}; result rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
