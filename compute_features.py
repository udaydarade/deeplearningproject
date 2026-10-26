"""Build the unified multi-dataset feature/label panel (one long CSV).

For every (dataset, ticker, date) triple in the seven registered datasets:

* classical CSD features (rolling 50/100 AC1, variance, skew, excess kurtosis;
  20-day realised volatility; 20-day EWMA volatility; stride-refit GARCH(1,1)
  conditional volatility),
* per-ticker Takens TDA (H1 landscape L1/L2, persistence entropy, total
  persistence, finite-bar count and a 36-dim discretised landscape vector
  computed on a z-scored 50-day, 3-D delay-embedded block),
* joint group TDA (same summaries for the Gidea-Katz multivariate point
  cloud, inner-join dates only) and cross-sectional correlation-structure
  features (mean pairwise correlation and max correlation-matrix eigenvalue),
* the objective crash label and its raw forward-min return.

Run from the project root with ``py -3.12 compute_features.py``.
Per-ticker work (TDA + GARCH) is parallelised over a process pool.
Outputs:
  data/benchmark/unified_panel.csv
  data/benchmark/label_thresholds.csv
  data/benchmark/feature_report.txt
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from pipeline import datasets, features, labels

OUT = datasets.ROOT / "data" / "benchmark"


def _ticker_table(dataset: str, ticker: str, prices: pd.DataFrame) -> pd.DataFrame | None:
    """Classical + labels + Takens TDA for one (dataset, ticker)."""
    series = prices[ticker].dropna()
    if len(series) < 200:
        return None
    returns = np.log(series / series.shift(1)).dropna()
    label_frame, info = labels.label_series(series)
    table = label_frame.join(features.classical_features(returns), how="left")
    table = table.join(features.takens_features(returns), how="left")
    table.index.name = "date"
    table.insert(0, "ticker", ticker)
    table.insert(0, "dataset", dataset)
    info.update(dataset=dataset, ticker=ticker)
    table.attrs["info"] = info
    return table


def _dataset_frames(dataset: str) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Joint TDA + cross-sectional features for one dataset."""
    prices = datasets.inner_join(datasets.load_prices(dataset))
    return dataset, features.joint_tda_frame(prices), features.cross_sectional_features(prices)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    blocks, threshold_rows = [], []

    print("[stage 1/2] joint TDA + cross-sectional features (one task per dataset)")
    joint_lookup: dict[str, pd.DataFrame] = {}
    cross_lookup: dict[str, pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=7) as pool:
        futures = [pool.submit(_dataset_frames, name) for name in datasets.DATASETS]
        for future in as_completed(futures):
            name, joint, cross = future.result()
            joint_lookup[name], cross_lookup[name] = joint, cross
            print(f"  [{name}] joint={len(joint)} rows, cross={len(cross)} rows", flush=True)

    print("[stage 2/2] per-ticker features: CSD (+GARCH), labels, Takens TDA")
    ticker_tasks = []
    for dataset in datasets.DATASETS:
        prices = datasets.load_prices(dataset)
        for ticker in prices.columns:
            if prices[ticker].dropna().shape[0] >= 200:
                ticker_tasks.append((dataset, ticker, prices))
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_ticker_table, d, t, p): (d, t) for d, t, p in ticker_tasks}
        for future in as_completed(futures):
            dataset, ticker = futures[future]
            table = future.result()
            if table is None:
                continue
            info = table.attrs["info"]
            joint = joint_lookup[dataset]
            cross = cross_lookup[dataset]
            table = table.join(joint, how="left").join(cross, how="left")
            blocks.append(table)
            threshold_rows.append(info)
            print(f"  [{dataset}/{ticker}] rows={len(table)} {info['rule']}", flush=True)

    unified = pd.concat(blocks).reset_index()
    if "index" in unified.columns:
        unified = unified.rename(columns={"index": "date"})
    else:
        unified = unified.rename(columns={unified.columns[0]: "date"})
    feature_groups = [
        ["date", "dataset", "ticker", "label", "forward_min_return"],
        sorted(c for c in unified.columns if c.startswith(("csd_", "realized_", "ewma_"))),
        sorted(c for c in unified.columns if c.startswith("tda_idx_")),
        sorted(c for c in unified.columns if c.startswith("tda_joint_")),
    ]
    used = {c for group in feature_groups for c in group}
    column_order = [c for group in feature_groups for c in group] + [
        c for c in unified.columns if c not in used
    ]
    unified = unified[column_order].sort_values(["dataset", "ticker", "date"])
    unified.to_csv(OUT / "unified_panel.csv", index=False, date_format="%Y-%m-%d")

    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(OUT / "label_thresholds.csv", index=False)

    takens_cols = [c for c in unified.columns if c.startswith("tda_idx_")]
    joint_cols = [c for c in unified.columns if c.startswith("tda_joint_")]
    cross_cols = [c for c in unified.columns if c.startswith("csd_meanpaircorr")]
    lines = [
        "Unified feature/label panel report (v2)",
        f"Datasets: {len(datasets.DATASETS)}; rows: {len(unified):,}; "
        f"tickers: {unified.ticker.nunique()}",
        f"Date span: {unified.date.min()} -> {unified.date.max()}",
        "Classical block: rolling 50/100 AC1, variance, skew, excess kurtosis; "
        "20-day realised and EWMA volatility; GARCH(1,1) conditional volatility "
        f"(refit every {features.GARCH_STRIDE} sessions on a trailing "
        f"{features.GARCH_WINDOW}-day window, ffill forward only).",
        f"TDA per ticker: {len(takens_cols)} cols; joint multivariate: "
        f"{len(joint_cols)} cols (summaries + {features.LANDSCAPE_LEVELS}-level "
        f"x {features.LANDSCAPE_GRID}-point landscape grid; window "
        f"{features.TDA_WINDOW}; Takens dim {features.TAKENS_DIM}/delay "
        f"{features.TAKENS_DELAY}).",
        f"Cross-sectional: {len(cross_cols)} mean-eigen/correlation features.",
        f"Labels: forward {labels.HORIZON}-session minimum simple return <= per-ticker "
        f"{labels.LABEL_QUANTILE:.0%} threshold (calibration-only estimation).",
        "",
        "Threshold table:",
        thresholds.to_string(index=False),
    ]
    (OUT / "feature_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {len(unified):,} rows x {len(unified.columns)} cols -> {OUT / 'unified_panel.csv'}")


if __name__ == "__main__":
    main()
