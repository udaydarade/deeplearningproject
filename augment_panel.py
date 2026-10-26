"""Join momentum ("price-action") and peer-aggregate features onto the
existing unified panel - without recomputing the expensive TDA/GARCH blocks.

Two ideas from the independent peer-index/momentum experiments are absorbed
into the unified framework so they can be compared head-to-head on the same
labels and folds across all 39 tickers x 7 datasets (the original experiments
only covered the four US indices).

Outputs: overwrites ``data/benchmark/unified_panel.csv`` (the join is
outer on the panel's own rows; no rows added or dropped) and
``data/benchmark/feature_report.txt``.

Run with ``py -3.12 augment_panel.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import datasets, features

OUT = datasets.ROOT / "data" / "benchmark"
PANEL_FILE = OUT / "unified_panel.csv"


def main() -> None:
    panel = pd.read_csv(PANEL_FILE, parse_dates=["date"])
    panel = panel.sort_values(["dataset", "ticker", "date"])
    prices_all = {name: datasets.load_prices(name) for name in datasets.DATASETS}

    momentum_tables = []
    peer_tables = []
    for dataset, prices in prices_all.items():
        for ticker in prices.columns:
            series = prices[ticker].dropna()
            if len(series) < 252:
                continue
            momentum = features.momentum_features(series)
            momentum.insert(0, "ticker", ticker)
            momentum.insert(0, "dataset", dataset)
            momentum_tables.append(momentum.reset_index())

        inner = datasets.inner_join(prices)
        peer_frames = features.peer_features(inner)
        for ticker, frame in peer_frames.items():
            frame = frame.copy()
            frame.insert(0, "ticker", ticker)
            frame.insert(0, "dataset", dataset)
            peer_tables.append(frame.reset_index())

    momentum_panel = pd.concat(momentum_tables, ignore_index=True)
    peer_panel = pd.concat(peer_tables, ignore_index=True)
    momentum_panel = momentum_panel.rename(columns={momentum_panel.columns[0]: "date"})
    peer_panel = peer_panel.rename(columns={peer_panel.columns[0]: "date"})

    before = panel.columns.tolist()
    panel = panel.merge(momentum_panel, on=["dataset", "ticker", "date"], how="left")
    panel = panel.merge(peer_panel, on=["dataset", "ticker", "date"], how="left")
    added = [c for c in panel.columns if c not in before]

    panel.to_csv(PANEL_FILE, index=False, date_format="%Y-%m-%d")
    momentum_cols = [c for c in added if c.startswith("price_")]
    peer_cols = [c for c in added if c.startswith("peer_")]
    lines = [
        "Panel augmentation report",
        f"Joined columns: {len(added)} ({len(momentum_cols)} momentum/price-action, "
        f"{len(peer_cols)} peer-aggregate)",
        "Momentum block (per ticker): 1-day return, 5/20/60-day momentum, "
        "20/60-day volatility, 20-day downside volatility, 60/252-day rolling "
        "drawdown, the 20/60 vol ratio and a 20-day skew - all trailing.",
        "Peer block (per ticker, own series excluded): peer return mean/median, "
        "loss breadth, dispersion, mean peer 20d volatility and the mean "
        "trailing 50d pairwise correlation among its peers (inner-join dates).",
        f"Rows unchanged: {len(panel):,}; columns now {len(panel.columns)}.",
    ]
    (OUT / "augment_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
