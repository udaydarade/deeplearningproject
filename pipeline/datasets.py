"""Dataset registry and loaders for the unified multi-dataset benchmark.

Each entry points at a wide CSV of already-downloaded daily prices (an index
column and one column per ticker). The files themselves are produced by
``download_market_data.py`` (group A/B/C) and ``run_cross_asset_extension.py``
(sector groups), so no benchmark step touches the network.

Adding a new market dataset therefore only requires appending a line to
``DATASETS`` pointing at a prices CSV with the same schema.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Stable registry: dataset name -> prices CSV path. Column names must be the
# exact Yahoo/Stooq tickers because labels and features are computed per column.
DATASETS: dict[str, Path] = {
    "group_a_us_indices": DATA / "group_a_original_indices" / "prices.csv",
    "group_b_cross_asset": DATA / "group_b_cross_asset" / "prices.csv",
    "crypto": DATA / "group_c_panel" / "crypto_prices.csv",
    "international": DATA / "group_c_panel" / "international_prices.csv",
    "financials": DATA / "extension" / "financials_prices.csv",
    "energy": DATA / "extension" / "energy_prices.csv",
    "tech": DATA / "extension" / "tech_prices.csv",
}


def load_prices(dataset: str) -> pd.DataFrame:
    """Load one dataset's price panel (wide frame indexed by date)."""
    path = DATASETS[dataset]
    prices = pd.read_csv(path, parse_dates=[0], index_col=0).sort_index()
    prices.index.name = "date"
    return prices.apply(pd.to_numeric, errors="coerce")


def inner_join(prices: pd.DataFrame) -> pd.DataFrame:
    """Restrict to rows where every ticker quotes (needed for joint TDA)."""
    return prices.dropna(how="any")
