"""V3 feature/label point-in-time audit on ten seeded dates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from v3_build_features import _asset_features, _market_row, _tda_one_date
from v3_download_data import OUT
from v3_prepare_data import forward_min


def _close(a, b, feature: str, date, asset, rows: list[dict]) -> None:
    if pd.isna(a) and pd.isna(b):
        return
    if not np.isfinite(a) or not np.isfinite(b):
        if pd.isna(a) == pd.isna(b):
            return
        raise AssertionError(f"finite mismatch at {date}/{asset}/{feature}: {a} vs {b}")
    diff = abs(float(a) - float(b))
    rows.append({"date": date, "asset": asset, "block": feature.split("_", 1)[0],
                 "feature": feature, "abs_diff": diff})
    if diff >= 1e-5:
        raise AssertionError(f"leak/recompute mismatch at {date}/{asset}/{feature}: diff={diff}")


def main() -> None:
    returns = pd.read_parquet(OUT / "returns.parquet").astype("float32")
    prices = pd.read_parquet(OUT / "prices.parquet").astype("float32")
    states = pd.read_parquet(OUT / "states.parquet")
    registry = pd.read_csv(OUT / "registry.csv")
    panel = pd.read_parquet(OUT / "panel.parquet")
    shared = pd.read_parquet(OUT / "market_features.parquet")
    labels = pd.read_parquet(OUT / "labels.parquet")
    classes = registry.set_index("ticker").asset_class
    nonstock = [c for c in returns if classes.get(c) != "stock"]
    market_ret = returns[nonstock].mean(axis=1)
    candidates = np.arange(300, len(returns) - 300, max(1, len(returns) // 20))
    rng = np.random.default_rng(1701)
    chosen = np.sort(rng.choice(candidates, size=10, replace=False))
    rows = []
    for i in chosen:
        date = returns.index[i]
        expected_market = _market_row(returns.loc[:date], prices.loc[:date],
                                      states.loc[:date], registry, len(returns.loc[:date]) - 1)
        for col, value in expected_market.items():
            if col in shared.columns:
                _close(shared.at[date, col], value, col, date, "MARKET", rows)
        eligible = [c for c in prices if pd.notna(prices.at[date, c])]
        asset = str(rng.choice(eligible))
        p = prices[asset].loc[:date].dropna()
        r = returns[asset].reindex(p.index)
        expected_local = _asset_features(p, r, market_ret.loc[:date].reindex(p.index)).iloc[-1]
        saved = panel.loc[(panel.asset == asset) & (panel.date == date)]
        if saved.empty:
            continue
        saved = saved.iloc[0]
        for col, value in expected_local.items():
            if col in panel.columns:
                _close(saved[col], value, col, date, asset, rows)
        # Expanding label threshold is rebuilt from outcomes whose full horizon
        # had ended by this origin. The origin's own future label is not used.
        label_row = labels.loc[(labels.asset == asset) & (labels.date == date)]
        if len(label_row):
            n, q = 15, .02
            loc = p.index.get_loc(date)
            forward = forward_min(p, n)
            # Match the labels' exact construction: m(t) is the min over the
            # next N observed asset sessions; q(t) sees m values through t-N.
            prior_m = forward.iloc[:loc - n + 1].dropna().to_numpy(dtype=float)
            if len(prior_m) >= 750:
                expected_q = float(np.quantile(prior_m, q, method="linear"))
                _close(label_row.iloc[0]["q_BASE"], expected_q, "q_BASE", date, asset, rows)

    # Recompute full TDA rows at ten stride dates using only the history up to
    # each date; compare every available raw topological coordinate.
    all_nonstock = [c for c in returns if classes.get(c) != "stock"]
    history_count = returns[all_nonstock].notna().cumsum().to_numpy(dtype=np.int32)
    stride_dates = returns.index[9::10]
    tda_chosen = rng.choice(stride_dates[max(0, len(stride_dates) - 200):], size=10, replace=False)
    ntda = 0
    for date in sorted(tda_chosen):
        ix = returns.index.get_loc(date)
        hist = returns.iloc[ix - 119:ix + 1][all_nonstock].to_numpy(dtype=np.float32)
        date = pd.Timestamp(date)
        _, expected, _ = _tda_one_date((date, int(date.strftime("%Y%m%d")), hist,
                                        history_count[ix], 3))
        for col, value in expected.items():
            if col in shared.columns:
                _close(shared.at[date, col], value, col, date, "TDA", rows)
                ntda += 1
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "leak_test.csv", index=False)
    print(f"[leak test] passed: 10 dates; {len(rows)} numeric comparisons; {ntda} TDA comparisons; max_diff={result.abs_diff.max() if len(result) else 0:.3g}")


if __name__ == "__main__":
    main()
