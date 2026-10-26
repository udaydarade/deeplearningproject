"""Classical critical-slowing-down (CSD) and topological (TDA) features.

Everything here is backward-looking only: a feature dated t uses returns
through t and never observations after t.

Classical CSD block (per ticker, trailing rolling windows of 50 and 100
sessions): lag-1 autocorrelation within the window, variance, skewness and
excess kurtosis, plus realised volatility (20d), an EWMA volatility
(RiskMetrics-style span 20) and GARCH(1,1) conditional volatility. Fitting a
GARCH model on every trailing window is prohibitive for ~39 tickers, so the
model is refit on a trailing 250-day window every ``GARCH_STRIDE`` sessions
and the last conditional variance (a past-only quantity) is forward-filled in
between - no future information leaks into the standing estimate.

Cross-sectional block (per dataset, multivariate, broadcast to group
members): mean off-diagonal pairwise correlation and the largest eigenvalue
of the correlation matrix over trailing 50/100-day windows. Rising average
correlation ("market mode" taking over) is the classic cross-market early
warning; these features are what bind "combining multiple indices" into the
feature space alongside the joint TDA landscape.

TDA block has two variants:

* Joint (per dataset, multivariate) - the Gidea-Katz sliding-window scheme.
  The point cloud at date t is the trailing n=50 block of the group's jointly
  available log-return vectors (z-scored per window), filtered with
  Euclidean Vietoris-Rips, and summarised via H1 persistence-landscape norms
  (closed-form integrals), persistence entropy, total persistence and the
  finite-bar count, plus a discretised landscape curve vector
  (``LANDSCAPE_LEVELS`` x ``LANDSCAPE_GRID``) usable directly as deep-model
  input channels. Joint features only exist on dates where every ticker in
  the group quotes.
* Per-ticker (Takens) - the same scheme applied to a scalar series via
  time-delay embedding. The trailing 50-day z-scored return block is folded
  into 3-D delay vectors (delay 1), giving a 48x3 point cloud, again filtered
  and reduced to the same summary set. This gives every ticker its own
  topological signal even inside heterogeneous groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from arch import arch_model
except ImportError:  # GARCH block degrades to NaN, never blocks the pipeline
    arch_model = None

from ripser import ripser

ROLL_WINDOWS = (50, 100)
TDA_WINDOW = 50       # trailing window length for both TDA variants
TAKENS_DIM = 3        # embedding dimension for the per-ticker point cloud
TAKENS_DELAY = 1      # embedding delay
LANDSCAPE_GRID = 12   # grid points per landscape level
LANDSCAPE_LEVELS = 3  # top-k landscape levels kept as a vector
LANDSCAPE_UPPER = 2.5 # grid upper bound (z-scored log-return units)
GARCH_WINDOW = 250    # trailing fit window for GARCH(1,1)
GARCH_STRIDE = 5      # refit cadence, sessions


# ---------------------------------------------------------------- classical
def classical_features(returns: pd.Series) -> pd.DataFrame:
    """CSD indicator block for one return series (dated by window end)."""
    out = pd.DataFrame(index=returns.index)
    for window in ROLL_WINDOWS:
        roll = returns.rolling(window, min_periods=window)
        out[f"csd_ac1_w{window}"] = returns.rolling(
            window - 1, min_periods=window - 1
        ).corr(returns.shift(1))
        out[f"csd_variance_w{window}"] = roll.var()
        out[f"csd_skew_w{window}"] = roll.skew()
        out[f"csd_kurtosis_w{window}"] = roll.kurt()
    out["realized_vol_w20"] = returns.rolling(20, min_periods=20).std()
    out["ewma_vol_w20"] = returns.ewm(span=20, min_periods=20).std()
    out["csd_garch_vol"] = garch_vol(returns)
    return out


def garch_vol(returns: pd.Series, window: int = GARCH_WINDOW,
              stride: int = GARCH_STRIDE) -> pd.Series:
    """GARCH(1,1) conditional volatility, point-in-time with infrequent refits.

    A model refit at date t yields the conditional volatility *at t* - reusing
    it for earlier dates would leak, so refit outputs are only carried forward
    (ffill) until the next refit, never backward.
    """
    if arch_model is None:
        return pd.Series(np.nan, index=returns.index)
    values = returns.to_numpy(float)
    estimates = pd.Series(np.nan, index=returns.index)
    for end in range(window - 1, len(values), stride):
        block = values[end - window + 1 : end + 1]
        try:
            fit = arch_model(block * 100.0, mean="Constant", vol="GARCH", p=1, q=1,
                             dist="normal", rescale=False).fit(
                                 disp="off", show_warning=False, options={"maxiter": 200})
            estimates.iloc[end] = float(fit.conditional_volatility[-1]) / 100.0
        except Exception:
            continue
    # Carry the most recent (past-only) estimate forward; refs happen backward-
    # indexing the block only, so no future information ever enters date t.
    return estimates.ffill(limit=stride - 1)


def cross_sectional_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Group-level correlation-structure early warnings (dated by window end)."""
    returns = np.log(prices / prices.shift(1)).dropna(how="any")
    values = returns.to_numpy(float)
    n_assets = values.shape[1]
    if n_assets < 2:
        return pd.DataFrame(index=returns.index)
    out = pd.DataFrame(index=returns.index)
    for window in ROLL_WINDOWS:
        mean_corr, max_eig = [], []
        for end in range(window, len(values) + 1):
            block = values[end - window : end]
            std = block.std(axis=0, ddof=0)
            std[std == 0] = np.nan
            normed = block / std
            corr = np.corrcoef(normed, rowvar=False)
            offdiag = corr[np.triu_indices(n_assets, k=1)]
            mean_corr.append(float(np.nanmean(offdiag)))
            eigenvalues = np.linalg.eigvalsh(corr)
            max_eig.append(float(eigenvalues[-1]))
        dates = returns.index[window - 1 :]
        out[f"csd_meanpaircorr_w{window}"] = pd.Series(mean_corr, index=dates)
        out[f"csd_maxeig_w{window}"] = pd.Series(max_eig, index=dates)
    return out


# ----------------------------------------------------------------- TDA core
def finite_h1_bars(points: np.ndarray) -> np.ndarray:
    """Finite H1 intervals of the Euclidean Vietoris-Rips filtration."""
    diagrams = ripser(points, maxdim=1, metric="euclidean")["dgms"]
    if len(diagrams) < 2:
        return np.empty((0, 2))
    bars = diagrams[1]
    return bars[np.isfinite(bars).all(axis=1)]


def landscape_stats(bars: np.ndarray) -> tuple[float, float, float, int]:
    """Closed-form landscape L1/L2, persistence entropy and bar count.

    L1 = sum(p^2)/4 and L2 = sqrt(sum(p^3)/12) integrate each bar's landscape
    tent analytically (same formulas as tda_replication.py), so no numerical
    grid is involved and values are directly comparable across variants.
    """
    if bars.size == 0:
        return 0.0, 0.0, 0.0, 0
    persistence = bars[:, 1] - bars[:, 0]
    l1 = float(np.square(persistence).sum() / 4.0)
    l2 = float(np.sqrt(np.power(persistence, 3).sum() / 12.0))
    probs = persistence / persistence.sum()
    entropy = float(-(probs * np.log(probs)).sum())
    return l1, l2, entropy, int(len(bars))


LANDSCAPE_GRID_POINTS = np.linspace(0.0, LANDSCAPE_UPPER, LANDSCAPE_GRID)


def landscape_vector(bars: np.ndarray) -> np.ndarray:
    """Top-k landscape curves sampled on a fixed grid -> flat channel vector.

    The grid is defined in z-scored log-return units and shared by all dates,
    so the vector is comparable across time and becomes native deep-model
    input (persistences computed on standardised windows are O(1)).
    """
    if bars.size == 0:
        return np.zeros(LANDSCAPE_LEVELS * LANDSCAPE_GRID)
    grid = LANDSCAPE_GRID_POINTS
    births, deaths = bars[:, 0][:, None], bars[:, 1][:, None]
    tents = np.maximum(0.0, np.minimum(grid[None, :] - births, deaths - grid[None, :]))
    tents.sort(axis=0)
    tents = tents[::-1]
    fixed = np.zeros((LANDSCAPE_LEVELS, LANDSCAPE_GRID))
    kept = min(LANDSCAPE_LEVELS, tents.shape[0])
    fixed[:kept] = tents[:kept]
    return fixed.reshape(-1).copy()  # level-major flattening


def extended_landscape_stats(bars: np.ndarray) -> dict:
    """All scalar summaries plus the discretised landscape vector."""
    l1, l2, entropy, betti = landscape_stats(bars)
    persistence = (bars[:, 1] - bars[:, 0]) if bars.size else np.zeros(0)
    values = {
        "l1": l1, "l2": l2, "entropy": entropy, "betti": betti,
        "total_persistence": float(persistence.sum()),
    }
    prefix = "grid_L{}_g{}"
    vector = landscape_vector(bars)
    for level in range(LANDSCAPE_LEVELS):
        for j in range(LANDSCAPE_GRID):
            values[prefix.format(level + 1, j)] = float(vector[level * LANDSCAPE_GRID + j])
    return values


def _nominal_columns() -> list[str]:
    """Nominal landscape-summary names for an empty bars set, stable order."""
    return ["l1", "l2", "entropy", "betti", "totalpers"] + [
        f"grid_L{level}_g{j}"
        for level in range(1, LANDSCAPE_LEVELS + 1) for j in range(LANDSCAPE_GRID)
    ]


def stats_row(stats: dict) -> list[float]:
    """Stats dict -> row values ordered as _nominal_columns."""
    keys = ["l1", "l2", "entropy", "betti", "total_persistence"]
    ordered = [stats[key] for key in keys] + [
        stats[f"grid_L{level}_g{j}"]
        for level in range(1, LANDSCAPE_LEVELS + 1) for j in range(LANDSCAPE_GRID)
    ]
    return ordered


# ------------------------------------------------------------- per-ticker TDA
def takens_point_cloud(block: np.ndarray) -> np.ndarray:
    """Delay-embedding point cloud from a z-scored return block."""
    std = block.std(ddof=0)
    if not np.isfinite(std) or std == 0.0:
        std = 1.0
    z = (block - block.mean()) / std
    return np.stack(
        [z[i : i + TAKENS_DIM] for i in range(len(z) - TAKENS_DIM + 1)], axis=0
    )


def takens_features(returns: pd.Series, window: int = TDA_WINDOW) -> pd.DataFrame:
    """Per-ticker TDA summary for every date with a full trailing window."""
    values = returns.to_numpy(float)
    rows = []
    for end in range(window, len(values) + 1):
        block = values[end - window : end]
        stats = extended_landscape_stats(finite_h1_bars(takens_point_cloud(block)))
        rows.append((returns.index[end - 1], *stats_row(stats)))
    columns = ["date", *[f"tda_idx_{name}" for name in _nominal_columns()]]
    return pd.DataFrame(rows, columns=columns).set_index("date")


# --------------------------------------------------- momentum ("price") block
# Absorbed from the independent peer/momentum experiments: reactive
# price-action features (returns, multi-scale momentum, realised and downside
# volatility, rolling drawdowns, volatility ratio, short skew).
MOMENTUM_SCALES = (5, 20, 60)


def momentum_features(prices: pd.Series) -> pd.DataFrame:
    """Per-ticker price-action block (all strictly trailing windows)."""
    prices = prices.sort_index()
    returns = np.log(prices / prices.shift(1))
    out = pd.DataFrame(index=prices.index)
    out["price_return_1d"] = returns
    for scale in MOMENTUM_SCALES:
        out[f"price_momentum_{scale}d"] = returns.rolling(scale, min_periods=scale).sum()
    out["price_volatility_20d"] = returns.rolling(20, min_periods=20).std()
    out["price_volatility_60d"] = returns.rolling(60, min_periods=60).std()
    downside_sq = returns.clip(upper=0.0).pow(2)
    out["price_downside_vol_20d"] = np.sqrt(downside_sq.rolling(20, min_periods=20).mean())
    out["price_drawdown_60d"] = prices / prices.rolling(60, min_periods=60).max() - 1
    out["price_drawdown_252d"] = prices / prices.rolling(252, min_periods=252).max() - 1
    out["price_volratio_20_60"] = (out["price_volatility_20d"] /
                                   out["price_volatility_60d"].clip(lower=1e-12))
    out["price_skew_20d"] = returns.rolling(20, min_periods=20).skew()
    return out


# -------------------------------------------------------- peer-aggregate block
def peer_features(prices: pd.DataFrame, corr_window: int = 50) -> dict[str, pd.DataFrame]:
    """Per-ticker peer aggregates from the *other* members of its group.

    Generalises the independent peer-index experiments (which ran only the
    four US indices): every ticker gets the joint behaviour of the other
    members of its dataset - peer return mean/median, loss breadth, return
    dispersion, average peer volatility and the mean trailing pairwise
    correlation among its peers. Everything is computed on the group's
    inner-join dates; the target's own series is always excluded.
    """
    returns = np.log(prices / prices.shift(1)).dropna(how="any")
    names = list(returns.columns)
    if len(names) < 2:
        return {}
    # Group-wide mean pairwise rolling correlation (w50), once.
    pair_curves = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            pair_curves.append(returns[names[left]].rolling(
                corr_window, min_periods=corr_window).corr(returns[names[right]]))
    group_corr = pd.concat(pair_curves, axis=1).mean(axis=1).rename("peer_corr_w50")

    result: dict[str, pd.DataFrame] = {}
    for ticker in names:
        peers = returns.drop(columns=[ticker])
        frame = pd.DataFrame(index=returns.index)
        frame["peer_return_mean"] = peers.mean(axis=1)
        frame["peer_return_median"] = peers.median(axis=1)
        frame["peer_breadth"] = (peers < 0).mean(axis=1)
        frame["peer_dispersion"] = peers.std(axis=1, ddof=1)
        frame["peer_vol_mean"] = peers.rolling(20, min_periods=20).std().mean(axis=1)
        peer_pairs = [
            returns[a].rolling(corr_window, min_periods=corr_window).corr(returns[b])
            for i, a in enumerate(names) if a != ticker
            for b in names[i + 1:] if b != ticker
        ]
        frame["peer_corr_w50"] = pd.concat(peer_pairs, axis=1).mean(axis=1)
        result[ticker] = frame
    return result
    """Group-level multivariate TDA on the inner-joined return panel."""
    returns = np.log(prices / prices.shift(1)).dropna(how="any")
    values = returns.to_numpy(float)
    rows = []
    for end in range(window, len(values) + 1):
        block = values[end - window : end]
        std = block.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        stats = extended_landscape_stats(
            finite_h1_bars((block - block.mean(0)) / std))
        rows.append((returns.index[end - 1], *stats_row(stats)))
    columns = ["date", *[f"tda_joint_{name}" for name in _nominal_columns()]]
    return pd.DataFrame(rows, columns=columns).set_index("date")
