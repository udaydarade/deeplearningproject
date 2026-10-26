"""Past-only severity ranks for V6 crash and surge labels."""
from __future__ import annotations
import numpy as np


def severity_ranks(price: np.ndarray, horizon: int, min_history: int = 750):
    """Past-only severity ranks attached to the forecast origin.

    At origin ``i``, the target is the worst/best return over the next
    ``horizon`` sessions. The expanding reference set contains only prior
    origins whose complete horizon has ended by ``i`` (origins <= i-horizon).
    This keeps the outcome at its forecast date and prevents future outcomes
    from entering its threshold.
    """
    p = np.asarray(price, dtype=float)
    n = len(p)
    down = np.full(n, np.nan)
    up = np.full(n, np.nan)
    if horizon < 1 or min_history < 0:
        raise ValueError("horizon must be positive and min_history non-negative")
    length = n - horizon
    if length <= 0:
        return down, up
    # One row per forecast origin; columns are returns over 1..N sessions.
    paths = np.column_stack([p[k:k + length] / p[:length] - 1.0
                             for k in range(1, horizon + 1)])
    finite_path = np.isfinite(paths)
    m = np.min(np.where(finite_path, paths, np.inf), axis=1)
    M = np.max(np.where(finite_path, paths, -np.inf), axis=1)
    # Match the project label definition: an origin is valid only when its
    # entire forward window is observed. Partial windows would change the
    # effective horizon from asset to asset.
    finite = finite_path.all(axis=1) & np.isfinite(m) & np.isfinite(M)

    # Coordinate compression plus Fenwick counts gives exact empirical ranks
    # without using future outcomes in the counts or quadratic list insertion.
    down_coords = np.unique(m[finite])
    up_coords = np.unique(M[finite])
    down_tree = np.zeros(len(down_coords) + 1, dtype=np.int64)
    up_tree = np.zeros(len(up_coords) + 1, dtype=np.int64)

    def add(tree, index):
        while index < len(tree):
            tree[index] += 1
            index += index & -index

    def prefix(tree, index):
        total = 0
        while index:
            total += tree[index]
            index -= index & -index
        return total

    count = 0
    for origin in range(length):
        matured = origin - horizon
        if matured >= 0 and finite[matured]:
            add(down_tree, int(np.searchsorted(down_coords, m[matured])) + 1)
            add(up_tree, int(np.searchsorted(up_coords, M[matured])) + 1)
            count += 1
        if finite[origin] and count >= min_history:
            d_le = prefix(down_tree, int(np.searchsorted(down_coords, m[origin], side="right")))
            u_lt = prefix(up_tree, int(np.searchsorted(up_coords, M[origin], side="left")))
            down[origin] = (1 + d_le) / (1 + count)
            up[origin] = (1 + count - u_lt) / (1 + count)
    return down, up
