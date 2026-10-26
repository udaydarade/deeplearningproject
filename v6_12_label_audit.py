"""Fast correctness checks for V6 forward labels and expanding ranks."""
from __future__ import annotations
import numpy as np
from pipeline_v6.labels import severity_ranks


def brute(price: np.ndarray, horizon: int, min_history: int):
    p = np.asarray(price, float)
    down = np.full(len(p), np.nan); up = np.full(len(p), np.nan)
    length = len(p) - horizon
    m = np.full(max(0, length), np.nan); M = np.full(max(0, length), np.nan)
    for i in range(length):
        r = p[i + 1:i + horizon + 1] / p[i] - 1
        m[i], M[i] = np.min(r), np.max(r)
    for i in range(length):
        # At forecast origin i, only target windows ending by i are known.
        stop = max(0, i - horizon + 1)
        past_d = m[:stop][np.isfinite(m[:stop])]
        past_u = M[:stop][np.isfinite(M[:stop])]
        if len(past_d) >= min_history:
            down[i] = (1 + np.sum(past_d <= m[i])) / (1 + len(past_d))
            up[i] = (1 + np.sum(past_u >= M[i])) / (1 + len(past_u))
    return down, up


def main():
    rng = np.random.default_rng(20261025)
    p = np.exp(np.cumsum(rng.normal(0, .02, 1200)))
    got = severity_ranks(p, horizon=7, min_history=31)
    want = brute(p, horizon=7, min_history=31)
    assert np.allclose(got[0], want[0], equal_nan=True, atol=1e-12)
    assert np.allclose(got[1], want[1], equal_nan=True, atol=1e-12)
    assert np.flatnonzero(np.isfinite(got[0]))[0] == 7 + 31 - 1

    # Future prices beyond the current target window cannot change its rank.
    i, horizon = 700, 7
    changed = p.copy(); changed[i + horizon + 1:] *= np.exp(np.linspace(.2, 1, len(p)-i-horizon-1))
    changed_ranks = severity_ranks(changed, horizon=horizon, min_history=31)
    assert np.allclose(got[0][:i+1], changed_ranks[0][:i+1], equal_nan=True, atol=1e-12)
    assert np.allclose(got[1][:i+1], changed_ranks[1][:i+1], equal_nan=True, atol=1e-12)

    # A missing price inside a target window invalidates that origin.
    gap_price = p[:250].copy(); gap_price[100] = np.nan
    gap_down, gap_up = severity_ranks(gap_price, horizon=5, min_history=10)
    assert np.isnan(gap_down[95:101]).all() and np.isnan(gap_up[95:101]).all()

    inv = severity_ranks(1 / p, horizon=7, min_history=31)
    assert np.allclose(got[0], inv[1], equal_nan=True, atol=1e-12)
    assert np.allclose(got[1], inv[0], equal_nan=True, atol=1e-12)
    print("V6 forward-label tests passed: origin alignment, past-only rank, future-invariance, inverse symmetry")


if __name__ == "__main__":
    main()
