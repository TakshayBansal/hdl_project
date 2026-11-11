"""Numpy-only acceptance statistics helpers (no torch import, so they test anywhere)."""
import numpy as np


def smooth_tail(pk, start=4):
    """Optional power-law fit of the tail p_k, k >= start (spec 7.9). Returns a new list; the
    head is kept as measured and the total mass of the tail is preserved."""
    pk = np.asarray(pk, dtype=float)
    k = np.arange(1, len(pk) + 1)
    m = (k >= start) & (pk > 0)
    if m.sum() < 3:
        return pk.tolist()
    b, a = np.polyfit(np.log(k[m]), np.log(pk[m]), 1)
    fit = np.exp(a) * k ** b
    out = pk.copy()
    tail = k >= start
    if fit[tail].sum() > 0:
        out[tail] = fit[tail] * pk[tail].sum() / fit[tail].sum()
    return out.tolist()


def depth_matrix(visit_counts, D, B, fallback):
    """Rebuild P[d, k] from decoder visit logs (spec 8.4).
    visit_counts: dict keyed 'd,nk,i' -> count, where d = depth of the visited node,
    nk = its number of children, i = accepted child index (-1 = all rejected).
    p_{d,k} = (# visits at depth d whose child k was accepted) / (# visits at depth d with
    at least k children). Ranks never present in the tree fall back to the 1-D vector.
    Returns an array of shape (D, B + 1) with column 0 unused (dp.build_dp format)."""
    fb = np.zeros(B + 1)
    m = min(B, len(fallback))
    fb[1:m + 1] = np.asarray(fallback, dtype=float)[:m]
    P = np.tile(fb, (D, 1))
    acc = np.zeros((D, B + 1))
    reach = np.zeros((D, B + 1))
    for key, c in visit_counts.items():
        d, nk, i = (int(x) for x in key.split(','))
        if d >= D:
            continue
        reach[d, 1:min(nk, B) + 1] += c
        if 0 <= i < B:
            acc[d, i + 1] += c
    ok = reach >= 30                       # do not trust a rank seen fewer than 30 times
    P[ok] = acc[ok] / reach[ok]
    P[:, 0] = 0
    return P, reach
