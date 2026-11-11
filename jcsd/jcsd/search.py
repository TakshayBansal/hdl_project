import math
import numpy as np
from jcsd.dp import build_dp, get_tree, expected_tokens


def level_widths(tree):
    widths, cur = [1], [tree]
    while True:
        nxt = [c for node in cur for c in node]
        if not nxt:
            break
        widths.append(len(nxt))
        cur = nxt
    return widths


def t_ar_of(prof):
    # the measured AR baseline samples and syncs every token, so use the full-step time if profiled
    return prof.get('t_ar_step', prof['t_ar'])


def make_cost(prof, overhead=0.0, per_node=0.0, level0=False):
    """Seconds per decoding step for a tree with per-level widths `widths` (spec 4.6):
         t_verify(n) + sum_{levels with children} t_draft(width) + overhead + per_node * G
    level0=True prices the draft's level-0 forward at width ceil(G) (the pending chunk of
    tokens accepted last step, spec 9.2 item 2) instead of width 1."""
    fv = lambda n: float(np.interp(n, prof['sizes'], prof['t_verify']))
    fd = lambda w: float(np.interp(w, prof['sizes'], prof['t_draft_level']))

    def cost(widths, G=None):
        n = sum(widths)
        ws = list(widths[:-1])                     # the deepest level needs no further drafting
        if level0 and G is not None and ws:
            ws[0] = max(1, math.ceil(G))
        draft = sum(fd(w) for w in ws)
        extra = per_node * G if (per_node and G is not None) else 0.0
        return fv(n) + draft + overhead + extra
    return cost


def acc_row_fn(acc):
    acc = np.asarray(acc, dtype=float)
    if acc.ndim == 1:
        return lambda l, k: acc[k - 1] if k <= len(acc) else 0.0
    return lambda l, k: acc[min(l, acc.shape[0] - 1), k] if k < acc.shape[1] else 0.0


def evaluate_tree(tree, acc, prof, **cost_kw):
    """Predicted G and speedup of a fixed tree under acceptance acc (1-D vector or 2-D matrix)."""
    G = expected_tokens(tree, acc_row_fn(acc))
    w = level_widths(tree)
    c = make_cost(prof, **cost_kw)(w, G)
    return dict(G=float(G), speedup=float(G * t_ar_of(prof) / c), widths=w, cost=float(c))


def _dp_tables(acc, N, D, B):
    """Yield (d, best_G_by_n, tree_fn) for every depth cap d = 1..D."""
    acc = np.asarray(acc, dtype=float)
    if acc.ndim == 1:
        dp = build_dp(acc, N, D, B)       # one run is valid for every depth cap (spec 7.5)
        for d in range(1, D + 1):
            yield d, dp['best'][D - d], (lambda n, d=d: get_tree(dp, n, L=D - d))
    else:
        for d in range(1, D + 1):         # a depth-dependent matrix needs one run per cap
            dp = build_dp(acc[:d], N, d, B)
            yield d, dp['best'][0], (lambda n, dp=dp: get_tree(dp, n, L=0))


def best_for_acc(acc, cfg, prof, overhead=0.0, n_values=None, per_node=0.0, level0=False,
                 surface=False):
    # n_values: optional coarse grid of tree sizes to scan (default: every n up to cfg.max_tree)
    N, D, B = cfg.max_tree, cfg.max_depth, cfg.max_children
    acc = np.asarray(acc, dtype=float)
    if acc.ndim == 2:
        acc = acc[:D, :B + 1]
        if acc.shape[1] < B + 1:
            acc = np.pad(acc, ((0, 0), (0, B + 1 - acc.shape[1])))
        if acc.shape[0] < D:
            acc = np.vstack([acc, np.tile(acc[-1:], (D - acc.shape[0], 1))])
    cost = make_cost(prof, overhead, per_node, level0)
    t_ar = t_ar_of(prof)
    ns = list(range(1, N + 1)) if n_values is None else [n for n in n_values if 1 <= n <= N]
    best = dict(speedup=0.0)
    surf = {}
    for d, bestG, tree_fn in _dp_tables(acc, N, D, B):
        for n in ns:
            G = bestG[n]
            if G < 0:
                continue
            tree = tree_fn(n)
            w = level_widths(tree)
            sp = G * t_ar / cost(w, G)
            if surface:
                surf['%d,%d' % (d, n)] = float(sp)
            if sp > best['speedup']:
                best = dict(speedup=float(sp), n=n, d=len(w) - 1, G=float(G), widths=w, tree=tree)
    if surface:
        best['surface'] = surf
    return best


def _feasible(acc_by_alpha, q_by_alpha, q_min):
    alphas = [a for a in acc_by_alpha if q_by_alpha[a] >= q_min]
    if not alphas:
        raise ValueError('no alpha meets q_min=%s (best quality %s)' % (q_min, max(q_by_alpha.values())))
    return alphas


def joint_search(acc_by_alpha, q_by_alpha, q_min, cfg, prof, overhead=0.0, n_values=None, **kw):
    res = {}
    for a in _feasible(acc_by_alpha, q_by_alpha, q_min):
        res[a] = best_for_acc(acc_by_alpha[a], cfg, prof, overhead, n_values, **kw)
    a_star = max(res, key=lambda a: res[a]['speedup'])
    return a_star, res


def sequential_search(acc_ref, acc_by_alpha, q_by_alpha, q_min, cfg, prof, overhead=0.0,
                      n_values=None, **kw):
    # step 1: best tree for the exact target; step 2: tune alpha for that fixed tree
    ref = best_for_acc(acc_ref, cfg, prof, overhead, n_values, **kw)
    out = {}
    for a in _feasible(acc_by_alpha, q_by_alpha, q_min):
        out[a] = evaluate_tree(ref['tree'], acc_by_alpha[a], prof, overhead=overhead, **kw)['speedup']
    a_seq = max(out, key=out.get)
    return a_seq, out[a_seq], ref


def fit_overhead(rows):
    """Calibrate the step-overhead model from end-to-end runs (spec 9.2 item 1).
    rows: dicts with measured 'step_time' (s), 'pred_forward' (model cost without overhead, s)
    and 'G' (mean new tokens per step). Least squares fit of
        step_time - pred_forward = c0 + c1 * G,   with c0, c1 >= 0.
    Returns (c0, c1)."""
    y = np.array([r['step_time'] - r['pred_forward'] for r in rows])
    X = np.array([[1.0, r['G']] for r in rows])
    if len(rows) < 3:
        return max(float(np.median(y)), 0.0), 0.0
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    c0, c1 = float(c[0]), float(c[1])
    if c1 < 0:
        c0, c1 = max(float(np.mean(y)), 0.0), 0.0
    elif c0 < 0:
        c0, c1 = 0.0, max(float((X[:, 1] @ y) / (X[:, 1] @ X[:, 1])), 0.0)
    return c0, c1
