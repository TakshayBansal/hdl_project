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


def make_cost(prof, overhead=0.0):
    fv = lambda n: float(np.interp(n, prof['sizes'], prof['t_verify']))
    fd = lambda w: float(np.interp(w, prof['sizes'], prof['t_draft_level']))

    def cost(widths):
        n = sum(widths)
        draft = sum(fd(w) for w in widths[:-1])   # the deepest level needs no further drafting
        return fv(n) + draft + overhead
    return cost


def best_for_acc(acc, cfg, prof, overhead=0.0, n_values=None):
    # n_values: optional coarse grid of tree sizes to scan (default: every n up to cfg.max_tree)
    N, D, B = cfg.max_tree, cfg.max_depth, cfg.max_children
    dp = build_dp(acc, N, D, B)       # valid for every depth cap because acc is 1-D
    cost = make_cost(prof, overhead)
    ns = range(1, N + 1) if n_values is None else [n for n in n_values if 1 <= n <= N]
    best = dict(speedup=0.0)
    for d in range(1, D + 1):
        for n in ns:
            G = dp['best'][D - d, n]
            if G < 0:
                continue
            tree = get_tree(dp, n, L=D - d)
            w = level_widths(tree)
            sp = G * prof['t_ar'] / cost(w)
            if sp > best['speedup']:
                best = dict(speedup=float(sp), n=n, d=len(w) - 1, G=float(G), widths=w, tree=tree)
    return best


def _feasible(acc_by_alpha, q_by_alpha, q_min):
    alphas = [a for a in acc_by_alpha if q_by_alpha[a] >= q_min]
    if not alphas:
        raise ValueError('no alpha meets q_min=%s (best quality %s)' % (q_min, max(q_by_alpha.values())))
    return alphas


def joint_search(acc_by_alpha, q_by_alpha, q_min, cfg, prof, overhead=0.0, n_values=None):
    res = {}
    for a in _feasible(acc_by_alpha, q_by_alpha, q_min):
        res[a] = best_for_acc(acc_by_alpha[a], cfg, prof, overhead, n_values)
    a_star = max(res, key=lambda a: res[a]['speedup'])
    return a_star, res


def sequential_search(acc_ref, acc_by_alpha, q_by_alpha, q_min, cfg, prof, overhead=0.0, n_values=None):
    # step 1: best tree for the exact target; step 2: tune alpha for that fixed tree
    ref = best_for_acc(acc_ref, cfg, prof, overhead, n_values)
    cost = make_cost(prof, overhead)(ref['widths'])
    out = {}
    for a in _feasible(acc_by_alpha, q_by_alpha, q_min):
        acc = acc_by_alpha[a]
        row = lambda l, k, acc=acc: acc[k - 1] if k <= len(acc) else 0.0
        G = expected_tokens(ref['tree'], row)
        out[a] = G * prof['t_ar'] / cost
    a_seq = max(out, key=out.get)
    return a_seq, out[a_seq], ref
