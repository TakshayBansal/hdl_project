import numpy as np

NEG = -1e18


def build_dp(acc, N, D, B):
    # acc: 1-D vector p_1..p_K, or 2-D array shape (D, B+1) with column 0 unused
    acc = np.asarray(acc, dtype=float)
    if acc.ndim == 1:
        row = np.zeros(B + 1)
        m = min(B, len(acc))
        row[1:m + 1] = acc[:m]
        acc = np.tile(row, (D, 1))
    assert acc.shape == (D, B + 1)
    W = np.full((D + 1, N + 1, B + 1), NEG)
    best = np.full((D + 1, N + 1), NEG)
    arg_b = np.zeros((D + 1, N + 1), dtype=int)
    arg_m = np.zeros((D + 1, N + 1, B + 1), dtype=int)
    for L in range(D, -1, -1):
        W[L, 1, 0] = 1.0
        best[L, 1] = 1.0
        if L == D:
            continue
        for n in range(2, N + 1):
            ms = np.arange(1, n)                 # size of the last child's subtree
            sub = best[L + 1, ms]
            for b in range(1, B + 1):
                left = W[L, n - ms, b - 1]
                cand = np.where((sub > NEG / 2) & (left > NEG / 2),
                                left + acc[L, b] * sub, NEG)
                j = int(np.argmax(cand))
                W[L, n, b] = cand[j]
                arg_m[L, n, b] = ms[j]
            bb = int(np.argmax(W[L, n, :]))
            best[L, n] = W[L, n, bb]
            arg_b[L, n] = bb
    return dict(W=W, best=best, arg_b=arg_b, arg_m=arg_m, D=D, B=B, N=N)


def get_tree(dp, n, L=0):
    # returns nested lists: a node is the list of its children; a leaf is []
    def best_tree(L, n):
        return with_b(L, n, int(dp['arg_b'][L, n]))

    def with_b(L, n, b):
        kids, rem = [], n
        for bb in range(b, 0, -1):
            m = int(dp['arg_m'][L, rem, bb])
            kids.append(best_tree(L + 1, m))
            rem -= m
        kids.reverse()
        return kids
    return best_tree(L, n)


def expected_tokens(tree, acc_row_fn, L=0):
    # recompute G for a tree under acceptance acc_row_fn(L, k); used as an independent check
    tot, stack = 0.0, [(tree, 1.0, L)]
    while stack:
        node, f, l = stack.pop()
        tot += f
        for k, ch in enumerate(node, start=1):
            stack.append((ch, f * acc_row_fn(l, k), l + 1))
    return tot
