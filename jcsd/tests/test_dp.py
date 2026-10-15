import numpy as np
from jcsd.dp import build_dp, get_tree, expected_tokens


def test_chain():
    a = 0.8
    dp = build_dp([a], N=6, D=10, B=3)
    assert abs(dp['best'][0, 6] - sum(a ** i for i in range(6))) < 1e-9


def test_paper_vector_small_trees_are_chains():
    # acceptance vector from SEQUOIA Appendix E (Llama3-70B target, Llama3-8B draft, CNN/DM)
    acc = [0.7732, 0.1039, 0.0402, 0.0206, 0.0128, 0.0081, 0.0064, 0.0043]
    dp = build_dp(acc, N=16, D=16, B=8)
    t = get_tree(dp, 8)

    def max_children(node):
        return max([len(node)] + [max_children(c) for c in node])
    assert max_children(t) == 1          # their Figure 5a: the 8-node tree is a chain


def test_recomputed_value_matches_dp():
    acc = np.array([0.6, 0.2, 0.1, 0.05])
    dp = build_dp(acc, N=20, D=8, B=4)
    t = get_tree(dp, 20)
    row = lambda l, k: acc[k - 1] if k <= len(acc) else 0.0
    assert abs(expected_tokens(t, row) - dp['best'][0, 20]) < 1e-9
