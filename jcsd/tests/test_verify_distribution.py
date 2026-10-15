import torch, numpy as np, pytest
from scipy.stats import chisquare
from jcsd.verify import draft_children, verify_node, gumbel_topk
from jcsd.cascade import tokenv3_target


def run(k, alpha, q, p, n=30000, seed=0):
    g = torch.Generator().manual_seed(seed)
    pi = tokenv3_target(q, p, q, p, alpha).double()
    counts = np.zeros(len(q))
    for _ in range(n):
        ch = draft_children(q, k, g)
        _, tok = verify_node(pi, q, ch, g)
        counts[tok] += 1
    exp = pi.numpy() * counts.sum()
    mask = exp > 5          # chi-square needs expected counts that are not tiny
    cs = counts[mask]; ex = exp[mask]
    ex = ex * cs.sum() / ex.sum()
    return chisquare(cs, ex).pvalue


def rand_dist(V, g, scale=1.5):
    return torch.softmax(torch.randn(V, generator=g) * scale, -1)


@pytest.mark.parametrize('k', [1, 2, 3, 5])
@pytest.mark.parametrize('alpha', [0.0, 0.3, 0.8, 1.0])
def test_output_matches_target(k, alpha):
    g = torch.Generator().manual_seed(1)
    q, p = rand_dist(8, g), rand_dist(8, g)
    assert run(k, alpha, q, p) > 1e-3


def test_zero_support_draft():
    q = torch.tensor([0.5, 0.5, 0, 0, 0, 0, 0, 0])
    g = torch.Generator().manual_seed(2)
    p = rand_dist(8, g)
    assert run(4, 0.2, q, p) > 1e-3        # k=4 exceeds the draft support


def test_alpha_one_first_child_always_accepted():
    g = torch.Generator().manual_seed(3)
    q, p = rand_dist(8, g), rand_dist(8, g)
    pi = tokenv3_target(q, p, q, p, 1.0).double()
    for _ in range(500):
        ch = draft_children(q, 3, g)
        i, _ = verify_node(pi, q, ch, g)
        assert i == 0


def test_gumbel_topk_matches_loop_sampler():
    g = torch.Generator().manual_seed(4)
    q = rand_dist(6, g)
    n = 50000
    qm = q[None].expand(n, -1).contiguous()
    top = gumbel_topk(qm, 2, g)
    first = np.bincount(top[:, 0].numpy(), minlength=6)
    exp1 = q.double().numpy()
    exp1 = exp1 / exp1.sum() * first.sum()
    assert chisquare(first, exp1).pvalue > 1e-3
    pair = np.zeros((6, 6))
    for a, b in top.numpy():
        pair[a, b] += 1
    ref = np.zeros((6, 6))
    for a in range(6):
        for b in range(6):
            if a != b:
                ref[a, b] = n * q[a].item() * q[b].item() / (1 - q[a].item())
    m = ref > 5
    r = ref[m] * pair[m].sum() / ref[m].sum()
    assert chisquare(pair[m], r).pvalue > 1e-3


def test_gumbel_topk_zero_support_fills_uniformly():
    g = torch.Generator().manual_seed(5)
    q = torch.tensor([0.5, 0.5, 0, 0, 0, 0])
    qm = q[None].expand(20000, -1).contiguous()
    top = gumbel_topk(qm, 4, g)
    assert set(top[:, :2].flatten().tolist()) == {0, 1}      # support comes first
    third = np.bincount(top[:, 2].numpy(), minlength=6)[2:]
    assert chisquare(third).pvalue > 1e-3                    # the rest in uniform order
