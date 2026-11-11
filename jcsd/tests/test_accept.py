import numpy as np
import pytest
import torch
from scipy.stats import chisquare
from jcsd.accept import verify_batch, AcceptCounter
from jcsd.verify import draft_children, verify_node, gumbel_topk
from jcsd.cascade import tokenv3_target


def rand_dist(V, g, scale=1.5):
    return torch.softmax(torch.randn(V, generator=g) * scale, -1)


def _chi(counts, probs):
    exp = probs * counts.sum()
    m = exp > 5
    return chisquare(counts[m], exp[m] * counts[m].sum() / exp[m].sum()).pvalue


@pytest.mark.parametrize('k,alpha', [(1, 0.0), (3, 0.3), (5, 0.0), (3, 0.8)])
def test_verify_batch_output_is_target(k, alpha):
    g = torch.Generator().manual_seed(10 + k)
    q, p = rand_dist(8, g), rand_dist(8, g)
    pi = tokenv3_target(q, p, q, p, alpha)
    n = 30000
    qm, pm = q[None].expand(n, -1).contiguous(), pi[None].expand(n, -1).contiguous()
    ch = gumbel_topk(qm, k, g)
    acc, R = verify_batch(pm, qm, ch, g)
    tok = torch.where(acc >= 0, ch.gather(1, acc.clamp_min(0)[:, None]).squeeze(1),
                      torch.multinomial(R.float(), 1, generator=g).squeeze(1))
    counts = np.bincount(tok.numpy(), minlength=8).astype(float)
    assert _chi(counts, pi.double().numpy()) > 1e-3


def test_verify_batch_zero_support_draft():
    g = torch.Generator().manual_seed(2)
    q = torch.tensor([0.5, 0.5, 0, 0, 0, 0, 0, 0])
    p = rand_dist(8, g)
    pi = tokenv3_target(q, p, q, p, 0.2)
    n = 30000
    qm, pm = q[None].expand(n, -1).contiguous(), pi[None].expand(n, -1).contiguous()
    ch = gumbel_topk(qm, 4, g)
    acc, R = verify_batch(pm, qm, ch, g)
    tok = torch.where(acc >= 0, ch.gather(1, acc.clamp_min(0)[:, None]).squeeze(1),
                      torch.multinomial(R.float(), 1, generator=g).squeeze(1))
    assert _chi(np.bincount(tok.numpy(), minlength=8).astype(float), pi.double().numpy()) > 1e-3


def test_accept_index_matches_loop_verifier():
    g = torch.Generator().manual_seed(3)
    q, p = rand_dist(8, g), rand_dist(8, g)
    pi = tokenv3_target(q, p, q, p, 0.3).double()
    k, n = 4, 20000
    loop = np.zeros(k + 1)
    for _ in range(n):
        i, _t = verify_node(pi, q, draft_children(q, k, g), g)
        loop[i if i >= 0 else k] += 1
    qm, pm = q[None].expand(n, -1).contiguous(), pi[None].expand(n, -1).contiguous()
    acc, _ = verify_batch(pm, qm, gumbel_topk(qm, k, g), g)
    bat = np.bincount(np.where(acc.numpy() < 0, k, acc.numpy()), minlength=k + 1).astype(float)
    m = (loop > 5) & (bat > 5)
    assert chisquare(bat[m], loop[m] * bat[m].sum() / loop[m].sum()).pvalue > 1e-3


def test_early_exit_same_result_as_full():
    g1 = torch.Generator().manual_seed(4)
    g2 = torch.Generator().manual_seed(4)
    q = rand_dist(8, torch.Generator().manual_seed(5))
    ch = torch.tensor([[3, 1, 0]])
    a1, _ = verify_batch(q[None], q[None], ch, g1, early_exit=True)
    a2, _ = verify_batch(q[None], q[None], ch, g2)
    assert a1.item() == a2.item() == 0            # pi == q: first child always accepted


def test_counter_alpha_one_and_analytic_p1():
    g = torch.Generator().manual_seed(6)
    q = torch.softmax(torch.randn(50, 16, generator=g) * 2, -1)
    p = torch.softmax(torch.randn(50, 16, generator=g) * 2, -1)
    c = AcceptCounter(4, 'cpu', seed=0)
    c.add(q, tokenv3_target(q, p, q, p, 1.0), sims=4)
    r = c.result()
    assert r['pk'][0] == 1.0 and r['all_rejected'] == 0.0
    c = AcceptCounter(4, 'cpu', seed=0)
    pi = tokenv3_target(q, p, q, p, 0.3)
    for _ in range(40):
        c.add(q, pi, sims=4)
    r = c.result()
    # simulated P(first child accepted) agrees with the analytic 1 - TV(pi, q)
    assert abs(r['pk'][0] - np.mean(r['p1_samples'])) < 0.02
    assert abs(sum(r['pk']) + r['all_rejected'] - 1) < 1e-9
