import torch
from jcsd.cascade import tokenv3_target


def test_worked_example():
    q = torch.tensor([0.6, 0.3, 0.1])
    p = torch.tensor([0.4, 0.5, 0.1])
    pi = tokenv3_target(q, p, q, p, 0.3)
    assert torch.allclose(pi, torch.tensor([0.64, 0.35, 0.01]), atol=1e-6)


def test_alpha_one_is_draft():
    q = torch.softmax(torch.randn(50), -1)
    p = torch.softmax(torch.randn(50), -1)
    assert torch.allclose(tokenv3_target(q, p, q, p, 1.0), q, atol=1e-6)


def test_sums_to_one():
    for a in (0.0, 0.1, 0.5, 0.9, 1.0):
        q = torch.softmax(torch.randn(4, 50), -1)
        p = torch.softmax(torch.randn(4, 50), -1)
        pi = tokenv3_target(q, p, q, p, a)
        assert torch.allclose(pi.sum(-1), torch.ones(4), atol=1e-5)


def test_batched_matches_single():
    q = torch.softmax(torch.randn(3, 20), -1)
    p = torch.softmax(torch.randn(3, 20), -1)
    b = tokenv3_target(q, p, q, p, 0.3)
    for i in range(3):
        assert torch.allclose(b[i], tokenv3_target(q[i], p[i], q[i], p[i], 0.3), atol=1e-6)
