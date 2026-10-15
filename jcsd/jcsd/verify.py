import torch


def _next_D(D, proposed, q):
    if D.sum() <= 0:
        D = torch.ones_like(q)
        if proposed:
            D[proposed] = 0
    return D / D.sum()


def draft_children(q, k, gen):
    # k distinct tokens sampled sequentially without replacement from q
    D = q.clone().double()
    out = []
    for _ in range(k):
        D = _next_D(D, out, q.double())
        s = int(torch.multinomial(D, 1, generator=gen).item())
        out.append(s)
        D[s] = 0
    return out


def verify_node(pi, q, children, gen):
    # returns (index of accepted child or -1, token produced at this node)
    R = pi.clone().double()
    D = q.clone().double()
    proposed = []
    for i, s in enumerate(children):
        D = _next_D(D, proposed, q.double())
        u = torch.rand(1, generator=gen, device=R.device, dtype=torch.double).item()
        ratio = (R[s] / D[s]).item() if D[s] > 0 else 0.0
        if u < min(1.0, ratio):
            return i, s
        R = (R - D).clamp_min(0)
        tot = R.sum()
        R = R / tot if tot > 1e-12 else pi.clone().double()
        proposed.append(s)
        D[s] = 0
    tok = int(torch.multinomial(R, 1, generator=gen).item())
    return -1, tok


def gumbel_topk(qmat, k, gen):
    # qmat: [m, V]. Returns [m, k] token ids in sequential-without-replacement order.
    u = torch.rand(qmat.shape, generator=gen, device=qmat.device).clamp_min(1e-20)
    g = -torch.log(-torch.log(u))
    key = torch.where(qmat > 0, torch.log(qmat.clamp_min(1e-30)) + g, g - 1e4)
    return key.topk(k, dim=-1).indices
