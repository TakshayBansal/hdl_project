"""Acceptance statistics: batched node verification, acceptance vectors P(alpha), and the
depth-dependent matrix P[d, k] rebuilt from decoder visit logs (spec 7.9).

verify_batch runs SEQUOIA's sampling-without-replacement verification (spec 4.3) on m
independent rows at once. It has the same semantics as verify.verify_node, row by row,
but needs no host sync per child, so it can run inside the generation loop on the GPU.
"""
import torch
from jcsd.stats import smooth_tail, depth_matrix   # re-exported


@torch.no_grad()
def verify_batch(pi, q, children, gen, early_exit=False):
    """pi, q: [m, V] target / draft distributions; children: [m, k] long token ids drawn
    sequentially without replacement from q (e.g. verify.gumbel_topk).
    Returns (acc, R): acc[m] = index of the accepted child or -1, R[m, V] = the residual
    distribution after all rejections (the extra token is sampled from it when acc == -1).
    early_exit stops as soon as every row has accepted (one host sync per child)."""
    m, k = children.shape
    dev = pi.device
    R = pi.double().clone()
    D = q.double().clone()
    pi0 = pi.double()
    proposed = torch.zeros(D.shape, dtype=torch.bool, device=dev)
    acc = torch.full((m,), -1, dtype=torch.long, device=dev)
    alive = torch.ones(m, dtype=torch.bool, device=dev)
    for i in range(k):
        s = D.sum(-1, keepdim=True)
        D = torch.where(s <= 0, (~proposed).double(), D)        # uniform over unproposed
        D = D / D.sum(-1, keepdim=True)
        c = children[:, i:i + 1]
        Dc = D.gather(1, c).squeeze(1)
        Rc = R.gather(1, c).squeeze(1)
        ratio = torch.where(Dc > 0, Rc / Dc.clamp_min(1e-300), torch.zeros_like(Dc))
        u = torch.rand(m, generator=gen, device=dev, dtype=torch.double)
        ok = alive & (u < ratio.clamp(max=1.0))
        acc = torch.where(ok, torch.full_like(acc, i), acc)
        alive = alive & ~ok
        Rn = (R - D).clamp_min(0)
        tot = Rn.sum(-1, keepdim=True)
        R = torch.where(tot > 1e-12, Rn / tot.clamp_min(1e-300), pi0)
        proposed.scatter_(1, c, True)
        D = D.scatter(1, c, 0.0)
        if early_exit and not bool(alive.any()):
            break
    return acc, R


class AcceptCounter:
    """Accumulates simulated node verifications with B children on the GPU (no host sync)."""

    def __init__(self, B, device, keep_p1=20000, seed=0):
        self.B = B
        self.counts = torch.zeros(B + 1, dtype=torch.long, device=device)   # B = all rejected
        self.p1 = []
        self.keep_p1 = keep_p1
        self.n_p1 = 0
        self.gen = torch.Generator(device=device).manual_seed(seed)

    @torch.no_grad()
    def add(self, q, pi, sims=4):
        # q, pi: [r, V] for the active rows at one generation step
        if q.shape[0] == 0:
            return
        from jcsd.verify import gumbel_topk
        if self.n_p1 < self.keep_p1:
            self.p1.append(torch.minimum(q, pi).sum(-1).float())       # analytic P(first child)
            self.n_p1 += q.shape[0]
        qq = q.repeat_interleave(sims, 0)
        pp = pi.repeat_interleave(sims, 0)
        ch = gumbel_topk(qq, self.B, self.gen)
        a, _ = verify_batch(pp, qq, ch, self.gen)
        a = torch.where(a < 0, torch.full_like(a, self.B), a)
        self.counts += torch.bincount(a, minlength=self.B + 1)

    def result(self):
        c = self.counts.double().cpu().numpy()
        tot = max(c.sum(), 1.0)
        p1 = torch.cat(self.p1).cpu().numpy().tolist() if self.p1 else []
        return dict(pk=(c[:self.B] / tot).tolist(), all_rejected=float(c[self.B] / tot),
                    counts=c.astype(int).tolist(), p1_samples=p1[:self.keep_p1])
