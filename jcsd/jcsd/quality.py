"""Stage 3 + 4: sample from the cascade target pi_alpha token by token (no trees needed, by the
decoupling property of spec 3.3), score quality, and collect acceptance statistics inline.

Prompts are batched with left padding so a whole calibration split runs in minutes rather
than hours. Every row gets its own explicit position ids, so padding never shifts positions.
"""
import torch
from jcsd.cascade import probs, tokenv3_target
from jcsd.accept import AcceptCounter


class PairStepper:
    """Runs draft and target side by side over a left-padded batch with KV caches."""

    def __init__(self, draft, target, V, pad_id, device, need_target=True):
        self.draft, self.target, self.V = draft, target, V
        self.pad_id, self.dev, self.need_target = pad_id, device, need_target

    @torch.no_grad()
    def prefill(self, prompts):
        b, L = len(prompts), max(len(p) for p in prompts)
        ids = torch.full((b, L), self.pad_id, dtype=torch.long)
        mask = torch.zeros((b, L), dtype=torch.long)
        for i, p in enumerate(prompts):
            ids[i, L - len(p):] = torch.tensor(p, dtype=torch.long)
            mask[i, L - len(p):] = 1
        ids, mask = ids.to(self.dev), mask.to(self.dev)
        pos = (mask.cumsum(-1) - 1).clamp_min(0)
        self.mask, self.pos = mask, pos[:, -1]
        od = self.draft(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
        self.dc = od.past_key_values
        lq = od.logits[:, -1, :self.V].float()
        lp = None
        if self.need_target:
            ot = self.target(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
            self.tc = ot.past_key_values
            lp = ot.logits[:, -1, :self.V].float()
        return lq, lp

    @torch.no_grad()
    def step(self, tok):
        # tok: [b] long, the token just sampled for every row
        b = tok.shape[0]
        self.mask = torch.cat([self.mask, torch.ones((b, 1), dtype=self.mask.dtype, device=self.dev)], 1)
        self.pos = self.pos + 1
        kw = dict(input_ids=tok[:, None], attention_mask=self.mask, position_ids=self.pos[:, None],
                  use_cache=True)
        od = self.draft(past_key_values=self.dc, **kw)
        self.dc = od.past_key_values
        lq = od.logits[:, -1, :self.V].float()
        lp = None
        if self.need_target:
            ot = self.target(past_key_values=self.tc, **kw)
            self.tc = ot.past_key_values
            lp = ot.logits[:, -1, :self.V].float()
        return lq, lp


def target_dist(mode, alpha, lq, lp, T):
    q_s = probs(lq, T)
    if mode == 'q':
        return q_s, q_s
    p_s = probs(lp, T)
    if mode == 'p':
        return q_s, p_s
    return q_s, tokenv3_target(q_s, probs(lp), q_s, p_s, alpha)   # deferral test on RAW p


@torch.no_grad()
def sample_batch(draft, target, V, prompts, cfg, alpha, mode, gen, eos_ids, max_new,
                 pad_id, counter=None, sims=4):
    """prompts: list of token-id lists. mode: 'cascade' (pi_alpha), 'p' (target), 'q' (draft).
    Returns a list of generated token-id lists (each cut after the first EOS).
    If counter (AcceptCounter) is given, every active position also contributes `sims`
    simulated node verifications with B children against the same target pi."""
    st = PairStepper(draft, target, V, pad_id, cfg.device, need_target=(mode != 'q'))
    lq, lp = st.prefill(prompts)
    b = len(prompts)
    eos = torch.tensor(sorted(eos_ids), dtype=torch.long, device=cfg.device)
    done = torch.zeros(b, dtype=torch.bool, device=cfg.device)
    out = torch.zeros((b, max_new), dtype=torch.long, device=cfg.device)
    n_out = 0
    for t in range(max_new):
        q_s, pi = target_dist(mode, alpha, lq, lp, cfg.temperature)
        tok = torch.multinomial(pi, 1, generator=gen).squeeze(1)
        if counter is not None and mode != 'q':
            act = ~done
            counter.add(q_s[act], pi[act], sims)
        out[:, t] = torch.where(done, torch.full_like(tok, -1), tok)
        n_out = t + 1
        done = done | torch.isin(tok, eos)
        if t % 16 == 15 and bool(done.all()):          # occasional host sync only
            break
        if t + 1 < max_new:
            lq, lp = st.step(tok)
    res = []
    for row in out[:, :n_out].tolist():
        r = []
        for x in row:
            if x < 0:
                break
            r.append(x)
            if x in eos_ids:
                break
        res.append(r)
    return res


@torch.no_grad()
def sample_from_target(draft, target, V, prompt_ids, cfg, alpha, gen, eos_id, mode='cascade'):
    """Unbatched reference sampler from the spec (7.8). Kept for tests and debugging."""
    dev = cfg.device
    ids = prompt_ids.to(dev)
    od = draft(input_ids=ids, use_cache=True)
    ot = target(input_ids=ids, use_cache=True)
    lq, dc = od.logits[0, -1, :V].float(), od.past_key_values
    lp, tc = ot.logits[0, -1, :V].float(), ot.past_key_values
    out = []
    for _ in range(cfg.max_new_tokens):
        _, pi = target_dist(mode, alpha, lq, lp, cfg.temperature)
        tok = int(torch.multinomial(pi, 1, generator=gen).item())
        out.append(tok)
        if tok == eos_id:
            break
        t = torch.tensor([[tok]], device=dev)
        od = draft(input_ids=t, past_key_values=dc, use_cache=True)
        ot = target(input_ids=t, past_key_values=tc, use_cache=True)
        lq, dc = od.logits[0, -1, :V].float(), od.past_key_values
        lp, tc = ot.logits[0, -1, :V].float(), ot.past_key_values
    return out


def make_counter(B, device, seed):
    return AcceptCounter(B, device, seed=seed)
