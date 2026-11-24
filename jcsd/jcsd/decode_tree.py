"""End-to-end tree decoder (spec 8.4) and the autoregressive baseline (spec 8.5).

Invariants (keep these true after every step):
  * Target cache holds every committed token except the last one (the root). The verify pass
    feeds the root plus the n-1 drafted nodes, with position ids ctx + depth.
  * Draft cache holds committed tokens up to d_len. Tokens committed since then are fed as one
    chunk at the start of the next step; tree-level forwards are cropped away afterwards.
  * keep_path keeps the root and accepted nodes of the target cache after every step.
  * Deferral test on RAW target probabilities; blend on temperature-scaled ones (float32).
  * Cascade mode only: if the walk ends at a deepest-level leaf, the accepted path is fed to the
    draft right away (q at that leaf is needed for pi_alpha); d_len advances accordingly.
"""
import time
import torch
from jcsd.cascade import probs, tokenv3_target
from jcsd.verify import gumbel_topk
from jcsd.accept import verify_batch
from jcsd.tree import flatten, ancestor_allow, additive_mask, keep_path, crop_to
from jcsd.dp import chain  # noqa: F401  (re-exported)


def _sync(dev):
    if str(dev).startswith('cuda'):
        torch.cuda.synchronize()


def _truncate(new_tokens, eos_ids):
    for j, t in enumerate(new_tokens):
        if t in eos_ids:
            return new_tokens[:j + 1], True
    return new_tokens, False


class TreeDecoder:
    def __init__(self, draft, target, V, cfg, alpha, tree, mode='cascade'):
        assert mode in ('cascade', 'p')
        self.draft, self.target, self.V, self.cfg = draft, target, V, cfg
        self.alpha, self.mode = alpha, mode
        self.par, self.dep, self.rank, self.kids = flatten(tree)
        self.n, dev = len(self.par), cfg.device
        self.depth = max(self.dep)
        self.tdt = next(target.parameters()).dtype
        self.ddt = next(draft.parameters()).dtype
        self.use_draft = not (mode == 'p' and self.n == 1)
        self.levels = {}
        for i, d in enumerate(self.dep):
            self.levels.setdefault(d, []).append(i)
        self.lv_idx = {l: torch.tensor(v, device=dev) for l, v in self.levels.items()}
        self.allow = ancestor_allow(self.par).to(dev)
        self.dep_t = torch.tensor([self.dep], device=dev)
        self.kid_t = [torch.tensor(k, device=dev)[None] if k else None for k in self.kids]
        self.lv_allow, self.assign = {}, {}
        for l in range(0, self.depth):
            idx = self.levels[l]
            rows, ranks, childs = [], [], []
            for r, i in enumerate(idx):
                for k, c in enumerate(self.kids[i]):
                    rows.append(r); ranks.append(k); childs.append(c)
            kmax = max(len(self.kids[i]) for i in idx)
            self.assign[l] = (torch.tensor(rows, device=dev), torch.tensor(ranks, device=dev),
                              torch.tensor(childs, device=dev), kmax)
            if l >= 1:
                # draft cache during level l holds d_len committed + nodes 1..lo-1 (levels 1..l-1)
                lo, m = idx[0], len(idx)
                a = torch.zeros(m, lo - 1 + m, dtype=torch.bool)
                for r, i in enumerate(idx):
                    j = i
                    while j > 0:
                        a[r, j - 1] = True
                        j = self.par[j]
                self.lv_allow[l] = a.to(dev)

    @torch.no_grad()
    def _draft_tree(self, last_q_logits, d_len, dcache, gen, committed_last):
        dev, T, V, n = self.cfg.device, self.cfg.temperature, self.V, self.n
        qs = torch.zeros(n, V, device=dev)
        toks = torch.zeros(n, dtype=torch.long, device=dev)
        toks[0] = committed_last
        if not self.use_draft:
            return qs, toks
        qs[0] = probs(last_q_logits, T)
        for l in range(0, self.depth):
            idx = self.lv_idx[l]
            if l >= 1:
                pos = torch.full((1, len(idx)), d_len - 1 + l, device=dev, dtype=torch.long)
                mask = additive_mask(self.lv_allow[l], d_len, self.ddt)
                o = self.draft(input_ids=toks[idx][None], position_ids=pos, attention_mask=mask,
                               past_key_values=dcache, use_cache=True)
                qs[idx] = probs(o.logits[0, :, :V], T)
            rows, ranks, childs, kmax = self.assign[l]
            top = gumbel_topk(qs[idx], kmax, gen)
            toks[childs] = top[rows, ranks]
        if self.depth >= 2:
            crop_to(dcache, d_len)
        return qs, toks

    @torch.no_grad()
    def generate(self, prompt_ids, gen, eos_ids, max_new, debug=False):
        """prompt_ids: list of ints. Returns (new_tokens, log). Times exclude the prefill.
        debug=True asserts the cache invariants every step and returns the caches in the log."""
        dev, T, V = self.cfg.device, self.cfg.temperature, self.V
        eos_ids = set(eos_ids)
        committed = list(prompt_ids)
        plen = len(committed)
        ids = torch.tensor([committed], device=dev)
        _sync(dev); t_pre = time.perf_counter()
        tcache = self.target(input_ids=ids[:, :-1], use_cache=True).past_key_values
        dcache, d_len, last_q = None, 0, None
        if self.use_draft:
            o = self.draft(input_ids=ids, use_cache=True)
            dcache, d_len, last_q = o.past_key_values, len(committed), o.logits[0, -1, :V]
        _sync(dev); t0 = time.perf_counter()
        log = dict(steps=0, new_per_step=[], visits={}, path_len=[])
        finished = False
        while len(committed) - plen < max_new and not finished:
            # ---- draft: pending chunk, then one forward per tree level ----
            if self.use_draft and d_len < len(committed):
                pend = torch.tensor([committed[d_len:]], device=dev)
                o = self.draft(input_ids=pend, past_key_values=dcache, use_cache=True)
                dcache, d_len, last_q = o.past_key_values, len(committed), o.logits[0, -1, :V]
            qs, toks = self._draft_tree(last_q, d_len, dcache, gen, committed[-1])
            # ---- verify: one target pass over the whole tree ----
            ctx_t = len(committed) - 1
            mask = additive_mask(self.allow, ctx_t, self.tdt)
            o = self.target(input_ids=toks[None], position_ids=self.dep_t + ctx_t,
                            attention_mask=mask, past_key_values=tcache, use_cache=True)
            tcache = o.past_key_values
            lp = o.logits[0, :, :V].float()
            p_s = probs(lp, T)
            PI = p_s if self.mode == 'p' else tokenv3_target(qs, probs(lp), qs, p_s, self.alpha)
            tl = toks.tolist()
            node, path, new = 0, [0], []
            while True:
                kid = self.kids[node]
                if kid:
                    acc, R = verify_batch(PI[node:node + 1], qs[node:node + 1], toks[self.kid_t[node]],
                                          gen, early_exit=True)
                    i = int(acc.item())
                    key = '%d,%d,%d' % (self.dep[node], len(kid), i)
                    log['visits'][key] = log['visits'].get(key, 0) + 1
                    if i >= 0:
                        node = kid[i]
                        path.append(node)
                        new.append(tl[node])
                        continue
                    tok = int(torch.multinomial(R[0], 1, generator=gen).item())
                else:
                    pi_leaf = PI[node]
                    if self.mode == 'cascade' and self.depth >= 1 and self.dep[node] == self.depth:
                        # deepest leaves were never fed to the draft, so q (and pi_alpha) is unknown
                        # there. Feed the accepted path now: these tokens are committed anyway and
                        # would otherwise be the start of next step's pending chunk.
                        o = self.draft(input_ids=toks[torch.tensor(path[1:], device=dev)][None],
                                       past_key_values=dcache, use_cache=True)
                        dcache, d_len = o.past_key_values, d_len + len(path) - 1
                        q_leaf = probs(o.logits[0, -1, :V], T)
                        pi_leaf = tokenv3_target(q_leaf, probs(lp[node]), q_leaf, p_s[node], self.alpha)
                    tok = int(torch.multinomial(pi_leaf, 1, generator=gen).item())
                new.append(tok)
                break
            keep_path(tcache, ctx_t, path)
            new, finished = _truncate(new, eos_ids)
            committed += new
            log['steps'] += 1
            log['new_per_step'].append(len(new))
            log['path_len'].append(len(path))
            if debug and not finished:
                assert tcache.get_seq_length() == len(committed) - 1, 'target cache length'
                if self.use_draft:
                    assert dcache.get_seq_length() == d_len, 'draft cache length'
        _sync(dev); t1 = time.perf_counter()
        if debug:
            log.update(_tcache=tcache, _committed=committed)
        out = committed[plen:plen + max_new]
        log.update(time_prefill=t0 - t_pre, time_decode=t1 - t0, n_new=len(out))
        return out, log


@torch.no_grad()
def autoregressive(target, V, prompt_ids, cfg, gen, eos_ids, max_new):
    """Plain sampling from the target, one token per forward. Same log format as TreeDecoder."""
    dev, T = cfg.device, cfg.temperature
    eos_ids = set(eos_ids)
    _sync(dev); t_pre = time.perf_counter()
    out = target(input_ids=torch.tensor([list(prompt_ids)], device=dev), use_cache=True)
    cache, logits, res = out.past_key_values, out.logits[0, -1, :V], []
    _sync(dev); t0 = time.perf_counter()
    for _ in range(max_new):
        tok = int(torch.multinomial(probs(logits, T), 1, generator=gen).item())
        res.append(tok)
        if tok in eos_ids or len(res) == max_new:
            break
        o = target(input_ids=torch.tensor([[tok]], device=dev), past_key_values=cache, use_cache=True)
        cache, logits = o.past_key_values, o.logits[0, -1, :V]
    _sync(dev); t1 = time.perf_counter()
    return res, dict(steps=len(res), new_per_step=[1] * len(res), visits={}, path_len=[1] * len(res),
                     time_prefill=t0 - t_pre, time_decode=t1 - t0, n_new=len(res))
