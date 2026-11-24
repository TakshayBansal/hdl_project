"""Token-tree utilities (spec 8.2, 8.3): breadth-first flattening, tree attention masks,
KV-cache surgery, and the numerical check that the model honours a custom 4D mask."""
import torch


def flatten(tree):
    """Nested-list tree -> (parent, depth, rank, kids) in breadth-first order.
    Node 0 is the root; rank is the 1-based child index under its parent (0 for the root)."""
    par, dep, rank, kids = [-1], [0], [0], [[]]
    order, i = [(tree, 0)], 0
    while i < len(order):
        node, me = order[i]
        i += 1
        for k, ch in enumerate(node, start=1):
            par.append(me); dep.append(dep[me] + 1); rank.append(k); kids.append([])
            kids[me].append(len(par) - 1)
            order.append((ch, len(par) - 1))
    return par, dep, rank, kids


def ancestor_allow(par):
    """Bool [n, n]: allow[i, j] is True iff j is i or an ancestor of i."""
    n = len(par)
    allow = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        j = i
        while j != -1:
            allow[i, j] = True
            j = par[j]
    return allow


def additive_mask(allow, ctx, dtype):
    """[m, k] bool block -> [1, 1, m, ctx + k] additive float mask with the full prefix visible."""
    m = allow.shape[0]
    full = torch.cat([torch.ones(m, ctx, dtype=torch.bool, device=allow.device), allow], dim=1)
    out = torch.zeros(full.shape, dtype=dtype, device=allow.device)
    out.masked_fill_(~full, torch.finfo(dtype).min)
    return out[None, None]


def _layers(cache):
    """Yield (get, set) access to every layer's key/value tensors across transformers cache APIs."""
    if hasattr(cache, 'layers'):
        for lay in cache.layers:
            yield lay, 'layer'
    elif hasattr(cache, 'key_cache'):
        for i in range(len(cache.key_cache)):
            yield i, 'lists'
    else:
        raise TypeError('unsupported cache type %s; pin transformers to a version with '
                        'DynamicCache.layers or key_cache/value_cache' % type(cache))


def cache_len(cache):
    return cache.get_seq_length()


def crop_to(cache, length):
    """Truncate `cache` to its first `length` positions. A negative argument (remove that many
    tokens) means the same on the old DynamicCache.crop(max_length) and on transformers >= 5,
    which rejects a positive one; crop(0) would empty the old cache, so skip a no-op."""
    extra = cache.get_seq_length() - length
    if extra > 0:
        cache.crop(-extra)


def keep_path(cache, ctx, path):
    """After a verify pass appended n entries after `ctx` cached ones, keep only the nodes in
    `path` (root first, tree indices) and move them to consecutive slots ctx, ctx+1, ...
    Valid because a node at depth j has position ctx + j and lands in slot ctx + j."""
    sel = None
    for h, kind in _layers(cache):
        if kind == 'layer':
            k, v = h.keys, h.values
        else:
            k, v = cache.key_cache[h], cache.value_cache[h]
        if sel is None:
            sel = torch.tensor([ctx + j for j in path], device=k.device)
        k2 = torch.cat([k[:, :, :ctx], k.index_select(2, sel)], dim=2)
        v2 = torch.cat([v[:, :, :ctx], v.index_select(2, sel)], dim=2)
        if kind == 'layer':
            h.keys, h.values = k2, v2
        else:
            cache.key_cache[h], cache.value_cache[h] = k2, v2
    if hasattr(cache, '_seen_tokens'):
        cache._seen_tokens = ctx + len(path)


@torch.no_grad()
def check_tree_attention(model, tree, ctx_len, V, device, tol=2e-2, seed=0, vocab_hi=None):
    """Spec 8.3. Feed a tree through `model` with a 4D mask and explicit position ids, and compare
    each node's logits with a plain causal forward over prefix + its root-to-node path.
    Returns (ok, worst_abs_diff, baseline_noise) where baseline_noise is the difference between
    two plain forwards of different batch shapes (calibrates what fp16 rounding alone does)."""
    g = torch.Generator().manual_seed(seed)
    hi = vocab_hi or min(V, 1000)
    lo = 100 if hi > 200 else 0
    par, dep, rank, kids = flatten(tree)
    n = len(par)
    prefix = torch.randint(lo, hi, (1, ctx_len), generator=g).to(device)
    node_tok = torch.randint(lo, hi, (n,), generator=g).to(device)
    cache = model(input_ids=prefix[:, :-1], use_cache=True).past_key_values
    dt = next(model.parameters()).dtype
    mask = additive_mask(ancestor_allow(par).to(device), ctx_len - 1, dt)
    inp = torch.cat([prefix[0, -1:], node_tok[1:]])[None]
    pos = torch.tensor([[ctx_len - 1 + d for d in dep]], device=device)
    out = model(input_ids=inp, position_ids=pos, attention_mask=mask,
                past_key_values=cache, use_cache=True).logits[0, :, :V].float()
    worst = 0.0
    for i in range(n):
        path, j = [], i
        while j > 0:
            path.append(int(node_tok[j])); j = par[j]
        seq = prefix
        if path:
            seq = torch.cat([prefix, torch.tensor([path[::-1]], device=device, dtype=prefix.dtype)], 1)
        ref = model(input_ids=seq).logits[0, -1, :V].float()
        worst = max(worst, (out[i] - ref).abs().max().item())
    # rounding baseline: same last-token logits from a full pass vs an incremental pass
    a = model(input_ids=prefix).logits[0, -1, :V].float()
    c2 = model(input_ids=prefix[:, :-1], use_cache=True).past_key_values
    b = model(input_ids=prefix[:, -1:], past_key_values=c2, use_cache=True).logits[0, -1, :V].float()
    noise = (a - b).abs().max().item()
    return worst < max(tol, 3 * noise), worst, noise
