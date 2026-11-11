# Tree attention, KV-cache surgery and the end-to-end decoder (spec 8.3, 8.6) on tiny CPU models.
# If these fail, nothing measured by run_e2e.py can be trusted.
import numpy as np
import pytest
import torch
from scipy.stats import chisquare
pytest.importorskip('transformers')
from _tiny import tiny, pair, C, V
from jcsd.tree import flatten, ancestor_allow, keep_path, check_tree_attention, additive_mask
from jcsd.decode_tree import TreeDecoder, autoregressive, chain
from jcsd.cascade import probs, tokenv3_target

TREE = [[[], [[]]], [], [[]]]          # root: 3 children; first has 2 (second of which has 1); third has 1


def test_flatten_bfs_and_allow():
    par, dep, rank, kids = flatten(TREE)
    assert par == [-1, 0, 0, 0, 1, 1, 3, 5]
    assert dep == [0, 1, 1, 1, 2, 2, 2, 3]
    assert rank == [0, 1, 2, 3, 1, 2, 1, 1]
    a = ancestor_allow(par)
    assert a[7].nonzero().flatten().tolist() == [0, 1, 5, 7]
    assert a.sum().item() == 1 + 2 + 2 + 2 + 3 + 3 + 3 + 4


@pytest.mark.parametrize('attn', ['eager', 'sdpa'])
def test_tree_attention_matches_path_forward(attn):
    m = tiny(0, attn=attn)
    ok, worst, noise = check_tree_attention(m, TREE, ctx_len=12, V=V, device='cpu', tol=1e-4, vocab_hi=V)
    assert ok, (worst, noise)


def test_keep_path_gives_same_cache_as_plain_forward():
    m = tiny(0)
    torch.manual_seed(1)
    prefix = torch.randint(0, V, (1, 10))
    par, dep, rank, kids = flatten(TREE)
    n = len(par)
    node_tok = torch.randint(0, V, (n,))
    node_tok[0] = prefix[0, -1]
    with torch.no_grad():
        cache = m(input_ids=prefix[:, :-1], use_cache=True).past_key_values
        ctx = 9
        mask = additive_mask(ancestor_allow(par), ctx, torch.float32)
        pos = torch.tensor([[ctx + d for d in dep]])
        m(input_ids=node_tok[None], position_ids=pos, attention_mask=mask, past_key_values=cache, use_cache=True)
        path = [0, 1, 5, 7]
        keep_path(cache, ctx, path)
        assert cache.get_seq_length() == ctx + len(path)
        nxt = torch.tensor([[7]])
        a = m(input_ids=nxt, past_key_values=cache, use_cache=True).logits[0, -1]
        seq = torch.cat([prefix, node_tok[path[1:]][None], nxt], 1)
        b = m(input_ids=seq).logits[0, -1]
    assert (a - b).abs().max().item() < 1e-4


def test_root_only_tree_is_autoregressive():
    d, t = pair()
    prompt = [3, 5, 7, 11, 13]
    dec = TreeDecoder(d, t, V, C, 0.0, [], mode='p')
    g1, g2 = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)
    a, _ = dec.generate(prompt, g1, [], 20)
    b, _ = autoregressive(t, V, prompt, C, g2, [], 20)
    assert a == b


@pytest.mark.parametrize('tree', [TREE, chain(4), [[[[]], []], [[]]]])
def test_cache_invariants_and_final_cache(tree):
    d, t = pair()
    prompt = [1, 2, 3, 4, 5, 6]
    dec = TreeDecoder(d, t, V, C, 0.3, tree, mode='cascade')
    out, log = dec.generate(prompt, torch.Generator().manual_seed(0), [], 25, debug=True)
    assert len(out) == 25 and log['steps'] >= 25 / (max(flatten(tree)[1]) + 1)
    committed = log['_committed']
    with torch.no_grad():   # the surgically maintained cache must equal a fresh forward
        nxt = torch.tensor([[committed[-1]]])
        a = t(input_ids=nxt, past_key_values=log['_tcache'], use_cache=True).logits[0, -1]
        b = t(input_ids=torch.tensor([committed])).logits[0, -1]
    assert (a - b).abs().max().item() < 1e-4


def _first_token_hist(dec, prompt, runs, seed=0):
    counts = np.zeros(V)
    for r in range(runs):
        out, _ = dec.generate(prompt, torch.Generator().manual_seed(seed + r), [], 1)
        counts[out[0]] += 1
    return counts


@pytest.mark.slow
@pytest.mark.parametrize('mode,alpha', [('p', 0.0), ('cascade', 0.3), ('cascade', 0.0)])
def test_first_token_distribution(mode, alpha):
    d, t = pair()
    prompt = [5, 9, 2, 33, 17]
    dec = TreeDecoder(d, t, V, C, alpha, TREE, mode=mode)
    counts = _first_token_hist(dec, prompt, 4000)
    with torch.no_grad():
        q = probs(d(input_ids=torch.tensor([prompt])).logits[0, -1, :V])
        p = probs(t(input_ids=torch.tensor([prompt])).logits[0, -1, :V])
    want = p if mode == 'p' else tokenv3_target(q, p, q, p, alpha)
    want = want.double().numpy()
    exp = want * counts.sum()
    m = exp > 5
    lump_c = np.append(counts[m], counts[~m].sum())
    lump_e = np.append(exp[m], exp[~m].sum())
    keep = lump_e > 0
    assert chisquare(lump_c[keep], lump_e[keep] * lump_c[keep].sum() / lump_e[keep].sum()).pvalue > 1e-3


@pytest.mark.slow
def test_second_token_distribution_through_cache_surgery():
    # condition on the first token; the second must follow p(. | prompt, t1): exercises keep_path
    d, t = pair()
    prompt = [5, 9, 2, 33, 17]
    dec = TreeDecoder(d, t, V, C, 0.0, chain(1), mode='p')   # 1 drafted token: both tokens often in one step
    pairs = []
    for r in range(4000):
        out, _ = dec.generate(prompt, torch.Generator().manual_seed(r), [], 2)
        pairs.append(tuple(out[:2]))
    firsts = np.bincount([a for a, _ in pairs], minlength=V)
    t1 = int(firsts.argmax())
    seconds = np.bincount([b for a, b in pairs if a == t1], minlength=V).astype(float)
    with torch.no_grad():
        p = probs(t(input_ids=torch.tensor([prompt + [t1]])).logits[0, -1, :V]).double().numpy()
    exp = p * seconds.sum()
    m = exp > 5
    lump_c = np.append(seconds[m], seconds[~m].sum())
    lump_e = np.append(exp[m], exp[~m].sum())
    keep = lump_e > 0
    assert chisquare(lump_c[keep], lump_e[keep] * lump_c[keep].sum() / lump_e[keep].sum()).pvalue > 1e-3


def test_alpha_one_accepts_first_child_always():
    d, t = pair()
    dec = TreeDecoder(d, t, V, C, 1.0, TREE, mode='cascade')
    _, log = dec.generate([4, 8, 15, 16], torch.Generator().manual_seed(0), [], 30)
    root = {k: v for k, v in log['visits'].items() if k.startswith('0,')}
    assert root and all(k.endswith(',0') for k in root)


def test_eos_stops_generation():
    d, t = pair()
    dec = TreeDecoder(d, t, V, C, 0.5, TREE, mode='cascade')
    out, _ = dec.generate([4, 8, 15], torch.Generator().manual_seed(0), list(range(V)), 30)
    assert len(out) == 1                    # every token is EOS here
