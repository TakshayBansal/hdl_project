# Batched left-padded sampling (stage 3) must agree with unbatched forwards row by row.
import numpy as np
import pytest
import torch
from scipy.stats import chisquare
pytest.importorskip('transformers')
from _tiny import pair, C, V
from jcsd.quality import PairStepper, sample_batch, target_dist
from jcsd.accept import AcceptCounter
from jcsd.cascade import probs, tokenv3_target


@pytest.mark.parametrize('attn', ['eager', 'sdpa'])
def test_batched_left_padding_matches_unbatched(attn):
    d, t = pair(attn)
    prompts = [[3, 4, 5], [7, 8, 9, 10, 11, 12], [1, 2, 3, 4, 5]]
    st = PairStepper(d, t, V, pad_id=0, device='cpu')
    lq, lp = st.prefill(prompts)
    forced = [[10, 20, 30], [11, 21, 31], [12, 22, 32]]
    for s in range(3):
        lq, lp = st.step(torch.tensor([f[s] for f in forced]))
    with torch.no_grad():
        for i, p in enumerate(prompts):
            seq = torch.tensor([p + forced[i]])
            assert (d(input_ids=seq).logits[0, -1, :V] - lq[i]).abs().max().item() < 1e-4
            assert (t(input_ids=seq).logits[0, -1, :V] - lp[i]).abs().max().item() < 1e-4


@pytest.mark.slow
@pytest.mark.parametrize('mode,alpha', [('p', None), ('cascade', 0.3)])
def test_first_token_distribution(mode, alpha):
    d, t = pair()
    prompt = [5, 9, 2, 33, 17]
    other = [1, 2]
    counts = np.zeros(V)
    for r in range(150):
        g = torch.Generator().manual_seed(r)
        outs = sample_batch(d, t, V, [prompt] * 20 + [other], C, alpha, mode, g, [], 1, pad_id=0)
        for o in outs[:20]:
            counts[o[0]] += 1
    with torch.no_grad():
        lq = d(input_ids=torch.tensor([prompt])).logits[0, -1, :V]
        lp = t(input_ids=torch.tensor([prompt])).logits[0, -1, :V]
    _, want = target_dist(mode, alpha, lq, lp, 1.0)
    exp = want.double().numpy() * counts.sum()
    m = exp > 5
    lc, le = np.append(counts[m], counts[~m].sum()), np.append(exp[m], exp[~m].sum())
    k = le > 0
    assert chisquare(lc[k], le[k] * lc[k].sum() / le[k].sum()).pvalue > 1e-3


def test_eos_rows_stop_and_counter_fills():
    d, t = pair()
    c = AcceptCounter(4, 'cpu', seed=0)
    outs = sample_batch(d, t, V, [[1, 2, 3], [4, 5]], C, 0.5, 'cascade', torch.Generator().manual_seed(0),
                        [0, 1, 2, 3, 4, 5, 6, 7], 10, pad_id=0, counter=c)
    for o in outs:
        assert 1 <= len(o) <= 10
        assert all(x not in range(8) for x in o[:-1])
    r = c.result()
    assert sum(r['counts']) > 0 and abs(sum(r['pk']) + r['all_rejected'] - 1) < 1e-9
