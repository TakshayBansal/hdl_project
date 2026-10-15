# Smoke tests on tiny random-weight models, CPU only. They check plumbing (shapes, cache
# crop, the 4D-mask call path, JSON output), not speed or numerical agreement.
import json
import pytest
import torch
pytest.importorskip('transformers')
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
from jcsd.config import Cfg
from jcsd.models import common_vocab, logits_of
from jcsd import profile_hw as ph


def tiny(vocab=300, hidden=32, layers=2):
    torch.manual_seed(0)
    c = GPTNeoXConfig(vocab_size=vocab, hidden_size=hidden, num_hidden_layers=layers,
                      num_attention_heads=4, intermediate_size=64, max_position_embeddings=512)
    return GPTNeoXForCausalLM(c).eval()


def test_common_vocab_slices_to_min(capsys):
    assert common_vocab(tiny(vocab=300), tiny(vocab=280)) == 280
    assert 'slicing' in capsys.readouterr().out


def test_logits_of_shape_and_cache():
    m = tiny()
    ids = torch.randint(0, 300, (1, 32))
    lg, past = logits_of(m, ids, 290)
    assert lg.shape == (1, 32, 290) and lg.dtype == torch.float32
    lg2, _ = logits_of(m, ids[:, -1:], 290, past)
    assert lg2.shape == (1, 1, 290)


@pytest.mark.parametrize('masked', [False, True])
def test_time_forward_restores_cache(masked):
    m = tiny()
    t, raw = ph.time_forward(m, ctx_len=16, n=4, device='cpu', masked=masked, reps=3, warm=1)
    assert t > 0 and len(raw) == 3


def test_masked_forward_matches_plain_causal_on_chain():
    # a lower-triangular 4D mask over a chunk is plain causal attention, so logits must agree
    m = tiny()
    ctx, n = 12, 5
    ids = torch.randint(0, 300, (1, ctx))
    chunk = torch.randint(0, 300, (1, n))
    with torch.no_grad():
        c1 = m(input_ids=ids, use_cache=True).past_key_values
        a = m(input_ids=chunk, past_key_values=c1, use_cache=True).logits
        c2 = m(input_ids=ids, use_cache=True).past_key_values
        mask = ph._tree_like_mask(n, ctx, next(m.parameters()).dtype, 'cpu')
        pos = torch.arange(ctx, ctx + n)[None]
        b = m(input_ids=chunk, past_key_values=c2, use_cache=True, attention_mask=mask, position_ids=pos).logits
    assert (a - b).abs().max().item() < 1e-4


def test_profile_end_to_end_and_save(tmp_path):
    cfg = Cfg(device='cpu', dtype='float32')
    prof = ph.profile(tiny(), tiny(layers=3), cfg, ctx_len=16, sizes=(1, 2, 4), reps=3, warm=1)
    assert prof['sizes'] == [1, 2, 4] and len(prof['t_verify']) == 3 and prof['t_ar'] == prof['t_verify'][0]
    assert prof['t_ar_step'] > 0 and prof['hw'] == 'cpu'
    p = tmp_path / 'sub' / 'p.json'
    ph.save_profile(prof, str(p))
    assert json.load(open(p))['ctx_len'] == 16


def test_monotone_report():
    assert ph.monotone_report(dict(t_verify=[1.0, 1.0, 1.5, 2.0])) == []
    assert ph.monotone_report(dict(t_verify=[1.0, 1.0, 0.5, 2.0])) == [2]
