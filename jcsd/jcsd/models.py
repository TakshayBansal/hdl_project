import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_pair(cfg):
    dt = getattr(torch, cfg.dtype)
    attn = getattr(cfg, 'attn', 'sdpa')
    tok = AutoTokenizer.from_pretrained(cfg.target)
    tok_d = AutoTokenizer.from_pretrained(cfg.draft)
    assert tok.get_vocab() == tok_d.get_vocab(), 'tokenizers differ'
    kw = dict(torch_dtype=dt, attn_implementation=attn)
    draft = AutoModelForCausalLM.from_pretrained(cfg.draft, **kw).to(cfg.device).eval()
    target = AutoModelForCausalLM.from_pretrained(cfg.target, **kw).to(cfg.device).eval()
    V = common_vocab(draft, target)
    return tok, draft, target, V


def common_vocab(draft, target):
    # embedding matrices can be padded differently between sizes of one family; slice to the
    # common range and say so, because the logits beyond it belong to padding rows
    vd, vt = draft.config.vocab_size, target.config.vocab_size
    if vd != vt:
        print('NOTE: vocab sizes differ (draft %d, target %d); slicing both to %d' % (vd, vt, min(vd, vt)))
    return min(vd, vt)


@torch.no_grad()
def logits_of(model, ids, V, past=None):
    out = model(input_ids=ids, past_key_values=past, use_cache=True)
    return out.logits[..., :V].float(), out.past_key_values
