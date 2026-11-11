import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _dtype_kw(dt):
    # transformers >= 4.56 renamed torch_dtype to dtype; older versions silently ignore 'dtype'
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split('.')[:2])
    return dict(dtype=dt) if (major, minor) >= (4, 56) else dict(torch_dtype=dt)


def load_model(name, cfg):
    dt = getattr(torch, cfg.dtype)
    kw = dict(attn_implementation=getattr(cfg, 'attn', 'sdpa'), **_dtype_kw(dt))
    m = AutoModelForCausalLM.from_pretrained(name, **kw).to(cfg.device).eval()
    got = next(m.parameters()).dtype
    assert got == dt, 'model loaded as %s, wanted %s' % (got, dt)
    return m


def load_pair(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.target)
    tok_d = AutoTokenizer.from_pretrained(cfg.draft)
    assert tok.get_vocab() == tok_d.get_vocab(), 'tokenizers differ'
    draft = load_model(cfg.draft, cfg)
    target = load_model(cfg.target, cfg)
    # rows beyond len(tok) are embedding padding: never sample them
    V = min(common_vocab(draft, target), len(tok))
    return tok, draft, target, V


def pad_id_of(tok):
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


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
