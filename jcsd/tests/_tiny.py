# Tiny random-weight GPT-NeoX models for CPU tests. Output weights are scaled up so the next-token
# distributions are peaked (a near-uniform distribution would make every sampling test vacuous).
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

V = 64


def tiny(seed=0, layers=2, hidden=32, sharp=6.0, attn='eager'):
    torch.manual_seed(seed)
    c = GPTNeoXConfig(vocab_size=V, hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=4,
                      intermediate_size=64, max_position_embeddings=512)
    try:
        c._attn_implementation = attn
    except Exception:
        pass
    m = GPTNeoXForCausalLM(c).eval()
    with torch.no_grad():
        m.get_output_embeddings().weight.mul_(sharp)   # embed_out (4.x) / lm_head (5.x)
    return m


class C:
    device = 'cpu'
    temperature = 1.0
    max_new_tokens = 8


def pair(attn='eager'):
    return tiny(1, layers=1, attn=attn), tiny(2, layers=2, attn=attn)
