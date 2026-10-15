import torch


def probs(logits, T=1.0):
    return torch.softmax(logits.float() / T, dim=-1)


def tokenv3_target(q_raw, p_raw, q_s, p_s, alpha):
    # all tensors [..., V]; works for a single node or a batch of nodes
    thr = (1.0 - alpha) * p_raw.max(dim=-1, keepdim=True).values
    in_t = (p_raw >= thr)
    q_out = (q_s * (~in_t)).sum(dim=-1, keepdim=True)   # draft mass on implausible tokens
    return q_s * in_t + p_s * q_out


def plain_target(q_raw, p_raw, q_s, p_s, alpha=None):
    return p_s
