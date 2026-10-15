import json, os, statistics, subprocess
import torch


def _sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def cuda_time(fn, warm=5, reps=30, device='cuda'):
    # median seconds over reps, with warm-up discarded; CUDA events on GPU, perf_counter on CPU
    import time
    for _ in range(warm):
        fn()
    _sync(device)
    ts = []
    for _ in range(reps):
        if str(device).startswith('cuda'):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) / 1000.0)
        else:
            t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts), ts


def _vocab(model):
    return min(model.config.vocab_size, 1000)


def _tree_like_mask(n, ctx_len, dtype, device):
    # stand-in for a tree mask: full prefix plus a lower-triangular block. It forces the same
    # explicit-4D-mask attention path that real tree verification uses.
    allow = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
    full = torch.cat([torch.ones(n, ctx_len, dtype=torch.bool, device=device), allow], dim=1)
    mask = torch.zeros(full.shape, dtype=dtype, device=device)
    mask.masked_fill_(~full, torch.finfo(dtype).min)
    return mask[None, None]


@torch.no_grad()
def time_forward(model, ctx_len, n, device, masked=False, reps=30, warm=5):
    hi = _vocab(model)
    ids = torch.randint(min(100, hi - 1), hi, (1, ctx_len), device=device)
    cache = model(input_ids=ids, use_cache=True).past_key_values
    chunk = torch.randint(min(100, hi - 1), hi, (1, n), device=device)
    kw = {}
    if masked:
        dt = next(model.parameters()).dtype
        kw = dict(attention_mask=_tree_like_mask(n, ctx_len, dt, device),
                  position_ids=torch.arange(ctx_len, ctx_len + n, device=device)[None])

    def fn():
        model(input_ids=chunk, past_key_values=cache, use_cache=True, **kw)
        cache.crop(ctx_len)
    return cuda_time(fn, warm=warm, reps=reps, device=device)


@torch.no_grad()
def time_ar_step(target, ctx_len, V, T, device, reps=30, warm=5):
    # one real autoregressive step: forward of 1 token, sample, host sync (what the AR baseline does)
    hi = _vocab(target)
    ids = torch.randint(min(100, hi - 1), hi, (1, ctx_len), device=device)
    cache = target(input_ids=ids, use_cache=True).past_key_values
    tok = torch.randint(min(100, hi - 1), hi, (1, 1), device=device)

    def fn():
        lg = target(input_ids=tok, past_key_values=cache, use_cache=True).logits[0, -1, :V].float()
        int(torch.multinomial(torch.softmax(lg / T, -1), 1).item())
        cache.crop(ctx_len)
    return cuda_time(fn, warm=warm, reps=reps, device=device)


def env_info():
    info = dict(torch=torch.__version__)
    try:
        import transformers
        info['transformers'] = transformers.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        info['hw'] = torch.cuda.get_device_name()
        try:
            info['nvidia_smi'] = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout
        except Exception:
            pass
    else:
        info['hw'] = 'cpu'
    return info


def profile(draft, target, cfg, ctx_len=256, sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
            masked=False, reps=30, warm=5):
    dev = cfg.device
    cuda = str(dev).startswith('cuda')
    prof = dict(ctx_len=ctx_len, sizes=list(sizes), masked=masked, reps=reps,
                t_verify=[], t_draft_level=[], peak_mem_gb=[], raw_verify={}, raw_draft={})
    prof.update(env_info())
    for n in sizes:
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        tv, raw = time_forward(target, ctx_len, n, dev, masked, reps, warm)
        td, rawd = time_forward(draft, ctx_len, n, dev, masked, reps, warm)
        prof['t_verify'].append(tv)
        prof['t_draft_level'].append(td)
        prof['peak_mem_gb'].append(torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0)
        prof['raw_verify'][str(n)] = raw
        prof['raw_draft'][str(n)] = rawd
    prof['t_ar'] = prof['t_verify'][0]          # one autoregressive target step (forward only)
    V = min(draft.config.vocab_size, target.config.vocab_size)
    prof['t_ar_step'], _ = time_ar_step(target, ctx_len, V, cfg.temperature, dev, reps, warm)
    return prof


def monotone_report(prof, noise=0.10):
    # done-when check: t_verify non-decreasing within noise. Returns the violating indices.
    tv = prof['t_verify']
    return [i for i in range(1, len(tv)) if tv[i] < tv[i - 1] * (1 - noise)]


def save_profile(prof, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(prof, f, indent=2)
