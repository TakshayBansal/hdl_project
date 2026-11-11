import numpy as np

NL = chr(10)

# default generation lengths per task (spec 7.6)
MAX_NEW = dict(cnndm=64, gsm8k=256)


def last_number(s):
    s = s.replace(',', '')
    cur, found = '', ''
    for ch in s + ' ':
        if ch.isdigit() or (ch == '-' and cur == '') or (ch == '.' and cur and '.' not in cur):
            cur += ch
        else:
            if cur and cur not in ('-', '.'):
                found = cur
            cur = ''
    return found.rstrip('.')


def _load(name, split):
    from datasets import load_dataset
    if name == 'cnndm':
        try:
            return load_dataset('abisee/cnn_dailymail', '3.0.0', split=split)
        except Exception:
            return load_dataset('cnn_dailymail', '3.0.0', split=split)
    if name == 'gsm8k':
        return load_dataset('openai/gsm8k', 'main', split=split)
    raise ValueError('unknown task: ' + name)


def load_task(name, split, n, seed=0):
    ds = _load(name, split).shuffle(seed=seed).select(range(n))
    return [dict(ex) for ex in ds]


N_SHOTS = 4


def load_shots(name):
    # GSM8K: the first four training examples, fixed forever. CNN/DM is zero-shot.
    if name != 'gsm8k':
        return []
    ds = _load(name, 'train')
    return [dict(ds[i]) for i in range(N_SHOTS)]


def load_role(name, role, n, seed=0):
    """Disjoint splits: role 'calib' (alpha selection, acceptance), 'test' (reported numbers),
    'dev' (debugging). CNN/DM: calib = validation, test = test, dev = train.
    GSM8K: calib and dev come from train (shots excluded), test = test."""
    if name == 'cnndm':
        split = dict(calib='validation', test='test', dev='train')[role]
        return load_task(name, split, n, seed)
    if role == 'test':
        return load_task(name, 'test', n, seed)
    ds = _load(name, 'train').select(range(N_SHOTS, 7473)).shuffle(seed=seed)
    off = 0 if role == 'calib' else 3000          # calib and dev never overlap
    return [dict(ds[i]) for i in range(off, off + n)]


def make_prompt(name, ex, shots):
    if name == 'cnndm':
        art = ' '.join(ex['article'].split()[:300])
        return 'Article: ' + art + NL + 'Summary:'
    parts = ['Question: ' + s['question'] + NL + 'Answer: ' + s['answer'] for s in shots]
    parts.append('Question: ' + ex['question'] + NL + 'Answer:')
    return (NL + NL).join(parts)


def make_chat_prompt(name, ex, tok):
    # for instruction-tuned pairs (e.g. Qwen2.5-*-Instruct): zero-shot with the chat template
    if name == 'cnndm':
        art = ' '.join(ex['article'].split()[:300])
        msg = 'Summarize the following news article in two or three sentences.' + NL + NL + art
    else:
        msg = ex['question'] + NL + "Reason step by step, then finish with 'The answer is <number>'."
    return tok.apply_chat_template([dict(role='user', content=msg)], tokenize=False,
                                   add_generation_prompt=True)


def encode_prompts(tok, name, exs, chat=False, shots=None):
    shots = shots if shots is not None else (load_shots(name) if not chat else [])
    texts = [make_chat_prompt(name, e, tok) if chat else make_prompt(name, e, shots) for e in exs]
    return [tok(t, add_special_tokens=not chat)['input_ids'] for t in texts]


def eos_ids_of(tok):
    ids = {tok.eos_token_id}
    for t in ('<|im_end|>', '<|endoftext|>', '<|eot_id|>'):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(i)
    return sorted(i for i in ids if i is not None)


def clean_output(name, text):
    stop = 'Article:' if name == 'cnndm' else 'Question:'
    return text.split(stop)[0].strip()


def gsm8k_gold(ex):
    return ex['answer'].split('####')[-1].strip().replace(',', '')


def score(name, preds, exs):
    if name == 'cnndm':
        import evaluate
        r = evaluate.load('rouge')
        out = r.compute(predictions=preds, references=[e['highlights'] for e in exs], use_aggregator=False)
        return list(out['rouge2'])
    gold = [gsm8k_gold(e) for e in exs]
    return [1.0 if last_number(p) == g else 0.0 for p, g in zip(preds, gold)]


def bootstrap_ci(scores, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    s = np.asarray(scores, dtype=float)
    means = [s[rng.integers(0, len(s), len(s))].mean() for _ in range(n_boot)]
    return float(s.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap_ci(a, b, n_boot=2000, seed=0):
    # CI on mean(a - b) over the same examples; if it contains 0 the data show no difference
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bootstrap_ci(d, n_boot, seed)
