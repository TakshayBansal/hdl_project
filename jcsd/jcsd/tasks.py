import numpy as np

NL = chr(10)


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


def load_task(name, split, n, seed=0):
    from datasets import load_dataset
    if name == 'cnndm':
        ds = load_dataset('cnn_dailymail', '3.0.0', split=split)
    elif name == 'gsm8k':
        ds = load_dataset('openai/gsm8k', 'main', split=split)
    else:
        raise ValueError('unknown task: ' + name)
    ds = ds.shuffle(seed=seed).select(range(n))
    return [dict(ex) for ex in ds]


def make_prompt(name, ex, shots):
    if name == 'cnndm':
        art = ' '.join(ex['article'].split()[:300])
        return 'Article: ' + art + NL + 'Summary:'
    parts = ['Question: ' + s['question'] + NL + 'Answer: ' + s['answer'] for s in shots]
    parts.append('Question: ' + ex['question'] + NL + 'Answer:')
    return (NL + NL).join(parts)


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
