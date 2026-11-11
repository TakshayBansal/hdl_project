"""Stages 3+4: quality curve Q(alpha) and acceptance vectors P(alpha) on the calibration split.

For each setting (draft-only 'q', target-only 'p', and every alpha of the TokenV3 cascade) it
samples token by token from that setting's target distribution, scores the outputs against
ground truth, and simulates node verification with B children at every generated position.
Hardware independent: run once per (pair, task, temperature), reuse on every platform.

  python scripts/run_quality.py --pair A --task cnndm --n 200 --batch 16
Resumable: settings whose output file exists are skipped (use --force to redo).
"""
import argparse, os, time
from _common import add_pair_args, cfg_from
import torch
from jcsd.config import pair_name, setting_name
from jcsd.models import load_pair, pad_id_of
from jcsd.tasks import load_role, encode_prompts, eos_ids_of, clean_output, score, bootstrap_ci, MAX_NEW
from jcsd.quality import sample_batch
from jcsd.accept import AcceptCounter
from jcsd.profile_hw import env_info
from jcsd.paths import quality_path, save_json

ap = argparse.ArgumentParser()
add_pair_args(ap)
ap.add_argument('--n', type=int, default=200, help='calibration prompts')
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--alphas', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0])
ap.add_argument('--no_endpoints', action='store_true', help='skip the q and p settings')
ap.add_argument('--max_new', type=int, default=None)
ap.add_argument('--B', type=int, default=16, help='children per simulated node (DP width cap)')
ap.add_argument('--sims', type=int, default=4, help='simulated verifications per position')
ap.add_argument('--force', action='store_true')
a = ap.parse_args()

cfg = cfg_from(a)
pair = pair_name(cfg)
max_new = a.max_new or MAX_NEW[a.task]
tok, draft, target, V = load_pair(cfg)
exs = load_role(a.task, 'calib', a.n, seed=a.seed)
prompts = encode_prompts(tok, a.task, exs, chat=cfg.chat)
eos = eos_ids_of(tok)
pad = pad_id_of(tok)
# batch prompts of similar length together (less padding); fixed order -> common random numbers
order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
batches = [order[i:i + a.batch] for i in range(0, len(order), a.batch)]
print('pair', pair, '| task', a.task, '| V', V, '| eos', eos, '| prompts', len(prompts),
      '| mean prompt len %.0f' % (sum(map(len, prompts)) / len(prompts)), '| max_new', max_new)

settings = ([] if a.no_endpoints else [('q', None), ('p', None)]) + [('cascade', x) for x in a.alphas]
for mode, alpha in settings:
    name = setting_name(mode, alpha)
    path = quality_path(a.root, pair, a.task, name, cfg.temperature)
    if os.path.exists(path) and not a.force:
        print('skip', name, '(exists)')
        continue
    t0 = time.time()
    counter = AcceptCounter(a.B, cfg.device, seed=a.seed + 7)
    gens = [None] * len(prompts)
    for bi, b in enumerate(batches):
        g = torch.Generator(device=cfg.device).manual_seed(a.seed * 100003 + bi)   # same per batch for all settings
        outs = sample_batch(draft, target, V, [prompts[i] for i in b], cfg, alpha, mode, g, eos,
                            max_new, pad, counter=counter, sims=a.sims)
        for i, o in zip(b, outs):
            gens[i] = o
    texts = [clean_output(a.task, tok.decode(g_, skip_special_tokens=True)) for g_ in gens]
    sc = score(a.task, texts, exs)
    m, lo, hi = bootstrap_ci(sc, seed=a.seed)
    acc = counter.result() if mode != 'q' else None
    rec = dict(setting=name, mode=mode, alpha=alpha, pair=pair, task=a.task, temperature=cfg.temperature,
               n=len(exs), max_new=max_new, B=a.B, sims=a.sims, quality=m, ci=[lo, hi], scores=sc,
               texts=texts, gen_lens=[len(x) for x in gens], acc=acc, seconds=time.time() - t0,
               env={k: v for k, v in env_info().items() if k != 'nvidia_smi'})
    save_json(rec, path)
    p1 = ('p1=%.3f p2=%.3f rej=%.3f' % (acc['pk'][0], acc['pk'][1], acc['all_rejected'])) if acc else ''
    print('%-6s Q=%.4f [%.4f, %.4f]  %s  (%.0fs)' % (name, m, lo, hi, p1, rec['seconds']))
