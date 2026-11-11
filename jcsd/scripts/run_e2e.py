"""Stage 6: end-to-end measured decoding on the test split, every method through the same decoder.

  python scripts/run_e2e.py --pair A --task cnndm --hw tesla_t4_fp16 --n 100 --passes 3
  # transfer (H3): also run another platform's optimum here
  python scripts/run_e2e.py ... --hw tesla_t4_fp16 --transfer search/tesla_p100_..._cnndm.json

Methods are interleaved per prompt (order rotated) so throttling hits all of them alike, and
every method gets the same random seed for a given (prompt, pass). Batch size 1.
"""
import argparse, os, time
from _common import add_pair_args, cfg_from
import torch
from jcsd.config import pair_name
from jcsd.models import load_pair
from jcsd.tasks import load_role, encode_prompts, eos_ids_of, clean_output, score, bootstrap_ci, MAX_NEW
from jcsd.decode_tree import TreeDecoder, autoregressive
from jcsd.profile_hw import env_info
from jcsd.paths import search_path, e2e_path, load_json, save_json

ap = argparse.ArgumentParser()
add_pair_args(ap)
ap.add_argument('--hw', required=True)
ap.add_argument('--n', type=int, default=100, help='test prompts')
ap.add_argument('--passes', type=int, default=3)
ap.add_argument('--max_new', type=int, default=None)
ap.add_argument('--methods', nargs='*', default=None, help='subset of config names (default: all)')
ap.add_argument('--transfer', nargs='*', default=[], help='other platforms\' search files: run their joint optimum here')
ap.add_argument('--tag', default='', help='suffix for the output file (e.g. calib round)')
a = ap.parse_args()

cfg = cfg_from(a)
pair = pair_name(cfg)
T = cfg.temperature
max_new = a.max_new or MAX_NEW[a.task]
S = load_json(search_path(a.root, a.hw, pair, a.task, T))
configs = [dict(c, source=a.hw) for c in S['configs']]
for f in a.transfer:
    other = load_json(f)
    for c in other['configs']:
        if c['name'].startswith('joint') or c['name'] == 'sequoia':
            configs.append(dict(c, name=c['name'] + '@' + other['hw'], source=other['hw']))
if a.methods:
    configs = [c for c in configs if c['name'] in a.methods or c['name'] == 'ar']

tok, draft, target, V = load_pair(cfg)
exs = load_role(a.task, 'test', a.n, seed=a.seed)
prompts = encode_prompts(tok, a.task, exs, chat=cfg.chat)
eos = eos_ids_of(tok)

# one runner per distinct (mode, alpha, tree); names sharing it are aliases
runners, alias = {}, {}
for c in configs:
    key = 'ar' if c['name'] == 'ar' else repr((c['mode'], c['alpha'], c['tree']))
    alias[c['name']] = key
    if key in runners:
        continue
    if key == 'ar':
        runners[key] = lambda p, g: autoregressive(target, V, p, cfg, g, eos, max_new)
    else:
        dec = TreeDecoder(draft, target, V, cfg, c['alpha'] if c['alpha'] is not None else 0.0,
                          c['tree'], mode=c['mode'])
        runners[key] = (lambda d: (lambda p, g: d.generate(p, g, eos, max_new)))(dec)
keys = list(runners)
print('%d methods, %d distinct runners:' % (len(configs), len(keys)), [c['name'] for c in configs])

# warm-up on two dev prompts (kernels, allocator), not recorded
dev_p = encode_prompts(tok, a.task, load_role(a.task, 'dev', 2, seed=a.seed), chat=cfg.chat)
for k in keys:
    for p in dev_p:
        runners[k](p, torch.Generator(device=cfg.device).manual_seed(0))

R = {k: dict(time=[], n_new=[], steps=[], peak_gb=[], visits={}, path_len_hist={}, tokens=[None] * len(prompts))
     for k in keys}
out_path = e2e_path(a.root, a.hw, pair, a.task, T).replace('.json', ('_' + a.tag if a.tag else '') + '.json')
t_start = time.time()
for ps in range(a.passes):
    for i, p in enumerate(prompts):
        order = keys[(i + ps) % len(keys):] + keys[:(i + ps) % len(keys)]
        for k in order:
            g = torch.Generator(device=cfg.device).manual_seed(a.seed * 1000003 + ps * 10007 + i)
            torch.cuda.reset_peak_memory_stats()
            toks, log = runners[k](p, g)
            r = R[k]
            r['time'].append(log['time_decode']); r['n_new'].append(log['n_new']); r['steps'].append(log['steps'])
            r['peak_gb'].append(torch.cuda.max_memory_allocated() / 1e9)
            for vk, vc in log['visits'].items():
                r['visits'][vk] = r['visits'].get(vk, 0) + vc
            for L in log['path_len']:
                r['path_len_hist'][str(L)] = r['path_len_hist'].get(str(L), 0) + 1
            if ps == 0:
                r['tokens'][i] = toks
        if (i + 1) % 10 == 0:
            el = time.time() - t_start
            print('pass %d prompt %d/%d  (%.0f s elapsed)' % (ps + 1, i + 1, len(prompts), el), flush=True)

methods = {}
for c in configs:
    k = alias[c['name']]
    r = R[k]
    texts = [clean_output(a.task, tok.decode(t, skip_special_tokens=True)) for t in r['tokens']]
    sc = score(a.task, texts, exs)
    tot_t, tot_n, tot_s = sum(r['time']), sum(r['n_new']), sum(r['steps'])
    methods[c['name']] = dict(config=c, runner=k, ms_per_token=1000 * tot_t / max(tot_n, 1),
                              tokens_per_step=tot_n / max(tot_s, 1), quality=bootstrap_ci(sc, seed=a.seed),
                              scores=sc, peak_gb=max(r['peak_gb']), time=r['time'], n_new=r['n_new'],
                              steps=r['steps'], visits=r['visits'], path_len_hist=r['path_len_hist'])
ar = methods['ar']['ms_per_token']
for name, m in methods.items():
    m['speedup'] = ar / m['ms_per_token']
    print('%-22s %7.2f ms/tok  speedup %.3f (pred %.3f)  tok/step %.2f  Q %.4f  peak %.1f GB'
          % (name, m['ms_per_token'], m['speedup'], m['config'].get('predicted', float('nan')),
             m['tokens_per_step'], m['quality'][0], m['peak_gb']))
save_json(dict(hw=a.hw, pair=pair, task=a.task, temperature=T, n=len(prompts), passes=a.passes,
               max_new=max_new, search=S['profile'], methods=methods, env=env_info()), out_path)
print('saved', out_path)
