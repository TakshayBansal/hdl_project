"""Hypothesis tests H1-H4 (spec 9.3) and LaTeX tables for the report. CPU only, no torch.

  python scripts/summarize.py --pair A --task cnndm > summary.txt
"""
import argparse, glob, os
import numpy as np
from scipy.stats import spearmanr
from _common import add_pair_args, cfg_from
from jcsd.config import pair_name, setting_name
from jcsd.paths import load_quality, load_json, stored_path
from jcsd.search import evaluate_tree
from jcsd.tasks import paired_bootstrap_ci

ap = argparse.ArgumentParser()
add_pair_args(ap)
ap.add_argument('--tables', default='tables', help='directory for .tex tables')
a = ap.parse_args()
cfg = cfg_from(a, device='cpu')
pair, T, R = pair_name(cfg), cfg.temperature, a.root
tsuf = '' if T == 1.0 else '_T%g' % T
os.makedirs(a.tables, exist_ok=True)
print('=== %s | %s | T=%g ===' % (pair, a.task, T))

Q = load_quality(R, pair, a.task, T)
cas = sorted([r for r in Q.values() if r['mode'] == 'cascade'], key=lambda r: r['alpha'])
if cas:
    al = [r['alpha'] for r in cas]
    p1 = [r['acc']['pk'][0] for r in cas]
    rho, pv = spearmanr(al, p1)
    print('\nH1  p_1 vs alpha:', ' '.join('%g:%.3f' % x for x in zip(al, p1)))
    print('    Spearman rho = %.3f (p = %.3g); monotone non-decreasing: %s'
          % (rho, pv, all(b >= a_ - 1e-3 for a_, b in zip(p1, p1[1:]))))
    if 'p' in Q:
        print('    exact target p_1 = %.3f;  Q(p) = %.4f  Q(q) = %.4f' % (Q['p']['acc']['pk'][0], Q['p']['quality'],
                                                                         Q['q']['quality'] if 'q' in Q else float('nan')))
    for r in cas:
        s = np.asarray(r['acc']['p1_samples'])
        if len(s):
            print('    alpha %g: per-position p_1 mean %.3f, share > 0.95: %.2f, share < 0.5: %.2f'
                  % (r['alpha'], s.mean(), (s > 0.95).mean(), (s < 0.5).mean()))

searches = sorted(glob.glob(os.path.join(R, 'search', '*_%s_%s%s.json' % (pair, a.task, tsuf))))
S_by_hw = {}
for f in searches:
    S = load_json(f)
    S_by_hw[S['hw']] = S
    ks = sorted(S['per_alpha'], key=float)
    x = [float(k) for k in ks]
    b = [S['per_alpha'][k]['best'] for k in ks]
    print('\nH2  [%s] optimal tree vs alpha' % S['hw'])
    for k, bb in zip(ks, b):
        print('    alpha %s: n=%3d d=%2d widths=%s G=%.2f pred. speedup %.3f' % (k, bb['n'], bb['d'], bb['widths'], bb['G'], bb['speedup']))
    if len(x) > 2:
        print('    Spearman(alpha, depth) = %.3f   Spearman(alpha, max width) = %.3f'
              % (spearmanr(x, [bb['d'] for bb in b])[0], spearmanr(x, [max(bb['widths']) for bb in b])[0]))
    print('\nH4  [%s] predicted joint vs sequential' % S['hw'])
    for tol, r in S['by_tol'].items():
        if r.get('feasible'):
            print('    tol %-5s joint a=%g %.3fx | sequential a=%g %.3fx | ratio %.3f | fc chain %.3fx | sequoia %.3fx'
                  % (tol, r['joint']['alpha'], r['joint']['speedup'], r['sequential']['alpha'],
                     r['sequential']['speedup'], r['joint_over_sequential'], r['fc_chain']['speedup'],
                     S['sequoia']['speedup']))

# predicted transfer (H3): evaluate each platform's joint tree under every other profile
if len(S_by_hw) > 1:
    print('\nH3  predicted transfer (row = optimum from, col = run on), first feasible tolerance')
    hws = sorted(S_by_hw)
    for src in hws:
        cells = []
        for dst in hws:
            Sd, Ss = S_by_hw[dst], S_by_hw[src]
            tol = next((t for t, r in Ss['by_tol'].items() if r.get('feasible')), None)
            if tol is None:
                cells.append('   n/a'); continue
            c = next(c for c in Ss['configs'] if c['name'] == 'joint_tol%g' % float(tol))
            acc = Q[setting_name('cascade', c['alpha'])]['acc']['pk']      # acceptance is hardware independent
            ev = evaluate_tree(c['tree'], acc, load_json(stored_path(R, Sd['profile'])), **Sd['cost'])
            own = Sd['by_tol'][tol]['joint']['speedup'] if Sd['by_tol'][tol].get('feasible') else np.nan
            cells.append('%.3f (%+.1f%%)' % (ev['speedup'], 100 * (ev['speedup'] / own - 1)))
        print('    %-18s' % src, ' | '.join(cells))

e2es = sorted(glob.glob(os.path.join(R, 'results', 'e2e_*_%s_%s%s*.json' % (pair, a.task, tsuf))))
for f in e2es:
    E = load_json(f)
    M = E['methods']
    print('\nMeasured [%s] %s  (n=%d prompts x %d passes)' % (E['hw'], os.path.basename(f), E['n'], E['passes']))
    lines = []
    for name, m in M.items():
        q = m['quality']
        print('    %-24s %7.2f ms/tok  speedup %.3f  pred %.3f  tok/step %.2f  Q %.4f [%.4f, %.4f]'
              % (name, m['ms_per_token'], m['speedup'], m['config'].get('predicted') or 1.0,
                 m['tokens_per_step'], q[0], q[1], q[2]))
        lines.append('%s & %.2f & %.2f$\\times$ & %.2f$\\times$ & %.2f & %.4f [%.4f, %.4f] & %.1f \\\\'
                     % (name.replace('_', '\\_'), m['ms_per_token'], m['speedup'], m['config'].get('predicted') or 1.0,
                        m['tokens_per_step'], q[0], q[1], q[2], m['peak_gb']))
    tex = os.path.join(a.tables, os.path.basename(f).replace('.json', '.tex'))
    with open(tex, 'w') as fh:
        fh.write('\\begin{tabular}{lrrrrlr}\n\\toprule\nMethod & ms/tok & Speedup & Pred. & Tok/step & Quality [95\\%% CI] & Mem (GB) \\\\\n\\midrule\n')
        fh.write('\n'.join(lines) + '\n\\bottomrule\n\\end{tabular}\n')
    print('    wrote', tex)
    # H4 measured: paired bootstrap on per-(prompt, pass) ms/token
    for name in M:
        if not name.startswith('joint_tol') or '@' in name:
            continue
        seq = 'seq_tol' + name[len('joint_tol'):]
        if seq not in M:
            continue
        mj = 1000 * np.array(M[name]['time']) / np.maximum(M[name]['n_new'], 1)
        ms = 1000 * np.array(M[seq]['time']) / np.maximum(M[seq]['n_new'], 1)
        d, lo, hi = paired_bootstrap_ci(ms, mj)
        verdict = 'joint faster' if lo > 0 else ('sequential faster' if hi < 0 else 'no resolved difference')
        print('    H4 %s vs %s: sequential - joint = %+.2f ms/tok [%+.2f, %+.2f] -> %s'
              % (name, seq, d, lo, hi, verdict))
    # wasted siblings: share of visits at nodes with >1 child where the first child was taken
    for name, m in M.items():
        v = m.get('visits') or {}
        multi = {k: c for k, c in v.items() if int(k.split(',')[1]) > 1}
        tot = sum(multi.values())
        if tot:
            first = sum(c for k, c in multi.items() if int(k.split(',')[2]) == 0)
            print('    siblings unused (first child accepted at branching nodes) %-22s %.2f' % (name, first / tot))
