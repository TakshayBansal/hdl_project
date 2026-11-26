"""Cost-model calibration from an end-to-end run (spec 9.2). CPU only.

  python scripts/calibrate.py --pair A --task cnndm --hw tesla_t4_fp16
then re-run run_search.py (it picks up calib/...json automatically; add --depth_matrix to use the
depth-dependent acceptance matrix) and run_e2e.py --tag round2 to check the new predictions.

Fits   measured step time = forward-cost model + c0 + c1 * (tokens per step)
and rebuilds P[d, k] from the visit logs of the exact-target tree methods (sequoia, chains).
"""
import argparse
import numpy as np
from _common import add_pair_args, cfg_from
from jcsd.config import pair_name
from jcsd.paths import e2e_path, search_path, calib_path, load_json, save_json, stored_path
from jcsd.search import make_cost, fit_overhead, level_widths, t_ar_of
from jcsd.stats import depth_matrix

ap = argparse.ArgumentParser()
add_pair_args(ap)
ap.add_argument('--hw', required=True)
ap.add_argument('--e2e', default=None, help='e2e results file (default: the standard path)')
ap.add_argument('--D', type=int, default=12)
ap.add_argument('--B', type=int, default=16)
a = ap.parse_args()

cfg = cfg_from(a, device='cpu')
pair = pair_name(cfg)
T = cfg.temperature
E = load_json(a.e2e or e2e_path(a.root, a.hw, pair, a.task, T))
S = load_json(search_path(a.root, a.hw, pair, a.task, T))
prof = load_json(stored_path(a.root, S['profile']))
level0 = S['cost'].get('level0', True)
base = make_cost(prof, 0.0, 0.0, level0)

rows, errs = [], []
seen = set()
for name, m in E['methods'].items():
    c = m['config']
    if c['tree'] is None or m['runner'] in seen or c.get('source', a.hw) != a.hw:
        continue
    seen.add(m['runner'])
    G = m['tokens_per_step']
    step = sum(m['time']) / max(sum(m['steps']), 1)
    rows.append(dict(name=name, step_time=step, pred_forward=base(level_widths(c['tree']), G), G=G))
for name, m in E['methods'].items():
    if name != 'ar':
        errs.append(dict(name=name, predicted=m['config'].get('predicted'), measured=m['speedup'],
                         rel_err=(m['config'].get('predicted', np.nan) - m['speedup']) / m['speedup']))
c0, c1 = fit_overhead(rows)
print('fitted overhead: c0 = %.2f ms/step, c1 = %.2f ms per accepted token' % (1e3 * c0, 1e3 * c1))
for r in rows:
    print('  %-20s step %.2f ms  forward model %.2f ms  G %.2f  residual %.2f ms'
          % (r['name'], 1e3 * r['step_time'], 1e3 * r['pred_forward'], r['G'],
             1e3 * (r['step_time'] - r['pred_forward'] - c0 - c1 * r['G'])))
print('prediction error before calibration (predicted vs measured speedup):')
for e in errs:
    print('  %-22s pred %.3f  meas %.3f  rel.err %+.1f%%' % (e['name'], e['predicted'], e['measured'], 100 * e['rel_err']))

# depth-dependent acceptance from exact-target tree methods (alpha-free reference)
visits = {}
for name, m in E['methods'].items():
    if m['config']['mode'] == 'p' and m['config']['tree'] is not None and m['config'].get('source', a.hw) == a.hw:
        for k, v in m['visits'].items():
            visits[k] = visits.get(k, 0) + v
fallback = S.get('acc_ref') or []
P, reach = depth_matrix(visits, a.D, a.B, fallback)
print('depth-dependent p_{d,1} (rows with >= 30 visits):',
      ['d%d: %.3f' % (d, P[d, 1]) for d in range(a.D) if reach[d, 1] >= 30])
if fallback:
    print('1-D teacher-forced p_1 = %.3f' % fallback[0])
save_json(dict(overhead=c0, per_node=c1, rows=rows, pred_errors=errs, depth_matrix=P.tolist(),
               reach=reach.tolist(), source=a.e2e or e2e_path(a.root, a.hw, pair, a.task, T),
               t_ar=t_ar_of(prof)), calib_path(a.root, a.hw, pair, a.task, T))
print('saved', calib_path(a.root, a.hw, pair, a.task, T))
