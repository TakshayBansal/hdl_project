"""Stage 5: joint (alpha, tree) search per platform, plus every baseline's predicted speedup.
Offline, CPU only: needs quality/ files (run_quality.py) and a profile (run_profile.py).

  python scripts/run_search.py --pair A --task cnndm --hw tesla_t4_fp16 --tols 0 0.02 0.05

Writes search/{hw}_{pair}_{task}.json with, for each quality tolerance tol
(feasible: Q(alpha) >= (1 - tol) * Q(p), or the paired-CI version with --criterion paired):
  joint      best alpha and its own DP tree                      (this project)
  sequential SEQUOIA tree for the exact target, then best alpha   (H4 baseline)
  fc_chain   best (alpha, gamma) on a chain                       (Faster Cascades)
and tolerance-free baselines: sequoia (exact target, DP tree), chain{g} (exact target).
"""
import argparse, os, sys
from _common import add_pair_args, cfg_from
from jcsd.config import pair_name, setting_name
from jcsd.paths import load_quality, find_profile, search_path, calib_path, load_json, save_json
from jcsd.search import best_for_acc, evaluate_tree, t_ar_of
from jcsd.dp import chain
from jcsd.stats import smooth_tail
from jcsd.tasks import paired_bootstrap_ci, MAX_NEW

ap = argparse.ArgumentParser()
add_pair_args(ap)
ap.add_argument('--hw', required=True, help='platform tag used in the profile file name')
ap.add_argument('--ctx', type=int, default=None, help='pick the profile closest to this context length')
ap.add_argument('--tols', type=float, nargs='+', default=[0.02, 0.05, 0.10])
ap.add_argument('--criterion', default='mean', choices=['mean', 'paired'],
                help='mean: Q(alpha) >= (1-tol) Q(p); paired: upper end of the paired-bootstrap CI of Q(alpha)-Q(p) >= -tol Q(p)')
ap.add_argument('--gammas', type=int, nargs='+', default=[3, 5, 7])
ap.add_argument('--N', type=int, default=128, help='max tree size')
ap.add_argument('--D', type=int, default=12, help='max tree depth')
ap.add_argument('--B', type=int, default=16, help='max children per node')
ap.add_argument('--overhead', type=float, default=None, help='seconds per step (default: calib file or 0)')
ap.add_argument('--per_node', type=float, default=None, help='seconds per accepted token (default: calib or 0)')
ap.add_argument('--no_level0', action='store_true', help='price the draft level-0 forward at width 1')
ap.add_argument('--smooth', action='store_true', help='power-law smoothing of the acceptance tail')
ap.add_argument('--depth_matrix', action='store_true', help='use P[d,k] from the calib file (spec 9.2 item 4)')
a = ap.parse_args()

cfg = cfg_from(a, max_tree=a.N, max_depth=a.D, max_children=a.B, device='cpu')
pair = pair_name(cfg)
T = cfg.temperature
Qrec = load_quality(a.root, pair, a.task, T)
if 'p' not in Qrec:
    sys.exit('need the target-only quality file (setting p); run run_quality.py first')
alpha_recs = {r['alpha']: r for r in Qrec.values() if r['mode'] == 'cascade'}
alphas = sorted(alpha_recs)
prep = (lambda pk: smooth_tail(pk)) if a.smooth else (lambda pk: pk)
acc_ref = prep(Qrec['p']['acc']['pk'])
acc_by_alpha = {x: prep(alpha_recs[x]['acc']['pk']) for x in alphas}
q_by_alpha = {x: alpha_recs[x]['quality'] for x in alphas}
Qp = Qrec['p']['quality']
Qq = Qrec['q']['quality'] if 'q' in Qrec else None

prof_path, prof = find_profile(a.root, a.hw, pair, ctx=a.ctx or 512)
cal = {}
cpath = calib_path(a.root, a.hw, pair, a.task, T)
if os.path.exists(cpath):
    cal = load_json(cpath)
cost_kw = dict(overhead=a.overhead if a.overhead is not None else cal.get('overhead', 0.0),
               per_node=a.per_node if a.per_node is not None else cal.get('per_node', 0.0),
               level0=not a.no_level0)
if a.depth_matrix:
    if 'depth_matrix' not in cal:
        sys.exit('no depth_matrix in %s; run calibrate.py after an e2e run' % cpath)
    # scale the reference matrix by each alpha's change in the 1-D vector (first-order transfer)
    import numpy as np
    Mref = np.asarray(cal['depth_matrix'])
    def to_matrix(pk_alpha):
        r = np.ones(Mref.shape[1])
        ref = np.asarray(acc_ref, dtype=float)
        k = min(len(ref), Mref.shape[1] - 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(ref[:k] > 0, np.asarray(pk_alpha)[:k] / ref[:k], 1.0)
        r[1:k + 1] = ratio
        M = np.clip(Mref * r[None, :], 0, 1)
        s = M[:, 1:].sum(1, keepdims=True)
        return np.where(s > 1, M / np.maximum(s, 1e-12), M)
    acc_by_alpha = {x: to_matrix(acc_by_alpha[x]) for x in alphas}
    acc_ref = Mref
print('profile', prof_path, '| t_ar %.4f s | cost' % t_ar_of(prof), cost_kw)

GRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
per_alpha = {}
for x in alphas:
    b = best_for_acc(acc_by_alpha[x], cfg, prof, surface=True, **cost_kw)
    surf = {k: v for k, v in b.pop('surface').items() if int(k.split(',')[1]) in GRID}
    chains = {g: evaluate_tree(chain(g), acc_by_alpha[x], prof, **cost_kw) for g in a.gammas}
    d, lo, hi = paired_bootstrap_ci(alpha_recs[x]['scores'], Qrec['p']['scores'], seed=a.seed)
    per_alpha[x] = dict(alpha=x, quality=q_by_alpha[x], quality_ci=alpha_recs[x]['ci'],
                        diff_vs_p=[d, lo, hi], p1=alpha_recs[x]['acc']['pk'][0], best=b,
                        chains={str(g): c for g, c in chains.items()}, surface=surf)
    print('alpha %.2f Q=%.4f (vs p %+.4f [%+.4f,%+.4f]) p1=%.3f | best n=%d d=%d G=%.2f speedup %.3f'
          % (x, q_by_alpha[x], d, lo, hi, per_alpha[x]['p1'], b['n'], b['d'], b['G'], b['speedup']))

sequoia = best_for_acc(acc_ref, cfg, prof, **cost_kw)
spec_chains = {g: evaluate_tree(chain(g), acc_ref, prof, **cost_kw) for g in a.gammas}
print('sequoia (exact target): n=%d d=%d G=%.2f speedup %.3f' % (sequoia['n'], sequoia['d'], sequoia['G'], sequoia['speedup']))

configs = [dict(name='ar', mode='p', alpha=None, tree=None, predicted=1.0)]
for g in a.gammas:
    configs.append(dict(name='chain%d' % g, mode='p', alpha=None, tree=chain(g),
                        predicted=spec_chains[g]['speedup'], G=spec_chains[g]['G']))
configs.append(dict(name='sequoia', mode='p', alpha=None, tree=sequoia['tree'],
                    predicted=sequoia['speedup'], G=sequoia['G']))
by_tol = {}
for tol in a.tols:
    qmin = (1 - tol) * Qp
    if a.criterion == 'mean':
        feas = [x for x in alphas if q_by_alpha[x] >= qmin]
    else:
        feas = [x for x in alphas if per_alpha[x]['diff_vs_p'][2] >= -tol * Qp]
    if not feas:
        print('tol %.3f: no feasible alpha (Q(p)=%.4f)' % (tol, Qp))
        by_tol[str(tol)] = dict(q_min=qmin, feasible=[])
        continue
    aj = max(feas, key=lambda x: per_alpha[x]['best']['speedup'])
    seq = {x: evaluate_tree(sequoia['tree'], acc_by_alpha[x], prof, **cost_kw) for x in feas}
    aseq = max(feas, key=lambda x: seq[x]['speedup'])
    fc = max(((x, g) for x in feas for g in a.gammas), key=lambda xg: per_alpha[xg[0]]['chains'][str(xg[1])]['speedup'])
    J = per_alpha[aj]['best']
    rec = dict(q_min=qmin, feasible=feas,
               joint=dict(alpha=aj, n=J['n'], d=J['d'], G=J['G'], speedup=J['speedup'], widths=J['widths']),
               sequential=dict(alpha=aseq, G=seq[aseq]['G'], speedup=seq[aseq]['speedup']),
               fc_chain=dict(alpha=fc[0], gamma=fc[1], speedup=per_alpha[fc[0]]['chains'][str(fc[1])]['speedup']))
    rec['joint_over_sequential'] = J['speedup'] / seq[aseq]['speedup']
    by_tol[str(tol)] = rec
    t = '%g' % tol
    configs.append(dict(name='joint_tol' + t, mode='cascade', alpha=aj, tree=J['tree'], predicted=J['speedup'], G=J['G']))
    configs.append(dict(name='seq_tol' + t, mode='cascade', alpha=aseq, tree=sequoia['tree'],
                        predicted=seq[aseq]['speedup'], G=seq[aseq]['G']))
    configs.append(dict(name='fc_chain_tol' + t, mode='cascade', alpha=fc[0], tree=chain(fc[1]),
                        predicted=rec['fc_chain']['speedup'],
                        G=per_alpha[fc[0]]['chains'][str(fc[1])]['G']))
    print('tol %-5g q_min %.4f feasible %s | joint a=%.2f (n=%d,d=%d) %.3f | sequential a=%.2f %.3f | fc chain a=%.2f g=%d %.3f | joint/seq %.3f'
          % (tol, qmin, feas, aj, J['n'], J['d'], J['speedup'], aseq, seq[aseq]['speedup'],
             fc[0], fc[1], rec['fc_chain']['speedup'], rec['joint_over_sequential']))

out = dict(hw=a.hw, pair=pair, task=a.task, temperature=T, profile=prof_path, profile_hw=prof.get('hw'),
           cost=cost_kw, criterion=a.criterion, t_ar=t_ar_of(prof), N=a.N, D=a.D, B=a.B, smooth=a.smooth,
           depth_matrix=a.depth_matrix, Q_p=Qp, Q_q=Qq, Q_p_ci=Qrec['p']['ci'], acc_ref=list(map(float, acc_ref[:a.B])) if not a.depth_matrix else None,
           per_alpha={str(x): v for x, v in per_alpha.items()},
           sequoia=dict(n=sequoia['n'], d=sequoia['d'], G=sequoia['G'], speedup=sequoia['speedup'], widths=sequoia['widths']),
           chains={str(g): v for g, v in spec_chains.items()}, by_tol=by_tol, configs=configs,
           max_new=MAX_NEW[a.task])
p = search_path(a.root, a.hw, pair, a.task, T)
save_json(out, p)
print('saved', p)
