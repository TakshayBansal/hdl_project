"""Profile t_verify(n) and t_draft(n) on this machine.  Run once per platform (spec 7.7).

  python scripts/run_profile.py --pair A --ctx 512 1024 --masked --tag t4_fp16
--masked times the explicit-4D-mask path that tree verification really uses (use it for search).
--tag labels the platform; a different precision on the same GPU is a different tag (say so).
"""
import argparse, os
from _common import add_pair_args, cfg_from, hw_tag
from jcsd.config import pair_name
from jcsd.models import load_pair
from jcsd.profile_hw import profile, save_profile, monotone_report

ap = argparse.ArgumentParser()
add_pair_args(ap)
ap.add_argument('--ctx', type=int, nargs='+', default=[512])
ap.add_argument('--sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
ap.add_argument('--masked', action='store_true', help='time the explicit-4D-mask path (tree verification)')
ap.add_argument('--reps', type=int, default=30)
ap.add_argument('--tag', default=None, help='label for the platform, e.g. t4_fp16')
a = ap.parse_args()

cfg = cfg_from(a)
tok, draft, target, V = load_pair(cfg)
hw = hw_tag(a.tag)
pair = pair_name(cfg)
for ctx in a.ctx:
    prof = profile(draft, target, cfg, ctx_len=ctx, sizes=a.sizes, masked=a.masked, reps=a.reps)
    prof['tag'] = hw
    prof['pair'] = pair
    path = os.path.join(a.root, 'profiles', '%s_%s_ctx%d%s.json' % (hw, pair, ctx, '_masked' if a.masked else ''))
    save_profile(prof, path)
    print('ctx', ctx, 'saved', path)
    print('  sizes   ', prof['sizes'])
    print('  t_verify', ['%.4f' % t for t in prof['t_verify']])
    print('  t_draft ', ['%.4f' % t for t in prof['t_draft_level']])
    print('  t_ar (forward) %.4f s, t_ar (full step) %.4f s' % (prof['t_ar'], prof['t_ar_step']))
    bad = monotone_report(prof)
    if bad:
        print('  WARNING: t_verify decreases at sizes', [prof['sizes'][i] for i in bad], '- rerun, GPU may be noisy')
