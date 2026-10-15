"""Profile t_verify(n) and t_draft(n) on this machine.  Run once per platform.

  python scripts/run_profile.py --ctx 128 512 --masked --tag t4_fp16
"""
import argparse, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch
from jcsd.config import Cfg
from jcsd.models import load_pair
from jcsd.profile_hw import profile, save_profile, monotone_report

ap = argparse.ArgumentParser()
ap.add_argument('--draft', default=Cfg.draft)
ap.add_argument('--target', default=Cfg.target)
ap.add_argument('--dtype', default=Cfg.dtype)
ap.add_argument('--attn', default=Cfg.attn)
ap.add_argument('--ctx', type=int, nargs='+', default=[256])
ap.add_argument('--sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
ap.add_argument('--masked', action='store_true', help='time the explicit-4D-mask path (tree verification)')
ap.add_argument('--reps', type=int, default=30)
ap.add_argument('--tag', default=None, help='label for the platform, e.g. t4_fp16')
ap.add_argument('--out_dir', default='profiles')
a = ap.parse_args()

cfg = Cfg(draft=a.draft, target=a.target, dtype=a.dtype, attn=a.attn)
tok, draft, target, V = load_pair(cfg)
hw = a.tag or re.sub(r'\W+', '_', torch.cuda.get_device_name()).strip('_').lower()
pair = a.target.split('/')[-1] + '__' + a.draft.split('/')[-1]
for ctx in a.ctx:
    prof = profile(draft, target, cfg, ctx_len=ctx, sizes=a.sizes, masked=a.masked, reps=a.reps)
    prof['tag'] = hw
    path = os.path.join(a.out_dir, '%s_%s_ctx%d%s.json' % (hw, pair, ctx, '_masked' if a.masked else ''))
    save_profile(prof, path)
    print('ctx', ctx, 'saved', path)
    print('  sizes   ', prof['sizes'])
    print('  t_verify', ['%.4f' % t for t in prof['t_verify']])
    print('  t_draft ', ['%.4f' % t for t in prof['t_draft_level']])
    print('  t_ar (forward) %.4f s, t_ar (full step) %.4f s' % (prof['t_ar'], prof['t_ar_step']))
    bad = monotone_report(prof)
    if bad:
        print('  WARNING: t_verify decreases at sizes', [prof['sizes'][i] for i in bad], '- rerun, GPU may be noisy')
