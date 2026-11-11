"""Spec 8.3 on the real models: do they honour a custom 4D tree mask with explicit position ids?
Also runs a short real decode with the cache-invariant asserts switched on.

  python scripts/check_tree_attention.py --pair A
If it fails with sdpa, rerun with --attn eager; if that also fails, switch model family (pair B).
"""
import argparse, sys
from _common import add_pair_args, cfg_from
import torch
from jcsd.models import load_pair
from jcsd.tree import check_tree_attention
from jcsd.decode_tree import TreeDecoder
from jcsd.tasks import eos_ids_of

ap = argparse.ArgumentParser()
add_pair_args(ap)
a = ap.parse_args()
cfg = cfg_from(a)
tok, draft, target, V = load_pair(cfg)
TREE = [[[[]], []], [[]], []]
TREE += [[[[]]]]                      # root with 4 children, depth 3
ok_all = True
for name, m in (('target', target), ('draft', draft)):
    for ctx in (64, 400):
        ok, worst, noise = check_tree_attention(m, TREE, ctx, V, cfg.device)
        ok_all &= ok
        print('%-6s ctx %4d  max|logit diff| %.4f  (rounding baseline %.4f)  %s'
              % (name, ctx, worst, noise, 'OK' if ok else 'FAIL'))
ids = tok('The quick brown fox jumps over the lazy dog. Then the fox', add_special_tokens=False)['input_ids']
dec = TreeDecoder(draft, target, V, cfg, 0.3, TREE, mode='cascade')
out, log = dec.generate(ids, torch.Generator(device=cfg.device).manual_seed(0), eos_ids_of(tok), 40, debug=True)
print('decode with invariant checks OK: %d tokens in %d steps -> %r'
      % (len(out), log['steps'], tok.decode(out)[:120]))
sys.exit(0 if ok_all else 1)
