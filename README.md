# Hardware-aware joint optimization of speculative trees and cascaded decoding

Joint search over the Faster Cascades deferral parameter alpha (TokenV3 rule) and the SEQUOIA token-tree
topology, priced by a measured hardware cost profile. Spec and math: `hdl_md.md`. Working notes: `CONTEXT.md`.

## Layout
```
jcsd/jcsd/      library
  cascade.py      TokenV3 target pi_alpha
  verify.py       node verification (sampling without replacement), Gumbel top-k drafting
  accept.py       batched verification on the GPU, acceptance counters
  stats.py        acceptance-tail smoothing, depth-dependent matrix from decoder logs (numpy)
  dp.py           SEQUOIA dynamic program, tree reconstruction, chain()
  search.py       cost model, joint / sequential search, overhead fit
  tree.py         flattening, tree masks, KV-cache surgery, tree-attention check
  decode_tree.py  end-to-end tree decoder (all speculative methods) + autoregressive baseline
  quality.py      batched sampling from pi_alpha (quality curve + inline acceptance)
  profile_hw.py   t_verify(n), t_draft(n) timing
  tasks.py        CNN/DM and GSM8K data, prompts, metrics, bootstrap CIs
jcsd/scripts/   pipeline stages (each prints what it saved)
jcsd/tests/     unit + CPU tests (tiny random models) + offline pipeline test
notebooks/jcsd_pipeline.ipynb   the whole pipeline on Colab / Kaggle
```

## Run order (per pair and task; `--pair A --task cnndm` shown)
| Stage | Command | Where | Output |
|---|---|---|---|
| tests | `python -m pytest -q` | GPU box (CPU tests) | must pass |
| 0 | `python scripts/check_tree_attention.py --pair A` | GPU | OK/FAIL per model |
| 2 | `python scripts/run_profile.py --pair A --ctx 512 [--masked] --tag t4_fp16` | each platform | `profiles/` |
| 3+4 | `python scripts/run_quality.py --pair A --task cnndm --n 200` | once (any GPU) | `quality/` |
| 5 | `python scripts/run_search.py --pair A --task cnndm --hw t4_fp16` | CPU | `search/` |
| 6 | `python scripts/run_e2e.py --pair A --task cnndm --hw t4_fp16 [--transfer search/<other>.json]` | each platform | `results/` |
| 6b | `python scripts/calibrate.py ...` then repeat 5 and 6 (`--tag round2`) | CPU, then GPU | `calib/` |
| 7 | `python scripts/make_figures.py ...` and `python scripts/summarize.py ...` | CPU | `figures/`, `tables/`, H1-H4 verdicts |

Methods compared end to end (same decoder, same seeds, interleaved): autoregressive; chain speculative (gamma 3/5/7);
SEQUOIA (DP tree, exact target); Faster Cascades (best alpha and gamma on a chain); sequential (SEQUOIA tree, then
best alpha); joint (best alpha with its own DP tree), at each quality tolerance; plus other platforms' joint optima (H3).
