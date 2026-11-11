# CONTEXT.md - Joint Cascade + Speculative-Tree Decoding (paste at the start of each session)

Spec: hdl_md.md. Code: jcsd/. Run tests from jcsd/ with `python -m pytest -q` (add `-m "not slow"` for a quick pass).
Full GPU pipeline: notebooks/jcsd_pipeline.ipynb (Colab or Kaggle, no Drive). Run order: see README.md.

## Notation
- q, p: draft / target probs at a prefix. pi_alpha: TokenV3 cascade target. alpha=1 -> pi=q; alpha=0 is NOT p.
- p_k(alpha): P(k-th child of an accepted node is the accepted one). G(T)=expected tokens per target step.
- Speedup = G * t_ar / Cost(T;H). Constraint: Quality(alpha) >= (1 - tol) * Quality(p). H (hardware) is an input.
- Cost(T;H) = t_verify(n) + sum over non-leaf levels of t_draft(width) [level 0 priced at width ceil(G)]
  + c0 + c1*G (c0, c1 fitted by calibrate.py from end-to-end runs; 0 before calibration).
- Tree = nested lists (node = list of children). Tree size n counts the root. Breadth-first flattening.

## Invariants (decoder, spec 8.4)
- Target cache holds all committed tokens except the last (the root); verify pass feeds root + n-1 drafted nodes.
- Draft cache holds tokens up to d_len; pending committed tokens are fed as one chunk next step, then cropped back.
- Position id = ctx + depth. Mask = full prefix + ancestors + self. keep_path after every step.
- Deferral test uses RAW target probs; blend uses temperature-scaled probs. Probabilities in float32 (verification in float64).
- `TreeDecoder.generate(..., debug=True)` asserts the cache lengths every step.

## Status
| Module | State |
|---|---|
| config, cascade, verify, dp, search, tasks, stats, paths | done; numpy parts tested (DP, search, offline pipeline) |
| accept (verify_batch, AcceptCounter) | written + CPU tests (test_accept.py); run them |
| quality (batched left-padded sampler, inline acceptance) | written + CPU tests (test_quality_cpu.py); run them |
| tree, decode_tree (TreeDecoder, autoregressive) | written + CPU tests (test_tree_decoder_cpu.py); run them |
| scripts: run_profile, check_tree_attention, run_quality, run_search, run_e2e, calibrate, make_figures, summarize | written; CPU-only ones (search, calibrate, figures, summarize) tested on synthetic artifacts |
| GPU results | NONE YET. Every number in the final report must come from these scripts. |

The torch modules were written without being executed (the authoring sandbox had no torch). The test files are the
contract: if a test fails, fix the module, not the test (unless the test itself is shown wrong).

## Decisions
- Platform 1: Colab T4 (fp16). Platform 2: Kaggle P100 (fp16), or T4 at fp32 labelled as such (spec 6.2).
- Pair A: pythia-160m -> pythia-2.8b (changed from 410m: in HF eager the draft's per-forward cost scales with its
  layer count, 24 for 410m vs 12 for 160m, so a 410m draft costs a large fraction of a 2.8b verify step on a T4).
  Pair A410 keeps the 410m draft for comparison. Check the profile: t_draft(1)/t_verify(1) is the number to watch.
- Pair B (second family, and the GSM8K pair): Qwen2.5-0.5B-Instruct -> Qwen2.5-3B-Instruct, chat template, zero-shot.
- Tasks: CNN/DM ROUGE-2 for pair A (calib = validation split, test = test split). GSM8K for pair B.
- Alpha grid: 0, .1, .2, .3, .5, .7, .9, 1.0. Quality tolerances: 2%, 5%, 10% of Q(p) (report all + Pareto).
- DP caps for the search: N=128, D=12, B=16. Temperature 1.0 (0.6 as an extra run if time permits).
- Library versions are recorded in every artifact (env). Pin them in requirements.txt after the first GPU run.

## Open items
- Run check_tree_attention.py on the real models (sdpa; eager if it fails).
- Positional assumption: compare 1-D p_1 with the depth matrix from calibrate.py; report the p_1 bimodality histogram.
- Spec's static-tree-wastes-siblings claim is a conjecture: summarize.py prints the measured share.
- Mid-term report Table I was not produced by this code (no quality/acceptance/e2e code existed then); replace it.
