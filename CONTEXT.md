# CONTEXT.md - Joint Cascade + Speculative-Tree Decoding (paste at the start of each session)

Spec: hdl_md.md. Code: jcsd/. Run tests from jcsd/ with `../.venv/bin/python -m pytest -q`.

## Notation
- q, p: draft / target probs at a prefix. pi_alpha: TokenV3 cascade target. alpha=1 -> pi=q; alpha=0 is NOT p.
- p_k(alpha): P(k-th child of an accepted node is the accepted one). G(T)=expected tokens per target step.
- Speedup = G * t_ar / Cost(T;H). Constraint: Quality(alpha) >= Q_min. H (hardware) is an input.
- Tree = nested lists (node = list of children). Tree size n counts the root. Breadth-first flattening.

## Invariants (decoder, spec 8.4)
- Target cache holds all committed tokens except the last (the root); verify pass feeds root + n-1 drafted nodes.
- Draft cache holds tokens up to d_len; pending committed tokens are fed as one chunk next step, then cropped back.
- Position id = ctx + depth. Mask = full prefix + ancestors + self. keep_path after every step.
- Deferral test uses RAW target probs; blend uses temperature-scaled probs. Probabilities in float32.

## Status
| Module | State |
|---|---|
| config, cascade, verify, dp, search, tasks | done, tested on CPU (47 tests) |
| models, profile_hw, scripts/run_profile.py | written; CPU smoke tests only. Run on GPU: notebooks/colab_profile.ipynb |
| quality, accept, tree, decode_tree, baselines, run_e2e | not started |

## Decisions
- Platform: Colab / Kaggle (T4 or P100 16 GB, fp16; T4 has no bf16). Pair A: pythia-410m (draft) / pythia-2.8b (target).
- Fallback pair B if Pythia rejects 4D masks: Qwen2.5-0.5B-Instruct / Qwen2.5-3B-Instruct.
- Second hardware profile (spec 6.2): different precision or CPU offload; label it as such via --tag.
- Library versions: record torch/transformers in every profile JSON (env_info does this). Pin after first GPU run.

## Open items
- Verify tree attention (8.3) on the real model before building decode_tree.
- Re-profile with --masked after that; compare to plain profile.
- Spec's static-tree-wastes-siblings claim is a conjecture: measure with visit logs.
