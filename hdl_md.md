# **Joint Cascade \+ Speculative-Tree Decoding: Full Project and Code Guide**

Hardware-aware joint optimization of the Faster Cascades deferral parameter alpha and the SEQUOIA token-tree topology.

Built from: Faster Cascades via Speculative Decoding (ICLR 2025), SEQUOIA (NeurIPS 2024), and your project proposal.

## **0\. How to use this document**

Parts 1 to 5 explain the project: what it claims, the exact math, and the system design. Part 6 onward is the build manual: environment, then one module at a time with code you can paste and run. Parts 11 and 12 show how to get an AI assistant to generate and debug the code module by module, plus a schedule.

Two honesty notes. First, the code here is scaffolding I wrote from the papers; it has not been executed, so every module comes with a test you must run before trusting it. Second, I have not seen the SEQUOIA repository (github.com/Infini-AI-Lab/Sequoia, cited in the paper). The plan therefore uses a self-contained implementation of tree verification and the DP, and uses the authors' repo only as a cross-check for baselines.

## **1\. Project at a glance**

### **1.1 One-paragraph summary**

LLM decoding is slow because each token needs a full read of the model weights. Speculative decoding hides this by having a small draft model propose tokens that the big target model verifies in parallel. SEQUOIA proposes a token tree and picks the tree shape (by dynamic programming) and size (by measuring hardware) that maximize speed while keeping the output identical to the target model. Faster Cascades relaxes the exactness requirement: the verifier targets a blend of the small and large model controlled by a parameter alpha, which accepts more drafts, runs faster, and sometimes even improves quality. Your project joins the two: alpha changes how often drafts are accepted, which changes the best tree, while the hardware sets the price of each tree. You optimize alpha and tree together, per hardware platform.

### **1.2 What each paper contributes and what it ignores**

| Component | Tunes | Ignores |
| ----- | ----- | ----- |
| Faster Cascades | deferral rule and alpha (quality vs rejection rate) | tree topology; hardware cost (uses gamma-length chains, and gamma=1 for Gemma) |
| SEQUOIA | tree shape by DP; tree size and depth by measured hardware cost | any quality trade-off; the target is always exactly the large model |
| Your project | alpha, tree, under a quality constraint, per hardware | (this is the contribution) |

### **1.3 Hypotheses to test**

* H1: the acceptance vector depends on alpha. Concretely, the probability that the first child is accepted rises with alpha.  
* H2: as alpha rises, the optimal tree becomes deeper and narrower; at small alpha it is wider.  
* H3: the optimal (alpha, tree) differs across hardware, and transplanting one platform's optimum to another loses measurable speedup.  
* H4: the joint optimum beats the sequential procedure (build the best tree for alpha=0, then tune alpha afterwards).

H4 is the one that justifies the project. If it fails, you still have a legitimate negative result: the coupling is weak for the model pairs and hardware you tested.

### **1.4 Deliverables**

1. A verified implementation of tree-based speculative sampling with a cascade target.  
2. Hardware cost profiles for each platform you can access.  
3. Quality curves Q(alpha) and acceptance vectors P(alpha).  
4. The joint search tool and its predicted-versus-measured validation.  
5. End-to-end comparison against autoregressive, chain speculative, Faster Cascades, SEQUOIA, and the sequential baseline.  
6. Hardware-transfer analysis.

## **2\. Background, self-contained**

### **2.1 Why decoding is memory-bound**

Generating one token requires loading all weights from GPU memory, and the arithmetic per loaded byte is tiny. The GPU therefore has spare compute. Processing, say, 64 tokens in one forward pass costs almost the same time as processing 1, until the pass becomes compute-bound. That spare capacity is what speculation spends.

### **2.2 Speculative decoding**

A draft model q proposes tokens x. The target model p scores all proposed positions in one pass. Each proposed token is accepted with probability min(1, p(x)/q(x)); at the first rejection a replacement token is sampled from the residual distribution, and later proposals are discarded. The output distribution is exactly p.

### **2.3 Token trees**

A chain wastes the pass whenever the first guess is wrong. A tree lets each position have several candidate children, all verified in one pass using a tree attention mask (each node attends only to its ancestors and the prefix). SEQUOIA samples children without replacement from q so the same wrong token is never proposed twice.

### **2.4 Cascades**

A cascade lets the small model answer when it is confident and defers to the big model otherwise. At token level, a deferral rule decides per step. The small model sometimes beats the large one on a given token, so a cascade can match or exceed the large model's quality at lower cost.

### **2.5 Speculative cascades**

Faster Cascades implements the deferral inside the verification step. Instead of comparing q against p, it compares q against a target pi built from both models. Because speculative sampling can target any distribution, the machinery stays exact with respect to pi. Parallel verification of drafts lets rules that need both q and p be used (they would be unusable in a sequential cascade, which would need to run p at every step).

## **3\. Precise problem formulation**

### **3.1 Objects**

* Vocabulary V; at a given prefix, q(v) and p(v) are the draft and target probabilities.  
* alpha in \[0,1): the deferral/leniency parameter of the chosen rule.  
* pi\_alpha(q, p): the cascade target distribution at that prefix.  
* A token tree T with n nodes and depth d, rooted at the current prefix.  
* p\_k(alpha): probability that the k-th child of an accepted node is the one accepted (k \= 1, 2, ...). These events are mutually exclusive, so the vector sums to at most 1\.  
* H: a hardware platform, which enters only through measured times.

### **3.2 The objective, corrected**

Your proposal states (alpha, T, H) \= argmax E\[tokens generated\] / measured execution cost(T, H). Two fixes are needed.

1. H is an input, not a decision variable. Solve the problem once per platform.  
2. Unconstrained, the ratio is maximized by pushing alpha to its leniency extreme: accept almost every draft, run fast, produce poor text. You need a quality constraint:

maximize   Speedup(alpha, T; H) \= G(T; alpha) / Cost(T; H)  
subject to Quality(alpha) \>= Q\_min

where G is expected tokens per target verification step and Cost is the wall-clock time of that step (verify \+ draft \+ overhead), normalized by one autoregressive target step. Faster Cascades makes the same move when it reports latency at matched large-model quality. Because the threshold Q\_min is a judgment call, also report the full Pareto frontier of (quality, speedup).

### **3.3 The decoupling property**

SEQUOIA's verification outputs a token distributed exactly as whatever target distribution it is given. Its correctness proof (Appendix F.2.1 of the paper) uses only that proposed tokens are valid samples from the current draft distribution; it never uses that the target is the large model. So with target pi\_alpha, the output distribution is pi\_alpha for every tree shape. Consequences:

* Quality depends on alpha only, and the tree affects speed only.  
* You can measure Quality(alpha) without any tree machinery, and you can sample from pi\_alpha directly when you only want quality.  
* The tree search for each alpha is a cheap offline computation (a DP plus table lookups).

The coupling is purely through p\_k(alpha): alpha reshapes the acceptance statistics, the statistics reshape the best tree.

### **3.4 Assumptions you should test, not trust**

* Positional acceptance (SEQUOIA Def. 3.1): acceptance depends only on child index k. Under a cascade the acceptance of the first child depends strongly on whether the draft token lies in the plausible set, so p\_1 is bimodal across contexts and its average hides this. Mitigation: use the depth-dependent matrix p\_{d,k}, and compare DP-predicted speedup with measured speedup.  
* Static trees under a cascade: if the first child is certain to be accepted in a context, its siblings are wasted work. This is my reading of the math, not a result from the papers; measure how often siblings are actually reached.  
* Plug-in confidence as a proxy for true loss (Faster Cascades Lemma 5): the deferral rule is only as good as the models are calibrated. The best rule may differ across model pairs.

## **4\. Math reference**

### **4.1 Total variation and rejection**

TV(p, q) \= sum\_v max(0, p(v) \- q(v)) \= 1 \- sum\_v min(p(v), q(v))

For target pi and draft q, a draft token x \~ q is accepted with probability min(1, pi(x)/q(x)). The rejection probability is TV(pi, q). On rejection, sample from norm(max(0, pi \- q)). The output is exactly pi because min(q, pi) \+ max(0, pi \- q) \= pi at every token.

### **4.2 TokenV3 cascade target**

The plausible set uses the large model's confidence on unscaled probabilities:

T\_alpha \= { v : p(v) \>= (1 \- alpha) \* max\_v' p(v') }  
pi(v)   \= q(v) \* 1\[v in T\_alpha\]  \+  p(v) \* sum\_{v' not in T\_alpha} q(v')

With temperature or top-P, the membership test uses the raw p, while the blend uses the scaled distributions S(q) and S(p) (Faster Cascades Appendix C.1). Under greedy decoding, TokenV3 reduces to Leviathan's lossy greedy criterion (their Lemma 7), which gives you a built-in test.

### **4.3 Verification at a tree node with k children**

R \= pi;  D \= q  
for child tokens s\_1..s\_k (drawn from D without replacement):  
    accept s\_i with prob min(1, R\[s\_i\] / D\[s\_i\])  
    on reject: R \= norm(max(R \- D, 0)); D\[s\_i\] \= 0; D \= norm(D)  
               (if D is all zero, D \= uniform over not-yet-proposed tokens)  
if all rejected: sample the extra token from R

### **4.4 Expected tokens of a tree**

With positional acceptance and acceptance vector p \= (p\_1, p\_2, ...):

f(root) \= 1;   f(v) \= product of p\_{k\_i} along the child indices on the path to v  
G(T)    \= sum over v in T of f(v)

Sanity check: a chain of length gamma with p\_1 \= a gives G \= 1 \+ a \+ ... \+ a^gamma, which is Leviathan et al.'s formula.

### **4.5 SEQUOIA dynamic program**

Let Tbl\[n, b\] be the best G among trees with n nodes whose root has exactly b children. Splitting at the last child (a subtree of m nodes that matters only if child b is accepted):

Tbl\[1, 0\] \= 1  
Tbl\[n, b\] \= max over 1 \<= m \<= n-1 of ( Tbl\[n-m, b-1\] \+ p\_b \* max\_j Tbl\[m, j\] )

With a depth index d (and optional depth-dependent acceptance P\[d, b\]) the same recursion bounds the tree depth. Cost is roughly O(N^2 \* B \* D), offline.

### **4.6 Cost model and speedup**

Cost(n, d; H) \= t\_verify(n) \+ t\_draft(tree) \+ t\_overhead(n)      (seconds)  
Speedup(n, d) \= G(n, d; alpha) \* t\_ar / Cost(n, d; H)

t\_ar is the time of one autoregressive target step (one token), so Speedup is in the same units as the papers. SEQUOIA's simplification t\_draft \= d \* c is a starting point; Part 7 measures the real draft cost.

### **4.7 The joint search**

for alpha in grid with Quality(alpha) \>= Q\_min:  
    P(alpha)   \= measured acceptance vector (or matrix)  
    for each (n, d): tree, G \= DP(P(alpha))  
        speedup(alpha, n, d) \= G \* t\_ar / Cost(n, d; H)  
    best(alpha) \= max over (n, d)  
answer \= argmax over alpha of best(alpha)

## **5\. System architecture**

### **5.1 Pipeline stages and artifacts**

stage 0  baselines            \-\> results/baselines\_{hw}.json  
stage 1  verification core    \-\> unit tests pass (no artifact)  
stage 2  hardware profile     \-\> profiles/{hw}\_{pair}.json    (t\_verify, t\_draft, memory)  
stage 3  quality curve        \-\> quality/{pair}\_{task}.json   (Q(alpha), CIs, token files)  
stage 4  acceptance vectors   \-\> accept/{pair}\_{task}\_a{alpha}.npy  
stage 5  joint search         \-\> search/{hw}\_{pair}.json      (best alpha, tree, predicted speedup)  
stage 6  end-to-end decoder   \-\> results/e2e\_{hw}.json       (measured)  
stage 7  analysis             \-\> figures, tables

Stages 3 and 4 are hardware independent (they depend on models and data only), so you run them once and reuse them across platforms. Stages 2, 5, 6 are per platform.

### **5.2 Repository layout**

jcsd/  
  README.md  
  requirements.txt  
  jcsd/  
    config.py            \# dataclasses: model pair, alpha grid, paths, seeds  
    models.py            \# load draft/target, tokenizer checks  
    cascade.py           \# TokenV3 target, temperature handling  
    verify.py            \# node verification (SEQUOIA sampling w/o replacement) for any target  
    dp.py                \# SEQUOIA dynamic program \+ tree reconstruction  
    tree.py              \# tree data structure, attention mask, path utilities  
    profile\_hw.py        \# t\_verify(n), t\_draft, memory  
    quality.py           \# sample from pi\_alpha, score tasks, bootstrap CIs  
    accept.py            \# acceptance vectors/matrices from token files  
    search.py            \# joint search  
    decode\_tree.py       \# end-to-end tree decoder with KV cache  
    baselines.py         \# autoregressive, chain speculative  
    tasks.py             \# dataset loading and metrics  
  tests/  
    test\_verify\_distribution.py  
    test\_dp.py  
    test\_cascade\_limits.py  
  scripts/  
    run\_profile.py  run\_quality.py  run\_accept.py  run\_search.py  run\_e2e.py  make\_figures.py

## **6\. Environment, models, data**

### **6.1 Install**

pip install torch transformers accelerate datasets evaluate rouge\_score sacrebleu numpy scipy pandas matplotlib tqdm pytest

Pin the versions you use in requirements.txt and record them in every results file. Tree attention uses a custom 4D attention mask, and support for this differs across transformers versions, so verify it on a tiny model first (Part 8 includes the check).

### **6.2 Hardware you can realistically get**

Colab and Kaggle availability changes and I cannot check what is offered today, so log the output of nvidia-smi with every run. Typical free options are a 16 GB T4 or P100, with larger GPUs on paid tiers. You need at least two distinct cost profiles for the transfer analysis. If you only get one GPU model, a second profile can come from running the target with weights offloaded to CPU (this is the memory-bound regime where SEQUOIA's trees grow to hundreds of nodes), or from a different precision (fp16 versus 8-bit). Treat these as different platforms only if you clearly say so.

### **6.3 Model pairs**

The pair must share a tokenizer, since drafts are token ids verified by the target. Memory check before choosing: fp16 weights take 2 bytes per parameter, so a 1.4B model needs about 2.8 GB, a 2.8B model about 5.6 GB, a 7B model about 14 GB (too tight on 16 GB once the KV cache and tree activations are added). Suggested starting pairs:

| Pair | Draft | Target | Notes |
| ----- | ----- | ----- | ----- |
| A (main) | Pythia-160M or 410M | Pythia-1.4B or 2.8B | SEQUOIA also uses Pythia for scaling plots; base models, so use few-shot prompts for tasks |
| B | Qwen2.5-0.5B-Instruct | Qwen2.5-3B-Instruct | instruction-tuned; fits a 16 GB GPU |
| C (optional) | any 7B-class pair | larger | only on bigger hardware or with offloading |

Verify with a one-line check that both tokenizers produce identical vocabularies. Some families pad the embedding matrix differently between sizes, so compare logits dimensions and slice to the common vocabulary if they differ, and record this.

### **6.4 Tasks and splits**

Quality must be scored against ground truth, not agreement with the target.

* Main pair of tasks: CNN/DailyMail summarization (ROUGE-2 or ROUGE-L) and GSM8K (exact match on the final answer).  
* Optional third: WMT En-De (BLEU) or MBPP (pass@1, which requires sandboxed execution and is the most work).  
* Speed-only profiling prompts: MT-Bench style prompts, as in SEQUOIA.

Use three disjoint subsets: a calibration split (about 200 prompts) for acceptance vectors and alpha selection, a test split (about 200 to 500\) for final reported numbers, and a small dev split (20) for debugging. Fix seeds and store them. Sampling at temperature 1 makes metrics noisy, and neither paper reports error bars, so plan bootstrap confidence intervals from the start.

### **6.5 Configuration object**

Keep every experimental knob in one dataclass so runs are reproducible and logged; Part 7 starts with it.

## **7\. Build manual: core modules**

Build in this order. Each module lists what it does, the code, and a done-when test. Do not start a module until the previous one passes its test, because later modules assume earlier ones are correct. All code uses single quotes and plain Python so it pastes cleanly; adapt names freely.

### **7.1 config.py**

One dataclass holds every knob. Save a JSON copy of it next to every result file.

from dataclasses import dataclass, asdict  
import json, os

@dataclass  
class Cfg:  
    draft: str \= 'EleutherAI/pythia-410m'  
    target: str \= 'EleutherAI/pythia-2.8b'  
    dtype: str \= 'float16'  
    device: str \= 'cuda'  
    temperature: float \= 1.0  
    alphas: tuple \= (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)  
    max\_new\_tokens: int \= 128  
    n\_calib: int \= 200  
    n\_test: int \= 300  
    max\_children: int \= 16      \# B in the DP  
    max\_tree: int \= 256         \# N in the DP  
    max\_depth: int \= 16         \# D in the DP  
    seed: int \= 0  
    out\_dir: str \= 'results'

    def save(self, name):  
        os.makedirs(self.out\_dir, exist\_ok=True)  
        with open(os.path.join(self.out\_dir, name), 'w') as f:  
            json.dump(asdict(self), f, indent=2)

Note on the alpha grid under TokenV3: alpha \= 1 makes the plausible set the whole vocabulary, so the target becomes the pure draft distribution q (fastest, lowest quality, the small-model end of the curve). alpha \= 0 keeps draft mass only on the target's single most likely token, which is strict but is not the same as the target p. Plain p is therefore a separate baseline, not a grid point.

### **7.2 models.py**

import torch  
from transformers import AutoModelForCausalLM, AutoTokenizer

def load\_pair(cfg):  
    dt \= getattr(torch, cfg.dtype)  
    tok \= AutoTokenizer.from\_pretrained(cfg.target)  
    tok\_d \= AutoTokenizer.from\_pretrained(cfg.draft)  
    assert tok.get\_vocab() \== tok\_d.get\_vocab(), 'tokenizers differ'  
    draft \= AutoModelForCausalLM.from\_pretrained(cfg.draft, torch\_dtype=dt).to(cfg.device).eval()  
    target \= AutoModelForCausalLM.from\_pretrained(cfg.target, torch\_dtype=dt).to(cfg.device).eval()  
    V \= min(draft.config.vocab\_size, target.config.vocab\_size)  
    return tok, draft, target, V

@torch.no\_grad()  
def logits\_of(model, ids, V, past=None):  
    out \= model(input\_ids=ids, past\_key\_values=past, use\_cache=True)  
    return out.logits\[..., :V\].float(), out.past\_key\_values

Done when: both models load, the tokenizer assertion passes, and a 32-token prompt gives logits of shape (1, 32, V) from both. If the vocab sizes differ, V slices both to the common range; write down that you did this.

### **7.3 cascade.py: the TokenV3 target**

Two inputs per distribution are needed: raw probabilities (softmax at temperature 1\) for the deferral test, and temperature-scaled probabilities for the blend, as in Faster Cascades Appendix C.1. Compute everything in float32 even when the models run in fp16, because ratios of small probabilities are used in acceptance tests.

import torch

def probs(logits, T=1.0):  
    return torch.softmax(logits.float() / T, dim=-1)

def tokenv3\_target(q\_raw, p\_raw, q\_s, p\_s, alpha):  
    \# all tensors \[..., V\]; works for a single node or a batch of nodes  
    thr \= (1.0 \- alpha) \* p\_raw.max(dim=-1, keepdim=True).values  
    in\_t \= (p\_raw \>= thr)  
    q\_out \= (q\_s \* (\~in\_t)).sum(dim=-1, keepdim=True)   \# draft mass on implausible tokens  
    return q\_s \* in\_t \+ p\_s \* q\_out

def plain\_target(q\_raw, p\_raw, q\_s, p\_s, alpha=None):  
    return p\_s

To use other deferral rules later (Chow, Diff, OPT), keep the same signature: any function that returns a normalized distribution can be dropped in, since the verification machinery accepts an arbitrary target.

Done when (tests/test\_cascade\_limits.py): the output sums to 1 within 1e-5 for random inputs; at alpha \= 1 the output equals q\_s; and with temperature 1 on a hand example it reproduces the numbers worked out earlier (q \= 0.6, 0.3, 0.1; p \= 0.4, 0.5, 0.1; alpha \= 0.3 gives 0.64, 0.35, 0.01).

import torch  
from jcsd.cascade import tokenv3\_target

def test\_worked\_example():  
    q \= torch.tensor(\[0.6, 0.3, 0.1\])  
    p \= torch.tensor(\[0.4, 0.5, 0.1\])  
    pi \= tokenv3\_target(q, p, q, p, 0.3)  
    assert torch.allclose(pi, torch.tensor(\[0.64, 0.35, 0.01\]), atol=1e-6)

def test\_alpha\_one\_is\_draft():  
    q \= torch.softmax(torch.randn(50), \-1); p \= torch.softmax(torch.randn(50), \-1)  
    assert torch.allclose(tokenv3\_target(q, p, q, p, 1.0), q, atol=1e-6)

### **7.4 verify.py: node verification for an arbitrary target**

This is the technical heart. It has two halves: drafting k children from q without replacement, and verifying them against the target pi. The draft distribution after j rejections is q with the j proposed tokens removed and renormalized, with a uniform-over-unproposed fallback if no mass remains (the cover property). Because the evolution of D depends only on the rejected tokens, which are exactly the earlier children, you can draw all k children up front at draft time and replay the same D sequence at verification time.

import torch

def \_next\_D(D, proposed, q):  
    if D.sum() \<= 0:  
        D \= torch.ones\_like(q)  
        if proposed:  
            D\[proposed\] \= 0  
    return D / D.sum()

def draft\_children(q, k, gen):  
    \# k distinct tokens sampled sequentially without replacement from q  
    D \= q.clone().double()  
    out \= \[\]  
    for \_ in range(k):  
        D \= \_next\_D(D, out, q.double())  
        s \= int(torch.multinomial(D, 1, generator=gen).item())  
        out.append(s)  
        D\[s\] \= 0  
    return out

def verify\_node(pi, q, children, gen):  
    \# returns (index of accepted child or \-1, token produced at this node)  
    R \= pi.clone().double()  
    D \= q.clone().double()  
    proposed \= \[\]  
    for i, s in enumerate(children):  
        D \= \_next\_D(D, proposed, q.double())  
        u \= torch.rand(1, generator=gen, dtype=torch.double).item()  
        ratio \= (R\[s\] / D\[s\]).item() if D\[s\] \> 0 else 0.0  
        if u \< min(1.0, ratio):  
            return i, s  
        R \= (R \- D).clamp\_min(0)  
        tot \= R.sum()  
        R \= R / tot if tot \> 1e-12 else pi.clone().double()  
        proposed.append(s)  
        D\[s\] \= 0  
    tok \= int(torch.multinomial(R, 1, generator=gen).item())  
    return \-1, tok

Two details worth understanding. After a rejection the residual is R \= norm(max(R \- D, 0)), where D is the draft distribution that was just used for that child (not the original q). The numerical guard on tot only fires when the target and draft coincide to machine precision, in which case the child would already have been accepted.

Done when (tests/test\_verify\_distribution.py), the empirical distribution of the produced token matches the target, for several k, several alpha, and including a draft with zero-probability tokens so the fallback path runs:

import torch, numpy as np, pytest  
from scipy.stats import chisquare  
from jcsd.verify import draft\_children, verify\_node  
from jcsd.cascade import tokenv3\_target

def run(k, alpha, q, p, n=30000, seed=0):  
    g \= torch.Generator().manual\_seed(seed)  
    pi \= tokenv3\_target(q, p, q, p, alpha).double()  
    counts \= np.zeros(len(q))  
    for \_ in range(n):  
        ch \= draft\_children(q, k, g)  
        \_, tok \= verify\_node(pi, q, ch, g)  
        counts\[tok\] \+= 1  
    exp \= pi.numpy() \* counts.sum()  
    mask \= exp \> 5          \# chi-square needs expected counts that are not tiny  
    cs \= counts\[mask\]; ex \= exp\[mask\]  
    ex \= ex \* cs.sum() / ex.sum()  
    return chisquare(cs, ex).pvalue

def rand\_dist(V, g, scale=1.5):  
    return torch.softmax(torch.randn(V, generator=g) \* scale, \-1)

@pytest.mark.parametrize('k', \[1, 2, 3, 5\])  
@pytest.mark.parametrize('alpha', \[0.0, 0.3, 0.8, 1.0\])  
def test\_output\_matches\_target(k, alpha):  
    g \= torch.Generator().manual\_seed(1)  
    q, p \= rand\_dist(8, g), rand\_dist(8, g)  
    assert run(k, alpha, q, p) \> 1e-3

def test\_zero\_support\_draft():  
    q \= torch.tensor(\[0.5, 0.5, 0, 0, 0, 0, 0, 0\])  
    g \= torch.Generator().manual\_seed(2)  
    p \= rand\_dist(8, g)  
    assert run(4, 0.2, q, p) \> 1e-3        \# k=4 exceeds the draft support

def test\_alpha\_one\_first\_child\_always\_accepted():  
    g \= torch.Generator().manual\_seed(3)  
    q, p \= rand\_dist(8, g), rand\_dist(8, g)  
    pi \= tokenv3\_target(q, p, q, p, 1.0).double()  
    for \_ in range(500):  
        ch \= draft\_children(q, 3, g)  
        i, \_ \= verify\_node(pi, q, ch, g)  
        assert i \== 0

A failing test here means a bug in the residual or fallback logic, and nothing downstream is trustworthy until it passes. A p-value of exactly 1e-3 is a reasonable failure threshold with this many parametrized cases, but if one case dips just below it, rerun with a different seed before concluding there is a bug; a real bug fails by orders of magnitude.

### **7.5 dp.py: SEQUOIA dynamic program**

I re-derived the DP rather than copying SEQUOIA's Algorithm 4, with a level index L so that depth-dependent acceptance rows are natural. State: W\[L\]\[n\]\[b\] is the best expected tokens for a subtree rooted at depth L with n nodes whose root has exactly b children; best\[L\]\[n\] is the maximum over b. A subtree at depth L has children at depth L+1, whose acceptance is acc\[L\]\[k\] (row L, column k \= child index). At the depth cap D, only a single node is allowed.

import numpy as np  
NEG \= \-1e18

def build\_dp(acc, N, D, B):  
    \# acc: 1-D vector p\_1..p\_K, or 2-D array shape (D, B+1) with column 0 unused  
    acc \= np.asarray(acc, dtype=float)  
    if acc.ndim \== 1:  
        row \= np.zeros(B \+ 1\)  
        m \= min(B, len(acc))  
        row\[1:m \+ 1\] \= acc\[:m\]  
        acc \= np.tile(row, (D, 1))  
    assert acc.shape \== (D, B \+ 1\)  
    W \= np.full((D \+ 1, N \+ 1, B \+ 1), NEG)  
    best \= np.full((D \+ 1, N \+ 1), NEG)  
    arg\_b \= np.zeros((D \+ 1, N \+ 1), dtype=int)  
    arg\_m \= np.zeros((D \+ 1, N \+ 1, B \+ 1), dtype=int)  
    for L in range(D, \-1, \-1):  
        W\[L, 1, 0\] \= 1.0  
        best\[L, 1\] \= 1.0  
        if L \== D:  
            continue  
        for n in range(2, N \+ 1):  
            ms \= np.arange(1, n)                 \# size of the last child's subtree  
            sub \= best\[L \+ 1, ms\]  
            for b in range(1, B \+ 1):  
                left \= W\[L, n \- ms, b \- 1\]  
                cand \= np.where((sub \> NEG / 2\) & (left \> NEG / 2),  
                                left \+ acc\[L, b\] \* sub, NEG)  
                j \= int(np.argmax(cand))  
                W\[L, n, b\] \= cand\[j\]  
                arg\_m\[L, n, b\] \= ms\[j\]  
            bb \= int(np.argmax(W\[L, n, :\]))  
            best\[L, n\] \= W\[L, n, bb\]  
            arg\_b\[L, n\] \= bb  
    return dict(W=W, best=best, arg\_b=arg\_b, arg\_m=arg\_m, D=D, B=B, N=N)

def get\_tree(dp, n, L=0):  
    \# returns nested lists: a node is the list of its children; a leaf is \[\]  
    def best\_tree(L, n):  
        return with\_b(L, n, int(dp\['arg\_b'\]\[L, n\]))  
    def with\_b(L, n, b):  
        kids, rem \= \[\], n  
        for bb in range(b, 0, \-1):  
            m \= int(dp\['arg\_m'\]\[L, rem, bb\])  
            kids.append(best\_tree(L \+ 1, m))  
            rem \-= m  
        kids.reverse()  
        return kids  
    return best\_tree(L, n)

def expected\_tokens(tree, acc\_row\_fn, L=0):  
    \# recompute G for a tree under acceptance acc\_row\_fn(L, k); used as an independent check  
    tot, stack \= 0.0, \[(tree, 1.0, L)\]  
    while stack:  
        node, f, l \= stack.pop()  
        tot \+= f  
        for k, ch in enumerate(node, start=1):  
            stack.append((ch, f \* acc\_row\_fn(l, k), l \+ 1))  
    return tot

For a one-dimensional acceptance vector every row is identical, so a single DP run with cap D also gives the best tree for any smaller depth d at no extra cost: the best tree of depth at most d with n nodes is best\[D \- d\]\[n\]. For a two-dimensional matrix this shortcut does not hold, and you re-run the DP for each depth cap you care about. Exact-size entries can be infeasible (for example, n larger than the cap allows), so in the search always take a running maximum over sizes up to n. Adding nodes can never lower the expected token count because all acceptance values are non-negative.

Done when (tests/test\_dp.py):

import numpy as np  
from jcsd.dp import build\_dp, get\_tree, expected\_tokens

def test\_chain():  
    a \= 0.8  
    dp \= build\_dp(\[a\], N=6, D=10, B=3)  
    assert abs(dp\['best'\]\[0, 6\] \- sum(a \*\* i for i in range(6))) \< 1e-9

def test\_paper\_vector\_small\_trees\_are\_chains():  
    \# acceptance vector from SEQUOIA Appendix E (Llama3-70B target, Llama3-8B draft, CNN/DM)  
    acc \= \[0.7732, 0.1039, 0.0402, 0.0206, 0.0128, 0.0081, 0.0064, 0.0043\]  
    dp \= build\_dp(acc, N=16, D=16, B=8)  
    t \= get\_tree(dp, 8\)  
    def max\_children(node):  
        return max(\[len(node)\] \+ \[max\_children(c) for c in node\])  
    assert max\_children(t) \== 1          \# their Figure 5a: the 8-node tree is a chain

def test\_recomputed\_value\_matches\_dp():  
    acc \= np.array(\[0.6, 0.2, 0.1, 0.05\])  
    dp \= build\_dp(acc, N=20, D=8, B=4)  
    t \= get\_tree(dp, 20\)  
    row \= lambda l, k: acc\[k \- 1\] if k \<= len(acc) else 0.0  
    assert abs(expected\_tokens(t, row) \- dp\['best'\]\[0, 20\]) \< 1e-9

The second test relies on my reading of SEQUOIA's Figure 5(a) and Appendix E, since I only saw the paper text and not their code; if it fails, compare the shapes for sizes 16 and 32 against their Figure 5(b) and 5(c) before suspecting your DP. The third test is the strongest: it recomputes the value of the reconstructed tree from scratch, so a bug in the reconstruction logic shows up immediately.

### **7.6 tasks.py: data, prompts, metrics**

Base models (Pythia) need few-shot prompts; instruction-tuned models can use their chat template instead. Whatever you choose, use the identical prompt builder for every method so only the decoding differs.

import numpy as np  
from datasets import load\_dataset

NL \= chr(10)

def last\_number(s):  
    s \= s.replace(',', '')  
    cur, found \= '', ''  
    for ch in s \+ ' ':  
        if ch.isdigit() or (ch \== '-' and cur \== '') or (ch \== '.' and cur and '.' not in cur):  
            cur \+= ch  
        else:  
            if cur and cur not in ('-', '.'):  
                found \= cur  
            cur \= ''  
    return found.rstrip('.')

def load\_task(name, split, n, seed=0):  
    if name \== 'cnndm':  
        ds \= load\_dataset('cnn\_dailymail', '3.0.0', split=split)  
    elif name \== 'gsm8k':  
        ds \= load\_dataset('openai/gsm8k', 'main', split=split)  
    ds \= ds.shuffle(seed=seed).select(range(n))  
    return \[dict(ex) for ex in ds\]

def make\_prompt(name, ex, shots):  
    if name \== 'cnndm':  
        art \= ' '.join(ex\['article'\].split()\[:300\])  
        return 'Article: ' \+ art \+ NL \+ 'Summary:'  
    parts \= \['Question: ' \+ s\['question'\] \+ NL \+ 'Answer: ' \+ s\['answer'\] for s in shots\]  
    parts.append('Question: ' \+ ex\['question'\] \+ NL \+ 'Answer:')  
    return (NL \+ NL).join(parts)

def clean\_output(name, text):  
    stop \= 'Article:' if name \== 'cnndm' else 'Question:'  
    return text.split(stop)\[0\].strip()

def score(name, preds, exs):  
    if name \== 'cnndm':  
        import evaluate  
        r \= evaluate.load('rouge')  
        out \= r.compute(predictions=preds, references=\[e\['highlights'\] for e in exs\], use\_aggregator=False)  
        return list(out\['rouge2'\])  
    gold \= \[e\['answer'\].split('\#\#\#\#')\[-1\].strip().replace(',', '') for e in exs\]  
    return \[1.0 if last\_number(p) \== g else 0.0 for p, g in zip(preds, gold)\]

def bootstrap\_ci(scores, n\_boot=2000, seed=0):  
    rng \= np.random.default\_rng(seed)  
    s \= np.asarray(scores, dtype=float)  
    means \= \[s\[rng.integers(0, len(s), len(s))\].mean() for \_ in range(n\_boot)\]  
    return float(s.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

For GSM8K use 4 training examples as shots (shots \= the first four of the train split, fixed forever) and max\_new\_tokens of about 256; for CNN/DM use about 64\. Base models at 1.4B to 2.8B will score low on GSM8K, so the absolute numbers will be small, which is fine as long as differences between methods are resolved by the confidence intervals; if every method scores near zero on GSM8K, replace it with a task the target can actually do.

Because the same prompts are used for every alpha, compare methods with a paired bootstrap on the per-example score differences, which is much tighter than comparing two independent intervals.

Done when: score() on the gold references returns near-perfect values (ROUGE of a text against itself is 1, and GSM8K gold answers parse to themselves).

### **7.7 profile\_hw.py: the hardware cost profile**

This stage produces the cost function that makes the whole optimization hardware-aware. Measurement rules: synchronize before and after timing, use CUDA events, discard warm-up iterations, report the median of at least 30 runs, and keep the raw samples. Shared Colab and Kaggle GPUs throttle and are noisy, so repeat across sessions and store the spread.

import torch, json, statistics, numpy as np

def cuda\_time(fn, warm=5, reps=30):  
    for \_ in range(warm):  
        fn()  
    torch.cuda.synchronize()  
    ts \= \[\]  
    for \_ in range(reps):  
        s \= torch.cuda.Event(enable\_timing=True)  
        e \= torch.cuda.Event(enable\_timing=True)  
        s.record(); fn(); e.record()  
        torch.cuda.synchronize()  
        ts.append(s.elapsed\_time(e) / 1000.0)  
    return statistics.median(ts), ts

@torch.no\_grad()  
def time\_forward(model, ctx\_len, n, device):  
    ids \= torch.randint(100, 1000, (1, ctx\_len), device=device)  
    cache \= model(input\_ids=ids, use\_cache=True).past\_key\_values  
    chunk \= torch.randint(100, 1000, (1, n), device=device)  
    def fn():  
        model(input\_ids=chunk, past\_key\_values=cache, use\_cache=True)  
        cache.crop(ctx\_len)  
    return cuda\_time(fn)

def profile(draft, target, cfg, ctx\_len=256, sizes=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)):  
    prof \= dict(hw=torch.cuda.get\_device\_name(), torch=torch.\_\_version\_\_,  
                ctx\_len=ctx\_len, sizes=list(sizes), t\_verify=\[\], t\_draft\_level=\[\],  
                peak\_mem\_gb=\[\], raw\_verify={})  
    for n in sizes:  
        torch.cuda.reset\_peak\_memory\_stats()  
        tv, raw \= time\_forward(target, ctx\_len, n, cfg.device)  
        td, \_ \= time\_forward(draft, ctx\_len, n, cfg.device)  
        prof\['t\_verify'\].append(tv)  
        prof\['t\_draft\_level'\].append(td)  
        prof\['peak\_mem\_gb'\].append(torch.cuda.max\_memory\_allocated() / 1e9)  
        prof\['raw\_verify'\]\[str(n)\] \= raw  
    prof\['t\_ar'\] \= prof\['t\_verify'\]\[0\]      \# one autoregressive target step  
    return prof

What the arrays mean: t\_verify\[i\] is the target's time to process sizes\[i\] tokens against a cached prefix, and t\_draft\_level\[i\] is the draft model's time to process sizes\[i\] tokens, which is what drafting one tree level of that width costs. Plot t\_verify against n on a log-x axis; you should see a flat region and then a rise, and the knee is the hardware's signature (compare to SEQUOIA Figure 8). Run it at two or three context lengths because attention cost grows with context.

Three caveats you must act on. First, the loop above times a plain causal chunk; tree verification uses a custom attention mask, which may take a different (slower) attention code path in your transformers version, so after Part 8 re-profile using the actual masked call. Second, the draft cache must also absorb the tokens accepted at each step, which the level model approximates as one width-1 forward; the end-to-end run will show if this matters. Third, verification overhead (computing the target for visited nodes, sampling without replacement, Python control flow) is not in these numbers; measure it as the difference between a full decode step and its two forward passes in your pilot run, and pass it to the search as a constant, in seconds.

Done when: t\_verify is non-decreasing within noise, t\_ar matches a hand timing of one generate step within about 10 percent, and repeated runs agree within a few percent.

### **7.8 quality.py: Q(alpha) without trees**

By the decoupling property, a sequence generated by speculative verification against pi\_alpha is distributed exactly as one sampled token by token from pi\_alpha. So to measure quality you do not need trees or speculative machinery at all: at every step compute q and p, form pi, and sample from it. The speculation only changes speed.

import torch  
from jcsd.models import logits\_of  
from jcsd.cascade import probs, tokenv3\_target

@torch.no\_grad()  
def sample\_from\_target(draft, target, V, prompt\_ids, cfg, alpha, gen, eos\_id, mode='cascade'):  
    \# mode: 'cascade' (pi\_alpha), 'p' (target only), 'q' (draft only)  
    dev \= cfg.device  
    ids \= prompt\_ids.to(dev)  
    lq, pq \= logits\_of(draft, ids, V)  
    lp, pp \= logits\_of(target, ids, V)  
    out \= \[\]  
    for \_ in range(cfg.max\_new\_tokens):  
        q\_raw, p\_raw \= probs(lq\[0, \-1\]), probs(lp\[0, \-1\])  
        q\_s, p\_s \= probs(lq\[0, \-1\], cfg.temperature), probs(lp\[0, \-1\], cfg.temperature)  
        if mode \== 'p':  
            pi \= p\_s  
        elif mode \== 'q':  
            pi \= q\_s  
        else:  
            pi \= tokenv3\_target(q\_raw, p\_raw, q\_s, p\_s, alpha)  
        tok \= int(torch.multinomial(pi, 1, generator=gen).item())  
        out.append(tok)  
        if tok \== eos\_id:  
            break  
        t \= torch.tensor(\[\[tok\]\], device=dev)  
        lq, pq \= logits\_of(draft, t, V, pq)  
        lp, pp \= logits\_of(target, t, V, pp)  
    return out

Create the generator with torch.Generator(device='cuda').manual\_seed(seed \+ prompt\_index) so each prompt gets a reproducible stream.

The driver script (scripts/run\_quality.py) loops over modes and alphas: for mode in \['q', 'p'\] plus each alpha in cfg.alphas, generate for all calibration prompts, decode, clean\_output, score, and store the per-example scores, the aggregate with a bootstrap CI, and the token sequences (prompt ids plus generated ids) to disk, because the acceptance stage reuses them. Expect this to be the slowest stage: roughly two small-batch forward passes per token, for n\_prompts times max\_new\_tokens times the number of settings. As a rough estimate, 100 prompts at 128 tokens for 11 settings is about 140,000 steps; at tens of milliseconds per step that is a few hours on a free GPU. Cut it down by using 100 calibration prompts, a coarser alpha grid (6 values), and by batching prompts with left padding if you have the patience to handle attention masks and position ids.

What to look for: the curve Q(alpha) between the draft-only and target-only endpoints. Faster Cascades found that moderate settings can exceed the large model, but that is a finding in their setups; do not assume it holds for yours. The feasible set is every alpha whose score is within your tolerance of the target (for example, the lower confidence bound at least 99 percent of the target's mean), and it need not be an interval.

Done when: mode 'p' reproduces the quality of ordinary sampling from the target using the same prompts (up to noise), and mode 'q' is clearly worse, which confirms the metric can see the difference at all.

### **7.9 accept.py: acceptance vectors P(alpha)**

Goal: for each alpha, estimate p\_k \= probability that the k-th child of an accepted node is the accepted one. Method: take the sequences generated under pi\_alpha in the quality stage, run both models over each complete sequence in a single teacher-forced pass to get q and p at every generated position, and at each position simulate one node verification with B children.

import torch, numpy as np  
from jcsd.cascade import probs, tokenv3\_target  
from jcsd.verify import draft\_children, verify\_node

@torch.no\_grad()  
def acceptance\_vector(draft, target, V, seqs, cfg, alpha, B=16, sims=4, mode='cascade', seed=0):  
    \# seqs: list of (prompt\_len, token\_id\_list)  
    g \= torch.Generator().manual\_seed(seed)  
    counts \= np.zeros(B \+ 1\)           \# index 0..B-1 \= child accepted, B \= all rejected  
    p1\_analytic \= \[\]                    \# per-position P(first child accepted) \= 1 \- TV(pi, q)  
    for plen, toks in seqs:  
        ids \= torch.tensor(\[toks\], device=cfg.device)  
        lq \= draft(input\_ids=ids).logits\[0, :, :V\].float()  
        lp \= target(input\_ids=ids).logits\[0, :, :V\].float()  
        for t in range(plen \- 1, len(toks) \- 1):      \# positions predicting a generated token  
            q\_raw, p\_raw \= probs(lq\[t\]), probs(lp\[t\])  
            q\_s, p\_s \= probs(lq\[t\], cfg.temperature), probs(lp\[t\], cfg.temperature)  
            pi \= p\_s if mode \== 'p' else tokenv3\_target(q\_raw, p\_raw, q\_s, p\_s, alpha)  
            q\_c, pi\_c \= q\_s.double().cpu(), pi.double().cpu()  
            p1\_analytic.append(float(torch.minimum(q\_c, pi\_c).sum()))  
            for \_ in range(sims):  
                ch \= draft\_children(q\_c, B, g)  
                i, \_tok \= verify\_node(pi\_c, q\_c, ch, g)  
                counts\[i if i \>= 0 else B\] \+= 1  
    pk \= counts\[:B\] / counts.sum()  
    return pk, float(counts\[B\] / counts.sum()), np.array(p1\_analytic)

The function returns the acceptance vector (length B, entry k-1 is p\_k), the all-rejected probability, and the per-position first-child acceptance probabilities. Plot a histogram of the last one for several alphas: if it piles up near 1 and near a low value, p\_1 is bimodal, which is the evidence that the positional assumption is under strain (Part 3.4). Save the vector as accept/{pair}\_{task}\_a{alpha}.npy and also the plain-target vector (mode 'p') as the reference for the sequential baseline.

If this runs too slowly (it works on CPU tensors over a 50k vocabulary), reduce sims, restrict to a random subset of positions, or truncate to the top 2000 tokens of q and renormalize while documenting it. Sampling error on small entries such as p\_8 is large; since the DP's gains from deep branches are small, it is acceptable to smooth the tail with a power-law fit, but then say so, and compare results with and without smoothing.

Done when: p\_1 increases with alpha (H1), the entries sum to one minus the all-rejected probability, and at alpha \= 1 the target equals q so p\_1 is exactly 1 and every other entry is 0\.

Depth-dependent acceptance cannot be estimated by this teacher-forced procedure, because a node deeper in a real tree sits on a drafted continuation that differs from the stored text. Estimate p\_{d,k} instead from decoder logs: during the end-to-end run (Part 8\) record the depth and child index of every accepted node, rebuild the matrix, re-run the DP, and iterate once or twice.

### **7.10 search.py: the joint optimizer**

Inputs: acceptance vectors per alpha, quality per alpha, the quality threshold, and the hardware profile. For each feasible alpha it builds the DP, scans every (n, d), computes the cost of the reconstructed tree from its per-level widths, and keeps the best speedup. It also implements the sequential baseline so H4 can be tested.

import numpy as np  
from jcsd.dp import build\_dp, get\_tree, expected\_tokens

def level\_widths(tree):  
    widths, cur \= \[1\], \[tree\]  
    while True:  
        nxt \= \[c for node in cur for c in node\]  
        if not nxt:  
            break  
        widths.append(len(nxt))  
        cur \= nxt  
    return widths

def make\_cost(prof, overhead=0.0):  
    fv \= lambda n: float(np.interp(n, prof\['sizes'\], prof\['t\_verify'\]))  
    fd \= lambda w: float(np.interp(w, prof\['sizes'\], prof\['t\_draft\_level'\]))  
    def cost(widths):  
        n \= sum(widths)  
        draft \= sum(fd(w) for w in widths\[:-1\])   \# the deepest level needs no further drafting  
        return fv(n) \+ draft \+ overhead  
    return cost

def best\_for\_acc(acc, cfg, prof, overhead=0.0):  
    N, D, B \= cfg.max\_tree, cfg.max\_depth, cfg.max\_children  
    dp \= build\_dp(acc, N, D, B)       \# valid for every depth cap because acc is 1-D  
    cost \= make\_cost(prof, overhead)  
    best \= dict(speedup=0.0)  
    for d in range(1, D \+ 1):  
        for n in range(1, N \+ 1):  
            G \= dp\['best'\]\[D \- d, n\]  
            if G \< 0:  
                continue  
            tree \= get\_tree(dp, n, L=D \- d)  
            w \= level\_widths(tree)  
            sp \= G \* prof\['t\_ar'\] / cost(w)  
            if sp \> best\['speedup'\]:  
                best \= dict(speedup=float(sp), n=n, d=len(w) \- 1, G=float(G), widths=w, tree=tree)  
    return best

def joint\_search(acc\_by\_alpha, q\_by\_alpha, q\_min, cfg, prof, overhead=0.0):  
    res \= {}  
    for a, acc in acc\_by\_alpha.items():  
        if q\_by\_alpha\[a\] \>= q\_min:  
            res\[a\] \= best\_for\_acc(acc, cfg, prof, overhead)  
    a\_star \= max(res, key=lambda a: res\[a\]\['speedup'\])  
    return a\_star, res

def sequential\_search(acc\_ref, acc\_by\_alpha, q\_by\_alpha, q\_min, cfg, prof, overhead=0.0):  
    \# step 1: best tree for the exact target; step 2: tune alpha for that fixed tree  
    ref \= best\_for\_acc(acc\_ref, cfg, prof, overhead)  
    cost \= make\_cost(prof, overhead)(ref\['widths'\])  
    out \= {}  
    for a, acc in acc\_by\_alpha.items():  
        if q\_by\_alpha\[a\] \< q\_min:  
            continue  
        row \= lambda l, k, acc=acc: acc\[k \- 1\] if k \<= len(acc) else 0.0  
        G \= expected\_tokens(ref\['tree'\], row)  
        out\[a\] \= G \* prof\['t\_ar'\] / cost  
    a\_seq \= max(out, key=out.get)  
    return a\_seq, out\[a\_seq\], ref

Notes. The d reported is the realized tree depth, which can be smaller than the cap. Tree reconstruction for every (n, d) is the slow part (about N times D reconstructions per alpha); it is fine for N \= 256, D \= 16, but if you push N to 768 for the offloading regime, restrict the scan to n values on a coarse grid. The search runs offline and uses no GPU.

To answer H4, compare joint\_search's best predicted speedup with sequential\_search's. If they agree to within the prediction error, the coupling is weak. Predicted gaps smaller than the typical prediction error from Part 9.2 are not evidence of anything.

Done when: with a synthetic profile (t\_verify flat, t\_draft proportional to width) and a hand-made acceptance vector, the returned tree and speedup can be checked by hand; and at alpha \= 1 with p\_1 \= 1 the best tree is a single chain of the maximum useful depth.

## **8\. Build manual: the end-to-end tree decoder**

The offline pipeline (Part 7\) predicts a speedup. This part builds the real decoder that measures it. It is the hardest engineering in the project: a tree attention mask, per-layer KV-cache surgery, and a verification walk. Take it in the sub-steps below and test each one.

### **8.1 Preparatory edit to verify.py**

The decoder runs on the GPU with a CUDA generator, so make the random draw in verify\_node device-aware. Change the line that draws u to:

       u \= torch.rand(1, generator=gen, device=R.device, dtype=torch.double).item()

The CPU unit tests still pass because there R.device is the CPU. Also add this batched Gumbel top-k sampler to verify.py. Sampling k tokens without replacement is exactly equivalent to taking the top k of (log q \+ Gumbel noise), and giving zero-probability tokens a large negative offset keeps them in uniform random order after all supported tokens, which reproduces the uniform fallback:

def gumbel\_topk(qmat, k, gen):  
    \# qmat: \[m, V\]. Returns \[m, k\] token ids in sequential-without-replacement order.  
    u \= torch.rand(qmat.shape, generator=gen, device=qmat.device).clamp\_min(1e-20)  
    g \= \-torch.log(-torch.log(u))  
    key \= torch.where(qmat \> 0, torch.log(qmat.clamp\_min(1e-30)) \+ g, g \- 1e4)  
    return key.topk(k, dim=-1).indices

Test it: draw 50,000 samples of the first and second child from a fixed q and compare to draft\_children (first-child frequencies should match q; second-child frequencies should match the pair statistics of the loop-based sampler).

### **8.2 tree.py: flattening, masks, cache surgery**

The tree from the DP is a nested list. Flatten it in breadth-first order so node indices are contiguous by level. Node 0 is the root, which is the last committed token (its KV is computed in the verification pass, so it counts toward the tree size n, matching SEQUOIA's size accounting).

import torch

def flatten(tree):  
    par, dep, rank, kids \= \[-1\], \[0\], \[0\], \[\[\]\]  
    order, i \= \[(tree, 0)\], 0  
    while i \< len(order):  
        node, me \= order\[i\]  
        i \+= 1  
        for k, ch in enumerate(node, start=1):  
            par.append(me); dep.append(dep\[me\] \+ 1); rank.append(k); kids.append(\[\])  
            kids\[me\].append(len(par) \- 1\)  
            order.append((ch, len(par) \- 1))  
    return par, dep, rank, kids

def keep\_path(cache, ctx, path):  
    \# after a verification pass appended n entries after ctx cached ones, keep only the  
    \# root and accepted nodes (indices in 'path', root first), moved to consecutive slots.  
    layers \= cache.layers if hasattr(cache, 'layers') else None  
    n\_layers \= len(layers) if layers is not None else len(cache.key\_cache)  
    sel \= None  
    for i in range(n\_layers):  
        if layers is not None:  
            k, v \= layers\[i\].keys, layers\[i\].values  
        else:  
            k, v \= cache.key\_cache\[i\], cache.value\_cache\[i\]  
        if sel is None:  
            sel \= torch.tensor(\[ctx \+ j for j in path\], device=k.device)  
        k2 \= torch.cat(\[k\[:, :, :ctx\], k.index\_select(2, sel)\], dim=2)  
        v2 \= torch.cat(\[v\[:, :, :ctx\], v.index\_select(2, sel)\], dim=2)  
        if layers is not None:  
            layers\[i\].keys, layers\[i\].values \= k2, v2  
        else:  
            cache.key\_cache\[i\], cache.value\_cache\[i\] \= k2, v2

The KV-cache API changed between transformers releases (a per-layer list versus key\_cache and value\_cache lists), which is why keep\_path handles both; check which your installed version exposes and delete the branch you do not need. Cached keys already contain their rotary position encoding, so moving an entry is valid only if its position id equals the slot it moves to. That holds here: a node at depth j gets position ctx \+ j, and if it is on the accepted path it lands in slot ctx \+ j.

### **8.3 Validate tree attention before anything else**

This test decides whether the whole approach works on your model and library version. Pass a small tree through the target with a 4D mask and explicit position ids, and compare each node's logits with a plain causal forward over the prefix plus that node's root-to-node path.

import torch  
from jcsd.tree import flatten

@torch.no\_grad()  
def check\_tree\_attention(target, tree, ctx\_len, V, device, tol=2e-2):  
    par, dep, rank, kids \= flatten(tree)  
    n \= len(par)  
    prefix \= torch.randint(100, 1000, (1, ctx\_len), device=device)  
    node\_tok \= torch.randint(100, 1000, (n,), device=device)  
    cache \= target(input\_ids=prefix\[:, :-1\], use\_cache=True).past\_key\_values  
    allow \= torch.zeros(n, n, dtype=torch.bool)  
    for i in range(n):  
        j \= i  
        while j \!= \-1:  
            allow\[i, j\] \= True  
            j \= par\[j\]  
    full \= torch.cat(\[torch.ones(n, ctx\_len \- 1, dtype=torch.bool), allow\], dim=1).to(device)  
    dt \= next(target.parameters()).dtype  
    mask \= torch.zeros(full.shape, dtype=dt, device=device)  
    mask.masked\_fill\_(\~full, torch.finfo(dt).min)  
    inp \= torch.cat(\[prefix\[0, \-1:\], node\_tok\[1:\]\])\[None\]  
    pos \= torch.tensor(\[\[ctx\_len \- 1 \+ d for d in dep\]\], device=device)  
    out \= target(input\_ids=inp, position\_ids=pos, attention\_mask=mask\[None, None\],  
                 past\_key\_values=cache, use\_cache=True).logits\[0, :, :V\].float()  
    worst \= 0.0  
    for i in range(n):  
        path, j \= \[\], i  
        while j \> 0:  
            path.append(int(node\_tok\[j\])); j \= par\[j\]  
        seq \= torch.cat(\[prefix, torch.tensor(\[path\[::-1\]\], device=device, dtype=prefix.dtype)\], dim=1) if path else prefix  
        ref \= target(input\_ids=seq).logits\[0, \-1, :V\].float()  
        worst \= max(worst, (out\[i\] \- ref).abs().max().item())  
    return worst \< tol, worst

Run it with a tree that has branching and depth of at least 3\. In fp16 expect small differences (the 2e-2 tolerance on raw logits is a starting guess; compare it to the difference between two plain forward passes at different batch shapes to calibrate). A large mismatch means one of: the model class ignores 4D masks, the attention implementation (sdpa versus eager) handles them differently, or position ids were not applied. Try attn\_implementation='eager' first when debugging. If your chosen model family cannot take a custom mask in your version, switch to a model family that can (Llama and Qwen architectures are the usual safe choices) rather than patching library internals.

Once this passes, rerun the Part 7.7 profile using the masked call so t\_verify reflects the real code path.

### **8.4 decode\_tree.py: the decoder**

Invariants, which make the bookkeeping manageable:

* Target cache holds every committed token except the last one (the root). The verification pass feeds the root plus the n-1 drafted nodes.  
* Draft cache holds committed tokens up to some index d\_len; the tokens committed since then are pending and are fed as one chunk at the start of the next step (this chunk is the draft's level-0 forward, so its width is about the number of tokens just accepted, not 1).  
* After drafting, the draft cache is cropped back to d\_len. After verification, keep\_path keeps only the root and accepted nodes in the target cache.  
* TokenV3 uses only the target's raw probabilities for the deferral test, so the draft needs only its temperature-scaled distribution per node.

import torch  
from jcsd.cascade import probs, tokenv3\_target  
from jcsd.verify import verify\_node, gumbel\_topk  
from jcsd.tree import flatten, keep\_path

def chain(gamma):  
    t \= \[\]  
    for \_ in range(gamma):  
        t \= \[t\]  
    return t

class TreeDecoder:  
    def \_\_init\_\_(self, draft, target, V, cfg, alpha, tree, mode='cascade'):  
        self.draft, self.target, self.V, self.cfg \= draft, target, V, cfg  
        self.alpha, self.mode \= alpha, mode  
        self.par, self.dep, self.rank, self.kids \= flatten(tree)  
        self.n, dev \= len(self.par), cfg.device  
        self.depth \= max(self.dep)  
        self.tdt \= next(target.parameters()).dtype  
        self.ddt \= next(draft.parameters()).dtype  
        self.levels \= {}  
        for i, d in enumerate(self.dep):  
            self.levels.setdefault(d, \[\]).append(i)  
        self.lv\_idx \= {l: torch.tensor(v, device=dev) for l, v in self.levels.items()}  
        allow \= torch.zeros(self.n, self.n, dtype=torch.bool)  
        for i in range(self.n):  
            j \= i  
            while j \!= \-1:  
                allow\[i, j\] \= True  
                j \= self.par\[j\]  
        self.allow \= allow.to(dev)  
        self.lv\_allow, self.assign \= {}, {}  
        for l in range(0, self.depth):  
            idx \= self.levels\[l\]  
            rows, ranks, childs \= \[\], \[\], \[\]  
            for r, i in enumerate(idx):  
                for k, c in enumerate(self.kids\[i\]):  
                    rows.append(r); ranks.append(k); childs.append(c)  
            kmax \= max(len(self.kids\[i\]) for i in idx)  
            self.assign\[l\] \= (torch.tensor(rows, device=dev), torch.tensor(ranks, device=dev),  
                              torch.tensor(childs, device=dev), kmax)  
            if l \>= 1:  
                lo, m \= idx\[0\], len(idx)  
                a \= torch.zeros(m, lo \- 1 \+ m, dtype=torch.bool)  
                for r, i in enumerate(idx):  
                    j \= i  
                    while j \> 0:  
                        a\[r, j \- 1\] \= True  
                        j \= self.par\[j\]  
                self.lv\_allow\[l\] \= a.to(dev)

    def \_mask(self, allow, ctx, dtype):  
        m \= allow.shape\[0\]  
        full \= torch.cat(\[torch.ones(m, ctx, dtype=torch.bool, device=allow.device), allow\], dim=1)  
        out \= torch.zeros(full.shape, dtype=dtype, device=full.device)  
        out.masked\_fill\_(\~full, torch.finfo(dtype).min)  
        return out\[None, None\]

    @torch.no\_grad()  
    def generate(self, prompt\_ids, gen, eos\_id, max\_new):  
        dev, T, V, n \= self.cfg.device, self.cfg.temperature, self.V, self.n  
        committed \= prompt\_ids\[0\].tolist()  
        plen \= len(committed)  
        tcache \= self.target(input\_ids=prompt\_ids\[:, :-1\].to(dev), use\_cache=True).past\_key\_values  
        dcache, d\_len \= None, 0  
        log \= dict(steps=0, new\_per\_step=\[\], visits=\[\])  
        while len(committed) \- plen \< max\_new:  
            \# \---- draft: pending chunk, then one forward per tree level \----  
            pend \= torch.tensor(\[committed\[d\_len:\]\], device=dev)  
            o \= self.draft(input\_ids=pend, past\_key\_values=dcache, use\_cache=True)  
            dcache, d\_len \= o.past\_key\_values, len(committed)  
            qs \= torch.zeros(n, V, device=dev)  
            qs\[0\] \= probs(o.logits\[0, \-1, :V\], T)  
            toks \= torch.zeros(n, dtype=torch.long, device=dev)  
            toks\[0\] \= committed\[-1\]  
            for l in range(0, self.depth):  
                idx \= self.lv\_idx\[l\]  
                if l \>= 1:  
                    pos \= torch.full((1, len(idx)), d\_len \- 1 \+ l, device=dev)  
                    mask \= self.\_mask(self.lv\_allow\[l\], d\_len, self.ddt)  
                    o \= self.draft(input\_ids=toks\[idx\]\[None\], position\_ids=pos, attention\_mask=mask,  
                                   past\_key\_values=dcache, use\_cache=True)  
                    dcache \= o.past\_key\_values  
                    qs\[idx\] \= probs(o.logits\[0, :, :V\], T)  
                rows, ranks, childs, kmax \= self.assign\[l\]  
                top \= gumbel\_topk(qs\[idx\], kmax, gen)  
                toks\[childs\] \= top\[rows, ranks\]  
            dcache.crop(d\_len)  
            \# \---- verify: one target pass over the whole tree \----  
            ctx\_t \= len(committed) \- 1  
            pos \= torch.tensor(\[\[ctx\_t \+ d for d in self.dep\]\], device=dev)  
            mask \= self.\_mask(self.allow, ctx\_t, self.tdt)  
            o \= self.target(input\_ids=toks\[None\], position\_ids=pos, attention\_mask=mask,  
                            past\_key\_values=tcache, use\_cache=True)  
            tcache, lp \= o.past\_key\_values, o.logits\[0, :, :V\]  
            tl \= toks.tolist()  
            node, path, new \= 0, \[0\], \[\]  
            while True:  
                p\_raw, p\_s \= probs(lp\[node\]), probs(lp\[node\], T)  
                pi \= p\_s if self.mode \== 'p' else tokenv3\_target(qs\[node\], p\_raw, qs\[node\], p\_s, self.alpha)  
                kid \= self.kids\[node\]  
                i, tok \= verify\_node(pi, qs\[node\], \[tl\[c\] for c in kid\], gen)  
                if len(kid) \> 0:  
                    log\['visits'\].append((self.dep\[node\], len(kid), i))  
                if i \>= 0:  
                    node \= kid\[i\]; path.append(node); new.append(tok)  
                else:  
                    new.append(tok)  
                    break  
            keep\_path(tcache, ctx\_t, path)  
            committed \+= new  
            log\['steps'\] \+= 1  
            log\['new\_per\_step'\].append(len(new))  
            if eos\_id in new:  
                committed \= committed\[:plen \+ len(committed\[plen:\]) \- (len(new) \- new.index(eos\_id) \- 1)\]  
                break  
        return committed\[plen:\], log

Visit logs record (depth, number of children, accepted child index or \-1) for every visited node that had children. Those are the raw material for the depth-dependent acceptance matrix: p\_{d,k} is the fraction of visits at depth d, among visits whose node had at least k children, in which child k was accepted. It is estimated only for ranks that actually appear in the tree; for ranks absent from the tree, fall back to the teacher-forced values from Part 7.9.

### **8.5 Baselines through the same code**

Running every method through the same decoder keeps the comparison fair and means any Python overhead affects all methods alike.

* Standard chain speculative decoding: TreeDecoder with tree \= chain(gamma) and mode \= 'p', for gamma in 3, 5, 7\.  
* SEQUOIA-style: TreeDecoder with mode \= 'p' and the tree from best\_for\_acc applied to the plain-target acceptance vector.  
* Faster Cascades (chain): chain(gamma) with mode \= 'cascade'. For a chain this is the same algorithm as their Algorithm 5; cross-check the quality and rejection-rate trend against their paper's plots or the Colab notebook linked in their paper.  
* Autoregressive: one-token-at-a-time sampling from the target with the same cache handling.

@torch.no\_grad()  
def autoregressive(target, V, prompt\_ids, cfg, gen, eos\_id, max\_new):  
    dev, T \= cfg.device, cfg.temperature  
    out \= target(input\_ids=prompt\_ids.to(dev), use\_cache=True)  
    cache, logits, res \= out.past\_key\_values, out.logits\[0, \-1, :V\], \[\]  
    for \_ in range(max\_new):  
        tok \= int(torch.multinomial(probs(logits, T), 1, generator=gen).item())  
        res.append(tok)  
        if tok \== eos\_id:  
            break  
        o \= target(input\_ids=torch.tensor(\[\[tok\]\], device=dev), past\_key\_values=cache, use\_cache=True)  
        cache, logits \= o.past\_key\_values, o.logits\[0, \-1, :V\]  
    return res

### **8.6 Correctness test for the whole decoder**

Exactness is the property that justifies the quality-versus-speed separation, so test it end to end. Fix one prompt, generate only one new token many times with different seeds, and compare the histogram of that first token to the distribution it should follow: p\_s at the prompt for mode 'p', and pi\_alpha at the prompt for mode 'cascade'. Use a tree with branching (for example, root with 3 children, the first of which has 2). A few thousand runs on a tokenizer's most likely tokens (lump the rest into an 'other' bin) with a chi-square test is enough. Failures at this test but passes at 7.4 point to bookkeeping errors (token-to-child assignment, position ids, or cache surgery) rather than to the sampling logic.

A second check: with the tree reduced to the root alone (n \= 1\) and mode \= 'p', the decoder must reduce to plain autoregressive sampling. A third: with alpha \= 1 in cascade mode the target is q, so the first child should be accepted every time (visits with accepted index 0 at depth 0 should be 100 percent).

### **8.7 Running the end-to-end benchmark**

scripts/run\_e2e.py: for each method, load its tree and alpha, run over the test prompts at batch size 1, and for each prompt record the wall-clock time (synchronize before starting the timer and after the last token), the number of generated tokens, the step count, and the visit log; score quality from the decoded text. Interleave methods across prompts, or at least repeat the whole sweep in at least three passes in alternating order, so that throttling and neighbor noise hit all methods equally. Report ms per token, speedup relative to the autoregressive run from the same session, tokens per target call (new tokens divided by steps), acceptance rate, peak memory, and quality with a bootstrap interval.

Absolute speedups from this Python decoder will be below those in the papers, which use CUDA graphs and optimized kernels. Do not compare your absolute numbers to theirs; compare your own methods to each other.

## **9\. Experiments and analysis**

### **9.1 Experimental matrix**

| Factor | Levels |
| ----- | ----- |
| Method | autoregressive; chain speculative (gamma 3, 5, 7); SEQUOIA-style (mode p, DP tree); Faster Cascades chain (best alpha under the quality constraint); sequential baseline; joint (yours) |
| Model pair | A (main), B (second family) |
| Hardware | at least two distinct cost profiles |
| Task | CNN/DM, GSM8K (or your replacement) |
| Temperature | 0.6 and 1.0 (SEQUOIA picks deeper trees at lower temperature) |

Measured per run: ms per token, speedup versus autoregressive in the same session, tokens per target call, acceptance rate, target-model calls, peak memory, quality with a bootstrap interval. Use batch size 1 for all main results, as both papers do; state clearly that conclusions may not transfer to batched serving, where the cost curve t(n) changes.

### **9.2 Validating the cost model (prediction versus measurement)**

For each candidate (alpha, tree) from the search, compare predicted speedup with measured speedup. Report the relative error. If the error is large (a reasonable bar is within 15 to 20 percent, though that is my rule of thumb rather than a standard), fix the model before drawing conclusions about H3 and H4, because those rely on predicted rankings. Likely culprits, in the order I would check them:

1. Overhead: calibrate the constant by regressing measured step time against forward-time sums across several trees.  
2. The draft level-0 cost: the pending chunk has about G tokens, not 1, so replace the first drafting term with the draft time at width ceil(G).  
3. The masked-attention code path being slower than the plain one profiled.  
4. The positional assumption: use the depth-dependent matrix from the visit logs, rebuild the trees, and see whether predicted and measured gaps close.  
5. Context length drift: t\_verify depends on context, so profile at the typical lengths of your tasks.

A useful summary figure is a scatter of predicted versus measured speedup for all configurations you actually ran, with the y \= x line.

### **9.3 Testing the hypotheses**

* H1 (acceptance depends on alpha): plot p\_1 versus alpha with the per-position distribution as a violin or histogram; show the bimodality diagnostic.  
* H2 (tree shape shifts with alpha): for each feasible alpha plot optimal depth, maximum width, and node count; draw two or three example trees.  
* H3 (hardware dependence): build the transfer matrix. Rows are the platform whose optimum you take, columns the platform you run it on, entries the measured speedup; the transfer loss is one minus the off-diagonal entry divided by the diagonal entry in that column. Include the t\_verify curves and note where each knee sits.  
* H4 (joint beats sequential): compare the sequential optimum and the joint optimum, predicted and measured, using a paired bootstrap over prompts on measured ms per token. If the confidence interval includes zero, say that the data do not show an advantage.

The quality threshold deserves care. Present results at two or three thresholds, plus the Pareto frontier of quality versus speedup across alpha, so that conclusions do not hinge on one arbitrary Q\_min. Remember that Faster Cascades reports the quality-matching comparison and the best-quality-at-large-model-latency comparison; adopting the same two framings makes your numbers directly comparable to their Table 2\.

### **9.4 Figures and tables to produce**

1. t\_verify and t\_draft versus n for each platform.  
2. Q(alpha) with confidence intervals, plus the target-only and draft-only lines.  
3. p\_k versus alpha, and the p\_1 histogram.  
4. Predicted speedup surface over (n, d) for a few alphas, with the optimum marked.  
5. Optimal tree depth and width versus alpha, per platform.  
6. Quality versus measured speedup (Pareto) for all methods.  
7. Predicted versus measured speedup scatter.  
8. Transfer matrix heat map.

### **9.5 Reporting honestly**

Possible outcomes and how to frame each. If joint beats sequential, quantify the gain with a confidence interval and show where it comes from (acceptance shape change versus hardware knee). If it does not, report that the coupling is weak for these models and hardware, and characterize when it would matter (the dependence of the optimal tree on alpha, measured by how different the trees at different alphas actually are). Either way, state the limitations: small model pairs, batch size 1, a Python decoder, plug-in deferral rules, and the positional assumption. The papers themselves note related limitations: Faster Cascades notes higher total compute than a sequential cascade and local per-token objectives, and SEQUOIA notes that the optimal tree depends on the exact acceptance vector for each model pair, temperature and domain.

## **10\. Pitfalls and a debugging checklist**

| Symptom | Likely cause |
| ----- | ----- |
| Verification unit test fails | residual uses the original q rather than the current D; forgot to zero the proposed token in D; missing uniform fallback |
| Quality of mode p differs from plain sampling | temperature applied twice, or applied to the deferral test; wrong dtype handling in softmax |
| Tree-attention check fails | model ignores 4D masks in this version; position ids missing; sdpa versus eager; wrong mask dtype or polarity |
| Decoder first-token test fails | child-to-token assignment off by one; keep\_path slot mismatch; draft cache not cropped; pending chunk wrong |
| Output degenerates after a few steps | KV cache of accepted nodes inconsistent (check positions and ordering in keep\_path) |
| Speedup below 1 with large trees | Python overhead, per-child GPU syncs, or t\_verify past its knee; check the profile and shrink the tree |
| Predicted far above measured | overhead not modeled; masked path slower; draft level-0 width ignored |
| Noisy or non-reproducible timings | clock throttling, shared GPU, no warm-up, no synchronize; repeat and interleave |
| Out of memory | tree too large for the cache plus qs tensor (n by vocabulary floats, about 50 MB at n \= 256 for a 50k vocabulary); reduce n or use offloading deliberately |
| All methods score near zero | task too hard for the model; change the task or pair |

Things I could not verify and you should check yourself: the current Colab and Kaggle GPU options; the exact structure of the SEQUOIA repository (I only had the paper); which transformers version supports custom 4D masks and the cache API for each model family; and my reading that static trees waste sibling branches under cascades, which is a conjecture to measure with the visit logs.

Small numerical points: compute probabilities in float32 even when models run in fp16; fp16 logits from a batched tree pass differ slightly from sequential passes, which perturbs the sampled distribution at the level of rounding noise, so keep that in mind when interpreting chi-square tests with very large sample sizes; and make seeds explicit everywhere (per-prompt generators).

## **11\. How to generate the code with an AI assistant**

The modules above are written to be generated, tested, and fixed one at a time. The workflow that works best is below.

### **11.1 Working rules**

1. One module per conversation turn. Never ask for the whole project at once; errors compound across modules and become impossible to localize.  
2. Give the assistant the contract, not just the goal: the function signatures, input and output shapes, the formulas (copy Part 4), and the done-when test. Ask it to write the test first.  
3. Run the tests yourself. Paste the full failing output (traceback and values) back, not a summary.  
4. Keep a CONTEXT.md in the repo with the notation, the invariants of Part 8.4, the file list, and decisions made (model pair, alpha grid, library versions). Paste it at the start of each session.  
5. Pin versions and tell the assistant which ones, because APIs for caches and masks changed across transformers releases.  
6. Freeze a module once its test passes; later changes go through a new test.  
7. Review generated code against the checklist in 11.3 before running anything long.

### **11.2 Prompt template**

Context: \<paste CONTEXT.md\>  
Task: implement jcsd/\<module\>.py.  
Contract: \<signatures, shapes, dtypes, devices\>  
Math: \<the formulas from Part 4 that this module implements\>  
Constraints: PyTorch \<version\>, transformers \<version\>, no new dependencies, plain Python style.  
First write tests/test\_\<module\>.py covering: \<list of cases\>. Then the implementation.  
After the code, list the assumptions you made and anything you were unsure about.

Recommended order of prompts: (1) cascade.py and its limit tests; (2) verify.py with the chi-square tests; (3) dp.py with the three tests; (4) tasks.py; (5) profile\_hw.py; (6) quality.py and its driver; (7) accept.py; (8) search.py with a synthetic-profile test; (9) tree.py and the tree-attention check; (10) decode\_tree.py with the first-token test; (11) the e2e driver and plotting scripts. A concrete example for step 2:

Implement verify\_node(pi, q, children, gen) and draft\_children(q, k, gen) for tree  
speculative sampling WITHOUT replacement (SEQUOIA Algorithm 2\) with an arbitrary target pi.  
After each rejected child s: R \= norm(max(R \- D, 0)), then zero D\[s\] and renormalize D;  
if D becomes all zero, set D uniform over tokens not yet proposed. Return (accepted index  
or \-1, produced token). Write a chi-square test showing the produced token is distributed  
as pi for k in 1,2,3,5, several pi, including a draft with fewer than k nonzero tokens.

### **11.3 Review checklist for generated code**

* The deferral test uses unscaled target probabilities; the blend uses temperature-scaled ones.  
* Residual and acceptance use the current D, not the original q, and D is updated after each rejection.  
* Tree size n counts the root; the verification pass feeds n tokens.  
* Position ids are ctx plus depth; the mask is ancestors plus self, plus the full prefix.  
* Cache cropping and path selection happen after every step; nothing is left appended.  
* Timing uses CUDA events or synchronizes before and after; warm-up is discarded.  
* Per-prompt generators are seeded; no global state is shared between methods.  
* The same prompt builder and metric are used for all methods.  
* Probabilities are computed in float32.  
* Nothing silently falls back to a different attention implementation without warning.

### **11.4 When the assistant gets stuck**

Shrink the problem: a 6-token vocabulary on the CPU for sampling logic, a tiny random-weight model for tree attention, a three-node tree for the decoder. Ask the assistant to add assertions and print intermediate quantities rather than guessing at fixes, and compare against the hand-computed worked example from Part 4.2 (and the numbers I gave: 0.64, 0.35, 0.01).

## **12\. Schedule and scope control**

The effort estimates are rough guesses for one person; adjust for your experience and GPU availability.

| Phase | Work | Rough effort |
| ----- | ----- | ----- |
| 1 | Environment, models, tasks, autoregressive baseline, cascade.py, verify.py with tests | 1 week |
| 2 | dp.py with tests; profile\_hw.py on first platform | 1 week |
| 3 | quality.py and accept.py on one pair and one task | 1 to 1.5 weeks (GPU time dominates) |
| 4 | search.py; first predicted joint versus sequential comparison | 3 to 4 days |
| 5 | tree.py, tree-attention check, decoder, correctness tests | 1.5 to 2 weeks |
| 6 | End-to-end benchmarks; cost-model calibration; second platform and transfer analysis | 1.5 weeks |
| 7 | Second pair or task, figures, writing | 1 to 2 weeks |

### **Minimum viable project**

If time or compute runs short, keep this core and cut everything else: one model pair, one task, two hardware profiles (a second profile via a different precision or offloading is acceptable if labeled as such), the TokenV3 rule only, the 1-D acceptance vector, the offline search with predicted results for all alphas, and end-to-end validation of three configurations (the joint optimum, the sequential optimum, and chain speculative decoding). Phases 1 to 4 alone already give you H1, H2 and a predicted H4; Phases 5 and 6 turn the prediction into a measured result, which is what makes the project convincing.

### **Natural extensions**

Other deferral rules (Chow, Diff, OPT, TokenV1 and V2) through the same target interface; dynamic trees that branch only where the draft is uncertain (relevant to the wasted-sibling issue); batch sizes above 1; a learned router in place of plug-in estimates, as suggested in Faster Cascades' conclusion; and more than two models.

