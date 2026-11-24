from dataclasses import dataclass, asdict
import json, os

# model pairs (spec 6.3). chat=True uses the tokenizer's chat template (instruction-tuned pairs).
PAIRS = {
    'A': dict(draft='EleutherAI/pythia-160m', target='EleutherAI/pythia-2.8b', chat=False),
    'A410': dict(draft='EleutherAI/pythia-410m', target='EleutherAI/pythia-2.8b', chat=False),
    'B': dict(draft='Qwen/Qwen2.5-0.5B-Instruct', target='Qwen/Qwen2.5-3B-Instruct', chat=True),
}


@dataclass
class Cfg:
    draft: str = 'EleutherAI/pythia-410m'
    target: str = 'EleutherAI/pythia-2.8b'
    chat: bool = False
    dtype: str = 'float16'
    device: str = 'cuda'
    attn: str = 'sdpa'          # try 'eager' when debugging tree attention
    temperature: float = 1.0
    alphas: tuple = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)
    max_new_tokens: int = 128
    n_calib: int = 200
    n_test: int = 300
    max_children: int = 16      # B in the DP
    max_tree: int = 256         # N in the DP
    max_depth: int = 16         # D in the DP
    seed: int = 0
    out_dir: str = 'results'

    def save(self, name):
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, name), 'w') as f:
            json.dump(asdict(self), f, indent=2)


def pair_name(cfg):
    return cfg.target.split('/')[-1] + '__' + cfg.draft.split('/')[-1]


def cfg_for_pair(key, **kw):
    p = dict(PAIRS[key])
    p.update(kw)
    return Cfg(**p)


def setting_name(mode, alpha=None):
    # two decimals as before (a0.30), more only when needed (a0.995): '%.2f' would map 0.995 and
    # 0.998 onto the existing a0.99 / a1.00 files, which run_quality then skips as done
    if mode in ('p', 'q'):
        return mode
    s = '%.2f' % alpha
    if abs(float(s) - alpha) > 1e-9:
        s = ('%.6f' % alpha).rstrip('0')
    return 'a' + s
