from dataclasses import dataclass, asdict
import json, os


@dataclass
class Cfg:
    draft: str = 'EleutherAI/pythia-410m'
    target: str = 'EleutherAI/pythia-2.8b'
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
