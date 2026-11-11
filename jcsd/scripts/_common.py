import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from jcsd.config import Cfg, PAIRS, pair_name  # noqa: E402


def add_pair_args(ap):
    ap.add_argument('--pair', default='A', help='preset in jcsd/config.py PAIRS (A, A410, B)')
    ap.add_argument('--draft', default=None, help='override the preset draft model')
    ap.add_argument('--target', default=None, help='override the preset target model')
    ap.add_argument('--chat', default=None, choices=['0', '1'], help='override chat-template use')
    ap.add_argument('--task', default='cnndm', choices=['cnndm', 'gsm8k'])
    ap.add_argument('--dtype', default='float16')
    ap.add_argument('--attn', default='sdpa')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--root', default='.', help='directory holding profiles/, quality/, search/, results/')
    ap.add_argument('--seed', type=int, default=0)


def cfg_from(a, **kw):
    p = dict(PAIRS[a.pair])
    if a.draft:
        p['draft'] = a.draft
    if a.target:
        p['target'] = a.target
    if a.chat is not None:
        p['chat'] = a.chat == '1'
    return Cfg(draft=p['draft'], target=p['target'], chat=p['chat'], dtype=a.dtype, attn=a.attn,
               temperature=a.temperature, seed=a.seed, **kw)


def hw_tag(tag=None):
    if tag:
        return tag
    import torch
    name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'
    return re.sub(r'\W+', '_', name).strip('_').lower()
