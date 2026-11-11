# Runs the CPU-only half of the pipeline (search -> calibrate -> figures -> summary) on synthetic
# artifacts, so script plumbing is checked without a GPU or torch.
import json, os, subprocess, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAIR = 'pythia-2.8b__pythia-160m'
SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def _w(root, rel, obj):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p, 'w'))


def _profile(knee, base):
    tv = [base if n <= knee else base * (1 + (n - knee) / knee) for n in SIZES]
    td = [0.004 + 0.00002 * n for n in SIZES]
    return dict(sizes=SIZES, t_verify=tv, t_draft_level=td, t_ar=tv[0], t_ar_step=tv[0] * 1.05, hw='fake')


def _quality(root, rng):
    n = 40
    base = rng.uniform(0.05, 0.15, n)
    def rec(setting, mode, alpha, qshift, pk):
        sc = list(np.clip(base + qshift + rng.normal(0, 0.01, n), 0, 1))
        m = float(np.mean(sc))
        return dict(setting=setting, mode=mode, alpha=alpha, quality=m, ci=[m - 0.01, m + 0.01], scores=sc,
                    acc=None if pk is None else dict(pk=pk, all_rejected=1 - sum(pk), p1_samples=list(rng.uniform(0, 1, 200))))
    def pk_for(p1):
        tail = [(1 - p1) * 0.5 ** k for k in range(1, 16)]
        return [p1] + tail
    _w(root, 'quality/%s_cnndm_p.json' % PAIR, rec('p', 'p', None, 0.0, pk_for(0.55)))
    _w(root, 'quality/%s_cnndm_q.json' % PAIR, rec('q', 'q', None, -0.05, None))
    for a in (0.0, 0.2, 0.5, 0.8, 1.0):
        _w(root, 'quality/%s_cnndm_a%.2f.json' % (PAIR, a),
           rec('a%.2f' % a, 'cascade', a, -0.06 * a ** 2, pk_for(min(0.6 + 0.4 * a, 1.0))))


def _run(script, *args, root):
    r = subprocess.run([sys.executable, os.path.join(HERE, 'scripts', script), '--root', root, *args],
                       capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def _fake_e2e(root, hw, transfer=None):
    S = json.load(open(os.path.join(root, 'search', '%s_%s_cnndm.json' % (hw, PAIR))))
    cfgs = [dict(c, source=hw) for c in S['configs']]
    if transfer:
        O = json.load(open(transfer))
        cfgs += [dict(c, name=c['name'] + '@' + O['hw'], source=O['hw']) for c in O['configs'] if c['name'].startswith('joint')]
    rng = np.random.default_rng(1)
    M = {}
    for c in cfgs:
        sp = 1.0 if c['name'] == 'ar' else c['predicted'] * 0.85
        G = c.get('G', 1.0)
        n_new = [64] * 20
        tpt = 0.03 / sp
        t = list(tpt * 64 * (1 + rng.normal(0, 0.02, 20)))
        steps = [int(64 / G)] * 20
        visits = {} if c['tree'] is None else {'0,%d,0' % max(1, len(c['tree'])): 50, '0,%d,-1' % max(1, len(c['tree'])): 20}
        M[c['name']] = dict(config=c, runner=repr((c['mode'], c['alpha'], c['tree'])), ms_per_token=1000 * tpt,
                            tokens_per_step=G, quality=[0.1, 0.09, 0.11], scores=[0.1] * 20, peak_gb=6.0,
                            time=t, n_new=n_new, steps=steps, visits=visits, path_len_hist={}, speedup=sp)
    _w(root, 'results/e2e_%s_%s_cnndm.json' % (hw, PAIR),
       dict(hw=hw, pair=PAIR, task='cnndm', temperature=1.0, n=20, passes=1, max_new=64, methods=M))


@pytest.mark.slow
def test_offline_pipeline(tmp_path):
    pytest.importorskip('matplotlib')
    root = str(tmp_path)
    rng = np.random.default_rng(0)
    _quality(root, rng)
    _w(root, 'profiles/hwA_%s_ctx512_masked.json' % PAIR, _profile(knee=64, base=0.03))
    _w(root, 'profiles/hwB_%s_ctx512_masked.json' % PAIR, _profile(knee=4, base=0.03))
    small = ['--N', '48', '--D', '8', '--B', '6']
    for hw in ('hwA', 'hwB'):
        out = _run('run_search.py', '--pair', 'A', '--task', 'cnndm', '--hw', hw, *small, root=root)
        assert 'joint' in out
    S = json.load(open(os.path.join(root, 'search', 'hwA_%s_cnndm.json' % PAIR)))
    for r in S['by_tol'].values():
        if r['feasible']:
            assert r['joint']['speedup'] >= r['sequential']['speedup'] - 1e-9
    _fake_e2e(root, 'hwA', transfer=os.path.join(root, 'search', 'hwB_%s_cnndm.json' % PAIR))
    _fake_e2e(root, 'hwB', transfer=os.path.join(root, 'search', 'hwA_%s_cnndm.json' % PAIR))
    out = _run('calibrate.py', '--pair', 'A', '--task', 'cnndm', '--hw', 'hwA', '--D', '8', '--B', '6', root=root)
    assert 'fitted overhead' in out
    out = _run('run_search.py', '--pair', 'A', '--task', 'cnndm', '--hw', 'hwA', *small, '--depth_matrix', root=root)
    assert 'saved' in out
    _run('make_figures.py', '--pair', 'A', '--task', 'cnndm', '--out', os.path.join(root, 'fig'), root=root)
    figs = os.listdir(os.path.join(root, 'fig'))
    for k in ('fig1', 'fig2', 'fig3', 'fig4', 'fig5', 'fig6', 'fig7', 'fig8'):
        assert any(f.startswith(k) for f in figs), (k, figs)
    out = _run('summarize.py', '--pair', 'A', '--task', 'cnndm', '--tables', os.path.join(root, 'tab'), root=root)
    assert 'H1' in out and 'H4' in out and 'H3' in out
