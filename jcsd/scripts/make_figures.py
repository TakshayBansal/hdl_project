"""Stage 7: the figures of spec 9.4, from whatever artifacts exist. CPU only, no torch.

  python scripts/make_figures.py --pair A --task cnndm --out figures
Each figure is skipped (with a note) if its inputs are missing.
"""
import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from _common import add_pair_args, cfg_from
from jcsd.config import pair_name
from jcsd.paths import load_quality, load_json
from jcsd.dp import get_tree  # noqa: F401

# validated categorical palette (fixed order, never cycled); contrast < 3:1 for some slots,
# so every series also gets a marker and a legend entry
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
MK = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 2,
                     'lines.markersize': 5, 'legend.frameon': False, 'figure.dpi': 150,
                     'savefig.bbox': 'tight', 'axes.facecolor': '#fcfcfb', 'figure.facecolor': 'white'})

ap = argparse.ArgumentParser()
add_pair_args(ap)
ap.add_argument('--out', default='figures')
a = ap.parse_args()
cfg = cfg_from(a, device='cpu')
pair = pair_name(cfg)
T = cfg.temperature
R = a.root
os.makedirs(a.out, exist_ok=True)
tsuf = '' if T == 1.0 else '_T%g' % T


def save(fig, name):
    p = os.path.join(a.out, name)
    fig.savefig(p)
    plt.close(fig)
    print('wrote', p)


def skip(what, why):
    print('skip %s: %s' % (what, why))


# 1. hardware profiles -------------------------------------------------------------
profs = sorted(glob.glob(os.path.join(R, 'profiles', '*_%s_ctx*.json' % pair)))
if profs:
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, f in enumerate(profs[:8]):
        P = load_json(f)
        lab = os.path.basename(f).replace('_' + pair, '').replace('.json', '')
        ax[0].plot(P['sizes'], np.array(P['t_verify']) * 1e3, marker=MK[i], color=C[i], label=lab)
        ax[1].plot(P['sizes'], np.array(P['t_draft_level']) * 1e3, marker=MK[i], color=C[i], label=lab)
    for x, t in zip(ax, ['target forward t_verify(n)', 'draft forward t_draft(width)']):
        x.set_xscale('log', base=2); x.set_xlabel('tokens in one forward'); x.set_ylabel('ms'); x.set_title(t)
    ax[0].legend(fontsize=7)
    save(fig, 'fig1_profiles.png')
else:
    skip('fig1', 'no profiles for ' + pair)

# 2-3. quality and acceptance ---------------------------------------------------------
Q = load_quality(R, pair, a.task, T)
cas = sorted([r for r in Q.values() if r['mode'] == 'cascade'], key=lambda r: r['alpha'])
if cas:
    al = np.array([r['alpha'] for r in cas])
    q = np.array([r['quality'] for r in cas])
    lo = np.array([r['ci'][0] for r in cas]); hi = np.array([r['ci'][1] for r in cas])
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ax.fill_between(al, lo, hi, color=C[0], alpha=0.18, lw=0)
    ax.plot(al, q, marker='o', color=C[0], label='TokenV3 cascade')
    for key, c, ls, lab in (('p', C[1], '--', 'target only (p)'), ('q', C[2], ':', 'draft only (q)')):
        if key in Q:
            ax.axhline(Q[key]['quality'], color=c, ls=ls, lw=1.5, label=lab)
    ax.set_xlabel('alpha'); ax.set_ylabel('ROUGE-2' if a.task == 'cnndm' else 'accuracy')
    ax.set_title('Quality vs alpha (95% bootstrap CI)'); ax.legend(fontsize=7)
    save(fig, 'fig2_quality%s.png' % tsuf)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    pk = np.array([r['acc']['pk'][:4] for r in cas])
    for k in range(4):
        ax[0].plot(al, pk[:, k], marker=MK[k], color=C[k], label='p_%d' % (k + 1))
    if 'p' in Q:
        ax[0].axhline(Q['p']['acc']['pk'][0], color=INK2, ls='--', lw=1, label='p_1, exact target')
    ax[0].set_xlabel('alpha'); ax[0].set_ylabel('acceptance probability'); ax[0].legend(fontsize=7)
    ax[0].set_title('Acceptance vector vs alpha (H1)')
    pick = [cas[0], cas[len(cas) // 2], cas[-2] if len(cas) > 2 else cas[-1]]
    for i, r in enumerate(pick):
        s = np.asarray(r['acc']['p1_samples'])
        if len(s):
            ax[1].hist(s, bins=40, range=(0, 1), histtype='step', lw=2, color=C[i], density=True,
                       label='alpha=%.2f' % r['alpha'])
    ax[1].set_xlabel('per-position P(first child accepted) = 1 - TV(pi, q)'); ax[1].set_ylabel('density')
    ax[1].set_title('Bimodality of p_1 (positional assumption)'); ax[1].legend(fontsize=7)
    save(fig, 'fig3_acceptance%s.png' % tsuf)
else:
    skip('fig2-3', 'no quality files')

# 4-5. search -------------------------------------------------------------------------
searches = sorted(glob.glob(os.path.join(R, 'search', '*_%s_%s%s.json' % (pair, a.task, tsuf))))
if searches:
    S = load_json(searches[0])
    al = sorted(S['per_alpha'], key=float)
    pick = [al[0], al[len(al) // 2], al[-1]]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for x, key in zip(ax, pick):
        surf = S['per_alpha'][key]['surface']
        ds = sorted({int(k.split(',')[0]) for k in surf}); ns = sorted({int(k.split(',')[1]) for k in surf})
        M = np.full((len(ds), len(ns)), np.nan)
        for k, v in surf.items():
            d, n = map(int, k.split(','))
            M[ds.index(d), ns.index(n)] = v
        im = x.imshow(M, origin='lower', aspect='auto', cmap='Blues')
        b = S['per_alpha'][key]['best']
        ni = int(np.argmin([abs(np.log(v / b['n'])) for v in ns]))      # nearest grid size
        di = int(np.argmin([abs(v - b['d']) for v in ds]))
        x.plot(ni, di, marker='*', ms=12, color=C[1], mec='white')
        x.set_xticks(range(len(ns))); x.set_xticklabels(ns, rotation=90, fontsize=6)
        x.set_yticks(range(len(ds))); x.set_yticklabels(ds, fontsize=6)
        x.set_xlabel('tree size n'); x.set_title('alpha=%s: best n=%d, d=%d, %.2fx' % (key, b['n'], b['d'], b['speedup']), fontsize=8)
        x.grid(False)
        fig.colorbar(im, ax=x, fraction=0.046)
    ax[0].set_ylabel('depth cap d')
    fig.suptitle('Predicted speedup surface (%s); star = optimum' % S['hw'], y=1.03)
    save(fig, 'fig4_surface%s.png' % tsuf)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for i, f in enumerate(searches[:8]):
        S = load_json(f)
        al = sorted(S['per_alpha'], key=float)
        x = [float(k) for k in al]
        best = [S['per_alpha'][k]['best'] for k in al]
        for axx, vals in zip(ax, ([b['d'] for b in best], [max(b['widths']) for b in best], [b['n'] for b in best])):
            axx.plot(x, vals, marker=MK[i], color=C[i], label=S['hw'])
    for axx, t in zip(ax, ['optimal depth', 'max level width', 'node count']):
        axx.set_xlabel('alpha'); axx.set_title(t + ' (H2)')
    ax[0].legend(fontsize=7)
    save(fig, 'fig5_tree_vs_alpha%s.png' % tsuf)
else:
    skip('fig4-5', 'no search files')

# 6-8. end-to-end ---------------------------------------------------------------------
e2es = sorted(glob.glob(os.path.join(R, 'results', 'e2e_*_%s_%s%s*.json' % (pair, a.task, tsuf))))
if e2es:
    E = load_json(e2es[0])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    fam = [('ar', 'autoregressive'), ('chain', 'chain spec.'), ('sequoia', 'SEQUOIA'),
           ('fc_chain', 'Faster Cascades'), ('seq_', 'sequential'), ('joint', 'joint (ours)')]
    for i, (pre, lab) in enumerate(fam):
        pts = [(m['speedup'], m['quality'][0], m['quality'][1], m['quality'][2])
               for n, m in E['methods'].items() if n.startswith(pre) and '@' not in n]
        if pts:
            s, qq, l, h = map(np.array, zip(*pts))
            ax.errorbar(s, qq, yerr=[qq - l, h - qq], fmt=MK[i], color=C[i], label=lab, ms=6, lw=1, capsize=2)
    ax.set_xlabel('measured speedup vs autoregressive'); ax.set_ylabel('quality')
    ax.set_title('Quality vs speed (%s)' % E['hw']); ax.legend(fontsize=7)
    save(fig, 'fig6_pareto%s.png' % tsuf)

    fig, ax = plt.subplots(figsize=(4.2, 4))
    allv = []
    for i, f in enumerate(e2es[:8]):
        E = load_json(f)
        pts = [(m['config']['predicted'], m['speedup']) for n, m in E['methods'].items()
               if n != 'ar' and '@' not in n and m['config'].get('predicted')]
        if pts:
            p, s = zip(*pts)
            allv += list(p) + list(s)
            ax.plot(p, s, MK[i], color=C[i], label=os.path.basename(f).replace('.json', ''), ms=6)
    if allv:
        lo_, hi_ = min(allv) * 0.9, max(allv) * 1.1
        ax.plot([lo_, hi_], [lo_, hi_], color=INK2, lw=1, ls='--', label='y = x')
        ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_)
    ax.set_xlabel('predicted speedup'); ax.set_ylabel('measured speedup')
    ax.set_title('Cost-model validation'); ax.legend(fontsize=6)
    save(fig, 'fig7_pred_vs_meas%s.png' % tsuf)

    # transfer: rows = platform whose optimum, cols = platform run on
    hws, cell = [], {}
    for f in e2es:
        E = load_json(f)
        hws.append(E['hw'])
        for n, m in E['methods'].items():
            if n.startswith('joint'):
                src = n.split('@')[1] if '@' in n else E['hw']
                tol = n.split('@')[0]
                cell.setdefault(tol, {})[(src, E['hw'])] = m['speedup']
    hws = sorted(set(hws))
    tol = sorted(cell)[0] if cell else None
    if tol and len(hws) > 1 and any(s != r for s, r in cell[tol]):
        M = np.array([[cell[tol].get((s, r), np.nan) for r in hws] for s in hws])
        fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(hws), 1.2 + 1.0 * len(hws)))
        im = ax.imshow(M, cmap='Blues'); ax.grid(False)
        for i in range(len(hws)):
            for j in range(len(hws)):
                if not np.isnan(M[i, j]):
                    loss = 1 - M[i, j] / M[j, j] if not np.isnan(M[j, j]) else np.nan
                    ax.text(j, i, '%.2fx\n(%+.0f%%)' % (M[i, j], -100 * loss) if i != j else '%.2fx' % M[i, j],
                            ha='center', va='center', fontsize=7, color=INK if M[i, j] < np.nanmax(M) * 0.7 else 'white')
        ax.set_xticks(range(len(hws))); ax.set_xticklabels(hws, rotation=30, fontsize=7)
        ax.set_yticks(range(len(hws))); ax.set_yticklabels(hws, fontsize=7)
        ax.set_xlabel('run on'); ax.set_ylabel('optimum taken from')
        ax.set_title('Transfer of joint optimum (%s), H3' % tol)
        fig.colorbar(im, ax=ax, fraction=0.046)
        save(fig, 'fig8_transfer%s.png' % tsuf)
    else:
        skip('fig8', 'needs e2e runs on >= 2 platforms with --transfer')
else:
    skip('fig6-8', 'no e2e results')
