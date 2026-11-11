"""File layout of every pipeline artifact (spec 5.1). One place, so scripts agree."""
import glob, json, os


def _t(T):
    return '' if abs(T - 1.0) < 1e-9 else '_T%g' % T


def quality_path(root, pair, task, setting, T=1.0):
    return os.path.join(root, 'quality', '%s_%s%s_%s.json' % (pair, task, _t(T), setting))


def search_path(root, hw, pair, task, T=1.0):
    return os.path.join(root, 'search', '%s_%s_%s%s.json' % (hw, pair, task, _t(T)))


def e2e_path(root, hw, pair, task, T=1.0):
    return os.path.join(root, 'results', 'e2e_%s_%s_%s%s.json' % (hw, pair, task, _t(T)))


def calib_path(root, hw, pair, task, T=1.0):
    return os.path.join(root, 'calib', '%s_%s_%s%s.json' % (hw, pair, task, _t(T)))


def load_quality(root, pair, task, T=1.0):
    """-> dict setting -> record, for every quality file of this pair/task/temperature."""
    pat = os.path.join(root, 'quality', '%s_%s%s_*.json' % (pair, task, _t(T)))
    out = {}
    for f in sorted(glob.glob(pat)):
        r = load_json(f)
        out[r['setting']] = r
    return out


def find_profile(root, hw, pair, ctx=None, masked=True):
    """Pick the profile for this platform/pair; prefer the masked (tree) path and the context
    length closest to `ctx`."""
    cands = glob.glob(os.path.join(root, 'profiles', '%s_%s_ctx*.json' % (hw, pair)))
    if masked and any(c.endswith('_masked.json') for c in cands):
        cands = [c for c in cands if c.endswith('_masked.json')]
    if not cands:
        raise FileNotFoundError('no profile for hw=%s pair=%s under %s/profiles' % (hw, pair, root))
    def ctx_of(c):
        return int(c.split('_ctx')[-1].split('_')[0].split('.')[0])
    c = min(cands, key=lambda c: abs(ctx_of(c) - (ctx or 0)))
    return c, load_json(c)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
