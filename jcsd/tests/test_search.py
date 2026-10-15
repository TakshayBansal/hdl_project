import numpy as np
import pytest
from types import SimpleNamespace
from jcsd.search import (level_widths, make_cost, best_for_acc, joint_search,
                         sequential_search)

SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def prof_flat(c=0.02, draft_per_token=0.0, knee=64, slope=1e-3):
    # t_verify flat up to the knee then linear; t_draft proportional to width
    tv = [c if n <= knee else c + slope * (n - knee) for n in SIZES]
    td = [draft_per_token * n for n in SIZES]
    return dict(sizes=SIZES, t_verify=tv, t_draft_level=td, t_ar=tv[0])


def cfg(N=32, D=8, B=3):
    return SimpleNamespace(max_tree=N, max_depth=D, max_children=B)


def test_level_widths():
    assert level_widths([[[], []], [[]]]) == [1, 2, 3]
    assert level_widths([]) == [1]


def test_cost_matches_hand():
    prof = dict(sizes=[1, 4], t_verify=[1.0, 2.0], t_draft_level=[0.1, 0.4], t_ar=1.0)
    # widths [1,2,1]: n=4 -> verify 2.0; drafting levels 0 and 1 (widths 1,2) -> 0.1 + 0.2
    assert make_cost(prof, overhead=0.5)([1, 2, 1]) == pytest.approx(2.0 + 0.1 + 0.2 + 0.5)


def test_hand_checked_small_tree():
    # free drafting and constant verify cost: speedup equals G of the best 3-node tree.
    # chain: 1 + .6 + .36 = 1.96 beats root with two children: 1 + .6 + .3 = 1.9
    prof = dict(sizes=[1, 2, 4], t_verify=[1.0, 1.0, 1.0], t_draft_level=[0, 0, 0], t_ar=1.0)
    b = best_for_acc([0.6, 0.3], cfg(N=3, D=8, B=2), prof)
    assert b['speedup'] == pytest.approx(1.96)
    assert b['widths'] == [1, 1, 1]


def test_alpha_one_best_tree_is_max_depth_chain():
    c = cfg(N=32, D=8, B=3)
    prof = prof_flat(c=0.02, draft_per_token=0.001)
    b = best_for_acc([1.0], c, prof)
    assert b['widths'] == [1] * 9          # chain of the maximum useful depth
    assert b['d'] == 8 and b['n'] == 9
    cost = np.interp(9, SIZES, prof['t_verify']) + 8 * 0.001
    assert b['speedup'] == pytest.approx(9 * prof['t_ar'] / cost)


def test_bigger_trees_pay_off_only_when_verify_is_flat():
    acc = [0.5, 0.2, 0.1]
    cheap = best_for_acc(acc, cfg(N=64, D=8, B=3), prof_flat(knee=64))
    costly = best_for_acc(acc, cfg(N=64, D=8, B=3), prof_flat(knee=2, slope=0.02))
    assert cheap['n'] > costly['n']


def test_joint_at_least_sequential_and_strictly_better_when_trees_differ():
    c = cfg(N=48, D=10, B=4)
    prof = prof_flat(c=0.02, draft_per_token=0.0005, knee=64)
    acc_ref = [0.45, 0.2, 0.1, 0.05]                     # exact target: low p1, wide tree
    acc_by_alpha = {0.0: [0.5, 0.2, 0.1, 0.05], 0.5: [0.8, 0.1, 0.03, 0.01], 1.0: [1.0]}
    q = {0.0: 1.0, 0.5: 0.99, 1.0: 0.5}
    a_star, res = joint_search(acc_by_alpha, q, 0.95, c, prof)
    a_seq, sp_seq, ref = sequential_search(acc_ref, acc_by_alpha, q, 0.95, c, prof)
    assert 1.0 not in res                                # infeasible alpha is excluded
    assert res[a_star]['speedup'] >= sp_seq - 1e-9
    assert res[a_star]['speedup'] > sp_seq + 1e-6        # the best tree changed with alpha


def test_no_feasible_alpha_raises():
    c = cfg(N=8, D=4, B=2)
    with pytest.raises(ValueError):
        joint_search({0.0: [0.5]}, {0.0: 0.1}, 0.9, c, prof_flat())


def test_coarse_grid_never_beats_full_scan():
    c = cfg(N=32, D=6, B=3)
    prof = prof_flat()
    full = best_for_acc([0.6, 0.2, 0.1], c, prof)
    coarse = best_for_acc([0.6, 0.2, 0.1], c, prof, n_values=[1, 4, 8, 16, 32])
    assert coarse['speedup'] <= full['speedup'] + 1e-12
