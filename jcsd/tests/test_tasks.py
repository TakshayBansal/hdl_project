import numpy as np
import pytest
from jcsd.tasks import (last_number, make_prompt, clean_output, score, bootstrap_ci,
                        paired_bootstrap_ci, gsm8k_gold)

GSM = [dict(question='q%d' % i, answer='work out something\n#### %s' % a)
       for i, a in enumerate(['18', '1,250', '-3', '7.5'])]


@pytest.mark.parametrize('text,want', [
    ('so the answer is 18.', '18'),
    ('total 1,250 dollars', '1250'),
    ('it is -3 degrees', '-3'),
    ('costs 7.5', '7.5'),
    ('first 4 then 9', '9'),
    ('no digits here', ''),
    ('- and . only', ''),
])
def test_last_number(text, want):
    assert last_number(text) == want


def test_gsm8k_gold_parses_to_itself():
    preds = [e['answer'] for e in GSM]
    assert score('gsm8k', preds, GSM) == [1.0] * 4
    assert [gsm8k_gold(e) for e in GSM] == ['18', '1250', '-3', '7.5']


def test_gsm8k_wrong_answer_scores_zero():
    assert score('gsm8k', ['the answer is 19'], GSM[:1]) == [0.0]


def test_prompts_and_cleaning():
    p = make_prompt('gsm8k', GSM[3], GSM[:2])
    assert p.startswith('Question: q0') and p.endswith('Question: q3\nAnswer:')
    assert p.count('Question:') == 3
    c = make_prompt('cnndm', dict(article='word ' * 400), [])
    assert c.endswith('\nSummary:') and len(c.split()) < 310
    assert clean_output('gsm8k', ' it is 5\nQuestion: next') == 'it is 5'
    assert clean_output('cnndm', 'a summary Article: junk') == 'a summary'


def test_cnndm_gold_scores_one():
    pytest.importorskip('evaluate'); pytest.importorskip('rouge_score')
    exs = [dict(highlights='The cat sat on the mat. It was happy.')]
    assert score('cnndm', [exs[0]['highlights']], exs) == pytest.approx([1.0])


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    s = np.random.default_rng(1).normal(0.5, 0.1, 200)
    m, lo, hi = bootstrap_ci(s)
    assert lo < m < hi and (m, lo, hi) == bootstrap_ci(s)


def test_paired_bootstrap_detects_shift_and_null():
    rng = np.random.default_rng(2)
    base = rng.normal(0.5, 0.2, 300)
    m, lo, hi = paired_bootstrap_ci(base + 0.05, base)
    assert lo > 0                                     # consistent shift is resolved
    noise = rng.normal(0, 0.1, 300)
    m, lo, hi = paired_bootstrap_ci(base + noise, base)
    assert lo < 0 < hi                                # pure noise contains zero
