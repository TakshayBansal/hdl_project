from jcsd.config import setting_name


def test_setting_names_are_distinct_and_backward_compatible():
    assert setting_name('p') == 'p' and setting_name('q') == 'q'
    assert [setting_name('cascade', a) for a in (0.0, 0.3, 0.99, 1.0)] == ['a0.00', 'a0.30', 'a0.99', 'a1.00']
    fine = [0.99, 0.995, 0.998, 0.999, 1.0]          # '%.2f' used to collapse these onto a0.99 / a1.00
    assert len({setting_name('cascade', a) for a in fine}) == len(fine)
