import random

import pytest

from _support import load

S = load(__file__)


def test_type_one_error_near_alpha():
    p = S.simulate_power(0.0, 1.0, 50, alpha=0.05, n_sims=2000, seed=1)
    assert 0.03 <= p <= 0.08


def test_type_one_error_tracks_alpha():
    lo = S.simulate_power(0.0, 1.0, 50, alpha=0.01, n_sims=2000, seed=2)
    hi = S.simulate_power(0.0, 1.0, 50, alpha=0.20, n_sims=2000, seed=2)
    assert lo < 0.035 and 0.16 < hi < 0.25


def test_large_effect_has_power_near_one():
    assert S.simulate_power(2.0, 1.0, 30, n_sims=500, seed=3) > 0.99


def test_matches_normal_approximation_for_half_sd():
    p = S.simulate_power(0.5, 1.0, 64, n_sims=2000, seed=4)
    assert 0.74 <= p <= 0.88  # theory is about 0.81


def test_result_is_a_fraction_of_sims():
    p = S.simulate_power(0.5, 1.0, 20, n_sims=250, seed=5)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0
    assert (p * 250) == pytest.approx(round(p * 250), abs=1e-9)


def test_power_increases_with_n():
    small = S.simulate_power(0.5, 1.0, 20, n_sims=600, seed=6)
    big = S.simulate_power(0.5, 1.0, 80, n_sims=600, seed=6)
    assert big > small + 0.3


def test_power_increases_with_effect():
    assert S.simulate_power(1.0, 1.0, 30, n_sims=600, seed=7) > S.simulate_power(0.2, 1.0, 30, n_sims=600, seed=7) + 0.3


def test_power_decreases_with_noise():
    assert S.simulate_power(1.0, 1.0, 30, n_sims=600, seed=8) > S.simulate_power(1.0, 3.0, 30, n_sims=600, seed=8) + 0.3


def test_power_scale_invariant_in_effect_over_sd():
    a = S.simulate_power(0.5, 1.0, 40, n_sims=1500, seed=9)
    b = S.simulate_power(5.0, 10.0, 40, n_sims=1500, seed=10)
    assert abs(a - b) < 0.07


def test_negative_effect_has_same_power_as_positive():
    a = S.simulate_power(0.5, 1.0, 40, n_sims=1500, seed=11)
    b = S.simulate_power(-0.5, 1.0, 40, n_sims=1500, seed=12)
    assert abs(a - b) < 0.07  # two-sided test


def test_seed_reproducible_and_global_rng_untouched():
    assert S.simulate_power(0.5, 1.0, 20, n_sims=100, seed=42) == S.simulate_power(0.5, 1.0, 20, n_sims=100, seed=42)
    random.seed(5)
    before = random.getstate()
    S.simulate_power(0.5, 1.0, 20, n_sims=50, seed=1)
    S.simulate_power(0.5, 1.0, 20, n_sims=50)
    assert random.getstate() == before


@pytest.mark.parametrize(
    "kwargs",
    [dict(n=1), dict(n=0), dict(sd=0), dict(sd=-1), dict(n_sims=0), dict(alpha=0), dict(alpha=1), dict(alpha=-0.1)],
)
def test_simulate_power_validation(kwargs):
    args = dict(effect=0.5, sd=1.0, n=10, alpha=0.05, n_sims=10, seed=1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        S.simulate_power(**args)


def test_required_n_for_half_sd_is_near_textbook():
    n = S.required_n(0.5, 1.0, n_sims=500, seed=1)
    assert isinstance(n, int)
    assert 40 <= n <= 95  # theory: about 63 per group; simulation noise allowed


def test_required_n_shrinks_for_big_effect():
    big = S.required_n(1.0, 1.0, n_sims=500, seed=2)
    half = S.required_n(0.5, 1.0, n_sims=500, seed=2)
    assert 8 <= big <= 30
    assert half > big + 15


def test_returned_n_really_has_decent_power():
    n = S.required_n(1.0, 1.0, n_sims=500, seed=3)
    assert S.simulate_power(1.0, 1.0, n, n_sims=2000, seed=99) >= 0.6


def test_required_n_reproducible_with_seed():
    assert S.required_n(2.0, 4.0, n_sims=300, seed=4) == S.required_n(2.0, 4.0, n_sims=300, seed=4)


def test_required_n_minimum_is_two():
    assert S.required_n(50.0, 1.0, n_sims=200, seed=5) >= 2


def test_required_n_unreachable_raises():
    with pytest.raises(ValueError):
        S.required_n(0.001, 1.0, n_sims=100, seed=1, max_n=40)
