import random
import statistics

import pytest

from _support import load

S = load(__file__)


def gauss_data(seed, n, mu=0.0, sd=1.0):
    rng = random.Random(seed)
    return [rng.gauss(mu, sd) for _ in range(n)]


def test_distribution_size_and_resample_size_and_membership():
    data = [1, 2, 3, 4, 5, 6, 7]
    seen = []

    def stat(sample):
        seen.append(list(sample))
        return sum(sample)

    out = S.bootstrap_distribution(data, stat, n_resamples=300, seed=1)
    assert len(out) == 300 == len(seen)
    assert all(len(s) == len(data) for s in seen)
    assert all(x in data for s in seen for x in s)
    assert out == [sum(s) for s in seen]


def test_resampling_is_with_replacement_and_not_a_permutation():
    data = list(range(20))
    seen = []
    S.bootstrap_distribution(data, lambda s: seen.append(sorted(s)) or 0, n_resamples=50, seed=3)
    assert any(len(set(s)) < len(data) for s in seen)  # duplicates appear


def test_resample_means_cover_both_sides_of_the_sample_mean():
    data = [1, 2, 3, 4, 100]
    out = S.bootstrap_distribution(data, statistics.mean, n_resamples=500, seed=0)
    m = statistics.mean(data)
    assert min(out) < m < max(out)
    assert statistics.mean(out) == pytest.approx(m, rel=0.15)


def test_does_not_mutate_input():
    data = [3, 1, 2]
    S.bootstrap_distribution(data, statistics.mean, 10, seed=1)
    S.bootstrap_ci(data, seed=1)
    assert data == [3, 1, 2]


def test_seed_reproducible_and_seeds_differ():
    d = gauss_data(1, 30)
    assert S.bootstrap_ci(d, seed=7) == S.bootstrap_ci(d, seed=7)
    assert S.bootstrap_ci(d, seed=7) != S.bootstrap_ci(d, seed=8)
    assert S.bootstrap_distribution(d, statistics.mean, 50, seed=2) == S.bootstrap_distribution(d, statistics.mean, 50, seed=2)


def test_global_random_state_untouched():
    random.seed(123)
    before = random.getstate()
    S.bootstrap_ci(gauss_data(1, 20), seed=5)
    S.bootstrap_ci(gauss_data(1, 20), seed=None)
    assert random.getstate() == before


def test_ci_is_ordered_and_brackets_the_mean_for_normal_data():
    d = gauss_data(2, 60, mu=5, sd=2)
    lo, hi = S.bootstrap_ci(d, seed=1)
    assert lo <= hi
    assert lo < statistics.mean(d) < hi
    assert min(d) <= lo and hi <= max(d)


def test_constant_data_gives_zero_width():
    lo, hi = S.bootstrap_ci([4.2] * 10, seed=1)
    assert lo == pytest.approx(4.2) and hi == pytest.approx(4.2)


def test_single_observation():
    lo, hi = S.bootstrap_ci([3.0], seed=1)
    assert lo == hi == pytest.approx(3.0)


def test_small_n_example_stays_inside_data_range():
    lo, hi = S.bootstrap_ci([1, 2, 3, 4, 5], seed=0)
    assert 1 <= lo <= 3 <= hi <= 5


def test_smaller_alpha_gives_nested_wider_interval():
    d = gauss_data(3, 40)
    lo99, hi99 = S.bootstrap_ci(d, alpha=0.01, seed=4)
    lo80, hi80 = S.bootstrap_ci(d, alpha=0.20, seed=4)
    assert lo99 <= lo80 and hi80 <= hi99
    assert (hi99 - lo99) > (hi80 - lo80)


def test_width_shrinks_with_n():
    def avg_width(n):
        ws = []
        for s in range(6):
            lo, hi = S.bootstrap_ci(gauss_data(50 + s, n), n_resamples=400, seed=s)
            ws.append(hi - lo)
        return statistics.mean(ws)

    w25, w400 = avg_width(25), avg_width(400)
    assert w400 < w25 / 2.5  # theory: 4x smaller


def test_coverage_near_nominal():
    hits, trials = 0, 200
    for s in range(trials):
        d = gauss_data(1000 + s, 30)  # true mean 0
        lo, hi = S.bootstrap_ci(d, n_resamples=300, seed=s)
        hits += lo <= 0 <= hi
    assert 0.86 <= hits / trials <= 0.995  # percentile bootstrap at n=30 is slightly liberal


def test_median_statistic():
    d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    lo, hi = S.bootstrap_ci(d, stat=statistics.median, seed=1)
    assert lo <= statistics.median(d) <= hi
    assert hi < 100  # the median is not dragged to the outlier at this confidence


def test_custom_statistic_proportion():
    d = [1] * 9 + [0] * 3  # 12 pass/fail scores
    lo, hi = S.bootstrap_ci(d, stat=lambda s: sum(s) / len(s), seed=2)
    assert 0 <= lo <= 0.75 <= hi <= 1


def test_diff_ci_excludes_zero_for_big_effect():
    a, b = gauss_data(1, 30, mu=3), gauss_data(2, 30, mu=0)
    lo, hi = S.bootstrap_diff_ci(a, b, seed=1)
    assert lo > 0 and lo <= statistics.mean(a) - statistics.mean(b) <= hi


def test_diff_ci_covers_zero_for_no_effect_and_is_antisymmetric():
    a = gauss_data(5, 40)
    b = list(reversed(a))  # identical values: the true difference is exactly 0
    lo, hi = S.bootstrap_diff_ci(a, b, seed=1)
    assert lo < 0 < hi
    lo2, hi2 = S.bootstrap_diff_ci(b, a, seed=1)
    assert lo2 < 0 < hi2
    assert (hi - lo) == pytest.approx(hi2 - lo2, rel=0.35)


def test_diff_ci_allows_unequal_sizes():
    a, b = gauss_data(1, 8, mu=1), gauss_data(2, 20, mu=0)
    lo, hi = S.bootstrap_diff_ci(a, b, seed=1)
    assert lo <= hi


def test_validation():
    with pytest.raises(ValueError):
        S.bootstrap_ci([], seed=1)
    with pytest.raises(ValueError):
        S.bootstrap_distribution([], statistics.mean, 10, seed=1)
    for bad in (0, 1, -0.1, 1.5):
        with pytest.raises(ValueError):
            S.bootstrap_ci([1, 2, 3], alpha=bad)
    with pytest.raises(ValueError):
        S.bootstrap_ci([1, 2, 3], n_resamples=0)
    with pytest.raises(ValueError):
        S.bootstrap_diff_ci([], [1, 2])
    with pytest.raises(ValueError):
        S.bootstrap_diff_ci([1, 2], [])
