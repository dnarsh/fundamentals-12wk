import math

import pytest

from _support import load

S = load(__file__)

SIZES = [2 ** k for k in range(4, 15)]


def synth(f, c=1.0):
    return [(n, c * f(n)) for n in SIZES]


def test_time_call_runs_exactly_repeats_times_and_passes_args():
    calls = []

    def fn(a, b):
        calls.append((a, b))

    S.time_call(fn, 1, 2, repeats=4)
    assert calls == [(1, 2)] * 4


def test_time_call_default_repeats_is_five():
    calls = []
    S.time_call(lambda: calls.append(1))
    assert len(calls) == 5


def test_time_call_returns_minimum_not_mean():
    import time

    delays = iter([0.05, 0.0, 0.05])

    def fn():
        time.sleep(next(delays))

    t = S.time_call(fn, repeats=3)
    assert 0 <= t < 0.02


def test_time_call_returns_positive_float_in_seconds():
    import time

    t = S.time_call(lambda: time.sleep(0.02), repeats=2)
    assert isinstance(t, float)
    assert 0.015 < t < 0.5


@pytest.mark.parametrize("bad", [0, -1])
def test_time_call_rejects_nonpositive_repeats(bad):
    with pytest.raises(ValueError):
        S.time_call(lambda: None, repeats=bad)


def test_measure_shape_and_order():
    made = []

    def make(n):
        made.append(n)
        return list(range(n))

    pts = S.measure(sum, make, [10, 100, 1000], repeats=2)
    assert [n for n, _ in pts] == [10, 100, 1000]
    assert all(isinstance(t, float) and t >= 0 for _, t in pts)
    assert made == [10, 100, 1000]  # one build per size, in order


def test_measure_excludes_input_construction():
    import time

    def slow_make(n):
        time.sleep(0.05)
        return n

    pts = S.measure(lambda x: x + 1, slow_make, [1, 2], repeats=2)
    assert all(t < 0.02 for _, t in pts)


def test_measure_sees_quadratic_growth():
    def quad(n):
        total = 0
        for i in range(n):
            for j in range(n):
                total += 1
        return total

    (_, t_small), (_, t_big) = S.measure(quad, lambda n: n, [100, 400], repeats=3)
    assert t_big > 3 * t_small  # ideal ratio is 16


@pytest.mark.parametrize("k", [0.5, 1, 2, 3])
def test_exponent_of_exact_power_laws(k):
    assert S.estimate_exponent(synth(lambda n: n ** k, c=7.0)) == pytest.approx(k, abs=1e-6)


def test_exponent_of_constant_is_zero():
    assert S.estimate_exponent(synth(lambda n: 1.0, c=3.0)) == pytest.approx(0.0, abs=1e-9)


def test_exponent_n_log_n_is_a_bit_above_one():
    e = S.estimate_exponent(synth(lambda n: n * math.log(n)))
    assert 1.0 < e < 1.3


def test_exponent_works_with_unsorted_points():
    pts = synth(lambda n: n * n)
    pts.reverse()
    assert S.estimate_exponent(pts) == pytest.approx(2.0, abs=1e-6)


@pytest.mark.parametrize("pts", [[], [(10, 1.0)], [(0, 1.0), (10, 2.0)], [(1, 1.0), (2, 0.0)], [(1, 1.0), (-2, 1.0)]])
def test_exponent_rejects_bad_input(pts):
    with pytest.raises(ValueError):
        S.estimate_exponent(pts)


@pytest.mark.parametrize("c", [1e-9, 1.0, 123.4])
@pytest.mark.parametrize(
    "name,f",
    [
        ("O(1)", lambda n: 1.0),
        ("O(log n)", lambda n: math.log2(n)),
        ("O(n)", lambda n: float(n)),
        ("O(n log n)", lambda n: n * math.log2(n)),
        ("O(n^2)", lambda n: float(n * n)),
    ],
)
def test_classify_exact_synthetic_curves(name, f, c):
    assert S.classify(synth(f, c)) == name
