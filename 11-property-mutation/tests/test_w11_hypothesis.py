import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, strategies as st  # noqa: E402

from _support import load  # noqa: E402

S = load(__file__)

ints = st.integers(min_value=-1000, max_value=1000)
sorted_ints = st.lists(ints, max_size=30).map(sorted)
brackets = st.text(alphabet="()[]{}", max_size=20)


@given(st.text(max_size=50))
def test_rle_round_trip(s):
    assert S.rle_decode(S.rle_encode(s)) == s


@given(st.text(alphabet="abc", max_size=50))
def test_rle_runs_are_positive_and_maximal(s):
    pairs = S.rle_encode(s)
    assert all(c >= 1 and len(ch) == 1 for ch, c in pairs)
    assert all(x[0] != y[0] for x, y in zip(pairs, pairs[1:]))
    assert sum(c for _, c in pairs) == len(s)


@given(ints, ints, ints)
def test_clamp_in_range_and_idempotent(x, a, b):
    lo, hi = min(a, b), max(a, b)
    y = S.clamp(x, lo, hi)
    assert lo <= y <= hi
    assert S.clamp(y, lo, hi) == y
    if lo <= x <= hi:
        assert y == x


@given(sorted_ints, sorted_ints)
def test_merge_equals_sorted_concat(a, b):
    assert S.merge_sorted(a, b) == sorted(a + b)


@given(sorted_ints, sorted_ints)
def test_merge_is_stable_on_tagged_items(a, b):
    ta = [(x, "a", i) for i, x in enumerate(a)]
    tb = [(x, "b", i) for i, x in enumerate(b)]
    got = S.merge_sorted(ta, tb, key=lambda t: t[0])
    assert got == sorted(ta + tb, key=lambda t: t[0])  # sorted() is stable: a-items first on ties


@given(brackets, brackets)
def test_balanced_closed_under_concatenation(s, t):
    if S.is_balanced(s) and S.is_balanced(t):
        assert S.is_balanced(s + t)


@given(brackets)
def test_balanced_closed_under_wrapping_and_rejects_stray_closer(s):
    if S.is_balanced(s):
        for o, c in ("()", "[]", "{}"):
            assert S.is_balanced(o + s + c)
        assert not S.is_balanced(s + ")")
        assert not S.is_balanced("(" + s)


@given(st.text(max_size=20))
def test_balanced_ignores_non_bracket_characters(s):
    only = "".join(ch for ch in s if ch in "()[]{}")
    assert S.is_balanced(s) == S.is_balanced(only)
