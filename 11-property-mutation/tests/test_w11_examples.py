import random

import pytest

from _support import load

S = load(__file__)


# ---------- rle ----------

@pytest.mark.parametrize(
    "s,pairs",
    [
        ("", []),
        ("a", [("a", 1)]),
        ("aaabcc", [("a", 3), ("b", 1), ("c", 2)]),
        ("abab", [("a", 1), ("b", 1), ("a", 1), ("b", 1)]),
        ("aaAA", [("a", 2), ("A", 2)]),
        ("  \n\n", [(" ", 2), ("\n", 2)]),
        ("ééx", [("é", 2), ("x", 1)]),
        ("z" * 1000, [("z", 1000)]),
    ],
)
def test_rle_examples_both_ways(s, pairs):
    assert S.rle_encode(s) == pairs
    assert S.rle_decode(pairs) == s


def test_rle_decode_accepts_adjacent_equal_symbols():
    assert S.rle_decode([("a", 2), ("a", 3)]) == "aaaaa"


@pytest.mark.parametrize("bad", [[("a", 0)], [("a", -1)], [("ab", 2)], [("", 2)]])
def test_rle_decode_validation(bad):
    with pytest.raises(ValueError):
        S.rle_decode(bad)


@pytest.mark.parametrize("seed", range(5))
def test_rle_random_round_trip_and_canonical_form(seed):
    rng = random.Random(seed)
    s = "".join(rng.choice("aab") for _ in range(rng.randrange(0, 60)))
    pairs = S.rle_encode(s)
    assert S.rle_decode(pairs) == s
    assert all(c >= 1 for _, c in pairs)
    assert all(x[0] != y[0] for x, y in zip(pairs, pairs[1:]))


# ---------- clamp ----------

@pytest.mark.parametrize(
    "x,lo,hi,want",
    [(5, 0, 10, 5), (-1, 0, 10, 0), (11, 0, 10, 10), (0, 0, 10, 0), (10, 0, 10, 10), (3, 3, 3, 3), (7, 3, 3, 3), (0.5, 0.0, 1.0, 0.5), (-2.5, -1.5, 1.5, -1.5)],
)
def test_clamp_examples(x, lo, hi, want):
    assert S.clamp(x, lo, hi) == want


def test_clamp_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        S.clamp(1, 5, 0)


# ---------- merge_sorted ----------

def test_merge_examples():
    assert S.merge_sorted([], []) == []
    assert S.merge_sorted([1, 2], []) == [1, 2]
    assert S.merge_sorted([], [1, 2]) == [1, 2]
    assert S.merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
    assert S.merge_sorted([1, 2, 3], [4, 5]) == [1, 2, 3, 4, 5]
    assert S.merge_sorted([4, 5], [1, 2, 3]) == [1, 2, 3, 4, 5]
    assert S.merge_sorted([1, 1, 2], [1, 3]) == [1, 1, 1, 2, 3]


def test_merge_is_stable_a_before_b_on_ties():
    a = [(1, "a1"), (2, "a2"), (2, "a3")]
    b = [(1, "b1"), (2, "b2")]
    got = S.merge_sorted(a, b, key=lambda t: t[0])
    assert got == [(1, "a1"), (1, "b1"), (2, "a2"), (2, "a3"), (2, "b2")]


def test_merge_with_key_orders_by_key():
    assert S.merge_sorted(["bb", "dddd"], ["a", "ccc"], key=len) == ["a", "bb", "ccc", "dddd"]


def test_merge_does_not_mutate_inputs_and_returns_new_list():
    a, b = [1, 4], [2, 3]
    out = S.merge_sorted(a, b)
    assert a == [1, 4] and b == [2, 3]
    out.append(99)
    assert a == [1, 4] and b == [2, 3]


@pytest.mark.parametrize("seed", range(8))
def test_merge_random_matches_sorted(seed):
    rng = random.Random(seed)
    a = sorted(rng.randrange(20) for _ in range(rng.randrange(0, 30)))
    b = sorted(rng.randrange(20) for _ in range(rng.randrange(0, 30)))
    assert S.merge_sorted(a, b) == sorted(a + b)


# ---------- is_balanced ----------

@pytest.mark.parametrize("s", ["", "()", "[]", "{}", "()[]{}", "([{}])", "{[()()]}", "a(b)c", "no brackets", "(a[b]{c})d", "((()))"])
def test_balanced_true(s):
    assert S.is_balanced(s) is True


@pytest.mark.parametrize("s", ["(", ")", ")(", "(]", "([)]", "((", "(()", "())", "{[}]", "}{", "[(])", "a(b", "x)y", "(((((", "))))))"])
def test_balanced_false(s):
    assert S.is_balanced(s) is False


@pytest.mark.parametrize("seed", range(5))
def test_balanced_closure_on_random_wrapping(seed):
    rng = random.Random(seed)
    s = ""
    for _ in range(rng.randrange(1, 8)):
        o, c = rng.choice(["()", "[]", "{}"])
        s = o + s + c if rng.random() < 0.5 else s + o + c
        assert S.is_balanced(s)
    assert not S.is_balanced(s + ")")
    assert not S.is_balanced("(" + s)
