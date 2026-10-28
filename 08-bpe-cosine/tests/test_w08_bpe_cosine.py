import random

import pytest

from _support import load

S = load(__file__)

CLASSIC = "aaabdaaabac"
CLASSIC_MERGES = [("a", "a"), ("a", "b"), ("aa", "ab")]


# ---------- cosine ----------

def test_cosine_hand_values():
    assert S.cosine_similarity({"a": 3, "b": 4}, {"a": 1}) == pytest.approx(0.6)
    assert S.cosine_similarity({"a": 1, "b": 1}, {"a": 1, "c": 1}) == pytest.approx(0.5)


def test_cosine_identical_orthogonal_opposite():
    v = {"x": 2.0, "y": -1.0, "z": 0.5}
    assert S.cosine_similarity(v, dict(v)) == pytest.approx(1.0)
    assert S.cosine_similarity({"a": 1}, {"b": 1}) == 0.0
    assert S.cosine_similarity({"a": 1}, {"a": -1}) == pytest.approx(-1.0)


def test_cosine_scale_invariant_and_symmetric():
    u, v = {"a": 1.0, "b": 2.0, "c": 0.0}, {"a": 4.0, "c": 1.0, "d": 3.0}
    base = S.cosine_similarity(u, v)
    assert S.cosine_similarity({k: 10 * x for k, x in u.items()}, v) == pytest.approx(base)
    assert S.cosine_similarity(v, u) == pytest.approx(base)


@pytest.mark.parametrize("u,v", [({}, {}), ({}, {"a": 1}), ({"a": 0, "b": 0}, {"a": 1}), ({"a": 1}, {})])
def test_cosine_zero_vectors_give_zero(u, v):
    assert S.cosine_similarity(u, v) == 0.0


def test_cosine_does_not_mutate_inputs():
    u, v = {"a": 1}, {"a": 2, "b": 3}
    S.cosine_similarity(u, v)
    assert u == {"a": 1} and v == {"a": 2, "b": 3}


@pytest.mark.parametrize("seed", range(5))
def test_cosine_bounds_on_random_vectors(seed):
    rng = random.Random(seed)
    keys = "abcdefgh"
    u = {k: rng.uniform(-3, 3) for k in keys if rng.random() < 0.7}
    v = {k: rng.uniform(-3, 3) for k in keys if rng.random() < 0.7}
    c = S.cosine_similarity(u, v)
    assert -1 - 1e-9 <= c <= 1 + 1e-9


# ---------- train ----------

def test_classic_example_merges():
    assert S.train_bpe(CLASSIC, 3) == CLASSIC_MERGES


def test_training_prefix_property():
    assert S.train_bpe(CLASSIC, 2) == CLASSIC_MERGES[:2]
    assert S.train_bpe(CLASSIC, 0) == []


def test_stops_early_when_nothing_to_merge():
    assert len(S.train_bpe(CLASSIC, 500)) < 500  # runs out of pairs eventually
    assert S.train_bpe("ab ab ab", 5) == [("a", "b")]
    assert S.train_bpe("", 3) == []
    assert S.train_bpe("   ", 3) == []


def test_pairs_do_not_cross_word_boundaries():
    assert S.train_bpe("a b a b", 5) == []


def test_word_frequency_weights_counts():
    assert S.train_bpe("ab ab cd", 2) == [("a", "b"), ("c", "d")]
    assert S.train_bpe("cd ab cd", 2) == [("c", "d"), ("a", "b")]


def test_ties_broken_by_smallest_pair():
    assert S.train_bpe("ba ab", 2) == [("a", "b"), ("b", "a")]


def test_overlapping_pairs_counted_naively():
    assert S.train_bpe("aaa", 2) == [("a", "a"), ("aa", "a")]


def test_negative_merges_rejected():
    with pytest.raises(ValueError):
        S.train_bpe("abc", -1)


def test_training_is_deterministic_and_bounded():
    text = "the quick brown fox jumps over the lazy dog the end"
    m1, m2 = S.train_bpe(text, 12), S.train_bpe(text, 12)
    assert m1 == m2 and len(m1) <= 12
    assert all(isinstance(p, tuple) and len(p) == 2 for p in m1)
    assert m1[0] == ("h", "e")  # (t,h) and (h,e) tie at 3 (from 'the' x3); smallest tuple wins


# ---------- encode ----------

def test_encode_classic():
    assert S.bpe_encode(CLASSIC, CLASSIC_MERGES) == ["aaab", "d", "aaab", "a", "c"]
    assert S.bpe_encode(CLASSIC, CLASSIC_MERGES[:1]) == ["aa", "a", "b", "d", "aa", "a", "b", "a", "c"]


def test_encode_no_merges_gives_characters_and_drops_whitespace():
    assert S.bpe_encode("hi  there", []) == list("hithere")
    assert S.bpe_encode("", [("a", "b")]) == []


def test_encode_applies_left_to_right_non_overlapping():
    assert S.bpe_encode("aaa", [("a", "a")]) == ["aa", "a"]
    assert S.bpe_encode("aaaa", [("a", "a")]) == ["aa", "aa"]


def test_encode_merge_order_matters():
    assert S.bpe_encode("low", [("o", "w"), ("l", "ow")]) == ["low"]
    assert S.bpe_encode("low", [("l", "ow"), ("o", "w")]) == ["l", "ow"]


def test_encode_multiple_words_and_unseen_symbols():
    assert S.bpe_encode("ab ab", [("a", "b")]) == ["ab", "ab"]
    assert S.bpe_encode("xyz ab", [("a", "b")]) == ["x", "y", "z", "ab"]
    assert S.bpe_encode("héllo 你好", [("l", "l")]) == ["h", "é", "ll", "o", "你", "好"]


def test_encode_does_not_merge_across_words():
    assert S.bpe_encode("a b", [("a", "b")]) == ["a", "b"]


@pytest.mark.parametrize("seed", range(5))
def test_roundtrip_and_monotone_token_count(seed):
    rng = random.Random(seed)
    words = ["".join(rng.choice("abcd") for _ in range(rng.randint(1, 8))) for _ in range(40)]
    text = " ".join(words)
    merges = S.train_bpe(text, 30)
    counts = []
    for k in range(0, len(merges) + 1, 5):
        toks = S.bpe_encode(text, merges[:k])
        assert "".join(toks) == "".join(words)
        counts.append(len(toks))
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] < counts[0]  # training on this text really compresses it


def test_trained_text_compresses():
    text = "the cat sat on the mat the cat ate the rat"
    merges = S.train_bpe(text, 10)
    assert len(S.bpe_encode(text, merges)) < len("".join(text.split()))


# ---------- bpe_cosine ----------

def test_bpe_cosine_identical_and_disjoint():
    assert S.bpe_cosine("hello world", "hello world", []) == pytest.approx(1.0)
    assert S.bpe_cosine("aaa", "bbb", []) == 0.0


def test_bpe_cosine_is_bag_of_tokens():
    assert S.bpe_cosine("ab", "ba", []) == pytest.approx(1.0)  # same characters
    assert S.bpe_cosine("ab", "ba", [("a", "b")]) == 0.0  # ['ab'] vs ['b', 'a']


def test_bpe_cosine_symmetric_and_partial():
    m = [("a", "b")]
    x, y = S.bpe_cosine("ab ab c", "ab d", m), S.bpe_cosine("ab d", "ab ab c", m)
    assert x == pytest.approx(y)
    assert 0 < x < 1


def test_bpe_cosine_empty_text_is_zero():
    assert S.bpe_cosine("", "abc", []) == 0.0
