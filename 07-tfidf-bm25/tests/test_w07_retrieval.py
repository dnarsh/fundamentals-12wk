import math

import pytest

from _support import load

S = load(__file__)

CORPUS = [
    "the cat sat on the mat",
    "dogs bark loudly at night",
    "cat cat cat",
    "the dog and the cat",
]


def docs_of(texts):
    return [S.tokenize(t) for t in texts]


# ---------- tokenize ----------

def test_tokenize_basic():
    assert S.tokenize("Don't PANIC, 42!") == ["don", "t", "panic", "42"]


@pytest.mark.parametrize("text,expected", [("", []), ("   ", []), ("!!!", []), ("a-b_c", ["a", "b", "c"]), ("A1b2", ["a1b2"])])
def test_tokenize_edges(text, expected):
    assert S.tokenize(text) == expected


# ---------- tf / idf / tf-idf ----------

def test_tf_values():
    doc = ["a", "b", "a", "c"]
    assert S.tf("a", doc) == pytest.approx(0.5)
    assert S.tf("b", doc) == pytest.approx(0.25)
    assert S.tf("z", doc) == 0.0
    assert S.tf("a", []) == 0.0


def test_idf_values():
    docs = [["a", "b"], ["a"], ["c"]]
    assert S.idf("a", docs) == pytest.approx(math.log(4 / 3) + 1)
    assert S.idf("c", docs) == pytest.approx(math.log(4 / 2) + 1)
    assert S.idf("zzz", docs) == pytest.approx(math.log(4 / 1) + 1)


def test_idf_counts_documents_not_occurrences():
    assert S.idf("a", [["a", "a", "a"], ["b"]]) == pytest.approx(math.log(3 / 2) + 1)
    assert S.idf("a", [["a"], ["b"]]) == pytest.approx(math.log(3 / 2) + 1)


def test_idf_rarer_is_larger_and_never_below_one():
    docs = [["a", "b"], ["a"], ["a", "c"]]
    assert S.idf("b", docs) > S.idf("a", docs) >= 1.0


def test_idf_empty_corpus_raises():
    with pytest.raises(ValueError):
        S.idf("a", [])


def test_tfidf_vector_values_and_keys():
    docs = [["a", "b"], ["a", "c"], ["a", "d"]]
    vec = S.tfidf_vector(["a", "b"], docs)
    assert set(vec) == {"a", "b"}
    assert vec["a"] == pytest.approx(0.5 * S.idf("a", docs))
    assert vec["b"] == pytest.approx(0.5 * (math.log(4 / 2) + 1))
    assert vec["b"] > vec["a"]  # rare beats common at equal tf


def test_tfidf_vector_uses_distinct_terms_and_handles_empty():
    vec = S.tfidf_vector(["x", "x", "y"], [["x", "x", "y"]])
    assert set(vec) == {"x", "y"}
    assert vec["x"] == pytest.approx(2 / 3 * (math.log(2 / 2) + 1))
    assert S.tfidf_vector([], [["a"]]) == {}


# ---------- BM25 ----------

def test_bm25_idf_value_and_nonnegative():
    docs = [["a"], ["a"], ["a"]]
    assert S.bm25_idf("a", docs) == pytest.approx(math.log(1 + 0.5 / 3.5))
    assert S.bm25_idf("a", docs) > 0
    assert S.bm25_idf("zzz", docs) == pytest.approx(math.log(1 + 3.5 / 0.5))


def test_bm25_single_doc_hand_value():
    # N=1, df=1, len(doc)==avgdl so the length factor is exactly 1; tf=1 -> 1*2.5/(1+1.5)=1
    assert S.bm25_score(["a"], ["a"], [["a"]]) == pytest.approx(math.log(4 / 3))


def test_bm25_k1_zero_ignores_frequency():
    docs = [["a", "b"], ["c"]]
    assert S.bm25_score(["a"], docs[0], docs, k1=0) == pytest.approx(math.log(2))
    assert S.bm25_score(["a"], ["a", "a", "a", "b"], docs, k1=0) == pytest.approx(math.log(2))
    assert S.bm25_score(["a"], docs[1], docs, k1=0) == 0.0


def test_bm25_b_zero_closed_form():
    docs = [["a", "b"], ["c", "d"], ["e"]]  # N=3, df(a)=1 -> idf = ln(8/3)
    doc = ["a"] * 3 + ["z"] * 10  # length is irrelevant when b=0
    got = S.bm25_score(["a"], doc, docs, k1=1.2, b=0)
    assert got == pytest.approx(math.log(8 / 3) * 3 * 2.2 / 4.2)


def test_bm25_term_frequency_saturates():
    docs = [["a", "b"], ["c", "d"], ["e"]]

    def score(t):
        return S.bm25_score(["a"], ["a"] * t, docs, k1=1.5, b=0)

    s = [score(t) for t in (1, 2, 3, 4, 100)]
    assert all(x < y for x, y in zip(s, s[1:]))  # monotone
    assert (s[1] - s[0]) > (s[2] - s[1]) > (s[3] - s[2])  # diminishing returns
    assert s[4] < math.log(8 / 3) * 2.5  # bounded by idf * (k1 + 1)


def test_bm25_length_normalisation():
    docs = [["a", "x"], ["b", "y", "z", "w"], ["c"]]
    short = ["a", "q"]
    long = ["a"] + ["q"] * 9
    assert S.bm25_score(["a"], short, docs) > S.bm25_score(["a"], long, docs)
    assert S.bm25_score(["a"], short, docs, b=0) == pytest.approx(S.bm25_score(["a"], long, docs, b=0))


def test_bm25_additive_over_distinct_terms_and_repeats_count_once():
    docs = docs_of(CORPUS)
    d = docs[0]
    both = S.bm25_score(["cat", "mat"], d, docs)
    assert both == pytest.approx(S.bm25_score(["cat"], d, docs) + S.bm25_score(["mat"], d, docs))
    assert S.bm25_score(["cat", "cat", "cat"], d, docs) == pytest.approx(S.bm25_score(["cat"], d, docs))


def test_bm25_zero_cases():
    docs = docs_of(CORPUS)
    assert S.bm25_score(["cat"], docs[1], docs) == 0.0
    assert S.bm25_score(["unseenterm"], docs[0], docs) == 0.0
    assert S.bm25_score([], docs[0], docs) == 0.0
    assert S.bm25_score(["cat"], [], docs) == 0.0


def test_bm25_scores_are_nonnegative_even_for_common_terms():
    docs = [["a"], ["a"], ["a"], ["a", "b"]]
    assert S.bm25_score(["a"], docs[0], docs) > 0


# ---------- ranking ----------

def test_rank_toy_corpus_exact_order():
    docs = docs_of(CORPUS)
    ranked = S.rank_bm25(["cat"], docs)
    assert [i for i, _ in ranked] == [2, 3, 0, 1]
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
    assert scores[-1] == 0.0
    assert scores[0] == pytest.approx(S.bm25_score(["cat"], docs[2], docs))


def test_rank_returns_every_doc_and_breaks_ties_by_index():
    docs = [["x", "a"], ["y"], ["x", "a"], ["z"]]
    ranked = S.rank_bm25(["a"], docs)
    assert len(ranked) == 4
    assert [i for i, _ in ranked][:2] == [0, 2]
    assert [i for i, _ in ranked][2:] == [1, 3]


def test_rank_multiple_query_terms_prefers_matching_both():
    docs = docs_of(["red apple pie", "red car", "green apple", "blue sky"])
    order = [i for i, _ in S.rank_bm25(["red", "apple"], docs)]
    assert order[0] == 0 and order[-1] == 3
