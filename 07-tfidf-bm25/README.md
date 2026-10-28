# Week 07: TF-IDF and BM25

## Goal
Implement term weighting and the BM25 ranking function from the formulas, then rank a toy corpus and explain every number.

## Resource
Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft), the vector-semantics / TF-IDF material and the information-retrieval chapter: https://web.stanford.edu/~jurafsky/slp3/ . For BM25 take the formula from the book or from a derivation you can cite; the exact variant used here is fixed below.

## Conventions (fixed so tests can check values)
`docs` is a list of documents; a document is a list of lowercase tokens.

- `tokenize(text)`: lowercase, then maximal runs of ASCII letters and digits. `"Don't PANIC, 42!"` gives `["don", "t", "panic", "42"]`.
- `tf(term, doc) = count(term in doc) / len(doc)`, and `0.0` for an empty doc.
- `idf(term, docs) = ln((N + 1) / (df + 1)) + 1` (smoothed; `N = len(docs)`, `df` = number of docs containing the term). Empty `docs` raises `ValueError`.
- `tfidf_vector(doc, docs) -> dict`: term -> `tf * idf` for each distinct term of `doc`.
- `bm25_idf(term, docs) = ln(1 + (N - df + 0.5) / (df + 0.5))` (never negative).
- `bm25_score(query, doc, docs, k1=1.5, b=0.75)`: sum over the **distinct** terms `t` of `query` of `bm25_idf(t) * tf_raw * (k1 + 1) / (tf_raw + k1 * (1 - b + b * len(doc) / avgdl))`, where `tf_raw` is the raw count in `doc` and `avgdl` is the mean length of `docs`. A term with count 0 contributes 0. `avgdl == 0` or empty `doc` gives `0.0`.
- `rank_bm25(query, docs, k1=1.5, b=0.75) -> list[tuple[int, float]]`: `(doc_index, score)` for **every** doc, best first, ties by lower index first.

## Behaviours the tests check
- Tokenizer edge cases; tf, idf and bm25_idf against hand-computed values.
- Rare terms outweigh common ones in a TF-IDF vector.
- BM25 hand-computed values (single doc corpus, `k1 = 0`, `b = 0`), term-frequency saturation, length normalisation (`b` on/off), additivity over distinct query terms, repeated query terms counted once, unseen terms, empty docs.
- Ranking of a small corpus gives an exact expected order, including a tie and a zero-score doc.

## Aloud-questions (no notes)
1. Why does IDF exist? What goes wrong with raw term frequency alone?
2. What does `k1` control, and what happens as `k1` goes to 0 or to infinity?
3. What does `b` control, and when would you set it to 0?
4. Why does BM25 saturate term frequency, and why is that good for ranking?
5. Why is ranking by TF-IDF dot product biased toward long documents, and how does your code deal with it?

## What I got wrong
