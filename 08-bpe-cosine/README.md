# Week 08: tiny BPE tokenizer + cosine similarity

## Goal
Learn byte-pair-encoding merges from a text by counting adjacent pairs, apply them to new text, and compare two texts as bags of tokens with cosine similarity.

## Resources
- Andrej Karpathy, "Let's build the GPT Tokenizer": https://youtu.be/zduSFxRajkE
- Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft), the tokenization (BPE) and vector-semantics (cosine) sections: https://web.stanford.edu/~jurafsky/slp3/

## Conventions (fixed so tests can check exact outputs)
- A text is split into **words** with `text.split()`. Tokens never cross word boundaries and whitespace is dropped. Each word starts as a list of single characters (no end-of-word marker, no bytes: characters).
- A **pair** is two adjacent tokens inside one word. Counts are summed over all words, weighted by how often each word occurs, and adjacent pairs are counted naively (in `aaa` the pair `(a, a)` occurs twice).
- Each merge picks the pair with the highest count; ties go to the lexicographically smallest pair (Python tuple order). Merging a pair replaces non-overlapping occurrences scanning left to right (`aaa` with `(a, a)` becomes `aa`, `a`).
- Training stops early when no pair is left to merge.

## Implement in `solution.py`
- `cosine_similarity(u, v) -> float`: `u`, `v` are sparse vectors, `dict[str, float]`; absent keys are 0. If either vector is all zeros return `0.0`. Inputs are not mutated.
- `train_bpe(text, num_merges) -> list[tuple[str, str]]`: the merges in the order learned (at most `num_merges`). Negative `num_merges` raises `ValueError`.
- `bpe_encode(text, merges) -> list[str]`: tokens of all words in order, applying `merges` in the order given.
- `bpe_cosine(a, b, merges) -> float`: encode both texts, build token-count dicts, return their cosine similarity.

## Behaviours the tests check
- The classic `aaabdaaabac` example gives merges `(a,a)`, `(a,b)`, `(aa,ab)` and encodes to `aaab d aaab a c`.
- Frequency weighting, tie-breaking, overlap counting, word boundaries, early stop.
- Merge order matters when encoding; unseen characters and unicode survive; `"".join(tokens)` equals the text with whitespace removed; more merges never produce more tokens.
- Cosine: hand values (0.6, 0.5), identical = 1, orthogonal = 0, scale invariance, symmetry, negatives, zero vectors.
- `bpe_cosine` distinguishes `ab` from `ba` only when a merge splits them.

## Aloud-questions (no notes)
1. Why does BPE start from characters and what do the first few merges tend to capture?
2. Why must `bpe_encode` apply merges in training order, not by pair frequency in the new text?
3. What changes if you run BPE on bytes instead of characters, and why does GPT-style tokenization do that?
4. Why is cosine similarity preferred over raw dot product or Euclidean distance for comparing documents?
5. What does a cosine of 0 between two token-count vectors say, and what does it not say?

## What I got wrong
