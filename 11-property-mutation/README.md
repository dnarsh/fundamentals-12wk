# Week 11: property-based tests and mutation testing

## Goal
Learn to trust a test suite by attacking it. You write five small functions, then (a) write your **own** property-based tests for them with Hypothesis and (b) run a mutation tester over your code to see which deliberate bugs your tests fail to catch.

## Resources
- Hypothesis: https://hypothesis.readthedocs.io/en/latest/ (the quickstart and the strategies reference)
- mutmut: https://mutmut.readthedocs.io/en/latest/ (check the docs for the configuration your installed version expects)

Both are optional dev installs (`pip install hypothesis mutmut`); the repo's CI installs only pytest, so the Hypothesis tests here skip without it.

## Implement in `solution.py`
- `rle_encode(s) -> list[tuple[str, int]]`: run-length encoding, `"aaabcc"` gives `[("a", 3), ("b", 1), ("c", 2)]`; `""` gives `[]`.
- `rle_decode(pairs) -> str`: the inverse. A count below 1, or a symbol that is not a one-character string, raises `ValueError`.
- `clamp(x, lo, hi)`: `lo` if `x < lo`, `hi` if `x > hi`, else `x`. `lo > hi` raises `ValueError`.
- `merge_sorted(a, b, key=None) -> list`: merge two already-sorted lists in one linear pass (no `sorted`, no `heapq.merge`); stable, so on equal keys items of `a` come first; inputs are not mutated.
- `is_balanced(s) -> bool`: are the brackets `()`, `[]`, `{}` in `s` balanced and properly nested? Other characters are ignored.

## Your own tests (this is the real exercise)
Create `11-property-mutation/my_tests/test_my_properties.py`, starting with `hypothesis = pytest.importorskip("hypothesis")`. Write at least one property per function **before** looking at `tests/test_w11_hypothesis.py`. Then:
1. Run `mutmut` against `solution.py` with **only your tests** (see its docs for how to point it at your test directory).
2. List every surviving mutant in `LEARNING-LOG.md`, and for each say what test would kill it.
3. Add those tests until the survivors are gone or provably equivalent.

## What the repo tests check
- Example-based tests (stdlib only): exact outputs, edge cases, validation, stability, no mutation of inputs, and seeded random round trips.
- Hypothesis tests (skipped unless installed): round trip `decode(encode(s)) == s`, adjacent runs differ, `clamp` is idempotent and in range, `merge_sorted(a, b) == sorted(a + b)`, bracket-balance closure properties.

## Aloud-questions (no notes)
1. What is a property, and how is it different from an example? Give one for `merge_sorted`.
2. What does Hypothesis do when it finds a failing input, and why does shrinking matter?
3. What is a mutant, what does "killed" mean, and what is an equivalent mutant?
4. Why can 100% line coverage still leave surviving mutants?
5. Name a bug a round-trip property cannot catch.

## What I got wrong
