# fundamentals-12wk

Twelve weeks of core CS, statistics and AI-fundamentals exercises, **written by hand**, each with a spec and a test suite. Learning in public.

## The honest premise

I direct AI agents well. I am not yet good at writing core algorithms myself, and I want to be. This repo is the fix: every `solution.py` is typed by me, no AI code in it. An AI agent wrote only the scaffold: specs, tests, CI.

## How to use it

1. Open `NN-topic/README.md`: goal, resource, exact signatures, behaviours tested, five aloud-questions.
2. Do the resource first (6-8 hours per week in total).
3. Fill in `NN-topic/solution.py` (it starts as stubs that raise `NotImplementedError`).
4. `python -m pytest NN-topic -q`. **Skipped means not implemented yet.** Once a function exists, its tests run for real.
5. Answer the five aloud-questions without notes. Fill the "what I got wrong" section and `LEARNING-LOG.md`.

Python 3.8+, standard library only. `pytest` is a dev tool. Week 11 additionally uses `hypothesis` and `mutmut` (optional installs; their tests skip without them).

## Rules

- No AI-written code in any `solution.py`.
- AI may critique only **after** I commit my attempt.
- Never edit a test to make it pass; if a test seems wrong, write down why first.

## Progress

| Week | Topic | Status |
| --- | --- | --- |
| 01 | Complexity + timing harness | not started |
| 02 | Hash table with resizing | not started |
| 03 | BST + binary heap | not started |
| 04 | BFS / DFS / Dijkstra on a dependency graph | not started |
| 05 | Bootstrap confidence interval | not started |
| 06 | Power and sample-size simulation | not started |
| 07 | TF-IDF and BM25 | not started |
| 08 | Tiny BPE tokenizer + cosine similarity | not started |
| 09 | Logistic regression, plain-Python gradient descent | not started |
| 10 | Micrograd-style autograd | not started |
| 11 | Property-based and mutation testing | not started |
| 12 | File lock: design note + stress test | not started |

See `PROVENANCE.md` and `LEARNING-LOG.md`. MIT licensed.
