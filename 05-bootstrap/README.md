# Week 05: bootstrap confidence interval

## Goal
Build the percentile bootstrap yourself, then apply it to a small-n experiment of your own (for example 8 vs 8 timing runs of two implementations, or 12 pass/fail scores of two prompts) and write down what the interval does and does not tell you.

## Resource
Allen Downey, *Think Stats* 2e, chapter 8 (estimation, resampling): https://greenteapress.com/thinkstats2/html/thinkstats2009.html

## Implement in `solution.py`
- `bootstrap_distribution(data, stat, n_resamples=2000, seed=None) -> list[float]`: draw `n_resamples` resamples of `data`, each of the **same size as data, with replacement**, return `stat(resample)` for each. Uses `random.Random(seed)` so the global RNG state is untouched and a fixed seed reproduces the output exactly.
- `bootstrap_ci(data, stat=None, n_resamples=2000, alpha=0.05, seed=None) -> tuple[float, float]`: percentile interval `(low, high)` at confidence `1 - alpha`. `stat=None` means the mean.
- `bootstrap_diff_ci(a, b, stat=None, ...) -> tuple[float, float]`: interval for `stat(a) - stat(b)`, resampling `a` and `b` independently.
- Validation: empty data raises `ValueError`; `alpha` outside the open interval (0, 1) raises `ValueError`; `n_resamples < 1` raises `ValueError`.
- Do not call `scipy`/`numpy`. `statistics.mean` and `random` are fine.

The percentile-index convention is yours to choose and document; tests allow for that.

## Behaviours the tests check (properties, no reference numbers)
- Every resample has the same size; resampled values come from the data; the number of results equals `n_resamples`.
- Same seed gives identical output; different seeds differ; global `random` state is not consumed.
- `low <= high`; a constant dataset gives a zero-width interval at that constant; the interval lies within `[min(data), max(data)]` for the mean.
- Smaller `alpha` gives a wider (nested) interval for the same seed.
- Width shrinks as n grows (about 1/sqrt(n)).
- Simulated coverage of a known true mean is near the nominal 95%.
- Works for median and an arbitrary custom statistic; the difference CI is centred near the true difference and excludes 0 for a big effect but covers 0 for none.

## Aloud-questions (no notes)
1. Why resample with replacement, and why the same size as the original?
2. What does a 95% bootstrap interval promise, and what does it not promise?
3. Why does the percentile bootstrap undercover for very small n or for statistics like the max?
4. What is the bootstrap standard error, and how does it relate to the CI?
5. For your small-n experiment: what would make you distrust the interval?

## What I got wrong
