# Week 06: power and sample-size simulation

## Goal
Estimate statistical power by brute-force simulation, then find the sample size you need, and compare with the textbook normal-approximation answer (about 63 per group for effect 0.5 sd at 80% power).

## Resource
Allen Downey, *Think Stats* 2e, chapter 8 and the hypothesis-testing chapter (same book as week 5): https://greenteapress.com/thinkstats2/html/thinkstats2009.html . Use your week 5 thinking about sampling variability. `statistics.NormalDist().inv_cdf` is allowed for the critical value.

## Experiment model (fixed so tests can check it)
Two groups of size `n`. Group A ~ Normal(0, sd). Group B ~ Normal(effect, sd). Test statistic: Welch `t = (mean_B - mean_A) / sqrt(var_A/n + var_B/n)` with sample variances (`n - 1` denominator). Reject when `|t| > z`, where `z = NormalDist().inv_cdf(1 - alpha/2)` (a normal critical value, a deliberate simplification). Power = fraction of simulated experiments that reject.

## Implement in `solution.py`
- `simulate_power(effect, sd, n, alpha=0.05, n_sims=1000, seed=None) -> float`: `n` is per group; uses `random.Random(seed)`; global RNG untouched. `n < 2`, `sd <= 0`, `n_sims < 1` or `alpha` outside (0, 1) raise `ValueError`. Result is `rejections / n_sims`.
- `required_n(effect, sd, power=0.8, alpha=0.05, n_sims=500, seed=None, max_n=500) -> int`: the smallest per-group `n` (>= 2) whose simulated power reaches `power`, found by any search you can justify (linear, or doubling plus bisection). If `max_n` is not enough raise `ValueError`.

## Behaviours the tests check
- Effect 0: rejection rate is about `alpha` (type I error).
- Large effect: power near 1. Power grows with `n` and `effect`, falls with `sd`, and grows with `alpha` (same seed).
- Effect 0.5 sd with n = 64: power within a band around the theoretical 0.81.
- `required_n` for effect 0.5 is in a broad band around 63; for effect 1.0 it is much smaller; the returned n really has decent power on fresh draws.
- Same seed reproduces; validation errors; unreachable target raises.

## Aloud-questions (no notes)
1. Define power, type I error and type II error in one sentence each.
2. Why does the rejection rate at effect 0 equal alpha, and what would it mean if it did not?
3. Why is sample size per group proportional to `(sd/effect)^2`?
4. Why is a simulated "smallest n reaching 80%" itself noisy, and how would you reduce the noise?
5. A pilot with n = 8 per group suggests an effect. Why is using that observed effect to plan the main study risky?

## What I got wrong
