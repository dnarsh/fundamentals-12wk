# Week 01: complexity + timing harness

## Goal
Measure how running time grows with input size, and turn the measurements into a Big-O guess, using only `time.perf_counter` and your own maths.

## Resource
MIT 6.006 Introduction to Algorithms (Spring 2020): https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/ (the asymptotics / models-of-computation lectures and problem set 1).

## Implement in `solution.py`
- `time_call(fn, *args, repeats=5) -> float`: call `fn(*args)` `repeats` times, return the **minimum** wall time in seconds. `repeats < 1` raises `ValueError`.
- `measure(fn, make_input, sizes, repeats=3) -> list[tuple[int, float]]`: for each `n` in `sizes` build `make_input(n)` **outside** the timed region, time `fn(that_input)`, return `[(n, seconds), ...]` in the order of `sizes`.
- `estimate_exponent(points) -> float`: least-squares slope of `log(t)` against `log(n)` (hand-written regression, no `statistics.linear_regression`). Fewer than 2 points, any `n <= 0` or `t <= 0` raises `ValueError`.
- `classify(points) -> str`: return one of `"O(1)"`, `"O(log n)"`, `"O(n)"`, `"O(n log n)"`, `"O(n^2)"`. Must be right on exact synthetic data `t = c * f(n)` for `n = 2**4 ... 2**14`, any positive constant `c`. Decide how; one approach is to compare how constant `t / f(n)` is for each candidate `f`.

## Behaviours the tests check
- `time_call` runs the function exactly `repeats` times, passes args through, and returns the minimum (not mean).
- `measure` calls `make_input` once per size and keeps its cost out of the measurement.
- Exponent of exact `n`, `n^2`, `sqrt(n)`, constant data is 1, 2, 0.5, 0.
- Classification of synthetic `1`, `log n`, `n`, `n log n`, `n^2`, with arbitrary scale factors.
- A real quadratic function measures much slower at 4x the size.

## Aloud-questions (no notes)
1. Why take the minimum of repeats instead of the mean?
2. What does the slope of a log-log plot tell you, and why does n log n give a slope slightly above 1?
3. Why must input construction stay outside the timed region?
4. Name two reasons a timing-based classification can be wrong on real hardware.
5. What is the difference between O(n log n) and Theta(n log n)?

## What I got wrong
