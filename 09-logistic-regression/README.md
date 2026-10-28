# Week 09: logistic regression with gradient descent in plain Python

## Goal
Fit a logistic regression with batch gradient descent using only lists and `math`. No numpy, no scikit-learn. Derive the gradient on paper first; the tests check it numerically so a wrong derivation cannot hide.

## Resource
Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft), the logistic regression chapter (cross-entropy loss, gradient, SGD): https://web.stanford.edu/~jurafsky/slp3/

## Implement in `solution.py`
- `sigmoid(z) -> float`: must not raise `OverflowError` for any finite float (`sigmoid(-1000)` and `sigmoid(1000)` are fine).
- `log_loss(y_true, p) -> float`: mean binary cross-entropy; clamp each `p` into `[1e-15, 1 - 1e-15]`. Different lengths or empty input raise `ValueError`.
- `class LogisticRegression(lr=0.1, n_iter=1000)`
  - `fit(X, y) -> self`: `X` is a list of equal-length feature lists, `y` a list of 0/1. Initialise `weights` (a **list**, one per feature) and `bias` to zero, then run `n_iter` steps of full-batch gradient descent on the mean log loss. Calling `fit` again starts from zero again. Errors (all `ValueError`): empty `X`, `len(X) != len(y)`, ragged rows, labels outside {0, 1}.
  - `predict_proba(X) -> list[float]`: P(y = 1). Before `fit` raise `RuntimeError`; wrong feature count raises `ValueError`.
  - `predict(X, threshold=0.5) -> list[int]`: 1 when probability >= threshold, else 0.
  - Public attributes after fitting: `weights` (list of floats) and `bias` (float). The tests nudge them to check your gradient numerically, so `predict_proba` must read them live.

## Behaviours the tests check
- Sigmoid values, symmetry, bounds, monotonicity, no overflow.
- Log-loss hand values, clamping, errors.
- Separable 1-D and 2-D data are classified correctly; a symmetric dataset gives bias 0 and a positive weight; constant-zero features leave the weight at exactly 0.
- Loss falls as `n_iter` grows. After training on overlapping classes, the numerical gradient of the loss at your parameters is about 0 (this is what catches a wrong gradient).
- Refitting resets; same inputs give identical results.

## Aloud-questions (no notes)
1. Derive d(loss)/dw for one example. Why does it simplify to `(p - y) * x`?
2. Why is cross-entropy used rather than squared error for classification?
3. What happens to the weights on perfectly separable data if you train forever?
4. What does the learning rate do, and what do you see when it is too large?
5. Why is the loss convex here, and what does that guarantee about gradient descent?

## What I got wrong
