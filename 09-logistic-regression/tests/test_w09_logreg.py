import math
import random

import pytest

from _support import load

S = load(__file__)


def blobs(seed, n, sep, d=2):
    """n points per class, class 0 centred at -sep, class 1 at +sep, unit noise."""
    rng = random.Random(seed)
    X, y = [], []
    for label, c in ((0, -sep), (1, sep)):
        for _ in range(n):
            X.append([rng.gauss(c, 1.0) for _ in range(d)])
            y.append(label)
    return X, y


def accuracy(model, X, y):
    return sum(p == t for p, t in zip(model.predict(X), y)) / len(y)


# ---------- sigmoid / log_loss ----------

def test_sigmoid_values():
    assert S.sigmoid(0) == pytest.approx(0.5)
    assert S.sigmoid(1) == pytest.approx(1 / (1 + math.exp(-1)))
    assert S.sigmoid(-3) == pytest.approx(1 / (1 + math.exp(3)))


@pytest.mark.parametrize("z", [0.3, 2, 10, 35])
def test_sigmoid_symmetry(z):
    assert S.sigmoid(-z) == pytest.approx(1 - S.sigmoid(z), abs=1e-12)


@pytest.mark.parametrize("z", [-1000, -745, -710, 710, 745, 1000, 1e6, -1e6])
def test_sigmoid_extremes_do_not_overflow(z):
    v = S.sigmoid(z)
    assert 0.0 <= v <= 1.0
    assert abs(v - (1.0 if z > 0 else 0.0)) < 1e-12


def test_sigmoid_is_monotone_increasing():
    zs = [-50 + i * 0.5 for i in range(201)]
    vs = [S.sigmoid(z) for z in zs]
    assert all(a <= b for a, b in zip(vs, vs[1:]))


def test_log_loss_hand_values():
    assert S.log_loss([1, 0], [0.5, 0.5]) == pytest.approx(math.log(2))
    assert S.log_loss([1, 0], [0.9, 0.2]) == pytest.approx((-math.log(0.9) - math.log(0.8)) / 2)


def test_log_loss_perfect_and_clamped():
    assert 0 <= S.log_loss([1, 0], [1.0, 0.0]) < 1e-9
    wrong = S.log_loss([1], [0.0])
    assert math.isfinite(wrong) and 30 < wrong < 40  # -ln(1e-15) is about 34.5
    assert math.isfinite(S.log_loss([0], [1.0]))


@pytest.mark.parametrize("y,p", [([], []), ([1], [0.5, 0.5]), ([1, 0], [0.5])])
def test_log_loss_validation(y, p):
    with pytest.raises(ValueError):
        S.log_loss(y, p)


# ---------- model ----------

def test_separable_1d():
    X, y = [[-2], [-1], [1], [2]], [0, 0, 1, 1]
    m = S.LogisticRegression(lr=0.5, n_iter=500).fit(X, y)
    assert m.predict(X) == y
    p = m.predict_proba(X)
    assert p[0] < p[1] < 0.5 < p[2] < p[3]
    assert m.weights[0] > 0


def test_fit_returns_self_and_exposes_weights_and_bias():
    m = S.LogisticRegression(lr=0.1, n_iter=10)
    assert m.fit([[0.0, 1.0], [1.0, 0.0]], [0, 1]) is m
    assert isinstance(m.weights, list) and len(m.weights) == 2
    assert isinstance(m.bias, float)


def test_predict_returns_ints_zero_one():
    X, y = blobs(1, 20, 2.0)
    out = S.LogisticRegression().fit(X, y).predict(X)
    assert set(out) <= {0, 1} and all(isinstance(v, int) for v in out)


def test_symmetric_data_gives_zero_bias_and_positive_weight():
    m = S.LogisticRegression(lr=0.5, n_iter=1000).fit([[-1.0], [1.0]], [0, 1])
    assert abs(m.bias) < 1e-9
    assert m.weights[0] > 0.5


def test_zero_features_leave_weight_at_zero_and_bias_positive():
    m = S.LogisticRegression(lr=0.5, n_iter=200).fit([[0.0], [0.0], [0.0]], [1, 1, 0])
    assert m.weights == [0.0]
    assert m.bias > 0  # two thirds positives
    assert m.predict_proba([[0.0]])[0] == pytest.approx(2 / 3, abs=0.02)


def test_two_d_blobs_are_classified():
    X, y = blobs(2, 100, 2.0)
    m = S.LogisticRegression(lr=0.1, n_iter=1000).fit(X, y)
    assert accuracy(m, X, y) >= 0.95
    assert S.log_loss(y, m.predict_proba(X)) < 0.3


def test_irrelevant_feature_gets_small_weight():
    rng = random.Random(7)
    X, y = [], []
    for _ in range(150):
        label = rng.randint(0, 1)
        X.append([rng.gauss(3.0 if label else -3.0, 1.0), rng.gauss(0.0, 1.0)])
        y.append(label)
    m = S.LogisticRegression(lr=0.1, n_iter=1500).fit(X, y)
    assert m.weights[0] > 0 and abs(m.weights[1]) < m.weights[0] / 2


def test_loss_decreases_with_more_iterations():
    X, y = blobs(3, 60, 0.8)
    losses = []
    for n in (1, 10, 100, 600):
        m = S.LogisticRegression(lr=0.1, n_iter=n).fit(X, y)
        losses.append(S.log_loss(y, m.predict_proba(X)))
    assert all(a > b for a, b in zip(losses, losses[1:]))
    assert losses[0] < math.log(2) + 1e-9  # one step already helps (or at worst equals chance)


def test_gradient_is_numerically_zero_at_the_optimum():
    X, y = blobs(4, 30, 0.5)  # overlapping classes: a finite optimum exists
    m = S.LogisticRegression(lr=1.0, n_iter=3000).fit(X, y)

    def loss():
        return S.log_loss(y, m.predict_proba(X))

    h = 1e-5
    for j in range(len(m.weights)):
        orig = m.weights[j]
        m.weights[j] = orig + h
        up = loss()
        m.weights[j] = orig - h
        down = loss()
        m.weights[j] = orig
        assert abs((up - down) / (2 * h)) < 5e-3
    ob = m.bias
    m.bias = ob + h
    up = loss()
    m.bias = ob - h
    down = loss()
    m.bias = ob
    assert abs((up - down) / (2 * h)) < 5e-3


def test_first_step_matches_hand_gradient():
    # one step from zero on X=[[1],[0]], y=[1,0]: p=0.5 everywhere,
    # d/dw = mean((p-y)x) = -0.25, d/db = mean(p-y) = 0, so w = lr*0.25
    m = S.LogisticRegression(lr=0.4, n_iter=1).fit([[1.0], [0.0]], [1, 0])
    assert m.weights[0] == pytest.approx(0.1)
    assert m.bias == pytest.approx(0.0, abs=1e-12)


def test_refit_resets_and_is_deterministic():
    XA, yA = blobs(5, 20, 1.5)
    XB, yB = blobs(6, 20, 1.5)
    m = S.LogisticRegression(lr=0.1, n_iter=50)
    m.fit(XA, yA)
    m.fit(XB, yB)
    fresh = S.LogisticRegression(lr=0.1, n_iter=50).fit(XB, yB)
    assert m.weights == fresh.weights and m.bias == fresh.bias


def test_threshold_parameter():
    X, y = blobs(8, 30, 2.0)
    m = S.LogisticRegression().fit(X, y)
    assert set(m.predict(X, threshold=0.0)) == {1}
    assert sum(m.predict(X, threshold=0.999999)) <= sum(m.predict(X))


def test_validation_errors():
    m = S.LogisticRegression()
    with pytest.raises(RuntimeError):
        m.predict_proba([[1.0]])
    with pytest.raises(RuntimeError):
        m.predict([[1.0]])
    with pytest.raises(ValueError):
        m.fit([], [])
    with pytest.raises(ValueError):
        m.fit([[1.0], [2.0]], [1])
    with pytest.raises(ValueError):
        m.fit([[1.0], [2.0, 3.0]], [0, 1])
    with pytest.raises(ValueError):
        m.fit([[1.0], [2.0]], [0, 2])
    m.fit([[1.0], [2.0]], [0, 1])
    with pytest.raises(ValueError):
        m.predict_proba([[1.0, 2.0]])
