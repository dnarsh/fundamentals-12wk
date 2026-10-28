import math

import pytest

from _support import load

S = load(__file__)
V = S.Value


def relu(x):
    return x.relu() if isinstance(x, V) else max(0.0, x)


def tanh(x):
    return x.tanh() if isinstance(x, V) else math.tanh(x)


def exp(x):
    return x.exp() if isinstance(x, V) else math.exp(x)


def log(x):
    return x.log() if isinstance(x, V) else math.log(x)


def autograd(f, xs):
    vs = [V(x) for x in xs]
    out = f(*vs)
    out.backward()
    return out.data, [v.grad for v in vs]


def numeric(f, xs, h=1e-6):
    grads = []
    for i in range(len(xs)):
        up = list(xs)
        dn = list(xs)
        up[i] += h
        dn[i] -= h
        grads.append((f(*up) - f(*dn)) / (2 * h))
    return grads


def check(f, xs, tol=1e-5):
    val, grads = autograd(f, xs)
    assert val == pytest.approx(f(*xs), rel=1e-9, abs=1e-12)
    for g, n in zip(grads, numeric(f, xs)):
        assert g == pytest.approx(n, rel=tol, abs=tol)


# ---------- forward values ----------

def test_leaf_defaults():
    v = V(3.0)
    assert v.data == 3.0 and v.grad == 0.0


def test_forward_values_both_operand_orders():
    a, b = V(6.0), V(4.0)
    assert (a + b).data == 10 and (a + 1).data == 7 and (1 + a).data == 7
    assert (a - b).data == 2 and (a - 1).data == 5 and (10 - a).data == 4
    assert (a * b).data == 24 and (a * 2).data == 12 and (2 * a).data == 12
    assert (a / b).data == pytest.approx(1.5) and (a / 2).data == pytest.approx(3) and (12 / a).data == pytest.approx(2)
    assert (-a).data == -6
    assert (a ** 2).data == 36 and (b ** 0.5).data == 2 and (b ** -1).data == 0.25


def test_operations_do_not_mutate_operands():
    a, b = V(2.0), V(3.0)
    _ = a + b, a * b, a - b, a / b, a ** 2, -a, a.tanh(), a.relu()
    assert a.data == 2.0 and b.data == 3.0 and a.grad == 0.0


def test_function_values():
    assert V(0.5).tanh().data == pytest.approx(math.tanh(0.5))
    assert V(1.5).exp().data == pytest.approx(math.exp(1.5))
    assert V(2.5).log().data == pytest.approx(math.log(2.5))
    assert V(-2.0).relu().data == 0 and V(3.0).relu().data == 3


@pytest.mark.parametrize("bad", ["2", None, [2]])
def test_pow_rejects_non_numeric_exponents(bad):
    with pytest.raises(TypeError):
        V(2.0) ** bad


def test_pow_rejects_value_exponent():
    with pytest.raises(TypeError):
        V(2.0) ** V(2.0)


def test_add_rejects_strings():
    with pytest.raises(TypeError):
        V(2.0) + "a"


# ---------- reference expressions ----------

def test_karpathy_neuron():
    x1, x2 = V(2.0), V(0.0)
    w1, w2 = V(-3.0), V(1.0)
    b = V(6.8813735870195432)
    n = x1 * w1 + x2 * w2 + b
    o = n.tanh()
    o.backward()
    assert o.data == pytest.approx(0.7071067811865476, abs=1e-9)
    assert x1.grad == pytest.approx(-1.5, abs=1e-9)
    assert x2.grad == pytest.approx(0.5, abs=1e-9)
    assert w1.grad == pytest.approx(1.0, abs=1e-9)
    assert w2.grad == pytest.approx(0.0, abs=1e-9)
    assert b.grad == pytest.approx(0.5, abs=1e-9)
    assert o.grad == 1.0


def test_micrograd_relu_stress_expression():
    x = V(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    assert y.data == -20.0
    assert x.grad == 46.0


def test_micrograd_more_ops_expression():
    def f(a, b):
        c = a + b
        d = a * b + b ** 3
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + relu(b + a)
        d += 3 * d + relu(b - a)
        e = c - d
        f_ = e ** 2
        g = f_ / 2.0
        g += 10.0 / f_
        return g

    val, _ = autograd(f, [-4.0, 2.0])
    assert val == pytest.approx(24.70408163265306)
    check(f, [-4.0, 2.0], tol=1e-6)


# ---------- gradients per operator (vs finite differences) ----------

@pytest.mark.parametrize(
    "f,xs",
    [
        (lambda a, b: a + b, [1.5, -2.0]),
        (lambda a, b: a - b, [1.5, -2.0]),
        (lambda a, b: a * b, [1.5, -2.0]),
        (lambda a, b: a / b, [1.5, -2.0]),
        (lambda a, b: -a * b, [1.5, 2.5]),
        (lambda a: a ** 3, [-2.0]),
        (lambda a: a ** 2, [-2.0]),
        (lambda a: a ** 0.5, [4.0]),
        (lambda a: a ** -2, [0.7]),
        (lambda a: tanh(a), [0.3]),
        (lambda a: exp(a), [0.8]),
        (lambda a: log(a), [2.5]),
        (lambda a: relu(a), [2.0]),
        (lambda a: relu(a), [-2.0]),
        (lambda a: 3 - a, [1.0]),
        (lambda a: 3 / a, [1.5]),
        (lambda a: 2 * a + 7, [1.0]),
    ],
)
def test_single_operator_gradients(f, xs):
    check(f, xs)


def test_relu_gradient_is_zero_at_zero():
    assert autograd(lambda a: relu(a), [0.0])[1] == [0.0]


def test_composite_expression_gradients():
    def f(a, b, c):
        return tanh(a * b + c) * exp(a - b) / (c ** 2 + 1) + log(b * b + 1) - relu(c - a)

    check(f, [0.4, -0.7, 1.3])


# ---------- graph structure ----------

def test_shared_node_accumulates():
    x = V(3.0)
    y = x * x
    y.backward()
    assert y.data == 9 and x.grad == 6.0


def test_node_used_three_ways():
    check(lambda x: x * x * x + x + x, [2.0])
    assert autograd(lambda x: x + x, [5.0])[1] == [2.0]


def test_diamond_graph():
    def f(x):
        a = x * 2
        b = x + 3
        return a * b

    assert autograd(f, [4.0])[1] == [pytest.approx(2 * 4 + 2 * 7)]


def test_backward_on_leaf():
    x = V(2.0)
    x.backward()
    assert x.grad == 1.0


def test_intermediate_grads_are_filled():
    a, b = V(2.0), V(3.0)
    c = a * b
    d = c + 1
    d.backward()
    assert d.grad == 1.0 and c.grad == 1.0
    assert a.grad == 3.0 and b.grad == 2.0


def test_deep_chain_no_recursion_blowup():
    x = V(1.0)
    y = x
    for _ in range(150):
        y = y * 1.001 + 0.5
    y.backward()
    assert x.grad == pytest.approx(1.001 ** 150, rel=1e-9)


def test_wide_sum():
    xs = [V(float(i)) for i in range(200)]
    total = xs[0]
    for v in xs[1:]:
        total = total + v
    total.backward()
    assert total.data == sum(range(200))
    assert all(v.grad == 1.0 for v in xs)


# ---------- using the engine ----------

def test_gradient_descent_finds_minimum():
    x = V(0.0)
    for _ in range(100):
        x.grad = 0.0
        loss = (x - 3) ** 2
        loss.backward()
        x.data -= 0.1 * x.grad
    assert x.data == pytest.approx(3.0, abs=1e-6)


def test_gradient_descent_fits_a_line():
    pts = [(0.0, 1.0), (1.0, 3.0), (2.0, 5.0), (3.0, 7.0)]  # y = 2x + 1
    w, b = V(0.0), V(0.0)
    for _ in range(400):
        w.grad = b.grad = 0.0
        loss = V(0.0)
        for x, y in pts:
            loss = loss + (w * x + b - y) ** 2
        loss = loss / len(pts)
        loss.backward()
        w.data -= 0.05 * w.grad
        b.data -= 0.05 * b.grad
    assert w.data == pytest.approx(2.0, abs=1e-2)
    assert b.data == pytest.approx(1.0, abs=2e-2)
