# Week 10: micrograd-style autograd

## Goal
Build a scalar reverse-mode automatic differentiation engine: a `Value` that records how it was computed, so that `backward()` fills in `d(output)/d(every input)`. Then use it to minimise a function by gradient descent.

## Resource
Andrej Karpathy, "The spelling-out of neural nets and backpropagation: building micrograd": https://youtu.be/VMj-3S1tku0 . Watch it once, then close it and write the engine from the chain rule, not from memory of the video.

## Implement in `solution.py`
`class Value(data, _children=(), _op="")` with
- attributes `data` (the number) and `grad` (starts at `0.0`);
- `+`, `-`, `*`, `/`, unary `-`, and `**` with a plain `int` or `float` exponent (any other exponent type raises `TypeError`); each works with a plain number on **either** side (`2 * v`, `v - 1`, `1 / v`);
- `tanh()`, `exp()`, `log()` (natural log), `relu()` (gradient 0 at exactly 0);
- `backward()`: sets this node's grad to 1.0 and **accumulates** (`+=`) gradients into every ancestor via the chain rule, in reverse topological order. A node used several times (for example `x * x`) must receive the sum of all its contributions.

Operations must return new `Value`s and must not mutate their operands' `data`. Plain numbers that appear in an expression do not need gradients.

## Behaviours the tests check
- Forward values for every operator, both operand orders.
- Karpathy's two reference expressions: the tiny neuron (`tanh`, gradients -1.5 / 0.5 / 1.0 / 0.0 with the famous bias 6.88137...), and the `relu` stress expression whose value is -20 with `x.grad == 46`.
- Every operator's gradient against a central finite difference of the same function on plain floats; a larger mixed expression with `relu`, `**`, `/`.
- Shared nodes in diamond shapes; a 300-node chain (150 multiply-adds); `backward()` on a leaf.
- Gradient descent on `(x - 3)^2` using your engine converges to 3.
- `TypeError` for bad exponents and bad operands.

## Aloud-questions (no notes)
1. State the chain rule as your `backward` uses it: what does each node's `_backward` add to its children?
2. Why must gradients accumulate with `+=` instead of being assigned?
3. Why a topological order, and what goes wrong if you just recurse from the output without one?
4. Why is `d(tanh x)/dx = 1 - tanh(x)^2` convenient to implement?
5. Why do you have to zero the gradients between gradient-descent steps?

## What I got wrong
