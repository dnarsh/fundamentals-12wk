import heapq
import random

import pytest

from _support import load

S = load(__file__)


def build(keys):
    t = S.BST()
    for k in keys:
        t.insert(k)
    return t


# ---------- BST ----------

def test_empty_bst():
    t = S.BST()
    assert len(t) == 0 and t.inorder() == [] and t.height() == 0
    assert 1 not in t
    with pytest.raises(ValueError):
        t.min()
    with pytest.raises(ValueError):
        t.max()
    with pytest.raises(KeyError):
        t.delete(1)


def test_insert_contains_len_and_duplicates_ignored():
    t = build([5, 3, 8, 3, 5])
    assert len(t) == 3
    assert 3 in t and 8 in t and 4 not in t
    assert t.inorder() == [3, 5, 8]


def test_min_max():
    t = build([5, 3, 8, 1, 9, 7])
    assert t.min() == 1 and t.max() == 9


def test_height_degenerate_and_balanced():
    assert build(range(50)).height() == 50
    assert build(range(50, 0, -1)).height() == 50
    assert build([4, 2, 6, 1, 3, 5, 7]).height() == 3
    assert build([1]).height() == 1


def test_delete_leaf_one_child_two_children_root():
    t = build([50, 30, 70, 20, 40, 60, 80, 35])
    t.delete(20)  # leaf
    assert t.inorder() == [30, 35, 40, 50, 60, 70, 80]
    t.delete(40)  # one child (35)
    assert t.inorder() == [30, 35, 50, 60, 70, 80]
    t.delete(70)  # two children
    assert t.inorder() == [30, 35, 50, 60, 80]
    t.delete(50)  # root with two children
    assert t.inorder() == [30, 35, 60, 80]
    assert len(t) == 4
    for k in (30, 35, 60, 80):
        assert k in t
    for k in (20, 40, 70, 50):
        assert k not in t


def test_delete_everything():
    keys = [8, 4, 12, 2, 6, 10, 14]
    t = build(keys)
    for k in keys:
        t.delete(k)
    assert len(t) == 0 and t.inorder() == [] and t.height() == 0
    t.insert(3)
    assert t.inorder() == [3]


def test_delete_missing_raises_and_leaves_tree_intact():
    t = build([2, 1, 3])
    with pytest.raises(KeyError):
        t.delete(99)
    assert t.inorder() == [1, 2, 3]


def test_delete_single_node_root():
    t = build([7])
    t.delete(7)
    assert len(t) == 0 and 7 not in t


@pytest.mark.parametrize("seed", range(6))
def test_bst_random_ops_match_set_oracle(seed):
    rng = random.Random(seed)
    t, ref = S.BST(), set()
    for _ in range(1500):
        k = rng.randrange(80)
        if rng.random() < 0.55:
            t.insert(k)
            ref.add(k)
        elif k in ref:
            t.delete(k)
            ref.remove(k)
        else:
            with pytest.raises(KeyError):
                t.delete(k)
        assert len(t) == len(ref)
        assert (k in t) == (k in ref)
    assert t.inorder() == sorted(ref)
    if ref:
        assert t.min() == min(ref) and t.max() == max(ref)
        # n nodes need at least ceil(log2(n + 1)) levels, i.e. n.bit_length()
        assert len(ref).bit_length() <= t.height() <= len(ref)


@pytest.mark.parametrize("seed", range(4))
def test_bst_height_bounds_random_insertion(seed):
    keys = random.Random(seed).sample(range(10000), 255)
    assert 8 <= build(keys).height() <= 255


def test_bst_strings():
    t = build(["pear", "apple", "fig"])
    assert t.inorder() == ["apple", "fig", "pear"]
    assert t.min() == "apple"


# ---------- MinHeap ----------

def test_empty_heap_errors():
    h = S.MinHeap()
    assert len(h) == 0
    with pytest.raises(IndexError):
        h.pop()
    with pytest.raises(IndexError):
        h.peek()


def test_push_peek_pop_basic():
    h = S.MinHeap()
    for x in [5, 3, 8, 1]:
        h.push(x)
    assert h.peek() == 1 and len(h) == 4
    assert [h.pop() for _ in range(4)] == [1, 3, 5, 8]
    assert len(h) == 0


def test_peek_does_not_remove():
    h = S.MinHeap()
    h.push(2)
    assert h.peek() == 2 and h.peek() == 2 and len(h) == 1


def test_duplicates_survive():
    h = S.MinHeap()
    for x in [2, 2, 1, 2, 1]:
        h.push(x)
    assert [h.pop() for _ in range(5)] == [1, 1, 2, 2, 2]


def test_tuples_as_priority_items():
    h = S.MinHeap()
    for item in [(2, "b"), (1, "z"), (1, "a")]:
        h.push(item)
    assert h.pop() == (1, "a")


@pytest.mark.parametrize("seed", range(5))
def test_heapsort_property(seed):
    rng = random.Random(seed)
    xs = [rng.randrange(-50, 50) for _ in range(rng.randrange(0, 300))]
    h = S.MinHeap()
    for x in xs:
        h.push(x)
    assert [h.pop() for _ in range(len(xs))] == sorted(xs)


@pytest.mark.parametrize("seed", range(5))
def test_interleaved_ops_match_heapq(seed):
    rng = random.Random(seed)
    h, ref = S.MinHeap(), []
    for _ in range(2000):
        if ref and rng.random() < 0.4:
            assert h.pop() == heapq.heappop(ref)
        else:
            x = rng.randrange(1000)
            h.push(x)
            heapq.heappush(ref, x)
        assert len(h) == len(ref)
        if ref:
            assert h.peek() == ref[0]


@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 100])
def test_from_iterable_matches_sorted(n):
    rng = random.Random(n)
    xs = [rng.randrange(1000) for _ in range(n)]
    h = S.MinHeap.from_iterable(xs)
    assert len(h) == n
    assert [h.pop() for _ in range(n)] == sorted(xs)


def test_from_iterable_does_not_mutate_input_and_accepts_generators():
    xs = [3, 1, 2]
    S.MinHeap.from_iterable(xs)
    assert xs == [3, 1, 2]
    h = S.MinHeap.from_iterable(x for x in (9, 4, 6))
    assert h.pop() == 4
