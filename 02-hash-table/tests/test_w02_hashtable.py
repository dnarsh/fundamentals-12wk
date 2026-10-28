import random

import pytest

from _support import load

S = load(__file__)


class Collide:
    """All instances hash alike; equality by value."""

    def __init__(self, v):
        self.v = v

    def __hash__(self):
        return 1

    def __eq__(self, other):
        return isinstance(other, Collide) and self.v == other.v


def test_empty_table():
    t = S.HashTable()
    assert len(t) == 0 and t.capacity == 8
    assert "x" not in t
    assert t.keys() == [] and t.items() == []


def test_put_get_and_overwrite():
    t = S.HashTable()
    t.put("a", 1)
    t.put("b", 2)
    t.put("a", 10)
    assert t.get("a") == 10 and t.get("b") == 2
    assert len(t) == 2


def test_missing_key_raises_keyerror():
    t = S.HashTable()
    with pytest.raises(KeyError):
        t.get("nope")
    with pytest.raises(KeyError):
        t.delete("nope")


def test_delete_then_reinsert():
    t = S.HashTable()
    t.put("a", 1)
    t.delete("a")
    assert "a" not in t and len(t) == 0
    with pytest.raises(KeyError):
        t.get("a")
    t.put("a", 2)
    assert t.get("a") == 2 and len(t) == 1


def test_stores_none_value_distinct_from_missing():
    t = S.HashTable()
    t.put("k", None)
    assert "k" in t and t.get("k") is None


@pytest.mark.parametrize("bad", [0, -3])
def test_bad_initial_capacity(bad):
    with pytest.raises(ValueError):
        S.HashTable(bad)


def test_capacity_property_reflects_initial():
    assert S.HashTable(5).capacity == 5


def test_varied_key_types():
    t = S.HashTable()
    keys = [None, -1, 0, 10 ** 20, "s", (1, 2), (1, (2, 3)), 2.5, frozenset([1])]
    for i, k in enumerate(keys):
        t.put(k, i)
    for i, k in enumerate(keys):
        assert t.get(k) == i
    assert len(t) == len(keys)


def test_python_equality_semantics():
    t = S.HashTable()
    t.put(1, "one")
    t.put(1.0, "float one")  # 1 == 1.0 and equal hashes: same key
    assert len(t) == 1 and t.get(1) == "float one"


def test_negative_hash_keys():
    # hash(-1) == hash(-2) in CPython; both must still be distinct keys
    t = S.HashTable()
    t.put(-1, "a")
    t.put(-2, "b")
    assert t.get(-1) == "a" and t.get(-2) == "b"


def test_all_colliding_keys():
    t = S.HashTable()
    for i in range(50):
        t.put(Collide(i), i)
    assert len(t) == 50
    for i in range(50):
        assert t.get(Collide(i)) == i
    for i in range(0, 50, 2):
        t.delete(Collide(i))
    for i in range(50):
        assert (Collide(i) in t) == (i % 2 == 1)
    t.put(Collide(0), "back")
    assert t.get(Collide(0)) == "back" and t.get(Collide(1)) == 1


def test_resize_boundary_exact():
    t = S.HashTable(8)
    for i in range(6):
        t.put(i, i)
    assert t.capacity == 8  # 6/8 == 0.75 is not > 0.75
    t.put(6, 6)
    assert t.capacity == 16  # 7/8 > 0.75


def test_overwrite_does_not_trigger_resize():
    t = S.HashTable(8)
    for i in range(6):
        t.put(i, i)
    for _ in range(20):
        t.put(0, "again")
    assert t.capacity == 8


def test_load_factor_bounded_and_capacity_only_doubles():
    t = S.HashTable(4)
    seen = {t.capacity}
    for i in range(500):
        t.put(i, i)
        assert len(t) / t.capacity <= 0.75
        seen.add(t.capacity)
    assert seen == {4 * 2 ** k for k in range(len(seen))}


def test_delete_never_shrinks():
    t = S.HashTable()
    for i in range(100):
        t.put(i, i)
    cap = t.capacity
    for i in range(100):
        t.delete(i)
    assert t.capacity == cap and len(t) == 0


def test_keys_and_items_each_live_entry_once():
    t = S.HashTable()
    for i in range(40):
        t.put(i, i * i)
    t.delete(3)
    assert sorted(t.keys()) == [i for i in range(40) if i != 3]
    assert sorted(t.items()) == [(i, i * i) for i in range(40) if i != 3]


@pytest.mark.parametrize("seed", range(5))
def test_random_ops_match_dict_oracle(seed):
    rng = random.Random(seed)
    t, ref = S.HashTable(2), {}
    for _ in range(3000):
        k = rng.randrange(150)
        op = rng.random()
        if op < 0.5:
            v = rng.random()
            t.put(k, v)
            ref[k] = v
        elif op < 0.8:
            if k in ref:
                t.delete(k)
                del ref[k]
            else:
                with pytest.raises(KeyError):
                    t.delete(k)
        else:
            if k in ref:
                assert t.get(k) == ref[k]
            else:
                assert k not in t
        assert len(t) == len(ref)
    assert sorted(t.items()) == sorted(ref.items())


def test_absent_lookups_terminate_after_heavy_churn():
    t = S.HashTable(8)
    for i in range(2000):
        t.put(i, i)
        t.delete(i)
    assert len(t) == 0
    assert 12345 not in t  # must not loop forever on a table full of tombstones
    t.put(12345, "x")
    assert t.get(12345) == "x"
