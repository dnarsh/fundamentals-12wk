# Week 02: hash table with resizing

## Goal
Build a key-value map from a plain list of buckets (chaining) or slots (open addressing). No `dict`, `set` or `OrderedDict` inside your class; `hash()` and `==` on keys are fine.

## Resource
MIT 6.006 Spring 2020, the hashing lectures: https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/

## Implement in `solution.py`
`class HashTable`
- `__init__(self, initial_capacity=8)`: `initial_capacity < 1` raises `ValueError`.
- `capacity` (property): number of buckets/slots right now.
- `put(key, value)`: insert or overwrite.
- `get(key)`: value, or `KeyError(key)` if missing.
- `delete(key)`: remove, or `KeyError(key)` if missing.
- `key in table`, `len(table)`.
- `keys()` and `items()`: lists, any order, each live entry exactly once.

Resize rule (fixed so tests can check it): after a `put` that adds a **new** key, if `len(self) / capacity > 0.75`, double the capacity and rehash. Deleting never shrinks.

## Behaviours the tests check
- Overwrite does not change `len`; missing key raises `KeyError` on `get` and `delete`.
- Keys that all collide (constant `__hash__`) still work, including delete and reinsert.
- Negative hashes, `None`, tuples, strings, ints as keys; `1` and `1.0` are the same key (Python equality).
- Capacity: 8 after 6 puts, 16 after the 7th; load factor stays <= 0.75 after every put.
- Thousands of random put/get/delete operations agree with a `dict` oracle.
- Lookups of absent keys terminate after heavy churn (tombstones!).

## Aloud-questions (no notes)
1. What happens to lookup cost as the load factor approaches 1, for chaining versus open addressing?
2. Why double the capacity rather than add a constant?
3. Why must you rehash every entry on resize instead of copying buckets?
4. If you use open addressing, why can a delete not simply empty the slot?
5. Why must keys that are `==` have equal hashes, and what breaks if they do not?

## What I got wrong
