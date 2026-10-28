# Week 03: BST + binary heap

## Goal
Implement an unbalanced binary search tree and an array-backed binary min-heap, and trust them because of property tests.

## Resource
MIT 6.006 Spring 2020, binary trees / BSTs and heaps lectures: https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/

## Implement in `solution.py`
`class BST` (set semantics, no balancing)
- `insert(key)`: duplicates are ignored.
- `delete(key)`: `KeyError` if absent. Must handle leaf, one child and two children.
- `key in tree`, `len(tree)`.
- `inorder() -> list`: sorted keys.
- `height() -> int`: number of nodes on the longest root-to-leaf path; empty tree is 0.
- `min()`, `max()`: `ValueError` on an empty tree.

`class MinHeap` (array-backed, smallest first)
- `MinHeap()`, `MinHeap.from_iterable(items)` (build in O(n), not n pushes; must not mutate the input).
- `push(item)`, `pop()` (smallest), `peek()`, `len(heap)`.
- `pop()` and `peek()` on empty raise `IndexError`.

## Behaviours the tests check
- BST `inorder()` is always the sorted distinct keys after any random mix of inserts and deletes (compared with a `set` oracle).
- Deleting a leaf, a one-child node, a two-child node, the root; deleting until empty; failed delete leaves the tree intact.
- Sorted insertion degenerates: height equals n. A balanced insertion order gives height about log2(n).
- Heap pops come out sorted, match `heapq` on random interleaved push/pop, duplicates survive, `from_iterable` agrees with sorting.
- Empty-structure errors.

## Aloud-questions (no notes)
1. Why is the array index of a heap node's children `2i+1` and `2i+2`?
2. Why is building a heap from n items O(n) and not O(n log n)?
3. What is the worst-case height of your BST, and what input causes it?
4. Walk through deleting a BST node with two children. Why the in-order successor (or predecessor)?
5. Why can a heap not give you sorted order by iterating the array?

## What I got wrong
