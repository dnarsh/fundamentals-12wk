"""Week 03: binary search tree and binary min-heap. Write by hand; no heapq, no bisect."""


class BST:
    def __init__(self):
        raise NotImplementedError

    def insert(self, key):
        raise NotImplementedError

    def delete(self, key):
        raise NotImplementedError

    def __contains__(self, key):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def inorder(self):
        raise NotImplementedError

    def height(self):
        raise NotImplementedError

    def min(self):
        raise NotImplementedError

    def max(self):
        raise NotImplementedError


class MinHeap:
    def __init__(self):
        raise NotImplementedError

    @classmethod
    def from_iterable(cls, items):
        raise NotImplementedError

    def push(self, item):
        raise NotImplementedError

    def pop(self):
        raise NotImplementedError

    def peek(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
