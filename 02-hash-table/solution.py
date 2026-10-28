"""Week 02: hash table with resizing. Write by hand; do not use dict/set/OrderedDict inside."""


class HashTable:
    def __init__(self, initial_capacity=8):
        raise NotImplementedError

    @property
    def capacity(self):
        raise NotImplementedError

    def put(self, key, value):
        raise NotImplementedError

    def get(self, key):
        raise NotImplementedError

    def delete(self, key):
        raise NotImplementedError

    def __contains__(self, key):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError

    def items(self):
        raise NotImplementedError
