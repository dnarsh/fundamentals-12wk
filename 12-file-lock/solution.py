"""Week 12: cross-process file lock. Write by hand (stdlib only)."""


class FileLock:
    def __init__(self, path):
        raise NotImplementedError

    def acquire(self, timeout=None):
        raise NotImplementedError

    def try_acquire(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError

    def __enter__(self):
        raise NotImplementedError

    def __exit__(self, exc_type, exc, tb):
        raise NotImplementedError
