# Week 12: file lock design note + stress test

## Goal
Build a cross-process mutual-exclusion lock on a file, write a short design note defending your choices, and make the repo's multi-process stress test pass.

## Resources
- OSTEP (Operating Systems: Three Easy Pieces), the locks and concurrency chapters: https://pages.cs.wisc.edu/~remzi/OSTEP/
- Python `fcntl` docs: https://docs.python.org/3/library/fcntl.html . **`fcntl` is Unix-only.** On Windows use `msvcrt.locking` (see the `msvcrt` docs), or another approach you can justify. Your `solution.py` must work on the platform you run the tests on; handling both (import-time `try`/`except ImportError`, or `sys.platform` check) is part of the exercise. The repo's CI runs on Linux.

## Implement in `solution.py`
`class FileLock(path)`
- `acquire(timeout=None)`: block until the lock is held. With a `timeout` in seconds, raise `TimeoutError` if it cannot be taken in time. `timeout=0` means try once.
- `try_acquire() -> bool`: take the lock if free, never block, return whether you now hold it.
- `release()`: give it up. Releasing a lock this object does not hold raises `RuntimeError`.
- `with FileLock(path): ...`: acquire on enter, release on exit, **including when the body raises**.
- Calling `acquire` or `try_acquire` on an object that already holds the lock raises `RuntimeError` (not re-entrant).
- Creates the lock file if it does not exist. A lock must be released by the operating system if the holding process dies (crash safety); a design that leaves a stale lock file that blocks everyone forever fails the crash test on purpose.

Tests use separate Python processes (via `tests/_worker.py`, which only imports your `FileLock`). They never rely on two lock objects in one process excluding each other, because that depends on your locking primitive.

## Design note (you write it)
Create `12-file-lock/DESIGN-NOTE.md` (150+ words). Cover: which primitive you chose and why (`flock`, `lockf`, `msvcrt.locking`, `O_EXCL` lock file), what happens when the holder crashes, how waiting works (polling versus blocking, and the cost of each), and what breaks on Windows or on network filesystems. The test only checks that the note exists, is long enough and mentions crashes and Windows; the content is on you.

## Behaviours the tests check
- Acquire, release, re-acquire; `try_acquire` result; context manager releases on exceptions (verified from another process).
- Another process cannot take a held lock and can once it is released.
- A blocked `acquire(timeout=...)` raises `TimeoutError` roughly on time; a blocked `acquire()` wakes up when the holder releases.
- Four processes doing a read-modify-write on one counter file lose no updates.
- A holder killed without releasing does not wedge the lock.
- Misuse errors (`release` while not holding, double acquire).

## Aloud-questions (no notes)
1. What is a race condition, and why does a counter without a lock lose updates?
2. Why can `if not exists(lockfile): create(lockfile)` never be a correct lock?
3. What is the difference between advisory and mandatory locking, and which do you have?
4. What happens to your lock if the process is killed, and why does that matter?
5. Spinning, sleeping and blocking in the kernel: what is the trade-off in each for `acquire`?

## What I got wrong
