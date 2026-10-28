"""Subprocess helper for the week 12 tests. Test plumbing, not a solution.

usage: _worker.py WEEK_DIR MODE LOCK_PATH [ARG ...]
modes: try | timeout SECONDS | hold | incr COUNTER_PATH ITERATIONS
"""
import sys
import time

week_dir, mode, lock_path = sys.argv[1:4]
args = sys.argv[4:]
sys.path.insert(0, week_dir)
from solution import FileLock  # noqa: E402

lock = FileLock(lock_path)

if mode == "try":
    got = lock.try_acquire()
    print("True" if got else "False", flush=True)
    if got:
        lock.release()
elif mode == "timeout":
    start = time.time()
    try:
        lock.acquire(timeout=float(args[0]))
        print("acquired %.3f" % (time.time() - start), flush=True)
        lock.release()
    except TimeoutError:
        print("timeout %.3f" % (time.time() - start), flush=True)
elif mode == "hold":
    lock.acquire(timeout=10)
    print("ready", flush=True)
    sys.stdin.readline()  # parent either writes a line, closes the pipe or kills us
    lock.release()
elif mode == "incr":
    counter, n = args[0], int(args[1])
    for _ in range(n):
        with lock:
            with open(counter) as f:
                value = int(f.read() or "0")
            time.sleep(0.001)  # widen the race window
            with open(counter, "w") as f:
                f.write(str(value + 1))
else:
    sys.exit("unknown mode " + mode)
