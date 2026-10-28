import subprocess
import sys
import time
from pathlib import Path

import pytest

from _support import load

S = load(__file__)

WEEK = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).resolve().parent / "_worker.py"


@pytest.fixture(autouse=True)
def implemented(tmp_path):
    """Skip everything until FileLock exists (stubs raise NotImplementedError)."""
    probe = S.FileLock(str(tmp_path / "probe.lock"))
    probe.try_acquire()
    probe.release()


def spawn(mode, lock_path, *args, **kw):
    cmd = [sys.executable, str(WORKER), str(WEEK), mode, str(lock_path)] + [str(a) for a in args]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, universal_newlines=True, **kw)


def run(mode, lock_path, *args):
    p = spawn(mode, lock_path, *args)
    out, err = p.communicate(timeout=60)
    assert p.returncode == 0, err
    return out.strip()


def other_can_lock(lock_path):
    return run("try", lock_path) == "True"


def test_acquire_release_cycle(tmp_path):
    path = tmp_path / "a.lock"
    lock = S.FileLock(str(path))
    assert lock.try_acquire() is True
    lock.release()
    assert lock.try_acquire() is True
    lock.release()
    lock.acquire()
    lock.release()
    lock.acquire(timeout=1)
    lock.release()
    assert path.exists()  # the lock file is created on demand


def test_other_process_blocked_while_held_and_free_after_release(tmp_path):
    path = tmp_path / "b.lock"
    assert other_can_lock(path) is True  # nobody holds it
    lock = S.FileLock(str(path))
    lock.acquire()
    try:
        assert other_can_lock(path) is False
    finally:
        lock.release()
    assert other_can_lock(path) is True


def test_context_manager_holds_then_releases(tmp_path):
    path = tmp_path / "c.lock"
    with S.FileLock(str(path)):
        assert other_can_lock(path) is False
    assert other_can_lock(path) is True


def test_context_manager_releases_when_body_raises(tmp_path):
    path = tmp_path / "d.lock"
    with pytest.raises(KeyError):
        with S.FileLock(str(path)):
            raise KeyError("boom")
    assert other_can_lock(path) is True


def test_release_without_holding_raises(tmp_path):
    lock = S.FileLock(str(tmp_path / "e.lock"))
    with pytest.raises(RuntimeError):
        lock.release()
    lock.acquire()
    lock.release()
    with pytest.raises(RuntimeError):
        lock.release()


def test_double_acquire_on_same_object_raises(tmp_path):
    lock = S.FileLock(str(tmp_path / "f.lock"))
    lock.acquire()
    try:
        with pytest.raises(RuntimeError):
            lock.acquire()
        with pytest.raises(RuntimeError):
            lock.try_acquire()
    finally:
        lock.release()


def test_timeout_raises_after_roughly_the_timeout(tmp_path):
    path = tmp_path / "g.lock"
    lock = S.FileLock(str(path))
    lock.acquire()
    try:
        kind, elapsed = run("timeout", path, 0.4).split()
    finally:
        lock.release()
    assert kind == "timeout"
    assert 0.3 <= float(elapsed) < 5


def test_zero_timeout_tries_once_and_fails_fast(tmp_path):
    path = tmp_path / "h.lock"
    with S.FileLock(str(path)):
        kind, elapsed = run("timeout", path, 0).split()
    assert kind == "timeout" and float(elapsed) < 1


def test_blocking_acquire_wakes_up_when_holder_releases(tmp_path):
    path = tmp_path / "i.lock"
    lock = S.FileLock(str(path))
    lock.acquire()
    waiter = spawn("timeout", path, 20)
    time.sleep(0.5)
    assert waiter.poll() is None  # still waiting
    lock.release()
    out, err = waiter.communicate(timeout=30)
    assert waiter.returncode == 0, err
    kind, elapsed = out.split()
    assert kind == "acquired" and 0.4 <= float(elapsed) < 15


def test_stress_four_processes_lose_no_updates(tmp_path):
    path = tmp_path / "j.lock"
    counter = tmp_path / "counter.txt"
    counter.write_text("0")
    procs = [spawn("incr", path, counter, 25) for _ in range(4)]
    for p in procs:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
    assert int(counter.read_text()) == 100


def test_killed_holder_does_not_wedge_the_lock(tmp_path):
    path = tmp_path / "k.lock"
    holder = spawn("hold", path)
    try:
        assert holder.stdout.readline().strip() == "ready"
        assert other_can_lock(path) is False
        holder.kill()
        holder.wait(timeout=10)
        deadline = time.time() + 10
        freed = False
        while time.time() < deadline and not freed:
            freed = other_can_lock(path)
            if not freed:
                time.sleep(0.1)
        assert freed, "lock stayed held after its owner process was killed"
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait()
        for stream in (holder.stdin, holder.stdout, holder.stderr):
            stream.close()


def test_design_note_if_present():
    note = WEEK / "DESIGN-NOTE.md"
    if not note.exists():
        pytest.skip("write 12-file-lock/DESIGN-NOTE.md (required by the README)")
    text = note.read_text(encoding="utf-8")
    assert len(text.split()) >= 150
    assert "crash" in text.lower() and "windows" in text.lower()
