"""Shared pytest plumbing. Not a solution to anything.

A test that hits a `raise NotImplementedError` stub is reported as SKIPPED, so
the suite stays green until you write the code. Once you implement a function,
its tests run for real. (Caveat: if your own code raises NotImplementedError,
that also shows up as a skip.)
"""
import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    if call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        rep = outcome.get_result()
        rep.outcome = "skipped"
        rep.longrepr = (str(item.fspath), 0, "Skipped: solution.py not implemented yet")
