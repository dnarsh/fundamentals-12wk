"""Loads the week's solution.py by path so every week can use the module name `solution`."""
import sys
from pathlib import Path

import pytest


def load(test_file):
    week = str(Path(test_file).resolve().parents[1])
    sys.modules.pop("solution", None)
    sys.path.insert(0, week)
    try:
        return pytest.importorskip("solution")
    finally:
        sys.path.remove(week)
        sys.modules.pop("solution", None)
