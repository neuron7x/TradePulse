# SPDX-License-Identifier: MIT
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.fpma.src.core import main

def test_add():
    assert main.add(2,3)==5
