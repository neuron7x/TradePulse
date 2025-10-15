# SPDX-License-Identifier: MIT
from analytics.fpma.src.core import main

def test_add():
    assert main.add(2, 3) == 5
