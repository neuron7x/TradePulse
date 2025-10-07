# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_fixture_path = Path(__file__).parent / "fixtures" / "conftest.py"
spec = importlib.util.spec_from_file_location("tradepulse_tests_fixtures", _fixture_path)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load fixtures from {_fixture_path}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

globals().update({name: getattr(module, name) for name in dir(module) if not name.startswith("__")})
