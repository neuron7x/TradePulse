# SPDX-License-Identifier: MIT
"""Regression tests for the lightweight YAML shim."""
from __future__ import annotations

from yaml import safe_load


def test_safe_load_parses_nested_mappings_and_lists() -> None:
    text = """
    root:
      flag: true
      numbers: [1, 2, 3.5]
      child:
        name: example
        threshold: 0.25
    """
    config = safe_load(text)
    assert config["root"]["flag"] is True
    assert config["root"]["numbers"] == [1, 2, 3.5]
    assert config["root"]["child"]["threshold"] == 0.25


def test_safe_load_handles_null_and_empty_scalars() -> None:
    text = """
    items:
      value: null
      empty: ""
    """
    config = safe_load(text)
    assert config["items"]["value"] is None
    assert config["items"]["empty"] == ""
