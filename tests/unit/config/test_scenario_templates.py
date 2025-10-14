from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from interfaces.scenario_templates import load_scenario_templates


def test_templates_loadable_by_python_and_node() -> None:
    """Ensure the shared JSON can be parsed by Python and Node.js toolchains."""

    templates = load_scenario_templates()
    assert templates, 'scenario templates should not be empty'

    repo_root = Path(__file__).resolve().parents[3]
    json_path = repo_root / 'configs' / 'scenarios' / 'templates.json'
    raw = json.loads(json_path.read_text(encoding='utf-8'))
    assert len(raw) == len(templates)

    script = """
const templates = require(process.argv[1]);
if (!Array.isArray(templates)) {
  process.exit(1);
}
process.stdout.write(String(templates.length));
"""

    if shutil.which('node') is None:
        pytest.skip('node binary is required to validate TypeScript/Node consumption of templates')

    result = subprocess.run(
        ['node', '-e', script, str(json_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == str(len(templates))
