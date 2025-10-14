"""Helpers for loading shared scenario templates across Python interfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TEMPLATES_PATH = _ROOT / 'configs' / 'scenarios' / 'templates.json'


class ScenarioConfig(TypedDict):
    """Structure of a scenario defaults block."""

    initialBalance: float
    riskPerTrade: float
    maxPositions: int
    timeframe: str


class ScenarioTemplate(TypedDict):
    """Structure of a scenario template entry."""

    id: str
    label: str
    description: str
    defaults: ScenarioConfig
    notes: list[str]


def _validate_template(payload: dict[str, Any], index: int) -> ScenarioTemplate:
    try:
        defaults = payload['defaults']
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f'Scenario template at index {index} is missing defaults') from exc

    if not isinstance(defaults, dict):
        raise TypeError(f'Scenario template defaults at index {index} must be an object')

    required_defaults = {'initialBalance', 'riskPerTrade', 'maxPositions', 'timeframe'}
    if not required_defaults.issubset(defaults):
        missing = required_defaults.difference(defaults)
        raise ValueError(f'Scenario template defaults at index {index} missing keys: {sorted(missing)}')

    if not isinstance(defaults['initialBalance'], (int, float)):
        raise TypeError(f'initialBalance at index {index} must be numeric')
    if not isinstance(defaults['riskPerTrade'], (int, float)):
        raise TypeError(f'riskPerTrade at index {index} must be numeric')
    if not isinstance(defaults['maxPositions'], (int, float)):
        raise TypeError(f'maxPositions at index {index} must be numeric')
    if not isinstance(defaults['timeframe'], str):
        raise TypeError(f'timeframe at index {index} must be a string')

    notes = payload.get('notes', [])
    if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
        raise TypeError(f'notes at index {index} must be a list of strings')

    template: ScenarioTemplate = {
        'id': str(payload.get('id', '')),
        'label': str(payload.get('label', '')),
        'description': str(payload.get('description', '')),
        'defaults': {
            'initialBalance': float(defaults['initialBalance']),
            'riskPerTrade': float(defaults['riskPerTrade']),
            'maxPositions': int(defaults['maxPositions']),
            'timeframe': defaults['timeframe'],
        },
        'notes': list(notes),
    }

    for field in ('id', 'label', 'description'):
        if not template[field]:
            raise ValueError(f'Scenario template at index {index} missing required field: {field}')

    return template


def load_scenario_templates(path: str | Path | None = None) -> list[ScenarioTemplate]:
    """Load scenario templates from the shared JSON configuration."""

    location = Path(path) if path else _DEFAULT_TEMPLATES_PATH
    raw = location.read_text(encoding='utf-8')
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise TypeError('Scenario templates JSON must contain an array at the top level')

    templates = [_validate_template(entry, index) for index, entry in enumerate(parsed)]
    return templates


__all__ = ['ScenarioConfig', 'ScenarioTemplate', 'load_scenario_templates']
