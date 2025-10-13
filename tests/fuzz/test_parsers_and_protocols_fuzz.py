# SPDX-License-Identifier: MIT
"""Fuzz and protocol resilience tests for parsers and messaging adapters."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

try:
    from hypothesis import assume, given, settings, strategies as st
except ImportError:  # pragma: no cover - optional dependency
    pytest.skip("hypothesis not installed", allow_module_level=True)

yaml = pytest.importorskip("yaml")

from cli.tradepulse_cli import cli
from core.config.cli_models import ReportConfig
from core.config.template_manager import ConfigTemplateManager
from core.data.adapters.unified import Deduplicator, WebSocketIngestionAdapter
from core.data.feature_catalog import FeatureCatalog
from core.reporting import generate_markdown_report


json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.text(min_size=0, max_size=32, alphabet=st.characters(blacklist_categories=("Cs",))),
)

json_structures = st.recursive(
    json_scalars,
    lambda children: st.one_of(
        st.lists(children, min_size=0, max_size=6),
        st.dictionaries(
            st.text(min_size=0, max_size=16, alphabet=st.characters(blacklist_categories=("Cs",))),
            st.one_of(json_scalars, children),
            max_size=6,
        ),
    ),
    max_leaves=20,
)

mapping_structures = st.recursive(
    st.dictionaries(
        st.text(min_size=0, max_size=16, alphabet=st.characters(blacklist_categories=("Cs",))),
        json_scalars,
        max_size=6,
    ),
    lambda children: st.dictionaries(
        st.text(min_size=0, max_size=16, alphabet=st.characters(blacklist_categories=("Cs",))),
        st.one_of(json_scalars, st.lists(children, min_size=0, max_size=4), children),
        max_size=6,
    ),
    max_leaves=12,
)

non_mapping_structures = st.one_of(
    json_scalars,
    st.lists(json_structures, min_size=0, max_size=6),
)


def _safe_filename(seed: str) -> str:
    if not seed:
        return "artifact"
    sanitized = [ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in seed]
    candidate = "".join(sanitized).strip("._")
    return candidate or "artifact"


@settings(max_examples=100, deadline=None)
@given(entries=st.lists(json_structures, max_size=20))
def test_feature_catalog_loads_arbitrary_json(entries: list[Any], tmp_path: Path) -> None:
    """FeatureCatalog should tolerate arbitrary JSON payloads in the artifacts list."""

    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"artifacts": entries}, ensure_ascii=False), encoding="utf-8")

    catalog = FeatureCatalog(catalog_path)
    loaded = catalog._load_entries()
    assert isinstance(loaded, list)
    assert len(loaded) == len(entries)


@settings(max_examples=75, deadline=None)
@given(config=mapping_structures)
def test_template_manager_loads_varied_yaml(config: dict[str, Any], tmp_path: Path) -> None:
    """ConfigTemplateManager should return the raw mapping for arbitrary YAML content."""

    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    manager = ConfigTemplateManager(template_dir)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    loaded = manager.load_raw(path)
    assert loaded == config


@settings(max_examples=75, deadline=None)
@given(data=non_mapping_structures)
def test_template_manager_rejects_non_mapping_yaml(data: Any, tmp_path: Path) -> None:
    """Non-mapping YAML payloads should be rejected with a ValueError."""

    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    manager = ConfigTemplateManager(template_dir)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError):
        manager.load_raw(path)


@st.composite
def websocket_streams(draw) -> list[dict[str, Any]]:
    unique_messages = draw(
        st.lists(
            st.fixed_dictionaries(
                {
                    "timestamp": st.integers(min_value=0, max_value=10**12),
                    "price": st.floats(min_value=0.01, max_value=10_000.0, allow_nan=False, allow_infinity=False),
                    "volume": st.floats(min_value=0.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
                }
            ),
            min_size=1,
            max_size=12,
            unique_by=lambda item: item["timestamp"],
        )
    )
    stream: list[dict[str, Any]] = []
    for message in unique_messages:
        copies = draw(st.integers(min_value=1, max_value=4))
        stream.extend({**message} for _ in range(copies))
    assume(stream)
    permuted_indices = draw(st.permutations(tuple(range(len(stream)))))
    return [stream[idx] for idx in permuted_indices]


@settings(max_examples=100, deadline=None)
@given(stream=websocket_streams())
def test_websocket_adapter_deduplicates_and_orders(stream: list[dict[str, Any]]) -> None:
    """WebSocket adapter should drop duplicates while preserving arrival order."""

    adapter = WebSocketIngestionAdapter(lambda: stream, deduplicator=Deduplicator())
    emitted = list(adapter.messages())

    expected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for message in stream:
        key = str(message["timestamp"])
        if key in seen:
            continue
        seen.add(key)
        expected.append(message)

    assert emitted == expected


@st.composite
def cli_generation_commands(draw) -> tuple[list[str], Path | None]:
    command = draw(st.sampled_from(["ingest", "backtest", "exec", "optimize", "report"]))
    include_output = draw(st.booleans())
    args = [command, "--generate-config"]
    destination = None
    if include_output:
        raw_name = draw(st.text(min_size=1, max_size=24, alphabet=st.characters(blacklist_categories=("Cs",))))
        filename = _safe_filename(raw_name)
        if draw(st.booleans()):
            subdir = _safe_filename(
                draw(st.text(min_size=1, max_size=16, alphabet=st.characters(blacklist_categories=("Cs",))))
            )
            destination = Path(subdir) / Path(filename).with_suffix(".yaml")
        else:
            destination = Path(filename).with_suffix(".yaml")
        args.extend(["--template-output", str(destination)])
    return args, destination


@settings(max_examples=60, deadline=None)
@given(command=cli_generation_commands())
def test_cli_generate_config_handles_varied_paths(command: tuple[list[str], Path | None], tmp_path: Path) -> None:
    """CLI template generation should not crash for unusual but valid paths."""

    args, destination = command
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, args)
        assert result.exit_code == 0, result.output

        if destination is not None:
            path = Path(destination)
            assert path.exists(), f"Expected generated config at {path}"


@settings(max_examples=60, deadline=None)
@given(contents=st.lists(json_scalars.map(lambda value: str(value)), min_size=1, max_size=5))
def test_generate_markdown_report_handles_random_inputs(contents: list[str], tmp_path: Path) -> None:
    """Report generation should concatenate arbitrary text payloads safely."""

    paths: list[Path] = []
    for idx, text in enumerate(contents):
        filename = tmp_path / f"artifact_{idx}.txt"
        filename.write_text(text, encoding="utf-8")
        paths.append(filename)

    cfg = ReportConfig(name="fuzz", inputs=paths, output_path=tmp_path / "report.md")
    report = generate_markdown_report(cfg)

    for text in contents:
        snippet = text.strip()
        if snippet:
            assert snippet in report
