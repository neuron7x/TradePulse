"""Template utilities for rendering and validating TradePulse CLI configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Type, TypeVar

import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, ValidationError

__all__ = ["ConfigTemplateManager", "load_yaml_file", "model_to_hash"]

T = TypeVar("T", bound=BaseModel)


def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load a YAML document returning a dictionary."""

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError(f"YAML file {path} must contain a mapping at the top level")
    return dict(data)


def model_to_hash(model: BaseModel) -> str:
    """Return a deterministic sha256 hash for a Pydantic model."""

    payload = json.dumps(model.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    return __import__("hashlib").sha256(payload).hexdigest()


class ConfigTemplateManager:
    """Render configuration templates and validate configs against models."""

    def __init__(self, templates_dir: Path | None = None) -> None:
        self.templates_dir = Path(templates_dir or Path("configs/templates")).resolve()
        if not self.templates_dir.exists():
            raise FileNotFoundError(f"Templates directory {self.templates_dir} does not exist")
        loader = FileSystemLoader(str(self.templates_dir))
        self._env = Environment(loader=loader, autoescape=select_autoescape(enabled_extensions=(".j2",)))

    def available_templates(self) -> Dict[str, Path]:
        """Return all available templates keyed by stem."""

        templates: Dict[str, Path] = {}
        for path in self.templates_dir.glob("*.yaml.j2"):
            templates[path.stem.replace(".yaml", "")] = path
        return templates

    @property
    def environment(self) -> Environment:
        """Expose the underlying Jinja environment for advanced rendering."""

        return self._env

    def render(self, template_name: str, destination: Path, **context: Any) -> Path:
        """Render *template_name* into *destination* with optional context."""

        template_file = None
        for candidate in self.templates_dir.glob(f"{template_name}.yaml.j2"):
            template_file = candidate
            break
        if template_file is None:
            raise FileNotFoundError(f"Template '{template_name}' not found in {self.templates_dir}")

        destination.parent.mkdir(parents=True, exist_ok=True)
        template = self._env.get_template(template_file.name)
        rendered = template.render(**context)
        destination.write_text(rendered, encoding="utf-8")
        return destination

    def load_config(self, path: Path, model: Type[T]) -> T:
        """Load YAML config from *path* and validate against *model*."""

        data = load_yaml_file(path)
        try:
            return model.model_validate(data, context={"base_path": str(path.parent.resolve())})
        except ValidationError as exc:  # pragma: no cover - pydantic formats details
            raise ValueError(f"Configuration validation failed for {path}: {exc}") from exc

    def validate(self, path: Path, model: Type[T]) -> T:
        """Alias of :meth:`load_config` for readability."""

        return self.load_config(path, model)

    def write_json(self, destination: Path, payload: Mapping[str, Any]) -> Path:
        """Write payload to JSON ensuring parents exist."""

        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        return destination

