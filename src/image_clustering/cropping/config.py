from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    data: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.data[name])

    def get(self, dotted: str, default: Any = None) -> Any:
        value: Any = self.data
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                return default
            value = value[part]
        return value

    def fingerprint(self) -> str:
        """Return a stable hash of the complete configuration."""
        encoded = json.dumps(
            self.data,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(default_path: Path, override_path: Path | None = None) -> Config:
    """Load default YAML and an optional recursive override.

    Args:
        default_path: Repository default configuration.
        override_path: Optional user configuration.

    Returns:
        Merged immutable configuration wrapper.
    """
    base = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}
    if override_path is None:
        return Config(base)
    override = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
    return Config(_deep_merge(base, override))


def load_default_config(override_path: Path | None = None) -> Config:
    """Load the packaged default crop configuration."""
    return load_config(Path(__file__).with_name("default.yaml"), override_path)
