"""Config loading: YAML + dotted key CLI overrides."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from topic_modeling.config.schema import ExperimentConfig


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _coerce(value: str) -> Any:
    """Best-effort type coercion for --set override values."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in ("null", "none"):
        return None
    return value


def _set_nested(d: Dict[str, Any], dotted_key: str, raw_value: str) -> None:
    """Apply a dotted key path override into dict d in place."""
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = _coerce(raw_value)


def load_config(
    config_path: str,
    overrides: Optional[List[str]] = None,
) -> ExperimentConfig:
    """Load a YAML experiment config and apply optional --set overrides.

    Args:
        config_path: Path to the top-level experiment YAML.
        overrides:   List of "dotted.key=value" strings from --set flags.

    Returns:
        Validated ExperimentConfig instance.
    """
    data = _load_yaml(config_path)

    if overrides:
        for override in overrides:
            if "=" not in override:
                raise ValueError(
                    f"Override must be key=value format, got: {override!r}"
                )
            key, val = override.split("=", 1)
            _set_nested(data, key.strip(), val.strip())

    return ExperimentConfig(**data)
