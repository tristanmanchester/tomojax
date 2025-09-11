from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load a JSON or YAML config file into a dict. YAML optional.

    If PyYAML is not installed, only JSON is supported.
    """
    if path.endswith(('.yaml', '.yml')):
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("YAML config requested but PyYAML not installed")
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    with open(path, 'r') as f:
        return json.load(f)


def dump_config(obj: Any) -> Dict[str, Any]:
    """Convert dataclass or object to plain dict for logging/serialization."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    # Fallback: use __dict__ if exists
    return {k: getattr(obj, k) for k in dir(obj) if not k.startswith('_')}

