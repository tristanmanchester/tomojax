from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from importlib import metadata
import numpy as np
import pytest

import tomojax
from tomojax.cli import manifest as manifest_helpers


@dataclass
class _NestedConfig:
    path: Path
    values: np.ndarray


class _DictBacked:
    def to_dict(self) -> dict[str, object]:
        return {"kind": "custom", "scale": np.float32(1.5)}


def test_build_manifest_is_json_serializable_and_accepts_fixed_timestamp(monkeypatch):
    monkeypatch.setattr(
        manifest_helpers,
        "_versions",
        lambda: {
            "tomojax": "test-version",
            "python": {"version": "3.test"},
            "jax": None,
            "jaxlib": None,
        },
    )
    monkeypatch.setattr(
        manifest_helpers,
        "_jax_runtime",
        lambda: {"available": False, "backend": None, "devices": []},
    )

    manifest = manifest_helpers.build_manifest(
        "tomojax-recon",
        ["tomojax-recon", "--out", "out.nxs"],
        argparse.Namespace(out=Path("out.nxs"), grid=(2, 3, 4)),
        {
            "config": _NestedConfig(Path("input.nxs"), np.asarray([1, 2, 3])),
            "detector": _DictBacked(),
            "scalar": np.float32(2.25),
        },
        timestamp=datetime(2026, 4, 19, 12, 30, 0, tzinfo=timezone.utc),
    )

    json.dumps(manifest, allow_nan=False)
    assert manifest["schema_version"] == 1
    assert manifest["created_at"] == "2026-04-19T12:30:00Z"
    assert manifest["cli_args"]["out"] == "out.nxs"
    assert manifest["cli_args"]["grid"] == [2, 3, 4]
    assert manifest["resolved_config"]["config"]["path"] == "input.nxs"
    assert manifest["resolved_config"]["config"]["values"] == [1, 2, 3]
    assert manifest["resolved_config"]["detector"]["scale"] == pytest.approx(1.5)


def test_tomojax_version_falls_back_to_package_dunder_version(monkeypatch):
    def _missing_distribution(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", _missing_distribution)

    assert manifest_helpers._tomojax_version() == tomojax.__version__


def test_jax_runtime_reports_unavailable_without_raising(monkeypatch):
    monkeypatch.setitem(sys.modules, "jax", None)

    runtime = manifest_helpers._jax_runtime()

    assert runtime["available"] is False
    assert runtime["backend"] is None
    assert runtime["devices"] == []
    assert "error" in runtime


def test_jax_runtime_records_introspection_errors(monkeypatch):
    def _boom() -> str:
        raise RuntimeError("backend probe failed")

    fake_jax = SimpleNamespace(
        default_backend=_boom,
        devices=lambda: (_ for _ in ()).throw(RuntimeError("device probe failed")),
    )
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    runtime = manifest_helpers._jax_runtime()

    assert runtime["available"] is True
    assert runtime["backend"] is None
    assert runtime["devices"] == []
    assert "backend probe failed" in runtime["error"]
    assert "device probe failed" in runtime["error"]


def test_save_manifest_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "manifest.json"

    manifest_helpers.save_manifest(path, {"schema_version": 1, "ok": True})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"ok": True, "schema_version": 1}
