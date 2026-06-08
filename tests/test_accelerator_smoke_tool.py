from __future__ import annotations

import builtins
from collections.abc import Mapping, Sequence
import importlib.util
import json
from pathlib import Path
from typing import Protocol, cast

import pytest


class SmokeAcceleratorModule(Protocol):
    def main(self) -> int: ...


def _load_smoke_accelerator() -> SmokeAcceleratorModule:
    path = Path(__file__).resolve().parents[1] / "tools" / "smoke_accelerator.py"
    spec = importlib.util.spec_from_file_location("_tomojax_smoke_accelerator_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load smoke_accelerator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast("SmokeAcceleratorModule", module)


@pytest.mark.parametrize(
    "preflight",
    [
        {"status": "cuInit_failed", "library": "libcuda.so.1", "cuInit": 999},
        {"status": "not_found"},
        {"status": "error", "library": "libcuda.so.1", "error": "boom"},
    ],
)
def test_strict_cuda_preflight_fails_before_importing_jax(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    preflight: dict[str, object],
) -> None:
    module = _load_smoke_accelerator()
    monkeypatch.setenv("TOMOJAX_REQUIRE_CUDA", "1")
    monkeypatch.setattr(
        module,
        "_cuda_driver_preflight",
        lambda: preflight,
    )

    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> object:
        if name == "jax" or name.startswith("jax."):
            raise AssertionError("strict CUDA preflight should return before importing JAX")
        return cast("object", real_import(name, globals, locals, fromlist, level))

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert module.main() == 1
    payload = cast("dict[str, object]", json.loads(capsys.readouterr().out))
    assert payload == {
        "cuda_driver": preflight,
        "pallas_interpret": "not_run",
        "pallas_real": "failed_cuda_driver",
    }


def test_strict_cuda_fails_when_jax_reports_cpu_backend(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_smoke_accelerator()
    monkeypatch.setenv("TOMOJAX_REQUIRE_CUDA", "1")
    monkeypatch.setattr(
        module,
        "_cuda_driver_preflight",
        lambda: {"status": "ok", "library": "libcuda.so.1", "cuInit": 0},
    )

    assert module.main() == 1
    payload = cast("dict[str, object]", json.loads(capsys.readouterr().out))
    assert payload["backend"] == "cpu"
    assert payload["cuda_driver"] == {
        "status": "ok",
        "library": "libcuda.so.1",
        "cuInit": 0,
    }
    assert payload["pallas_interpret"] == "passed"
    assert payload["pallas_real"] == "failed_cpu_backend"
