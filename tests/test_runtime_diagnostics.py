from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import pytest

# check-public-imports: allow-private
from tomojax.align._objectives import fixed_volume

# check-public-imports: allow-private
from tomojax.backends import _memory, estimate_views_per_batch_info
from tomojax.geometry import Detector, Grid


def test_device_memory_probe_logs_debug_before_host_fallback(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_devices(_kind: str) -> list[object]:
        raise RuntimeError("device probe unavailable")

    monkeypatch.setattr(jax, "devices", raise_devices)
    monkeypatch.setattr(_memory, "_host_available_memory_bytes", lambda: None)

    with caplog.at_level(logging.DEBUG, logger="tomojax.backends._memory"):
        free_bytes = _memory.device_free_memory_bytes()

    assert free_bytes is None
    assert "Device memory probe failed" in caplog.text
    assert "device probe unavailable" in caplog.text


def test_invalid_views_per_batch_env_logs_selected_default(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOMOJAX_MAX_VIEWS_PER_BATCH", "not-an-int")

    with caplog.at_level(logging.DEBUG, logger="tomojax.backends._memory"):
        estimate = estimate_views_per_batch_info(
            n_views=16,
            grid_nxyz=(4, 4, 4),
            det_nuv=(4, 4),
            free_bytes_override=1_000_000_000,
            fallback_batch=8,
        )

    assert estimate.views_per_batch == 8
    assert estimate.fallback_used is False
    assert "Ignoring invalid TOMOJAX_MAX_VIEWS_PER_BATCH value" in caplog.text


def test_gpu_compute_capability_probe_logs_subprocess_failure(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_probe(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("nvidia-smi missing")

    _memory._gpu_compute_capability.cache_clear()
    monkeypatch.setattr(_memory, "check_output_resolved_command", raise_probe)

    with caplog.at_level(logging.DEBUG, logger="tomojax.backends._memory"):
        capability = _memory._gpu_compute_capability()

    assert capability is None
    assert "CUDA compute capability probe failed" in caplog.text
    assert "nvidia-smi missing" in caplog.text


def test_pallas_support_probe_logs_debug_before_jax_fallback(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_support(*_args: object, **_kwargs: object) -> str | None:
        raise RuntimeError("support probe failed")

    monkeypatch.setattr(
        fixed_volume,
        "resolve_pallas_callable",
        lambda *_args, **_kwargs: (raise_support, None),
    )

    with caplog.at_level(
        logging.DEBUG,
        logger="tomojax.align._objectives.fixed_volume",
    ):
        reason = fixed_volume._pallas_sinogram_fallback_reason(
            pose_stack=jnp.eye(4, dtype=jnp.float32)[None, :, :],
            grid=Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0),
            detector=Detector(nu=2, nv=2, du=1.0, dv=1.0),
            volume=jnp.ones((2, 2, 2), dtype=jnp.float32),
            det_grid=(
                jnp.zeros((4,), dtype=jnp.float32),
                jnp.zeros((4,), dtype=jnp.float32),
            ),
            gather_dtype="fp32",
        )

    assert reason == "RuntimeError: support probe failed"
    assert "Pallas sinogram support probe failed" in caplog.text
