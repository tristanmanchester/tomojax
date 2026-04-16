from __future__ import annotations

from contextlib import nullcontext
import logging
import os
import sys
import types

import numpy as np
import pytest
import jax.numpy as jnp

from tomojax.cli import _runtime as runtime_helpers
from tomojax.cli import convert as convert_cli
from tomojax.cli import runtime_checks
from tomojax.core.geometry.views import stack_view_poses
from tomojax.recon._tv_ops import div3, grad3
from tomojax.utils import logging as logging_utils
from tomojax.utils import memory as memory_utils


class DummyGeometry:
    def pose_for_view(self, i: int) -> np.ndarray:
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = float(i)
        pose[2, 3] = -float(i)
        return pose


def test_convert_main_parses_paths_and_calls_convert(monkeypatch):
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        convert_cli,
        "convert",
        lambda in_path, out_path: captured.update({"in_path": in_path, "out_path": out_path}),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["convert", "--in", "input.npz", "--out", "output.nxs"],
    )

    convert_cli.main()

    assert captured == {"in_path": "input.npz", "out_path": "output.nxs"}


def test_runtime_checks_entrypoints_delegate_to_runtime_printer(monkeypatch):
    calls: list[str | None] = []

    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setattr(
        runtime_checks,
        "_print_runtime",
        lambda: calls.append(os.environ.get("JAX_PLATFORM_NAME")),
    )

    runtime_checks.test_gpu_main()
    runtime_checks.test_cpu_main()

    assert calls == [None, "cpu"]
    assert os.environ["JAX_PLATFORM_NAME"] == "cpu"


def test_transfer_guard_context_covers_disabled_and_jax_paths(monkeypatch):
    disabled = runtime_helpers.transfer_guard_context("off")
    assert isinstance(disabled, type(nullcontext()))

    calls: list[str] = []
    sentinel = object()
    fake_jax = types.SimpleNamespace(transfer_guard=lambda mode: calls.append(mode) or sentinel)
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    enabled = runtime_helpers.transfer_guard_context("log")

    assert enabled is sentinel
    assert calls == ["log"]


def test_stack_view_poses_preserves_order_and_dtype():
    poses = stack_view_poses(DummyGeometry(), 3, dtype=jnp.float16)

    assert poses.shape == (3, 4, 4)
    assert poses.dtype == jnp.float16
    np.testing.assert_allclose(np.asarray(poses[:, 0, 3]), [0.0, 1.0, 2.0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(poses[:, 2, 3]), [0.0, -1.0, -2.0], atol=1e-6)


def test_grad3_and_div3_are_negative_adjoints():
    u = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)
    px = (jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) + 1.0) / 10.0
    py = (jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) + 2.0) / 11.0
    pz = (jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) + 3.0) / 12.0

    dx, dy, dz = grad3(u)
    lhs = jnp.vdot(dx, px) + jnp.vdot(dy, py) + jnp.vdot(dz, pz)
    rhs = -jnp.vdot(u, div3(px, py, pz))

    assert lhs == pytest.approx(float(rhs), rel=1e-6, abs=1e-6)


def test_div3_handles_singleton_axes():
    px = jnp.ones((1, 2, 3), dtype=jnp.float32)
    py = jnp.ones((1, 1, 3), dtype=jnp.float32)
    pz = jnp.ones((1, 2, 1), dtype=jnp.float32)

    div = div3(px, py, pz)

    assert div.shape == (1, 2, 3)
    assert np.isfinite(np.asarray(div)).all()


def test_logging_helpers_cover_progress_and_duration(monkeypatch):
    backend_logger = logging.getLogger("jax._src.xla_bridge")
    backend_logger.setLevel(logging.INFO)
    monkeypatch.delenv("TOMOJAX_BACKEND_LOG", raising=False)
    logging_utils.setup_logging("warning")
    assert backend_logger.level == logging.WARNING

    monkeypatch.setenv("TOMOJAX_PROGRESS", "0")
    assert list(logging_utils.progress_iter([1, 2, 3], desc="plain")) == [1, 2, 3]

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda iterable, **kwargs: iterable
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm)
    monkeypatch.setenv("TOMOJAX_PROGRESS", "1")
    assert list(logging_utils.progress_iter([1, 2], total=2, desc="bar")) == [1, 2]

    assert logging_utils.format_duration(None) == "-"
    assert logging_utils.format_duration(0.0005) == "500µs"
    assert logging_utils.format_duration(0.25) == "0.25s"
    assert logging_utils.format_duration(65.0) == "1m05.0s"


def test_memory_helpers_cover_batch_estimation_and_backend_defaults(monkeypatch):
    monkeypatch.setenv("TOMOJAX_MAX_VIEWS_PER_BATCH", "3")

    capped = memory_utils.estimate_views_per_batch(
        n_views=10,
        grid_nxyz=(16, 16, 16),
        det_nuv=(8, 8),
        free_bytes_override=10**9,
    )
    fallback = memory_utils.estimate_views_per_batch(
        n_views=10,
        grid_nxyz=(16, 16, 16),
        det_nuv=(8, 8),
        free_bytes_override=0,
    )

    assert 1 <= capped <= 3
    assert fallback == 8
    assert memory_utils._bytes_per("bf16") == 2
    assert memory_utils._bytes_per("fp32") == 4

    monkeypatch.setattr(memory_utils, "_current_backend", lambda: "cpu")
    assert memory_utils.default_gather_dtype() == "fp32"
