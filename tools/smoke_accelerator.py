# pyright: reportAny=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""Smoke-test accelerator-sensitive projector paths against the JAX reference."""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax

    from tomojax.geometry import Detector, Grid


CudaDriverStatus = dict[str, object]
ProjectionCase = tuple["jax.Array", "Grid", "Detector", "jax.Array"]


def _cuda_driver_preflight() -> CudaDriverStatus:
    """Probe the CUDA driver before importing JAX, which may initialize plugins."""
    libcuda_path = ctypes.util.find_library("cuda")
    if libcuda_path is None:
        return {"status": "not_found"}
    try:
        libcuda = ctypes.CDLL(libcuda_path)
        cu_init = libcuda.cuInit
        cu_init.argtypes = [ctypes.c_uint]
        cu_init.restype = ctypes.c_int
        rc = int(cu_init(0))
    except Exception as exc:
        return {
            "status": "error",
            "library": libcuda_path,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "status": "ok" if rc == 0 else "cuInit_failed",
        "library": libcuda_path,
        "cuInit": rc,
    }


def _tiny_projection_case() -> ProjectionCase:
    import jax.numpy as jnp

    from tomojax.geometry import Detector, Grid

    transform = jnp.eye(4, dtype=jnp.float32)
    grid = Grid(nx=2, ny=2, nz=2, vx=1.0, vy=1.0, vz=1.0)
    detector = Detector(nu=2, nv=2, du=1.0, dv=1.0)
    volume = jnp.arange(8, dtype=jnp.float32).reshape((2, 2, 2))
    return transform, grid, detector, volume


def _assert_close(name: str, actual: object, expected: object) -> None:
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"{name} did not match the JAX reference projector",
    )


def main() -> int:
    """Run the accelerator smoke check and print a compact JSON summary."""
    cuda_driver = _cuda_driver_preflight()
    require_cuda = os.environ.get("TOMOJAX_REQUIRE_CUDA") == "1"
    cuda_status = str(cuda_driver["status"])
    if cuda_status != "ok":
        _ = os.environ.setdefault("JAX_PLATFORMS", "cpu")
        if require_cuda:
            print(
                json.dumps(
                    {
                        "cuda_driver": cuda_driver,
                        "pallas_interpret": "not_run",
                        "pallas_real": "failed_cuda_driver",
                    },
                    indent=2,
                )
            )
            return 1

    import jax

    from tomojax.core import projector
    from tomojax.core.pallas import api as pallas_api

    transform, grid, detector, volume = _tiny_projection_case()
    expected = projector.forward_project_view_T(
        transform,
        grid,
        detector,
        volume,
        projector_backend="jax",
    )
    _ = expected.block_until_ready()

    interpret_options = pallas_api.PallasProjectorOptions(
        interpret=True,
        tile_shape=(1, 1),
        num_warps=1,
    )
    interpreted = pallas_api.forward_project_view_T_pallas(
        transform,
        grid,
        detector,
        volume,
        options=interpret_options,
    )
    _ = interpreted.block_until_ready()
    _assert_close("Pallas interpret projector", interpreted, expected)

    backend = jax.default_backend()
    force_real = os.environ.get("TOMOJAX_FORCE_REAL_PALLAS_SMOKE") == "1"
    real_pallas_status = "skipped_cpu_backend"
    if require_cuda and backend == "cpu":
        print(
            json.dumps(
                {
                    "backend": backend,
                    "devices": [str(device) for device in jax.devices()],
                    "cuda_driver": cuda_driver,
                    "pallas_interpret": "passed",
                    "pallas_real": "failed_cpu_backend",
                },
                indent=2,
            )
        )
        return 1

    if backend != "cpu" or force_real:
        real_options = pallas_api.PallasProjectorOptions(tile_shape=(1, 1), num_warps=1)
        reason = pallas_api.pallas_projector_unsupported_reason(
            transform,
            grid,
            detector,
            volume,
            options=real_options,
        )
        if reason is not None:
            raise RuntimeError(f"real Pallas projector unsupported on backend {backend}: {reason}")
        actual = pallas_api.forward_project_view_T_pallas(
            transform,
            grid,
            detector,
            volume,
            options=real_options,
        )
        _ = actual.block_until_ready()
        _assert_close("Real Pallas projector", actual, expected)
        real_pallas_status = "passed"

    print(
        json.dumps(
            {
                "backend": backend,
                "devices": [str(device) for device in jax.devices()],
                "cuda_driver": cuda_driver,
                "pallas_interpret": "passed",
                "pallas_real": real_pallas_status,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
