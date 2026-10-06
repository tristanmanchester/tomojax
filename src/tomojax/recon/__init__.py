"""Public reconstruction API.

Names resolve on first use, so a Fourier reconstruction does not import JAX
and an iterative solve does not import the Fourier/CuPy stack.
"""

from __future__ import annotations

from importlib import import_module
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tomojax.recon._operators import backproject, project
    from tomojax.recon.cgls import CGLSConfig, cgls
    from tomojax.recon.cgls_multires import cgls_multires
    from tomojax.recon.cone_axis import ConeAxisCalibration, ConeAxisConfig, calibrate_cone_axis
    from tomojax.recon.fbp import (
        FBPConfig,
        fbp,
    )
    from tomojax.recon.fbp_host import FBPHostConfig, fbp_host
    from tomojax.recon.fdk import FDKConfig, FDKHostConfig, fdk, fdk_host
    from tomojax.recon.fista_tv import FistaConfig, fista_tv
    from tomojax.recon.fourier import FourierConfig, fourier_reconstruct
    from tomojax.recon.spdhg_tv import SPDHGConfig, spdhg_tv

_SOURCES = {
    "backproject": "_operators",
    "project": "_operators",
    "CGLSConfig": "cgls",
    "ConeAxisCalibration": "cone_axis",
    "ConeAxisConfig": "cone_axis",
    "FBPConfig": "fbp",
    "FBPHostConfig": "fbp_host",
    "FDKConfig": "fdk",
    "FDKHostConfig": "fdk",
    "FistaConfig": "fista_tv",
    "FourierConfig": "fourier",
    "SPDHGConfig": "spdhg_tv",
    "calibrate_cone_axis": "cone_axis",
    "cgls": "cgls",
    "cgls_multires": "cgls_multires",
    "fbp": "fbp",
    "fbp_host": "fbp_host",
    "fdk": "fdk",
    "fdk_host": "fdk",
    "fista_tv": "fista_tv",
    "fourier_reconstruct": "fourier",
    "spdhg_tv": "spdhg_tv",
}

__all__ = [
    "CGLSConfig",
    "ConeAxisCalibration",
    "ConeAxisConfig",
    "FBPConfig",
    "FBPHostConfig",
    "FDKConfig",
    "FDKHostConfig",
    "FistaConfig",
    "FourierConfig",
    "SPDHGConfig",
    "backproject",
    "calibrate_cone_axis",
    "cgls",
    "cgls_multires",
    "fbp",
    "fbp_host",
    "fdk",
    "fdk_host",
    "fista_tv",
    "fourier_reconstruct",
    "project",
    "spdhg_tv",
]


class _Facade(ModuleType):
    def __setattr__(self, name: str, value: Any) -> None:
        # Importing the submodule ``tomojax.recon.fbp`` must not hide the public
        # ``fbp`` function that shares its name.
        if name in _SOURCES and isinstance(value, ModuleType):
            return
        super().__setattr__(name, value)


def __getattr__(name: str) -> Any:
    source = _SOURCES.get(name)
    if source is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{source}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


sys.modules[__name__].__class__ = _Facade
