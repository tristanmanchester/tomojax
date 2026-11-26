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
    from tomojax.recon.cgls import CGLSConfig, cgls
    from tomojax.recon.cgls_multires import cgls_multires
    from tomojax.recon.fbp import (
        FBPConfig,
        fbp,
    )
    from tomojax.recon.fbp_host import FBPHostConfig, fbp_host
    from tomojax.recon.fdk import FDKConfig, FDKHostConfig, fdk, fdk_host
    from tomojax.recon.fista_tv import FISTAConfig, fista
    from tomojax.recon.fourier import FourierConfig, fourier
    from tomojax.recon.mlem import MLEMConfig, mlem
    from tomojax.recon.sirt import SIRTConfig, sirt
    from tomojax.recon.spdhg_tv import SPDHGConfig, spdhg

_SOURCES = {
    "CGLSConfig": "cgls",
    "FBPConfig": "fbp",
    "FBPHostConfig": "fbp_host",
    "FDKConfig": "fdk",
    "FDKHostConfig": "fdk",
    "FISTAConfig": "fista_tv",
    "FourierConfig": "fourier",
    "MLEMConfig": "mlem",
    "mlem": "mlem",
    "SPDHGConfig": "spdhg_tv",
    "SIRTConfig": "sirt",
    "sirt": "sirt",
    "cgls": "cgls",
    "cgls_multires": "cgls_multires",
    "fbp": "fbp",
    "fbp_host": "fbp_host",
    "fdk": "fdk",
    "fdk_host": "fdk",
    "fista": "fista_tv",
    "fourier": "fourier",
    "spdhg": "spdhg_tv",
}

__all__ = [
    "CGLSConfig",
    "FBPConfig",
    "FBPHostConfig",
    "FDKConfig",
    "FDKHostConfig",
    "FISTAConfig",
    "FourierConfig",
    "MLEMConfig",
    "SIRTConfig",
    "SPDHGConfig",
    "cgls",
    "cgls_multires",
    "fbp",
    "fbp_host",
    "fdk",
    "fdk_host",
    "fista",
    "fourier",
    "mlem",
    "sirt",
    "spdhg",
]


class _Facade(ModuleType):
    def __setattr__(self, name: str, value: object) -> None:
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
