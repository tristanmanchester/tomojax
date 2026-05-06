"""Public reconstruction API."""
# pyright: reportAny=false

from __future__ import annotations

from types import ModuleType
from typing import Any

from . import fbp as _fbp_module, fista_tv as _fista_tv_module, spdhg_tv as _spdhg_tv_module
from ._reference import reconstruct_average_reference
from .types import Regulariser


class _CallableReconModule(ModuleType):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        function_name = self.__name__.rsplit(".", 1)[-1]
        return getattr(self, function_name)(*args, **kwargs)


def _make_callable_module(module: ModuleType) -> ModuleType:
    module.__class__ = _CallableReconModule
    return module


fbp = _make_callable_module(_fbp_module)
fista_tv = _make_callable_module(_fista_tv_module)
spdhg_tv = _make_callable_module(_spdhg_tv_module)

FBPConfig = _fbp_module.FBPConfig
FistaConfig = _fista_tv_module.FistaConfig
SPDHGConfig = _spdhg_tv_module.SPDHGConfig

__all__ = [
    "FBPConfig",
    "FistaConfig",
    "Regulariser",
    "SPDHGConfig",
    "fbp",
    "fista_tv",
    "reconstruct_average_reference",
    "spdhg_tv",
]
