"""Public data, simulation, and phantom API."""

from __future__ import annotations

from types import ModuleType
from typing import Any

from .io_hdf5 import (
    LoadedNXTomo,
    NXTomoMetadata,
    ValidationReport,
    load_nxtomo,
    save_nxtomo,
    validate_nxtomo,
)
from .phantoms import (
    blobs,
    cube,
    lamino_disk,
    lamino_disk_legacy,
    random_cubes_spheres,
    rotated_centered_cube,
    shepp_logan_3d,
    sphere,
)
from . import simulate as _simulate_module


class _CallableDataModule(ModuleType):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        function_name = self.__name__.rsplit(".", 1)[-1]
        return getattr(self, function_name)(*args, **kwargs)


def _make_callable_module(module: ModuleType) -> ModuleType:
    module.__class__ = _CallableDataModule
    return module


simulate = _make_callable_module(_simulate_module)
SimConfig = _simulate_module.SimConfig
SimulatedData = _simulate_module.SimulatedData
simulate_to_file = _simulate_module.simulate_to_file

__all__ = [
    "LoadedNXTomo",
    "NXTomoMetadata",
    "SimConfig",
    "SimulatedData",
    "ValidationReport",
    "blobs",
    "cube",
    "lamino_disk",
    "lamino_disk_legacy",
    "load_nxtomo",
    "random_cubes_spheres",
    "rotated_centered_cube",
    "save_nxtomo",
    "shepp_logan_3d",
    "simulate",
    "simulate_to_file",
    "sphere",
    "validate_nxtomo",
]
