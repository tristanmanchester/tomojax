"""Transitional public API for legacy data helpers.

New production IO should use `tomojax.io`; deterministic synthetic generation
should use `tomojax.datasets`. This API file exists so the retained transitional
package still follows the v2 deep-module shape while its responsibilities are
migrated.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

from tomojax.data.artefacts import SimulationArtefacts, apply_simulation_artefacts
from tomojax.data.contrast import (
    absorption_to_transmission,
    flat_dark_to_absorption,
    transmission_to_absorption,
)
from tomojax.data.io_hdf5 import (
    LoadedNXTomo,
    NXTomoMetadata,
    ValidationReport,
    load_nxtomo,
    save_nxtomo,
    validate_nxtomo,
)
from tomojax.data.phantoms import (
    blobs,
    cube,
    lamino_disk,
    lamino_disk_legacy,
    random_cubes_spheres,
    rotated_centered_cube,
    shepp_logan_3d,
    sphere,
)

_simulate_module = import_module("tomojax.data.simulate")


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
    "SimulationArtefacts",
    "ValidationReport",
    "absorption_to_transmission",
    "apply_simulation_artefacts",
    "blobs",
    "cube",
    "flat_dark_to_absorption",
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
    "transmission_to_absorption",
    "validate_nxtomo",
]
