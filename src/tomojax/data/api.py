"""Retained lower-level API for data helpers.

New production IO should use `tomojax.io`; deterministic synthetic generation
should use `tomojax.datasets`. This API file exists so the retained lower-level
package still follows the v2 deep-module shape while its responsibilities are
moved behind the owning modules.
"""

from __future__ import annotations

from tomojax.data.artefacts import SimulationArtefacts, apply_simulation_artefacts
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
    random_cubes_spheres,
    rotated_centered_cube,
    shepp_logan_3d,
    sphere,
)
from tomojax.data.simulate import SimConfig, SimulatedData, simulate, simulate_to_file

__all__ = [
    "LoadedNXTomo",
    "NXTomoMetadata",
    "SimConfig",
    "SimulatedData",
    "SimulationArtefacts",
    "ValidationReport",
    "apply_simulation_artefacts",
    "blobs",
    "cube",
    "lamino_disk",
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
