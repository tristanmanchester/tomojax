"""Public synthetic simulation facade."""

from __future__ import annotations

from tomojax.data.artefacts import (
    SimulationArtefacts,
    apply_simulation_artefacts,
    validate_simulation_artefacts,
)
from tomojax.data.simulate import (
    LaminoGeometryMeta,
    SimConfig,
    SimMetadata,
    SimulatedData,
    make_phantom,
    simulate,
    simulate_to_file,
)

__all__ = [
    "LaminoGeometryMeta",
    "SimConfig",
    "SimMetadata",
    "SimulatedData",
    "SimulationArtefacts",
    "apply_simulation_artefacts",
    "make_phantom",
    "simulate",
    "simulate_to_file",
    "validate_simulation_artefacts",
]
