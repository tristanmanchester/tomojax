"""Public synthetic simulation facade."""

from __future__ import annotations

from tomojax.datasets._impl.artefacts import (
    SimulationArtefacts,
    apply_simulation_artefacts,
    validate_simulation_artefacts,
)
from tomojax.datasets._impl.simulate import (
    SimConfig,
    SimMetadata,
    SimulatedData,
    make_phantom,
    simulate,
    simulate_to_file,
)

__all__ = [
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
