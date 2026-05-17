"""Public synthetic simulation facade."""

from __future__ import annotations

from tomojax.data.artefacts import SimulationArtefacts, validate_simulation_artefacts
from tomojax.data.simulate import SimConfig, SimulatedData, make_phantom, simulate, simulate_to_file

__all__ = [
    "SimConfig",
    "SimulatedData",
    "SimulationArtefacts",
    "make_phantom",
    "simulate",
    "simulate_to_file",
    "validate_simulation_artefacts",
]
