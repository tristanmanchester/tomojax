"""The beam hardening methods the bench scores, behind one call.

A method takes raw ``tj.Frames`` and the settings a scanner records (tube
voltage, anode, window, filters, scintillator; never the truth) plus the
sample material a user would name, and returns
a :class:`Result`: a volume in mm^-1, the gauge it is in, and its corrected
line integrals if it makes them. Every method reconstructs with :func:`image`
unless it is itself a reconstruction method. ``oracle`` and ``exact`` read the
simulated truth to bound what the bench can resolve: ``oracle`` applies the
exact curve of the material carrying most attenuation (area times
attenuation), which a correct single-material calibration gives and which
is exact for one material, and ``exact`` reconstructs the noiseless true line
integrals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


LAMINOGRAPHY_ITERATIONS = 50


@dataclass
class Result:
    """A method's volume (mm^-1), gauge, corrected line integrals, whether it declined, record."""

    volume: np.ndarray
    gauge: tuple
    projections: np.ndarray | None = None
    declined: bool = False
    info: dict | None = None


def image(scan: Any) -> np.ndarray:
    """The bench's reconstruction of line integrals, on the host.

    FDK for cone beams; for laminography, where filtered backprojection is
    not quantitative, CGLS with a fixed budget.
    """
    import tomojax as tj

    if isinstance(scan.geometry, tj.LaminographyGeometry):
        return np.asarray(tj.reconstruct(scan, "cgls", iterations=LAMINOGRAPHY_ITERATIONS).volume)
    return np.asarray(tj.reconstruct(scan, "fbp").volume)


def uncorrected(frames: Any, metadata: dict, truth: dict | None = None) -> Result:  # noqa: ARG001
    """Flat, dark and log only; its image is in the slope gauge for small path lengths."""
    scan = frames.corrected()
    return Result(image(scan), ("slope",), np.asarray(scan.projections))


def oracle(frames: Any, metadata: dict, truth: dict | None = None) -> Result:  # noqa: ARG001
    """Invert the true curve of the main material: p = mu_bar L where q(L) is measured."""
    import tomojax as tj

    if truth is None:
        raise ValueError("the oracle needs the simulated truth")
    weights, mu = truth["weights"], truth["main_mu"]
    lengths = np.linspace(0.0, 60.0 / max(mu.min(), 1e-3), 4097)
    q = -np.log(np.exp(-np.multiply.outer(lengths, mu)) @ weights)
    slope = mu @ weights
    scan = frames.corrected()
    measured = np.asarray(scan.projections, np.float64)
    # Beyond the table the true curve is linear with the hardest component's slope.
    p = np.where(
        measured <= q[-1],
        np.interp(measured, q, lengths * slope),
        lengths[-1] * slope + (measured - q[-1]) * slope / mu.min(),
    ).astype(np.float32)
    corrected = tj.Scan(p, scan.geometry)
    return Result(image(corrected), ("slope",), p)


def exact(frames: Any, metadata: dict, truth: dict | None = None) -> Result:  # noqa: ARG001
    """The exact noiseless line integrals in the slope gauge, reconstructed: the bench's floor."""
    import tomojax as tj

    if truth is None:
        raise ValueError("the exact control needs the simulated truth")
    return Result(
        image(tj.Scan(truth["line_integrals"], frames.geometry)),
        ("slope",),
        truth["line_integrals"],
    )


SCINTILLATOR_DENSITY = {"CsI": 4.51, "Gd2O2S": 7.34, "CdWO4": 7.9}


def selfcal(frames: Any, metadata: dict, truth: dict | None = None) -> Result:  # noqa: ARG001
    """``BeamHardening.fit``: segment a first reconstruction, reproject, fit the curve.

    When the fit declines (``ValueError``: no material stands out), the scan
    is left uncorrected and scored as such.
    """
    from tomojax.corrections import BeamHardening

    scan = frames.corrected()
    try:
        step = BeamHardening.fit(scan)
    except ValueError:
        return Result(image(scan), ("slope",), np.asarray(scan.projections), declined=True)
    corrected = scan.corrected(step)
    return Result(image(corrected), ("slope",), np.asarray(corrected.projections))


def spectrum(frames: Any, metadata: dict, truth: dict | None = None) -> Result:  # noqa: ARG001
    """``BeamHardening.from_spectrum`` of the sample's material and the modelled tube.

    The tube spectrum comes from the recorded voltage, window, filters and
    scintillator (0.5 mm if its thickness is not recorded); the material is
    what the user names as the sample's. The image is attenuation at the
    modelled spectrum's mean detected energy.
    """
    from tomojax.corrections import BeamHardening, tube_spectrum

    if metadata.get("monochromatic_kev"):
        energies, weights = [metadata["monochromatic_kev"]], [1.0]
    else:
        layers = [tuple(layer) for layer in (metadata["window"], *metadata["filters"]) if layer[1]]
        material, thickness = metadata["scintillator"]
        energies, weights = tube_spectrum(
            metadata["kv"],
            filters=layers,
            detector=(material, thickness or 0.5, SCINTILLATOR_DENSITY[material]),
            target_angle=metadata.get("anode_angle_deg") or 12.0,
        )
    energy = float(np.dot(energies, weights) / np.sum(weights))
    step = BeamHardening.from_spectrum(energies, weights, metadata["material"], energy_kev=energy)
    corrected = frames.corrected().corrected(step)
    return Result(image(corrected), ("energy", energy), np.asarray(corrected.projections))


def joint(frames: Any, metadata: dict, truth: dict | None = None) -> Result:  # noqa: ARG001
    """``tj.reconstruct(scan, "polychromatic")``: the image and its curve fitted together.

    Its line integrals are the measured ones through the inverse of the fitted
    curve. Where it declines, as ``selfcal`` does, the scan is left uncorrected.
    """
    import tomojax as tj
    from tomojax.corrections import BeamHardening

    scan = frames.corrected()
    try:
        result = tj.reconstruct(scan, "polychromatic")
    except ValueError:  # its self-calibrated start declined: no material stands out
        return Result(image(scan), ("slope",), np.asarray(scan.projections), declined=True)
    step = BeamHardening(**result.info["beam_hardening"])
    projections = np.asarray(step.inverse(np.asarray(scan.projections)), np.float32)
    record = {k: v for k, v in result.info.items() if k in ("beam_hardening", "misfit", "start")}
    return Result(np.asarray(result.volume), ("slope",), projections, info=record)


METHODS: dict[str, Callable[..., Result]] = {
    "none": uncorrected,
    "oracle": oracle,
    "exact": exact,
    "selfcal": selfcal,
    "spectrum": spectrum,
    "joint": joint,
}
