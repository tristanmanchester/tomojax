"""Detected X-ray tube spectra and material attenuation, from xraylib's cross sections."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable
    from types import ModuleType

    type Layer = tuple[str, float] | tuple[str, float, float]

# Kanaya-Okayama electron range in tungsten: R = 0.0276 A E^1.67 / (Z^0.889 rho) um.
_TUNGSTEN = {"Z": 74, "A": 183.84, "rho": 19.3}


def _xraylib() -> ModuleType:
    try:
        import xraylib
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError("spectrum models need xraylib: pip install 'tomojax[spectra]'") from error
    return xraylib


def attenuation(material: str, energies: np.ndarray) -> np.ndarray:
    """Mass attenuation (cm²/g) of a chemical formula or NIST material name at ``energies`` keV.

    Coherent and incoherent scattering are included, as a narrow beam sees them.
    """
    xraylib = _xraylib()
    try:
        return np.array([xraylib.CS_Total_CP(material, float(e)) for e in np.ravel(energies)])
    except ValueError as error:
        raise ValueError(
            f"{material!r} is neither a chemical formula nor a NIST material name"
        ) from error


def density(material: str) -> float:
    """Density (g/cm³) of an element or a NIST material."""
    xraylib = _xraylib()
    try:
        return float(xraylib.ElementDensity(xraylib.SymbolToAtomicNumber(material)))
    except ValueError:
        pass
    try:
        return float(xraylib.GetCompoundDataNISTByName(material)["density"])
    except ValueError as error:
        raise ValueError(
            f"no density for {material!r}: give (material, thickness_mm, density_g_cm3)"
        ) from error


def _layer(layer: Layer) -> tuple[str, float, float]:
    """``(material, thickness_mm, density)``, the density looked up if not given."""
    material, thickness, *given = layer
    rho = float(given[0]) if given else density(str(material))
    if float(thickness) < 0 or rho <= 0:
        raise ValueError(f"layer {layer!r}: thickness must be >= 0 and density > 0")
    return str(material), float(thickness), rho


def _layers(layers: Iterable[Layer], energies: np.ndarray) -> np.ndarray:
    """Transmission of a stack of ``(material, mm)`` or ``(material, mm, g/cm³)`` layers."""
    total = np.zeros_like(energies)
    for layer in layers:
        material, thickness, rho = _layer(layer)
        total += attenuation(material, energies) * rho * thickness / 10
    return np.exp(-total)


def tube_spectrum(
    kv: float,
    *,
    filters: Iterable[Layer] = (),
    detector: Layer | None = None,
    target_angle: float = 12.0,
    step_kev: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """The detected spectrum of a tungsten-target X-ray tube.

    Bremsstrahlung follows Kramers' thin-target law, with each photon
    absorbed on its way out of the target from the depth where the electron
    made it (Thomson-Whiddington slowing, Kanaya-Okayama range, leaving the
    target at ``target_angle`` degrees). Tungsten's K lines are not modelled;
    above 70 kV they add a few percent of the photons near 60 keV. The beam is
    then filtered by ``filters`` and detected by ``detector``, a scintillator
    whose signal is the energy it absorbs; ``None`` counts each photon's full
    energy.

    Parameters
    ----------
    kv : float
        Tube voltage in kV.
    filters : iterable of tuple, optional
        Layers between source and detector, including the tube window, as
        ``(material, thickness_mm)`` or ``(material, thickness_mm, density_g_cm3)``.
        A material is a chemical formula ("Al", "CuO") or a NIST name ("Water, Liquid").
    detector : tuple or None, optional
        Scintillator ``(material, thickness_mm[, density])``; None absorbs every photon.
    target_angle : float, optional
        Angle in degrees between the target surface and the beam. Default 12.
    step_kev : float, optional
        Energy bin width in keV. Default 0.5.

    Returns
    -------
    energies : numpy.ndarray
        Bin centres in keV.
    weights : numpy.ndarray
        Detected signal in each bin, summing to one.

    Raises
    ------
    ValueError
        If a setting is out of range or a material is unknown.
    ImportError
        If xraylib is not installed.
    """
    if not (np.isfinite(kv) and kv > 1):
        raise ValueError("tube_spectrum needs a voltage above 1 kV")
    if not (0 < target_angle <= 90 and step_kev > 0):
        raise ValueError("target_angle must be in (0, 90] degrees and step_kev positive")
    energies = np.arange(step_kev / 2, kv, step_kev)
    energies = energies[energies >= 1.0]
    w = _TUNGSTEN
    range_cm = 0.0276e-4 * w["A"] * kv**1.67 / (w["Z"] ** 0.889 * w["rho"])
    mu = attenuation("W", energies) * w["rho"] / np.sin(np.radians(target_angle))
    # Kramers: each electron at energy T makes photons of E < T at a rate ~ 1/(E T);
    # Thomson-Whiddington: it reaches T at depth range * (1 - T^2 / kv^2), with dT/dx ~ 1/T.
    # So the photons of energy E come from T in (E, kv), uniformly in T.
    t = np.linspace(0.0, 1.0, 65)[:, None]
    electron = energies + t * (kv - energies)
    depth = range_cm * (1 - (electron / kv) ** 2)
    escape = np.trapezoid(np.exp(-mu * depth), t, axis=0)
    photons = (kv - energies) / energies * escape
    photons = photons * _layers(filters, energies)
    if detector is None:
        signal = photons * energies
    else:
        material, thickness, rho = _layer(detector)
        xraylib = _xraylib()
        total = attenuation(material, energies)
        absorbed = np.array([xraylib.CS_Energy_CP(material, float(e)) for e in energies])
        stopped = 1 - np.exp(-total * rho * thickness / 10)
        signal = photons * energies * stopped * absorbed / total
    if not signal.sum() > 0:
        raise ValueError("no signal reaches the detector through these filters")
    return energies, signal / signal.sum()
