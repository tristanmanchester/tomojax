#!/usr/bin/env python3
"""Render one polychromatic bench case with gVXR, SpekPy or xpecgen spectra and xraylib.

Run in the gVXR environment, not TomoJAX's::

    python simulate.py --split dev --index 3 --output case.npz

gVXR rasterises the path length through every cylinder mesh for each view
(its L-buffer); nested shapes become per-material path lengths by
subtraction. The tube spectrum is filtered by the window, the recorded
filters and some unrecorded aluminium, and detected through gVXR's
scintillator energy response, with energy-weighted quantum noise and
scintillator blur. Two checks run on every case: the L-buffers against exact
ray-cylinder chords in every view (the rare pixel gVXR rasterises beyond a
silhouette takes the chord), and the spectral integration against gVXR's own
image (99th percentile, as that image has the same rare silhouette columns).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cases
import numpy as np

ENERGY_GRID = np.arange(1.0, 301.0)  # keV, for attenuation at any reference energy


def _mu(formula: str, density: float, energies: np.ndarray) -> np.ndarray:
    import xraylib

    return np.array([xraylib.CS_Total_CP(formula, float(e)) for e in energies]) * density / 10


FILTER_DENSITY = {"Be": 1.848, "Al": 2.699, "Cu": 8.96, "Sn": 7.31}


def spectrum(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Bin centres (keV) and relative photon fluence of the filtered tube output."""
    kv, angle = spec["kv"], spec["hidden"]["anode_angle_deg"]
    if spec["generator"] == "spekpy":
        import spekpy

        energies, fluence = spekpy.Spek(kvp=kv, th=angle, dk=1.0, targ="W").get_spectrum()
    else:
        from xpecgen import xpecgen

        tube = xpecgen.calculate_spectrum(kv, angle, 3.0, int(kv), epsrel=0.5, monitor=None, z=74)
        energies = np.arange(1.5, np.floor(kv) + 0.5)
        fluence = np.interp(energies, tube.x, tube.y, left=0.0, right=0.0)
        for line in tube.discrete:
            k = int(np.clip(np.floor(line[0] - 1.0), 0, energies.size - 1))
            fluence[k] += line[1]
    filtration = [spec["window"], *spec["filters"], ["Al", spec["hidden"]["extra_al_mm"]]]
    for element, thickness in filtration:
        fluence = fluence * np.exp(-_mu(element, FILTER_DENSITY[element], energies) * thickness)
    return np.asarray(energies, float), np.maximum(np.asarray(fluence, float), 0.0)


def cylinder_chords(
    start: np.ndarray,
    end: np.ndarray,
    centre: list[float],
    radius: float,
    half_height: float | None,
) -> np.ndarray:
    """Exact chord (mm) of each ray segment through a z-aligned cylinder, finite or not."""
    d = end - start
    rel = start[..., :2] - np.asarray(centre)
    a = d[..., 0] ** 2 + d[..., 1] ** 2
    b = 2 * (d[..., 0] * rel[..., 0] + d[..., 1] * rel[..., 1])
    c = (rel**2).sum(-1) - radius**2
    root = np.sqrt(np.maximum(b**2 - 4 * a * c, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        # A ray along the axis is inside the cylinder throughout or nowhere.
        low = np.where(a > 0, (-b - root) / (2 * a), np.where(c <= 0, -np.inf, np.inf))
        high = np.where(a > 0, (-b + root) / (2 * a), np.where(c <= 0, np.inf, -np.inf))
    if half_height is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            bottom = (-half_height - start[..., 2]) / d[..., 2]
            top = (half_height - start[..., 2]) / d[..., 2]
        flat = d[..., 2] == 0  # parallel to the faces: inside them or not at all
        inside = np.abs(start[..., 2]) <= half_height
        low = np.maximum(
            low, np.where(flat, np.where(inside, -np.inf, np.inf), np.minimum(bottom, top))
        )
        high = np.minimum(
            high, np.where(flat, np.where(inside, np.inf, -np.inf), np.maximum(bottom, top))
        )
    return np.maximum(high - low, 0.0) * np.linalg.norm(d, axis=-1)


def _pose(case: dict, angle: float) -> np.ndarray:
    """World-from-object rotation of one view: about z, then the laminography tilt."""
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    t = np.deg2rad(case.get("tilt_deg", 0.0))  # the axis leans from +z towards the beam (+y)
    tilt = np.array([[1, 0, 0], [0, np.cos(t), np.sin(t)], [0, -np.sin(t), np.cos(t)]])
    return tilt @ turn


def _rays(case: dict) -> tuple[np.ndarray, np.ndarray]:
    """Lab-frame start and end (mm) of the ray to every detector pixel, ``(nv, nu, 3)``."""
    g = cases.GEOMETRIES[case["geometry"]]
    u = (np.arange(g["nu"]) - (g["nu"] - 1) / 2) * g["pixel"]
    v = (np.arange(g["nv"]) - (g["nv"] - 1) / 2) * g["pixel"]
    if g["beam"] == "cone":
        far = g["source_to_detector"] - g["source_to_axis"]
        end = np.stack(np.broadcast_arrays(u[None, :], far, v[:, None]), axis=-1)
        return np.broadcast_to([0.0, -g["source_to_axis"], 0.0], end.shape), end
    end = np.stack(np.broadcast_arrays(u[None, :], 100.0, v[:, None]), axis=-1)
    return end - [0.0, 200.0, 0.0], end


def _scene(gvxr: Any, case: dict, energies: np.ndarray, fluence: np.ndarray) -> np.ndarray:
    """Detector, spectrum and cylinder meshes; returns gVXR's open-beam image."""
    g = cases.GEOMETRIES[case["geometry"]]
    if g["beam"] == "cone":
        gvxr.useConeBeam()
    else:
        gvxr.useParallelBeam()
    gvxr.setDetectorNumberOfPixels(g["nu"], g["nv"])
    gvxr.setDetectorPixelSize(g["pixel"], g["pixel"], "mm")
    for k, e in enumerate(energies):
        gvxr.addEnergyBinToSpectrumPerPixelAtSDD(float(e), "keV", float(fluence[k]))
    _place_beam(gvxr, case, 0.0)
    # gVXR's open beam falls off towards the detector edges; checks compare transmission.
    # It renders only with a mesh in the scene, so one sits far outside the beam.
    shapes = [{"material": "air", "centre": [0.0, 0.0], "radius": 0.1}, *case["shapes"]]
    for i, shape in enumerate(shapes):
        label = f"s{i - 1}"
        if i:
            height = 2 * shape["half_height"] if "half_height" in shape else 60.0
            gvxr.makeCylinder(label, 1000, height, shape["radius"], "mm")
            # Node transforms compose locally: this turns the axis from y to z, then moves it.
            gvxr.translateNode(label, *shape["centre"], 0.0, "mm")
            gvxr.rotateNode(label, 0, 0, 90)
        else:
            gvxr.makeSphere(label, 4, 4, 0.1, "mm")
            gvxr.translateNode(label, 0.0, 0.0, 1000.0, "mm")
        formula, density = cases.MATERIALS.get(shape["material"], ("N2", 0.0012))
        gvxr.setCompound(label, formula)
        gvxr.setDensity(label, density, "g/cm3")
        gvxr.addPolygonMeshAsInnerSurface(label)
        if not i:
            open_beam = 1000 * np.array(gvxr.computeXRayImage(True), np.float64)
    return open_beam


def _place_beam(gvxr: Any, case: dict, angle: float) -> np.ndarray:
    """Put the source and detector where the object, turned to ``angle``, sees them."""
    g = cases.GEOMETRIES[case["geometry"]]
    to_object = _pose(case, angle).T
    if g["beam"] == "cone":
        source, detector = (
            [0.0, -g["source_to_axis"], 0.0],
            [0.0, g["source_to_detector"] - g["source_to_axis"], 0.0],
        )
    else:
        source, detector = [0.0, -100.0, 0.0], [0.0, 100.0, 0.0]
    gvxr.setSourcePosition(*(to_object @ source), "mm")
    gvxr.setDetectorPosition(*(to_object @ detector), "mm")
    gvxr.setDetectorUpVector(*(to_object @ [0.0, 0.0, 1.0]))
    return to_object


def _lengths(gvxr: Any, case: dict, angle: float, checks: dict) -> tuple[list, list]:
    """Path length (mm) through every shape at one view, from gVXR: as rendered and repaired."""
    to_object = _place_beam(gvxr, case, angle)
    start, end = (r @ to_object.T for r in _rays(case))
    rendered, repaired = [], []
    for i, shape in enumerate(case["shapes"]):
        length = 10 * np.array(gvxr.computeLBuffer(f"s{i}"), np.float64)
        chord = cylinder_chords(
            start, end, shape["centre"], shape["radius"], shape.get("half_height")
        )
        error = length - chord
        # gVXR occasionally rasterises one column beyond a silhouette; take the chord there.
        glitch = np.abs(error) > 0.2
        checks["repaired_pixels"] += int(glitch.sum())
        checks["chord_max_mm"] = max(checks["chord_max_mm"], float(np.abs(error[~glitch]).max()))
        rms = float(np.sqrt(np.mean(error[~glitch] ** 2)))
        checks["chord_rms_mm"] = max(checks["chord_rms_mm"], rms)
        rendered.append(length)
        repaired.append(np.where(glitch, chord, length))
    return rendered, repaired


def render(case: dict) -> tuple[dict, dict]:
    """Counts, flats, darks and per-material truth for one case."""
    from importlib.metadata import version

    from gvxrPython3 import gvxr

    g, spec, shapes = cases.GEOMETRIES[case["geometry"]], case["spectrum"], case["shapes"]
    energies, fluence = spectrum(spec)
    materials = sorted({s["material"] for s in shapes} - {"air"})
    children = [
        [j for j, s in enumerate(shapes) if s.get("parent") == i] for i in range(len(shapes))
    ]
    shape = (g["views"], g["nv"], g["nu"])
    rng = np.random.default_rng(case["seed"])
    checks = {"chord_max_mm": 0.0, "chord_rms_mm": 0.0, "repaired_pixels": 0}
    checks["spectral_p99_relative"] = 0.0
    gvxr.createNewContext("EGL")
    try:
        gvxr.setScintillator(spec["scintillator"][0], spec["hidden"]["scintillator_mm"], "mm")
        table = np.array(gvxr.getEnergyResponse("keV"))
        # Energy-integrating detector: mean deposit R(E) per photon, variance ~ R(E) E.
        weight = fluence * np.interp(energies, table[:, 0], table[:, 1])
        keep = weight > weight.sum() * 1e-7
        energies, fluence, weight = energies[keep], fluence[keep], weight[keep]
        variance = weight * energies
        mu = {m: _mu(*cases.MATERIALS[m], energies) for m in materials}
        # Counts hold flat_quanta noise-equivalent quanta in the flat field.
        gain = variance.sum() / weight.sum()
        scale = case["flat_quanta"] * variance.sum() / weight.sum() ** 2
        open_beam = _scene(gvxr, case, energies, fluence)
        counts = np.empty(shape, np.float32)
        paths = {m: np.zeros(shape, np.float32) for m in materials}
        for view in range(g["views"]):
            rendered, lengths = _lengths(gvxr, case, view * 360.0 / g["views"], checks)
            for i, s in enumerate(shapes):
                if s["material"] != "air":
                    paths[s["material"]][view] += lengths[i] - sum(lengths[j] for j in children[i])
            depth = sum(np.multiply.outer(mu[m], paths[m][view]) for m in materials)
            transmitted = np.exp(-depth)
            mean = np.tensordot(weight, transmitted, 1) * scale / gain
            sd = np.sqrt(np.tensordot(variance, transmitted, 1) * scale) / gain
            counts[view] = _blur(mean + sd * rng.standard_normal(mean.shape), case["blur_px"])
            if view % 90 == 0:
                # gVXR sums attenuation over overlapping meshes; integrate the same here.
                total = sum(
                    np.multiply.outer(_mesh_mu(s, energies), rendered[i])
                    for i, s in enumerate(shapes)
                )
                ours = np.tensordot(weight, np.exp(-total), 1) / weight.sum()
                theirs = 1000 * np.array(gvxr.computeXRayImage(True), np.float64) / open_beam
                # Its image can rasterise a silhouette column differently from its L-buffers.
                error = float(np.percentile(np.abs(ours - theirs), 99))
                checks["spectral_p99_relative"] = max(checks["spectral_p99_relative"], error)
        checks["versions"] = {p: version(p) for p in ("gvxr", "xraylib", "spekpy", "xpecgen")}
    finally:
        gvxr.destroy()
    # Near-tangent rays see the 1000-sided polygon, not the circle: bound the maximum loosely.
    limits = {"chord_max_mm": 0.2, "chord_rms_mm": 0.01, "spectral_p99_relative": 1e-3}
    limits["repaired_pixels"] = 1e-4 * counts.size * len(shapes)
    if any(checks[k] > limit for k, limit in limits.items()):
        raise RuntimeError(f"simulation check failed: {checks}")
    dark, flat = 100.0, weight.sum() * scale / gain
    arrays = {
        "counts": counts + np.float32(dark),
        "flat": (dark + flat + rng.normal(0, np.sqrt(flat / 20), shape[1:]))[None].astype(
            np.float32
        ),
        "dark": np.full((1, *shape[1:]), dark, np.float32),
        "energies": energies,
        "weights": weight / weight.sum(),
        "mu_grid": np.stack([_mu(*cases.MATERIALS[m], ENERGY_GRID) for m in materials]),
        **{f"path_{m}": paths[m] for m in materials},
    }
    return arrays, {"case": case, "materials": materials, "checks": checks}


def _mesh_mu(shape: dict, energies: np.ndarray) -> np.ndarray:
    if shape["material"] == "air":
        return _mu("N2", 0.0012, energies)
    return _mu(*cases.MATERIALS[shape["material"]], energies)


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(image, sigma, mode="nearest")


def main() -> None:
    """Render the requested case to an NPZ file and print its checks."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--split", choices=["dev", "heldout"], required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--geometry", choices=sorted(cases.GEOMETRIES), default="cone")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrays, info = render(cases.sample(args.split, args.index, args.geometry))
    np.savez(args.output, **arrays, info=json.dumps(info))
    print(json.dumps({"case": info["case"]["name"], **info["checks"]}))


if __name__ == "__main__":
    main()
