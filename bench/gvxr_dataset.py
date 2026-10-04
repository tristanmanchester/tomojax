#!/usr/bin/env python3
"""Generate independent mesh projections using an optional, isolated gVXR install.

All lengths are millimetres; reconstructed attenuation is mm^-1. The synthetic
five-bin spectrum is a stress case, not a calibrated tube or detector model.
Only NumPy is needed to import the geometry/normalization helpers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np

VERSION = "gvxr-materials-v1"
ENERGIES_KEV = np.array([40, 60, 80, 100, 120], dtype=np.float64)
PHOTON_FRACTIONS = np.array([0.12, 0.26, 0.30, 0.22, 0.10])
MATERIALS = (
    {
        "label": "water",
        "compound": "H2O",
        "density_g_cm3": 1.0,
        "center_mm": [-5.0, -1.0, 0.0],
        "extent_mm": [18.0, 24.0, 22.0],
    },
    {
        "label": "aluminium",
        "compound": "Al",
        "density_g_cm3": 2.70,
        "center_mm": [8.0, 2.0, -3.0],
        "extent_mm": [3.0, 10.0, 14.0],
    },
    {
        "label": "pmma",
        "compound": "C5H8O2",
        "density_g_cm3": 1.18,
        "center_mm": [7.0, -9.0, 5.0],
        "extent_mm": [5.0, 8.0, 12.0],
    },
)


def box_chords(
    base: np.ndarray, direction: np.ndarray, center: np.ndarray, extent: np.ndarray
) -> np.ndarray:
    """Exact infinite-ray intersection lengths through an axis-aligned cuboid."""
    lower, upper = center - extent / 2, center + extent / 2
    parallel = np.abs(direction) < 1e-14
    divisor = np.where(parallel, 1.0, direction)
    a, b = (lower - base) / divisor, (upper - base) / divisor
    inside = (base >= lower) & (base <= upper)
    near = np.where(parallel, np.where(inside, -np.inf, np.inf), np.minimum(a, b))
    far = np.where(parallel, np.where(inside, np.inf, -np.inf), np.maximum(a, b))
    length = np.maximum(np.min(far, axis=-1) - np.max(near, axis=-1), 0.0)
    return length * np.linalg.norm(direction, axis=-1)


def log_attenuation(
    counts: np.ndarray, incident_photons: float, *, floor_photons: float | None = None
) -> np.ndarray:
    """Normalize photon counts by a known flat field, optionally flooring zeros."""
    data = np.asarray(counts, dtype=np.float64)
    if not np.isfinite(incident_photons) or incident_photons <= 0:
        raise ValueError("Incident photon count must be finite and positive")
    if not np.all(np.isfinite(data)) or np.any(data < 0):
        raise ValueError("Photon counts must be finite and nonnegative")
    if floor_photons is not None:
        if not np.isfinite(floor_photons) or floor_photons <= 0:
            raise ValueError("Photon floor must be finite and positive")
        data = np.maximum(data, floor_photons)
    elif np.any(data == 0):
        raise ValueError("Zero photon counts require an explicit floor")
    return -np.log(data / incident_photons)


def rotation(angle: float, tilt: float) -> np.ndarray:
    """World-from-object R_x(tilt) R_z(angle), with angles in degrees."""
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    ct, st = np.cos(np.deg2rad(tilt)), np.sin(np.deg2rad(tilt))
    return np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]]) @ np.array(
        [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    )


def add_materials(gvxr: Any, coordinates: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Create three disjoint mesh materials and their voxel-centre reference."""
    materials = []
    truth = np.zeros(coordinates.shape[:-1], dtype=np.float64)
    for item in MATERIALS:
        label = item["label"]
        gvxr.makeCuboid(label, *item["extent_mm"], "mm")
        gvxr.translateNode(label, *item["center_mm"], "mm")
        gvxr.setCompound(label, item["compound"])
        gvxr.setDensity(label, item["density_g_cm3"], "g/cm3")
        gvxr.addPolygonMeshAsInnerSurface(label)
        # gVXR returns cm^-1, while every geometric length here is mm.
        mu = (
            np.array(
                [gvxr.getLinearAttenuationCoefficient(label, float(e), "keV") for e in ENERGIES_KEV]
            )
            / 10
        )
        materials.append({**item, "mu_mm_inverse": mu.tolist()})
        inside = np.all(
            np.abs(coordinates - item["center_mm"]) <= np.asarray(item["extent_mm"]) / 2,
            axis=-1,
        )
        truth[inside] += mu[2]
    return truth, materials


def reference_for_view(camera: np.ndarray, pose: np.ndarray, materials: list[dict]) -> np.ndarray:
    """Integrate material attenuation along physical rays, independently of meshes."""
    base = (camera - pose[:3, 3]) @ pose[:3, :3]
    optical_depth = np.zeros((*camera.shape[:-1], len(ENERGIES_KEV)))
    for item in materials:
        chord = box_chords(
            base, pose[1, :3], np.asarray(item["center_mm"]), np.asarray(item["extent_mm"])
        )
        optical_depth += chord[..., None] * item["mu_mm_inverse"]
    return optical_depth


def generate(size: int, views: int, tilt: float, photons: float, seed: int) -> tuple[dict, dict]:
    """Render a fixed material phantom and check it against analytic ray lengths."""
    from gvxrPython3 import gvxr

    spacing = 48.0 / size
    # Odd dimensions and sub-pixel offsets exercise orientation and centre conventions.
    nu, nv = size + 5 if size % 2 == 0 else size + 4, size + 3 if size % 2 == 0 else size + 2
    center = np.array([0.23 * spacing, -0.31 * spacing])
    angles = np.linspace(0, 180, views, endpoint=False)
    poses = np.broadcast_to(np.eye(4), (views, 4, 4)).copy()
    poses[:, :3, :3] = np.stack([rotation(float(a), tilt) for a in angles])
    poses[:, :3, 3] = [0.13, -0.17, 0.29]
    u, v = np.meshgrid(
        (np.arange(nu) - (nu - 1) / 2) * spacing + center[0],
        (np.arange(nv) - (nv - 1) / 2) * spacing + center[1],
    )
    camera = np.stack([u, np.zeros_like(u), v], axis=-1)
    coordinates = np.stack(
        np.meshgrid(*[(np.arange(size) - (size - 1) / 2) * spacing] * 3, indexing="ij"), axis=-1
    )
    raw_mono, raw_poly, ref_mono, ref_poly = [], [], [], []
    gvxr.createNewContext("EGL")
    try:
        gvxr.useParallelBeam()
        gvxr.disablePoissonNoise()
        # Finer raster coordinates reduce hardware triangle interpolation error.
        # Odd refinement preserves each original centre exactly; no area averaging.
        gvxr.setDetectorNumberOfPixels(nu * 5, nv * 5)
        gvxr.setDetectorPixelSize(spacing / 5, spacing / 5, "mm")
        truth, materials = add_materials(gvxr, coordinates)
        for pose in poses:
            r, t = pose[:3, :3], pose[:3, 3]
            detector_center = np.array([center[0], 0, center[1]])
            source = r.T @ (detector_center + np.array([0, -100, 0]) - t)
            detector = r.T @ (detector_center + np.array([0, 100, 0]) - t)
            gvxr.setSourcePosition(*source, "mm")
            gvxr.setDetectorPosition(*detector, "mm")
            gvxr.setDetectorUpVector(*(r.T @ [0, 0, 1]))
            np.testing.assert_allclose(gvxr.getDetectorRightVector(), r.T @ [1, 0, 0], atol=2e-6)
            gvxr.setMonoChromaticPerPixelAtSDD(80, "keV", photons)
            raw_mono.append(np.array(gvxr.computeXRayImage(False), dtype=np.float64)[2::5, 2::5])
            gvxr.resetBeamSpectrum()
            for energy, fraction in zip(ENERGIES_KEV, PHOTON_FRACTIONS, strict=True):
                gvxr.addEnergyBinToSpectrumPerPixelAtSDD(
                    float(energy), "keV", float(photons * fraction)
                )
            raw_poly.append(np.array(gvxr.computeXRayImage(False), dtype=np.float64)[2::5, 2::5])
            optical_depth = reference_for_view(camera, pose, materials)
            ref_mono.append(optical_depth[..., 2])
            ref_poly.append(-np.log(np.sum(PHOTON_FRACTIONS * np.exp(-optical_depth), axis=-1)))
        versions = {
            "gvxr": importlib.metadata.version("gvxr"),
            "numpy": np.__version__,
            "core": gvxr.getVersionOfCoreGVXR(),
            "simple": gvxr.getVersionOfSimpleGVXR(),
        }
    finally:
        gvxr.destroy()
    mono, poly = np.asarray(raw_mono), np.asarray(raw_poly)
    mono_log, poly_log = log_attenuation(mono, photons), log_attenuation(poly, photons)
    mono_ref, poly_ref = np.asarray(ref_mono), np.asarray(ref_poly)
    errors = {
        "monochromatic_relative_l2": float(
            np.linalg.norm(mono_log - mono_ref) / np.linalg.norm(mono_ref)
        ),
        "polychromatic_relative_l2": float(
            np.linalg.norm(poly_log - poly_ref) / np.linalg.norm(poly_ref)
        ),
        "monochromatic_max_absolute": float(np.max(np.abs(mono_log - mono_ref))),
        "polychromatic_max_absolute": float(np.max(np.abs(poly_log - poly_ref))),
    }
    # Frozen simulator checks, independent of TomoJAX output. Do not adapt the limit to a run.
    if max(errors[k] for k in errors if k.endswith("relative_l2")) > 1e-4:
        raise RuntimeError(f"gVXR projection geometry/attenuation check failed: {errors}")
    rng = np.random.Generator(np.random.PCG64(seed))
    noisy = rng.poisson(poly).astype(np.uint32)
    metadata = {
        "fixture_version": VERSION,
        "versions": versions,
        "platform": platform.platform(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "length_unit": "mm",
        "volume_unit": "mm^-1 at 80 keV",
        "renderer": "EGL",
        "shape": [size, size, size],
        "views": views,
        "tilt_deg": tilt,
        "grid": {"nx": size, "ny": size, "nz": size, "vx": spacing, "vy": spacing, "vz": spacing},
        "detector": {
            "nu": nu,
            "nv": nv,
            "du": spacing,
            "dv": spacing,
            "det_center": center.tolist(),
        },
        "pose_convention": "world-from-object; detector x,z plane; rays along +y",
        "materials": materials,
        "energy_keV": ENERGIES_KEV.tolist(),
        "photon_fractions": PHOTON_FRACTIONS.tolist(),
        "incident_photons_per_pixel": photons,
        "raster_refinement": {"factor": 5, "reduction": "centre sample only; no averaging"},
        "detector_model": "ideal photon-counting, point-sampled; no blur or scatter",
        "noise": {
            "model": "independent Poisson from gVXR mean counts",
            "rng": "PCG64",
            "seed": seed,
            "flat_field": "known noiseless incident photon count",
            "floor_photons": 0.5,
            "zero_count_pixels": int(np.count_nonzero(noisy == 0)),
        },
        "validation": errors,
        "validation_relative_l2_limit": 1e-4,
        "reference": "Independent analytic box-ray lengths using gVXR material attenuation tables",
        "limitations": (
            "Synthetic spectrum, not a calibrated scanner. Poly data do not satisfy "
            "a single-energy linear model. Voxel-centre truth has boundary "
            "discretization error."
        ),
    }
    arrays = {
        "truth": truth.astype(np.float32),
        "poses": poses.astype(np.float32),
        "angles": angles.astype(np.float32),
        "mono_counts": mono.astype(np.float32),
        "poly_counts": poly.astype(np.float32),
        "poly_noisy_counts": noisy,
        "mono_log": mono_log.astype(np.float32),
        "poly_log": poly_log.astype(np.float32),
        "poly_noisy_log": log_attenuation(noisy, photons, floor_photons=0.5).astype(np.float32),
        "mono_analytic": mono_ref.astype(np.float32),
        "poly_analytic": poly_ref.astype(np.float32),
    }
    return arrays, metadata


def main() -> None:
    """Generate a portable NPZ fixture and its human-readable metadata sidecar."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--views", type=int, default=180)
    parser.add_argument("--tilt", type=float, choices=[0.0, 30.0], default=0.0)
    parser.add_argument("--photons", type=float, default=100000)
    parser.add_argument("--seed", type=int, default=128904)
    parser.add_argument("--output", type=Path, default=Path("bench/results/gvxr-materials-v1.npz"))
    args = parser.parse_args()
    if (
        args.size < 8
        or args.views < 1
        or not np.isfinite(args.photons)
        or not 0 < args.photons <= 1e8
    ):
        parser.error("Require size >= 8, views >= 1 and 0 < photons <= 1e8")
    arrays, metadata = generate(args.size, args.views, args.tilt, args.photons, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays, metadata=json.dumps(metadata))
    metadata["npz_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "validation": metadata["validation"]}, indent=2))


if __name__ == "__main__":
    main()
