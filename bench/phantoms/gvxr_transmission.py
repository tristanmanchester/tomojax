"""Pixel-averaged transmission of closed STL meshes in parallel or cone beams, traced by gVXR.

Run it in the gVXR environment (``.artifacts/gvxr-env``, with ``gvxr``)::

    .artifacts/gvxr-env/bin/python bench/phantoms/gvxr_transmission.py PLAN.npz OUT.npz

``PLAN.npz`` holds ``meshes`` (a directory), ``names`` and ``mu_mm`` (one
linear attenuation coefficient per ``NAME.stl``, in mm^-1), the detector
``shape`` ``(nv, nu)``, ``pixel_mm``, ``centre_mm`` ``(u, v)``, ``supersample``
and ``poses``, TomoJAX world-from-object transforms ``(views, 4, 4)``.
Optional ``source_to_axis_mm`` and ``source_to_detector_mm`` select cone beam:
the source is at ``(0, -source_to_axis_mm, 0)`` and the detector at
``(centre_u, source_to_detector_mm - source_to_axis_mm, centre_v)`` with fixed
world x/z column/row directions. Without these keys the beam is parallel +y.
Detector coordinates follow TomoJAX:
``x = (u - nu/2 + 1/2) * pixel + centre_u`` (and likewise z for v).

gVXR traces each mesh's path length on a detector ``supersample`` times finer
in both directions. Each pixel is the mean of ``exp(-sum(mu * length))`` over
its sub-pixels, the photons it collects, so ``OUT.npz`` holds ``transmission``
``(views, nv, nu)`` plus gVXR version and renderer source hash; noise and the
logarithm are the caller's. Optional odd ``raster_refinement`` (default 1)
renders on a finer raster but keeps only each intended subray's exact centre;
it reduces rasterization error without changing the ``supersample`` quadrature.
Cone meshes must remain wholly between source and detector in every view.

For polychromatic data, ``mu_mm`` is ``(materials, energies)`` and
``photon_fractions`` contains nonnegative relative detected photon weights.
The weights are normalized; transmission is averaged over energies before
pixel averaging. This models an ideal photon counter with a spatially uniform
flat field, no scatter, blur or inverse-square illumination variation. Energy
values and material provenance belong in the caller's corpus metadata.
``save_energy_transmission=True`` additionally stores ``energy_transmission``
``(energy, view, v, u)`` as float32 pixel means. Their weighted sum recovers
``transmission`` to float32 roundoff, permitting new spectra without tracing
meshes again. No subray or path-length arrays are retained.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time

import numpy as np


def _spectrum(plan: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Material coefficients and normalized detected photon fractions."""
    mu = np.asarray(plan["mu_mm"], np.float64)
    if mu.ndim == 1:
        mu = mu[:, None]
        fractions = np.ones(1)
    else:
        fractions = np.asarray(plan["photon_fractions"], np.float64)
        if fractions.shape != (mu.shape[1],) or np.any(fractions < 0) or fractions.sum() <= 0:
            raise ValueError("photon_fractions must match energies and have positive total weight")
        fractions = fractions / fractions.sum()
    return mu, fractions


def render(plan: dict[str, np.ndarray]) -> dict[str, np.ndarray | str]:
    """Trace mesh lengths and integrate the plan's detected spectrum and pixels."""
    from gvxrPython3 import gvxr  # pyright: ignore[reportMissingImports]

    meshes = Path(str(plan["meshes"]))
    names = [str(n) for n in plan["names"]]
    mu, fractions = _spectrum(plan)
    nv, nu = (int(n) for n in plan["shape"])
    pixel = float(plan["pixel_mm"])
    centre_u, centre_v = (float(c) for c in plan["centre_mm"])
    s = int(plan["supersample"])
    refinement = int(plan.get("raster_refinement", 1))
    if refinement < 1 or refinement % 2 == 0:
        raise ValueError("raster_refinement must be a positive odd integer")
    poses = np.asarray(plan["poses"], np.float64)
    out = np.empty((len(poses), nv, nu), np.float32)
    energy_out = (
        np.empty((mu.shape[1], *out.shape), np.float32)
        if bool(plan.get("save_energy_transmission", False))
        else None
    )
    gvxr.createNewContext("EGL")
    try:
        cone = "source_to_axis_mm" in plan
        if cone:
            sod, sdd = float(plan["source_to_axis_mm"]), float(plan["source_to_detector_mm"])
            if not 0 < sod < sdd:
                raise ValueError(
                    "cone distances require 0 < source_to_axis_mm < source_to_detector_mm"
                )
            gvxr.usePointSource()
            source = np.array([0.0, -sod, 0.0])
            screen = np.array([centre_u, sdd - sod, centre_v])
        else:
            gvxr.useParallelBeam()
            source = np.array([centre_u, -100.0, centre_v])
            screen = np.array([centre_u, 100.0, centre_v])
        gvxr.setDetectorNumberOfPixels(nu * s * refinement, nv * s * refinement)
        gvxr.setDetectorPixelSize(pixel / (s * refinement), pixel / (s * refinement), "mm")
        for name in names:
            gvxr.loadMeshFile(name, str(meshes / f"{name}.stl"), "mm")
        gvxr.setMonoChromatic(100, "keV", 1)  # gVXR requires a spectrum to initialize shaders
        for i, pose in enumerate(poses):
            r, t = pose[:3, :3], pose[:3, 3]
            gvxr.setSourcePosition(*(r.T @ (source - t)), "mm")
            gvxr.setDetectorPosition(*(r.T @ (screen - t)), "mm")
            gvxr.setDetectorUpVector(*(r.T @ [0.0, 0.0, 1.0]))
            if cone:
                gvxr.setDetectorRightVector(*(r.T @ [1.0, 0.0, 0.0]))
            absorption = np.zeros((nv * s, nu * s, mu.shape[1]))
            for name, coefficient in zip(names, mu, strict=True):
                length_mm = (
                    10
                    * np.asarray(gvxr.computePathLength(name), np.float64)[
                        refinement // 2 :: refinement, refinement // 2 :: refinement
                    ]
                )
                absorption += length_mm[..., None] * coefficient
            transmission = np.exp(-absorption).reshape(nv, s, nu, s, -1).mean(axis=(1, 3))
            out[i] = transmission @ fractions
            if energy_out is not None:
                energy_out[:, i] = np.moveaxis(transmission, -1, 0)
        version = gvxr.getVersionOfCoreGVXR()
    finally:
        gvxr.destroy()
    return {
        "transmission": out,
        "gvxr": version,
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **({"energy_transmission": energy_out} if energy_out is not None else {}),
    }


def main() -> None:
    """Render the plan named first on the command line into the file named second."""
    start = time.perf_counter()
    with np.load(sys.argv[1]) as data:
        result = render(dict(data))
    np.savez(sys.argv[2], **result)
    print(f"gVXR {result['gvxr']}: rendered in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
