"""Pixel-averaged transmission of closed STL meshes in a parallel beam, traced by gVXR.

Run it in the gVXR environment (``.artifacts/gvxr-env``, with ``gvxr``)::

    .artifacts/gvxr-env/bin/python bench/phantoms/gvxr_transmission.py PLAN.npz OUT.npz

``PLAN.npz`` holds ``meshes`` (a directory), ``names`` and ``mu_mm`` (one
linear attenuation coefficient per ``NAME.stl``, in mm^-1), the detector
``shape`` ``(nv, nu)``, ``pixel_mm``, ``centre_mm`` ``(u, v)``, ``supersample``
and ``poses``, TomoJAX world-from-object transforms ``(views, 4, 4)``. As in
TomoJAX, rays run along world +y through detector pixels at
``x = (u - nu/2 + 1/2) * pixel + centre_u`` (and likewise z for v).

gVXR traces each mesh's path length on a detector ``supersample`` times finer
in both directions. Each pixel is the mean of ``exp(-sum(mu * length))`` over
its sub-pixels, the photons it collects, so ``OUT.npz`` holds ``transmission``
``(views, nv, nu)`` and nothing else: noise and the logarithm are the caller's.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np


def main() -> None:
    """Render the plan named first on the command line into the file named second."""
    from gvxrPython3 import gvxr  # pyright: ignore[reportMissingImports]

    plan = np.load(sys.argv[1])
    meshes = Path(str(plan["meshes"]))
    names = [str(n) for n in plan["names"]]
    mu = np.asarray(plan["mu_mm"], np.float64)
    nv, nu = (int(n) for n in plan["shape"])
    pixel = float(plan["pixel_mm"])
    centre_u, centre_v = (float(c) for c in plan["centre_mm"])
    s = int(plan["supersample"])
    poses = np.asarray(plan["poses"], np.float64)
    out = np.empty((len(poses), nv, nu), np.float32)
    start = time.perf_counter()
    gvxr.createNewContext("EGL")
    try:
        gvxr.useParallelBeam()
        gvxr.setDetectorNumberOfPixels(nu * s, nv * s)
        gvxr.setDetectorPixelSize(pixel / s, pixel / s, "mm")
        for name in names:
            gvxr.loadMeshFile(name, str(meshes / f"{name}.stl"), "mm")
            gvxr.setCompound(name, "Si")  # unused: attenuation is formed from path lengths
            gvxr.setDensity(name, 1.0, "g/cm3")
            gvxr.addPolygonMeshAsInnerSurface(name)
        gvxr.setMonoChromatic(100, "keV", 1)  # unused, as above
        source = np.array([centre_u, -100.0, centre_v])
        screen = np.array([centre_u, 100.0, centre_v])
        for i, pose in enumerate(poses):
            r, t = pose[:3, :3], pose[:3, 3]
            gvxr.setSourcePosition(*(r.T @ (source - t)), "mm")
            gvxr.setDetectorPosition(*(r.T @ (screen - t)), "mm")
            gvxr.setDetectorUpVector(*(r.T @ [0.0, 0.0, 1.0]))
            absorption = np.zeros((nv * s, nu * s))
            for name, coefficient in zip(names, mu, strict=True):
                length_mm = 10 * np.asarray(gvxr.computePathLength(name), np.float64)
                absorption += coefficient * length_mm
            out[i] = np.exp(-absorption).reshape(nv, s, nu, s).mean(axis=(1, 3))
        version = gvxr.getVersionOfCoreGVXR()
    finally:
        gvxr.destroy()
    np.savez(sys.argv[2], transmission=out, gvxr=version)
    print(f"gVXR {version}: {len(poses)} views of {nu}x{nv} in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
