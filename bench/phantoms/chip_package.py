#!/usr/bin/env python3
"""Simulate a DIAD-like laminography scan of the chip-package phantom with gVXR.

The phantom (``chip_package_blender.py``) is eight closed STL meshes in mm.
gVXR traces each mesh's path length through a parallel monochromatic beam; this
script turns those into measurements with physics gVXR does not apply itself:

- attenuation and refractive decrement per material from xraylib at the beam energy
- propagation-based phase contrast: Fresnel propagation of the exit wave
- scintillator/optics blur (Gaussian PSF) and detector pixel integration
- Poisson noise, a fixed-pattern pixel gain and flat fields with their own noise
- per-view sample motion and a centre-of-rotation (detector-u) offset

Scattering, harmonics, beam drift and partial coherence are not modelled.

Three steps run in two environments. ``plan`` and ``finish`` use TomoJAX
(geometry, exact voxel truth, a consistency check against TomoJAX's projector,
NeXus output); ``render`` uses the isolated gVXR environment
(``.artifacts/gvxr-env``, with ``gvxr`` and ``xraylib``). ``build`` runs all three:

    uv run --no-sync python bench/phantoms/chip_package.py build OUT_DIR

Volumes are attenuation in mm^-1 at the beam energy, ``(x, y, z)`` order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GVXR_PYTHON = ROOT / ".artifacts" / "gvxr-env" / "bin" / "python"
BLENDER = ROOT / ".artifacts" / "tools" / "blender" / "blender-4.5.14-linux-x64" / "blender"

# (formula, mass fraction) components and density in g/cm^3.
EPOXY = "C21H24O4"  # bisphenol-A epoxy resin
MATERIALS: dict[str, dict[str, Any]] = {
    "substrate": {
        "components": [
            (EPOXY, 0.45),
            ("SiO2", 0.30),
            ("CaO", 0.13),
            ("Al2O3", 0.08),
            ("B2O3", 0.04),
        ],
        "density": 1.85,
    },
    "ground_plane": {"components": [("Cu", 1.0)], "density": 8.96},
    "vias": {"components": [("Cu", 1.0)], "density": 8.96},
    "traces": {"components": [("Cu", 1.0)], "density": 8.96},
    "die": {"components": [("Si", 1.0)], "density": 2.33},
    "wires": {"components": [("Au", 1.0)], "density": 19.32},
    "mold": {"components": [(EPOXY, 0.12), ("SiO2", 0.88)], "density": 1.95},
    "solder": {"components": [("Sn", 0.965), ("Ag", 0.03), ("Cu", 0.005)], "density": 7.38},
}
NESTED = {"die": "mold", "wires": "mold", "traces": "mold"}
SUPERSAMPLE = 3  # odd, so the centre sub-pixel is the detector pixel centre


def _defaults(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("out", type=Path)
    parser.add_argument("--energy-kev", type=float, default=25.0)
    parser.add_argument("--voxel-mm", type=float, default=0.008)
    parser.add_argument("--views", type=int, default=720)
    parser.add_argument("--tilt-deg", type=float, default=30.0)
    parser.add_argument("--propagation-mm", type=float, default=50.0)
    parser.add_argument("--psf-px", type=float, default=0.7, help="Gaussian PSF sigma, pixels")
    parser.add_argument("--photons", type=float, default=2e4, help="flat-field counts per pixel")
    parser.add_argument("--flats", type=int, default=20)
    parser.add_argument("--gain-percent", type=float, default=1.0)
    parser.add_argument("--cor-px", type=float, default=3.2, help="true detector-u offset")
    parser.add_argument("--shift-px", type=float, default=2.0, help="sample wobble amplitude")
    parser.add_argument("--rotation-deg", type=float, default=0.1, help="sample tilt jitter")
    parser.add_argument("--seed", type=int, default=20261006)


# ------------------------------------------------------------- the TomoJAX plan


def plan(args: argparse.Namespace) -> None:
    """Write the grid, nominal and true detector, poses and motion to ``plan.npz``."""
    import jax.numpy as jnp

    from tomojax.alignment.api import apply_pose_updates
    from tomojax.geometry import Detector, Grid, LaminographyGeometry, stack_view_poses

    v = args.voxel_mm
    # The package spans x, y in [-0.9, 0.9] mm and z in [-0.36, 0.30] mm.
    grid = Grid(240, 240, 88, v, v, v, vol_center=(0.0, 0.0, -0.03))
    angles = np.linspace(0.0, 360.0, args.views, endpoint=False)
    probe = Detector(1, 1, v, v)
    poses = np.asarray(
        stack_view_poses(LaminographyGeometry(grid, probe, angles, args.tilt_deg), args.views)
    )
    corners = (
        np.array(np.meshgrid([-0.9, 0.9], [-0.9, 0.9], [-0.37, 0.31], indexing="ij"))
        .reshape(3, -1)
        .T
    )
    world = corners @ poses[:, :3, :3].transpose(0, 2, 1) + poses[:, None, :3, 3]
    margin = 6 + int(np.ceil(abs(args.cor_px) + args.shift_px))
    nu = 2 * (int(np.ceil(np.abs(world[..., 0]).max() / v)) + margin)
    nv = 2 * (int(np.ceil(np.abs(world[..., 2]).max() / v)) + margin)
    nominal = Detector(nu, nv, v, v)
    geometry = LaminographyGeometry(grid, nominal, angles, args.tilt_deg)
    nominal_poses = stack_view_poses(geometry, args.views)
    rng = np.random.default_rng(args.seed)
    theta = np.deg2rad(angles)
    rotation = np.deg2rad(args.rotation_deg)
    params = np.zeros((args.views, 5))
    # Slow drift plus jitter in tilt; eccentric wobble plus jitter in position.
    params[:, 0] = rotation * (0.5 * np.sin(theta + 0.4) + rng.uniform(-0.5, 0.5, args.views))
    params[:, 1] = rotation * (0.5 * np.cos(2 * theta) + rng.uniform(-0.5, 0.5, args.views))
    params[:, 2] = rotation * rng.uniform(-0.5, 0.5, args.views)
    params[:, 3] = (
        v * args.shift_px * (0.7 * np.sin(theta + 1.1) + rng.uniform(-0.3, 0.3, args.views))
    )
    params[:, 4] = (
        v * args.shift_px * (0.4 * np.sin(2 * theta) + rng.uniform(-0.2, 0.2, args.views))
    )
    true_poses = np.asarray(
        apply_pose_updates(
            nominal_poses, jnp.asarray(params, jnp.float32), translation_frame="detector"
        ),
        np.float64,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out / "plan.npz",
        grid=json.dumps(grid.to_dict()),
        detector=json.dumps(nominal.to_dict()),
        angles_deg=angles,  # the plan file's key
        tilt_deg=args.tilt_deg,
        nominal_poses=np.asarray(nominal_poses, np.float64),
        true_poses=true_poses,
        static_poses=np.asarray(nominal_poses, np.float64),
        true_params=params,
        cor_mm=args.cor_px * v,
    )
    print(f"plan: grid {grid.nx}x{grid.ny}x{grid.nz}, detector {nu}x{nv}, {args.views} views")


# ------------------------------------------------------------------ render (gVXR env)


def material_table(energy_kev: float) -> dict[str, dict[str, float]]:
    """Linear attenuation (mm^-1) and refractive decrement per material."""
    import xraylib

    r_e_cm, avogadro = 2.8179403262e-13, 6.02214076e23
    wavelength_cm = 1.23984198e-7 / energy_kev
    table = {}
    for name, spec in MATERIALS.items():
        weights: dict[int, float] = {}
        for formula, fraction in spec["components"]:
            parsed = xraylib.CompoundParser(formula)
            for z, w in zip(parsed["Elements"], parsed["massFractions"], strict=True):
                weights[z] = weights.get(z, 0.0) + fraction * w
        rho = spec["density"]
        mu_cm = rho * sum(w * xraylib.CS_Total(z, energy_kev) for z, w in weights.items())
        electrons = sum(
            w * (z + xraylib.Fi(z, energy_kev)) / xraylib.AtomicWeight(z)
            for z, w in weights.items()
        )
        delta = r_e_cm * wavelength_cm**2 / (2 * np.pi) * avogadro * rho * electrons
        table[name] = {"mu_mm": mu_cm / 10, "delta": float(delta), "density": rho}
    return table


def _effective(table: dict[str, dict[str, float]], key: str) -> dict[str, float]:
    # A nested part replaces its parent's material along its own path length.
    out = {}
    for name, entry in table.items():
        parent = NESTED.get(name)
        out[name] = entry[key] - (table[parent][key] if parent else 0.0)
    return out


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    pad = int(np.ceil(4 * sigma))
    padded = np.pad(image, pad, mode="edge")
    fy = np.fft.fftfreq(padded.shape[0])[:, None]
    fx = np.fft.rfftfreq(padded.shape[1])[None, :]
    kernel = np.exp(-2 * (np.pi * sigma) ** 2 * (fx**2 + fy**2))
    blurred = np.fft.irfft2(np.fft.rfft2(padded) * kernel, s=padded.shape)
    return blurred[pad:-pad, pad:-pad]


def _propagate(
    wave: np.ndarray, pixel_mm: float, wavelength_mm: float, distance_mm: float
) -> np.ndarray:
    # Free-space Fresnel propagation; the field beyond the detector is unit plane wave.
    if distance_mm <= 0:
        return wave
    ny, nx = wave.shape
    padded = np.ones((2 * ny, 2 * nx), np.complex128)
    padded[ny // 2 : ny // 2 + ny, nx // 2 : nx // 2 + nx] = wave
    fy = np.fft.fftfreq(2 * ny, pixel_mm)[:, None]
    fx = np.fft.fftfreq(2 * nx, pixel_mm)[None, :]
    transfer = np.exp(-1j * np.pi * wavelength_mm * distance_mm * (fx**2 + fy**2))
    out = np.fft.ifft2(np.fft.fft2(padded) * transfer)
    return out[ny // 2 : ny // 2 + ny, nx // 2 : nx // 2 + nx]


def render(args: argparse.Namespace) -> None:
    """Trace mesh path lengths with gVXR and form ideal and DIAD-like measurements."""
    from gvxrPython3 import gvxr

    data = np.load(args.out / "plan.npz")
    detector = json.loads(str(data["detector"]))
    nu, nv, du, dv = detector["nu"], detector["nv"], detector["du"], detector["dv"]
    table = material_table(args.energy_kev)
    mu, delta = _effective(table, "mu_mm"), _effective(table, "delta")
    wavelength_mm = 1.23984198e-6 / args.energy_kev
    k = 2 * np.pi / wavelength_mm
    s = SUPERSAMPLE
    outputs: dict[str, np.ndarray] = {}
    gvxr.createNewContext("EGL")
    try:
        gvxr.useParallelBeam()
        gvxr.setDetectorNumberOfPixels(nu * s, nv * s)
        gvxr.setDetectorPixelSize(du / s, dv / s, "mm")
        for name in MATERIALS:
            gvxr.loadMeshFile(name, str(args.meshes / f"{name}.stl"), "mm")
            gvxr.setCompound(name, "Si")  # unused: attenuation is formed from path lengths
            gvxr.setDensity(name, 1.0, "g/cm3")
            gvxr.addPolygonMeshAsInnerSurface(name)
        gvxr.setMonoChromatic(args.energy_kev, "keV", 1)
        for key, centre_u in (("static", 0.0), ("moving", float(data["cor_mm"]))):
            poses = data["static_poses" if key == "static" else "true_poses"]
            ideal, measured = [], []
            for pose in poses:
                r, t = pose[:3, :3], pose[:3, 3]
                source = np.array([centre_u, -100.0, 0.0])
                screen = np.array([centre_u, 100.0, 0.0])
                gvxr.setSourcePosition(*(r.T @ (source - t)), "mm")
                gvxr.setDetectorPosition(*(r.T @ (screen - t)), "mm")
                gvxr.setDetectorUpVector(*(r.T @ [0, 0, 1]))
                absorption = np.zeros((nv * s, nu * s))
                phase = np.zeros_like(absorption)
                for name in MATERIALS:
                    length_mm = 10 * np.asarray(gvxr.computePathLength(name), np.float64)
                    absorption += mu[name] * length_mm
                    phase += delta[name] * length_mm
                ideal.append(absorption[s // 2 :: s, s // 2 :: s])
                wave = np.exp(-absorption / 2 - 1j * k * phase)
                intensity = (
                    np.abs(_propagate(wave, du / s, wavelength_mm, args.propagation_mm)) ** 2
                )
                intensity = _gaussian_blur(intensity, args.psf_px * s)
                measured.append(intensity.reshape(nv, s, nu, s).mean(axis=(1, 3)))
            outputs[f"{key}_line_integrals"] = np.asarray(ideal, np.float32)
            outputs[f"{key}_intensity"] = np.asarray(measured, np.float32)
            print(f"render: {key} done")
        version = gvxr.getVersionOfCoreGVXR()
    finally:
        gvxr.destroy()
    np.savez(args.out / "render.npz", **outputs)
    (args.out / "materials.json").write_text(
        json.dumps({"energy_keV": args.energy_kev, "gvxr": version, "materials": table}, indent=2)
    )


# ---------------------------------------------------------- the TomoJAX finish


def _truth_volume(meshes: Path, grid: Any, table: dict[str, dict[str, float]]) -> np.ndarray:
    sys.path.insert(0, str(HERE))
    from mesh_voxels import occupancy, read_stl

    from tomojax.geometry import grid_volume_origin

    origin = tuple(grid_volume_origin(grid))
    spacing = (grid.vx, grid.vy, grid.vz)
    mu = _effective(table, "mu_mm")
    volume = np.zeros((grid.nx, grid.ny, grid.nz))
    for name in MATERIALS:
        fraction = occupancy(
            read_stl(meshes / f"{name}.stl"), (grid.nx, grid.ny, grid.nz), spacing, origin
        )
        volume += mu[name] * fraction
    return volume.astype(np.float32)


def finish(args: argparse.Namespace) -> None:
    """Voxel truth, a projector consistency check, noise, and NeXus datasets."""
    import jax.numpy as jnp

    from tomojax.forward import project_joseph
    from tomojax.geometry import Detector, Grid
    from tomojax.io.api import NXTomoMetadata, save_nxtomo

    data = np.load(args.out / "plan.npz")
    rendered = np.load(args.out / "render.npz")
    grid = Grid(**json.loads(str(data["grid"])))
    nominal = Detector.from_dict(json.loads(str(data["detector"])))
    table = json.loads((args.out / "materials.json").read_text())["materials"]
    truth = _truth_volume(args.meshes, grid, table)
    # The Beer-Lambert line integrals must match TomoJAX's projector applied to
    # the voxel truth, up to the voxelisation of thin parts.
    shifted = Detector(nominal.nu, nominal.nv, nominal.du, nominal.dv, (float(data["cor_mm"]), 0.0))
    checks = {}
    for key, detector in (("static", nominal), ("moving", shifted)):
        poses = jnp.asarray(data["static_poses" if key == "static" else "true_poses"], jnp.float32)
        reference = rendered[f"{key}_line_integrals"]
        predicted = np.concatenate(
            [
                np.asarray(project_joseph(jnp.asarray(truth), poses[i : i + 32], grid, detector))
                for i in range(0, len(poses), 32)
            ]
        )
        checks[key] = float(np.linalg.norm(predicted - reference) / np.linalg.norm(reference))
    rng = np.random.default_rng(args.seed + 1)
    gain = 1 + args.gain_percent / 100 * rng.standard_normal((nominal.nv, nominal.nu))
    flat = rng.poisson(args.photons * gain, size=(args.flats, *gain.shape)).mean(axis=0)
    metadata = NXTomoMetadata(
        angles=data["angles_deg"].astype(np.float32),
        grid=grid.to_dict(),
        detector=nominal.to_dict(),
        geometry_type="lamino",
        geometry_meta={"tilt_deg": float(data["tilt_deg"]), "tilt_about": "x"},
    )
    for key in ("static", "moving"):
        save_nxtomo(
            str(args.out / f"{key}-ideal.nxs"), rendered[f"{key}_line_integrals"], metadata=metadata
        )
        counts = rng.poisson(args.photons * gain * rendered[f"{key}_intensity"])
        realistic = -np.log(np.maximum(counts, 0.5) / flat)
        save_nxtomo(
            str(args.out / f"{key}-realistic.nxs"), realistic.astype(np.float32), metadata=metadata
        )
    np.savez_compressed(
        args.out / "truth.npz",
        volume=truth,
        true_params=data["true_params"],
        true_poses=data["true_poses"],
        nominal_poses=data["nominal_poses"],
        cor_mm=data["cor_mm"],
    )
    summary = {
        "projector_check_relative_l2": checks,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"finish: projector check {checks}")


def build(args: argparse.Namespace) -> None:
    """Model the meshes if needed, then plan, render and finish."""
    if not (args.meshes / "mold.stl").exists():
        subprocess.run(
            [
                str(BLENDER),
                "-b",
                "--factory-startup",
                "-P",
                str(HERE / "chip_package_blender.py"),
                "--",
                str(args.meshes),
            ],
            check=True,
        )
    plan(args)
    forwarded = [arg for arg in sys.argv[2:] if arg != "build"]
    subprocess.run([str(GVXR_PYTHON), __file__, "render", *forwarded], check=True)
    finish(args)


def main() -> None:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "render", "finish", "build"):
        sub = commands.add_parser(name)
        _defaults(sub)
        sub.add_argument(
            "--meshes", type=Path, default=ROOT / ".artifacts" / "phantoms" / "chip_package"
        )
    args = parser.parse_args()
    {"plan": plan, "render": render, "finish": finish, "build": build}[args.command](args)


if __name__ == "__main__":
    main()
