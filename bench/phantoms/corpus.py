"""Prepare and package independent mesh tomography cases; rendering is a separate GPU step.

    JAX_PLATFORMS=cpu python bench/phantoms/corpus.py prepare CONFIG.json CASE_DIR
    # Run in the gVXR environment, once per plan:
    python bench/phantoms/gvxr_transmission.py CASE/clean-plan.npz CASE/clean-render.npz
    # Likewise clean-motion, exterior, motion, exterior-only and moving-exterior plans.
    python bench/phantoms/corpus.py pack CASE_DIR --photons 1e8 1e5 1e4 1e3

CONFIG supplies geometry (kind, grid, detector, angles_deg, tilt/beam), full,
interior and exterior mesh directories, material names and reference mu_mm,
reference_energy_keV, and seed. Optional mu_spectrum/energies_keV/photon_fractions
select a photon-counting spectrum. Grid lengths and mesh coordinates are mm.
Material truth is voxelised directly from closed meshes, never projected by
TomoJAX. The mesh partition must match the grid's physical box.

Prepare runs on CPU; only the external renderer requires the GPU. Pack writes
self-contained NPZs with truth, nominal/true matrices and simulator provenance.
Held-out indices are fixed before noise generation. A corpus is not validation:
inspect renders and reconstructions before using a case as realism evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mesh_voxels import occupancy, read_stl
import numpy as np


def sha256(path: Path) -> str:
    """Hash a source or mesh without allocating a second mesh-sized byte buffer."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def moved_poses(nominal: np.ndarray, pixel: float) -> np.ndarray:
    """Known axis error, detector offset, smooth drift and two jumps in physical units.

    The fixed 0.35 degree axis error is a lab-x rotation. Small object rotations
    are Ry(beta) Rx(alpha) Rz(phi); translations are lab xyz. Parallel beam-y
    translation is zero (unobservable). Absolute truth keeps this original gauge.
    """
    n = len(nominal)
    phase = 2 * np.pi * np.arange(n) / n
    result = nominal.copy()
    axis = np.deg2rad(0.35)
    tilt = np.array([[1, 0, 0], [0, np.cos(axis), -np.sin(axis)], [0, np.sin(axis), np.cos(axis)]])
    for i, phase_i in enumerate(phase):
        a, b, c = np.deg2rad(
            [0.12 * np.sin(phase_i), 0.08 * np.cos(2 * phase_i), 0.1 * np.sin(3 * phase_i)]
        )
        rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
        result[i, :3, :3] = tilt @ nominal[i, :3, :3] @ ry @ rx @ rz
    result[:, 0, 3] += pixel * (1.7 + 1.2 * np.sin(phase) + (phase >= 2) - (phase >= 4))
    result[:, 2, 3] += pixel * (0.8 * np.cos(2 * phase) - 0.7 * (phase >= 4))
    return result


def prepare(config: dict, out: Path) -> None:
    """Voxelise physical truth and write six independent-simulator plans on CPU."""
    import tomojax as tj
    from tomojax.geometry import grid_volume_origin, stack_view_poses

    out.mkdir(parents=True, exist_ok=False)
    geometry = config["geometry"]
    grid = tj.Grid(**geometry["grid"])
    detector = tj.Detector.from_dict(geometry["detector"])
    if detector.du != detector.dv:
        raise ValueError("the corpus renderer requires square detector pixels")
    angles = np.asarray(geometry["angles_deg"], np.float64)
    options = {key: geometry[key] for key in ("tilt_deg", "tilt_about") if key in geometry}
    if geometry["kind"] == "cone":
        unsupported = set(geometry["beam"]) - {"source_to_axis", "source_to_detector"}
        if any(geometry["beam"][key] != 0 for key in unsupported):
            raise ValueError("the corpus renderer requires an untilted, centred cone camera")
        geo = tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(**geometry["beam"]), **options)
        beam = {
            "source_to_axis_mm": geo.beam.source_to_axis,
            "source_to_detector_mm": geo.beam.source_to_detector,
        }
    elif geometry["kind"] == "laminography":
        geo = tj.LaminographyGeometry(grid, detector, angles, **options)
        beam = {}
    elif geometry["kind"] == "parallel":
        geo = tj.ParallelGeometry(grid, detector, angles)
        beam = {}
    else:
        raise ValueError(f"unknown geometry kind: {geometry['kind']}")
    nominal = np.asarray(stack_view_poses(geo, len(angles)), np.float64)
    magnification = geometry.get("beam", {}).get("source_to_detector", 1) / geometry.get(
        "beam", {}
    ).get("source_to_axis", 1)
    true = moved_poses(nominal, detector.du / magnification)
    truth = np.zeros((grid.nz, grid.ny, grid.nx), np.float32)
    for material in config["materials"]:
        path = Path(config["meshes"]) / (material["name"] + ".stl")
        truth += material["mu_mm"] * occupancy(
            read_stl(path),
            truth.shape,
            (grid.vx, grid.vy, grid.vz),
            grid_volume_origin(grid),
            supersample=config.get("truth_supersample", 4),
        ).astype(np.float32)
    np.save(out / "truth.npy", truth)
    np.savez(out / "poses.npz", nominal_poses=nominal, poses=true)
    spectral = "energies_keV" in config
    plan_hashes = {}
    for name, mesh_key, poses in (
        ("clean", "interior_meshes", nominal),
        ("clean-motion", "interior_meshes", true),
        ("exterior", "meshes", nominal),
        ("motion", "meshes", true),
        ("exterior-only", "exterior_meshes", nominal),
        ("moving-exterior", "exterior_meshes", true),
    ):
        meshes = Path(config[mesh_key]).resolve()
        materials = [m for m in config["materials"] if (meshes / (m["name"] + ".stl")).exists()]
        plan_hashes[name] = {m["name"]: sha256(meshes / (m["name"] + ".stl")) for m in materials}
        coefficients = np.array([m["mu_spectrum"] if spectral else m["mu_mm"] for m in materials])
        if spectral:
            coefficients = coefficients.reshape(len(materials), len(config["energies_keV"]))
        np.savez(
            out / f"{name}-plan.npz",
            meshes=str(meshes),
            names=[m["name"] for m in materials],
            mu_mm=coefficients,
            shape=(detector.nv, detector.nu),
            pixel_mm=detector.du,
            centre_mm=detector.center,
            supersample=config.get("supersample", 2),
            raster_refinement=config.get("raster_refinement", 1),
            poses=poses,
            **beam,
            **({"photon_fractions": config["photon_fractions"]} if spectral else {}),
        )
    config = dict(
        config,
        generator_sha256=sha256(Path(__file__)),
        plan_mesh_sha256=plan_hashes,
        held_out_indices=np.flatnonzero(np.arange(len(angles)) % 5 == 2).tolist(),
    )
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def pack(work: Path, photons: list[float]) -> None:
    """Package rendered scenarios at frozen count levels; zero-count correction is 0.5."""
    config = json.loads((work / "config.json").read_text())
    geometry = config["geometry"]
    truth = np.load(work / "truth.npy")
    pose_data = np.load(work / "poses.npz")
    nominal, moved = pose_data["nominal_poses"], pose_data["poses"]
    for scenario in ("clean", "clean-motion", "exterior", "motion"):
        with np.load(work / f"{scenario}-render.npz") as data:
            transmission = data["transmission"]
            simulator = str(data["gvxr"])
            renderer_hash = (
                str(data["renderer_sha256"]) if "renderer_sha256" in data else "unrecorded"
            )
        if scenario in ("clean", "clean-motion"):
            exterior = np.zeros_like(transmission)
        else:
            name = "moving-exterior" if scenario == "motion" else "exterior-only"
            with np.load(work / f"{name}-render.npz") as data:
                exterior = -np.log(data["transmission"])
        for count in photons:
            seed = config["seed"]
            counts = np.random.default_rng(seed).poisson(count * transmission.astype(np.float64))
            projections = -np.log(np.maximum(counts, 0.5) / count)
            metadata = dict(
                config,
                scenario=scenario,
                photons_per_pixel=count,
                simulator=simulator,
                renderer_sha256=renderer_hash,
                truth_level="T1-independent-mesh",
                noise_model="Poisson; max(count,0.5) before log",
                gauge="original mesh frame; compare fitted poses only after rigid gauge matching",
                detector_response="photon counting",
                zero_count_fraction=float(np.mean(counts == 0)),
            )
            np.savez_compressed(
                work / f"{scenario}-{count:g}.npz",
                projections=projections.astype(np.float32),
                transmission=transmission,
                truth=truth,
                exterior_projection=exterior,
                poses=moved if scenario.endswith("motion") else nominal,
                nominal_poses=nominal,
                angles_deg=geometry["angles_deg"],
                geometry_json=json.dumps(geometry),
                metadata_json=json.dumps(metadata),
            )


def main() -> None:
    """Run one CPU stage of the corpus pipeline."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="stage", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("config", type=Path)
    command.add_argument("out", type=Path)
    command = commands.add_parser("pack")
    command.add_argument("work", type=Path)
    command.add_argument("--photons", type=float, nargs="+", default=[1e8, 1e5, 1e4, 1e3])
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(json.loads(args.config.read_text()), args.out)
    else:
        pack(args.work, args.photons)


if __name__ == "__main__":
    main()
