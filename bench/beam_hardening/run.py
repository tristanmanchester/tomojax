#!/usr/bin/env python3
"""Score beam hardening methods on one family of scans: one command per family.

    python bench/beam_hardening/run.py simulation --split dev --methods none,oracle
    python bench/beam_hardening/run.py 2detect --split dev --methods none

``simulation`` renders each case with gVXR in its own environment
(``GVXR_PYTHON``), scores every method on it and deletes it; ``--split
heldout`` scores the held-out families (titanium, above 150 kV) and is run
once per frozen method. ``lamino`` does the same for plates in parallel-beam
laminography. ``2detect`` corrects the 60 kV unfiltered scans and
scores them against the 90 kV filtered scans of the same slices; its held-out
split is the out-of-distribution slices. ``walnut`` and ``mono`` are the
do-no-harm checks (see ``donoharm.py``). The JSON report has every case's
metrics and each method's summary; a figure per case, every image on the
reference's window beside radial profiles, goes in ``<output>-figures/``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cases  # noqa: E402
import figures as plots  # noqa: E402
import methods  # noqa: E402
import score  # noqa: E402

GVXR_PYTHON = os.environ.get(
    "GVXR_PYTHON", str(Path("~/Projects/tomojax/.artifacts/gvxr-env/bin/python").expanduser())
)


def _render(split: str, index: int, geometry: str, folder: Path) -> subprocess.Popen:
    output = folder / f"{split}-{index}.npz"
    command = [GVXR_PYTHON, str(HERE / "simulate.py"), "--split", split, "--index", str(index)]
    return subprocess.Popen(
        [*command, "--geometry", geometry, "--output", str(output)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def tomojax_geometry(case: dict) -> object:
    """The TomoJAX geometry the case was rendered with."""
    import tomojax as tj

    g = cases.GEOMETRIES[case["geometry"]]
    voxel = g["voxel"]
    grid = tj.Grid(nx=g["grid"], ny=g["grid"], nz=g["slices"], vx=voxel, vy=voxel, vz=voxel)
    detector = tj.Detector(nu=g["nu"], nv=g["nv"], du=g["pixel"], dv=g["pixel"])
    angles = np.arange(g["views"]) * 360.0 / g["views"]
    if g["beam"] == "cone":
        beam = tj.ConeBeam(g["source_to_axis"], g["source_to_detector"])
        return tj.ConeGeometry(grid=grid, detector=detector, angles=angles, beam=beam)
    return tj.LaminographyGeometry(
        grid=grid, detector=detector, angles=angles, tilt_deg=case["tilt_deg"]
    )


def recorded(spectrum: dict, material: str) -> dict:
    """What a scanner records about its beam, and the material a user names as the sample's."""
    keys = ("kv", "anode", "anode_angle_deg", "window", "filters", "scintillator")
    return {**{k: spectrum[k] for k in keys}, "material": cases.MATERIALS[material][0]}


def score_case(arrays: dict, info: dict, names: list[str], figures: Path | None = None) -> dict:
    """Run every method on one rendered case, score it, and draw it into ``figures``."""
    import tomojax as tj

    case, materials = info["case"], info["materials"]
    frames = tj.Frames(
        counts=arrays["counts"],
        geometry=tomojax_geometry(case),
        flats=arrays["flat"],
        darks=arrays["dark"],
    )
    paths = {m: arrays[f"path_{m}"] for m in materials}
    slope = score.gauge_mu(materials, arrays, ("slope",))
    # The main material carries the most attenuation: area times attenuation.
    mass = dict.fromkeys(materials, 0.0)
    for i, s in enumerate(case["shapes"]):
        if s["material"] != "air":
            holes = sum(c["radius"] ** 2 for c in case["shapes"] if c.get("parent") == i)
            mass[s["material"]] += (s["radius"] ** 2 - holes) * slope[s["material"]]
    main = max(mass, key=mass.__getitem__)
    truth = {
        "line_integrals": sum(slope[m] * paths[m] for m in materials).astype(np.float32),
        "weights": arrays["weights"],
        "main_mu": np.interp(
            arrays["energies"], np.arange(1.0, 301.0), arrays["mu_grid"][materials.index(main)]
        ),
    }
    images: dict[str, np.ndarray] = {}
    # Figures show every image in units of the largest region's material.
    fractions = score.fractions(case)
    level = case["shapes"][int(np.argmax(fractions.sum(axis=(1, 2))))]["material"]
    record = {
        "case": case["name"],
        "family": case["family"],
        "kv": case["spectrum"]["kv"],
        "checks": info["checks"],
    }
    # Laminography never measures part of frequency space, so even exact line
    # integrals reconstruct off: image scores there are read against that floor.
    lamino = case["geometry"] == "lamino"
    for name in [*names, "exact"] if lamino and "exact" not in names else names:
        start = time.perf_counter()
        try:
            result = methods.METHODS[name](frames, recorded(case["spectrum"], main), truth)
        except Exception as error:  # a failing method is a result, not a stop
            record[name] = {"error": f"{type(error).__name__}: {error}"}
            continue
        seconds = time.perf_counter() - start
        mu = score.gauge_mu(materials, arrays, result.gauge)
        metrics = score.score_volume(result.volume, case, mu)
        if result.projections is not None:
            metrics["line_integral"] = score.score_line_integrals(result.projections, paths, mu)
        record[name] = {"seconds": seconds, **metrics}
        images[name] = score.slab(result.volume, case) / mu[level]
    if lamino and "mu" in record.get("exact", {}):
        _against_exact(record)
    if figures is not None:
        shapes = case["shapes"]
        relative = {s["material"]: slope.get(s["material"], 0.0) / slope[level] for s in shapes}
        truth = sum(f * relative[s["material"]] for f, s in zip(fractions, shapes, strict=True))
        g = cases.GEOMETRIES[case["geometry"]]
        outer = max(shapes, key=lambda s: s["radius"])
        centre = tuple(c / g["voxel"] + (g["grid"] - 1) / 2 for c in outer["centre"])
        title = f"{case['name']}: {case['spectrum']['kv']:.0f} kV, in units of {level}"
        figures.mkdir(parents=True, exist_ok=True)
        path = figures / f"{case['name']}.png"
        plots.save(path, truth, images, centre, g["voxel"], title, window=(0.0, 1.5))
    return record


def _against_exact(record: dict) -> None:
    """Score each method's mu bias and cupping relative to the exact control's image.

    The truth-relative values move to ``vs_truth``; line integrals are unchanged.
    """
    floor = {m: v["bias"] for m, v in record["exact"]["mu"].items()}
    cupping = record["exact"].get("cupping")
    for row in record.values():
        if not isinstance(row, dict) or "mu" not in row:
            continue
        row["vs_truth"] = {"mu": {m: v["bias"] for m, v in row["mu"].items()}}
        for m, v in row["mu"].items():
            v["bias"] = (1 + v["bias"]) / (1 + floor[m]) - 1
        if "cupping" in row and cupping is not None:
            row["vs_truth"]["cupping"] = row["cupping"]
            row["cupping"] -= cupping


def simulation(
    split: str, indices: list[int], names: list[str], geometry: str, figures: Path | None = None
) -> list[dict]:
    """Render, score and delete each case; the next case renders while one is scored."""
    records = []
    with tempfile.TemporaryDirectory(dir=os.environ.get("LAB_ARTIFACTS")) as folder:
        pending = _render(split, indices[0], geometry, Path(folder))
        for n, index in enumerate(indices):
            job, path = pending, Path(folder) / f"{split}-{index}.npz"
            job.communicate()
            if n + 1 < len(indices):
                pending = _render(split, indices[n + 1], geometry, Path(folder))
            if job.returncode:
                records.append({"case": f"{split}-{geometry}-{index}", "error": "render failed"})
                continue
            with np.load(path) as archive:
                arrays = {k: archive[k] for k in archive.files}
            path.unlink()
            info = json.loads(str(arrays.pop("info")))
            records.append(score_case(arrays, info, names, figures))
            print(json.dumps(records[-1]), flush=True)
    return records


def _headline(row: dict) -> dict[str, float]:
    """One case's headline magnitudes for one method."""
    found = {
        "mu_bias": max((abs(v["bias"]) for v in row["mu"].values()), default=None),
        "ratio_error": row.get("ratio_error") if len(row["mu"]) > 1 else None,
        "cupping": abs(row["cupping"]) if "cupping" in row else None,
        "dark_band": abs(row["dark_band"]) if "dark_band" in row else None,
        "radius_mm": max((abs(v) for v in row["radius"].values()), default=None),
        "line_integral": row["line_integral"]["all"] if "line_integral" in row else None,
        "line_shape": row["line_integral"].get("shape") if "line_integral" in row else None,
    }
    return {k: v for k, v in found.items() if v is not None}


def summary(records: list[dict], names: list[str]) -> dict:
    """Per family and method, the median and 90th percentile of each headline magnitude.

    ``change`` is the 25th, 50th and 75th percentile over cases of each
    magnitude's relative change against uncorrected; ``beats_none`` is the
    share of cases where the method's mu bias and line-integral bias are both
    below the uncorrected ones.
    """
    out: dict = {}
    for family in sorted({r["family"] for r in records if "family" in r}):
        rows = [r for r in records if r.get("family") == family]
        for name in names:
            values: dict[str, list] = {}
            changes: dict[str, list] = {}
            wins = []
            failed = 0
            for r in rows:
                if "mu" not in r.get(name, {}):
                    failed += name in r
                    continue
                mine = _headline(r[name])
                for k, v in mine.items():
                    values.setdefault(k, []).append(v)
                if "mu" in r.get("none", {}) and name != "none":
                    base = _headline(r["none"])
                    keys = [k for k in ("mu_bias", "line_integral") if k in mine and k in base]
                    wins.append(all(mine[k] < base[k] for k in keys))
                    for k in mine.keys() & base.keys():
                        if base[k] > 0:
                            changes.setdefault(k, []).append(mine[k] / base[k] - 1)
            entry = {
                k: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90)), "n": len(v)}
                for k, v in values.items()
            }
            for k, v in changes.items():  # relative change against uncorrected, per case
                entry[k]["change"] = [float(np.percentile(v, p)) for p in (25, 50, 75)]
            if wins:
                entry["beats_none"] = float(np.mean(wins))
            if failed:
                entry["failed"] = failed
            out.setdefault(family, {})[name] = entry
    return out


def _indices(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        low, _, high = part.partition("-")
        out.extend(range(int(low), int(high or low) + 1))
    return out


def main() -> None:
    """Score the requested methods on one family and write the JSON report."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("family", choices=["simulation", "lamino", "2detect", "walnut", "mono"])
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument(
        "--cases",
        help="case indices (simulation, default 0-23) or slice numbers (2detect, default all)",
    )
    parser.add_argument("--methods", default="none,oracle,exact")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("LAB_ARTIFACTS", ".")) / "beam_hardening.json",
    )
    args = parser.parse_args()
    names = args.methods.split(",")
    figures = args.output.with_name(f"{args.output.stem}-figures")
    if args.family in ("simulation", "lamino"):
        geometry = "cone" if args.family == "simulation" else "lamino"
        default = "0-23" if geometry == "cone" else "0-11"
        indices = _indices(args.cases or default)
        records = simulation(args.split, indices, names, geometry, figures)
        totals = summary(records, names)
    elif args.family == "2detect":
        import detect2_score

        chosen = _indices(args.cases) if args.cases else None
        records = detect2_score.run(args.split, names, chosen, figures)
        totals = detect2_score.summary(records, names)
    else:
        import donoharm

        records = getattr(donoharm, args.family)(names, figures)
        totals = {name: [r.get(name) for r in records] for name in names}
    report = {
        "family": args.family,
        "split": args.split,
        "methods": names,
        "summary": totals,
        "cases": records,
    }
    args.output.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
