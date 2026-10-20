"""Score beam hardening methods on 2DeteCT: 60 kV unfiltered against 90 kV filtered.

Each slice's mode 2 scan (90 kV, Thoraeus filter, high dose) reconstructed
by FBP is the reference; a method corrects and reconstructs the mode 3 scan
(60 kV, no filter) of the same slice, on the same grid. The published
segmentation gives the sample tube's wall, the filler and the objects; the
objects split at 0.028 mm^-1 in the 90 kV image into light ones (nuts, fruit)
and dense ones (lava stone; titanium screws out of distribution). Every
class is eroded by 3 pixels. Attenuation at 60 and 90 kV differs by a
material-dependent factor, so the scores compare the ratio ``r`` of the two
images (each Gaussian-smoothed by 3 pixels):

- ``ratio``: per class, the 60 kV mean over the 90 kV mean. The tube wall is
  the same material in every slice, so a correct method gives every slice
  the same wall ratio whatever the tube holds; the summary's ``*_ratio_cv``
  is the spread across slices;
- ``light_vs_filler``: the light objects' ratio over the filler's, minus 1.
  Both are organic, with nearly the same ratio; self-hardening of the
  denser objects lowers theirs;
- ``low_z_spread``: the RMS of ``r`` over the filler and the light objects
  (skipping pores below half the filler's attenuation), relative to its
  median and clipped at 3: cupping, streaks and self-hardening together;
- ``filler_cupping``: ``r`` of the filler near the wall minus that near the
  tube's centre, relative to the filler's (``reference_cupping`` is the same
  profile of the 90 kV image alone, its own hardening);
- ``wall_sectors``: the spread of the wall ratio over 36 angular sectors,
  relative to its mean, which streaks from dense objects raise.

Single-material corrections cannot be right for the dense objects, whose
``ratio`` is reported but not summarised.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import detect2 as d2
import figures as plots
import methods
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

METADATA = {
    3: {
        "kv": 60.0,
        "anode": "W",
        "window": ["Be", 0.3],
        "filters": [],
        "scintillator": ["CsI", None],
        "material": "H2O",
    },
    2: {
        "kv": 90.0,
        "anode": "W",
        "window": ["Be", 0.3],
        "filters": [["Sn", 0.1], ["Cu", 0.2], ["Al", 0.5]],
        "scintillator": ["CsI", None],
        "material": "H2O",
    },
}
LIGHT_LIMIT = 0.028  # mm^-1 at 90 kV, in the gap between food and lava stone


def _frames(slice_id: int, mode: int) -> object:
    import tomojax as tj

    f = d2.frames(slice_id, mode)
    return tj.Frames(
        counts=f["counts"], geometry=d2.geometry(slice_id), flats=f["flat"], darks=f["dark"]
    )


def compare(image: np.ndarray, reference: np.ndarray, segmentation: np.ndarray) -> dict:
    """Metrics of one 60 kV image against the 90 kV image of the same slice."""
    from scipy.ndimage import binary_erosion, gaussian_filter

    objects = segmentation == d2.OBJECTS
    classes = {
        "wall": segmentation == d2.WALL,
        "filler": segmentation == d2.FILLER,
        "light": objects & (reference <= LIGHT_LIMIT),
        "dense": objects & (reference > LIGHT_LIMIT),
    }
    masks = {k: binary_erosion(v, iterations=3) for k, v in classes.items()}
    smooth_reference = gaussian_filter(reference, 3.0)
    ratio = gaussian_filter(image, 3.0) / smooth_reference
    result: dict = {"ratio": {}}
    for name, mask in masks.items():
        if mask.sum() > 100:
            result["ratio"][name] = float(image[mask].mean() / reference[mask].mean())
    filler, light = result["ratio"]["filler"], result["ratio"].get("light")
    if light is not None:
        result["light_vs_filler"] = light / filler - 1
    # Pores in the filler leave too little attenuation for a ratio; skip them.
    solid = smooth_reference > 0.5 * np.median(smooth_reference[masks["filler"]])
    low_z = ratio[(masks["filler"] | masks["light"]) & solid]
    low_z = np.clip(low_z / np.median(low_z), 0.0, 3.0)
    result["low_z_spread"] = float(np.sqrt(np.mean((low_z - 1) ** 2)))
    x, y = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing="ij")
    wall = segmentation == d2.WALL
    cx, cy = x[wall].mean(), y[wall].mean()
    rho, phi = np.hypot(x - cx, y - cy), np.arctan2(y - cy, x - cx)
    inner = np.percentile(rho[wall], 1)
    fill = masks["filler"]
    centre, rim = fill & (rho < 0.3 * inner), fill & (rho > 0.7 * inner) & (rho < 0.9 * inner)
    result["filler_cupping"] = float((ratio[rim].mean() - ratio[centre].mean()) / filler)
    result["reference_cupping"] = float(
        (reference[rim].mean() - reference[centre].mean()) / reference[fill].mean()
    )
    sector = np.floor((phi + np.pi) / (2 * np.pi) * 36).astype(int) % 36
    w = masks["wall"]
    per_sector = np.bincount(sector[w], image[w], 36) / np.bincount(sector[w], reference[w], 36)
    result["wall_sectors"] = float(per_sector.std() / abs(per_sector.mean()))
    return result


def run(
    split: str, names: list[str], chosen: list[int] | None = None, figures: Path | None = None
) -> list[dict]:
    """Score every method on every slice of the split, or on the ``chosen`` ones.

    Each slice's figure shows the images divided by their filler's median,
    so the 60 and 90 kV scans share a window, with profiles about the tube's centre.
    """
    records = []
    for slice_id in chosen or d2.slices(split):
        reference = methods.uncorrected(_frames(slice_id, 2), METADATA[2]).volume[:, :, 0]
        segmentation = d2.segmentation(slice_id)
        frames = _frames(slice_id, 3)
        record: dict = {"case": f"slice{slice_id:05d}"}
        filler = segmentation == d2.FILLER
        images = {}
        for name in names:
            start = time.perf_counter()
            try:
                result = methods.METHODS[name](frames, METADATA[3], None)
            except Exception as error:  # truth-only controls, or a failing method
                record[name] = {"error": f"{type(error).__name__}: {error}"}
                continue
            image = np.asarray(result.volume)[:, :, 0]
            record[name] = {
                "seconds": time.perf_counter() - start,
                "declined": result.declined,
                **compare(image, reference, segmentation),
            }
            images[name] = image / np.median(image[filler])
        records.append(record)
        print(record, flush=True)
        if figures is not None:
            figures.mkdir(parents=True, exist_ok=True)
            x, y = np.nonzero(segmentation == d2.WALL)
            title = f"2DeteCT slice {slice_id}: 60 kV methods against 90 kV filtered"
            plots.save(
                figures / f"{record['case']}.png",
                reference / np.median(reference[filler]),
                images,
                (x.mean(), y.mean()),
                d2.VOXEL,
                title,
                window=(0.0, 6.0),
            )
    return records


def summary(records: list[dict], names: list[str]) -> dict:
    """Per method: spread of the class ratios across slices, medians of the rest.

    ``*_change`` is the 25th, 50th and 75th percentile over slices of the
    magnitude's relative change against uncorrected.
    """
    keys = ("light_vs_filler", "low_z_spread", "filler_cupping", "reference_cupping")
    out = {}
    for name in names:
        rows = [r for r in records if "ratio" in r.get(name, {})]
        if not rows:
            continue
        entry = {}
        for c in ("wall", "filler", "light"):
            ratios = np.array([r[name]["ratio"][c] for r in rows if c in r[name]["ratio"]])
            entry[f"{c}_ratio_cv"] = float(ratios.std() / abs(ratios.mean()))
        for key in (*keys, "wall_sectors"):
            entry[key] = float(np.median([abs(r[name][key]) for r in rows if key in r[name]]))
            change = [
                abs(r[name][key]) / abs(r["none"][key]) - 1
                for r in rows
                if name != "none" and key in r[name] and r.get("none", {}).get(key)
            ]
            if change and key != "reference_cupping":
                entry[f"{key}_change"] = [float(np.percentile(change, p)) for p in (25, 50, 75)]
        out[name] = {**entry, "n": len(rows)}
    return out
