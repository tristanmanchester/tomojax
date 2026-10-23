"""Reconstruction error of a beam hardening method against simulated truth.

Truth is per-material path length for every ray and an analytic phantom of
z-aligned cylinders. A method declares its gauge, ``("slope",)`` for units of
the unhardened beam's mean attenuation (the curve's slope at zero) or
``("energy", keV)`` for attenuation at a reference energy, and is scored in
that gauge:

- ``mu``: per material, the relative bias and RMS error of the attenuation
  inside the material, 3 voxels clear of every edge (or the inner half of a
  thinner wall); the RMS includes noise and the reconstruction's own blur of
  strong edges, which ``exact`` shows;
- ``ratio_error``: the RMS over materials of the relative error of their mean
  attenuation after the one scale factor that minimises it, so independent
  of the gauge: whether the materials keep their true ratios (0 for one);
- ``cupping``: in the largest region, the mean attenuation near its rim minus
  that near its centre, relative to the region's mean (0 when flat);
- ``dark_band``: between every pair of dense rods in a matrix, the matrix
  attenuation on the line joining them minus that at the same radius
  elsewhere, relative to the region's mean (0 without streaks; the worst pair);
- ``radius``: per shape, the edge position from the local mid-level between
  the attenuation just inside and just outside, minus the true radius, in mm;
- ``line_integral``: for methods that return corrected projections, their
  systematic relative error over rays through the object, overall, per
  material for rays through one material only, and after the best scale.

Flatness and data residual are never scores: they are what correction
methods optimise.
"""

from __future__ import annotations

from cases import GEOMETRIES
import numpy as np

MARGIN = 0.3  # mm clear of edges for interior statistics


def coordinates(case: dict) -> tuple[np.ndarray, np.ndarray]:
    """Voxel-centre x and y (mm) of the case's grid, indexed ``[iy, ix]``."""
    g = GEOMETRIES[case.get("geometry", "cone")]
    axis = (np.arange(g["grid"]) - (g["grid"] - 1) / 2) * g["voxel"]
    return np.meshgrid(axis, axis, indexing="xy")


def _owned(case: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Index of the innermost shape containing each point, or -1."""
    owner = np.full(x.shape, -1)
    for i, shape in enumerate(case["shapes"]):  # parents precede their children
        inside = np.hypot(x - shape["centre"][0], y - shape["centre"][1]) <= shape["radius"]
        owner[inside] = i
    return owner


def fractions(case: dict, supersample: int = 5) -> np.ndarray:
    """Area fraction of every shape's own region in each voxel, ``(shapes, ny, nx)``."""
    x, y = coordinates(case)
    voxel = GEOMETRIES[case.get("geometry", "cone")]["voxel"]
    offsets = (np.arange(supersample) - (supersample - 1) / 2) * voxel / supersample
    result = np.zeros((len(case["shapes"]), *x.shape))
    for dx in offsets:
        for dy in offsets:
            owner = _owned(case, x + dx, y + dy)
            for i in range(len(case["shapes"])):
                result[i] += owner == i
    return result / supersample**2


def gauge_mu(materials: list[str], arrays: dict, gauge: tuple) -> dict[str, float]:
    """True attenuation (mm^-1) of each material in a method's gauge.

    ``("slope",)``: the unhardened beam's mean attenuation; ``("energy", keV)``:
    attenuation at that energy.
    """
    grid = np.arange(1.0, 301.0)
    detected = {
        m: np.interp(arrays["energies"], grid, arrays["mu_grid"][k])
        for k, m in enumerate(materials)
    }
    if gauge[0] == "slope":
        return {m: float(mu @ arrays["weights"]) for m, mu in detected.items()}
    if gauge[0] == "energy":
        return {
            m: float(np.interp(gauge[1], grid, arrays["mu_grid"][k]))
            for k, m in enumerate(materials)
        }
    raise ValueError(f"unknown gauge {gauge!r}")


def _distance_to_edges(case: dict, i: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance from each point to the boundary of shape ``i``'s own region."""
    shape = case["shapes"][i]
    d = np.abs(np.hypot(x - shape["centre"][0], y - shape["centre"][1]) - shape["radius"])
    for child in case["shapes"]:
        if child.get("parent") == i:
            d = np.minimum(
                d,
                np.abs(np.hypot(x - child["centre"][0], y - child["centre"][1]) - child["radius"]),
            )
    return d


def _interior(case: dict, i: int, owner: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Voxels of shape ``i``'s own region clear of its edges; the inner half of thin walls."""
    distance = np.where(owner == i, _distance_to_edges(case, i, x, y), -1.0)
    return distance >= min(MARGIN, 0.5 * distance.max())


def _radius(image: np.ndarray, case: dict, i: int, x: np.ndarray, y: np.ndarray) -> float | None:
    """Median over clean angular sectors of the local mid-level edge radius, minus truth."""
    shape, voxel = case["shapes"][i], GEOMETRIES[case.get("geometry", "cone")]["voxel"]
    rho = np.hypot(x - shape["centre"][0], y - shape["centre"][1])
    phi = np.arctan2(y - shape["centre"][1], x - shape["centre"][0])
    near = np.abs(rho - shape["radius"]) <= 6 * voxel
    others = [s for k, s in enumerate(case["shapes"]) if k != i]
    clash = np.zeros(x.shape, bool)
    for s in others:
        clash |= np.abs(np.hypot(x - s["centre"][0], y - s["centre"][1]) - s["radius"]) <= 7 * voxel
    sectors = int(np.clip(2 * np.pi * shape["radius"] / 1.0, 4, 24))
    which = np.floor((phi + np.pi) / (2 * np.pi) * sectors).astype(int) % sectors
    found = []
    for k in range(sectors):
        sel = near & (which == k)
        if not sel.any() or (clash & sel).any():
            continue
        r, values = rho[sel] - shape["radius"], image[sel]
        inner = values[(r >= -6 * voxel) & (r <= -3 * voxel)].mean()
        outer = values[(r >= 3 * voxel) & (r <= 6 * voxel)].mean()
        edges = np.arange(-3, 3.01, 0.5) * voxel
        counts = np.histogram(r, edges)[0]
        if counts.min() == 0:
            continue
        centres = 0.5 * (edges[1:] + edges[:-1])
        profile = np.histogram(r, edges, weights=values)[0] / counts
        side = np.sign(profile - 0.5 * (inner + outer)) * np.sign(inner - outer)
        crossing = np.nonzero((side[:-1] > 0) & (side[1:] <= 0))[0]
        if crossing.size:
            j = crossing[0]
            level = 0.5 * (inner + outer)
            t = (profile[j] - level) / (profile[j] - profile[j + 1])
            found.append(centres[j] + t * (centres[j + 1] - centres[j]))
    return float(np.median(found)) if found else None


def slab(volume: np.ndarray, case: dict) -> np.ndarray:
    """Mean of the scored slices: the central half, or a plate 0.5 mm clear of its faces."""
    g, nz = GEOMETRIES[case.get("geometry", "cone")], volume.shape[0]
    z = (np.arange(nz) - (nz - 1) / 2) * g["voxel"]
    half = min(z.max() / 2, case["shapes"][0].get("half_height", np.inf) - 0.5)
    return np.asarray(volume[np.abs(z) <= max(half, g["voxel"])], np.float64).mean(axis=0)


def score_volume(volume: np.ndarray, case: dict, truth_mu: dict[str, float]) -> dict:
    """Image metrics of a ``(nz, ny, nx)`` volume against a case in one gauge."""
    image = slab(volume, case)
    x, y = coordinates(case)
    owner = _owned(case, x, y)
    shapes = case["shapes"]
    result: dict = {"mu": {}, "radius": {}}
    for material, true in truth_mu.items():
        mask = np.zeros(x.shape, bool)
        for i, s in enumerate(shapes):
            if s["material"] == material:
                mask |= _interior(case, i, owner, x, y)
        values = image[mask] / true
        result["mu"][material] = {
            "bias": float(values.mean() - 1),
            "rms": float(np.sqrt(np.mean((values - 1) ** 2))),
        }
    ratios = np.array([1 + v["bias"] for v in result["mu"].values()])
    best = ratios.sum() / (ratios @ ratios)
    result["ratio_error"] = float(np.sqrt(np.mean((best * ratios - 1) ** 2)))
    big = int(np.argmax([np.sum(owner == i) for i in range(len(shapes))]))
    if shapes[big]["material"] in truth_mu:
        result |= _matrix(image, case, big, owner)
    for i, shape in enumerate(shapes):
        error = _radius(image, case, i, x, y)
        if error is not None:
            result["radius"][f"{i}:{shape['material']}"] = error
    return result


def _matrix(image: np.ndarray, case: dict, big: int, owner: np.ndarray) -> dict:
    """Cupping and dark bands of the largest region, clear of the rods inside it.

    Both are relative to the region's own mean attenuation, so independent of the gauge.
    """
    x, y = coordinates(case)
    matrix = case["shapes"][big]
    rho = np.hypot(x - matrix["centre"][0], y - matrix["centre"][1])
    region = _interior(case, big, owner, x, y)
    rods = [s for s in case["shapes"] if s.get("parent") == big]
    for rod in rods:
        region &= np.hypot(x - rod["centre"][0], y - rod["centre"][1]) > 1.5 * rod["radius"] + 0.5
    result = {}
    own = image[region].mean()
    radius = matrix["radius"]
    centre, rim = (
        region & (rho < 0.3 * radius),
        region & (rho > 0.7 * radius) & (rho < 0.9 * radius),
    )
    if centre.sum() > 20 and rim.sum() > 20:
        result["cupping"] = float((image[rim].mean() - image[centre].mean()) / own)
    dense = [r for r in rods if r["material"] != "air"]
    reference = region.copy()
    for r in dense:
        reference &= np.hypot(x - r["centre"][0], y - r["centre"][1]) > 2 * r["radius"] + 0.5
    bins = np.floor(rho / 0.5).astype(int)
    bands = []
    for a, rod in enumerate(dense):
        for other in dense[a + 1 :]:
            band = _segment_band(x, y, rod, other) & region
            # The matrix at the same distance from its centre, away from rods and this band.
            elsewhere = reference & ~band
            level = np.bincount(bins[elsewhere], image[elsewhere], bins.max() + 1)
            count = np.bincount(bins[elsewhere], minlength=bins.max() + 1)
            ok = count[bins[band]] >= 10
            if ok.sum() >= 10:
                expected = level[bins[band]][ok] / count[bins[band]][ok]
                bands.append(float((image[band][ok] - expected).mean() / own))
    if bands:
        result["dark_band"] = min(bands)
    return result


def _segment_band(x: np.ndarray, y: np.ndarray, a: dict, b: dict) -> np.ndarray:
    """Points within half the smaller rod radius of the segment joining two rods, between them."""
    p, q = np.asarray(a["centre"]), np.asarray(b["centre"])
    d = q - p
    t = ((x - p[0]) * d[0] + (y - p[1]) * d[1]) / (d @ d)
    distance = np.hypot(x - p[0] - t * d[0], y - p[1] - t * d[1])
    clear = (np.hypot(x - p[0], y - p[1]) > a["radius"] + 0.3) & (
        np.hypot(x - q[0], y - q[1]) > b["radius"] + 0.3
    )
    return (t > 0) & (t < 1) & (distance < 0.5 * min(a["radius"], b["radius"])) & clear


def score_line_integrals(
    projections: np.ndarray, paths: dict[str, np.ndarray], truth_mu: dict[str, float]
) -> dict:
    """Systematic relative error of corrected line integrals, overall and per lone material.

    Rays are binned by true line integral and the error averaged in each bin,
    so noise cancels and the bias of the curve remains; ``shape`` is the same
    after the one scale factor that minimises it (independent of the gauge),
    and ``rms`` keeps the noise.
    """
    truth = sum(truth_mu[m] * path.astype(np.float64) for m, path in paths.items())
    inside = truth > 0.05 * truth.max()
    error = projections - truth
    p = projections[inside].astype(np.float64)
    scaled = projections * ((p @ truth[inside]) / (p @ p)) - truth

    def systematic(error: np.ndarray, mask: np.ndarray) -> float:
        bins = np.minimum((truth[mask] / truth[mask].max() * 20).astype(int), 19)
        count = np.bincount(bins, minlength=20)
        mean_error = np.bincount(bins, error[mask], 20) / np.maximum(count, 1)
        mean_truth = np.bincount(bins, truth[mask], 20) / np.maximum(count, 1)
        return float(np.sqrt((count * mean_error**2).sum() / (count * mean_truth**2).sum()))

    result = {
        "all": systematic(error, inside),
        "shape": systematic(scaled, inside),
        "rms": float(np.sqrt(np.mean(error[inside] ** 2) / np.mean(truth[inside] ** 2))),
    }
    for m, path in paths.items():
        others = sum((q for o, q in paths.items() if o != m), np.zeros_like(truth))
        alone = inside & (path > 0) & (others < 0.01)
        if alone.sum() > 100:
            result[m] = systematic(error, alone)
    return result
