"""Volume-render TomoJAX's reconstruction of a watch movement.

The 29 mm movement of ``_watch.py`` is scanned whole in a 120 keV parallel
beam, 600 views over a half turn with 3 x 10^4 photons per pixel, and
reconstructed on a 768 x 768 x 164 grid of 38.5 µm voxels with FISTA-TV. The
reconstruction is drawn by direct volume rendering (``_volume_render.py``): a
transfer function of the reconstructed attenuation draws steel white, brass
and the balance gold, the timing screws yellow and the ruby jewels red, and
titanium as blue glass, by its boundaries, so the train shows through the
plate and bridges. The meshes are used only to make the projections; what is
drawn is the reconstructed volume. Writes ``images/watch-render.png`` and a
JSON record.

    uv run --no-sync python examples/render_watch.py
"""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from _figure import MUTED, Card
import _volume_render
import _watch
import tomojax as tj
from tomojax.geometry import stack_view_poses

SIZE, LAYERS, FIELD, MIDDLE = 768, 164, 29.6, 3.0
VIEWS, PHOTONS = 600, 3e4
RECONSTRUCTION = {"iterations": 60, "tv_weight": 0.01, "nonnegative": True, "warm_start": True}
TRANSFER = _volume_render.Transfer(  # attenuation in mm^-1 at 120 keV
    stops=(
        0.0,
        0.028,
        0.040,
        0.052,
        0.062,
        0.072,
        0.085,
        0.115,
        0.14,
        0.17,
        0.23,
        0.27,
        0.45,
        1.2,
        4.5,
    ),
    colours=(
        (0.0, 0.0, 0.0),
        (0.25, 0.34, 0.50),
        (0.32, 0.44, 0.66),  # silicon escape wheel and pallet lever
        (0.40, 0.30, 0.45),
        (0.95, 0.14, 0.20),  # ruby jewels
        (0.72, 0.84, 1.00),
        (0.76, 0.87, 1.00),  # titanium plate and bridges, as blue glass
        (0.80, 0.89, 1.00),
        (0.80, 0.84, 0.93),
        (0.88, 0.90, 0.96),  # steel pinions, arbors, screws and springs
        (0.93, 0.92, 0.90),
        (1.00, 0.74, 0.38),  # brass wheels and barrel, and the Glucydur balance
        (1.00, 0.76, 0.40),
        (1.00, 0.84, 0.36),  # gold timing screws
        (1.00, 0.95, 0.60),
    ),
    opacity=(0.0, 0.0, 0.5, 0.45, 1.2, 0.6, 0.75, 0.75, 1.2, 5.0, 9.0, 12.0, 14.0, 30.0, 40.0),
)
CLEAR = 0.15  # below this the materials are drawn by their boundaries
CAMERA = _volume_render.Camera(
    target=(0.0, -0.4, 3.3),
    distance=7.9 * FIELD / 2,
    azimuth_deg=-112,
    elevation_deg=48,
    fov_deg=24,
)


def run(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, object]]:
    meshes = _watch.meshes(args.blender)
    voxel = FIELD / SIZE
    grid = tj.Grid(SIZE, SIZE, LAYERS, voxel, voxel, voxel, vol_center=(0.0, 0.0, MIDDLE))
    detector = tj.Detector(SIZE, LAYERS, voxel, voxel, center=(0.0, MIDDLE))
    geometry = tj.ParallelGeometry(grid, detector, np.linspace(0, 180, VIEWS, endpoint=False))
    light = _watch.transmission(
        meshes,
        np.asarray(stack_view_poses(geometry, VIEWS)),
        (LAYERS, SIZE),
        voxel,
        centre=(0.0, MIDDLE),
        gvxr_python=args.gvxr_python,
    )
    scan = tj.Scan(_watch.measure(light, PHOTONS, seed=4), geometry)
    del light
    start = time.perf_counter()
    volume = np.asarray(tj.reconstruct(scan, "fista", **RECONSTRUCTION).volume, np.float32)
    seconds = time.perf_counter() - start
    np.save(_watch.CACHE / "render-volume.npy", volume)
    low = np.array([-FIELD / 2, -FIELD / 2, MIDDLE - LAYERS * voxel / 2])
    start = time.perf_counter()
    rgb = _volume_render.render(
        volume,
        voxel,
        low,
        CAMERA,
        TRANSFER,
        width=args.width,
        height=args.height,
        light=np.array([-0.5, -0.6, 1.0]),
        background=(0.063, 0.075, 0.094),
        edge=1.5,
        clear=CLEAR,
    )
    rgb = (rgb * 255).astype(np.uint8)
    render_seconds = time.perf_counter() - start
    metrics = {
        "object": "29 mm watch movement, meshes from bench/phantoms/watch_blender.py",
        "beam": f"parallel, monochromatic {_watch.ENERGY_KEV:g} keV",
        "views": VIEWS,
        "photons_per_pixel": PHOTONS,
        "voxel_mm": voxel,
        "grid": [SIZE, SIZE, LAYERS],
        "reconstruction": {
            "method": "fista",
            **RECONSTRUCTION,
            "warm_start": "fbp",
            "seconds": seconds,
        },
        "transfer_function": {
            "stops_mm": TRANSFER.stops,
            "colours": TRANSFER.colours,
            "opacity_per_mm": TRANSFER.opacity,
        },
        "camera": {
            "azimuth_deg": CAMERA.azimuth_deg,
            "elevation_deg": CAMERA.elevation_deg,
            "fov_deg": CAMERA.fov_deg,
        },
        "render_seconds": render_seconds,
        "renderer": "direct volume rendering, examples/_volume_render.py",
        "device": str(__import__("jax").devices()[0]),
        "versions": {name: version(name) for name in ("tomojax", "jax", "numpy", "matplotlib")},
    }
    return rgb, metrics


def plot(rgb: np.ndarray, metrics: dict[str, object], path: Path) -> None:
    """The render on the dark card, titled over its empty top left corner."""
    margin = 30
    h, w = rgb.shape[:2]
    card = Card(w + 2 * margin, h + 2 * margin)
    card.picture(margin, margin, rgb)
    recon = metrics["reconstruction"]
    card.text(
        margin + 8, margin + 36, "A watch movement, reconstructed by TomoJAX", 32, weight="bold"
    )
    card.text(
        margin + 8,
        margin + 74,
        f"{metrics['views']} simulated 120 keV views  ·  FISTA-TV, {recon['iterations']} "
        f"iterations in {recon['seconds']:.0f} s  ·  {metrics['voxel_mm'] * 1000:.1f} µm "
        "voxels  ·  direct volume rendering",
        21,
        MUTED,
    )
    card.save(path, colours=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=_watch.ROOT / "images" / "watch-render.png")
    parser.add_argument("--blender", type=Path, default=_watch.BLENDER)
    parser.add_argument("--gvxr-python", type=Path, default=_watch.GVXR_PYTHON)
    parser.add_argument("--width", type=int, default=1740)
    parser.add_argument("--height", type=int, default=980)
    args = parser.parse_args()
    start = time.perf_counter()
    rgb, metrics = run(args)
    plot(rgb, metrics, args.out)
    metrics["total_seconds"] = time.perf_counter() - start
    args.out.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
