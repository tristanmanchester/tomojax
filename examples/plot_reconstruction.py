"""Generate the README's synthetic reconstruction figure and its provenance."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from simulate_and_reconstruct import reconstruct_example


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("images/reconstruction-example.png"))
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--views", type=int, default=90)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--backend", choices=("jax", "pallas"), default="jax")
    args = parser.parse_args()
    truth, volume, metrics = reconstruct_example(
        size=args.size, views=args.views, iterations=args.iterations, backend=args.backend
    )

    mid = args.size // 2
    slices = [
        (truth[:, :, mid].T, volume[:, :, mid].T, "x", "y"),
        (truth[:, mid, :].T, volume[:, mid, :].T, "x", "z"),
    ]
    vmin, vmax = float(min(truth.min(), volume.min())), float(max(truth.max(), volume.max()))
    error_max = max(float(np.abs(reference - recon).max()) for reference, recon, _, _ in slices)
    extent = (-args.size / 2, args.size / 2, -args.size / 2, args.size / 2)
    with plt.rc_context({"font.size": 10, "axes.titlesize": 12, "figure.facecolor": "white"}):
        fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.8), layout="constrained")
        for row, (reference, recon, horizontal, vertical) in enumerate(slices):
            for col, data in enumerate((reference, recon, np.abs(reference - recon))):
                ax = axes[row, col]
                artist = ax.imshow(
                    data,
                    origin="lower",
                    extent=extent,
                    cmap="magma" if col == 2 else "gray",
                    vmin=0.0 if col == 2 else vmin,
                    vmax=error_max if col == 2 else vmax,
                    interpolation="nearest",
                )
                ax.set_xlabel(f"{horizontal} (unit voxel pitch)")
                ax.set_ylabel(f"{vertical} (unit voxel pitch)")
                ax.set_title(("Reference", "CGLS reconstruction", "Absolute error")[col])
                fig.colorbar(artist, ax=ax, shrink=0.78, label="Attenuation (1 / length unit)")
        fig.suptitle(
            f"Parallel tomography · {args.size}³ voxels · {args.views} views\n"
            f"Matched Joseph model · {args.iterations} CGLS iterations · "
            f"full-volume relative L2 = {float(metrics['volume_relative_l2']):.3%}",
            fontsize=13,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=150, metadata={"Software": "TomoJAX public API example"})
        plt.close(fig)

    sources = (Path(__file__), Path(__file__).with_name("simulate_and_reconstruct.py"))
    provenance = {
        "description": "Matched-model usage example; not independent accuracy evidence.",
        "metrics": metrics,
        "geometry": {
            "type": "parallel",
            "angle_range_deg": [0, 180],
            "endpoint": False,
            "voxel_pitch": [1.0, 1.0, 1.0],
            "detector_pitch": [1.0, 1.0],
        },
        "display": {
            "slice_index": mid,
            "planes": ["xy", "xz"],
            "attenuation_range": [vmin, vmax],
            "absolute_error_range": [0.0, error_max],
        },
        "versions": {
            name: version(name) for name in ("tomojax", "jax", "jaxlib", "numpy", "matplotlib")
        },
        "sources_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "image_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
    }
    args.out.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
