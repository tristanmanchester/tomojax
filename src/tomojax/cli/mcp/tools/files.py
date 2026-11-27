"""What the inspect tool reads from disk itself: directory listings, HDF5 trees, previews."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp_types import ImageContent

if TYPE_CHECKING:
    from tomojax.cli.mcp.supervisor import Object


def preview_image(path: Path) -> ImageContent:
    """The central projection, or the central z slice of a volume saved without its scan."""
    from io import BytesIO

    from PIL import Image

    from tomojax.cli import _images, _previews

    if path.suffix.lower() == ".npz":
        from tomojax.io.projection_layout import saved_npz_layout

        _, arrays = saved_npz_layout(path)
        stack = arrays["projections" if "projections" in arrays else "volume"]
        pixels = stack[len(stack) // 2]  # the central view, or the central z plane
    else:
        pixels = _previews.projection_image(path)
        if pixels is None:
            (plane,) = _previews.saved_requests(path, None)[:1]
            pixels = _previews.saved_plane(path, *plane)
    image = Image.fromarray(_images.to_uint8(pixels, _images.limits(pixels)))
    image.thumbnail((1024, 1024))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return ImageContent(
        type="image", mime_type="image/png", data=base64.b64encode(buffer.getvalue()).decode()
    )


def tree(path: Path) -> Object:
    """List HDF5 dataset names, shapes and dtypes without loading their values."""
    import h5py

    datasets = {}
    if path.is_file() and h5py.is_hdf5(path):
        with h5py.File(path, "r") as file:

            def visit(name: str, item: Any) -> None:
                if isinstance(item, h5py.Dataset):
                    datasets["/" + name] = {"shape": list(item.shape), "dtype": str(item.dtype)}

            file.visititems(visit)
    return datasets


def inspect_directory(path: Path) -> Object:
    """Summarize acquisition entry points and count TIFF frames in one directory pass."""
    formats = {
        ".xtekct": "Nikon",
        ".txrm": "Zeiss",
        ".txm": "Zeiss",
        ".log": "acquisition log",
        ".info": "acquisition metadata",
        ".xml": "acquisition metadata",
        ".npz": "NumPy archive",
        ".mat": "PSI MATLAB acquisition",
        **dict.fromkeys((".nxs", ".nx", ".hdf", ".hdf5", ".h5"), "HDF5"),
    }
    found: dict[str, str] = {}
    subdirectories: list[str] = []
    checkpoints: list[str] = []  # stopped runs' saved state: inspect one to resume it
    directory_count = frames = candidates = 0
    with os.scandir(path) as entries:
        for entry in entries:
            suffix = Path(entry.name).suffix.lower()
            if entry.is_dir():
                directory_count += 1
                if len(subdirectories) < 64:
                    subdirectories.append(entry.name)
                if Path(entry.path, "checkpoint.json").is_file():
                    checkpoints.append(entry.path)
            elif suffix in {".tiff", ".tif"} and entry.is_file():
                frames += 1
            elif suffix in formats and entry.is_file():
                candidates += 1
                if len(found) < 64:
                    found[entry.path] = formats[suffix]
    return {
        "directory": str(path),
        "datasets": dict(sorted(found.items())),
        "candidate_count": candidates,
        "subdirectories": sorted(subdirectories),
        "subdirectory_count": directory_count,
        "tiff_files": frames,
        "checkpoints": sorted(checkpoints),
        "omitted": candidates - len(found),
    }
