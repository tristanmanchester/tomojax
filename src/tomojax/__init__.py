"""TomoJAX: tomography, laminography and lab cone-beam CT reconstruction and alignment.

The workflow API takes a :class:`Scan` (projections and their geometry) and
returns results that carry their geometry::

    import tomojax as tj

    scan = tj.load("scan.nxs")  # or a Nikon .xtekct
    recon = tj.reconstruct(scan)  # FBP (FDK for cone beams)
    recon = tj.reconstruct(scan, method="cgls", iterations=50)
    result = tj.align(scan, mode="cor-then-pose")  # result.scan carries the corrections
    tj.save("recon.nxs", tj.reconstruct(result.scan))

Build a scan from arrays with ``tj.Scan(projections, geometry)`` and the
geometry classes below; ``tj.project`` simulates projections for any geometry.
The subpackages (``tomojax.recon``, ``tomojax.alignment``, ``tomojax.geometry``,
``tomojax.io``) hold the building blocks these use. Names load on first use,
so ``import tomojax`` does not import JAX.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tomojax._workflow import (
        Alignment,
        Reconstruction,
        Scan,
        align,
        backproject,
        load,
        load_reconstruction,
        project,
        reconstruct,
        save,
    )
    from tomojax.geometry import (
        ConeBeam,
        ConeGeometry,
        Detector,
        Grid,
        LaminographyGeometry,
        ParallelGeometry,
    )

__version__ = "0.3.0"

_SOURCES = {
    "Alignment": "tomojax._workflow",
    "Reconstruction": "tomojax._workflow",
    "Scan": "tomojax._workflow",
    "align": "tomojax._workflow",
    "backproject": "tomojax._workflow",
    "load": "tomojax._workflow",
    "load_reconstruction": "tomojax._workflow",
    "project": "tomojax._workflow",
    "reconstruct": "tomojax._workflow",
    "save": "tomojax._workflow",
    "ConeBeam": "tomojax.geometry",
    "ConeGeometry": "tomojax.geometry",
    "Detector": "tomojax.geometry",
    "Grid": "tomojax.geometry",
    "LaminographyGeometry": "tomojax.geometry",
    "ParallelGeometry": "tomojax.geometry",
}

__all__ = [
    "Alignment",
    "ConeBeam",
    "ConeGeometry",
    "Detector",
    "Grid",
    "LaminographyGeometry",
    "ParallelGeometry",
    "Reconstruction",
    "Scan",
    "__version__",
    "align",
    "backproject",
    "load",
    "load_reconstruction",
    "project",
    "reconstruct",
    "save",
]


def __getattr__(name: str) -> Any:
    module = _SOURCES.get(name)
    if module is None:
        raise AttributeError(f"module 'tomojax' has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
