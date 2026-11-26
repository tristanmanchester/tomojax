"""TomoJAX: tomography, laminography and lab cone-beam CT reconstruction and alignment.

The workflow API takes a :class:`Scan` (projections and their geometry) and
returns results that carry their geometry::

    import tomojax as tj

    scan = tj.load("scan.nxs")  # or a Nikon .xtekct
    recon = tj.reconstruct(scan)  # Geometry-aware automatic reconstruction
    recon = tj.reconstruct(scan, method="cgls", iterations=50)
    result = tj.align(scan)  # result.scan carries the corrections
    tj.save("recon.nxs", tj.reconstruct(result.scan))

Build a scan from arrays with ``tj.Scan(projections, geometry)`` and the
geometry classes below; ``tj.project`` simulates projections for any geometry.
The subpackages (``tomojax.recon``, ``tomojax.alignment``, ``tomojax.geometry``,
``tomojax.io``) hold the building blocks these use. Names load on first use,
so ``import tomojax`` does not import JAX. Importing sets growing GPU allocation
defaults for future JAX backends; explicit allocator environment settings are
preserved. See https://tomojax.dev/large-scans/ for process-wide memory configuration.
"""

from __future__ import annotations

from importlib import import_module
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tomojax._loading.files import export, inspect
    from tomojax._loading.loader import Frames, load, load_frames
    from tomojax._scan.model import Alignment, Reconstruction, Scan
    from tomojax._scan.simulation import Simulation, simulate
    from tomojax._workflow.operations import (
        align,
        backproject,
        load_alignment,
        load_reconstruction,
        load_simulation,
        project,
        reconstruct,
        reconstruct_slabs,
        save,
    )
    from tomojax._workflow.region import RegionEstimate, estimate_region
    from tomojax.errors import InvalidParameter
    from tomojax.geometry import (
        ConeBeam,
        ConeGeometry,
        Detector,
        Grid,
        LaminographyGeometry,
        ParallelGeometry,
    )
    from tomojax.recon.api import Projector

from tomojax._version import __version__

# Grow JAX's pool on demand, to JAX's own ceiling, for Python and the shell alike.
# JAX eagerly reserves 75% of device memory by default, leaving too little for
# cuSolver initialization and TomoJAX's CuPy workspaces on small GPUs; growing
# the pool on demand avoids that reservation while retaining reuse. The ceiling
# stays JAX's 75%. The pool never hands memory back, so what lives outside it
# (the CUDA context, kernel modules, cuSolver, CuPy) has only what the ceiling
# leaves: at 90% that was 0.3 GiB on an 8 GB card once a level had grown the
# pool, and K11 67208's refinement could not load a kernel. An explicit
# ``XLA_CLIENT_MEM_FRACTION`` (or JAX's legacy ``XLA_PYTHON_CLIENT_MEM_FRACTION``),
# preallocation or allocator still wins. An existing JAX backend cannot be
# reconfigured: this default applies at import, before its first operation.
_ = os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_SOURCES = {
    "InvalidParameter": "tomojax.errors",
    "Projector": "tomojax.recon.api",
    "Alignment": "tomojax._scan.model",
    "Frames": "tomojax._loading.loader",
    "Reconstruction": "tomojax._scan.model",
    "RegionEstimate": "tomojax._workflow.region",
    "Scan": "tomojax._scan.model",
    "Simulation": "tomojax._scan.simulation",
    "align": "tomojax._workflow.operations",
    "backproject": "tomojax._workflow.operations",
    "estimate_region": "tomojax._workflow.region",
    "export": "tomojax._loading.files",
    "inspect": "tomojax._loading.files",
    "simulate": "tomojax._scan.simulation",
    "load": "tomojax._loading.loader",
    "load_alignment": "tomojax._workflow.operations",
    "load_simulation": "tomojax._workflow.operations",
    "load_frames": "tomojax._loading.loader",
    "load_reconstruction": "tomojax._workflow.operations",
    "project": "tomojax._workflow.operations",
    "reconstruct": "tomojax._workflow.operations",
    "reconstruct_slabs": "tomojax._workflow.operations",
    "save": "tomojax._workflow.operations",
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
    "Frames",
    "Grid",
    "InvalidParameter",
    "LaminographyGeometry",
    "ParallelGeometry",
    "Projector",
    "Reconstruction",
    "RegionEstimate",
    "Scan",
    "Simulation",
    "__version__",
    "align",
    "backproject",
    "estimate_region",
    "export",
    "inspect",
    "load",
    "load_alignment",
    "load_frames",
    "load_reconstruction",
    "load_simulation",
    "project",
    "reconstruct",
    "reconstruct_slabs",
    "save",
    "simulate",
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
