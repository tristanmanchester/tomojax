"""Domain constructors behind the worker's typed configuration forms."""

from typing import get_args

import tomojax as tj
from tomojax import alignment, corrections, datasets
from tomojax._loading_geometry import DeclaredGeometry
from tomojax.recon.api import MethodConfig


def option_types() -> tuple[type, ...]:
    """Return constructors for all tagged steps, method settings and nested value objects."""
    return (
        DeclaredGeometry,
        tj.Grid,
        tj.Detector,
        datasets.SimulationArtifacts,
        alignment.AlignConfig,
        *get_args(MethodConfig.__value__),
        *(
            cls
            for name in corrections.__all__
            if isinstance(cls := getattr(corrections, name), type)
            and issubclass(cls, corrections.Step)
            and cls is not corrections.Step
        ),
    )
