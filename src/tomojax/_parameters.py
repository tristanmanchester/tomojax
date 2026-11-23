"""The closed parameter choices shared by Python operations and their front ends."""

from importlib import import_module
from types import ModuleType
from typing import Any, Literal, get_args

from tomojax.errors import InvalidParameter

type Method = Literal["auto", "fbp", "fourier", "cgls", "fista", "spdhg", "sirt", "mlem"]
type Filter = Literal["ramp", "hann", "shepp-logan"]
type RegionName = Literal["auto", "bbox", "cube", "cyl"]
type AlignMethod = Literal["joint", "opposite-views"]
type Fit = Literal["center", "tilt", "roll", "translations", "rotations"]
type Phantom = Literal["shepp", "cube", "sphere", "blobs", "random_shapes", "lamino_disk"]
type GeometryName = Literal["parallel", "lamino", "cone"]
type TiltAbout = Literal["x", "z"]

type ProjectorBackend = Literal["auto", "jax", "cuda"]


def check_backend(backend: str, context: str) -> None:
    """Check the backend vocabulary before querying any device."""
    if backend not in get_args(ProjectorBackend.__value__):
        raise InvalidParameter("backend", f"{context}: backend must be 'auto', 'jax' or 'cuda'")


class AnnotationModule(ModuleType):
    """Load a numerical module only when a runtime consumer resolves an annotation."""

    def __getattr__(self, name: str) -> Any:
        return getattr(import_module(self.__name__), name)
