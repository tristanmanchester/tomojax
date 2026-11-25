"""Typed array adapters for third-party libraries with incomplete stubs."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from tomojax._parameters import AnnotationModule

if TYPE_CHECKING:
    import jax
else:
    jax = AnnotationModule("jax")

type JaxArray = jax.Array
type Float32Array = NDArray[np.float32]
type Float64Array = NDArray[np.float64]


class Device(Protocol):
    """A ``jax.Device``, as TomoJAX uses one (jaxlib's class has no type stubs)."""

    @property
    def id(self) -> int: ...

    @property
    def platform(self) -> str: ...

    @property
    def local_hardware_id(self) -> int: ...


def device_id(device: Device) -> str:
    """Identify a participating JAX device on the worker protocol."""
    return f"{device.platform}:{device.id}"


def numpy_array(value: object) -> NDArray[np.object_]:
    """Convert an array-like value to a NumPy array at an untrusted IO boundary."""
    return cast("NDArray[np.object_]", np.asarray(value))


def numpy_float32_array(value: object) -> Float32Array:
    """Convert an array-like value to a NumPy float32 array."""
    return cast("Float32Array", np.asarray(value, dtype=np.float32))


def numpy_float64_array(value: object) -> Float64Array:
    """Convert an array-like value to a NumPy float64 array."""
    return cast("Float64Array", np.asarray(value, dtype=np.float64))


def object_list(value: object) -> list[object]:
    """Copy a runtime sequence into a typed object list."""
    if isinstance(value, list | tuple):
        return [cast("object", item) for item in value]
    return []


def object_mapping(value: object) -> dict[str, object]:
    """Copy a runtime mapping into a string-keyed object mapping."""
    if not isinstance(value, dict):
        return {}
    return {str(key): cast("object", item) for key, item in value.items()}


def write_image(path: str | os.PathLike[str], image: np.ndarray) -> None:
    """Write an image through a typed boundary for imageio's overloads, moved on once whole."""
    from pathlib import Path

    import imageio.v3 as iio

    from tomojax.core.files import replaced_when_complete

    target = Path(path)
    with replaced_when_complete(target, "image") as temporary:
        iio.imwrite(temporary, image, extension=target.suffix or ".png")


def true_count(mask: object) -> int:
    """Count true values in a mask."""
    return int(np.asarray(mask).sum())


def any_true(mask: object) -> bool:
    """Return whether any mask value is true."""
    return bool(np.asarray(mask).any())


def finite_values_float64(values: object, mask: object) -> Float64Array:
    """Return finite values as float64 through NumPy's dynamic indexing boundary."""
    return numpy_float64_array(np.asarray(values)[np.asarray(mask)]).astype(np.float64, copy=False)


def linspace_indices(start: int, stop: int, num: int) -> list[int]:
    """Return integer indices sampled evenly across a range."""
    return [int(value) for value in np.linspace(start, stop, num=num, dtype=np.int64).tolist()]


def sum_float64(values: object) -> float:
    """Sum numeric values as float64."""
    return float(np.asarray(values, dtype=np.float64).sum(dtype=np.float64))


def min_float(values: object) -> float:
    """Return the minimum numeric value."""
    return float(np.asarray(values).min())


def max_float(values: object) -> float:
    """Return the maximum numeric value."""
    return float(np.asarray(values).max())


def unique_int_count_map(values: object) -> dict[str, int]:
    """Return string-keyed counts for integer values."""
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {str(int(key)): int(count) for key, count in zip(unique, counts, strict=True)}
