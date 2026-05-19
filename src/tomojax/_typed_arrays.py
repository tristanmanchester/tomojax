"""Typed array adapters for third-party libraries with incomplete stubs."""

from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.floating[Any]]
type Float32Array = NDArray[np.float32]
type Float64Array = NDArray[np.float64]
type Int64Array = NDArray[np.int64]
type BoolArray = NDArray[np.bool_]


def jax_float32_array(value: object) -> jax.Array:
    """Convert an array-like value to a JAX float32 array."""
    return cast("jax.Array", jnp.asarray(value, dtype=np.float32))


def numpy_array(value: object) -> NDArray[np.object_]:
    """Convert an array-like value to a NumPy array at an untrusted IO boundary."""
    return cast("NDArray[np.object_]", np.asarray(value))


def numpy_float32_array(value: object) -> Float32Array:
    """Convert an array-like value to a NumPy float32 array."""
    return cast("Float32Array", np.asarray(value, dtype=np.float32))


def numpy_float64_array(value: object) -> Float64Array:
    """Convert an array-like value to a NumPy float64 array."""
    return cast("Float64Array", np.asarray(value, dtype=np.float64))


def numpy_int64_array(value: object) -> Int64Array:
    """Convert an array-like value to a NumPy int64 array."""
    return cast("Int64Array", np.asarray(value, dtype=np.int64))


def finite_mask(value: NDArray[object] | FloatArray | Float32Array | Float64Array) -> BoolArray:
    """Return a typed finite mask for numeric arrays."""
    return cast("BoolArray", np.isfinite(value))


def object_list(value: object) -> list[object]:
    """Copy a runtime sequence into a typed object list."""
    if isinstance(value, list | tuple):
        return [cast("object", item) for item in value]
    return []


def float_list(value: object) -> list[float]:
    """Copy numeric runtime values into a typed float list."""
    if not isinstance(value, list | tuple):
        return []
    return [float(item) for item in value if isinstance(item, int | float)]


def object_mapping(value: object) -> dict[str, object]:
    """Copy a runtime mapping into a string-keyed object mapping."""
    if not isinstance(value, dict):
        return {}
    return {str(key): cast("object", item) for key, item in value.items()}


def object_mapping_list(value: object) -> list[dict[str, object]]:
    """Copy a runtime list of mappings into typed string-keyed mappings."""
    if not isinstance(value, list | tuple):
        return []
    return [object_mapping(item) for item in value if isinstance(item, dict)]


def update_jax_config(name: str, val: object) -> None:
    """Update a JAX config option through a typed project boundary."""
    import jax

    jax.config.update(name, val)


def write_image(path: object, image: object) -> None:
    """Write an image through a typed boundary for imageio's broad overloads."""
    import imageio.v3 as iio

    iio.imwrite(path, image)


def shape2(array: object) -> tuple[int, int]:
    """Return the last two dimensions as concrete ints."""
    shape = np.shape(array)
    return int(shape[-2]), int(shape[-1])


def shape1(array: object) -> tuple[int]:
    """Return a one-dimensional shape as a concrete int tuple."""
    shape = np.shape(array)
    return (int(shape[0]),)


def shape3(array: object) -> tuple[int, int, int]:
    """Return a three-dimensional shape as concrete ints."""
    shape = np.shape(array)
    return int(shape[0]), int(shape[1]), int(shape[2])


def equal_to_int(array: object, value: int) -> BoolArray:
    """Compare an integer array to a scalar."""
    return cast("BoolArray", np.equal(array, value))


def scalar_int(value: object) -> int:
    """Convert a scalar-like value to int."""
    return int(value)


def scalar_float(value: object) -> float:
    """Convert a scalar-like value to float."""
    return float(value)


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


def mean_float32(values: object, axis: object) -> Float32Array:
    """Mean-reduce values and return float32."""
    return numpy_float32_array(np.asarray(values).mean(axis=axis, dtype=np.float32))


def unique_int_count_map(values: object) -> dict[str, int]:
    """Return string-keyed counts for integer values."""
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {str(int(key)): int(count) for key, count in zip(unique, counts, strict=True)}
