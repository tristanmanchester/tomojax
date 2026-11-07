"""Tiled float32 CUDA implementation of the adaptive spatial ring reference."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
from jax.experimental.buffer_callback import buffer_callback
import numpy as np

from tomojax.core.cuda import launch_kernel

from ._steps import median_network

_REDUCTION_THREADS = 512
_TILE = 128


def _literal(value: float) -> str:
    return f"{float(np.float32(value)):.17e}f"


@lru_cache(maxsize=32)
def _source(
    rows: int,
    columns: int,
    gamma: float,
    local: int,
    taps: tuple[tuple[float, ...], ...],
) -> str:
    constants = {
        "N": columns,
        "V": rows,
        "B": _REDUCTION_THREADS,
        "C": (columns + _REDUCTION_THREADS - 1) // _REDUCTION_THREADS,
        "T": _TILE,
    }
    source = "\n".join(f"#define {key} {value}" for key, value in constants.items())
    source += "\n" + Path(__file__).with_name("_rings_reference.cu").read_text()
    offsets = list(range(-local, local + 1, 8))
    operations, result = median_network(len(offsets))
    median = [
        f"float a{i}=fabsf(r[max(0,min(N-1,x+({offset})))]);" for i, offset in enumerate(offsets)
    ]
    for dest, minimum, a, b in operations:
        median.append(f"float a{dest}={'fminf' if minimum else 'fmaxf'}(a{a},a{b});")
    median.append(f"return a{result};")
    source = source.replace(
        "/* LOCAL_MEDIAN */",
        "__device__ float local_median(const float* r,int x) {" + "".join(median) + "}",
    )
    narrow = "float fit=0.0f;\n" + "\n".join(
        f"fit+={_literal((taps[-1][6 - j] + taps[-1][6 + j]) * 0.5)}"
        f"*(v[max(0,x-{j})]+v[min(N-1,x+{j})]);"
        for j in range(2, 7)
    )
    source = source.replace("/* NARROW */", narrow)
    source = source.replace(
        "SCATTER_NORMALIZATION", _literal(1 / np.sqrt(1 + np.square(taps[-1]).sum()))
    )
    blends: list[str] = []
    for index, tap in enumerate(taps[1:], 1):
        window = len(tap) // 2
        difference = np.pad(tap, 30 - window) - np.asarray(taps[0])
        variance = _literal(np.square(difference).sum())
        lag1 = _literal((difference[1:] * difference[:-1]).sum())
        lag2 = _literal((difference[2:] * difference[:-2]).sum())
        blends.append(
            f"{{float noise=scatter*sqrtf({variance}+2.0f*(rho1*{lag1}+rho2*{lag2}));"
            f"float diff=fits[({index}*V+row)*N+x]-out;"
            f"out+=(diff>0?1.0f:diff<0?-1.0f:0.0f)*fmaxf(fabsf(diff)-{_literal(gamma)}*noise,0.0f);}}"
        )
    return source.replace("/* BLEND */", "\n".join(blends))


def reference(
    values: jax.Array,
    taps: tuple[tuple[float, ...], ...],
    gamma: float,
    local: int,
) -> jax.Array:
    """Same weighted quadratic fits and noise blending; no detector-sized shared arrays."""
    rows, columns = values.shape
    source = _source(rows, columns, gamma, local, taps)

    def launch(ctx: Any, outputs: tuple[Any, ...], values: Any) -> None:
        result, residual, sigma, floor, fits, narrow_residual, correlations = outputs
        launches = (
            ("residual_mad", (rows,), _REDUCTION_THREADS, (values, residual, sigma)),
            ("sigma_floor", (1,), _REDUCTION_THREADS, (sigma, floor)),
            (
                "weighted_tiles",
                ((columns + _TILE - 1) // _TILE, rows),
                _TILE,
                (values, residual, sigma, floor, fits, narrow_residual),
            ),
            ("row_correlations", (rows,), _REDUCTION_THREADS, (narrow_residual, correlations)),
            (
                "blend_tiles",
                ((columns + _TILE - 1) // _TILE, rows),
                _TILE,
                (fits, narrow_residual, correlations, result),
            ),
        )
        for name, grid, block, arguments in launches:
            launch_kernel(
                source,
                name,
                options=("--fmad=true",),
                stream=ctx.stream,
                grid=grid,
                block=(block,),
                arguments=arguments,
            )

    shapes = (
        values.shape,
        values.shape,
        (rows,),
        (1,),
        (5, rows, columns),
        values.shape,
        (rows, 2),
    )
    outputs = tuple(jax.ShapeDtypeStruct(shape, values.dtype) for shape in shapes)
    return buffer_callback(
        launch, outputs, vmap_method="sequential", command_buffer_compatible=True
    )(values)[0]
