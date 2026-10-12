"""Bounded calibrated raw-row correction through terminal detector averaging.

Each band owns its output rows and log diagnostics. Calibration is computed
once on the full detector; the optical halo contains measured input rows only.
Global/view-selecting and FFT steps are deliberately not windowed here.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.array_ops import mean_detector_pixels

from ._calibration import count_ceiling_readings, dtype_ceiling
from ._distortion import RadialDistortion, pullback_rows, source_rows
from ._normalization import log_transmission, normalize
from ._source_batches import batches
from ._steps import BeamHardening


def lens_row_slab(counts, ordered, batch_views, factor, budget):
    if batch_views is not None or factor == 1:
        return None
    transmission = [s for d, s in ordered if d == "transmission"]
    if (
        len(transmission) != 1
        or not isinstance(transmission[0], RadialDistortion)
        or any(d == "counts" for d, _ in ordered)
        or any(
            d == "line integrals"
            and (s.selects_views or not (s.whole_rows or isinstance(s, BeamHardening)))
            for d, s in ordered
        )
    ):
        return None
    if not isinstance(counts, np.ndarray) and not getattr(counts, "_efficient_row_windows", False):
        return None
    views, rows, cols = counts.shape
    slab = (min(rows, budget // (8 * views * cols)) // factor) * factor
    return slab if slab >= factor else None


@partial(jax.jit, static_argnames=("lens", "rows", "epsilon", "head"))
def _warp_log(transmission, *, lens, first, begin, rows, epsilon, head):
    warped = pullback_rows(
        lens,
        transmission,
        input_first=begin,
        output_first=first,
        output_rows=rows,
    )
    return log_transmission(warped, epsilon=epsilon, line_steps=head)


@partial(jax.jit, static_argnames=("lens", "rows", "epsilon", "head"))
def _fused_pass(raw, lo, hi, t, fields, *, lens, first, begin, rows, epsilon, head):
    transmission = normalize(raw, lo, hi, t, *fields, epsilon=epsilon)
    return _warp_log(
        transmission,
        lens=lens,
        first=first,
        begin=begin,
        rows=rows,
        epsilon=epsilon,
        head=head,
    )


@partial(jax.jit, static_argnames=("step",))
def _stored_step(step, data):
    return step.apply(data).astype(jnp.float32)


def correct_lens_rows(
    counts,
    fields,
    interpolation,
    lens,
    head,
    tail,
    *,
    epsilon,
    factor,
    slab,
    budget,
    target,
):
    views, height, width = counts.shape
    output = (
        np.empty((views, height // factor, width // factor), np.float32)
        if target is None
        else target
    )
    lo, hi, t = (jax.device_put(q) for q in interpolation)
    calibration = (fields.flats, fields.dark, fields.flat_variance, fields.dark_variance)
    tally = np.zeros(2, np.int64)
    ceiling, ceiling_readings = dtype_ceiling(counts), 0
    for first in range(0, height, slab):
        stop = min(first + slab, height)
        begin, end = source_rows(lens, first, stop, height, width)
        if ceiling is not None:
            # Input provenance owns acquired rows, not repeatedly sampled halo
            # rows. Include that ownership band even for a contracting lens.
            begin, end = min(begin, first), max(end, stop)
        source = (
            counts[:, begin:end]
            if isinstance(counts, np.ndarray)
            else counts.window(slice(begin, end), slice(None))
        )
        # Keep one bounded all-view output band for exact whole-row statistics.
        part = np.empty((views, stop - first, width), np.float32)
        window_fields = tuple(
            jax.device_put(a[:, begin:end] if a.ndim == 3 else a[begin:end]) for a in calibration
        )
        input_bytes = max(4, np.dtype(counts.dtype).itemsize)
        batch = max(
            1,
            min(64, views, budget // (width * (input_bytes * (end - begin) + 8 * (stop - first)))),
        )

        def diagnose(raw, owned=(first - begin, stop - begin)):
            nonlocal ceiling_readings
            if ceiling is not None:
                ceiling_readings += count_ceiling_readings(raw[:, owned[0] : owned[1]], ceiling)

        pending = None
        for v0, v1, raw in batches(
            source, views, batch, after_read=diagnose if ceiling is not None else None
        ):
            kwargs = {
                "lens": lens,
                "first": first,
                "begin": begin,
                "rows": stop - first,
                "epsilon": epsilon,
                "head": head,
            }
            if raw.dtype.itemsize > 4:
                transmission = normalize(
                    jnp.asarray(raw),
                    lo[v0:v1],
                    hi[v0:v1],
                    t[v0:v1],
                    *window_fields,
                    epsilon=epsilon,
                )
                done = _warp_log(transmission, **kwargs)
            else:
                done = _fused_pass(
                    jnp.asarray(raw), lo[v0:v1], hi[v0:v1], t[v0:v1], window_fields, **kwargs
                )
            done[0].copy_to_host_async()
            if pending is not None:
                start, finish, result = pending
                part[start:finish] = np.asarray(result[0])
                tally += [int(n) for n in result[1:]]
            pending = (v0, v1, done)
        if pending is not None:
            start, finish, result = pending
            part[start:finish] = np.asarray(result[0])
            tally += [int(n) for n in result[1:]]
        for step in tail:
            part[:] = np.asarray(_stored_step(step, jnp.asarray(part)))
        # Discarded edge rows still contributed their original log diagnostics.
        output[:, first // factor : stop // factor] = mean_detector_pixels(part, factor)
    return output, tally, ceiling_readings
