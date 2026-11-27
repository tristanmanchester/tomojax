"""The exterior fit on physical horizontal chords of a rolled detector."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
import math
from typing import TYPE_CHECKING

import numpy as np

from tomojax import progress
from tomojax.core.geometry.cone import ConeGeometry, ConeSegments, cone_parts
from tomojax.core.host_volume import host_array
from tomojax.geometry import Detector, Geometry

from ._continuation import Continuation, cone_axis_distance, parallel_axis_distance, radius_views
from ._detector_rows import DetectorRows, RowSamples

if TYPE_CHECKING:
    from collections.abc import Iterator

    import jax
    from numpy.typing import NDArray

    from tomojax._typed_arrays import Float32Array
    from tomojax.core.cuda_types import CuPyValue
    from tomojax.core.geometry import ConeBeam


@dataclass(frozen=True)
class _Panel:
    first: int
    stop: int
    beam: ConeBeam | None
    layout: DetectorRows
    turns: int


def _horizontal(geometry: Geometry, detector: Detector) -> Geometry:
    """Keep the per-view poses, source and panel center while removing in-plane roll."""
    if not is_dataclass(geometry) or isinstance(geometry, type):
        raise TypeError(f"cannot rectify detector on {type(geometry).__name__} geometry")
    if isinstance(geometry, ConeSegments):
        return replace(
            geometry, segments=tuple(_horizontal(s, detector) for s in geometry.segments)
        )
    if (base := getattr(geometry, "base", None)) is not None:
        return replace(geometry, base=_horizontal(base, detector))
    own: Detector = getattr(geometry, "detector", detector)
    panel = replace(detector, center=own.center)
    if isinstance(geometry, ConeGeometry):
        return replace(geometry, detector=panel, beam=replace(geometry.beam, detector_roll=0.0))
    return replace(geometry, detector=panel, detector_roll=0.0)


@dataclass(frozen=True)
class ObliqueExterior:
    geometry: Geometry
    detector: Detector
    panels: tuple[_Panel, ...]

    @classmethod
    def for_scan(cls, geometry: Geometry, detector: Detector, views: int) -> ObliqueExterior | None:
        parts = cone_parts(geometry, detector)
        roll = float(getattr(geometry, "detector_roll", 0.0))
        if not roll and not any(beam.detector_roll for _, beam, _ in parts or ()):
            return None
        panels: list[_Panel] = []
        first = 0
        for count, beam, own in parts or ((views, None, detector),):
            angle = roll if beam is None else beam.detector_roll
            # Physically identical rectangular panels can have transposed or reversed
            # storage. Use the same tap order before interpolating their measured rays.
            turns = math.floor((angle + 45) / 90)
            canonical = (
                replace(own, nu=own.nv, nv=own.nu, du=own.dv, dv=own.du) if turns % 2 else own
            )
            layout = DetectorRows(canonical, angle - turns * 90)
            stop = first + (count or views)
            horizontal_beam = None if beam is None else replace(beam, detector_roll=0.0)
            panels.append(_Panel(first, stop, horizontal_beam, layout, turns))
            first = stop
        detectors = [p.layout.horizontal for p in panels]
        du, dv = min(d.du for d in detectors), min(d.dv for d in detectors)
        panel = replace(
            detectors[0],
            nu=max(math.ceil(d.nu * d.du / du - 1e-10) for d in detectors),
            nv=max(math.ceil(d.nv * d.dv / dv - 1e-10) for d in detectors),
            du=du,
            dv=dv,
        )
        return cls(_horizontal(geometry, panel), panel, tuple(panels))

    def completed(
        self, data: np.ndarray | jax.Array | CuPyValue, coarse: Detector, unseen: NDArray[np.bool_]
    ) -> tuple[Float32Array, NDArray[np.bool_]]:
        """Fill the bounded auxiliary detector, continuing only actual measured chords."""
        views, rows, cols = data.shape
        full = host_array((views, coarse.nv, 3 * coarse.nu), copies=4)
        missing = host_array(full.shape, dtype=bool)
        missing[:] = unseen[:, None, :]
        vertical = np.arange(coarse.nv)
        batch = max(1, (16 << 20) // (4 * rows * cols))
        pad = coarse.nu
        groups: dict[tuple[DetectorRows, ConeBeam | None], list[_Panel]] = {}
        for panel in self.panels:
            groups.setdefault((panel.layout, panel.beam), []).append(panel)
        for (layout, beam), panels in groups.items():
            det = replace(coarse, center=layout.measured.center)
            low, high = layout.limits(det, vertical)
            u, v = layout.coordinates(det, vertical)
            samples = RowSamples(
                np.clip(u, 0, layout.measured.nu - 1), np.clip(v, 0, layout.measured.nv - 1)
            )
            columns = np.arange(det.nu)
            inside = (columns >= low[:, None]) & (columns < high[:, None])
            for panel in panels:
                for begin in range(panel.first, panel.stop, batch):
                    progress.check_canceled()
                    part = slice(begin, min(begin + batch, panel.stop))
                    block = data[part]
                    transfer = getattr(block, "get", None)
                    host = np.asarray(transfer() if transfer is not None else block)
                    measured = samples.read(np.rot90(host, -panel.turns, axes=(1, 2)), 0)
                    np.copyto(full[part, :, pad : pad + det.nu], measured, where=inside)
            indices = np.concatenate([np.arange(p.first, p.stop) for p in panels])
            selected = indices[radius_views(len(indices))]
            for row in range(det.nv):
                lo, hi = low.item(row), high.item(row)
                width = hi - lo
                if not width:
                    continue
                start, stop = pad + lo, pad + hi
                center = (lo + (width - det.nu) / 2) * det.du + det.center[0]
                chord = replace(det, nu=width, center=(float(center), det.center[1]))
                distance = (
                    parallel_axis_distance(chord, width, width)
                    if beam is None
                    else cone_axis_distance(beam, chord, width)
                )
                fit = Continuation(distance, width, width)
                radius = fit.stack_radius(full[selected, row : row + 1, start:stop])[None]
                for panel in panels:
                    part = slice(panel.first, panel.stop)
                    missing[part, row, start:stop] = False
                    measured = full[part, row : row + 1, start:stop]
                    left, right = fit.continued(np.asarray, measured, radius)
                    full[part, row : row + 1, start - width : start] = left
                    full[part, row : row + 1, stop : stop + width] = right
        return full, missing

    def measured_rays(
        self,
        blocks: Iterator[tuple[slice[int, int, None], Float32Array | CuPyValue]],
        *,
        positive: bool,
    ) -> Iterator[tuple[slice[int, int, None], Float32Array]]:
        """Sample each horizontal background on the original measured detector pixels."""
        horizontal = self.detector
        panel_index = 0
        samples = None
        for part, block in blocks:
            host = np.asarray(block if isinstance(block, np.ndarray) else block.get())
            begin = part.start
            while begin < part.stop:
                panel = self.panels[panel_index]
                det = panel.layout.measured
                if samples is None:
                    c, s = panel.layout.axes
                    u = (np.arange(det.nu) - (det.nu - 1) / 2) * det.du
                    v = (np.arange(det.nv)[:, None] - (det.nv - 1) / 2) * det.dv
                    x = (c * u - s * v) / horizontal.du + (horizontal.nu - 1) / 2
                    z = (s * u + c * v) / horizontal.dv + (horizontal.nv - 1) / 2
                    samples = RowSamples(x, z)
                stop = min(part.stop, panel.stop)
                native = samples.read(host[begin - part.start : stop - part.start], 0)
                if positive:
                    _ = np.maximum(native, 0, out=native)
                yield slice(begin, stop), np.rot90(native, panel.turns, axes=(1, 2))
                begin = stop
                if begin == panel.stop:
                    panel_index += 1
                    samples = None
