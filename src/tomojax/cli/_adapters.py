"""Run admitted operations on the worker's execution thread."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial, singledispatchmethod
import json
from pathlib import Path
from typing import Any, cast

import h5py
import msgspec
import numpy as np

import tomojax as tj
from tomojax import progress
from tomojax._loading import load_raw_frames_with_declaration, load_scan_with_declaration
from tomojax._planning import resolve_reconstruction
from tomojax._resources import sample
from tomojax._scan import combined_geometry
from tomojax._simulate import build_sim_config, simulate_config
from tomojax._smooth_alignment import resolve_alignment
from tomojax._workflow import align_resolved, reconstruct_resolved
from tomojax.cli import execution as ex, operations as op, products as p
from tomojax.cli._arguments import arguments, source
from tomojax.cli._handles import Root, handle, stage_array, unit_of
from tomojax.cli._plans import plan_product
from tomojax.cli._results import VOLUME_DATASET, length_unit, scan_axes
from tomojax.core.array_ops import check_binning
from tomojax.errors import InvalidParameter
from tomojax.wire.arrays import Array, Axis, Coordinates, Hdf5, Record, Uniform
from tomojax.wire.progress import Event

type Work = Callable[[], Root | p.DataProduct]


def execute(command: op.ExecutionInput, context: ex.ExecutionContext) -> ex.Success:
    """Forward watch events and attach provenance from this command's admission."""
    context.control.checkpoint()
    operation = (
        command.operation if isinstance(command, op.Run | op.Plan | op.Validate) else command
    )
    parameters = msgspec.to_builtins(
        command if isinstance(command, op.Validate | op.Plan) else operation
    )
    manifest = p.Provenance(parameters.pop("type"), parameters, context.lineage(command))

    def event(record: progress.Event) -> None:
        context.event(msgspec.convert(progress.to_json(record), type=Event))

    def diagnostic(pointer: str, data: Any) -> Array:
        a = np.asarray(data)
        axes = tuple(Axis(f"axis{i}", "diagnostic", "index", Uniform(0, 1)) for i in range(a.ndim))
        return Array(stage_array(context, a), a.shape, a.dtype.str, axes, "unspecified", pointer)

    with progress.watch(event, control=cast("progress.Control", context.control)):
        if isinstance(operation, op.Plane):
            if not isinstance(command, op.Validate):
                return ex.browse_plane(operation, context)
            ex.plane_source(operation, context)
            item = p.Validated()
        else:
            work = Admission(context, manifest, isinstance(command, op.Plan)).prepare(operation)
            item = p.Validated() if isinstance(command, op.Validate) else work()
        context.control.checkpoint()
        info = item.info if isinstance(item, tj.Reconstruction | tj.Alignment) else {}
        summary = ex.summarise(info, externalise=diagnostic)
        if isinstance(item, tj.Scan | tj.Frames | tj.Reconstruction | tj.Alignment | tj.Simulation):
            storage = unit = None
            if (
                isinstance(operation, op.Open)
                and isinstance(item, tj.Reconstruction)
                and h5py.is_hdf5(operation.path)
            ):
                with h5py.File(operation.path, "r") as f:
                    # A canonical member also covers files saved in a legacy axis order.
                    if f["/entry/processing/tomojax"].attrs.get("volume_axes_order") in (
                        "zyx",
                        b"zyx",
                    ):
                        storage = Hdf5(operation.path, VOLUME_DATASET)
                    if (group := f.get("/entry/geometry")) is not None:
                        unit = length_unit(
                            json.loads(
                                cast("str | bytes", group.attrs.get("geometry_meta_json", "{}"))
                            )
                        )
            return ex.Success(handle(item, volume_storage=storage, unit=unit), summary, manifest)
        return ex.Success(item, summary, manifest)


@dataclass(frozen=True)
class Admission:
    """Shared inputs to operation preflight. The returned callable performs the numerical work."""

    context: ex.ExecutionContext
    manifest: p.Provenance
    plan: bool

    @singledispatchmethod
    def prepare(self, operation: Record) -> Work:
        raise ValueError(f"operation unavailable: {operation.__struct_config__.tag}")

    @prepare.register
    def inspect(self, operation: op.Inspect) -> Work:
        return lambda: p.Inspection(dict(tj.inspect(operation.path)))

    @prepare.register
    def open(self, operation: op.Open) -> Work:
        options = arguments(operation, self.context, omit=("path", "kind"))
        declared = options.pop("geometry")
        if operation.kind in ("scan", "frames"):
            loader = (
                load_scan_with_declaration
                if operation.kind == "scan"
                else load_raw_frames_with_declaration
            )
            item = loader(operation.path, declared=declared, **options)
        else:
            for field in msgspec.structs.fields(operation):
                if (
                    field.name not in ("path", "kind")
                    and getattr(operation, field.name) != field.default
                ):
                    raise InvalidParameter(
                        field.name, f"{field.name} applies only when opening a scan or frames"
                    )
            load = {
                "reconstruction": tj.load_reconstruction,
                "alignment": tj.load_alignment,
                "simulation": tj.load_simulation,
            }[operation.kind]
            item = load(operation.path)
        return lambda: item

    @prepare.register
    def simulate(self, operation: op.Simulate) -> Work:
        grid, detector, cfg = build_sim_config(**arguments(operation, self.context))
        return partial(simulate_config, cfg, grid=grid, detector=detector)

    @prepare.register
    def combine(self, operation: op.Combine) -> Work:
        scans = [
            source(ref, self.context, kinds=(tj.Scan,), field=f"sources[{i}]")
            for i, ref in enumerate(operation.sources)
        ]
        try:
            combined_geometry(scans)
        except ValueError as err:
            raise InvalidParameter("sources", str(err)) from err
        return lambda: tj.Scan.combine(scans, name=operation.name)

    @prepare.register
    def fit(self, operation: op.Reconstruct | op.Align) -> Work:
        scan = source(operation.source, self.context, kinds=(tj.Scan,))
        options = arguments(operation, self.context, omit=("source", "axis", "index", "centres"))
        if isinstance(operation, op.Align):
            alignment = resolve_alignment(scan, **options)

            def apply_alignment():
                with progress.task(
                    "align",
                    ", ".join(operation.fit) if isinstance(operation.fit, tuple) else operation.fit,
                    None,
                    "fit",
                ):
                    return align_resolved(scan, operation.fit, alignment)

            compute = apply_alignment
            work = alignment
        else:
            reconstruction = resolve_reconstruction(scan, **options)
            if isinstance(operation, op.Slice):
                if operation.index >= reconstruction.scan.grid.shape["zyx".index(operation.axis)]:
                    raise InvalidParameter("index", "plane index outside volume")
                if isinstance(operation, op.CentreSweep) and (
                    not operation.centres or not np.isfinite(operation.centres).all()
                ):
                    raise InvalidParameter(
                        "centres", "centres must be a nonempty sequence of finite detector offsets"
                    )
            compute = partial(
                reconstruct_resolved, reconstruction, warm_start=bool(operation.warm_start)
            )
            work = reconstruction
        snapshots = sample(work)
        for snapshot in snapshots:
            self.context.snapshot(snapshot)
        if self.plan:
            return partial(plan_product, scan, work, snapshot=snapshots)
        return compute

    @prepare.register
    def region(self, operation: op.RegionPreview) -> Work:
        from tomojax._region import check_region_geometry

        scan = source(operation.source, self.context, kinds=(tj.Scan,))
        check_region_geometry(scan)
        return p.Validated

    @prepare.register
    def project(self, operation: op.Project | op.Backproject) -> Work:
        scan = source(operation.source, self.context, kinds=(tj.Scan,))
        options = arguments(operation, self.context, omit=("source",))
        grid = options["grid"] or scan.grid
        forward = isinstance(operation, op.Project)
        name = "volume" if forward else "projections"
        data = options.pop(name)
        expected = grid.shape if forward else scan.projections.shape
        if grid.nz == scan.detector.nv == 1 and data.ndim == 2:
            data = data[None, :, :] if forward else data[:, None, :]
        if tuple(data.shape) != expected:
            raise InvalidParameter(name, f"{name} shape {data.shape} does not match {expected}")

        def apply():
            unit = unit_of(scan)
            if forward:
                pixels = np.asarray(tj.project(scan.geometry, data, **options))
                view = Axis(
                    "view",
                    "acquisition",
                    "degree",
                    Coordinates(stage_array(self.context, scan.angles)),
                )
                return p.Projections(
                    Array(
                        stage_array(self.context, pixels),
                        pixels.shape,
                        pixels.dtype.str,
                        scan_axes(scan.detector, view, unit),
                        "1",
                        "line_integrals",
                    )
                )
            pixels = np.asarray(tj.backproject(scan.geometry, data, **options))
            return p.Volume(
                ex.volume_metadata(
                    grid,
                    storage=stage_array(self.context, pixels),
                    dtype=pixels.dtype.str,
                    length_unit=unit,
                )
            )

        return apply

    @prepare.register
    def edit(self, operation: op.Select | op.Crop | op.Bin | op.Poses) -> Work:
        from tomojax._scan import detector_window
        from tomojax.core.validation import view_indices

        kinds = (tj.Scan, tj.Frames) if isinstance(operation, op.Select | op.Crop) else (tj.Scan,)
        item = source(operation.source, self.context, kinds=kinds)
        detector = item.geometry.detector
        if isinstance(operation, op.Select):
            try:
                kept = view_indices(operation.views, len(item.geometry.angles))
            except ValueError as err:
                raise InvalidParameter("views", str(err)) from err
            return partial(item.selected, kept)
        if isinstance(operation, op.Crop):
            rows, cols = slice(*operation.rows), slice(*operation.cols)
            try:
                detector_window(rows, cols, detector)
            except ValueError as err:
                lo, hi, _ = rows.indices(detector.nv)
                raise InvalidParameter("rows" if lo >= hi else "cols", str(err)) from err
            return partial(item.cropped, rows, cols)
        if isinstance(operation, op.Bin):
            check_binning(operation.binning, detector.nv, detector.nu)
            return partial(item.binned, operation.binning)
        poses = arguments(operation, self.context, omit=("source",))["poses"]
        try:
            posed = item.with_poses(poses)
        except ValueError as err:
            raise InvalidParameter("poses", str(err)) from err
        return lambda: posed

    @prepare.register
    def correct(self, operation: op.Correct | op.ReconstructSlabs) -> Work:
        from tomojax._loading import is_raw_input
        from tomojax.corrections.api import (
            check_scan_steps,
            positive_finite,
            validated_views_per_batch,
        )

        raw = isinstance(operation, op.CorrectionPreview | op.ReconstructSlabs)
        if isinstance(operation.source, op.PathInput) and not raw:
            raw = is_raw_input(operation.source.path)
        item = source(
            operation.source,
            self.context,
            kinds=(tj.Frames,) if raw else (tj.Frames, tj.Scan),
            raw=raw,
        )
        if isinstance(operation, op.ReconstructSlabs):
            settings = arguments(operation, self.context, omit=("source",))
            from tomojax._workflow import slab_settings

            grid, _, _, _ = slab_settings(item, **settings)

            def fill():
                with tj.reconstruct_slabs(item, **settings) as slabs:
                    stage = self.context.staging.array()
                    volume = np.lib.format.open_memmap(
                        stage.path, mode="w+", dtype=np.float32, shape=grid.shape
                    )
                    for interval, slab in slabs:
                        volume[interval] = slab.volume
                        del slab
                        self.context.control.checkpoint()
                    volume.flush()
                return p.Volume(
                    ex.volume_metadata(
                        grid,
                        storage=stage.reference,
                        dtype=volume.dtype.str,
                        length_unit=unit_of(item),
                    )
                )

            return fill

        settings = arguments(operation, self.context, omit=("source", "rows"))
        steps = settings.pop("corrections")
        validated_views_per_batch(operation.views_per_batch)
        if isinstance(item, tj.Scan):
            check_scan_steps(steps)
            if operation.epsilon != 1e-6 or operation.binning != 1:
                raise InvalidParameter(
                    "epsilon" if operation.epsilon != 1e-6 else "binning",
                    "epsilon and correction binning apply to raw frames",
                )
            return partial(item.corrected, *steps, views_per_batch=operation.views_per_batch)
        positive_finite(operation.epsilon, "epsilon")
        check_binning(operation.binning, *item.counts.shape[1:])
        if isinstance(operation, op.CorrectionPreview):
            low, high = operation.rows
            if not (0 <= low < high <= item.counts.shape[1] // operation.binning):
                raise InvalidParameter(
                    "rows", "expected a nonempty interval of binned detector rows"
                )
        return partial(item.corrected, *steps, **settings)

    @prepare.register
    def durable(self, operation: op.Save | op.Export) -> Work:
        from tomojax._files import validate_export_options

        kinds = (tj.Reconstruction, tj.Alignment, tj.Simulation)
        value = source(
            operation.source,
            self.context,
            kinds=(*kinds, tj.Scan) if isinstance(operation, op.Save) else kinds,
        )
        path = Path(operation.path)
        sidecar = path.with_name(path.name + ".manifest.json")
        paths = [path, sidecar]
        raw = path.suffix.lower() == ".raw"
        if isinstance(operation, op.Export):
            validate_export_options(path, operation.dtype, operation.range, operation.prefix)
            if isinstance(value, tj.Alignment) and value.image is None:
                raise InvalidParameter("source", "alignment holds no working image to export")
            if raw:
                paths.append(path.with_suffix(".json"))
        for output in paths:
            if not operation.force and (output.exists() or output.is_symlink()):
                raise FileExistsError(f"output exists: {output}")

        def write():
            staged = {p: self.context.staging.output(p, force=operation.force) for p in paths}
            if isinstance(operation, op.Save):
                tj.save(staged[path], value, include_scan=operation.include_scan)
            else:
                tj.export(
                    staged[path],
                    value,
                    dtype=operation.dtype,
                    range=operation.range,
                    prefix=operation.prefix,
                )
                if raw:
                    staged[path].with_suffix(".json").replace(staged[path.with_suffix(".json")])
            staged[sidecar].write_bytes(msgspec.json.encode(self.manifest) + b"\n")
            outputs = [str(p) for p in paths]
            if isinstance(operation, op.Export) and not raw:
                outputs.append(str(path / f"{operation.prefix}.json"))
            return p.Files(tuple(outputs))

        return write
