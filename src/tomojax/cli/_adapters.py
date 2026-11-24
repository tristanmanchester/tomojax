"""Run admitted operations on the worker's execution thread."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial, singledispatchmethod
from pathlib import Path
from typing import Any, cast

import h5py
import msgspec
import numpy as np

import tomojax as tj
from tomojax import interactive, progress
from tomojax._loading import load_raw_frames_with_declaration, load_scan_with_declaration
from tomojax._planning import resolve_reconstruction
from tomojax._resources import sample
from tomojax._scan import combined_geometry
from tomojax._simulate import build_sim_config, simulate_config
from tomojax._smooth_alignment import resolve_alignment
from tomojax._workflow import align_resolved, reconstruct_resolved, validate_save
from tomojax.cli import execution as ex, operations as op, products as p
from tomojax.cli._arguments import arguments, source
from tomojax.cli._handles import Root, handle, stage_array
from tomojax.cli._interactive import correction_product, region_product, slice_product
from tomojax.cli._options import check_destinations
from tomojax.cli._plans import plan_product
from tomojax.cli._results import scan_axes, unit_of
from tomojax.cli.products import VOLUME_DATASET
from tomojax.core.array_ops import check_binning
from tomojax.errors import InvalidParameter
from tomojax.wire.arrays import Array, Axis, Coordinates, Hdf5, Record, Uniform
from tomojax.wire.progress import Event, Preview

type Work = Callable[[], Root | p.DataProduct | ex.NewHandle]

EXECUTABLE = (
    op.Inspect,
    op.Open,
    op.Validate,
    op.Plan,
    op.Run,
    op.Plane,
    op.Slice,
    op.CentreSweep,
    op.CorrectionPreview,
    op.RegionPreview,
)


def execute(command: op.ExecutionInput, context: ex.ExecutionContext) -> ex.Success:
    """Forward watch events and attach provenance from this command's admission."""
    context.control.checkpoint()
    if not isinstance(command, EXECUTABLE):
        raise ValueError(f"operation unavailable: {command.__struct_config__.tag}")
    operation = (
        command.operation if isinstance(command, op.Run | op.Plan | op.Validate) else command
    )
    parameters = msgspec.to_builtins(
        command if isinstance(command, op.Validate | op.Plan) else operation
    )
    manifest = p.Provenance(parameters.pop("type"), parameters, context.lineage(command))

    def event(record: progress.Event) -> None:
        if isinstance(record, progress.Preview):
            arrays = {
                name: sample.store(lambda a: stage_array(context, a))
                for name, sample in record.arrays.items()
            }
            context.event(Preview(schema=2, task=record.task, time=record.time, arrays=arrays))
        else:
            context.event(msgspec.convert(progress.to_json(record), type=Event))

    def diagnostic(pointer: str, data: Any) -> Array:
        a = np.asarray(data)
        axes = tuple(Axis(f"axis{i}", "diagnostic", "index", Uniform(0, 1)) for i in range(a.ndim))
        return Array(stage_array(context, a), a.shape, a.dtype.str, axes, "unspecified", pointer)

    for snapshot in sample():
        context.snapshot(snapshot)
    with progress.watch(
        event,
        control=cast("progress.Control", context.control),
        previews=isinstance(command, op.Run) and command.previews,
    ):
        if isinstance(operation, op.Plane):
            if not isinstance(command, op.Validate):
                return ex.browse_plane(operation, context)
            ex.plane_source(operation, context)
            item = p.Validated()
        else:
            work = Admission(context, manifest, isinstance(command, op.Plan)).prepare(operation)
            item = p.Validated() if isinstance(command, op.Validate) else work()
        context.control.checkpoint()
        for snapshot in sample():
            context.snapshot(snapshot)
        info = item.info if isinstance(item, tj.Reconstruction | tj.Alignment) else {}
        if isinstance(item, tj.Alignment):
            poses = np.asarray(item.poses)
            info = {**info, "pose_rms": np.sqrt(np.mean(np.square(poses), axis=0)).tolist()}
        summary = ex.summarise(info, externalise=diagnostic)
        if isinstance(item, tj.Scan | tj.Frames | tj.Reconstruction | tj.Alignment | tj.Simulation):
            storage = None
            if isinstance(operation, op.Open) and isinstance(item, tj.Reconstruction):
                storage = saved_storage(operation.path)
            return ex.Success(handle(item, volume_storage=storage), summary, manifest)
        return ex.Success(item, summary, manifest)


def saved_storage(path: str) -> Hdf5 | None:
    """Use a saved dataset directly only when it has canonical in-memory axis order."""
    if not h5py.is_hdf5(path):
        return None
    with h5py.File(path, "r") as file:
        axes = file["/entry/processing/tomojax"].attrs.get("volume_axes_order")
        return Hdf5(path, VOLUME_DATASET) if axes in ("zyx", b"zyx") else None


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
        elif isinstance(operation, op.Slice):
            if isinstance(operation, op.CentreSweep):
                plans = interactive.resolve_sweep(
                    scan, operation.axis, operation.index, operation.centres, **options
                )
            else:
                plans = (
                    resolve_reconstruction(
                        scan, plane=(operation.axis, operation.index), **options
                    ),
                )
            compute = partial(slice_product, operation, plans, self.context)
            work = plans[0]
        else:
            work = resolve_reconstruction(scan, **options)
            compute = partial(reconstruct_resolved, work)
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
        return partial(region_product, scan, self.context)

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
            check_rows,
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
            from tomojax._scan import geometry_record
            from tomojax._workflow import slab_settings
            from tomojax.io.api import dataset_output_format, slab_volume
            from tomojax.recon.api import validate_slab_grid

            settings = arguments(operation, self.context, omit=("source", "path", "force"))
            grid, _, _ = slab_settings(item, **settings)
            validate_slab_grid(operation.method, item.geometry, grid, item.geometry.detector)
            path = Path(operation.path)
            if dataset_output_format(path) != "hdf5":
                raise InvalidParameter(
                    "path", "slab reconstruction requires a NeXus/HDF5 destination"
                )
            sidecar = path.with_name(path.name + ".manifest.json")
            inputs = set(ancestor_paths(self.manifest.inputs)) | set(
                storage_paths(self.manifest.parameters)
            )
            if isinstance(operation.source, op.PathInput):
                inputs.add(Path(operation.source.path).resolve())
            check_destinations(inputs, (path, sidecar), force=operation.force)

            def fill():
                record = geometry_record(
                    item.geometry, name=item.name, grid=grid, source=item.source
                )
                record.geometry_metadata.pop("volume_role", None)
                record.geometry_metadata.update(
                    reconstruction_method=operation.method,
                    reconstruction_info={
                        "host_slabs": True,
                        "source": item.source.source_path if item.source else None,
                    },
                )
                stage = self.context.staging.output(path, force=operation.force)
                manifest = self.context.staging.output(sidecar, force=operation.force)
                with (
                    slab_volume(stage, record.to_nxtomo_metadata(), grid.shape) as volume,
                    tj.reconstruct_slabs(item, **settings) as slabs,
                ):
                    volume.attrs["long_name"] = "volume"
                    for interval, slab in slabs:
                        volume[interval] = slab.volume
                        del slab
                        self.context.control.checkpoint()
                manifest.write_bytes(msgspec.json.encode(self.manifest) + b"\n")
                return p.Files((str(path), str(sidecar)))

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
            check_rows(operation.rows, item.counts.shape[1], operation.binning)
            return partial(correction_product, operation, item, steps, settings, self.context)
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
        if not path.name or path.name == "..":
            raise InvalidParameter("path", "expected an output file or directory name")
        sidecar = path.with_name(path.name + ".manifest.json")
        paths = [path, sidecar]
        raw = path.suffix.lower() == ".raw"
        if isinstance(operation, op.Save):
            validate_save(path, value, include_scan=operation.include_scan)
        else:
            validate_export_options(path, operation.dtype, operation.range, operation.prefix)
            if isinstance(value, tj.Alignment) and value.image is None:
                raise InvalidParameter("source", "alignment holds no working image to export")
            if raw:
                paths.append(path.with_suffix(".json"))
        check_destinations(set(ancestor_paths(self.manifest.inputs)), paths, force=operation.force)

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


def ancestor_paths(ancestors: tuple[p.Provenance, ...]) -> Iterator[Path]:
    """Identify caller-owned source files in a durable output's input history."""
    for node in ancestors:
        if node.operation == "open":
            yield Path(node.parameters["path"]).resolve()
            for field in ("flats", "darks"):
                if isinstance(path := node.parameters.get(field), str):
                    yield Path(path).resolve()
        yield from storage_paths(node.parameters)
        yield from ancestor_paths(node.inputs)


def storage_paths(value: object) -> Iterator[Path]:
    if isinstance(value, dict):
        if value.get("type") in {"path", "npy", "hdf5"}:
            yield Path(value["path"]).resolve()
        else:
            for option in value.values():
                yield from storage_paths(option)
    elif isinstance(value, tuple | list):
        for option in value:
            yield from storage_paths(option)
