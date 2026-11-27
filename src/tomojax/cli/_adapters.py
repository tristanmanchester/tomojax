"""Run admitted operations on the worker's execution thread."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext, suppress
from dataclasses import dataclass, field
from functools import partial, singledispatchmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import h5py
import msgspec
import numpy as np

import tomojax as tj
from tomojax import interactive, progress
from tomojax._loading.loader import (
    load_raw_frames_with_declaration,
    load_scan_with_declaration,
    projection_frames,
)
from tomojax._scan.model import combined_geometry, describe
from tomojax._scan.simulation import build_sim_config, simulate_config
from tomojax._typed_arrays import object_mapping
from tomojax._workflow.operations import (
    align_resolved,
    alignment_from_record,
    alignment_task,
    reconstruct_resolved,
    reconstruction_from_record,
    saved_info,
    simulation_from_record,
    validate_save,
)
from tomojax._workflow.planning import ResolvedReconstruction, resolve_reconstruction
from tomojax._workflow.resources import sample
from tomojax._workflow.smooth_alignment import resolve_alignment
from tomojax.cli import execution as ex, operations as op, products as p
from tomojax.cli._arguments import arguments, scan_layout, source
from tomojax.cli._browse import check_input_grid, source_array
from tomojax.cli._handles import Root, handle, stage_array
from tomojax.cli._inputs import check_scan_calibration
from tomojax.cli._interactive import correction_product, region_product, slice_product
from tomojax.cli._options import ancestor_paths, check_destinations, checkpointed
from tomojax.cli._plans import plan_product
from tomojax.cli._results import unit_of
from tomojax.cli.products import VOLUME_DATASET
from tomojax.core.array_grid import Grid
from tomojax.core.array_ops import check_binning
from tomojax.errors import InvalidParameter
from tomojax.wire.arrays import Array, Axis, Hdf5, Record, Uniform
from tomojax.wire.progress import Event, Preview

if TYPE_CHECKING:
    from collections.abc import Iterator
    from contextlib import AbstractContextManager
    from os import PathLike

    from tomojax._workflow.evaluation import SplitSettings
    from tomojax.core.json_values import JsonValue

type Work = Callable[[], Root | p.DataProduct | ex.NewHandle]

EXECUTABLE = (
    op.Python,
    op.Inspect,
    op.Open,
    op.Validate,
    op.Plan,
    op.Run,
    op.Read,
    op.Statistics,
    op.Plane,
    op.Slice,
    op.CenterSweep,
    op.CorrectionPreview,
    op.RegionPreview,
)


def execute(command: op.ExecutionInput, context: ex.ExecutionContext) -> ex.Success:
    """Forward watch events and attach provenance from this command's admission."""
    context.control.boundary()
    if not isinstance(command, EXECUTABLE):
        raise ValueError(f"operation unavailable: {command.__struct_config__.tag}")
    operation = (
        command.operation if isinstance(command, op.Run | op.Plan | op.Validate) else command
    )

    def diagnostic(pointer: str, data: object) -> Array:
        a = np.asarray(data)
        axes = tuple(Axis(f"axis{i}", "diagnostic", "index", Uniform(0, 1)) for i in range(a.ndim))
        return Array(stage_array(context, a), a.shape, a.dtype.str, axes, "unspecified", pointer)

    for snapshot in sample():
        context.snapshot(snapshot)
    with progress.watch(
        partial(report_event, context),
        control=cast("progress.Control", context.control),
        previews=isinstance(command, op.Run) and command.previews,
    ):
        if isinstance(operation, op.Python):
            from tomojax.cli._python import run_python

            try:
                return run_python(operation, context)
            except Exception:
                raise
            except BaseException as err:
                raise ValueError(str(err)) from err
        manifest = provenance(command, operation, context)
        if isinstance(operation, op.Plane):
            if not isinstance(command, op.Validate):
                return ex.browse_plane(operation, context)
            ex.plane_source(operation, context)
            item = p.Validated()
        else:
            admission = Admission(
                context,
                manifest,
                isinstance(command, op.Plan),
                isinstance(command, op.Validate),
            )
            work = admission.prepare(operation)
            item = p.Validated(admission.details) if isinstance(command, op.Validate) else work()
        context.control.boundary()
        for snapshot in sample():
            context.snapshot(snapshot)
        info = result_info(item)
        if isinstance(item, p.Files):
            info = admission.details
        summary = ex.summarize(info, externalize=diagnostic)
        if isinstance(item, tj.Scan | tj.Frames | tj.Reconstruction | tj.Alignment | tj.Simulation):
            storage = None
            if isinstance(operation, op.Open) and isinstance(item, tj.Reconstruction):
                storage = saved_storage(operation.path)
            return ex.Success(handle(item, volume_storage=storage), summary, manifest)
        return ex.Success(item, summary, manifest)


def report_event(context: ex.ExecutionContext, record: progress.Event) -> None:
    if isinstance(record, progress.Preview):
        arrays = {
            name: sample.store(lambda a: stage_array(context, a))
            for name, sample in record.arrays.items()
        }
        context.event(Preview(schema=2, task=record.task, time=record.time, arrays=arrays))
    else:
        context.event(msgspec.convert(progress.to_json(record), type=Event))


def result_info(item: Root | p.DataProduct | ex.NewHandle) -> Mapping[str, object]:
    """Geometric registration and solver diagnostics for the public product."""
    info = item.info if isinstance(item, tj.Reconstruction | tj.Alignment) else {}
    if isinstance(item, tj.Alignment):
        poses = np.asarray(item.poses)
        info = {**info, "pose_rms": np.sqrt(np.mean(np.square(poses), axis=0)).tolist()}
    if isinstance(item, tj.Scan | tj.Frames | tj.Simulation | tj.Alignment | tj.Reconstruction):
        info = {**info, "geometry": geometry_summary(item)}
        if calibration := info["geometry"].get("calibration"):
            info["calibration"] = calibration
    return info


def geometry_summary(item: Root, *, measured_angles: bool = True) -> dict[str, Any]:
    scan = item if isinstance(item, tj.Scan | tj.Frames) else item.scan
    geometry = (
        item.scan.geometry if isinstance(item, tj.Alignment | tj.Simulation) else item.geometry
    )
    angles = np.asarray(geometry.angles)
    summary: dict[str, Any] = {
        "kind": describe(geometry)[0],
        "center_uv": list(geometry.detector.center),
        "center_unit": unit_of(item if isinstance(item, tj.Reconstruction) else scan),
        "views": len(angles),
    }
    if measured_angles:
        summary["angle_range_degree"] = [float(angles.min()), float(angles.max())]
    metadata = scan.source.geometry_metadata if scan is not None and scan.source else {}
    # What the reader found: a center or geometry the file does not record, a cut stack.
    acquisition = object_mapping(object_mapping(metadata).get("acquisition"))
    found = {
        "detector_center_status": object_mapping(metadata).get("detector_center_status"),
        "geometry_status": acquisition.get("geometry_status"),
        "frame_records": acquisition.get("frame_records"),
    }
    summary.update({key: value for key, value in found.items() if value is not None})
    info = (
        item.info
        if isinstance(item, tj.Alignment | tj.Reconstruction)
        else metadata.get("alignment_info", {})
    )
    calibration = info.get("calibration", {})
    if isinstance(calibration, dict) and calibration:
        summary["calibration"] = {
            key: {
                **entry,
                "unit": summary["center_unit"] if key == "center" else "degree",
                "change_unit": "pixel" if key == "center" else "degree",
            }
            for key, entry in calibration.items()
        }
    return summary


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
    validating: bool
    details: dict[str, Any] = field(default_factory=dict)

    @singledispatchmethod
    def prepare(self, operation: Record) -> Work:
        raise ValueError(f"operation unavailable: {operation.__struct_config__.tag}")

    @prepare.register
    def feature(self, operation: op.Feature) -> Work:
        from tomojax.cli._features import make_feature

        work = partial(make_feature, operation, self.context, validate_only=self.validating)
        if self.validating:
            _ = work()
        return work

    @prepare.register
    def browse(self, operation: op.Read | op.Statistics) -> Work:
        from tomojax.cli._browse import browse

        return partial(browse, operation, self.context)

    @prepare.register
    def inspect(self, operation: op.Inspect) -> Work:
        return lambda: p.Inspection(dict(tj.inspect(operation.path)))

    @prepare.register
    def open(self, operation: op.Open) -> Work:
        if operation.kind in ("scan", "frames"):
            options = arguments(
                operation, self.context, omit=("path", "kind"), layout_only=self.validating
            )
            declared = options.pop("geometry")
            loader = (
                load_scan_with_declaration
                if operation.kind == "scan"
                else load_raw_frames_with_declaration
            )
            work = partial(loader, operation.path, declared=declared, **options)
            if self.validating:
                item, _ = projection_frames(
                    Path(operation.path),
                    require_raw=operation.kind == "frames",
                    declared=declared,
                    **options,
                )
        else:
            for field in msgspec.structs.fields(operation):
                if (
                    field.name not in ("path", "kind")
                    and getattr(operation, field.name) != field.default
                ):
                    raise InvalidParameter(
                        field.name, f"{field.name} applies only when opening a scan or frames"
                    )
            if operation.kind == "feature":
                from tomojax.cli._features import load_feature

                self.details.update(path=operation.path, kind="feature")
                return partial(load_feature, operation.path, self.context)
            load = {
                "reconstruction": tj.load_reconstruction,
                "alignment": tj.load_alignment,
                "simulation": tj.load_simulation,
            }[operation.kind]
            work = partial(load, operation.path)
            if self.validating:
                from tomojax.io.projection_layout import saved_projection_layout

                record, _ = saved_projection_layout(Path(operation.path))

                def stored_scan(path: str | PathLike[str]) -> tj.Scan:
                    frames, raw = projection_frames(Path(path))
                    return scan_layout(frames, raw=raw)

                build = {
                    "reconstruction": reconstruction_from_record,
                    "alignment": alignment_from_record,
                    "simulation": simulation_from_record,
                }[operation.kind]
                item = build(operation.path, record, stored_scan)
        if self.validating:
            self.details.update(
                path=operation.path,
                kind=operation.kind,
                geometry=geometry_summary(item, measured_angles=operation.angles is None),
            )
        return work

    @prepare.register
    def simulate(self, operation: op.Simulate) -> Work:
        grid, detector, cfg = build_sim_config(**arguments(operation, self.context))
        self.details.update(
            views=cfg.n_views,
            phantom_shape=grid.shape,
            projection_shape=(cfg.n_views, detector.nv, detector.nu),
        )
        return partial(simulate_config, cfg, grid=grid, detector=detector)

    @prepare.register
    def combine(self, operation: op.Combine) -> Work:
        scans = [
            source(
                ref,
                self.context,
                kinds=(tj.Scan,),
                layout_only=self.validating,
                field=f"sources[{i}]",
            )
            for i, ref in enumerate(operation.sources)
        ]
        try:
            combined_geometry(scans)
        except ValueError as err:
            raise InvalidParameter("sources", str(err)) from err
        return lambda: tj.Scan.combine(scans, name=operation.name)

    def check_support(self, config: object, grid: tj.Grid, unit: str) -> None:
        if (ref := getattr(config, "support", None)) is not None:
            with source_array(ref, self.context, field="config.support") as (_, meta):
                check_input_grid(
                    meta, Grid.volume(grid, unit), self.context, field="config.support"
                )

    @prepare.register
    def evaluate(self, operation: op.Evaluate) -> Work:
        from tomojax._workflow.evaluation import (
            local_motion,
            partition_layout,
            partitions,
            score_candidate,
        )

        if isinstance(operation.source, op.PathInput):
            data, raw = projection_frames(Path(operation.source.path))
        else:
            data = source(operation.source, self.context, kinds=(tj.Scan, tj.Frames), raw=None)
            raw = isinstance(data, tj.Frames)
        steps = arguments(
            operation, self.context, omit=("source", "split", "reference", "candidates")
        )["corrections"]
        split = operation.split
        spec: SplitSettings
        if isinstance(split, op.Angular):
            spec = {
                "type": "angular",
                "start": split.start,
                "width": split.width,
                "guard": split.guard,
                "period": split.period,
            }
        else:
            spec = {
                "type": "interleaved",
                "stride": split.stride,
                "offset": split.offset,
                "guard": split.guard,
            }
        if not operation.candidates:
            raise InvalidParameter("candidates", "supply at least one setting")
        if operation.reference.grid is not None or operation.reference.region is not None:
            raise InvalidParameter("reference", "alignment reference uses the scan's default grid")
        try:
            domain = (
                partition_layout(data, spec, steps, raw=raw)
                if self.validating
                else partitions(data, spec, steps, raw=raw)
            )
        except InvalidParameter:
            raise
        except ValueError as err:
            raise InvalidParameter("split", str(err)) from err
        reference = partial(arguments, operation.reference, self.context)
        train = domain[1].scan
        for setting in operation.candidates:
            config = setting.config
            settings = arguments(setting, self.context, layout_only=True)
            if isinstance(setting, op.AlignSettings):
                local_motion(settings)
                resolve_alignment(train, **settings)
                config = operation.reference.config
                settings = reference(layout_only=True)
            resolved = resolve_reconstruction(train, metadata_only=self.validating, **settings)
            self.check_support(config, resolved.scan.grid, unit_of(train))
        self.details.update(
            split_fingerprint=domain[0].fingerprint,
            training_views=len(domain[1].indices),
            test_views=len(domain[2].indices),
        )
        return lambda: p.Evaluation(
            domain[0].fingerprint,
            msgspec.to_builtins(operation.reference),
            tuple(
                score_candidate(
                    domain,
                    setting.__struct_config__.tag,
                    arguments(setting, self.context),
                    msgspec.to_builtins(setting),
                    reference,
                )
                for setting in operation.candidates
            ),
        )

    @prepare.register
    def fit(self, operation: op.Reconstruct | op.Slice | op.Align) -> Work:
        layout_only = self.validating or self.plan
        scan = source(operation.source, self.context, kinds=(tj.Scan,), layout_only=layout_only)
        options = arguments(
            operation,
            self.context,
            omit=("source", "axis", "index", "centers", "checkpoint"),
            layout_only=layout_only,
        )
        checkpoint = None if isinstance(operation, op.Slice) else operation.checkpoint
        if not isinstance(operation, op.Align):
            options["metadata_only"] = layout_only
        if isinstance(operation, op.Align):
            alignment = resolve_alignment(scan, **options)

            def apply_alignment() -> tj.Alignment:
                with alignment_task(operation.method, operation.fit):
                    return align_resolved(scan, alignment, checkpoint=checkpoint)

            compute = apply_alignment
            work = alignment
        elif isinstance(operation, op.Slice):
            if isinstance(operation, op.CenterSweep):
                plans = interactive.resolve_sweep(
                    scan, operation.axis, operation.index, operation.centers, **options
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
            compute = partial(reconstruct_resolved, work, checkpoint=checkpoint)
        if isinstance(work, ResolvedReconstruction):
            self.check_support(operation.config, work.scan.grid, unit_of(scan))
        snapshots = sample(work)
        for snapshot in snapshots:
            self.context.snapshot(snapshot)
        if self.plan:
            return partial(plan_product, scan, work, snapshot=snapshots)
        if checkpoint is None:
            return compute
        return partial(checkpointed, checkpoint, compute, request=self.request())

    def request(self) -> JsonValue:
        """How a checkpointed run was requested: the operation and its inputs' provenance."""
        return {"operation": msgspec.to_builtins(self.manifest)}

    @prepare.register
    def region(self, operation: op.RegionPreview) -> Work:
        from tomojax._workflow.region import check_region_geometry

        scan = source(operation.source, self.context, kinds=(tj.Scan,), layout_only=self.validating)
        check_region_geometry(scan)
        return partial(region_product, scan, self.context)

    @prepare.register
    def project(self, operation: op.Project | op.Backproject) -> Work:
        from tomojax.cli._operators import project_work

        return project_work(operation, self.context, validating=self.validating)

    @prepare.register
    def edit(self, operation: op.Select | op.Crop | op.Bin | op.Poses) -> Work:
        from tomojax._scan.model import detector_window
        from tomojax.core.validation import view_indices

        kinds = (tj.Scan, tj.Frames) if isinstance(operation, op.Select | op.Crop) else (tj.Scan,)
        item = source(operation.source, self.context, kinds=kinds, layout_only=self.validating)
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
        poses = arguments(operation, self.context, omit=("source",), layout_only=self.validating)[
            "poses"
        ]
        try:
            posed = item.with_poses(poses)
        except ValueError as err:
            raise InvalidParameter("poses", str(err)) from err
        return lambda: posed

    @prepare.register
    def slabs(self, operation: op.ReconstructSlabs) -> Work:
        from tomojax._scan.model import geometry_record
        from tomojax._workflow.operations import slab_settings
        from tomojax.io.api import dataset_output_format, slab_volume
        from tomojax.recon.api import validate_slab_grid

        item: tj.Frames = source(
            operation.source,
            self.context,
            kinds=(tj.Frames,),
            raw=True,
            layout_only=self.validating,
        )
        settings = arguments(
            operation,
            self.context,
            omit=("source", "path", "force", "checkpoint"),
            layout_only=self.validating,
        )
        grid, cfg, _ = slab_settings(item, **settings)
        self.check_support(operation.config, grid, unit_of(item))
        validate_slab_grid(cfg, item.geometry, grid, item.geometry.detector)
        path = Path(operation.path)
        if dataset_output_format(path) != "hdf5":
            raise InvalidParameter("path", "slab reconstruction requires a NeXus/HDF5 destination")
        sidecar = path.with_name(path.name + ".manifest.json")
        inputs = set(ancestor_paths((self.manifest,)))
        check_destinations(inputs, (path, sidecar), force=operation.force)

        def fill() -> p.Files:
            record = geometry_record(item.geometry, name=item.name, grid=grid, source=item.source)
            record.geometry_metadata.pop("volume_role", None)
            record.geometry_metadata["reconstruction_method"] = operation.method
            metadata = record.to_nxtomo_metadata()
            info: dict[str, object] = {"source": item.source.source_path if item.source else None}
            stage = self.context.staging.output(path, force=operation.force)
            manifest = self.context.staging.output(sidecar, force=operation.force)
            with slab_volume(stage, metadata, grid.shape) as volume:
                volume.attrs["long_name"] = "volume"
                write_slabs(
                    partial(tj.reconstruct_slabs, item, info=info, **settings),
                    volume,
                    None if operation.checkpoint is None else Path(operation.checkpoint),
                    self.request(),
                )
                assert metadata.geometry_meta is not None
                metadata.geometry_meta["reconstruction_info"] = saved_info(info)
                self.details.update(info)
            manifest.write_bytes(msgspec.json.encode(self.manifest) + b"\n")
            return p.Files((str(path), str(sidecar)))

        return fill

    @prepare.register
    def correct(self, operation: op.Correct) -> Work:
        from tomojax.corrections.api import (
            check_rows,
            check_scan_steps,
            positive_finite,
            validated_views_per_batch,
        )

        item = source(
            operation.source,
            self.context,
            kinds=(tj.Frames, tj.Scan),
            raw=None,
            layout_only=self.validating,
        )
        settings = arguments(operation, self.context, omit=("source", "rows"))
        steps = settings.pop("corrections")
        validated_views_per_batch(operation.views_per_batch)
        if isinstance(item, tj.Scan) and isinstance(operation, op.CorrectionPreview):
            # A scan's band bins after its line-integral steps, as Scan.binned does.
            check_scan_steps(steps)
            check_scan_calibration(operation.epsilon, operation.flat_weights)
            check_binning(operation.binning, *item.projections.shape[1:])
            check_rows(operation.rows, item.projections.shape[1], operation.binning)
            settings.pop("epsilon", None)
            settings.pop("flat_weights", None)
            return partial(correction_product, operation, item, steps, settings, self.context)
        if isinstance(item, tj.Scan):
            check_scan_steps(steps)
            check_scan_calibration(operation.epsilon, operation.flat_weights)
            if operation.binning != 1:
                raise InvalidParameter("binning", "correction binning applies to raw frames")
            return partial(item.corrected, *steps, views_per_batch=operation.views_per_batch)
        positive_finite(operation.epsilon, "epsilon")
        check_binning(operation.binning, *item.counts.shape[1:])
        if isinstance(operation, op.CorrectionPreview):
            check_rows(operation.rows, item.counts.shape[1], operation.binning)
            return partial(correction_product, operation, item, steps, settings, self.context)
        return partial(item.corrected, *steps, **settings)

    @prepare.register
    def save(self, operation: op.Save) -> Work:
        held = self.context.leases[operation.source].value
        path = Path(operation.path)
        if isinstance(held, ex.Feature):
            from tomojax.cli._features import SAVED_SUFFIXES, save_feature

            if path.suffix.lower() not in SAVED_SUFFIXES:
                raise InvalidParameter("path", f"save a feature as {', '.join(SAVED_SUFFIXES)}")
            feature = held
        else:
            value = source(
                operation.source,
                self.context,
                kinds=(tj.Scan, tj.Reconstruction, tj.Alignment, tj.Simulation),
            )
            validate_save(path, value, include_scan=operation.include_scan)
            feature = None
        paths = self.durable_paths(path, [path], force=operation.force)

        def write() -> p.Files:
            stage = self.context.staging.output(path, force=operation.force)
            if feature is not None:
                save_feature(stage, feature)
            else:
                tj.save(stage, value, include_scan=operation.include_scan)
            return self.write_manifest(paths, force=operation.force)

        return write

    def durable_paths(self, path: Path, outputs: list[Path], *, force: bool) -> list[Path]:
        if not path.name or path.name == "..":
            raise InvalidParameter("path", "expected an output file or directory name")
        paths = [*outputs, path.with_name(path.name + ".manifest.json")]
        self.details.update(
            paths=[str(p) for p in paths], destination_exists=any(p.exists() for p in paths)
        )
        check_destinations(
            set(ancestor_paths((self.manifest,))), paths, force=force or self.validating
        )
        return paths

    def write_manifest(self, paths: list[Path], *, force: bool) -> p.Files:
        stage = self.context.staging.output(paths[-1], force=force)
        stage.write_bytes(msgspec.json.encode(self.manifest) + b"\n")
        return p.Files(tuple(str(p) for p in paths))

    @prepare.register
    def export(self, operation: op.Export) -> Work:
        from tomojax._loading.files import volume_source
        from tomojax.io.api import ExportSource, plan_export

        lease = self.context.leases[operation.source]
        held = lease.value
        if isinstance(held, ex.RegisteredArray):
            if operation.source.member not in ("volume", "image", "phantom", "mask"):
                raise InvalidParameter("source", "choose a three-dimensional volume or mask")
            held = lease.owner
        if isinstance(held, ex.Feature):
            member = ExportSource(
                held.name,
                held.shape,
                held.dtype,
                held.registration.axes,
                "1",
                "mask",
                held.__getitem__,
                {"recipe": held.recipe},
            )
        elif isinstance(held, tj.Reconstruction | tj.Simulation | tj.Alignment):
            member = volume_source(held)
        else:
            raise InvalidParameter("source", "choose a volume or mask to export")
        member.record["provenance"] = {
            **object_mapping(member.record.get("provenance")),
            "lineage": msgspec.to_builtins(self.manifest),
        }
        path = Path(operation.path)
        plan = plan_export(
            path,
            member,
            dtype=operation.dtype,
            value_range=operation.range,
            crop=operation.crop,
            binning=operation.binning,
            axis=operation.axis,
            compress=operation.compress,
        )
        paths = self.durable_paths(path, list(plan.format.paths(path)), force=operation.force)
        self.details.update(plan.details)

        def write() -> p.Files:
            self.details.update(self.context.staging.export(plan, force=operation.force))
            return self.write_manifest(paths, force=operation.force)

        return write


def write_slabs(
    slabs: Callable[..., AbstractContextManager[Iterator[tuple[slice, tj.Reconstruction]]]],
    volume: h5py.Dataset,
    checkpoint: Path | None,
    request: JsonValue,
) -> None:
    """Fill ``volume`` from ``slabs(checkpoint=...)``, keeping finished slabs in ``checkpoint``.

    The staged output does not survive a kill, so with a checkpoint the finished
    slabs go to ``volume.npy`` in it, which a resumed run reopens; the slabs after
    the last finished one are made and the whole volume is copied out at the end.
    The checkpoint's own records go first when it completes, so a directory left
    holding only ``volume.npy`` is refused rather than mistaken for a finished run.
    """
    from tomojax.core.checkpoint import opened

    kept = None if checkpoint is None else checkpoint / "volume.npy"
    with nullcontext() if checkpoint is None else opened(checkpoint, request=request):
        with slabs(checkpoint=checkpoint) as finished:
            target = (
                volume
                if kept is None
                else np.lib.format.open_memmap(
                    kept, mode="r+" if kept.exists() else "w+", dtype=np.float32, shape=volume.shape
                )
            )
            for interval, slab in finished:
                target[interval] = slab.volume
                if isinstance(target, np.memmap):
                    target.flush()
                del slab
                progress.check_canceled()  # a pause holds here; a cancel waits for the save
        if isinstance(target, np.memmap):
            for start in range(0, len(target), 16):
                volume[start : start + 16] = target[start : start + 16]
            del target
    if kept is not None and checkpoint is not None:
        kept.unlink()
        with suppress(OSError):
            checkpoint.rmdir()


def provenance(
    command: op.ExecutionInput, operation: Record, context: ex.ExecutionContext
) -> p.Provenance:
    parameters = msgspec.to_builtins(
        command if isinstance(command, op.Validate | op.Plan) else operation
    )
    return p.Provenance(parameters.pop("type"), parameters, context.lineage(command))
