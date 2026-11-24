"""Fit and predict on disjoint acquisition partitions."""

from dataclasses import asdict, dataclass, replace
import math

import numpy as np

import tomojax as tj
from tomojax import progress
from tomojax._data import views_of
from tomojax._frame_sources import selected_frames
from tomojax._planning import resolve_reconstruction
from tomojax._scan import bind_step_geometry, frame_output_grid, selected_scan, with_grid
from tomojax._smooth_alignment import transfer_alignment
from tomojax._workflow import reconstruct_resolved
from tomojax.core.geometry.cone import segments_of
from tomojax.core.host_volume import host_array, release_planes, resident_limit
from tomojax.errors import InvalidParameter
from tomojax.evaluation import Score, Split
from tomojax.recon import api
from tomojax.recon.planning import prediction_block_views


@dataclass
class Partition:
    scan: tj.Scan
    indices: np.ndarray
    segments: np.ndarray


def native_split(data, kind, settings):
    shape = (data.counts if isinstance(data, tj.Frames) else data.projections).shape
    n = shape[0]
    positions = np.arange(n) if data.view_positions is None else np.asarray(data.view_positions)
    starts = np.r_[True, np.diff(positions) <= 0]
    if parts := segments_of(data.geometry):
        starts[np.cumsum(parts.trajectory_lengths)[:-1]] = True
    segments = np.cumsum(starts)
    guard = settings["guard"]
    if kind == "angular":
        split = Split.angular(shape, np.asarray(data.geometry.angles), **settings)
    else:
        if not 0 <= guard < settings["stride"] / 2:
            raise InvalidParameter("split.guard", "guards must leave training views")
        split = Split.interleaved(shape, stride=settings["stride"], offset=settings["offset"])
    train, test = split.train.ravel().copy(), split.test.ravel().copy()
    for group in np.unique(segments):
        ids = np.flatnonzero(segments == group)
        if kind == "interleaved":
            for k in range(1, guard + 1):
                train[ids[k:]] &= ~test[ids[:-k]]
                train[ids[:-k]] &= ~test[ids[k:]]
            support = ids[train[ids]]
            test[ids] &= (ids >= support.min()) & (ids <= support.max()) if support.size else False
        elif test[ids].any():
            support = ids[train[ids]]
            if (
                not support.size
                or min(ids[test[ids]]) < min(support)
                or max(ids[test[ids]]) > max(support)
            ):
                raise InvalidParameter(
                    "split", "angular arc must lie inside its acquisition segment"
                )
    if not train.any() or not test.any():
        raise InvalidParameter(
            "split", "guards and acquisition boundaries leave no bracketed test or training views"
        )
    split = Split(
        shape=shape,
        train=train[:, None, None],
        test=test[:, None, None],
        specification=split.specification,
    )

    return split, segments


_PARTITION_BYTES = 32 << 20


def _select_views(data, ids, *, raw):
    source = data.counts if isinstance(data, tj.Frames) else data.projections
    if raw and hasattr(source, "frames_at") and not hasattr(source, "copy_views"):
        return data.selected(ids)
    target = host_array((len(ids), *source.shape[1:]), dtype=source.dtype)
    if hasattr(source, "copy_views"):
        source.copy_views(ids, target)
    else:
        batch = max(1, _PARTITION_BYTES // (math.prod(source.shape[1:]) * source.dtype.itemsize))
        for first in range(0, len(ids), batch):
            progress.check_cancelled()
            group = ids[first : first + batch]
            if hasattr(source, "frames_at"):
                target[first : first + len(group)] = source[group]
            else:
                for row, view in enumerate(group, first):
                    target[row] = source[int(view)]
            if isinstance(source, np.ndarray):
                release_planes(source, int(group[0]), int(group[-1]) + 1)
            release_planes(target, first, first + len(group))
    if isinstance(data, tj.Scan):
        return selected_scan(data, ids, target)
    fields = selected_frames(data, ids, target)
    if raw:
        return fields
    return tj.Scan(
        target,
        fields.geometry,
        fields.name,
        fields.corrections,
        fields.source,
        fields.view_positions,
    )


def partition_layout(data, kind, settings, steps, *, raw):
    """Validate the native split before any data-dependent rejection or calibration."""
    from tomojax.corrections.api import check_scan_steps
    from tomojax.corrections.calibration import check_calibration_layout

    if raw:
        check_calibration_layout(data)
    else:
        check_scan_steps(steps)
    split, segments = native_split(data, kind, settings)
    positions = (
        np.arange(len(segments)) + 0.5 if data.view_positions is None else data.view_positions
    )

    samples = data.counts if isinstance(data, tj.Frames) else data.projections
    dtype = np.float32 if raw or steps else samples.dtype

    def layout(mask):
        ids = np.flatnonzero(mask)
        geometry = views_of(data.geometry, ids)
        bind_step_geometry(tuple(step.retained(ids) for step in steps), geometry)
        shape = (len(ids), *split.shape[1:])
        scan = tj.Scan(
            np.broadcast_to(np.zeros((), dtype), shape),
            frame_output_grid(geometry, data.source) if raw else geometry,
            data.name,
            data.corrections,
            data.source,
            positions[ids],
        )
        return Partition(scan, ids, segments[ids])

    return split, layout(split.train), layout(split.test)


def partitions(data, kind, settings, steps, *, raw):
    split, segments = native_split(data, kind, settings)

    def partition(mask):
        ids = np.flatnonzero(mask)
        selected = _select_views(data, ids, raw=raw)
        scan = (
            selected.corrected(*(step.retained(ids) for step in steps))
            if steps or isinstance(selected, tj.Frames)
            else selected
        )
        kept = np.ones(len(ids), bool)
        for record in scan.corrections[len(selected.corrections) :]:
            kept[np.asarray(record.found.get("rejected", []), dtype=np.intp)] = False
        ids = ids[kept]
        return Partition(scan, ids, segments[ids])

    token = resident_limit.set(_PARTITION_BYTES)
    try:
        train_part, test_part = partition(split.train.ravel()), partition(split.test.ravel())
    finally:
        resident_limit.reset(token)
    for group in np.unique(test_part.segments):
        fit = train_part.indices[train_part.segments == group]
        held = test_part.indices[test_part.segments == group]
        if not fit.size or min(held) < min(fit) or max(held) > max(fit):
            raise InvalidParameter(
                "split",
                "retained training views must bracket test views within each acquisition segment",
            )
    return split, train_part, test_part


def local_motion(settings):
    """Only global and smooth fits define poses for unseen views."""
    selection = settings["fit"]
    local = bool(
        {"rotations", "translations"}.intersection(
            (selection,) if isinstance(selection, str) else selection
        )
    )
    if (
        local
        and (settings["motion"] or getattr(settings["config"], "motion", "independent"))
        == "independent"
    ):
        raise InvalidParameter(
            "motion",
            "Independent per-view poses cannot predict unseen views. "
            "Use detector splits through python with a weighted solver. "
            "tj.align takes no observation weights.",
        )
    return local


def fit_candidate(train, test, kind, settings, reference):
    held = test.scan
    diagnostic = {}
    scan = train.scan
    if kind == "align":
        local = local_motion(settings)
        positions = scan.view_positions.copy()
        for group in np.unique(train.segments):
            members = train.segments == group
            positions[members] -= positions[members][0]
        fit = tj.align(replace(scan, view_positions=positions), **settings)
        held = transfer_alignment(
            train.scan,
            test.scan,
            fit,
            training_segments=train.segments,
            held_segments=test.segments,
            local=local,
        )
        report = fit.info.get("report")
        diagnostic = {
            "fit_method": fit.info["method"],
            "calibration": fit.info["calibration"],
            "fit_levels": report.get("levels", []) if isinstance(report, dict) else [],
        }
        scan = with_grid(fit.scan, train.scan.grid)
        del fit
        settings = reference()
    plan = resolve_reconstruction(scan, **settings)
    recon = reconstruct_resolved(plan)
    return (
        recon,
        held,
        plan.config,
        {
            "method": recon.method,
            "iterations": recon.info.get("effective_iterations"),
            **diagnostic,
        },
    )


_PREDICTION_BYTES = 32 << 20


def prediction_score(split, test, held, recon, config):
    """Project one bounded batch at a time, summing views in acquisition order."""
    detector = held.detector
    batch = max(1, _PREDICTION_BYTES // (detector.nv * detector.nu * np.dtype(np.float32).itemsize))
    views = len(test.indices)
    # XLA may round the same rays differently at a different vector width.
    # Keep the native width when it fits, shifting the tail like the one-shot forward.
    native = prediction_block_views(held.geometry, config) if batch < views else None
    batch = min(native, batch) if native else batch
    score = Score(0, 0, 0, 0)
    for first in range(0, views, batch):
        progress.check_cancelled()
        stop = min(views, first + batch)
        start = max(0, stop - batch) if native else first
        geometry = views_of(held.geometry, np.arange(start, stop))
        predicted = np.asarray(
            api.predict_projections(geometry, recon.volume, grid=recon.grid, config=config)
        )
        for i in range(first, stop):
            row = i - start
            k = int(test.indices[i])
            score += split.score(
                predicted[row : row + 1],
                test.scan.projections[i : i + 1],
                views=slice(k, k + 1),
            )
        release_planes(test.scan.projections, first, stop)
        del predicted
    return score


def score_candidate(domain, kind, settings, description, reference):
    split, train, test = domain
    recon, held, config, diagnostics = fit_candidate(train, test, kind, settings, reference)
    score = prediction_score(split, test, held, recon, config)
    return dict(
        candidate=description,
        **asdict(score),
        mse=score.mse,
        relative_l2=score.relative_l2,
        diagnostic=diagnostics,
    )
