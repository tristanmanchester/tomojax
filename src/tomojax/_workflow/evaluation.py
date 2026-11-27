"""Fit and predict on disjoint acquisition partitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import gc
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

import tomojax as tj
from tomojax import progress
from tomojax._data import views_of
from tomojax._loading.frame_sources import selected_frames
from tomojax._parameters import method_fit
from tomojax._scan.model import bind_step_geometry, frame_output_grid, selected_scan, with_grid
from tomojax._workflow.operations import reconstruct_resolved
from tomojax._workflow.planning import resolve_reconstruction
from tomojax._workflow.smooth_alignment import (
    acquisition_segments,
    trajectory_positions,
    transfer_alignment,
)
from tomojax.core.host_volume import release_planes, resident_limit, selected_stack
from tomojax.errors import InvalidParameter
from tomojax.evaluation import Score, Split
from tomojax.recon import api
from tomojax.recon.planning import prediction_block_views

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Literal

    from tomojax.corrections import Step
    from tomojax.recon.api import MethodConfig

    class _AngularSplit(TypedDict):
        type: Literal["angular"]
        start: float
        width: float
        guard: float
        period: float

    class _InterleavedSplit(TypedDict):
        type: Literal["interleaved"]
        stride: int
        offset: int
        guard: int

    type SplitSettings = _AngularSplit | _InterleavedSplit


# msgspec and the worker's argument resolver produce dynamic keyword dictionaries.
type WireValue = Any


@dataclass
class Partition:
    scan: tj.Scan
    indices: np.ndarray
    segments: np.ndarray


def native_split(
    data: tj.Scan | tj.Frames,
    settings: SplitSettings,
) -> tuple[Split, np.ndarray]:
    n, rows, cols = (data.counts if isinstance(data, tj.Frames) else data.projections).shape
    shape = (n, rows, cols)
    segments = acquisition_segments(data.geometry, data.view_positions)
    interleaved_guard = None
    if settings["type"] == "angular":
        split = Split.angular(
            shape,
            np.asarray(data.geometry.angles),
            start=settings["start"],
            width=settings["width"],
            guard=settings["guard"],
            period=settings["period"],
        )
    else:
        interleaved_guard = settings["guard"]
        if not 0 <= interleaved_guard < settings["stride"] / 2:
            raise InvalidParameter("split.guard", "guards must leave training views")
        split = Split.interleaved(shape, stride=settings["stride"], offset=settings["offset"])
    train, test = split.train.ravel().copy(), split.test.ravel().copy()
    for group in np.unique(segments):
        ids = np.flatnonzero(segments == group)
        if interleaved_guard is not None:
            for k in range(1, interleaved_guard + 1):
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


def _select_views(data: tj.Scan | tj.Frames, ids: np.ndarray, *, raw: bool) -> tj.Scan | tj.Frames:
    source = data.counts if isinstance(data, tj.Frames) else data.projections
    if raw and hasattr(source, "frames_at"):
        return data.selected(ids)
    target = selected_stack(source, ids, batch_bytes=_PARTITION_BYTES)
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


def partition_layout(
    data: tj.Scan | tj.Frames,
    settings: SplitSettings,
    steps: Sequence[Step],
    *,
    raw: bool,
) -> tuple[Split, Partition, Partition]:
    """Validate the native split before any data-dependent rejection or calibration."""
    from tomojax.corrections.api import check_scan_steps
    from tomojax.corrections.calibration import check_calibration_layout

    if raw:
        check_calibration_layout(data)
    else:
        check_scan_steps(steps)
    split, segments = native_split(data, settings)
    positions = (
        np.arange(len(segments)) + 0.5 if data.view_positions is None else data.view_positions
    )

    samples = data.counts if isinstance(data, tj.Frames) else data.projections
    dtype = np.float32 if raw or steps else samples.dtype

    def layout(mask: np.ndarray) -> Partition:
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


def partitions(
    data: tj.Scan | tj.Frames,
    settings: SplitSettings,
    steps: Sequence[Step],
    *,
    raw: bool,
) -> tuple[Split, Partition, Partition]:
    split, segments = native_split(data, settings)

    def partition(mask: np.ndarray) -> Partition:
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


def local_motion(settings: dict[str, WireValue]) -> bool:
    """Only global and smooth fits define poses for unseen views."""
    fit = method_fit(settings["method"], settings["fit"])
    local = bool({"rotations", "translations"}.intersection(fit))
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


def fit_candidate(
    train: Partition,
    test: Partition,
    kind: str | int | None,
    settings: dict[str, WireValue],
    reference: Callable[[], dict[str, WireValue]],
) -> tuple[tj.Reconstruction, tj.Scan, MethodConfig, dict[str, object]]:
    held = test.scan
    diagnostic = {}
    scan = train.scan
    if kind == "align":
        local = local_motion(settings)
        if scan.view_positions is None:
            raise ValueError(
                "alignment evaluation requires acquisition positions in the training partition"
            )
        positions = trajectory_positions(scan.view_positions, train.segments)
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


def prediction_score(
    split: Split, test: Partition, held: tj.Scan, recon: tj.Reconstruction, config: MethodConfig
) -> Score:
    """Project one bounded batch at a time, summing views in acquisition order."""
    measured = test.scan.projections
    detector = held.detector
    batch = max(1, _PREDICTION_BYTES // (detector.nv * detector.nu * np.dtype(np.float32).itemsize))
    views = len(test.indices)
    # XLA may round the same rays differently at a different vector width.
    # Keep the native width when it fits, shifting the tail like the one-shot forward.
    native = prediction_block_views(held.geometry, config) if batch < views else None
    batch = min(native, batch) if native else batch
    score = Score(0, 0, 0, 0)
    for first in range(0, views, batch):
        progress.check_canceled()
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
                np.asarray(measured[i : i + 1]),
                views=slice(k, k + 1),
            )
        if isinstance(measured, np.ndarray):
            release_planes(measured, first, stop)
        del predicted
    return score


def score_candidate(
    domain: tuple[Split, Partition, Partition],
    kind: str | int | None,
    settings: dict[str, WireValue],
    description: dict[str, object],
    reference: Callable[[], dict[str, WireValue]],
) -> dict[str, object]:
    """Fit and score one candidate, releasing its volume before the next one starts.

    Collecting cyclic garbage here frees device buffers that a solver's internal
    reference cycles would otherwise hold until an arbitrary later collection.
    """
    split, train, test = domain
    recon, held, config, diagnostics = fit_candidate(train, test, kind, settings, reference)
    score = prediction_score(split, test, held, recon, config)
    del recon, held
    gc.collect()
    return dict(
        candidate=description,
        **asdict(score),
        mse=score.mse,
        relative_l2=score.relative_l2,
        diagnostic=diagnostics,
    )
