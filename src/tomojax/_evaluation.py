"""Fit and predict on disjoint acquisition partitions."""

from dataclasses import asdict, dataclass, replace

import numpy as np

import tomojax as tj
from tomojax._planning import resolve_reconstruction
from tomojax._scan import with_grid
from tomojax._smooth_alignment import transfer_alignment
from tomojax._workflow import reconstruct_resolved
from tomojax.core.geometry.cone import segments_of
from tomojax.errors import InvalidParameter
from tomojax.evaluation import Score, Split
from tomojax.recon.api import predict_projections


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


def partitions(data, kind, settings, steps):
    split, segments = native_split(data, kind, settings)

    def partition(mask):
        ids = np.flatnonzero(mask)
        selected = data.selected(ids)
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

    train_part, test_part = partition(split.train.ravel()), partition(split.test.ravel())
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
        settings = reference
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


def evaluate(domain, candidates, reference):
    split, train, test = domain
    rows = []
    for kind, settings, description in candidates:
        recon, held, config, diagnostics = fit_candidate(train, test, kind, settings, reference)
        predicted = np.asarray(
            predict_projections(held.geometry, recon.volume, grid=recon.grid, config=config)
        )
        score: Score = sum(
            (
                split.score(
                    predicted[i : i + 1],
                    test.scan.projections[i : i + 1],
                    views=slice(int(k), int(k) + 1),
                )
                for i, k in enumerate(test.indices)
            ),
            start=Score(0, 0, 0, 0),
        )
        rows.append(
            dict(
                candidate=description,
                **asdict(score),
                mse=score.mse,
                relative_l2=score.relative_l2,
                diagnostic=diagnostics,
            )
        )
    return tuple(rows)
