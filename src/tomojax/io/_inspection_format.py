"""Terminal formatting for NXtomo/HDF5 inspection reports."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tomojax.io._inspection_types import (
        AlignmentReport,
        AnglesReport,
        DetectorMetadataReport,
        FlatsDarksReport,
        Inspection,
        LoadingReport,
        MemoryEstimatesReport,
        ProjectionReport,
        VolumeReport,
    )


def _fmt_value(value: object, *, precision: int = 6) -> str:
    if value is None:
        return "not found"
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def format_inspection_report(report: Inspection) -> str:
    """Format an inspection report for terminal output.

    Sections hold ``key: value`` lines, so the text is never cut or wrapped.

    Parameters
    ----------
    report : Inspection
        Structured dataset inspection report, as :func:`tomojax.inspect` returns it.

    Returns
    -------
    str
        Human-readable multiline summary for terminal output.
    """
    if (saved := report["checkpoint"]) is not None:  # a stopped run's saved state
        keys = ("path", "run", "request", "created", "saved", "entries", "bytes", "in_use")
        rows = [f"  {key + ':':<9} {json.dumps(saved.get(key))}" for key in keys]
        return "\n".join(["Checkpoint", *rows, f"  {saved['resume']}"])
    if (record := report.get("record")) and record.get("kind") == "export":
        lines = [f"Export {report['input_path']}"]
        for name, array in report.get("contents", {}).items():
            lines.append(f"  {name}  {array['shape']}  {array['dtype']}  {array['unit']}")
        return "\n".join(lines)
    corrections = report["corrections"]
    geometry = report["geometry"]
    sections = [
        ("File", [("path", report["input_path"])]),
        ("Projections", _projection_rows(report["projection"])),
        ("Angles", _angle_rows(report["angles"])),
        (
            "Geometry and detector",
            [
                ("geometry", _fmt_value(geometry["type"])),
                ("recorded geometry", "present" if geometry["meta_found"] else "not found"),
                *_detector_rows(report["detector_metadata"]),
            ],
        ),
        ("Calibration", [("flats and darks", _flats_darks_text(report["flats_darks"]))]),
        ("Corrections", [("applied", ", ".join(corrections) if corrections else "none recorded")]),
        ("Alignment", [("saved poses", _alignment_text(report["alignment"]))]),
        ("Memory", _memory_rows(report["memory_estimates"])),
        ("Volume", [("saved volume", _volume_text(report["volume"]))]),
    ]
    if report["loading"] is not None:
        sections.insert(1, ("Loading", _loading_rows(report["loading"])))
    lines: list[str] = []
    for title, rows in sections:
        width = max(len(key) for key, _ in rows)
        lines.append(title)
        lines.extend(f"  {key + ':':<{width + 1}} {value}" for key, value in rows)
    return "\n".join(lines)


def _loading_rows(loading: LoadingReport) -> list[tuple[str, str]]:
    if loading["message"] is not None:
        return [("as a scan", f"needs more: {loading['message']}")]
    rows = [("as a scan", "loads with no further arguments")]
    if loading.get("full_turn"):
        text = "full turn"
        near, far = loading.get("axis_to_edge_px"), loading.get("field_of_view_px")
        overlap = loading.get("overlap_px")
        gap = loading.get("central_gap_px")
        if gap is not None and near is not None and far is not None:
            text += (
                f", axis {-near:.2f} px outside the panel, outer diameter {far:.2f} px, "
                f"unmeasured central disk {gap:.2f} px across"
            )
        elif near is not None and far is not None and overlap is not None:
            text += (
                f", axis {near:.2f} px from the nearer edge, "
                f"field {far:.2f} px, overlap {overlap:.2f} px"
            )
        rows.append(("parallel acquisition", text))
    if loading["geometry"] == "not_recorded":
        rows.append(("geometry", "not recorded: declare geometry to reconstruct or align"))
    if loading["center"] == "not_recorded":
        rows.append(("rotation center", "not recorded: the detector middle until fitted"))
    if loading["length_unit"] == "pixel":
        rows.append(("lengths", "in detector pixels (no pixel size recorded)"))
    if records := loading["frame_records"]:
        counts = ", ".join(f"{count} {name}" for name, count in records.items())
        rows.append(("stopped early", f"{counts}: reads the first {min(records.values())} frames"))
    return rows


def _projection_rows(projection: ProjectionReport) -> list[tuple[str, str]]:
    if not projection["found"]:
        return [("projections", "not found")]
    stats = projection["stats"]
    nonfinite = projection["nonfinite"]
    detector = projection["detector_shape"]
    views, rows, columns = (projection["shape"] or [None, None, None])[:3]
    shape = f"{views} views of {rows} x {columns} pixels" if detector else str(projection["shape"])
    values = ", ".join(
        f"{name} {_fmt_value(value)}"
        for name, value in (
            ("min", stats["min"]),
            ("1%", stats["p01"]),
            ("median", stats["p50"]),
            ("99%", stats["p99"]),
        )
    )
    values += f", max {_fmt_value(stats['max'])}, mean {_fmt_value(stats['mean'])}"
    counts = (nonfinite["nan_count"], nonfinite["posinf_count"], nonfinite["neginf_count"])
    bad = "none" if not any(counts) else "NaN {}, +inf {}, -inf {}".format(*counts)
    return [
        ("shape", shape),
        ("dtype", str(projection["dtype"])),
        ("values", values),
        ("NaN and inf", bad),
    ]


def _angle_rows(angles: AnglesReport) -> list[tuple[str, str]]:
    if not angles["found"]:
        return [("angles", "not found")]
    return [
        ("coverage", f"{_fmt_value(angles['coverage'])} degrees"),
        (
            "range",
            f"{_fmt_value(angles['min'])} to {_fmt_value(angles['max'])}, "
            f"{angles['count']} angles ({_fmt_value(angles['units'])})",
        ),
    ]


def _detector_rows(detector: DetectorMetadataReport) -> list[tuple[str, str]]:
    if not detector["found"]:
        return [("detector", "not found")]
    center = detector["center"]
    center_text = (
        ", ".join(_fmt_value(c) for c in center)
        if isinstance(center, list | tuple)
        else str(center)
    )
    return [
        ("detector", f"{detector['nu']} x {detector['nv']} pixels (u x v)"),
        ("pixel size", f"{_fmt_value(detector['du'])} x {_fmt_value(detector['dv'])}"),
        ("center", f"({center_text})"),
    ]


def _flats_darks_text(flats_darks: FlatsDarksReport) -> str:
    if flats_darks["flats_present"] or flats_darks["darks_present"]:
        return (
            f"{flats_darks['flat_count']} flats, {flats_darks['dark_count']} darks, "
            f"{flats_darks['sample_count']} sample frames "
            f"(image_key {_fmt_value(flats_darks['image_key_path'])})"
        )
    if flats_darks["image_key_found"]:
        return f"none (image_key {_fmt_value(flats_darks['image_key_path'])} labels none)"
    return "not found"


def _alignment_text(alignment: AlignmentReport) -> str:
    if not alignment["found"]:
        return "none"
    parts: list[str] = []
    if alignment["params_found"]:
        parts.append(f"pose table {alignment['params_shape']}")
    if alignment["angle_offset_found"]:
        parts.append(f"angle offsets {alignment['angle_offset_shape']}")
    if alignment["misalign_spec_found"]:
        parts.append("misalignment spec")
    if alignment["gauge_fix_found"]:
        parts.append("gauge fix")
    return ", ".join(parts)


def _memory_rows(memory: MemoryEstimatesReport) -> list[tuple[str, str]]:
    if not memory["feasible"]:
        return [("volume working sets", f"not estimated ({memory['notes']})")]
    modes = memory["modes"]
    grid = " x ".join(str(n) for n in memory["reconstruction_grid_shape"] or [])
    sets = ", ".join(
        f"{name} {_size(modes[key]['estimated_working_set_bytes'])}"
        for name, key in (
            ("fbp", "fbp_fp32"),
            ("fista", "fista_tv_fp32"),
            ("spdhg", "spdhg_tv_fp32"),
        )
    )
    return [
        ("volume working sets", f"{sets} for {grid} voxels"),
        ("placement", "tomojax reconstruct INPUT --dry-run estimates a run's devices and memory"),
    ]


def _size(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"  # pragma: no cover - the loop always returns


def _volume_text(volume: VolumeReport) -> str:
    if not volume["found"]:
        return "none"
    shape = " x ".join(str(n) for n in volume["shape"] or [])
    method = f", {volume['method']}" if volume["method"] else ""
    return f"{shape} ({volume['axes']}){method}"


__all__ = ["format_inspection_report"]
