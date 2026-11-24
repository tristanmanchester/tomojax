"""Terminal formatting for NXtomo/HDF5 inspection reports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tomojax.io._inspection_types import InspectionReport


def _fmt_value(value: object, *, precision: int = 6) -> str:
    if value is None:
        return "not found"
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def format_inspection_report(report: InspectionReport) -> str:
    """Format an inspection report for terminal output.

    Sections hold ``key: value`` lines, so the text is never cut or wrapped.

    Parameters
    ----------
    report : InspectionReport
        Structured dataset inspection report.

    Returns
    -------
    str
        Human-readable multiline summary for terminal output.
    """
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
    lines: list[str] = []
    for title, rows in sections:
        width = max(len(key) for key, _ in rows)
        lines.append(title)
        lines.extend(f"  {key + ':':<{width + 1}} {value}" for key, value in rows)
    return "\n".join(lines)


def _projection_rows(projection: Mapping[str, Any]) -> list[tuple[str, str]]:
    if not projection["found"]:
        return [("projections", "not found")]
    stats = projection["stats"]
    nonfinite = projection["nonfinite"]
    detector = projection["detector_shape"] or {}
    views, rows, columns = (projection["shape"] or [None, None, None])[:3]
    shape = f"{views} views of {rows} x {columns} pixels" if detector else str(projection["shape"])
    values = ", ".join(
        f"{name} {_fmt_value(stats[key])}"
        for name, key in (("min", "min"), ("1%", "p01"), ("median", "p50"), ("99%", "p99"))
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


def _angle_rows(angles: Mapping[str, Any]) -> list[tuple[str, str]]:
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


def _detector_rows(detector: Mapping[str, Any]) -> list[tuple[str, str]]:
    if not detector["found"]:
        return [("detector", "not found")]
    centre = detector["center"]
    centre_text = (
        ", ".join(_fmt_value(c) for c in centre)
        if isinstance(centre, list | tuple)
        else str(centre)
    )
    return [
        ("detector", f"{detector['nu']} x {detector['nv']} pixels (u x v)"),
        ("pixel size", f"{_fmt_value(detector['du'])} x {_fmt_value(detector['dv'])}"),
        ("centre", f"({centre_text})"),
    ]


def _flats_darks_text(flats_darks: Mapping[str, Any]) -> str:
    if flats_darks["flats_present"] or flats_darks["darks_present"]:
        return (
            f"{flats_darks['flat_count']} flats, {flats_darks['dark_count']} darks, "
            f"{flats_darks['sample_count']} sample frames "
            f"(image_key {_fmt_value(flats_darks['image_key_path'])})"
        )
    if flats_darks["image_key_found"]:
        return f"none (image_key {_fmt_value(flats_darks['image_key_path'])} labels none)"
    return "not found"


def _alignment_text(alignment: Mapping[str, Any]) -> str:
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


def _memory_rows(memory: Mapping[str, Any]) -> list[tuple[str, str]]:
    if not memory["feasible"]:
        return [("volume working sets", f"not estimated ({memory['notes']})")]
    modes = memory["modes"]
    grid = " x ".join(str(n) for n in reversed(memory["reconstruction_grid_shape"]))
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


def _volume_text(volume: Mapping[str, Any]) -> str:
    if not volume["found"]:
        return "none"
    shape = " x ".join(str(n) for n in volume["shape"] or [])
    method = f", {volume['method']}" if volume["method"] else ""
    return f"{shape} ({volume['axes']}){method}"


__all__ = ["format_inspection_report"]
