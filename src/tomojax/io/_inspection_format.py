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
    lines = [
        f"TomoJAX inspection: {report['input_path']}",
        *_projection_lines(report["projection"]),
        _angles_line(report["angles"]),
        f"Geometry type: {_fmt_value(geometry['type'])}",
        f"Geometry metadata: {'present' if geometry['meta_found'] else 'not found'}",
        _detector_line(report["detector_metadata"]),
        _flats_darks_line(report["flats_darks"]),
        "Corrections: " + (", ".join(corrections) if corrections else "none recorded"),
        _alignment_line(report["alignment"]),
        _memory_line(report["memory_estimates"]),
        _volume_line(report["volume"]),
    ]
    return "\n".join(lines)


def _projection_lines(projection: Mapping[str, Any]) -> list[str]:
    if not projection["found"]:
        return ["Projection shape: not found"]
    stats = projection["stats"]
    nonfinite = projection["nonfinite"]
    return [
        f"Projection shape: {projection['shape']}",
        f"Dtype: {projection['dtype']}",
        f"Views: {projection['n_views']}",
        f"Detector shape: {projection['detector_shape']}",
        (
            "Stats: "
            f"min={_fmt_value(stats['min'])}, "
            f"p01={_fmt_value(stats['p01'])}, "
            f"mean={_fmt_value(stats['mean'])}, "
            f"p50={_fmt_value(stats['p50'])}, "
            f"p99={_fmt_value(stats['p99'])}, "
            f"max={_fmt_value(stats['max'])}"
        ),
        (
            "NaN/Inf counts: "
            f"nan={nonfinite['nan_count']}, "
            f"+inf={nonfinite['posinf_count']}, "
            f"-inf={nonfinite['neginf_count']}, "
            f"inf_total={nonfinite['inf_count']}"
        ),
    ]


def _angles_line(angles: Mapping[str, Any]) -> str:
    if not angles["found"]:
        return "Angle coverage: not found"
    return (
        "Angle coverage: "
        f"{_fmt_value(angles['coverage_deg'])} deg "
        f"(min={_fmt_value(angles['min_deg'])}, "
        f"max={_fmt_value(angles['max_deg'])}, "
        f"count={angles['count']}, "
        f"units={_fmt_value(angles['units'])})"
    )


def _detector_line(detector: Mapping[str, Any]) -> str:
    if not detector["found"]:
        return "Detector metadata: not found"
    return (
        "Detector metadata: "
        f"nu={detector['nu']}, nv={detector['nv']}, "
        f"du={detector['du']}, dv={detector['dv']}, "
        f"center={detector['center']}"
    )


def _flats_darks_line(flats_darks: Mapping[str, Any]) -> str:
    if flats_darks["flats_present"] or flats_darks["darks_present"]:
        return (
            "Flats/darks: "
            f"samples={flats_darks['sample_count']}, "
            f"flats={flats_darks['flat_count']}, "
            f"darks={flats_darks['dark_count']}, "
            f"image_key={_fmt_value(flats_darks['image_key_path'])}"
        )
    if flats_darks["image_key_found"]:
        return (
            "Flats/darks: not found "
            f"(image_key={_fmt_value(flats_darks['image_key_path'])}; no flat/dark frames)"
        )
    return "Flats/darks: not found"


def _alignment_line(alignment: Mapping[str, Any]) -> str:
    if not alignment["found"]:
        return "Alignment parameters: not found"
    parts: list[str] = []
    if alignment["params_found"]:
        parts.append(f"params shape={alignment['params_shape']}")
    if alignment["angle_offset_found"]:
        parts.append(f"angle_offset shape={alignment['angle_offset_shape']}")
    if alignment["misalign_spec_found"]:
        parts.append("misalign_spec present")
    if alignment["gauge_fix_found"]:
        parts.append("gauge_fix present")
    return f"Alignment parameters: {', '.join(parts)}"


def _memory_line(memory: Mapping[str, Any]) -> str:
    if not memory["feasible"]:
        return f"Memory estimates: not found ({memory['notes']})"
    modes = memory["modes"]
    return (
        "Memory estimates: "
        f"grid={memory['reconstruction_grid_shape']}, "
        f"fbp_fp32={modes['fbp_fp32']['estimated_working_set_bytes']} bytes, "
        f"fista_tv_fp32={modes['fista_tv_fp32']['estimated_working_set_bytes']} bytes, "
        f"spdhg_tv_fp32={modes['spdhg_tv_fp32']['estimated_working_set_bytes']} bytes"
    )


def _volume_line(volume: Mapping[str, Any]) -> str:
    if not volume["found"]:
        return "Volume: not found"
    method = f", method={volume['method']}" if volume["method"] else ""
    return f"Volume: shape={volume['shape']}, axes={volume['axes']}{method}"


__all__ = ["format_inspection_report"]
