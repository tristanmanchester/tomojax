"""Terminal formatting for NXtomo/HDF5 inspection reports."""

from __future__ import annotations

from tomojax.io._inspection_types import InspectionReport


def _fmt_value(value: object, *, precision: int = 6) -> str:
    if value is None:
        return "not found"
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def _fmt_bool_presence(found: bool) -> str:
    return "present" if found else "not found"


def format_inspection_report(report: InspectionReport) -> str:
    """Format an inspection report for terminal output."""
    projection = report["projection"]
    angles = report["angles"]
    geometry = report["geometry"]
    detector_metadata = report["detector_metadata"]
    flats_darks = report["flats_darks"]
    preprocess = report["preprocess"]
    alignment = report["alignment"]
    memory = report["memory_estimates"]

    lines = [f"TomoJAX inspection: {report['input_path']}"]
    if projection["found"]:
        stats = projection["stats"]
        nonfinite = projection["nonfinite"]
        lines.extend(
            [
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
        )
    else:
        lines.append("Projection shape: not found")

    if angles["found"]:
        lines.append(
            "Angle coverage: "
            f"{_fmt_value(angles['coverage_deg'])} deg "
            f"(min={_fmt_value(angles['min_deg'])}, "
            f"max={_fmt_value(angles['max_deg'])}, "
            f"count={angles['count']}, "
            f"units={_fmt_value(angles['units'])})"
        )
    else:
        lines.append("Angle coverage: not found")

    lines.append(f"Geometry type: {_fmt_value(geometry['type'])}")
    lines.append(f"Geometry metadata: {_fmt_bool_presence(bool(geometry['meta_found']))}")
    if detector_metadata["found"]:
        lines.append(
            "Detector metadata: "
            f"nu={detector_metadata['nu']}, nv={detector_metadata['nv']}, "
            f"du={detector_metadata['du']}, dv={detector_metadata['dv']}, "
            f"det_center={detector_metadata['det_center']}"
        )
    else:
        lines.append("Detector metadata: not found")

    if flats_darks["flats_present"] or flats_darks["darks_present"]:
        lines.append(
            "Flats/darks: "
            f"samples={flats_darks['sample_count']}, "
            f"flats={flats_darks['flat_count']}, "
            f"darks={flats_darks['dark_count']}, "
            f"image_key={_fmt_value(flats_darks['image_key_path'])}"
        )
    elif flats_darks["image_key_found"]:
        lines.append(
            "Flats/darks: not found "
            f"(image_key={_fmt_value(flats_darks['image_key_path'])}; no flat/dark frames)"
        )
    else:
        lines.append("Flats/darks: not found")

    if preprocess["found"]:
        lines.append(
            "Preprocess output: "
            f"domain={_fmt_value(preprocess['output_domain'])}, "
            f"epsilon={_fmt_value(preprocess['epsilon'])}, "
            f"clip_min={_fmt_value(preprocess['clip_min'])}"
        )
    else:
        lines.append("Preprocess output: not found")

    if alignment["found"]:
        parts: list[str] = []
        if alignment["params_found"]:
            parts.append(f"params shape={alignment['params_shape']}")
        if alignment["angle_offset_found"]:
            parts.append(f"angle_offset shape={alignment['angle_offset_shape']}")
        if alignment["misalign_spec_found"]:
            parts.append("misalign_spec present")
        if alignment["gauge_fix_found"]:
            parts.append("gauge_fix present")
        lines.append(f"Alignment parameters: {', '.join(parts)}")
    else:
        lines.append("Alignment parameters: not found")

    if memory["feasible"]:
        lines.append(
            "Memory estimates: "
            f"grid={memory['reconstruction_grid_shape']}, "
            f"fbp_fp32={memory['modes']['fbp_fp32']['estimated_working_set_bytes']} bytes, "
            "fista_tv_fp32="
            f"{memory['modes']['fista_tv_fp32']['estimated_working_set_bytes']} bytes, "
            f"spdhg_tv_fp32={memory['modes']['spdhg_tv_fp32']['estimated_working_set_bytes']} bytes"
        )
    else:
        lines.append(f"Memory estimates: not found ({memory['notes']})")

    return "\n".join(lines)


__all__ = ["format_inspection_report"]
