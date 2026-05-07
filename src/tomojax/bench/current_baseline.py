"""Normalize current/default TomoJAX benchmark metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sequence


def current_baseline_payload(
    *,
    source_path: str | Path,
    benchmark: str,
    profile: str,
    implementation: str = "current_default",
) -> dict[str, object]:
    """Return a normalized current/default baseline payload."""
    path = Path(source_path)
    raw = cast("object", json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict):
        raise ValueError("current baseline source JSON must contain an object")
    raw_payload = cast("dict[object, object]", raw)
    volume_nmse = _extract_volume_nmse(raw_payload)
    if volume_nmse is None:
        raise ValueError("current baseline source JSON must contain numeric volume_nmse")
    return {
        "schema": "tomojax.current_default_baseline.v1",
        "benchmark": benchmark,
        "implementation": implementation,
        "profile": profile,
        "source_path": str(path),
        "volume_nmse": volume_nmse,
        "reconstruction": {
            "volume_nmse": volume_nmse,
            "final_residual": _extract_optional_float(raw_payload, "final_residual"),
        },
        "raw": {str(key): value for key, value in raw_payload.items()},
    }


def write_current_baseline_artifacts(
    *,
    source_path: str | Path,
    output_dir: str | Path,
    benchmark: str,
    profile: str,
    implementation: str = "current_default",
) -> tuple[Path, Path]:
    """Write `benchmark_baseline_current.json` and markdown summary artifacts."""
    payload = current_baseline_payload(
        source_path=source_path,
        benchmark=benchmark,
        profile=profile,
        implementation=implementation,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_baseline_current.json"
    md_path = out_dir / "benchmark_baseline_current.md"
    _ = json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _ = md_path.write_text(_baseline_markdown(payload), encoding="utf-8")
    return json_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    """Normalize a current/default baseline metrics JSON from the command line."""
    parser = argparse.ArgumentParser(
        description="Normalize current/default TomoJAX benchmark metrics."
    )
    _ = parser.add_argument("source_json", help="Current/default metrics JSON to normalize.")
    _ = parser.add_argument("--out-dir", required=True, help="Output artifact directory.")
    _ = parser.add_argument("--benchmark", required=True, help="Synthetic benchmark name.")
    _ = parser.add_argument("--profile", required=True, help="Current/default profile name.")
    _ = parser.add_argument(
        "--implementation",
        default="current_default",
        help="Implementation label to record in the baseline payload.",
    )
    args = parser.parse_args(argv)
    json_path, md_path = write_current_baseline_artifacts(
        source_path=cast("str", args.source_json),
        output_dir=cast("str", args.out_dir),
        benchmark=cast("str", args.benchmark),
        profile=cast("str", args.profile),
        implementation=cast("str", args.implementation),
    )
    print(f"benchmark_baseline_current_json: {json_path}")
    print(f"benchmark_baseline_current_md: {md_path}")
    return 0


def _extract_volume_nmse(payload: dict[object, object]) -> float | None:
    direct = payload.get("volume_nmse")
    if isinstance(direct, int | float):
        return float(direct)
    reconstruction = payload.get("reconstruction")
    if isinstance(reconstruction, dict):
        value = cast("dict[object, object]", reconstruction).get("volume_nmse")
        if isinstance(value, int | float):
            return float(value)
    return None


def _extract_optional_float(payload: dict[object, object], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, int | float):
        return float(value)
    reconstruction = payload.get("reconstruction")
    if isinstance(reconstruction, dict):
        nested = cast("dict[object, object]", reconstruction).get(key)
        if isinstance(nested, int | float):
            return float(nested)
    return None


def _baseline_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Current TomoJAX Baseline",
        "",
        f"- Benchmark: `{payload.get('benchmark')}`",
        f"- Implementation: `{payload.get('implementation')}`",
        f"- Profile: `{payload.get('profile')}`",
        f"- Volume NMSE: `{payload.get('volume_nmse')}`",
        f"- Source: `{payload.get('source_path')}`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
