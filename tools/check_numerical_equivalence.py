"""Check that a change keeps TomoJAX's results as accurate as, and numerically equal to, before.

Each case simulates a standard phantom (Shepp-Logan for parallel and cone
beams, the laminography disk for laminography), then reconstructs it, or
aligns it after random per-view shifts, and measures the result against the
ground truth: reconstruction NRMSE and RMSE, pose error, data residual and
wall time.

``--save-baseline`` records these with each case's output. ``--compare-baseline``
reruns the baseline's cases and fails (exit status 1) when

* the ground-truth error (NRMSE, or pose error for alignment) exceeds the
  worst baseline repeat by more than ``--tolerance`` (relative), or
* the outputs are further from the baseline's, in float32 units in the last
  place (ULP), than the baseline's own repeats are from each other.

A GPU does not repeat its reductions bit for bit, so one rerun can differ from
the baseline by chance and a 1-vs-1 comparison proves nothing. Each case
therefore runs ``--repeats`` times (5 on a GPU) to measure that spread. A CPU
is bit-exact: one run suffices and any difference fails. Residual and time
are reported, never judged: a lower residual or a faster run only counts at
an equal or lower error.

Typical use::

    python tools/check_numerical_equivalence.py --quick --save-baseline before.npz
    # ... change the code ...
    python tools/check_numerical_equivalence.py --compare-baseline before.npz
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import TYPE_CHECKING, Literal, cast
import zipfile

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy.typing as npt

    from tomojax.geometry import ScanGeometry

type Geometry = Literal["parallel", "cone", "lamino"]
type Method = Literal["fbp", "cgls", "sirt", "fista", "align"]
type Device = Literal["cpu", "gpu"]
type Float32Array = npt.NDArray[np.float32]

GEOMETRIES: tuple[Geometry, ...] = ("parallel", "cone", "lamino")
METHODS: tuple[Method, ...] = ("fbp", "cgls", "sirt", "fista", "align")
DEVICES: tuple[Device, ...] = ("cpu", "gpu")
DEFAULT_METHODS: tuple[Method, ...] = ("fbp", "cgls", "align")
PHANTOMS: dict[Geometry, Literal["shepp", "lamino_disk"]] = {
    "parallel": "shepp",
    "cone": "shepp",
    "lamino": "lamino_disk",
}
# Standard deviation of the simulated per-view shifts that alignment must recover, in voxels.
SHIFT_SIGMA = 1.0


@dataclass(frozen=True)
class Case:
    """One simulated problem and the method run on it."""

    geometry: Geometry
    method: Method
    size: int
    n_views: int
    iterations: int | None  # None for fbp and align, which take no iteration count

    @property
    def name(self) -> str:
        """``geometry/method``, as the summary table and the records name the case."""
        return f"{self.geometry}/{self.method}"

    @property
    def error_metric(self) -> Literal["nrmse", "pose_error"]:
        """The ground-truth error a comparison judges."""
        return "pose_error" if self.method == "align" else "nrmse"


def plan_cases(
    geometries: Sequence[Geometry], methods: Sequence[Method], *, quick: bool
) -> list[Case]:
    """Every geometry with every method; ``quick`` cases take seconds on a CPU."""
    size, n_views, iterations = (24, 48, 20) if quick else (64, 128, 50)
    return [
        Case(g, m, size, n_views, None if m in ("fbp", "align") else iterations)
        for g in geometries
        for m in methods
    ]


@dataclass(frozen=True)
class UlpStats:
    """Median and maximum per-voxel float32 ULP distance between two outputs."""

    median: float
    max: int

    def within(self, envelope: UlpStats) -> bool:
        """Whether neither the median nor the maximum exceeds ``envelope``'s."""
        return self.median <= envelope.median and self.max <= envelope.max


def ulp_distance(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Per-element distance between two float32 arrays in units in the last place.

    Adjacent floats are 1 apart, ``+0.0`` and ``-0.0`` are 0 apart, and the
    count runs through zero: the smallest positive and negative subnormals are
    2 apart.
    """

    def ordered(x: npt.ArrayLike) -> npt.NDArray[np.int64]:
        bits = np.ascontiguousarray(x, dtype=np.float32).view(np.int32).astype(np.int64)
        return np.where(bits < 0, -(bits & 0x7FFFFFFF), bits)

    return np.abs(ordered(a) - ordered(b))


def ulp_stats(a: Float32Array, b: Float32Array) -> UlpStats:
    """Summarize :func:`ulp_distance` over all voxels."""
    distance = ulp_distance(a, b)
    return UlpStats(float(np.quantile(distance, 0.5)), int(np.max(distance)))


def spread_envelope(spread: Sequence[UlpStats]) -> UlpStats:
    """The largest median and maximum among repeat-to-repeat distances; zero for one run."""
    return UlpStats(
        max((s.median for s in spread), default=0.0), max((s.max for s in spread), default=0)
    )


@dataclass(frozen=True)
class Measured:
    """A case's metrics (one value per repeat) and outputs from this process."""

    case: Case
    metrics: dict[str, list[float]]
    outputs: list[Float32Array]

    @property
    def spread(self) -> list[UlpStats]:
        """How far repeats 2..N landed from repeat 1: this device's run-to-run noise."""
        return [ulp_stats(out, self.outputs[0]) for out in self.outputs[1:]]


@dataclass(frozen=True)
class BaselineCase:
    """A case as a baseline recorded it: metrics, run-to-run spread, and repeat 1's output."""

    case: Case
    metrics: dict[str, list[float]]
    spread: list[UlpStats]
    reference: Float32Array


@dataclass(frozen=True)
class Baseline:
    """A saved run to compare against: the device it ran on and its cases by name."""

    device: Device
    cases: dict[str, BaselineCase]


@dataclass(frozen=True)
class Check:
    """One pass/fail judgment of a case against its baseline."""

    name: Literal["error", "ulp"]
    passed: bool
    detail: str


def pose_error(estimated: np.ndarray, true: np.ndarray, geometry: ScanGeometry) -> float:
    """RMS detector-plane translation error after removing the best common object shift.

    Shifting the whole object is invisible to alignment (the image moves with
    it), so that gauge is fitted by least squares and removed. ``estimated`` and
    ``true`` are ``(views, 6)`` pose tables as :attr:`tomojax.Scan.poses` gives
    them; the result is in the geometry's length unit.
    """
    n_views = len(true)
    rotations = np.array([np.asarray(geometry.pose_matrix(i))[:3, :3] for i in range(n_views)])
    # Rows x and z of world-from-object: how an object shift moves each view on the detector.
    gauge = rotations[:, [0, 2], :].reshape(-1, 3)
    error = (estimated[:, 3:5] - true[:, 3:5]).reshape(-1)
    shift = np.linalg.lstsq(gauge, error, rcond=None)[0]
    remaining = error - gauge @ shift
    return float(np.linalg.norm(remaining)) / math.sqrt(n_views)


def measure(case: Case, repeats: int) -> Measured:
    """Simulate ``case`` once, then run its method ``repeats`` times and score each run."""
    import tomojax as tj

    simulation = tj.simulate(
        PHANTOMS[case.geometry],
        size=case.size,
        n_views=case.n_views,
        geometry=case.geometry,
        translation_sigma=SHIFT_SIGMA if case.method == "align" else 0.0,
    )
    scan, truth = simulation.scan, simulation.phantom
    measured = np.asarray(scan.projections)
    metrics: dict[str, list[float]] = {}
    outputs: list[Float32Array] = []
    for _ in range(repeats):
        start = time.perf_counter()
        if case.method == "align":
            result = tj.align(scan, fit="translations", grid=scan.grid)
            if result.image is None or simulation.true_poses is None:
                raise RuntimeError("aligning a simulated scan must give an image and true poses")
            volume = np.asarray(result.image)
            seconds = time.perf_counter() - start
            geometry = result.scan.geometry
            values = {"pose_error": pose_error(result.poses, simulation.true_poses, scan.geometry)}
            # The fitted translations: float32 in the fit, so the float32 cast is exact.
            output = result.poses[:, 3:].astype(np.float32)
        else:
            recon = tj.reconstruct(scan, case.method, iterations=case.iterations)
            volume = np.asarray(recon.volume)
            seconds = time.perf_counter() - start
            geometry = recon.geometry
            difference = volume - truth
            values = {
                "nrmse": float(np.linalg.norm(difference) / np.linalg.norm(truth)),
                "rmse": float(np.linalg.norm(difference)) / math.sqrt(difference.size),
            }
            output = volume.astype(np.float32, copy=False)
        predicted = np.asarray(tj.project(geometry, volume))
        values["residual"] = float(np.linalg.norm(predicted - measured) / np.linalg.norm(measured))
        values["seconds"] = seconds
        for key, value in values.items():
            metrics.setdefault(key, []).append(value)
        outputs.append(output)
    return Measured(case, metrics, outputs)


def compare(current: Measured, baseline: BaselineCase, *, tolerance: float) -> list[Check]:
    """Judge ``current`` against ``baseline``: its ground-truth error and its ULP distance."""
    metric = current.case.error_metric
    value = statistics.median(current.metrics[metric])
    worst = max(baseline.metrics[metric])
    limit = worst * (1 + tolerance)
    # "At most the limit" fails a NaN error, where "not above it" would pass one.
    detail = f"{metric} {value:.4g}, limit {limit:.4g} (baseline worst {worst:.4g})"
    error = Check("error", value <= limit, detail)
    shape = baseline.reference.shape
    if any(out.shape != shape for out in current.outputs):
        changed = current.outputs[0].shape
        return [error, Check("ulp", False, f"output shape {changed}, baseline {shape}")]
    # A rank test. For unchanged code each current repeat is one more draw of the baseline's
    # run-to-run noise, so every one of n repeats landing beyond all m baseline distances has
    # probability 1 / C(n + m, n): 1/126 for 5 repeats on both sides. On a bit-exact CPU the
    # baseline spread is zero and any difference fails.
    envelope = spread_envelope(baseline.spread)
    best = closest(current.outputs, baseline.reference, envelope)
    detail = f"ULP median/max {best.median:g}/{best.max}, spread {envelope.median:g}/{envelope.max}"
    return [error, Check("ulp", best.within(envelope), detail)]


def closest(
    outputs: Sequence[Float32Array], reference: Float32Array, envelope: UlpStats
) -> UlpStats:
    """The repeat nearest ``reference``: one within ``envelope`` if any is, else by maximum."""
    distances = [ulp_stats(out, reference) for out in outputs]
    return min(distances, key=lambda d: (not d.within(envelope), d.max, d.median))


def save_baseline(path: Path, device: Device, measured: Sequence[Measured]) -> None:
    """Write metrics and spread as ``meta.json`` and each case's first output as ``.npy``.

    The file is a zip that ``numpy.load`` also reads; it refuses an existing ``path``.
    """
    meta: dict[str, object] = {
        "device": device,
        "cases": [
            {"case": asdict(m.case), "metrics": m.metrics, "spread": [asdict(s) for s in m.spread]}
            for m in measured
        ],
    }
    with zipfile.ZipFile(path, "x") as archive:
        archive.writestr("meta.json", json.dumps(meta, indent=1))
        for i, m in enumerate(measured):
            with archive.open(f"output_{i}.npy", "w") as stream:
                np.lib.format.write_array(stream, m.outputs[0], allow_pickle=False)


def load_baseline(path: Path) -> Baseline:
    """Read a file :func:`save_baseline` wrote; raise ValueError if it holds something else."""
    with zipfile.ZipFile(path) as archive:
        meta = _record(cast("object", json.loads(archive.read("meta.json"))), "baseline")
        device: Device = _choice(meta.get("device"), DEVICES, "device")
        cases: dict[str, BaselineCase] = {}
        for i, raw in enumerate(_records(meta.get("cases"), "cases")):
            spec = _record(raw.get("case"), f"cases[{i}].case")
            iterations = spec.get("iterations")
            case = Case(
                _choice(spec.get("geometry"), GEOMETRIES, f"cases[{i}] geometry"),
                _choice(spec.get("method"), METHODS, f"cases[{i}] method"),
                _int(spec.get("size"), f"cases[{i}] size"),
                _int(spec.get("n_views"), f"cases[{i}] n_views"),
                None if iterations is None else _int(iterations, f"cases[{i}] iterations"),
            )
            raw_metrics = _record(raw.get("metrics"), f"{case.name} metrics")
            metrics = {k: _floats(v, f"{case.name} {k}") for k, v in raw_metrics.items()}
            if case.error_metric not in metrics:
                raise ValueError(f"{case.name} has no {case.error_metric}")
            spread = [
                UlpStats(_floats([s.get("median")], "median")[0], _int(s.get("max"), "max"))
                for s in _records(raw.get("spread"), f"{case.name} spread")
            ]
            with archive.open(f"output_{i}.npy") as stream:
                reference = np.lib.format.read_array(stream, allow_pickle=False)
            if reference.dtype != np.float32:
                raise ValueError(f"{case.name} output is {reference.dtype}, not float32")
            cases[case.name] = BaselineCase(case, metrics, spread, reference)
    return Baseline(device, cases)


def _record(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a JSON object")
    return {str(k): v for k, v in cast("dict[object, object]", value).items()}


def _records(value: object, where: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be a JSON list")
    return [_record(v, f"{where}[{i}]") for i, v in enumerate(cast("list[object]", value))]


def _choice[T: str](value: object, options: tuple[T, ...], where: str) -> T:
    for option in options:
        if value == option:
            return option
    raise ValueError(f"{where} must be one of {', '.join(options)}, not {value!r}")


def _int(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{where} must be an integer, not {value!r}")
    return value


def _floats(value: object, where: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty list of numbers")
    out: list[float] = []
    for v in cast("list[object]", value):
        if not isinstance(v, int | float) or isinstance(v, bool):
            raise ValueError(f"{where} must hold numbers, not {v!r}")
        out.append(float(v))
    return out


def summary_table(
    measured: Sequence[Measured], checks: Mapping[str, list[Check]], baseline: Baseline | None
) -> str:
    """One row per case, then each failed check spelled out."""
    header = ("case", "error", "residual", "first s", "rest s", "ULP med/max", "spread", "result")
    rows: list[tuple[str, ...]] = [header]
    failures: list[str] = []
    for m in measured:
        case, metrics, seconds = m.case, m.metrics, m.metrics["seconds"]
        reference = None if baseline is None else baseline.cases[case.name]
        ulp = "-"
        envelope = spread_envelope(m.spread if reference is None else reference.spread)
        if reference is not None and all(o.shape == reference.reference.shape for o in m.outputs):
            best = closest(m.outputs, reference.reference, envelope)
            ulp = f"{best.median:g}/{best.max}"
        judged = checks.get(case.name, [])
        result = "-" if not judged else "PASS" if all(c.passed for c in judged) else "FAIL"
        failures += [f"{case.name} {c.name}: {c.detail}" for c in judged if not c.passed]
        rows.append(
            (
                case.name,
                f"{case.error_metric} {statistics.median(metrics[case.error_metric]):.4g}",
                f"{statistics.median(metrics['residual']):.4g}",
                f"{seconds[0]:.2f}",
                f"{statistics.median(seconds[1:]):.2f}" if len(seconds) > 1 else "-",
                ulp,
                f"{envelope.median:g}/{envelope.max}",
                result,
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    lines = [
        "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip() for row in rows
    ]
    return "\n".join(lines + failures)


class Options(argparse.Namespace):
    """The parsed command line, typed."""

    quick: bool = False
    geometries: list[Geometry] | None = None
    methods: list[Method] | None = None
    device: Device | None = None
    repeats: int | None = None
    tolerance: float = 0.01
    save_baseline: Path | None = None
    compare_baseline: Path | None = None
    json_out: Path | None = None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check reconstruction and pose error, and ULP equivalence to a baseline.",
        epilog="Exit status: 0 all checks pass, 1 a check failed, 2 usage error.",
    )
    _ = parser.add_argument("--quick", action="store_true", help="small cases for a CPU")
    _ = parser.add_argument("--geometries", nargs="+", choices=GEOMETRIES, help="default: all")
    _ = parser.add_argument(
        "--methods", nargs="+", choices=METHODS, help=f"default: {' '.join(DEFAULT_METHODS)}"
    )
    _ = parser.add_argument("--device", choices=DEVICES, help="default: JAX's default backend")
    _ = parser.add_argument("--repeats", type=int, help="runs per case (default 5 gpu, 1 cpu)")
    _ = parser.add_argument("--tolerance", type=float, help="relative error growth allowed (0.01)")
    baseline = parser.add_mutually_exclusive_group()
    _ = baseline.add_argument("--save-baseline", type=Path, metavar="NPZ")
    _ = baseline.add_argument("--compare-baseline", type=Path, metavar="NPZ")
    _ = parser.add_argument("--json-out", type=Path, metavar="JSON", help="the full record")
    return parser


def _parse(argv: Sequence[str] | None) -> tuple[Options, Baseline | None, argparse.ArgumentParser]:
    """Parse and check the command line and read the baseline, all before JAX starts."""
    parser = _parser()
    options = parser.parse_args(argv, namespace=Options())
    if options.repeats is not None and options.repeats < 1:
        parser.error("--repeats must be at least 1")
    for path in (options.save_baseline, options.json_out):
        if path is not None and path.exists():
            parser.error(f"{path} exists; choose a new path")
    if options.compare_baseline is None:
        return options, None, parser
    if options.quick:
        parser.error("--quick sizes new cases; a comparison reruns the baseline's")
    try:
        baseline = load_baseline(options.compare_baseline)
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        parser.error(f"cannot read baseline {options.compare_baseline}: {exc}")
    cases = {
        name: c
        for name, c in baseline.cases.items()
        if (options.geometries is None or c.case.geometry in options.geometries)
        and (options.methods is None or c.case.method in options.methods)
    }
    if not cases:
        parser.error("no baseline case matches --geometries and --methods")
    return options, Baseline(baseline.device, cases), parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the cases, print the summary table and return the exit status."""
    options, baseline, parser = _parse(argv)
    if options.device is not None:
        os.environ["JAX_PLATFORMS"] = "cpu" if options.device == "cpu" else "cuda"
    import jax

    device: Device = "cpu" if jax.default_backend() == "cpu" else "gpu"
    if options.device is not None and device != options.device:
        parser.error(f"--device {options.device} requested but JAX runs on {device}")
    if baseline is not None and baseline.device != device:
        parser.error(f"the baseline ran on {baseline.device}; compare on the same device")
    repeats = options.repeats or (5 if device == "gpu" else 1)
    cases = (
        plan_cases(
            options.geometries or GEOMETRIES,
            options.methods or DEFAULT_METHODS,
            quick=options.quick,
        )
        if baseline is None
        else [c.case for c in baseline.cases.values()]
    )

    measured: list[Measured] = []
    checks: dict[str, list[Check]] = {}
    for case in cases:
        print(f"running {case.name} x{repeats}", file=sys.stderr, flush=True)
        m = measure(case, repeats)
        measured.append(m)
        if baseline is not None:
            checks[case.name] = compare(m, baseline.cases[case.name], tolerance=options.tolerance)
    passed = all(c.passed for judged in checks.values() for c in judged)

    print(f"device {device}, {repeats} repeat(s) per case")
    print(summary_table(measured, checks, baseline))
    if baseline is not None:
        print("PASS" if passed else "FAIL")
    if options.save_baseline is not None:
        save_baseline(options.save_baseline, device, measured)
    if options.json_out is not None:
        record: dict[str, object] = {
            "device": device,
            "repeats": repeats,
            "tolerance": options.tolerance,
            "baseline": None if options.compare_baseline is None else str(options.compare_baseline),
            "passed": passed,
            "cases": [
                {
                    "case": asdict(m.case),
                    "metrics": m.metrics,
                    "spread": [asdict(s) for s in m.spread],
                    "checks": [asdict(c) for c in checks.get(m.case.name, [])],
                }
                for m in measured
            ],
        }
        _ = options.json_out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
