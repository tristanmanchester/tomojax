"""Run the multi-GPU comparisons with ASTRA and TIGRE on one Modal machine.

``scaling.py`` (cone projection, its transpose, FDK and CGLS on synthetic
data, against ASTRA and TIGRE), ``walnut.py`` (FDK and non-negative least
squares on FIPS walnut 1, against ASTRA) and ``walnut_alignment.py`` (aligning
its three orbits) run for each GPU count, on one machine with the largest,
after the multi-device tests pass on that machine's GPUs.

    modal run bench/modal_benchmarks.py --gpu H100:4 --gpus 1,2,4 --sizes 512,1024
    modal run bench/modal_benchmarks.py --gpu L4:2 --only-check  # the tests alone

The run needs a clean checkout: its wheel is built from HEAD, and every
package is pinned to ``uv.lock`` (TIGRE to a tested commit). Walnut 1 (about
6 GB, from Zenodo) and the synthetic phantoms (made on a CPU machine before the
GPUs start, once, outside the time cap) are kept in the ``tomojax-walnut``
volume. On the GPU machine the multi-device tests and a smoke run of every
library come first, and stop the run if anything fails. Every result is saved
in the volume as soon as it is measured, under ``results/<date>-<gpu>-<commit>/``
with each step's log and status (``steps.jsonl``) and the run's record (commit,
wheel and script hashes, command); ``--run-name`` continues a run of the same
build, skipping its finished steps. One deadline, five minutes inside
``--max-hours``, bounds every step, and Modal ends the machine a minute after
``--max-hours``. Four H100s with the requested CPUs and memory cost about $17
an hour.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import modal

# The benchmark scripts' directory: this one locally, /root/bench in the container.
sys.path[:0] = [str(Path(__file__).resolve().parent), "/root/bench"]
from _measure import run_bounded  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WALNUT_ZIP = "https://zenodo.org/records/2686726/files/Walnut1.zip"
TIGRE_COMMIT = "6b0951a"
# The build and test tools, which uv.lock does not pin (pytest's is the lock's dev version).
BUILD_PINS = [
    "cython==3.1.2", "setuptools==80.9.0", "wheel==0.45.1", "tqdm==4.70.1",
    "pytest==8.4.2", "iniconfig==2.3.0", "pluggy==1.6.0", "packaging==26.2", "pygments==2.20.0",
]  # fmt: skip
# What a run's results depend on: a dirty copy of any of these would not be HEAD's.
TRACKED = ["src", "bench", "tests", "pyproject.toml", "uv.lock"]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _local_build() -> tuple[str, Path, Path]:
    """HEAD's commit, a wheel built from it and the lockfile's pinned versions."""
    dirty = _git("status", "--porcelain", "--", *TRACKED)
    if dirty:
        raise SystemExit(f"commit these first; the run must be HEAD's:\n{dirty}")
    commit = _git("rev-parse", "--short=12", "HEAD")
    out = ROOT / "dist" / f"modal-{commit}"
    if not list(out.glob("tomojax-*.whl")):
        subprocess.run(["uv", "build", "--wheel", "--out-dir", str(out)], cwd=ROOT, check=True)
    export = ["uv", "export", "--frozen", "--no-dev", "--extra", "cuda12", "--group", "benchmark"]
    pins = out / "constraints.txt"
    locked = subprocess.run(
        [*export, "--no-hashes", "--no-emit-project"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    pins.write_text(locked + "\n".join(BUILD_PINS) + "\n")
    return commit, next(out.glob("tomojax-*.whl")), pins


image = modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.12")
image = image.apt_install("git", "unzip", "aria2", "build-essential")
COMMIT = ""
if modal.is_local():  # the container imports this module too, without the files
    COMMIT, WHEEL, PINS = _local_build()
    image = image.add_local_file(PINS, "/root/constraints.txt", copy=True)
    # The lockfile's NumPy, so TIGRE builds against the version it will run with.
    image = image.run_commands(
        "pip install -c /root/constraints.txt numpy scipy imageio astra-toolbox "
        "cython setuptools wheel tqdm pytest"
    )
    image = image.run_commands(
        "git clone https://github.com/CERN/TIGRE /opt/TIGRE",
        f"cd /opt/TIGRE && git checkout {TIGRE_COMMIT} && "
        "pip install --no-build-isolation -c /root/constraints.txt .",
        # This Python was built with clang, which the image does not have.
        env={"CC": "gcc", "CXX": "g++", "LDSHARED": "gcc -pthread -shared"},
    )
    image = image.add_local_file(WHEEL, f"/root/{WHEEL.name}", copy=True)
    image = image.run_commands(f"pip install -c /root/constraints.txt '/root/{WHEEL.name}[cuda12]'")
    ignore = ["results/**", "reference/**", "phantoms/**", "**/__pycache__/**"]
    image = image.add_local_dir(ROOT / "bench", "/root/bench", copy=True, ignore=ignore)
    image = image.add_local_dir(
        ROOT / "tests", "/root/tests", copy=True, ignore=["**/__pycache__/**"]
    )
app = modal.App("tomojax-benchmarks", image=image)
data = modal.Volume.from_name("tomojax-walnut", create_if_missing=True)


@app.function(volumes={"/data": data}, timeout=12 * 3600)
def fetch_walnut() -> None:
    """Walnut 1's projections and reference reconstruction, unzipped into the volume once.

    Zenodo serves it at 0.1-0.6 MB/s a connection and drops long transfers, so
    aria2 fetches it over eight connections, retrying each piece, into the
    volume, which is committed every five minutes: a stopped attempt resumes
    where it was. Uploading a local copy can be quicker:
    ``modal volume put tomojax-walnut <WalnutN dir> /Walnut1``, then an empty
    ``/Walnut1/.complete``.
    """
    import shutil
    import threading

    done = Path("/data/Walnut1/.complete")
    if done.exists():
        return
    shutil.rmtree("/data/Walnut1", ignore_errors=True)  # an interrupted extraction
    archive = Path("/data/partial-Walnut1.zip")  # with aria2's record of what it has
    stop = threading.Event()

    def keep_committing() -> None:
        while not stop.wait(300):
            data.commit()

    committer = threading.Thread(target=keep_committing, daemon=True)
    committer.start()
    fetch = ["aria2c", "--split=8", "--max-connection-per-server=8", "--min-split-size=20M",
             "--continue=true", "--max-tries=0", "--retry-wait=15", "--timeout=120",
             "--allow-overwrite=true", "--auto-file-renaming=false", "--summary-interval=300",
             f"--dir={archive.parent}", f"--out={archive.name}", WALNUT_ZIP]  # fmt: skip
    try:
        subprocess.run(fetch, check=True)
    finally:
        stop.set()
        committer.join()
        data.commit()
    subprocess.run(["unzip", "-tq", str(archive)], check=True)  # whole, before unpacking
    subprocess.run(["unzip", "-q", str(archive), "-d", "/data"], check=True)
    if not list(Path("/data/Walnut1/Reconstructions").glob("full_AGD_50_*.tiff")):
        raise RuntimeError("walnut 1's reference reconstruction is missing from the download")
    done.write_text(WALNUT_ZIP)
    archive.unlink()
    data.commit()


def _case_name(size: int, views: int) -> str:
    """The cache file of ``compare_cone.py``'s phantom: its generator's hash is in the name."""
    source = Path("/root/bench/compare_cone.py").read_bytes()
    return f"cone-{size}-{views}-{hashlib.sha256(source).hexdigest()[:10]}.npz"


@app.function(volumes={"/data": data}, timeout=4 * 3600, cpu=4, memory=48 * 1024)
def make_cases(sizes: list[int], views: int) -> None:
    """``scaling.py``'s phantoms and the walnut's decoded projections, cached in the volume (CPU).

    Decoding the walnut's TIFFs from the volume took 8 minutes of each GPU step
    in the rehearsal; the cache is one file per selection of views.
    """
    for orbits, every in (((2,), 1), ((1, 2, 3), 4)):
        cache = Path(_walnut_cache(orbits, every))
        if not cache.exists():
            code = (
                "from pathlib import Path; from walnut import load_orbits; "
                f"load_orbits(Path('/data/Walnut1'), {list(orbits)}, {every}, "
                f"cache=Path('{cache}'))"
            )
            subprocess.run(["python", "-c", code], cwd="/root/bench", check=True)
            data.commit()
    cases = Path("/data/cases")
    cases.mkdir(parents=True, exist_ok=True)
    for size in sizes:
        case = cases / _case_name(size, views)
        if case.exists():
            continue
        partial = cases / f"partial-{case.name}"  # renamed within the volume once whole
        code = (
            "import numpy as np; from compare_cone import setup, make_case; "
            f"v, d = make_case(setup({size}, {views})); np.savez('{partial}', volume=v, data=d)"
        )
        subprocess.run(["python", "-c", code], cwd="/root/bench", check=True)
        partial.rename(case)
        data.commit()


@app.function(timeout=2400)
def check(gpus: int) -> str:
    """The multi-device tests on ``gpus`` GPUs: their report, or an error if any fails."""
    return _check(gpus, Path("/tmp"), timeout=2000)


def _check(gpus: int, out: Path, *, timeout: float) -> str:
    """Run the multi-device tests twice, the second time with TomoJAX's CUDA gather forced.

    Each report is saved in ``out`` before it is judged; raises unless both pass
    on ``gpus`` GPUs with nothing skipped, failed or in error.
    """
    import xml.etree.ElementTree as ET

    found = subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    if len(found) != gpus:
        raise RuntimeError(f"expected {gpus} GPUs, found {found}")
    reports = []
    for name, extra in (("tests", {}), ("tests-cuda-gather", {"TOMOJAX_CUDA_KERNELS": "1"})):
        junit = out / f"{name}.xml"
        command = ["python", "-m", "pytest", "-v", "-p", "no:cacheprovider", f"--junitxml={junit}",
                   "tests/test_devices.py"]  # fmt: skip
        status = run_bounded(
            command, log=out / f"{name}.log", timeout=timeout / 2, cwd="/root",
            env={**os.environ, **extra},
        )  # fmt: skip
        report = (out / f"{name}.log").read_text(errors="replace")
        reports.append(report)
        counts = ET.parse(junit).getroot().find("testsuite") if junit.exists() else None
        attrs = {} if counts is None else counts.attrib
        bad = sum(int(attrs.get(k, 0)) for k in ("failures", "errors", "skipped"))
        if status != "exit 0" or bad or int(attrs.get("tests", 0)) == 0:
            raise RuntimeError(f"{name}: {status}, {attrs}:\n{report[-4000:]}")
    return "\n".join(reports)


@app.function(volumes={"/data": data}, cpu=16, memory=96 * 1024, max_containers=1)
def benchmark(gpus: list[int], sizes: list[int], views: int, run: dict) -> None:
    """The tests, then every comparison, most valuable first, each saved as it comes.

    One deadline, ``run["max_hours"]`` after the start, bounds every step: each
    gets at most the time left, and none starts with less than a minute left.
    """
    import shutil

    deadline = time.monotonic() + run["max_hours"] * 3600 - 300  # time to save at the end
    out = Path(f"/data/results/{run['name']}")
    out.mkdir(parents=True, exist_ok=True)
    _record_attempt(out, run)
    finished = _finished_steps(out)
    try:
        if "tests" not in finished:
            _check(max(gpus), out, timeout=min(2400, deadline - time.monotonic()))
            _log_step(out, "tests", "exit 0")
    finally:
        data.commit()
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "TOMOJAX_COMMIT": run["commit"]}

    def step(name: str, command: list[str], limit: float, *, cpu_jax: bool = False) -> str:
        """Run ``command`` for at most ``limit`` s, or the time left; its status."""
        if name in finished:
            return "exit 0"  # in an earlier attempt of this run
        timeout = min(limit, deadline - time.monotonic())
        if timeout < 60:
            status = "skipped: the run's time is spent"
        else:
            if "--worker-timeout" in command:  # inside the step's own deadline, with time to save
                command = list(command)
                command[command.index("--worker-timeout") + 1] = str(max(30, timeout - 60))
            log = out / f"{name}.log"
            attempt = 1
            while log.exists():  # keep earlier attempts' logs
                attempt += 1
                log = out / f"{name}.{attempt}.log"
            status = run_bounded(
                command, log=log, timeout=timeout, cwd="/root/bench",
                env={**env, "JAX_PLATFORMS": "cpu"} if cpu_jax else env,
            )  # fmt: skip
        _log_step(out, name, status)
        print(f"{name}: {status}", flush=True)
        data.commit()
        return status

    # Each library working at all, on every GPU, in a minute: stop here if not.
    smoke = out / "smoke" / "smoke.json"
    command = ["python", "scaling.py", "--size", "64", "--views", "90", "--iterations", "3",
               "--repeats", "1", "--gpus", str(max(gpus)), "--output", str(smoke),
               "--worker-timeout", "600"]  # fmt: skip
    status = step("smoke", command, 900)
    if status.startswith("skipped"):
        return
    _require_smoke(status, smoke)
    if run.get("preflight"):
        _preflight(out, env)
        return
    for name, command, limit, cpu_jax in _plan(gpus, sizes, views, out):
        if command[1] == "scaling.py":
            case = Path(command[command.index("--case") + 1])
            if not case.exists():  # read once from the volume, if there is time to use it
                if deadline - time.monotonic() < 900:
                    _log_step(out, name, "skipped: the run's time is spent")
                    return
                shutil.copyfile(f"/data/cases/{case.name}", case)
        if step(name, command, limit, cpu_jax=cpu_jax).startswith("skipped"):
            return


# Each property the kernels depend on, for every GPU.
_PROPERTIES = (
    "name", "major", "minor", "textureAlignment", "texturePitchAlignment", "sharedMemPerBlock",
    "sharedMemPerBlockOptin", "maxTexture2DLinear",
)  # fmt: skip
_KERNEL_TESTS = (
    "tests/test_cone_beam.py", "tests/test_projector_adjoint.py", "tests/test_fbp_accuracy.py",
    "tests/test_workflow.py",
)  # fmt: skip


def _preflight(out: Path, env: dict[str, str]) -> None:
    """The kernels' own tests on these GPUs, and the GPUs' properties; raise if a test fails."""
    command = ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", *_KERNEL_TESTS]
    status = run_bounded(command, log=out / "kernel-tests.log", timeout=1800, cwd="/root", env=env)
    _log_step(out, "kernel-tests", status)
    keys = repr(_PROPERTIES)
    code = (
        "import cupy as cp, json; r = cp.cuda.runtime; "
        f"print(json.dumps([{{k: r.getDeviceProperties(i)[k] for k in {keys}}} "
        "for i in range(r.getDeviceCount())], default=str))"
    )
    run_bounded(["python", "-c", code], log=out / "device-properties.log", timeout=120, env=env)
    data.commit()
    print(f"kernel-tests: {status}", flush=True)
    if status != "exit 0":
        raise RuntimeError(f"kernel tests failed ({status}); see kernel-tests.log")


# What each library's smoke worker must have timed.
SMOKE_OPERATIONS = {
    "tomojax": {"forward", "backproject", "fdk", "cgls"},
    "astra": {"forward", "backproject", "fdk", "cgls"},
    "tigre": {"forward", "forward_siddon", "backproject", "fdk", "cgls"},
}


# Errors the smoke case (64^3, 90 views, 3 CGLS iterations) stays under: about twice
# what every library measured in the rehearsal.
SMOKE_CEILINGS = {"forward": 0.05, "forward_siddon": 0.05, "fdk": 0.15, "cgls": 0.4}


def _require_smoke(status: str, summary: Path) -> None:
    """Raise unless the smoke run finished, every library timed every operation, sensibly."""
    import math

    records = json.loads(summary.read_text())["records"] if summary.exists() else []
    timed = {r["library"]: {o["operation"] for o in r.get("operations", [])} for r in records}
    complete = all(r.get("complete") and "failed" not in r for r in records)
    if status != "exit 0" or not complete or timed != SMOKE_OPERATIONS:
        raise RuntimeError(f"the smoke run failed ({status}): {timed}; see smoke.log")
    wrong = []
    for record in records:
        for op in record["operations"]:
            name, seconds = op["operation"], op.get("best_seconds")
            where = f"{record['library']} {name}"
            if not (isinstance(seconds, float) and math.isfinite(seconds) and seconds > 0):
                wrong.append(f"{where}: time {seconds}")
            ceiling = SMOKE_CEILINGS.get(name)
            error = op.get("error")
            if ceiling is not None and not (isinstance(error, float) and 0 <= error < ceiling):
                wrong.append(f"{where}: error {error} (ceiling {ceiling})")
            termination = (op.get("solver") or {}).get("termination")
            if termination not in (None, "iteration_limit", "converged", "tolerance"):
                wrong.append(f"{where}: terminated by {termination}")
    if wrong:
        raise RuntimeError("the smoke run's numbers are wrong: " + "; ".join(wrong))


def _log_step(out: Path, name: str, status: str) -> None:
    with (out / "steps.jsonl").open("a") as steps:
        steps.write(json.dumps({"step": name, "status": status, "time": time.time()}) + "\n")


def _finished_steps(out: Path) -> set[str]:
    """The steps an earlier attempt of this run finished."""
    path = out / "steps.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    return {e["step"] for e in map(json.loads, lines) if e["status"] == "exit 0"}


# A run continued with --run-name must be the same build on the same kind of machine.
_IMMUTABLE = ("commit", "machine", "wheel", "scripts", "tigre_commit", "constraints", "plan")


def _record_attempt(out: Path, run: dict) -> None:
    """Record this attempt of ``run``; refuse to continue a run of another build or machine."""
    first = out / "run.json"
    if first.exists():
        earlier = json.loads(first.read_text())
        differs = [k for k in _IMMUTABLE if earlier.get(k) != run.get(k)]
        if differs:
            raise RuntimeError(f"run {run['name']} was another build or machine: {differs}")
    else:
        first.write_text(json.dumps(run, indent=2))
    with (out / "attempts.jsonl").open("a") as attempts:
        attempts.write(json.dumps({"time": time.time(), "command": run["command"]}) + "\n")


CGLS_BUDGETS = (10, 20, 50)  # fresh solves each: error against time, not one fixed point
WALNUT_BUDGETS = {"cgls": (10, 25, 50), "fista": (10, 20, 50)}


def _walnut_cache(orbits: tuple[int, ...], every: int) -> str:
    return f"/data/cache/walnut1-orbits{''.join(map(str, orbits))}-every{every}.npz"


def _plan(
    gpus: list[int], sizes: list[int], views: int, out: Path
) -> list[tuple[str, list[str], float, bool]]:
    """Every step, most valuable first: (name, command, time limit, whether JAX uses the CPU).

    First the comparisons the paper needs from the fewest and most GPUs: the
    smallest size, the walnut (FDK, CGLS and non-negative least squares at
    several iteration budgets, and the alignment), the larger sizes'
    projections and FDK. Then the GPU counts between, and last the larger
    sizes' CGLS, TIGRE's last of all. ASTRA's CGLS uses one GPU whatever it is
    given, so it runs once per size.
    """
    low, high = min(gpus), max(gpus)
    ends = sorted({low, high})
    middle = [g for g in sorted(set(gpus)) if g not in ends]
    libraries = ("tomojax", "astra", "tigre")
    plan: list[tuple[str, list[str], float, bool]] = []
    projections = ["forward", "forward_siddon", "backproject", "fdk"]

    def scaling(size: int, count: int, library: str, operations: list[str], limit: float) -> None:
        command = [
            "python", "scaling.py", "--size", str(size), "--views", str(views),
            "--repeats", "3", "--cgls-budgets", *map(str, CGLS_BUDGETS),
            "--cgls-repeats", "1" if size <= 512 else "0", "--gpus", str(count),
            "--libraries", library, "--case", f"/tmp/{_case_name(size, views)}",
            "--output", str(out / f"scaling-{size}.json"), "--worker-timeout", str(limit - 60),
            "--operations", *operations,
        ]  # fmt: skip
        tag = "all" if "cgls" in operations and len(operations) > 1 else "+".join(operations)
        plan.append((f"scaling-{size}-{library}-{count}-{tag}", command, limit, False))

    def first_size(count: int) -> None:
        for library in libraries:
            ops = [*projections, "cgls"] if library != "astra" or count == low else projections
            scaling(sizes[0], count, library, ops, 1800)

    def walnut(count: int, *, alignment: bool) -> None:
        fdk = ["--orbits", "2", "--repeats", "3", "--cache", _walnut_cache((2,), 1)]
        orbits = ["--orbits", "1", "2", "3", "--every", "4", "--bin", "2",
                  "--cache", _walnut_cache((1, 2, 3), 4)]  # fmt: skip
        runs = {"fdk": fdk}
        if count == low:  # iterative methods: ASTRA's run on one GPU
            for method, budgets in WALNUT_BUDGETS.items():
                runs[method] = [*orbits, "--method", method, "--budgets", *map(str, budgets)]
        elif count == high:
            runs["cgls"] = [
                *orbits,
                "--method",
                "cgls",
                "--budgets",
                *map(str, WALNUT_BUDGETS["cgls"]),
            ]
        for name, options in runs.items():
            for library in ("tomojax", "astra"):  # a process each: neither holds the GPUs
                if library == "astra" and name != "fdk" and count != low:
                    continue
                tag = f"walnut-{name}-{library}-{count}"
                command = ["python", "walnut.py", "/data/Walnut1", *options,
                           "--gpus", str(count), "--libraries", library,
                           "--output", str(out / f"{tag}.json")]  # fmt: skip
                if name == "fdk":
                    command += ["--method", "fbp"]
                plan.append((tag, command, 2400, library == "astra"))
        if alignment:
            tag = f"walnut-align-{count}"
            command = ["python", "walnut_alignment.py", "/data/Walnut1", "--levels", "4,2",
                       "--gpus", str(count), "--output", str(out / f"{tag}.json"),
                       "--slices", str(out / f"{tag}.npz"),
                       "--checkpoint", str(out / f"{tag}.ckpt"),
                       "--cache", _walnut_cache((1, 2, 3), 4)]  # fmt: skip
            plan.append((tag, command, 2400, False))

    for count in ends:
        first_size(count)
    for count in ends:
        walnut(count, alignment=True)
    for size in sizes[1:]:
        for count in ends:
            for library in libraries:
                scaling(size, count, library, projections, 3600)
    for count in middle:
        first_size(count)
        walnut(count, alignment=False)
        for size in sizes[1:]:
            for library in libraries:
                scaling(size, count, library, projections, 3600)
    for size in sizes[1:]:
        for library in libraries:
            counts = [low] if library == "astra" else [*ends, *middle]
            for count in counts:
                scaling(size, count, library, ["cgls"], 5400)
    return plan


@app.local_entrypoint()
def main(
    *,
    gpu: str = "H100:4",
    gpus: str = "",
    sizes: str = "512,1024",
    views: int = 720,
    max_hours: float = 3.0,
    run_name: str = "",
    only_check: bool = False,
    preflight: bool = False,
) -> None:
    """Run the comparisons on a ``gpu`` machine; copy the results to bench/results.

    ``gpus`` defaults to 1, 2, ... up to the machine's count, by doubling;
    ``run_name`` continues an earlier run (its finished steps are kept).
    ``preflight`` runs only the tests, the smoke run and the kernels' own tests
    on the machine, saving their reports, before paying for the whole run.
    """
    import sys

    machine = int(gpu.split(":")[1]) if ":" in gpu else 1
    counts = [int(g) for g in gpus.split(",")] if gpus else _doublings(machine)
    if max(counts) != machine or min(counts) < 1:
        raise SystemExit(f"--gpus {counts} must run up to the {machine} GPUs of {gpu}")
    if only_check:
        print(check.with_options(gpu=gpu).remote(machine)[-3000:])
        return
    kind = "preflight-" if preflight else ""
    name = run_name or f"{kind}{time.strftime('%Y%m%d-%H%M')}-{gpu.replace(':', 'x')}-{COMMIT}"
    files = [*sorted((ROOT / "bench").glob("*.py")), *sorted((ROOT / "tests").glob("*.py"))]
    run = {
        "name": name,
        "commit": COMMIT,
        "machine": gpu,
        "command": sys.argv,
        "max_hours": max_hours,
        "wheel": {WHEEL.name: hashlib.sha256(WHEEL.read_bytes()).hexdigest()},
        "scripts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "tigre_commit": TIGRE_COMMIT,
        "constraints": PINS.read_text(),
        "preflight": preflight,
        # What the steps measure; a resumed run must ask for the same.
        "plan": {
            "sizes": sorted(int(s) for s in sizes.split(",")),
            "views": views,
            "gpus": counts,
            "cgls_budgets": list(CGLS_BUDGETS),
            "walnut_budgets": {k: list(v) for k, v in WALNUT_BUDGETS.items()},
            "preflight": preflight,
        },
    }
    sizes_list = sorted(int(s) for s in sizes.split(","))
    fetch_walnut.remote()
    if not preflight:  # the smoke case needs no phantoms
        make_cases.remote(sizes_list, views)
    hard_limit = int(max_hours * 3600) + 60  # the deadline inside keeps 5 minutes to save
    try:
        benchmark.with_options(gpu=gpu, timeout=hard_limit).remote(counts, sizes_list, views, run)
    finally:
        target = ROOT / "bench" / "results"
        target.mkdir(parents=True, exist_ok=True)
        fetch = [
            "modal",
            "volume",
            "get",
            "--force",
            "tomojax-walnut",
            f"results/{name}",
            str(target),
        ]
        for attempt in range(3):
            if subprocess.run(fetch, check=False).returncode == 0:
                print(f"results in {target / name}")
                break
            time.sleep(30 * (attempt + 1))
        else:
            print(f"download failed: the results stay in the volume at results/{name}")
            raise SystemExit(1)


def _doublings(machine: int) -> list[int]:
    counts = [1]
    while counts[-1] * 2 <= machine:
        counts.append(counts[-1] * 2)
    return counts if counts[-1] == machine else [*counts, machine]
