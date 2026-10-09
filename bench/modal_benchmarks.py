"""Run the multi-GPU comparisons with ASTRA and TIGRE on a Modal machine.

``scaling.py`` (cone projection, its transpose, FDK and CGLS on synthetic
data) and ``walnut.py`` (FDK and non-negative least squares on FIPS walnut 1,
TomoJAX against ASTRA) run once for each GPU count, on one machine with the
largest count. Build the wheel first; walnut 1 is fetched from Zenodo into the
``tomojax-walnut`` Modal volume on the first run (about 6 GB), and kept.

    uv build --wheel
    modal run bench/modal_benchmarks.py --gpu H100:4 --gpus 1,2,4 --size 512 --views 720
    modal run bench/modal_benchmarks.py --gpu L4:2 --only-check  # the tests alone

Results go to ``bench/results/modal-<gpu>-<size>.json``. Modal bills by GPU time:
four H100s cost several pounds an hour, so start with a small ``--size``.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
WALNUT_ZIP = "https://zenodo.org/records/2686726/files/Walnut1.zip"
TIGRE_COMMIT = "6b0951a"

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "unzip")
    .pip_install("numpy>=2,<2.3", "cython>=3", "setuptools", "scipy", "imageio", "tqdm")
    .pip_install("astra-toolbox>=2.5,<2.6")
    # TIGRE builds for every GPU architecture nvcc knows; this is the commit tested locally.
    .run_commands(
        "git clone https://github.com/CERN/TIGRE /opt/TIGRE",
        f"cd /opt/TIGRE && git checkout {TIGRE_COMMIT} && pip install --no-build-isolation .",
    )
)
if modal.is_local():  # the container imports this module too, without the files
    wheels = sorted((ROOT / "dist").glob("tomojax-*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise SystemExit("build the wheel first: uv build --wheel")
    image = image.add_local_file(wheels[-1], f"/root/{wheels[-1].name}", copy=True)
    image = image.run_commands(f"pip install '/root/{wheels[-1].name}[cuda12]'")
    ignore = ["results/**", "reference/**", "phantoms/**", "**/__pycache__/**"]
    image = image.add_local_dir(ROOT / "bench", "/root/bench", copy=True, ignore=ignore)
    image = image.pip_install("pytest").add_local_file(
        ROOT / "tests" / "test_devices.py", "/root/tests/test_devices.py", copy=True
    )
app = modal.App("tomojax-benchmarks", image=image)
data = modal.Volume.from_name("tomojax-walnut", create_if_missing=True)


@app.function(volumes={"/data": data}, timeout=3600)
def fetch_walnut() -> None:
    """Walnut 1's projections and reference reconstruction, unzipped into the volume."""
    import subprocess

    if Path("/data/Walnut1/Projections").exists():
        return
    subprocess.run(["curl", "-L", "--fail", "-o", "/tmp/w.zip", WALNUT_ZIP], check=True)
    subprocess.run(["unzip", "-q", "/tmp/w.zip", "-d", "/data"], check=True)
    data.commit()


@app.function(timeout=1800)
def check() -> str:
    """The multi-device tests, on this machine's GPUs: pytest's summary."""
    import subprocess

    command = ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_devices.py"]
    run = subprocess.run(command, cwd="/root", capture_output=True, text=True, check=False)
    return run.stdout[-3000:] + run.stderr[-2000:]


@app.function(volumes={"/data": data}, timeout=6 * 3600, memory=256 * 1024)
def benchmark(gpus: list[int], size: int, views: int, iterations: int) -> dict:
    """Both comparisons for each GPU count; their JSON records."""
    import subprocess

    counts = [str(g) for g in gpus]
    scaling = "/tmp/scaling.json"
    common = ["--size", str(size), "--views", str(views), "--iterations", str(iterations)]
    subprocess.run(
        ["python", "scaling.py", *common, "--gpus", *counts, "--output", scaling],
        cwd="/root/bench",
        check=True,
    )
    results: dict = {"scaling": json.loads(Path(scaling).read_text()), "walnut": []}
    runs = {
        "fdk": ["--orbits", "2"],
        "nnls": ["--orbits", "1", "2", "3", "--every", "4", "--bin", "2", "--method", "fista"],
    }
    for count in counts:
        for name, options in runs.items():
            command = ["python", "walnut.py", "/data/Walnut1", *options, "--astra", "--gpus", count]
            if name == "nnls":
                command += ["--iterations", str(iterations)]
            run = subprocess.run(
                command, cwd="/root/bench", capture_output=True, text=True, check=False
            )
            record = {"run": name, "gpus": int(count)}
            if run.returncode:
                record["failed"] = (run.stderr.strip().splitlines() or ["no output"])[-1]
            else:
                record |= json.loads(run.stdout[run.stdout.index("{") :])
            print(json.dumps(record)[:400], flush=True)
            results["walnut"].append(record)
    return results


@app.local_entrypoint()
def main(
    *,
    gpu: str = "H100:4",
    gpus: str = "1,2,4",
    size: int = 512,
    views: int = 720,
    iterations: int = 20,
    only_check: bool = False,
) -> None:
    """Run the comparisons on ``gpu`` machines and write the results under bench/results.

    The multi-device tests run first, on the same GPUs; ``--only-check`` stops there.
    """
    print(check.with_options(gpu=gpu).remote())
    if only_check:
        return
    fetch_walnut.remote()
    counts = [int(g) for g in gpus.split(",")]
    results = benchmark.with_options(gpu=gpu).remote(counts, size, views, iterations)
    results["machine"] = gpu
    out = ROOT / "bench" / "results" / f"modal-{gpu.replace(':', 'x')}-{size}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
