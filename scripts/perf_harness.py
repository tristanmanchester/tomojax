from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None

from tomojax.backends import run_command


def _sample_child_rss(child) -> int | None:
    try:
        return int(child.memory_info().rss)
    except Exception:
        return None


def _run_with_child_memory(cmd: list[str], env: dict[str, str]):
    proc = subprocess.Popen(  # nosec B603
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=False,
    )
    try:
        child = psutil.Process(proc.pid)
    except Exception:
        stdout, _ = proc.communicate()
        return {
            "returncode": proc.returncode,
            "stdout": stdout,
            "child_peak_rss": None,
            "child_final_rss": None,
        }
    peak_rss: int | None = None
    final_rss: int | None = None

    def sample() -> None:
        nonlocal peak_rss, final_rss
        rss = _sample_child_rss(child)
        if rss is None:
            return
        final_rss = rss
        peak_rss = rss if peak_rss is None else max(peak_rss, rss)

    sample()
    while True:
        try:
            stdout, _ = proc.communicate(timeout=0.05)
            sample()
            break
        except subprocess.TimeoutExpired:
            sample()

    return {
        "returncode": proc.returncode,
        "stdout": stdout,
        "child_peak_rss": peak_rss,
        "child_final_rss": final_rss,
    }


def run(cmd: list[str]) -> dict:
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["TOMOJAX_PROGRESS"] = "0"
    if psutil is None:
        proc = run_command(cmd, env=env, stdout=-1, stderr=-2, text=True)  # nosec B603
        run_result = {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "child_peak_rss": None,
            "child_final_rss": None,
        }
    else:
        run_result = _run_with_child_memory(cmd, env)
    t1 = time.perf_counter()
    return {
        "cmd": cmd,
        "rc": run_result["returncode"],
        "secs": round(t1 - t0, 3),
        "child_peak_rss": run_result["child_peak_rss"],
        "child_final_rss": run_result["child_final_rss"],
        "stdout": run_result["stdout"][-2000:],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Quick perf sweep for tomojax CLIs")
    ap.add_argument("--data", required=True, help="Path to .nxs dataset")
    ap.add_argument("--outdir", default="runs/perf", help="Output directory for results")
    ap.add_argument("--modes", nargs="+", default=["fbp", "fista"], help="Which algos to run: fbp, fista")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    combos = []
    dtypes = ["fp32", "bf16"]
    ckpts = [True, False]

    for algo in args.modes:
        for dt in dtypes:
            for ck in ckpts:
                combos.append((algo, dt, ck))

    results = []
    for algo, dt, ck in combos:
        out = outdir / f"{algo}_{dt}_{'ck' if ck else 'nock'}.nxs"
        if algo == "fbp":
            cmd = [
                "python",
                "-m",
                "tomojax.cli.recon",
                "--data",
                args.data,
                "--algo",
                "fbp",
                "--filter",
                "ramp",
                "--gather-dtype",
                dt,
                ("--checkpoint-projector" if ck else "--no-checkpoint-projector"),
                "--out",
                str(out),
            ]
        else:
            cmd = [
                "python",
                "-m",
                "tomojax.cli.recon",
                "--data",
                args.data,
                "--algo",
                "fista",
                "--iters",
                "20",
                "--lambda-tv",
                "0.001",
                "--gather-dtype",
                dt,
                ("--checkpoint-projector" if ck else "--no-checkpoint-projector"),
                "--out",
                str(out),
            ]
        res = run(cmd)
        results.append({
            "algo": algo,
            "gather_dtype": dt,
            "checkpoint": ck,
            **res,
        })

    with open(outdir / "perf_results.json", "w") as f:
        json.dump(results, f, indent=2)
    # Print concise table to stdout
    rows = [
        f"{r['algo']:5s} dt={r['gather_dtype']:5s} ck={str(r['checkpoint']):5s} t={r['secs']:7.3f}s rc={r['rc']}"
        for r in results
    ]
    print("\n".join(rows))


if __name__ == "__main__":  # pragma: no cover
    main()
