from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional
    psutil = None


def run(cmd: list[str]) -> dict:
    t0 = time.perf_counter()
    rss0 = psutil.Process().memory_info().rss if psutil else 0
    env = os.environ.copy()
    env.setdefault("TOMOJAX_PROGRESS", "0")
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t1 = time.perf_counter()
    rss1 = psutil.Process().memory_info().rss if psutil else 0
    return {
        "cmd": cmd,
        "rc": proc.returncode,
        "secs": round(t1 - t0, 3),
        "rss_delta": int(max(0, rss1 - rss0)),
        "stdout": proc.stdout[-2000:],
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
