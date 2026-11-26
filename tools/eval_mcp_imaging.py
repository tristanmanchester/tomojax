"""Run the MCP imaging workflow on synthetic phantoms and score it against their truth.

Each scenario drives a real ``tomojax mcp`` server over stdio, as an agent would: it
builds the phantom with the ``python`` tool, reconstructs or aligns with the workflow
tools, points with ``view`` marks and makes features, and scores the result against
the phantom's ground truth, which stays in the worker:

- ``shell``: a bright shell three voxels thick around a dimmer core (at size 64; the
  phantom scales with ``--size``). A rough mark 12 voxels outside the shell must come
  back flagged with the shell as its nearest structure; marking that and snapping
  lands on the shell; a threshold feature grown from it is scored by Dice against the
  true shell.
- ``centre``: a scan whose recorded rotation centre is 3.7 detector pixels off. The
  centre fit is scored in detector pixels.
- ``pose``: a scan with smooth per-view drift. The translation fit is scored as the
  RMS error in detector pixels, up to the shifts that moving the object explains.

Both fits also report the relative error against the phantom of FBP reconstructions
before and after alignment and with the true geometry (``ideal``). The script prints
one JSON record per scenario and exits 1 when a score misses its bar.

Run it with ``uv run --no-sync python tools/eval_mcp_imaging.py [--size 64]``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

PROTOCOL = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientInfo": {"name": "eval_mcp_imaging", "version": "1"},
    "io.modelcontextprotocol/clientCapabilities": {},
}
SHELL = (16.0, 19.0)  # inner and outer radius of the shell, as fractions of size / 64
BARS = {
    "shell": {
        "miss_flagged": True,
        "nearest_radial_error": 2.0,
        "snapped_radial_error": 1.0,
        "dice": 0.8,
    },
    "centre": {"centre_error_px": 0.1},
    "pose": {"pose_rms_error_px": 0.1},
}
RECON_MARGIN = 0.01  # how much worse than the true geometry an aligned reconstruction may be


class Server:
    """A ``tomojax mcp`` process spoken to with stateless JSON-RPC over stdio."""

    def __init__(self, stdin: IO[str], stdout: IO[str]) -> None:
        self.stdin = stdin
        self.stdout = stdout
        self.serial = 0

    def rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send one request and return its result, failing on a protocol error."""
        self.serial += 1
        row = {"jsonrpc": "2.0", "id": self.serial, "method": method}
        row["params"] = {**params, "_meta": PROTOCOL}
        self.stdin.write(json.dumps(row) + "\n")
        self.stdin.flush()
        for line in self.stdout:
            reply = json.loads(line)
            if reply.get("id") == self.serial:
                if "error" in reply:
                    raise RuntimeError(f"{method}: {reply['error']}")
                return reply["result"]
        raise RuntimeError("the server closed stdout")

    def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        """Call a tool and return its whole structured result, following continuations."""
        result = self.rpc("tools/call", {"name": name, "arguments": arguments})
        if result["isError"]:
            raise RuntimeError(f"{name}: {result['content'][0]['text']}")
        data = result["structuredContent"]
        chunks, uri = [], data.get("continuation")
        while uri:
            item = json.loads(self.rpc("resources/read", {"uri": uri})["contents"][0]["text"])
            chunks.append(item["data"])
            uri = item["next"]
        return json.loads("".join(chunks)) if chunks else data

    def python(self, code: str) -> Any:
        """Run code in the worker and return the value of its last expression."""
        return self.call("python", code=code)["value"]


@contextmanager
def server(directory: Path) -> Iterator[Server]:
    """Start ``tomojax mcp`` in ``directory`` and stop it on exit."""
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(directory)}
    command = [sys.executable, "-m", "tomojax.cli.main", "mcp"]
    with subprocess.Popen(command, env=env, cwd=directory, stdin=-1, stdout=-1, text=True) as p:
        stdin, stdout = p.stdin, p.stdout
        assert stdin is not None
        assert stdout is not None
        try:
            yield Server(stdin, stdout)
        finally:
            stdin.close()
            p.wait(timeout=60)


def radial_error(at: list[float], size: int) -> float:
    """How far a position lies from the middle of the shell's wall, in voxels."""
    centre = (size - 1) / 2
    inner, outer = (r * size / 64 for r in SHELL)
    return abs(sum((v - centre) ** 2 for v in at) ** 0.5 - (inner + outer) / 2)


def shell(s: Server, size: int) -> dict[str, Any]:
    """Point at a thin shell, recover from a miss, and segment it."""
    inner, outer = (r * size / 64 for r in SHELL)
    recon = s.python(f"""
truth = tj.simulate("sphere", size={size}, n_views={2 * size})
z, y, x = np.mgrid[:{size}, :{size}, :{size}] - ({size} - 1) / 2
r = np.sqrt(z**2 + y**2 + x**2)
truth_shell = (r >= {inner}) & (r < {outer})
phantom = (0.4 * (r < {inner}) + truth_shell).astype(np.float32)
lines = np.asarray(tj.project(truth.scan.geometry, phantom))
lines = lines + 0.01 * lines.max() * np.random.default_rng(0).standard_normal(lines.shape)
register(tj.Scan(lines.astype(np.float32), truth.scan.geometry))""")
    volume = s.call("reconstruct", source=recon, method="fbp")["handle"]
    middle = size // 2
    window = {"source": volume, "axis": "z", "index": middle}
    # The walkthrough's miss: a point 12 voxels (at size 64) outside the shell, on the
    # row through the centre.
    rough = [middle, middle, round((size - 1) / 2 - outer - 12 * size / 64)]
    first = s.call("view", **window, marks=[{"type": "point", "at": rough}])
    report = first["marks"][0]["panels"][0]
    nearest = report.get("nearest", {}).get("at", rough)
    second = s.call("view", **window, marks=[{"type": "point", "at": nearest, "snap": "peak"}])
    snapped = second["marks"][0]["panels"][0]
    seed = [round(v) for v in snapped["snapped"]]
    found = s.python(f"""
from scipy import ndimage
v = np.asarray(handles[{volume!r}].volume)
seed = tuple({seed})
core = float(np.median(v[{middle} - 1 : {middle} + 2, {middle} - 1 : {middle} + 2,
                         {middle} - 1 : {middle} + 2]))
masks = {{}}
for fraction in (0.4, 0.5, 0.6):
    labels, _ = ndimage.label(v > core + fraction * (float(v[seed]) - core))
    masks[fraction] = labels == labels[seed]
def dice(a, b):
    return float(2 * (a & b).sum() / (a.sum() + b.sum()))
shell_id = register_feature(masks[0.5], like={volume!r}, name="shell")
{{"feature": shell_id, "dice": dice(masks[0.5], truth_shell),
  "stability": [dice(masks[0.4], masks[0.5]), dice(masks[0.5], masks[0.6])]}}""")
    s.call("view", **window, features=[found["feature"]])
    return {
        "miss_flagged": "warning" in report,
        "nearest_radial_error": round(radial_error(nearest, size), 2),
        "snapped_radial_error": round(radial_error(snapped["snapped"], size), 2),
        "snapped_value": snapped["value"],
        "dice": round(found["dice"], 4),
        "threshold_stability": [round(d, 4) for d in found["stability"]],
    }


def reconstruction_errors(s: Server, **scans: str) -> dict[str, float]:
    """Relative L2 error against ``phantom`` of an FBP reconstruction of each scan."""
    volumes = {k: s.call("reconstruct", source=x, method="fbp")["handle"] for k, x in scans.items()}
    errors = s.python(f"""
def error(handle):
    v = np.asarray(handles[handle].volume)
    return float(np.linalg.norm(v - phantom) / np.linalg.norm(phantom))
{{k: error(h) for k, h in {volumes!r}.items()}}""")
    return {f"recon_error_{k}": round(v, 4) for k, v in errors.items()}


def centre(s: Server, size: int) -> dict[str, Any]:
    """Fit a rotation centre the scan records 3.7 detector pixels off."""
    scan, ideal = s.python(f"""
from dataclasses import replace
detector = tj.Detector({size}, {size}, 1.0, 1.0)
true_centre = 3.7
moved = tj.simulate("shepp", size={size}, n_views={2 * size},
                    detector=replace(detector, center=(true_centre, 0.0)))
phantom = moved.phantom
nominal = replace(moved.scan.geometry, detector=detector)
[register(tj.Scan(moved.scan.projections, nominal)), register(moved.scan)]""")
    aligned = s.call("align", source=scan, fit="center")
    fitted = s.python(f"""
d = handles[{aligned["handle"]!r}].scan.detector
float(d.center[0] / d.du)""")
    return {
        "true_centre_px": 3.7,
        "fitted_centre_px": round(fitted, 4),
        "centre_error_px": round(abs(fitted - 3.7), 4),
        **reconstruction_errors(s, before=scan, after=aligned["members"]["scan"], ideal=ideal),
    }


def pose(s: Server, size: int) -> dict[str, Any]:
    """Fit smooth per-view drift in both detector directions.

    A parallel scan cannot tell a horizontal shift ``a + b cos(angle) + c sin(angle)``
    from moving the object, nor a constant vertical shift, so the fitted poses are
    compared, and reconstructed, after removing that gauge from their difference.
    """
    scan, ideal = s.python(f"""
sim = tj.simulate("shepp", size={size}, n_views={2 * size})
phantom = sim.phantom
t = np.linspace(0, 1, {2 * size})
true_poses = np.zeros(({2 * size}, 6))
true_poses[:, 3] = 2.0 * np.sin(2 * np.pi * t) + 1.5 * t
true_poses[:, 4] = 1.2 * t - 0.6
moving = tj.Scan(
    np.asarray(tj.project(sim.scan.with_poses(true_poses).geometry, phantom), np.float32),
    sim.scan.geometry,
)
[register(moving), register(moving.with_poses(true_poses))]""")
    aligned = s.call("align", source=scan, fit="translations")
    found = s.python(f"""
fit = handles[{aligned["handle"]!r}]
angle = np.deg2rad(np.asarray(fit.scan.angles))
gauges = [np.stack([np.ones_like(angle), np.cos(angle), np.sin(angle)], 1),
          np.ones((len(angle), 1))]
poses = np.array(fit.poses)
for column, basis in zip((3, 4), gauges):
    miss = poses[:, column] - true_poses[:, column]
    poses[:, column] -= basis @ np.linalg.lstsq(basis, miss, rcond=None)[0]
miss = (poses - true_poses)[:, 3:5] / fit.scan.detector.du
{{"error": float(np.sqrt(np.mean(miss**2))),
  "drift": float(np.sqrt(np.mean(true_poses[:, 3:5]**2)) / fit.scan.detector.du),
  "aligned": register(fit.scan.with_poses(poses))}}""")
    return {
        "true_pose_rms_px": round(found["drift"], 4),
        "pose_rms_error_px": round(found["error"], 4),
        **reconstruction_errors(s, before=scan, after=found["aligned"], ideal=ideal),
    }


SCENARIOS: dict[str, Callable[[Server, int], dict[str, Any]]] = {
    "shell": shell,
    "centre": centre,
    "pose": pose,
}


def misses(name: str, scores: dict[str, Any]) -> list[str]:
    """The scores of one scenario that miss their bars."""
    found = []
    for key, bar in BARS[name].items():
        value = scores[key]
        if isinstance(bar, bool):
            ok = value is bar
        else:
            ok = value >= bar if key == "dice" else value <= bar
        if not ok:
            found.append(f"{name}.{key} = {value} (bar {bar})")
    if "recon_error_after" in scores and (
        scores["recon_error_after"] > scores["recon_error_ideal"] + RECON_MARGIN
    ):
        found.append(
            f"{name}.recon_error_after = {scores['recon_error_after']} (bar: within "
            f"{RECON_MARGIN} of recon_error_ideal, {scores['recon_error_ideal']})"
        )
    return found


def main(argv: list[str] | None = None) -> int:
    """Run the requested scenarios, print their scores, and report any missed bar."""
    parser = argparse.ArgumentParser(
        description="Score the MCP imaging workflow on synthetic phantoms."
    )
    parser.add_argument("--size", type=int, default=64, help="phantom edge in voxels")
    parser.add_argument("--scenario", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    args = parser.parse_args(argv)
    failed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="eval-mcp-imaging-") as tmp, server(Path(tmp)) as s:
        for name in args.scenario:
            scores = SCENARIOS[name](s, args.size)
            print(json.dumps({"scenario": name, "size": args.size, **scores}), flush=True)
            failed += misses(name, scores)
    for line in failed:
        print(f"MISSED {line}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
