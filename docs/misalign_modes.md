# Misalignment Modes and Schedules (misalign CLI)

The `misalign` CLI can now generate deterministic, reproducible misalignments in addition to the original random per‑view jitter. Deterministic schedules are composable and target individual degrees of freedom (DOFs):

- DOFs: `angle`, `alpha`, `beta`, `phi`, `dx`, `dz`
  - `angle`: adds an offset (in degrees) to the scan angles (`thetas_deg`) before building poses.
  - `alpha`, `beta`, `phi`: small rotations (input in degrees; stored as radians).
  - `dx`, `dz`: detector translations in pixels (u and v axes; scaled by `du` and `dv` to world units).

You describe each schedule with `--pert` or an external JSON `--spec` file. Multiple schedules can be combined additively, in the order specified.

## Quick Examples

- Linear drift in angles 0 → +5° across the scan:
  - `pixi run misalign --data data/sim.nxs --out runs/mis_angle_lin.nxs --pert angle:linear:delta=5deg`

- Sinusoidal `dx` drift peaking mid‑scan at +5 px:
  - `pixi run misalign --data data/sim.nxs --out runs/mis_dx_sin.nxs --pert dx:sin-window:amp=5px`

- Sudden `dx` shift to +5 px at 90° (held to end):
  - `pixi run misalign --data data/sim.nxs --out runs/mis_dx_step.nxs --pert dx:step:at=90deg,to=5px`

Combine schedules by repeating `--pert`:

```
pixi run misalign --data data/sim.nxs --out runs/mis_combo.nxs \
  --pert angle:linear:delta=5deg \
  --pert dx:sin-window:amp=5px
```

## CLI Reference

- `--pert <dof>:<shape>[:k=v[,k=v...]]`
  - DOFs: `angle`, `alpha`, `beta`, `phi`, `dx`, `dz` (aliases: `x|u -> dx`, `y|v -> dz`).
  - Shapes: `linear`, `sin-window` (`sin`/`sinwin`), `step`, `box`.
  - Units on values via suffix: `deg`, `px` (default by DOF: angles/rotations are degrees; translations are pixels).
  - Domain: by default, parameters refer to scan angle (`domain=angle`). Index-based domain is supported with `domain=index`.

- `--spec <path.json>`: JSON file describing schedules (see schema below).

- `--with-random`: add the original random jitter on top of deterministic schedules. If omitted, schedules are used alone (no randomness).

Other options remain unchanged (`--rot-deg`, `--trans-px`, `--seed`, `--poisson`, ...). When `--with-random` is set, `--rot-deg` and `--trans-px` control the random ranges.

## Shapes and Parameters

All shapes operate on a window. If no window is given, they cover the full scan.

- Window parameters (angle domain): `start_deg`, `end_deg`
- Window parameters (index domain): `start_index`, `end_index`

Angle domain uses the nearest views to the specified degrees. Index domain clamps to `[0, n_views-1]`.

### linear

Apply a linear ramp within the window.

- Keys: `delta`, optional `start`, `end` (if `start`/`end` supplied, they override `delta`).
  - Examples:
    - `angle:linear:delta=5deg` → 0° at first view, +5° at last.
    - `dx:linear:start=0px,end=3px,start_deg=0,end_deg=90` → 0→3 px between 0–90°, clamped outside.

### sin-window (sinwin, sin)

Single-lobe sine within the window: `amp * sin(pi * t)` with `t ∈ [0, 1]` across the window.

- Keys: `amp`
  - Example: `dx:sin-window:amp=5px` → 0 px at 0°, +5 px at 90°, back to 0 px at 180°.

### step

Step at a specific angle or index; holds until the end or for a width.

- Keys:
  - Angle domain: `at` (alias `at_deg`), optional `width_deg` or `until_deg`
  - Index domain: `at_index`, optional `width_index` or `until_index`
  - Value: either `to` for an absolute target, or `delta` for a relative shift.
  - Examples:
    - `dx:step:at=90deg,to=5px` → `dx=+5 px` from ~90° to the end.
    - `dz:step:domain=index,at_index=10,delta=-3px,width_index=4` → −3 px for 4 views.

### box

Box pulse: step up by `delta`, then down after a width (or at `until`).

- Keys: same as `step` plus `width_deg`/`width_index`. Example:
  - `dz:box:at=60deg,width_deg=20,delta=-4px` → −4 px between ~60° and ~80°.

## JSON Spec File

Two accepted schemas:

1) Per-DOF lists

```json
{
  "angle": [{"kind": "linear", "delta": "5deg"}],
  "dx": [{"kind": "sin-window", "amp": "5px"}],
  "dz": [{"kind": "step", "at": "90deg", "to": "5px"}]
}
```

2) Unified list

```json
{
  "schedules": [
    {"dof": "angle", "kind": "linear", "delta": "5deg"},
    {"dof": "dx", "kind": "sin-window", "amp": "5px"}
  ]
}
```

## Metadata and Outputs

The output file persists:

- Modified `thetas_deg` (angles after applying `angle` schedules).
- `processing/tomojax/align/thetas`: per-view 5‑DOF `[alpha,beta,phi,dx,dz]` in radians (rot) and world units (trans).
- If schedules were used:
  - `processing/tomojax/align/angle_offset_deg`: per-view angle offset applied (degrees).
  - `processing/tomojax/align` attribute `misalign_spec_json`: normalized spec for reproducibility.

## Notes

- Units: rotations (`alpha`,`beta`,`phi`) accept degrees (default) and are stored in radians; translations (`dx`,`dz`) accept pixels and are converted using `du`/`dv`.
- Composition: schedules are added in the order given; `step:to=...` sets an absolute level by internally computing the needed delta at that point.
- Randomness: if any `--pert`/`--spec` is present, random jitter is disabled by default. Use `--with-random` to enable it and combine.

