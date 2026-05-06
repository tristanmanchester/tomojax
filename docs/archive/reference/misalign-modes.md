# Misalignment modes

The `tomojax-misalign` command can generate deterministic,
reproducible misalignment schedules in addition to random per-view
jitter. Use deterministic schedules when you need to test alignment
against known drift patterns — linear ramps, sinusoidal oscillations,
sudden steps, or box pulses.

## When to use schedules vs random jitter

- **Random jitter** (`--rot-deg`, `--trans-px`) produces independent
  per-view perturbations. It's good for general robustness testing.
- **Deterministic schedules** (`--pert`, `--spec`) produce structured
  patterns that mimic real-world mechanical behavior: stage drift,
  thermal expansion, or sudden shifts from collisions.
- **Combined** — use `--with-random` to add random jitter on top of
  deterministic schedules.

## Degrees of freedom

Schedules target individual DOFs:

| DOF | Description | Default unit |
|-----|-------------|-------------|
| `angle` | Offset added to scan angles (`thetas_deg`) | degrees |
| `alpha` | Tilt around x-axis | degrees (stored as radians) |
| `beta` | Tilt around y-axis | degrees (stored as radians) |
| `phi` | In-plane rotation around z-axis | degrees (stored as radians) |
| `dx` | Horizontal detector translation | pixels (scaled by `du`) |
| `dz` | Vertical detector translation | pixels (scaled by `dv`) |

Aliases: `x` or `u` map to `dx`; `y` or `v` map to `dz`.

## Schedule shapes

All shapes operate within a window. If no window is given, they cover
the full scan.

### Linear

Applies a linear ramp within the window:

```bash
# Angle drift from 0 to +5 degrees across the full scan
--pert angle:linear:delta=5deg

# dx ramp from 0 to +3 pixels between 0 and 90 degrees
--pert dx:linear:start=0px,end=3px,start_deg=0,end_deg=90
```

| Key | Description |
|-----|-------------|
| `delta` | Total change from start to end |
| `start`, `end` | Override delta with explicit start/end values |

### Sin-window

A single-lobe sine within the window:
`amp * sin(π * t)` where `t ∈ [0, 1]`:

```bash
# dx oscillation peaking at +5 pixels at mid-scan
--pert dx:sin-window:amp=5px
```

Aliases: `sin`, `sinwin`.

### Step

A sudden shift at a specific angle or index, held to the end (or for
a width):

```bash
# dx jumps to +5 pixels at 90 degrees, held to end
--pert dx:step:at=90deg,to=5px

# dz shifts by -3 pixels for 4 views starting at index 10
--pert dz:step:domain=index,at_index=10,delta=-3px,width_index=4
```

| Key | Description |
|-----|-------------|
| `at` (or `at_deg`) | Angle where the step occurs |
| `to` | Absolute target value |
| `delta` | Relative shift |
| `width_deg` / `until_deg` | Duration of the step |

### Box

A box pulse — step up, then step down:

```bash
# dz drops by 4 pixels between 60 and 80 degrees
--pert dz:box:at=60deg,width_deg=20,delta=-4px
```

Uses the same keys as `step`, plus `width_deg` or `width_index` to
define the pulse duration.

## Composing schedules

You can combine multiple schedules by repeating `--pert`:

```bash
uv run tomojax-misalign --data data/sim.nxs --out runs/combo.nxs \
  --pert angle:linear:delta=5deg \
  --pert dx:sin-window:amp=5px
```

Schedules are added in the order given. A `step:to=...` sets an
absolute level by computing the needed delta at that point.

## JSON spec files

For complex schedules, write them in a JSON file and pass `--spec`:

**Per-DOF format:**

```json
{
  "angle": [{"kind": "linear", "delta": "5deg"}],
  "dx": [{"kind": "sin-window", "amp": "5px"}],
  "dz": [{"kind": "step", "at": "90deg", "to": "5px"}]
}
```

**Unified list format:**

```json
{
  "schedules": [
    {"dof": "angle", "kind": "linear", "delta": "5deg"},
    {"dof": "dx", "kind": "sin-window", "amp": "5px"}
  ]
}
```

## Window parameters

By default, schedules use angle-domain windowing (nearest views to
specified degrees). You can switch to index-domain:

| Domain | Window keys |
|--------|------------|
| Angle (default) | `start_deg`, `end_deg` |
| Index | `start_index`, `end_index`, `domain=index` |

Index-domain values clamp to `[0, n_views-1]`.

## Output metadata

The misaligned output file stores:

- Modified `thetas_deg` (after `angle` schedule offsets)
- `processing/tomojax/align/thetas` — per-view 5-DOF parameters
  `[alpha, beta, phi, dx, dz]` in radians (rotation) and world
  units (translation)
- `processing/tomojax/align/angle_offset_deg` — per-view angle
  offsets (if angle schedules were used)
- `misalign_spec_json` attribute — normalized spec for
  reproducibility

## Next steps

- [misalign CLI](../cli/misalign.md) — full command reference
- [End-to-end tutorial](../tutorials/end-to-end.md) — using
  misalignment in a complete workflow
