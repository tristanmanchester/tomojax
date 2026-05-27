# Alignment guide

TomoJAX alignment estimates geometry or pose corrections while reconstructing
the volume. Start with the mode that matches your problem, then check the
output metadata to decide whether the correction is physically meaningful or
just useful for reconstruction.

## Choose an alignment mode

`tomojax align` has three modes. Use `pose` for per-projection sample motion,
`cor` for detector-centre calibration, and `auto` when you need to combine
setup and pose degrees of freedom.

| Problem | Recommended mode | Typical command |
| --- | --- | --- |
| Sample or object motion changes from projection to projection | `pose` | `tomojax align --data scan.nxs --mode pose --out aligned.nxs` |
| Detector centre or centre-of-rotation is wrong | `cor` | `tomojax align --data scan.nxs --mode cor --out aligned.nxs` |
| Mild setup error and pose motion are both plausible | `auto` | `tomojax align --data scan.nxs --mode auto --gauge-policy anchor_mean --out aligned.nxs` |
| Reference elevation or detector-v shift is uncertain | Inspect manually | `det_v_px` is not a reliably recoverable alignment target. |

## Use 5-DOF pose correction first

The default `pose` mode optimizes one 5-DOF pose vector per projection:
`alpha`, `beta`, `phi`, `dx`, and `dz`. Use this for scans where the sample moved during acquisition.

```bash
uv run tomojax align \
  --data corrected.nxs \
  --mode pose \
  --out aligned.nxs
```

Use `--quality reference` for a slower, higher-fidelity solve. Use explicit
levels when you want a specific coarse-to-fine schedule:

```bash
uv run tomojax align \
  --data corrected.nxs \
  --mode pose \
  --quality reference \
  --levels 4 2 1 \
  --out aligned.nxs
```

The aligned dataset stores the reconstruction and recovered parameters.
Inspect it with:

```bash
uv run tomojax inspect aligned.nxs
```

## Correction quality vs physical calibration

Pose-only correction can absorb some global setup errors and still produce a
good reconstruction, but that doesn't mean the recovered pose parameters are
a calibrated description of the machine.

- For the cleanest reconstruction, start with `--mode pose`.
- For a physically interpretable detector-centre correction, use `--mode cor`.
- For both setup and pose correction, use `--mode auto` with an explicit gauge
  policy.

## Use COR mode for detector-centre calibration

Use `cor` mode when the main problem is a detector-u or centre-of-rotation
offset rather than sample motion.

```bash
uv run tomojax align \
  --data corrected.nxs \
  --mode cor \
  --out aligned.nxs
```

Explicit COR correction recovers large detector-u offsets better than relying
on pose-only correction to absorb them. Use `cor` when you care about
calibration, and `pose` when you care about per-projection motion.

## Use mixed setup and pose as expert mode

`auto` mode combines setup and pose stages. Because setup and pose parameters
can represent similar image changes, mixed correction has gauge ambiguity.
You must choose how to handle that ambiguity.

```bash
uv run tomojax align \
  --data corrected.nxs \
  --mode auto \
  --gauge-policy anchor_mean \
  --out aligned.nxs
```

Gauge policies:

- `anchor_mean`: Anchors mean translation so setup and pose don't drift
  together. Best for reconstruction quality.
- `prior_required`: Requires physical setup priors.
- `diagnose_only`: Produces diagnostics without treating the result as a
  calibrated correction.
- `reject`: Fails instead of running an ambiguous mixed correction.

## Smooth pose models

The default pose model is `per_view`, which optimizes an independent 5-DOF
vector for every projection. Use a smooth model when you expect the motion to
change smoothly over the scan.

```bash
uv run tomojax align \
  --data corrected.nxs \
  --mode pose \
  --pose-model spline \
  --knot-spacing 8 \
  --out aligned.nxs
```

Smooth models reduce degrees of freedom but can hide abrupt jumps or outlier
views.

## Known hard cases

The solver handles many noisy pose cases well, but some failure modes remain:

- Abrupt jumps may need jump-aware pose handling.
- Short bursts of bad views may need robust loss or bad-view detection.
- Large combined setup and pose errors may need staged initialization.
- Detector-v or sample-elevation reference shifts are physically ambiguous and
  are not reliably recoverable.

## Evidence from the 128^3 sweep

A 128^3 synthetic sweep tested 32 phantom scenarios. Results:

- 25 strong, 5 usable, 1 partial, 1 poor.

The poor case was a detector-v/reference-elevation shift (not reliably
recoverable). The partial case was an abrupt jump (needs a jump-aware
workflow).

## Next steps

After alignment, inspect the output and compare against the naive
reconstruction. See [`support-matrix.md`](support-matrix.md) for supported
workflows and [`known-limitations.md`](known-limitations.md) for hard cases.
