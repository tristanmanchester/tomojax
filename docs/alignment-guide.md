# Alignment guide

TomoJAX alignment estimates geometry or pose corrections while reconstructing
the volume. Start with the workflow that matches the problem you believe you
have, then use the diagnostics and output metadata to decide whether the
correction is physically interpretable or only useful for reconstruction.

## Choose an alignment mode

The public `tomojax align` command exposes three practical alignment paths.
Use `pose` for per-projection sample motion, use `cor` for detector-centre
calibration, and use `auto` only when you deliberately combine setup and pose
degrees of freedom.

| Problem | Recommended mode | Typical command |
| --- | --- | --- |
| Sample or object motion changes from projection to projection | `pose` | `tomojax align --data scan.nxs --mode pose --out aligned.nxs` |
| Detector centre or centre-of-rotation is wrong | `cor` | `tomojax align --data scan.nxs --mode cor --out aligned.nxs` |
| Mild setup error and pose motion are both plausible | `auto` | `tomojax align --data scan.nxs --mode auto --gauge-policy anchor_mean --out aligned.nxs` |
| Reference elevation or detector-v shift is uncertain | Inspect manually | Don't treat `det_v_px` as a normal recoverable alignment target. |

## Use 5-DOF pose correction first

The default `pose` mode optimizes one 5-DOF pose vector per projection:
`alpha`, `beta`, `phi`, `dx`, and `dz`. This is the main practical correction
path for scans where the sample moved during acquisition.

```bash
uv run tomojax align \
  --data corrected.nxs \
  --mode pose \
  --out aligned.nxs
```

Use `--quality reference` when you want the slower reference posture, and use
explicit levels when you want the same coarse-to-fine structure used in the
128^3 system sweep:

```bash
uv run tomojax align \
  --data corrected.nxs \
  --mode pose \
  --quality reference \
  --levels 4 2 1 \
  --out aligned.nxs
```

The aligned dataset stores the corrected reconstruction, the recovered
parameters, the gauge metadata, and the alignment manifest. Inspect it with:

```bash
uv run tomojax inspect aligned.nxs
```

## Separate correction quality from physical calibration

Pose-only correction can absorb some global setup errors and still produce a
good reconstruction. That is useful for recovering the volume, but it does not
mean the recovered pose parameters are a calibrated description of the machine.

Use this rule when you interpret results:

- If you need the cleanest reconstruction, start with `--mode pose`.
- If you need a physically interpretable detector-centre correction, run
  `--mode cor`.
- If you need both setup and pose correction, use `--mode auto` with an
  explicit gauge policy and treat the result as expert output.

## Use COR mode for detector-centre calibration

The `cor` mode optimizes detector-centre geometry directly. Use it when the
dominant problem is a detector-u or centre-of-rotation offset rather than
sample motion.

```bash
uv run tomojax align \
  --data corrected.nxs \
  --mode cor \
  --out aligned.nxs
```

The 128^3 sweep found that explicit COR correction recovered large detector-u
offsets better than asking pose-only correction to absorb them. Use `cor` when
you care about calibration, and use `pose` when you care about per-projection
motion.

## Use mixed setup and pose as expert mode

The `auto` mode combines setup and pose stages. Because setup and pose
parameters can represent similar image changes, mixed correction has gauge
ambiguity. You must choose how TomoJAX handles that ambiguity.

```bash
uv run tomojax align \
  --data corrected.nxs \
  --mode auto \
  --gauge-policy anchor_mean \
  --out aligned.nxs
```

Use these gauge policies:

- `anchor_mean`: Use this for reconstruction-quality mixed correction. It
  anchors mean translation so setup and pose don't drift together.
- `prior_required`: Use this only when you provide physical setup priors.
- `diagnose_only`: Use this when you want diagnostics without promoting the
  result as a calibrated correction.
- `reject`: Use this when scripts must fail instead of running an ambiguous
  mixed setup and pose request.

## Use smooth pose models deliberately

The default pose model is `per_view`, which optimizes an independent 5-DOF
vector for every projection. Use a smooth pose model only when you have reason
to believe the motion changes smoothly over the scan.

```bash
uv run tomojax align \
  --data corrected.nxs \
  --mode pose \
  --pose-model spline \
  --knot-spacing 8 \
  --out aligned.nxs
```

Smooth models can reduce degrees of freedom, but they can also hide abrupt
jumps or outlier views. Use them as a model choice, not as a general quality
upgrade.

## Known hard cases

The current solver handles many random and noisy pose cases well, but some
failure modes need more diagnostics or specialized models.

- Abrupt jumps can need jump-aware pose handling.
- Short bursts of bad views can need robust loss or bad-view detection.
- Large combined setup and pose errors can need staged initialization.
- Detector-v or sample-elevation reference shifts are physically ambiguous and
  are not good default optimization targets.

## Evidence from the 128^3 sweep

The 128^3 synthetic sweep tested 32 phantom scenarios on a CUDA laptop. All
cases completed after rerunning one mixed setup and pose case with an explicit
gauge policy.

The final classification counts were:

- 25 strong cases.
- 5 usable cases.
- 1 partial case.
- 1 poor case.

The poor case was a detector-v/reference-elevation shift, which is not a
normal recoverable alignment target. The partial case was an abrupt jump, which
points to a future jump-aware workflow rather than a general failure of 5-DOF
pose correction.

## Next steps

After alignment, inspect the output dataset and compare the reconstruction
against the naive reconstruction. For more detail about currently supported
workflows, see [`support-matrix.md`](support-matrix.md). For known hard cases,
see [`known-limitations.md`](known-limitations.md).
