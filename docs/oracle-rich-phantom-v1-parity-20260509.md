The focused diagnosis is: **v2’s fixed-truth Schur alignment is now working, but the stopped loop is not v1-parity yet.** The stopped loop is still feeding Schur a preview volume that is in the wrong gauge or too degraded to support detector-shift recovery.

The two most important implementation mismatches I found are:

```text
1. The Otsu foreground mask is leaking into FISTA preview reconstruction.
   It is not only used for the alignment loss.

2. The “multires sidecar” attempt carries geometry between 32→64→128,
   but not the reconstructed volume, and the coarse sidecars are not
   forward-consistent under the v2 projector.
```

So the current result does **not** mean “v2 gradient det_u recovery cannot work.” It means:

```text
true volume + Otsu L2 Schur
    → det_u recovery works

v2 stopped preview + Otsu L2 Schur
    → preview volume is already a bad/wrong-gauge reconstruction
```

That is a stopped-loop implementation problem, not a Schur geometry-solver problem.

## 1. Plain-English diagnosis

The rich PHANTOM94 result splits the system cleanly.

Fixed truth works:

```text
128³ / 128 views / fixed_synthetic_truth / Otsu L2
det_u RMSE ≈ 0.057864 px
classification = independent_projection_losses_consistent
```

That is a real oracle result. It says the following are basically coherent for this supported-only `det_u` problem:

```text
nominal theta handling
sidecar geometry
projector convention
Otsu L2 geometry loss
Schur finite-difference/JTJ path
det_u setup update
```

The stopped production loop fails:

```text
128³ / 128 views / stopped_reconstruction / Otsu L2
det_u RMSE ≈ 4.10057 px
volume NMSE ≈ 0.710293
classification = reconstruction_absorbed_geometry
```

Same-resolution no-skip geometry improves only to about `3.29 px`. No-FISTA-first fails at about `4.70 px`. The first real sidecar multires attempt still fails at roughly `2.17 px → 1.62 px → 2.36 px`.

That pattern says: Schur can recover det_u from a correct volume, but the stopped preview volume does not contain the same alignment signal. It has either absorbed the setup error, been reconstructed under the wrong data support, or both.

The highest-confidence code issue is the Otsu mask path. In `src/tomojax/align/_alternating_inputs.py`, `build_smoke_inputs()` calls `_with_projection_loss_mask()`, and when `projection_loss_mode` starts with `otsu_`, it modifies the global mask. Then the FISTA preview path calls:

```text
fista_reconstruct_reference(..., mask=_preview_reconstruction_mask(...))
```

and in `src/tomojax/recon/_fista_reference.py`, that mask is used inside `_loss_and_explicit_gradient()`:

```text
apply_residual_filter_schedule(..., mask=mask)
_residual_filter_adjoint(..., mask=mask)
```

So the current `otsu_l2` stopped path is effectively:

```text
Otsu-masked FISTA reconstruction
then
Otsu-masked Schur alignment
```

That is not v1 parity. The v1-like behaviour you want is almost certainly:

```text
valid-mask FISTA reconstruction
then
Otsu-masked alignment loss
```

The stopped preview reconstruction needs broad detector-valid data to make a sensible volume. The Otsu foreground mask is useful for the geometry objective, but it is too narrow to be the reconstruction data mask.

## 2. v1 versus v2 differences that matter

Here is the practical comparison.

| Ingredient          | v1-like behaviour needed                                                       | Current v2 behaviour                                                             | Why it matters                                                                                                      |
| ------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Otsu mask           | Alignment/objective mask only.                                                 | Otsu modifies the global mask, which is also passed into FISTA.                  | The preview volume is reconstructed from a narrowed foreground-only residual and can become poor or gauge-absorbed. |
| Reconstruction mask | Detector-valid / sidecar validity mask.                                        | Same mask as Otsu geometry loss.                                                 | Alignment-mask design is coupled to reconstruction quality.                                                         |
| Preview volume      | Reconstructed using enough projection information to preserve object geometry. | Reconstructed under corrupted geometry with an Otsu-masked loss.                 | Schur receives a stopped volume that may already fit the wrong setup.                                               |
| Multires            | Coarse reconstructed volume and geometry are refined and carried upward.       | Sidecar ladder carries geometry only; each level starts its own stopped solve.   | This is not a true v1-style pyramid. The volume gauge is restarted at every level.                                  |
| Coarse sidecars     | Coarse data/operator should be self-consistent.                                | 32³/64³ `true_volume_true_geometry_loss` is large.                               | The coarse sidecars are not clean oracle alignment problems.                                                        |
| Detector shift      | Should be solved against a stable preview volume/gauge.                        | Schur tries to recover det_u from a volume reconstructed with wrong det_u.       | The x-step can absorb or smear the detector shift before alignment.                                                 |
| Theta               | Det_u-only gate should not include frozen theta error.                         | Theta error remains about `0.0218166 rad` while active setup is only `det_u_px`. | The stopped reconstruction must absorb an uncorrected orientation error too.                                        |
| No-FISTA-first      | Useful only if the initial volume has alignment signal.                        | It fails around `4.7 px`.                                                        | A neutral volume avoids absorption but may be too uninformative for Schur.                                          |

The important conclusion: **v2 has the superficial shape of `reconstruct → align → reconstruct → align`, but not the practical v1 parity loop.**

The two missing pieces are:

```text
separate reconstruction-valid mask from Otsu alignment mask
carry the reconstructed volume through a real multires pyramid
```

## 3. Top 3 likely root causes, ranked

### 1. Otsu mask leakage into reconstruction

This is the highest-confidence root cause.

The current benchmark is labelled “v1-style Otsu L2,” but the implementation uses Otsu in the preview x-step, not just in the Schur geometry loss.

That directly explains the observed split:

```text
fixed_synthetic_truth:
    bypasses preview reconstruction
    det_u recovery succeeds

stopped_reconstruction:
    depends on preview reconstruction
    det_u recovery fails
```

The volume numbers support this. Even the fixed-truth parity run has poor final volume quality:

```text
fixed_truth_otsu_l2_reference_128v:
    volume NMSE ≈ 0.638
    final_volume_final_geometry_loss ≈ 42.67
```

It recovers det_u because Schur used the true volume, not because v2’s preview reconstruction is good.

### 2. The multires sidecar ladder is not a v1-equivalent pyramid

The current sidecar multires attempt is useful as a diagnostic, but it is not the old workflow.

It does this:

```text
32³ sidecar solve → carry geometry
64³ sidecar solve → carry geometry
128³ sidecar solve
```

It does **not** do:

```text
32³ reconstruction → align → upsample volume + scale geometry
64³ reconstruction continuation → align → upsample volume + scale geometry
128³ reconstruction continuation → align
```

Also, the coarse sidecars are not forward-consistent under the v2 projector. In the evidence:

```text
32³ true_volume_true_geometry_loss ≈ 291.69
64³ true_volume_true_geometry_loss ≈ 127.03
128³ true_volume_true_geometry_loss = 0.0
```

A true volume under true geometry should not have a huge projection loss in a clean oracle sidecar. That means the downsampled projections and downsampled volume are not matched by the same forward operator. The 32³ and 64³ failures are therefore not clean evidence against multires alignment.

### 3. The det_u-only gate is contaminated by frozen theta error

The gate is named and configured as det_u-only:

```text
active setup = det_u_px
pose frozen
```

but the true theta offset is still present:

```text
theta RMSE ≈ 0.0218166 rad ≈ 1.25°
```

and it remains stuck in every run because theta is not active.

For fixed truth, this can still recover det_u well enough. For stopped reconstruction, the volume can absorb the uncorrected theta as object orientation or smearing. That makes det_u recovery harder and makes the benchmark less clean.

For a pure det_u gate, make the true theta offset zero.

For a calibration gate, activate theta and add an orientation anchor.

Do not mix those two tests.

## 4. Next 1–2 code changes most likely to make the gate pass

### Change 1: split reconstruction masks from alignment masks

This should be the next change.

Replace the current single-mask flow:

```text
sidecar mask
    ↓
Otsu modifies global mask
    ↓
FISTA uses Otsu mask
Schur uses Otsu mask
```

with:

```text
projection_valid_mask:
    sidecar detector-valid mask
    used by FISTA reconstruction

alignment_loss_mask:
    sidecar detector-valid mask ∧ Otsu foreground mask
    used by Schur geometry loss
```

In concrete terms:

```text
build_smoke_inputs()
    projection_valid_mask = sidecar mask
    alignment_loss_mask = sidecar mask & otsu_projection_mask

fista_reconstruct_reference(..., mask=projection_valid_mask)

solve_joint_schur_lm(..., mask=alignment_loss_mask)

verification:
    report valid-mask residuals and alignment-mask residuals separately
```

Do not make this a user-facing knob. Treat it as a correctness fix.

Expected result:

```text
volume NMSE improves materially below 0.71
final_volume_true_geometry_loss improves
det_u RMSE drops from ~4.1 px
classification moves away from reconstruction_absorbed_geometry
```

Initial success threshold:

```text
det_u RMSE < 2 px
volume NMSE clearly below 0.71
```

Then tighten:

```text
det_u RMSE < 1 px
then < 0.5 px
```

### Change 2: implement a real in-process v1-parity pyramid carrying the volume

Do not use the current sidecar ladder as the v1-parity proof.

Implement an in-process pipeline:

```text
level 32:
    valid-mask FISTA preview
    Otsu-mask det_u Schur
    save volume + geometry

level 64:
    upsample previous volume
    scale pixel DOFs
    continue valid-mask FISTA from carried volume
    Otsu-mask det_u Schur
    save volume + geometry

level 128:
    upsample previous volume
    scale pixel DOFs
    continue valid-mask FISTA from carried volume
    Otsu-mask det_u Schur
```

If you keep the sidecar-based harness, it needs an `--initial-volume` path so level 64 can start from the upsampled level-32 reconstruction and level 128 can start from the upsampled level-64 reconstruction.

Also fix the multires sidecar consistency problem. Either:

```text
generate each level’s projections by projecting that level’s downsampled phantom
with that level’s true geometry
```

or stop treating coarse `true_volume_true_geometry_loss` as an oracle metric.

As a benchmark cleanup, set:

```text
theta_offset = 0
```

for the det_u-only parity gate. If you want theta recovery, run a separate theta-active calibration gate.

## 5. Benchmark results that are misleading or invalid

The fixed-truth Otsu L2 result is valid as a **det_u oracle pass**, but not as a production pass.

It is also currently mislabelled:

```text
status = passed
```

even though the manifest evaluation says:

```text
theta_offset_error_deg_lt: failed
benchmark_manifest_evaluation_summary.status = failed
```

So either theta should be marked `not_evaluated` for a det_u-only run, or the top-level status should not be `passed`.

The stopped Otsu L2 result is a real failure:

```text
det_u RMSE ≈ 4.10057 px
volume NMSE ≈ 0.710293
classification = reconstruction_absorbed_geometry
```

That result is trustworthy.

The sidecar multires result is not a valid v1-parity result because:

```text
it carries geometry, not volume
the 32³ and 64³ true-volume/true-geometry losses are large
```

So it is useful as a diagnostic, but not as proof that multires cannot solve the problem.

The “Otsu L2 v1-style” label is misleading. The actual current behaviour is:

```text
Otsu-masked reconstruction + Otsu-masked alignment
```

not:

```text
valid-mask reconstruction + Otsu-masked alignment
```

The no-FISTA-first failure is useful but should not be overinterpreted. A neutral or unreconstructed volume can fail because it has no alignment signal, not because FISTA is necessarily harmful.

Finally, absolute projection losses across 32/64/128 sidecars are not comparable or cleanly interpretable until the sidecars are forward-consistent.

## 6. Recommended minimal success gate

Before broader v2 work resumes, use one narrow gate:

```text
name:
    rich_phantom94_det_u_only_v1_parity

phantom:
    PHANTOM94 / random_cubes_spheres

size:
    128³
    optional in-process 32→64→128 pyramid

views:
    128

geometry:
    parallel tomography
    true det_u = 14.5 px
    true theta_offset = 0
    det_v = 0
    detector_roll = 0
    axis rotations = 0
    pose frozen / zero
    nuisance off

reconstruction:
    stopped reconstruction
    FISTA uses projection_valid_mask, not Otsu mask
    nonnegative constraint allowed
    fixed support allowed if documented

alignment:
    Schur active setup = det_u_px only
    active pose = none
    Schur mask = Otsu foreground mask

forbidden:
    fixed_synthetic_truth
    weak-view exclusion
    candidate refresh
    nuisance fitting
    unsupported-DOF exemptions
    Pallas requirement
```

Initial pass:

```text
det_u RMSE < 1 px
volume NMSE materially better than 0.71
classification != reconstruction_absorbed_geometry
```

Then tighten:

```text
det_u RMSE < 0.5 px
```

V1-parity target:

```text
det_u RMSE < 0.2 px
no excluded views
valid-mask reconstruction
Otsu-mask alignment
in-process volume-carry pyramid
```

## What not to do next

Do not add nuisance. The clean supported-only case already fails.

Do not add more DOFs. The one-DOF stopped gate fails.

Do not tune Schur damping/trust first. Fixed-truth Otsu L2 says Schur can recover det_u when the volume is right.

Do not add more candidate-refresh variants. Candidate refresh cannot fix a geometry proposal computed from a bad stopped volume.

Do not spend more time on support, center penalties, or TV until the mask split is fixed. Those experiments were run through a likely-invalid reconstruction mask.

Do not treat the current sidecar multires failure as proof that multires does not work. It is not a v1-equivalent pyramid.

Do not require theta recovery in the det_u-only gate unless theta is active and the volume orientation is anchored.

## Bottom line

The current v2 stopped loop is structurally unlike v1 in two exact ways that matter:

```text
1. v2 applies the Otsu mask to reconstruction, not just alignment.
2. v2 multires carries geometry, not the reconstructed volume.
```

Fix those before anything else.

The next agent goal should be:

```text
Implement true rich-phantom v1-parity det_u gate:
- split projection_valid_mask from alignment_loss_mask,
- set theta_offset = 0 for det_u-only,
- run valid-mask FISTA,
- run Otsu-mask det_u-only Schur,
- implement in-process 32→64→128 pyramid carrying volume and geometry,
- remove candidate-refresh/no-FISTA variants from this gate.
```

If that still fails, then the deeper issue is likely the v2 preview reconstruction/backprojection itself. But right now the strongest evidence points to a **v1-parity workflow mismatch**, not a fundamental failure of the v2 Schur alignment method.
