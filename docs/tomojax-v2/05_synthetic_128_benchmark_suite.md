# 05 — Synthetic `128^3` Benchmark Suite

This document defines five synthetic datasets for implementing and benchmarking the reimagined TomoJAX. They are designed to test global setup errors, per-view 5-DOF pose, laminography geometry, thermal/object-frame drift, nuisance terms, and combined failure modes.

The datasets should be generated reproducibly and used to compare:

```text
current TomoJAX
current COR-only / setup-only comparators
reimagined TomoJAX JAX reference backend
reimagined TomoJAX Pallas backend where eligible
```

## Common dataset format

Each dataset directory should contain:

```text
dataset_manifest.json
ground_truth_volume.npy
projections.npy or projections.zarr
nominal_geometry.json
true_geometry.json
true_pose.csv
true_motion.csv
nuisance_truth.json
noise_truth.json
mask.npy, optional
preview.png
```

Suggested directory layout:

```text
data/synthetic128/
  synth128_setup_global_tomo/
  synth128_pose_random_extreme/
  synth128_lamino_axis_roll_pose/
  synth128_thermal_object_drift/
  synth128_combined_nuisance_jumps/
```

## Common volume size

```text
volume shape: 128 x 128 x 128
voxel size: 1.0 arbitrary unit
object support: mostly inside radius 52 voxels
background attenuation: 0
```

## Common detector/projection defaults

Unless overridden:

```text
projection model: parallel beam
detector shape: 160 x 160
detector pixel size: 1.0
number of projections: 256
theta range: 0 to 180 degrees, endpoint excluded for tomography
dtype: float32
log-attenuation projections
```

For laminography:

```text
projection model: parallel laminography
detector shape: 192 x 192
number of projections: 256
theta range: 0 to 360 degrees, endpoint excluded
laminography tilt: 60 degrees from beam normal, or project-specific convention
```

## Common phantom recipe

Use a procedural phantom richer than a Shepp-Logan sphere. The goal is enough structure to make alignment observable.

Recommended phantom components, all seeded:

```text
large low-contrast ellipsoids
high-contrast rods/fibres at oblique angles
small beads/fiducials
thin plates/sheets
voids/pores
one asymmetric marker cluster
weak texture/noise field inside material
```

Pseudo-code sketch:

```python
def make_phantom_128(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vol = np.zeros((128, 128, 128), np.float32)

    # centred coordinate grid in [-1, 1]
    z, y, x = grid()

    # base ellipsoid
    vol += 0.45 * inside_rotated_ellipsoid(...)

    # sub-ellipsoids
    for k in range(20):
        vol += amp_k * inside_rotated_ellipsoid(...)

    # rods/fibres
    for k in range(15):
        vol += amp_k * distance_to_line_segment(...) < radius_k

    # beads
    for k in range(40):
        vol += amp_k * gaussian_sphere(...)

    # voids
    for k in range(20):
        vol *= 1 - 0.6 * inside_sphere(...)

    # asymmetric marker cluster
    add 5 high-contrast beads near one quadrant

    # smooth texture
    vol += 0.03 * filtered_noise_inside_support

    return np.clip(vol, 0, None)
```

Use separate seeds per dataset.

## Common noise model

For alignment benchmarking, include both clean and noisy variants when practical:

```text
clean:
    no noise, no nuisance

standard:
    additive Gaussian noise after projection
    sigma = 1% of projection robust scale

hard:
    additive Gaussian noise sigma = 3-5%
    sparse hot pixels
    optional mild stripe bias
```

Store the noise configuration in `noise_truth.json`.

## Common benchmark metrics

Synthetic truth allows two classes of metrics.

### Reconstruction metrics

```text
volume RMSE
volume NMSE
volume MAE
gradient magnitude correlation
edge sharpness proxy
support leakage
final data residual
```

Register/canonicalise volume if the geometry gauge allows a global frame transform. Record whether registration was required.

### Geometry metrics

After gauge canonicalisation:

```text
det_u error
det_v error, if active
detector_roll error
axis_rot_x/y error
theta_offset error
theta_scale error
pose dx/dz RMSE
pose alpha/beta/phi RMSE
object-frame drift RMSE
```

Also record realised-forward-geometry agreement:

```text
projection residual using true volume + recovered geometry
projection residual using true volume + true geometry
```

This avoids over-penalising gauge-equivalent parameterisations.

### Runtime metrics

```text
time to verified geometry
total wall time
number of reconstruction calls
number of geometry evaluations
number of projector calls
backend actual vs requested
Pallas hit/fallback rate
```

## Benchmark protocols

Run each dataset with at least:

```text
current_tomojax_default
current_tomojax_cor_only_or_reference_comparator
reimagined_jax_reference
reimagined_auto
reimagined_max, optional
```

Suggested command placeholders:

```bash
tomojax synth generate synth128_setup_global_tomo -o data/synthetic128/synth128_setup_global_tomo

tomojax benchmark run \
  --dataset data/synthetic128/synth128_setup_global_tomo \
  --impl current \
  --profile current-default \
  -o runs/current/synth128_setup_global_tomo

tomojax benchmark run \
  --dataset data/synthetic128/synth128_setup_global_tomo \
  --impl reimagined \
  --profile lightning \
  -o runs/reimagined/synth128_setup_global_tomo

tomojax benchmark compare \
  runs/current/synth128_setup_global_tomo \
  runs/reimagined/synth128_setup_global_tomo \
  -o reports/synth128_setup_global_tomo.md
```

Adapt command names to current repo conventions.

---

# Dataset 1 — `synth128_setup_global_tomo`

## Purpose

Test whether TomoJAX can solve classic global setup errors without grid search:

```text
centre of rotation / detector horizontal shift
detector roll
axis not vertical
global theta offset
```

This dataset should be solvable by setup parameters alone. Per-view pose should not be necessary except as a gauge-equivalent alternative.

## Geometry

```text
mode: parallel tomography
volume: 128^3
detector: 160 x 160
views: 256
theta: 0..180 degrees, endpoint excluded
phantom seed: 1001
```

## True setup perturbations

```text
det_u_px: +14.5
det_v_px: 0.0
detector_roll_deg: +0.65
axis_rot_x_deg: +0.45
axis_rot_y_deg: -0.30
theta_offset_deg: +1.25
theta_scale: 1.0
```

## True per-view pose

```text
alpha_i: 0
beta_i: 0
phi_residual_i: 0
dx_i: 0
dz_i: 0
```

## Nuisance/noise

```text
noise: standard
gain drift: none
background drift: none
hot pixels: 0.05%
mask: valid full detector
```

## Expected solver behaviour

The reimagined solver should primarily place the correction into setup variables:

```text
det_u_px ≈ +14.5
detector_roll ≈ +0.65°
axis_rot_x/y ≈ true values
theta_offset ≈ +1.25°
mean(dx) ≈ 0 after canonicalisation
mean(phi_residual) ≈ 0 after canonicalisation
```

If the optimiser initially puts common `dx` into per-view pose, gauge canonicalisation should transfer it to `det_u_px`.

## Pass criteria

```text
final volume NMSE < current COR-only comparator
det_u error < 0.5 px after canonicalisation
roll error < 0.05°
axis_rot_x/y error < 0.1°
theta_offset error < 0.1°
time-to-verified-geometry lower than current staged setup path
```

---

# Dataset 2 — `synth128_pose_random_extreme`

## Purpose

Stress-test the all-5 per-view pose solver. This mirrors the type of synthetic result already described: large random projection shifts, noisy data, and large per-view angular errors.

## Geometry

```text
mode: parallel tomography
volume: 128^3
detector: 160 x 160
views: 256
theta: 0..180 degrees, endpoint excluded
phantom seed: 1002
```

## True setup perturbations

```text
det_u_px: 0.0
det_v_px: 0.0
detector_roll_deg: 0.0
axis_rot_x_deg: 0.0
axis_rot_y_deg: 0.0
theta_offset_deg: 0.0
theta_scale: 1.0
```

## True per-view pose

Use deterministic random arrays with seed `2002`:

```text
dx_i: uniform(-20, +20) px
dz_i: uniform(-20, +20) px
phi_residual_i: uniform(-10, +10) degrees
alpha_i: uniform(-2.0, +2.0) degrees
beta_i: uniform(-2.0, +2.0) degrees
```

Optionally create a harder variant:

```text
dx/dz: uniform(-30, +30) px
phi: uniform(-15, +15) degrees
alpha/beta: uniform(-4, +4) degrees
```

## Nuisance/noise

```text
noise: hard
Gaussian sigma: 5% projection robust scale
hot pixels: 0.1%
background offset: none
gain drift: none
```

## Expected solver behaviour

This should be solved by per-view pose, not by setup:

```text
setup remains near nominal after gauge canonicalisation
pose RMSE drops sharply by level 4
final reconstruction visibly sharp
```

## Pass criteria

```text
pose dx/dz RMSE < 1.0 px
phi RMSE < 0.25°
alpha/beta RMSE < 0.25° if identifiable
final volume NMSE substantially better than unaligned reconstruction
level 4 geometry explains most of the error
```

This dataset is the primary regression test for the all-5 pose LM.

---

# Dataset 3 — `synth128_lamino_axis_roll_pose`

## Purpose

Test laminography-specific setup, axis direction, detector roll, calibrated detector-grid fallback semantics, and per-view pose.

## Geometry

```text
mode: parallel laminography
volume: 128^3
detector: 192 x 192
views: 256
theta: 0..360 degrees, endpoint excluded
laminography tilt: 60 degrees
phantom seed: 1003
detector grid: calibrated/non-canonical variant enabled
```

## True setup perturbations

```text
det_u_px: -8.0
det_v_px: +3.0      # intentionally weak; observability-gated
detector_roll_deg: +1.10
axis_rot_x_deg: +0.70
axis_rot_y_deg: -0.55
theta_offset_deg: -0.80
theta_scale: 1.0
```

## True per-view pose

Use smooth harmonic wobble plus small jitter, seed `2003`:

```text
dx_i = 3.0 * sin(2 theta_i + 0.4) + normal(0, 0.5)
dz_i = 2.0 * cos(3 theta_i - 0.2) + normal(0, 0.5)

alpha_i = 0.35° * sin(theta_i + 0.7) + normal(0, 0.05°)
beta_i  = 0.30° * cos(theta_i - 0.5) + normal(0, 0.05°)

phi_residual_i = 0.15° * sin(4 theta_i) + normal(0, 0.03°)
```

After generation, canonicalise the true pose so:

```text
mean(dx_i) = 0
mean(dz_i) = 0
mean(phi_residual_i) = 0
```

## Nuisance/noise

```text
noise: standard
partial-FOV crop: mild
mask: detector corners invalid for some views
gain drift: low amplitude sinusoid, +/-1%
```

## Expected solver behaviour

This dataset should exercise:

```text
global laminography axis correction
detector roll correction
per-view harmonic pose residuals
det_v observability test
calibrated detector-grid JAX fallback unless Pallas path supports it
```

## Pass criteria

```text
final reconstruction improves over setup-only and pose-only baselines
axis_rot_x/y error < 0.15°
detector_roll error < 0.10°
det_u error < 1.0 px
det_v either recovered stably or reported unobservable
backend report explicitly records calibrated-grid fallback where expected
```

---

# Dataset 4 — `synth128_thermal_object_drift`

## Purpose

Test gradual specimen motion in its own reference frame. This simulates thermal expansion or glue creep moving the specimen left over the scan.

This dataset distinguishes:

```text
detector-frame dx/dz correction
vs
object-frame drift
```

## Geometry

```text
mode: parallel tomography
volume: 128^3
detector: 160 x 160
views: 256
theta: 0..180 degrees, endpoint excluded
phantom seed: 1004
```

## True setup perturbations

```text
det_u_px: +6.0
det_v_px: 0.0
detector_roll_deg: +0.20
axis_rot_x_deg: 0.0
axis_rot_y_deg: 0.0
theta_offset_deg: +0.35
theta_scale: 1.0008
```

## True object-frame motion

Let `t = i / (N-1)`.

```text
tx_obj_i = -12.0 * smoothstep(t) px
ty_obj_i = +2.0 * sin(2πt) px
tz_obj_i = +1.5 * t px
rot_obj_z_i = +0.20° * smoothstep(t)
```

where:

```text
smoothstep(t) = 3t^2 - 2t^3
```

## True per-view detector/stage pose

Small stage jitter:

```text
dx_i = normal(0, 0.4) px
dz_i = normal(0, 0.4) px
alpha_i = normal(0, 0.03°)
beta_i = normal(0, 0.03°)
phi_residual_i = normal(0, 0.02°)
```

Seed: `2004`.

## Nuisance/noise

```text
noise: standard
gain drift: linear from 0.98 to 1.03
background offset: linear small drift
hot pixels: 0.05%
```

## Expected solver behaviour

Initial implementation without object-frame motion may still reconstruct better by absorbing drift into per-view pose. That is acceptable for reconstruction-first mode, but the verification system should notice structured residuals and large smooth pose drift.

When object-frame drift model is enabled, it should explain most of the smooth drift.

## Pass criteria

For core solver without object-frame motion:

```text
final reconstruction substantially better than no-alignment
pose trace shows smooth drift
verification flags object_motion_suspected or large smooth pose component
```

For object-motion-enabled solver:

```text
object tx RMSE < 1.5 px
detector dx residual RMSE reduced versus no-object-motion run
final reconstruction equal or better
```

This dataset is an escalation-path benchmark, not a phase-1 requirement.

---

# Dataset 5 — `synth128_combined_nuisance_jumps`

## Purpose

Test a hard, realistic combination:

```text
global setup errors
per-view pose wobble
sparse jumps
theta scale error
noise
flat-field/gain drift
stripe artefacts
partial invalid detector regions
optional weak affine expansion
```

This is the “everything goes wrong” benchmark.

## Geometry

```text
mode: parallel laminography
volume: 128^3
detector: 192 x 192
views: 320
theta: 0..360 degrees, endpoint excluded
laminography tilt: 55 degrees
phantom seed: 1005
```

## True setup perturbations

```text
det_u_px: -14.5
det_v_px: +4.0          # weak/ambiguous
detector_roll_deg: -0.85
axis_rot_x_deg: +0.95
axis_rot_y_deg: +0.40
theta_offset_deg: +1.50
theta_scale: 0.9985
```

## True per-view pose

Let `t = i/(N-1)` and `theta_i` be nominal angle.

Smooth components:

```text
dx_smooth_i = -6.0 * t + 2.5 * sin(2 theta_i + 0.3)
dz_smooth_i = +3.0 * smoothstep(t) + 1.5 * cos(3 theta_i)

alpha_i = 0.45° * sin(theta_i) + 0.15° * sin(5 theta_i)
beta_i  = 0.40° * cos(theta_i + 0.2)
phi_residual_i = 0.25° * sin(2 theta_i - 0.8)
```

Sparse jumps:

```text
at view 90:
    dx += +7 px
    dz += -3 px
    phi += +0.4°

at view 190:
    dx += -10 px
    dz += +5 px
    alpha += -0.35°

at view 260:
    dx += +5 px
    beta += +0.25°
```

Jitter:

```text
dx/dz += normal(0, 0.8 px)
alpha/beta/phi += normal(0, 0.05°)
```

Seed: `2005`.

## Optional object-frame affine expansion

Escalation variant only:

```text
scale_x_i = 1.0 + 0.004 * t
scale_y_i = 1.0
scale_z_i = 1.0 - 0.002 * t
tx_obj_i = -4.0 * t
```

## Nuisance/noise

```text
Gaussian noise: 3% projection robust scale
hot pixels: 0.1%
dead pixels: 0.05%
stripe bias: 4 detector columns with additive bias
gain drift: 0.97 to 1.04 plus sinusoid
background drift: low-frequency vertical gradient, changing over time
partial-FOV invalid mask: 5% detector edge invalid for selected views
dropped/bad views: 3 views with 2x noise
```

## Expected solver behaviour

A good reconstruction-first solver should:

```text
solve the dominant setup
recover most per-view pose
not let stripe/gain artefacts become geometry
flag sparse jumps or bad views
treat det_v as weak unless evidence supports it
produce a crisp final reconstruction
```

## Pass criteria

```text
final volume NMSE beats current TomoJAX default
time-to-verified-geometry acceptable relative to current staged solver
det_u error < 1.5 px after canonicalisation
axis/roll errors < 0.2°
theta_offset error < 0.2°
pose dx/dz RMSE < 2 px except jump neighbourhoods
bad views flagged in verification
nuisance estimates reduce residual structure
```

This is the final benchmark for calling the system “general”.

---

# Current TomoJAX comparison protocol

For every dataset, run current TomoJAX in at least three modes if available:

```text
current_no_align:
    reconstruct with nominal geometry

current_setup_only:
    current setup / COR / roll / axis path if available

current_full_alignment:
    current lightning/default/staged setup+pose path
```

Record:

```text
command
git commit
config
backend actual
total wall time
setup stage time
pose stage time
reconstruction time
final loss
final volume metrics
geometry estimates
pose estimates
failure warnings
```

Do not require current TomoJAX to recover the same parameter decomposition as the new solver. Compare both:

```text
canonical parameter recovery
realised forward-geometry residual
final reconstruction metrics
```

## Benchmark table template

```markdown
| Dataset | Impl | Profile | Time to verified | Total time | Vol NMSE | Final residual | Setup error | Pose RMSE | Backend | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| synth128_setup_global_tomo | current | default | ... | ... | ... | ... | ... | ... | ... | ... |
| synth128_setup_global_tomo | reimagined | lightning | ... | ... | ... | ... | ... | ... | ... | ... |
```

## Required success bar for reimagined TomoJAX

The new architecture should not only be cleaner. It must win.

Minimum target for first experimental release:

```text
matches or beats current final reconstruction quality on all 5 datasets
is faster on at least 3/5 datasets
solves Dataset 1 without COR grid search
solves Dataset 2 with all-5 pose LM at level 4/2
handles Dataset 3 laminography setup with explicit backend provenance
flags object motion on Dataset 4
handles nuisance/jumps better than current solver on Dataset 5
```
