# 02 — Loss and Optimiser Specification

## Summary

Use one default loss family and one default optimiser family.

```text
Geometry loss:
    masked, whitened, robust projection-domain reprojection loss
    + weak setup/pose/nuisance priors
    + hard gauge canonicalisation after accepted steps

Geometry optimiser:
    robust Levenberg-Marquardt / Gauss-Newton
    with IRLS robust weights
    with Schur/block structure for setup + per-view pose

Reconstruction loss:
    projection-domain data term
    + Huber-TV / smoothed TV
    + non-negativity / support constraints where applicable

Reconstruction optimiser:
    FISTA or PDHG
```

Do not use Adam for core geometry. Adam belongs in neural/large-field experimental modules.

## Geometry residual

For projection `i`:

```text
x      = current stopped reconstruction
g      = global setup geometry
p_i    = per-view pose
m_i    = optional object-frame motion/deformation
n_i    = nuisance acquisition state
y_i    = measured projection
A_i    = differentiable forward projector
N_i    = nuisance model
M_i    = valid-pixel mask
sigma_i= robust/noise scale
B_l    = level-dependent residual filter
```

Predicted projection:

```text
ŷ_i = N_i(A_i(x; g, p_i, m_i))
```

Whitened residual:

```text
e_i = ŷ_i - y_i
r_i = M_i * B_l(e_i) / sigma_i
```

The filter `B_l` depends on continuation level:

```text
coarse level:
    low-pass residual

medium level:
    low-pass + band-pass residual

fine level:
    raw residual for verification or optional polish
```

## Default robust loss

Use pseudo-Huber / Charbonnier:

\[
\rho_\delta(r)
=
\delta^2
\left(
\sqrt{1 + (r/\delta)^2} - 1
\right)
\]

Properties:

```text
small residuals:
    approximately 0.5 * r^2

large residuals:
    approximately δ * |r|

gradient:
    rho'(r) = r / sqrt(1 + (r/δ)^2)

IRLS weight:
    w(r) = rho'(r) / r = 1 / sqrt(1 + (r/δ)^2)
```

Why this loss:

```text
smooth and differentiable
quadratic near optimum
robust to hot pixels, stripes, masks, bad frames, underconverged volume defects
easy to implement with IRLS LM
less brittle than raw L2
less discontinuous than hard L1
```

Default robust scale:

```text
delta_l = k * robust_scale(r at level l)
```

Start with:

```text
k = 1.5 to 3.0
robust_scale = median absolute deviation on valid residuals
```

Record the actual `delta_l` used per level.

## Full geometry objective

\[
L_{\text{geom}}
=
\frac{1}{N_{\text{valid}}}
\sum_{i,u,v} \rho_{\delta_l}(r_i[u,v])
+
L_{\text{setup-prior}}
+
L_{\text{pose-prior}}
+
L_{\text{motion-prior}}
+
L_{\text{nuisance-prior}}
\]

### Setup prior

\[
L_{\text{setup-prior}}
=
\|S_g(g-g_0)\|^2
\]

Use weak metadata priors. They should guide weak modes, not prevent correction.

Recommended defaults:

```text
det_u_px:
    weak prior or none

detector_roll_deg:
    weak prior around metadata/zero

axis_rot_x/y:
    weak prior around metadata/zero

theta_offset:
    weak prior around zero

theta_scale:
    strong prior unless observability test passes

det_v_px:
    strong prior or frozen unless observability test passes
```

### Pose prior

\[
L_{\text{pose-prior}}
=
\lambda_0 \sum_i \|S_p p_i\|^2
+
\lambda_1 \sum_i \|D p_i\|^2
+
\lambda_2 \sum_i \|D^2 p_i\|^2
\]

Use weak priors and trust radii, not hard suppression.

Suggested behaviour:

```text
coarse:
    moderate prior on alpha/beta/phi
    weak smoothness prior on dx/dz
    robust loss dominates

medium:
    weaker prior
    let data dominate

fine:
    priors only prevent nonsensical runaway
```

Do not over-smooth by default. Real scans may contain jumps, segment boundaries, or stage discontinuities.

### Nuisance prior

Nuisance forward model:

```text
ŷ_i' = a_i * ŷ_i + b_i + bg_i(u,v)
```

Priors:

```text
sum_i (a_i - 1)^2
sum_i b_i^2
smoothness(bg_i)
low-rank/low-frequency penalty for background fields
```

Nuisance terms prevent geometry from explaining intensity drift.

## Gauge handling

Do not mainly implement gauges as loss terms. Use canonicalisation after accepted updates.

Examples:

```text
mean_dx = mean(dx_i)
dx_i -= mean_dx
det_u_px += mean_dx

mean_phi = mean(phi_residual_i)
phi_residual_i -= mean_phi
theta_offset_deg += mean_phi

if det_v_px active:
    mean_dz = mean(dz_i)
    dz_i -= mean_dz
    det_v_px += mean_dz
```

For alpha/beta and axis direction, choose a policy:

```text
Option A:
    global axis direction absorbs common alpha/beta-like modes

Option B:
    axis direction remains metadata, alpha/beta may carry common mode

Default recommendation:
    prefer global setup for common modes if the transform is well-defined,
    but record canonicalisation deltas and observability.
```

Gauge canonicalisation is not instrument calibration. It is coordinate-system hygiene.

## Reconstruction objective

For fixed geometry:

\[
L_x
=
\frac{1}{N_{\text{valid}}}
\sum_{i,u,v}
\rho_{\delta_x}
\left(
W_i(A_i(g,p_i,m_i)x-y_i)
\right)
+
\lambda_{\text{TV}} TV_\epsilon(x)
+
I[x \ge 0]
+
L_{\text{support}}
\]

Default reconstruction regulariser:

```text
Huber-TV / smoothed isotropic TV
```

Smoothed TV:

\[
TV_\epsilon(x)
=
\sum_{\text{voxels}}
\sqrt{
(\nabla_x x)^2
+(\nabla_y x)^2
+(\nabla_z x)^2
+\epsilon^2
}
\]

Default constraints:

```text
non-negativity for attenuation
optional support mask
optional soft upper bound only if justified
```

Use shorter reconstructions during alignment and longer ones at the final pass.

## Reconstruction optimiser

Default:

```text
FISTA / Huber-TV FISTA
```

Why:

```text
simple
fast
GPU-friendly
works well for composite inverse problems
easy to warm start
easy to run at multiple resolutions
```

Also support later:

```text
PDHG / Chambolle-Pock
ADMM for specialised priors
PnP/RED experimental final recon
```

But default alignment should not require a learned prior.

## Geometry optimiser

Use robust Levenberg-Marquardt / Gauss-Newton.

Each geometry iteration:

```text
1. compute predicted projections
2. compute masked whitened residuals
3. compute pseudo-Huber robust weights
4. accumulate Jᵀr and JᵀJ
5. add priors and damping
6. solve LM step
7. apply per-DOF trust radii / bounds
8. evaluate actual loss decrease
9. accept/reject step
10. canonicalise gauges
11. update damping
```

LM system:

\[
(J^T W J + P + \lambda D)\Delta
=
-(J^T W r + \nabla P)
\]

where:

```text
W = robust IRLS weights
P = prior Hessian / regularisation contribution
λ = LM damping
D = diagonal scaling, usually diag(JᵀWJ + P)
```

## Why LM/GN, not Adam

The core geometry problem is structured nonlinear least squares:

```text
small global setup block
5 parameters per view
projection residuals
useful local curvature
natural trust regions
robust least-squares objective
```

LM/GN uses this structure. Adam throws it away.

Use Adam/AdamW only for:

```text
neural fields
learned scout modules
large nuisance/background fields
high-dimensional deformation fields
experimental learned priors
```

Do not use Adam as the default optimiser for:

```text
det_u_px
detector_roll
axis direction
theta_offset
alpha/beta/phi/dx/dz
```

unless LM is not implemented yet.

## Pose-only LM

When setup is frozen, each view is independent:

\[
(H_{p_i p_i}+\lambda D_i)\delta p_i = -b_{p_i}
\]

where `p_i = [alpha, beta, phi, dx, dz]`.

This is a `5×5` solve per projection.

Record per view:

```text
loss_before
loss_after
accepted
damping
condition_number
step_norm
per-DOF step
bounds_hits
```

## Setup-only LM

When pose is frozen, solve a small dense system over setup variables:

```text
det_u_px
detector_roll_deg
axis_rot_x_deg
axis_rot_y_deg
theta_offset_deg
optional det_v_px
optional theta_scale
```

Record:

```text
setup step vector
condition number
eigenvalues
correlations
observability status
```

## Joint setup+pose Schur LM

For fixed `x`, residuals depend on global setup `g` and per-view pose `p_i`.

Accumulate per projection:

```text
loss_i
b_g_i  = Jg_iᵀ r_i
b_p_i  = Jp_iᵀ r_i
H_gg_i = Jg_iᵀ Jg_i
H_pp_i = Jp_iᵀ Jp_i
H_gp_i = Jg_iᵀ Jp_i
```

Sum global terms:

```text
H_gg = Σ_i H_gg_i + setup_prior + damping
b_g  = Σ_i b_g_i  + setup_prior_gradient
```

Per-view terms include pose priors and damping.

Schur solve:

```text
S = H_gg - Σ_i H_gp_i H_pp_i^-1 H_pg_i
b = b_g  - Σ_i H_gp_i H_pp_i^-1 b_p_i

δg = -solve(S, b)
δp_i = -H_pp_i^-1 (b_p_i + H_pg_i δg)
```

Check sign convention in implementation and unit tests.

## Trust radii and parameter scaling

Parameters should be internally scaled before optimisation.

Example scales:

```text
dx, dz:
    pixels

alpha, beta, phi:
    radians internally
    report degrees externally

det_u, det_v:
    pixels

detector_roll, axis_rot_x/y, theta_offset:
    radians internally

theta_scale:
    dimensionless fractional scale
```

Suggested initial trust radii:

```text
coarse:
    dx/dz:        10-30 px at native equivalent
    phi:          5-15 degrees
    alpha/beta:   1-5 degrees
    det_u:        10-30 px
    roll/axis:    0.5-5 degrees

medium:
    half or quarter of coarse

fine:
    small polish only
```

Use actual LM acceptance to adapt.

## Loss evaluation for acceptance

Compute actual and predicted decrease:

```text
predicted_decrease = model_quadratic_loss_before - model_quadratic_loss_after
actual_decrease = true_robust_loss_before - true_robust_loss_after
ratio = actual_decrease / predicted_decrease
```

Policy:

```text
ratio > 0.75:
    accept, decrease damping aggressively

0.25 < ratio <= 0.75:
    accept, modest damping decrease

0 < ratio <= 0.25:
    accept cautiously or shrink step

ratio <= 0:
    reject, increase damping
```

Always respect NaN/Inf checks.

## Residual filtering

Implement residual operators as named internal policies:

```text
lowpass_gaussian
bandpass_difference_of_gaussians
gradient_lowpass
raw
```

Default schedule:

```text
level 4:
    1.0 * lowpass

level 2:
    0.7 * lowpass
    0.3 * bandpass

level 1:
    raw for verification
    optional small-weight bandpass/gradient for polish
```

Do not expose all filters in CLI by default.

## Backend API target

The geometry optimiser should request reductions:

```python
GeometryReductions(
    loss: scalar,
    jtr_setup: [G],
    jtj_setup: [G, G],
    per_view_jtr_pose: [N, 5],
    per_view_jtj_pose: [N, 5, 5],
    per_view_jtj_setup_pose: [N, G, 5],
    residual_stats: ...
)
```

The JAX backend may compute these using autodiff/JVP/VJP.

The Pallas backend should eventually compute reductions directly for canonical detector-grid cases.

## Default values to start with

These are implementation defaults, not gospel.

```text
geometry_loss:
    pseudo_huber

delta:
    2.0 * robust MAD scale

mask:
    required

noise scaling:
    per-projection robust scale if photon stats unavailable

setup prior:
    weak except det_v/theta_scale

pose prior:
    weak L2 + weak second-difference smoothness

LM initial damping:
    1e-3 to 1e-1 relative to diagonal scale

damping update:
    trust-region style, based on actual/predicted decrease

reconstruction:
    Huber-TV FISTA
    non-negative attenuation
    short preview schedule during alignment
    longer final schedule
```

## Acceptance criteria for this module

A correct implementation should pass:

```text
finite-difference gradient checks
pose-only synthetic recovery
setup-only synthetic recovery
joint setup+pose synthetic recovery
JAX/Pallas residual agreement where Pallas is active
robust loss outlier test
gauge canonicalisation invariance test
LM actual-vs-predicted decrease sanity test
```
