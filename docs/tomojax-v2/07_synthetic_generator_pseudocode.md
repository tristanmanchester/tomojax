# 07 — Synthetic Dataset Generator Pseudocode

This document gives implementation-oriented pseudocode for generating the five `synthetic128` benchmark datasets. It is intentionally not tied to the current TomoJAX internals, so the coding agent can adapt it to the real projector.

## Generator entry point

```python
def generate_synthetic128_dataset(name: str, output_dir: Path, *, clean: bool = False) -> None:
    spec = load_spec_from_benchmark_manifest(name)
    rng = np.random.default_rng(spec.phantom_seed)

    volume = make_phantom_128(spec.phantom_seed)
    nominal_geometry = make_nominal_geometry(spec)
    true_geometry = perturb_setup(nominal_geometry, spec.true_setup)
    true_pose = make_true_pose(spec)
    true_motion = make_true_motion(spec)
    nuisance = make_nuisance(spec)

    projections_clean = forward_project_truth(
        volume=volume,
        geometry=true_geometry,
        pose=true_pose,
        motion=true_motion,
        detector_shape=spec.detector_shape,
        theta=nominal_geometry.theta,
        mode=spec.mode,
    )

    projections = apply_nuisance_and_noise(
        projections_clean,
        nuisance=nuisance,
        clean=clean,
        seed=spec.noise_seed,
    )

    write_dataset(
        output_dir=output_dir,
        volume=volume,
        projections=projections,
        nominal_geometry=nominal_geometry,
        true_geometry=true_geometry,
        true_pose=true_pose,
        true_motion=true_motion,
        nuisance_truth=nuisance,
        manifest=spec,
    )
```

## Phantom generation

Use one common phantom family so all datasets share comparable intrinsic difficulty.

```python
def make_phantom_128(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = 128
    z, y, x = normalized_grid(n)  # each in [-1, 1]
    vol = np.zeros((n, n, n), np.float32)

    # Base body
    vol += 0.35 * rotated_ellipsoid(
        x, y, z,
        center=(0.02, -0.04, 0.01),
        radii=(0.72, 0.58, 0.50),
        euler=(0.2, -0.1, 0.35),
    )

    # Internal ellipsoids
    for _ in range(22):
        center = rng.uniform(-0.45, 0.45, size=3)
        radii = rng.uniform(0.05, 0.18, size=3)
        euler = rng.uniform(-np.pi, np.pi, size=3)
        amp = rng.uniform(0.06, 0.30)
        vol += amp * rotated_ellipsoid(x, y, z, center, radii, euler)

    # Oblique rods / fibres
    for _ in range(18):
        p0 = rng.uniform(-0.7, 0.7, size=3)
        p1 = rng.uniform(-0.7, 0.7, size=3)
        radius = rng.uniform(0.012, 0.035)
        amp = rng.uniform(0.12, 0.45)
        vol += amp * (distance_to_segment(x, y, z, p0, p1) < radius)

    # Beads / fiducials
    for _ in range(45):
        center = rng.uniform(-0.65, 0.65, size=3)
        radius = rng.uniform(0.012, 0.045)
        amp = rng.uniform(0.25, 0.75)
        vol += amp * gaussian_sphere(x, y, z, center, radius)

    # Asymmetric marker cluster
    for c in [(0.48, -0.35, 0.30), (0.55, -0.28, 0.25), (0.43, -0.42, 0.38)]:
        vol += 0.9 * gaussian_sphere(x, y, z, c, 0.025)

    # Voids
    for _ in range(18):
        center = rng.uniform(-0.55, 0.55, size=3)
        radius = rng.uniform(0.025, 0.09)
        void = gaussian_sphere(x, y, z, center, radius) > 0.4
        vol[void] *= rng.uniform(0.1, 0.5)

    # Weak texture
    texture = lowpass_noise((n, n, n), seed + 999, sigma=3.0)
    support = vol > 0.02
    vol += support * 0.025 * texture

    vol = np.clip(vol, 0, None).astype(np.float32)
    return vol
```

## Nominal geometry generation

```python
def make_nominal_geometry(spec):
    theta = np.linspace(
        np.deg2rad(spec.theta_range_deg[0]),
        np.deg2rad(spec.theta_range_deg[1]),
        spec.views,
        endpoint=False,
    )

    return {
        "mode": spec.mode,
        "volume_shape": [128, 128, 128],
        "detector_shape": spec.detector_shape,
        "theta_rad": theta,
        "det_u_px": 0.0,
        "det_v_px": 0.0,
        "detector_roll_rad": 0.0,
        "axis_rot_x_rad": 0.0,
        "axis_rot_y_rad": 0.0,
        "theta_offset_rad": 0.0,
        "theta_scale": 1.0,
        "laminography_tilt_rad": np.deg2rad(spec.laminography_tilt_deg)
            if spec.mode == "parallel_laminography" else None,
    }
```

## Pose generation templates

### Random extreme pose

```python
def pose_random_extreme(n_views: int, seed: int):
    rng = np.random.default_rng(seed)
    return {
        "dx_px": rng.uniform(-20, 20, n_views),
        "dz_px": rng.uniform(-20, 20, n_views),
        "phi_residual_rad": np.deg2rad(rng.uniform(-10, 10, n_views)),
        "alpha_rad": np.deg2rad(rng.uniform(-2, 2, n_views)),
        "beta_rad": np.deg2rad(rng.uniform(-2, 2, n_views)),
    }
```

### Laminography harmonic pose

```python
def pose_lamino_harmonic(theta: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    n = len(theta)
    pose = {
        "dx_px": 3.0*np.sin(2*theta + 0.4) + rng.normal(0, 0.5, n),
        "dz_px": 2.0*np.cos(3*theta - 0.2) + rng.normal(0, 0.5, n),
        "alpha_rad": np.deg2rad(0.35*np.sin(theta + 0.7) + rng.normal(0, 0.05, n)),
        "beta_rad": np.deg2rad(0.30*np.cos(theta - 0.5) + rng.normal(0, 0.05, n)),
        "phi_residual_rad": np.deg2rad(0.15*np.sin(4*theta) + rng.normal(0, 0.03, n)),
    }
    return canonicalise_true_pose(pose)
```

### Combined jumps

```python
def pose_combined_jumps(theta: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    n = len(theta)
    t = np.linspace(0, 1, n)

    dx = -6.0*t + 2.5*np.sin(2*theta + 0.3) + rng.normal(0, 0.8, n)
    dz = 3.0*smoothstep(t) + 1.5*np.cos(3*theta) + rng.normal(0, 0.8, n)
    alpha = 0.45*np.sin(theta) + 0.15*np.sin(5*theta) + rng.normal(0, 0.05, n)
    beta = 0.40*np.cos(theta + 0.2) + rng.normal(0, 0.05, n)
    phi = 0.25*np.sin(2*theta - 0.8) + rng.normal(0, 0.05, n)

    dx[90:] += 7.0
    dz[90:] -= 3.0
    phi[90:] += 0.4

    dx[190:] -= 10.0
    dz[190:] += 5.0
    alpha[190:] -= 0.35

    dx[260:] += 5.0
    beta[260:] += 0.25

    pose = {
        "dx_px": dx,
        "dz_px": dz,
        "alpha_rad": np.deg2rad(alpha),
        "beta_rad": np.deg2rad(beta),
        "phi_residual_rad": np.deg2rad(phi),
    }
    return canonicalise_true_pose(pose)
```

## Object-frame drift

```python
def smoothstep(t):
    return 3*t*t - 2*t*t*t

def object_motion_thermal(n_views: int):
    t = np.linspace(0, 1, n_views)
    return {
        "tx_obj_px": -12.0 * smoothstep(t),
        "ty_obj_px": 2.0 * np.sin(2*np.pi*t),
        "tz_obj_px": 1.5 * t,
        "rot_obj_z_rad": np.deg2rad(0.20 * smoothstep(t)),
    }
```

## Nuisance generation

```python
def apply_nuisance_and_noise(proj, nuisance, clean=False, seed=0):
    if clean:
        return proj.astype(np.float32)

    rng = np.random.default_rng(seed)
    out = proj.copy()

    if nuisance.gain is not None:
        out *= nuisance.gain[:, None, None]

    if nuisance.offset is not None:
        out += nuisance.offset[:, None, None]

    if nuisance.background is not None:
        out += nuisance.background

    if nuisance.stripe_columns is not None:
        for col, bias in nuisance.stripe_columns:
            out[:, :, col] += bias

    sigma = nuisance.noise_sigma_fraction * robust_scale(out)
    out += rng.normal(0, sigma, size=out.shape)

    if nuisance.hot_pixels_fraction:
        mask = rng.random(out.shape) < nuisance.hot_pixels_fraction
        out[mask] += rng.normal(5*sigma, sigma, size=mask.sum())

    if nuisance.dead_pixels_fraction:
        mask = rng.random(out.shape) < nuisance.dead_pixels_fraction
        out[mask] = 0.0

    return out.astype(np.float32)
```

## Truth projector

The preferred truth projector is the same high-quality forward model as TomoJAX reference, but with the true geometry. If that is not available yet, use a simple but explicit temporary projector and mark generated projections as provisional.

Important: the generator must write:

```json
{
  "truth_projector": "tomojax_jax_reference_v1",
  "projector_commit": "...",
  "provisional": false
}
```

Do not mix provisional and benchmark-grade datasets.

## Current TomoJAX baseline expectations

The benchmark runner should support an adapter layer:

```python
class CurrentTomoJAXAdapter:
    def run(self, dataset_dir: Path, profile: str, output_dir: Path) -> BenchmarkRun:
        # Convert synthetic manifest into current TomoJAX config.
        # Run current implementation.
        # Parse current run artifacts into common schema.
        ...
```

The adapter should not assume current TomoJAX emits all new artifacts. It should fill missing fields with:

```json
{
  "available": false,
  "reason": "not_emitted_by_current_tomojax"
}
```

## Reimagined solver benchmark expectations

```python
class ReimaginedTomoJAXAdapter:
    def run(self, dataset_dir: Path, profile: str, output_dir: Path) -> BenchmarkRun:
        result = tomojax.recon(
            dataset_dir,
            align="auto",
            profile=profile,
            output_dir=output_dir,
        )
        return parse_artifacts(output_dir)
```

## Report comparison

Compare current and reimagined on:

```text
final volume metrics
realised forward geometry metrics
canonical parameter metrics
runtime
number of reconstruction calls
backend provenance
failure labels
```

Do not require parameter decomposition to match exactly if realised geometry and reconstruction are better.
