# Public free-voxel alignment pilot v1

Declared before timed execution on 2026-10-04. This is the first controlled
public-API baseline attempt, not evidence that the large-motion or 99% recovery
targets have been met. A failed run has no successful-recovery baseline time.

The [completed baseline report](public-free-voxel-baseline-2026-10-04.md)
retains all 48 calls. All six cells missed the rotation gate.
The [exact-projector rerun](public-free-voxel-exact-2026-10-04.md) also completed
48 calls under these same fixture definitions, budgets and gates.
The [coupled experiment](public-free-voxel-joint-2026-10-04.md) subsequently
completed 48 calls with five accepted cells and one retained noisy rotation failure.

After that rerun, the requested [Opus review](opus-review-2026-10-04.md)
recommended coupling volume and pose updates. The next controlled comparison
uses `--ray-integrator exact --gn-coupling joint` in the existing driver:
40 matrix-free PCG iterations per outer, relative residual tolerance `1e-4`,
voxel increment damping `1e-3`, and the existing pose damping and central
stencil. All six cells, fixture arrays, acceptance gates, 64×20 outer/refresh
budget, and one cold plus seven warm calls remain fixed. Positivity and pose
constraints are applied before the volume-and-pose pair is scored and accepted.
Keeping only the pose increment and scoring against the old volume would reject
valid joint descent directions, as checked with an independent dense example.

The hypothesis is slow convergence caused by coupling between free voxels and
poses. All six cells are scheduled; anisotropic and tilted cells are expected
to benefit most. The parallel rotation plateau may have a different limiter.
If only a subset improves, this line stops. The 27-cell reconstruction matrix,
cold reconstruction setup, and large-motion recovery distribution are outside
this change and remain open.

Scheduled cells are parallel-clean, parallel-noisy, anisotropic-clean,
anisotropic-noisy, lamino-clean and lamino-noisy. Each uses nominal size 32,
61 irregular half-turn views and seed 461. The anisotropic grid is 32×29×16
with physical pitches 0.8, 1.2, 1.4 and a 37×19 detector shifted by (0.27, −0.31).
Other grids are 32³ with unit pitches and 32² detectors. Laminography tilt is
30°. View positions have fixed seeded jitter bounded by ±0.3°.

The object is a seeded, spatially correlated random voxel field within a
compact support. Every voxel remains an independent reconstruction unknown;
the solver receives neither a compact phantom model nor its support or true
volume. Independent FP64 data integrate its zero-extended trilinear basis
exactly up to rounding: rays are split at all voxel-centre planes and each
segment uses two-point Gauss–Legendre quadrature, exact for the resulting
cubic polynomial. This tests a specified voxel-basis imaging model; it does
not replace real scans, material physics, or independent continuous phantoms.

True per-view rotation components are uniform within ±0.25°, and lab x/z
translations within ±0.5 native pixels. Noisy cells add seeded Gaussian noise
with standard deviation 0.1% of clean projection RMS. These are modest-motion
pilot cases; the required ±3°/±10-pixel distribution remains separate and open.

The measured solver is `tomojax.align.align`, starting from zero free voxels
and nominal poses. It uses 64 outer iterations, 20 reconstruction iterations
per outer, FP32 gather, a request for Pallas reconstruction, the central Gauss–Newton
Jacobian, L2 data loss, nonnegative voxels and zero TV weight. All five pose
parameters are active. Every outer checks accepted recovery and may terminate
only upon reaching the fixed gates. The timing includes these oracle quality
checks; this is benchmark stopping, not an automatic real-scan stopping rule.

Acceptance requires rotation-matrix angular RMSE ≤0.01°, detector translation
**vector** RMSE ≤0.05 native pixels, and full-volume relative L2 ≤0.10 for
parallel/anisotropic or ≤0.20 for tilted geometry. No amplitude fit, crop or
per-view registration is allowed. One shared rigid object-frame transform
aligns both the recovered geometry and the volume to truth; the raw pose
errors and fitted transform are also recorded. Joint identifiability across
the eventual noisy robustness distribution is not established by this pilot.

Each isolated worker makes one cold and seven repeated complete calls.
Repeated calls restart from zero voxels and nominal poses. Cold time is
externally bracketed before worker/monitor launch and includes Python startup,
imports, fixture reading, device setup, transfers, compilation and verification.
It conservatively includes the brief memory-monitor startup. Fixture generation
is outside the timed worker. Warm times include fixture reading, initialization,
transfers, reconstruction, pose updates and verification. Failed calls retain
the same measurements. The per-worker limit is 1800 seconds across its cold and
warm attempts; a timeout remains a failure with any completed samples retained.
Process GPU memory is sampled per PID at requested 10 ms intervals and can miss
short-lived peaks. Source and fixture hashes accompany the results.

Implementation: [public alignment benchmark](../bench/public_alignment_benchmark.py),
[independent voxel integrator](../bench/voxel_truth.py). The reconstruction
[whole matrix](system-matrix-2026-10-04.md) remains frozen and separate.
