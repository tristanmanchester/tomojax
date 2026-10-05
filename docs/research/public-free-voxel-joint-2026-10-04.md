# Public coupled volume-and-pose experiment, 2026-10-04

5/6 cells passed all eight calls; 40/48 calls reached every fixed gate. All six cells and all repeats completed. A failed row reports time spent on a failed attempt, not time to an accepted result.

This is the first public free-voxel test of `AlignConfig(gn_coupling="joint", ray_integrator="exact")`. The opt-in solver uses a matrix-free block PCG update with 40 iterations, requested relative residual tolerance 1e-4, and voxel increment damping 1e-3. Pose damping, the physical central-difference stencil, the 64-outer/20-FISTA-refresh budget, initialization, fixture arrays and gates are unchanged from the [exact-projector run](public-free-voxel-exact-2026-10-04.md). Every voxel is independent; neither a restricted object model nor the truth support is supplied to the solver.

Each candidate updates both volume and poses. Positivity and pose constraints are applied before the actual nonlinear joint objective is scored. Damping enters only the increment system. Scoring just the pose increment against the old volume can reject valid joint descent; an independent dense counterexample verifies this distinction.

| Cell | Accepted calls | Cold s | Warm median [min, max] s | Cold image relative L2 | Cold rotation RMSE ° | Cold translation vector RMSE px | Peak process GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| parallel-clean | 8/8 | 11.013 | 6.255 [6.191, 6.292] | 0.001920 | 0.006376 | 0.000167 | 280 |
| parallel-noisy | 8/8 | 11.328 | 6.289 [6.245, 6.346] | 0.004950 | 0.007589 | 0.000319 | 280 |
| anisotropic-clean | 8/8 | 16.021 | 10.917 [10.882, 10.953] | 0.001169 | 0.008294 | 0.000124 | 280 |
| anisotropic-noisy | 0/8 | 48.079 | 43.106 [42.927, 43.238] | 0.003477 | 0.016243 | 0.000383 | 280 |
| lamino-clean | 8/8 | 47.942 | 43.860 [40.251, 46.492] | 0.086907 | 0.009772 | 0.000141 | 280 |
| lamino-noisy | 8/8 | 57.330 | 46.677 [42.247, 48.472] | 0.083119 | 0.009877 | 0.000249 | 280 |

The [raw record](../../bench/reference/public-free-voxel-v1-joint.json.gz) retains every cold/warm quality value, outer history, achieved linear residual, accepted step and backend. Gates remain rotation RMSE ≤0.01°, translation-vector RMSE ≤0.05 native pixels, and full-volume relative L2 ≤0.10 for parallel/anisotropic or ≤0.20 for laminography. One shared rigid object transform aligns both the estimated poses and volume; no amplitude fit, crop or per-view registration is permitted.

Cold time includes process startup, imports, setup, data loading/transfers, compilation and verification. Warm calls repeat the complete workflow from zero voxels and nominal poses. Fixture generation is outside the timed worker. Peak memory is sampled per worker PID across its cold and seven warm calls at requested 10 ms intervals; it may miss shorter peaks. Validation/GPU jobs finished before launch; during the queue the parent performed only light status/reporting and literature review.

## Comparison with the preceding exact-projector attempt

| Cell | Previous rotation ° | Coupled rotation ° | Previous / coupled outer count (cold) | Previous / coupled peak MiB |
|---|---:|---:|---:|---:|
| parallel-clean | 0.021702 | 0.006376 | 64 / 6 | 444 / 280 |
| parallel-noisy | 0.022308 | 0.007589 | 64 / 6 | 444 / 280 |
| anisotropic-clean | 0.122042 | 0.008294 | 64 / 14 | 444 / 280 |
| anisotropic-noisy | 0.122751 | 0.016243 | 64 / 64 | 444 / 280 |
| lamino-clean | 0.218720 | 0.009772 | 64 / 45 | 444 / 280 |
| lamino-noisy | 0.219512 | 0.009877 | 64 / 55 | 444 / 280 |

The preceding exact-projector experiment failed all 48 calls. Its times cannot supply a successful-recovery speedup denominator. Successful rows here provide individual accepted-result measurements, but the complete six-cell successful baseline and its 20× target remain open while any cell fails. The larger-motion ±3°/±10-pixel distribution and 99% target are not tested by this modest-motion pilot.

## Decision and next test

The current stacked-PCG variant achieves only partial acceptance. Stop tuning this variant; do not optimize its passing cells or promote it to the default. Rotation error improves across the geometries, but the unmet gate is retained. The requested [literature-informed plan](research-test-plan-2026-10-04.md) first checks achieved linear accuracy, derivative precision and weak/gauge modes, then tests exact pose-block elimination as a separately controlled solver. Do not treat a low data residual as proof of correct geometry.

The frozen [27-cell reconstruction matrix](system-matrix-2026-10-04.md), its [cold profile](system-profile-2026-10-04.md) and ASTRA/TIGRE time/memory denominators are unchanged. This alignment experiment makes no reconstruction speedup claim. The research plan includes a shared preconditioner experiment across both matrices.

## Validation and provenance

597 CPU tests passed, 7 skipped and 210 deselected; five targeted CUDA tests passed. Checks cover independent dense Schur and physical projector references, cached and streamed pose columns, active/frozen DOFs, weighted data, Huber-TV, positivity, bounds, public single-/multiresolution objective consistency, and checkpoint continuation. Formatting, lint, configured type checks and import contracts passed. An initial multiresolution schedule integration failure was fixed and the full CPU suite rerun before this source was frozen.

Both joint forward/adjoint and acceptance scoring use the exact Pallas CUDA operator; central pose columns use the exact JAX reference on CUDA. A bounded 64 MiB pose-column cache has a per-view recomputation fallback for larger acquisitions. Actual linear residuals are recomputed: hitting the PCG cap is not reported as convergence.

Source SHA-256: `73a6925946726db21367d405b72593a25e9f16952177220322a62c3bf00e1d07`. All six fixture arrays match the preceding exact rerun, and the source digest is unchanged at completion. The [archive](../../bench/reference/public-free-voxel-v1-joint-source.tar.gz) and [launch metadata](../../bench/reference/public-free-voxel-v1-joint-source.json) preserve code, tests and environment inputs for review.
