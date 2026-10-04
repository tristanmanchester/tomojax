# Public analytic-Jacobian ablation, 2026-10-04

5/6 cells passed all eight calls; 40/48 calls met every fixed gate. All calls completed. Failed rows report time spent on a failed attempt, not accepted-result time.

Only the existing `gn_jacobian="autodiff"` option changed from the [coupled central-stencil comparison](public-free-voxel-joint-2026-10-04.md). The public free-voxel path, exact integrator, 40-iteration stacked PCG, 1e-3 pose/volume increment damping, zero TV, 64 outer iterations, 20 FISTA refresh iterations, initialization, fixtures and gates stayed fixed. This follows the [independent derivative diagnosis](alignment-accuracy-2026-10-04.md), which found much smaller analytic-column error at terminal poses.

| Cell | Accepted calls | Cold s | Warm median [min, max] s | Cold image relative L2 | Cold rotation RMSE ° | Cold shift vector RMSE px | Peak process GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| parallel-clean | 8/8 | 11.619 | 6.456 [6.407, 6.510] | 0.002011 | 0.006371 | 0.000153 | 280 |
| parallel-noisy | 8/8 | 11.573 | 6.444 [6.397, 6.525] | 0.004976 | 0.007638 | 0.000298 | 280 |
| anisotropic-clean | 8/8 | 16.028 | 11.036 [11.015, 11.095] | 0.001178 | 0.008868 | 0.000124 | 280 |
| anisotropic-noisy | 0/8 | 47.053 | 41.862 [41.823, 42.149] | 0.003480 | 0.016165 | 0.000383 | 280 |
| lamino-clean | 8/8 | 50.863 | 44.731 [41.023, 45.587] | 0.086303 | 0.009910 | 0.000135 | 280 |
| lamino-noisy | 8/8 | 51.975 | 45.801 [39.640, 52.743] | 0.082889 | 0.009777 | 0.000245 | 280 |

The [raw record](../bench/reference/public-free-voxel-v1-analytic.json.gz) retains all quality values and outer histories, including actual linear residuals. Gates remain rotation RMSE ≤0.01°, shift-vector RMSE ≤0.05 native pixels, and full-volume relative L2 ≤0.10 for parallel/anisotropic or ≤0.20 for laminography. Verification uses one common rigid frame for both poses and volume, with no crop, attenuation fit or per-view registration.

Cold time includes process startup, imports, loading/transfers, setup, compilation and verification. Each warm call also starts from zero voxels and nominal poses. Independent fixture generation is outside the worker. Peak GPU memory is sampled per PID at requested 10 ms intervals across all eight calls; short peaks may be missed. GPU execution was exclusive. CPU-only numerical tests and small dense analyses ran during parts of the queue, so these observed timings may contain CPU contention and are not used to assert a speedup.

## Comparison and decision

| Cell | Central rotation ° | Analytic rotation ° | Central / analytic cold outer count |
|---|---:|---:|---:|
| parallel-clean | 0.006376 | 0.006371 | 6 / 6 |
| parallel-noisy | 0.007589 | 0.007638 | 6 / 6 |
| anisotropic-clean | 0.008294 | 0.008868 | 14 / 14 |
| anisotropic-noisy | 0.016243 | 0.016165 | 64 / 64 |
| lamino-clean | 0.009772 | 0.009910 | 45 / 48 |
| lamino-noisy | 0.009877 | 0.009777 | 55 / 50 |

The more accurate derivative alone does not establish complete recovery. Stop this derivative-only variant without tuning its passing cells or changing defaults. The noisy anisotropic gate remains binding. A successful six-cell baseline and the 20× denominator remain unavailable; this modest-motion pilot does not establish the 99% large-motion target.

The next candidate targets volume conditioning in plain reconstruction and the pose-eliminated alignment system together. All 27 frozen reconstruction cells and all six alignment cells remain scheduled; the existing ASTRA/TIGRE time and memory denominators are unchanged. Small dense preconditioner results are qualification evidence only, not workflow improvements.

## Provenance

Frozen runtime/driver SHA-256: `eedbdf7ba2c3d70f2894d8688edff6a1e155bf251b0b50c54b2242fd62efa7ea`. Source stayed unchanged and every fixture array matches the central-stencil run. The [source archive](../bench/reference/public-free-voxel-v1-analytic-source.tar.gz) and [launch metadata](../bench/reference/public-free-voxel-v1-analytic-source.json) retain code and tests. The runtime is the same implementation previously validated by 597 CPU and five CUDA tests; six existing fixture/verifier tests and lint/format checks passed after adding the driver argument. Independent FP64 checks also validated both analytic implementations across the six cells. Later experimental preconditioner edits in the working tree were excluded from this immutable run.
