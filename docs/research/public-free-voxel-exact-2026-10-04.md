# Public free-voxel exact-projector rerun, 2026-10-04

0/6 cells met all joint gates on every cold and warm call. The six-cell rerun completed all 48 public alignment calls. Rows that fail any gate show failed-attempt time, not time to an accepted result.

The [frozen pilot](public-free-voxel-pilot.md) uses independent free-voxel objects, 61 irregular views, parallel / shifted anisotropic / 30° tilted geometry, each clean and noisy. Every call starts with zero voxels and nominal poses. The solver receives neither the truth volume nor its support. The independent FP64 oracle, fixed gates, per-outer verification, 64×20 iteration budget and seven warm restarts are unchanged.

The operator is now `AlignConfig(ray_integrator="exact")`: line integration of the zero-extended trilinear voxel basis, split at voxel-centre planes. Both reconstruction and pose evaluation use that model. The matched CUDA forward/adjoint supports changing poses, so reconstruction actually runs on Pallas CUDA. Central-difference Gauss–Newton pose updates and acceptance scoring use the exact JAX CUDA reference. The central stencil and damping are unchanged. The fused analytic pose-normal kernel was validated separately but is not used by this rerun.

The previously corrected step bounds remain in place. Therefore the sampled [bound-correction diagnostic](public-free-voxel-baseline-2026-10-04.md#step-bound-correction-diagnostic) is the closest algorithmic comparison; the original 48-call baseline also contained repeated step-bound inflation. This is not an isolated quadrature-only speed comparison, because enabling the exact CUDA reconstruction path also changes the effective backend.

| Cell | Status | Accepted calls | Cold s | Warm median [min, max] s | Image relative L2 | Rotation RMSE ° | Translation vector RMSE px | Peak GPU MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| parallel-clean | target_not_reached | 0/8 | 52.745 | 47.849 [47.678, 47.912] | 0.007058 | 0.021702 | 0.000543 | 444 |
| parallel-noisy | target_not_reached | 0/8 | 52.995 | 47.840 [47.693, 47.955] | 0.007950 | 0.022308 | 0.000653 | 444 |
| anisotropic-clean | target_not_reached | 0/8 | 46.775 | 41.678 [41.526, 41.743] | 0.006055 | 0.122042 | 0.002636 | 444 |
| anisotropic-noisy | target_not_reached | 0/8 | 47.582 | 42.390 [42.364, 42.587] | 0.006542 | 0.122751 | 0.002651 | 444 |
| lamino-clean | target_not_reached | 0/8 | 63.300 | 58.180 [58.075, 58.246] | 0.121360 | 0.218720 | 0.001598 | 444 |
| lamino-noisy | target_not_reached | 0/8 | 63.210 | 57.941 [57.878, 57.989] | 0.121390 | 0.219512 | 0.001666 | 444 |

Quality columns show the cold call. The [complete raw record](../../bench/reference/public-free-voxel-v1-exact.json.gz) retains every warm quality value, outer iteration, backend, effective step bound and stopping decision, including failures. Acceptance requires rotation RMSE ≤0.01°, translation-vector RMSE ≤0.05 native pixels and image relative L2 ≤0.10 parallel/anisotropic or ≤0.20 tilted. One shared rigid object frame is applied to both poses and volume; no per-view registration or attenuation fit is permitted.

Cold time starts before worker launch and includes imports, input loading, device setup, transfers, compilation and every verification. Warm calls include fresh initialization and the complete solve and verification. Fixture generation precedes the timed worker. Per-PID GPU memory is sampled across the whole worker at requested 10 ms intervals; short allocation peaks can be missed. GPU jobs and validation jobs were completed before this queue. During the queue only light reporting and status reads ran in the parent.

## Rotation comparison

| Cell | Original sampled ° | Corrected-bound sampled ° | Exact ° |
|---|---:|---:|---:|
| parallel-clean | 0.078435 | 0.097590 | 0.021702 |
| parallel-noisy | 0.078196 | 0.098552 | 0.022308 |
| anisotropic-clean | 0.570186 | 0.323083 | 0.122042 |
| anisotropic-noisy | 0.571898 | 0.320626 | 0.122751 |
| lamino-clean | 0.462657 | 0.245558 | 0.218720 |
| lamino-noisy | 0.465505 | 0.252345 | 0.219512 |

The separate [frozen reconstruction matrix](system-matrix-2026-10-04.md) and [cold-time profile](system-profile-2026-10-04.md) are unchanged. This alignment rerun does not establish a reconstruction speedup against ASTRA/TIGRE. The ±3°/±10-pixel distribution, identifiability and 99% recovery target remain unverified.

The [source archive](../../bench/reference/public-free-voxel-v1-exact-source.tar.gz) and [launch digest](../../bench/reference/public-free-voxel-v1-exact-source.json) preserve the exact code used, with an unchanged source hash at completion. Validation before launch: 587 CPU tests passed, 7 skipped, 208 deselected; three targeted CUDA tests passed. Independent dense matrices check forward integration, the matched adjoint, weighted pose normal equations and weighted FISTA updates. Public single-level and multiresolution tests check operator-consistent loss and checkpoint compatibility. Lint, configured type checking and import checks passed.

A six-cell successful-recovery baseline and the required 20× denominator remain undefined. No failed timing is counted as accepted recovery.
