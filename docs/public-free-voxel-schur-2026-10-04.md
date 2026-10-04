# Public pose-elimination comparison, 2026-10-04

All 48 public calls completed. 5/6 cells passed all eight calls; 40/48 calls met all fixed gates. Failed rows report unsuccessful attempt costs, not accepted-result times.

The only solver change from the [stacked coupled baseline](public-free-voxel-joint-2026-10-04.md) is `gn_joint_solver="pose_eliminated"`. The small pose normal block is factored exactly, a matrix-free volume Schur system is solved by PCG, and pose increments are back-substituted. Both increments still pass the same constrained nonlinear line search. Without pose smoothness the factors are independent 5-by-5 blocks; the implementation also supports nonzero smoothness with a block-banded factorization, independently tested against a dense matrix.

The public free-voxel path, exact projector, central physical stencil, 40 inner iterations, 1e-4 full-joint residual threshold, 1e-3 pose/volume increment damping, zero TV, 64 outer iterations, 20 FISTA refresh iterations, initialization and acceptance gates remain fixed. Damping is retained in the Schur system; the implementation does not square a damped residual projection and thereby solve a different problem.

| Cell | Accepted calls | Cold s | Warm median [min, max] s | Cold image relative L2 | Cold / worst-call rotation RMSE ° | Cold shift vector RMSE px | Peak process GPU MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| parallel-clean | 8/8 | 10.110 | 5.105 [5.012, 5.173] | 0.002328 | 0.004062 / 0.004062 | 0.000111 | 320 |
| parallel-noisy | 8/8 | 10.153 | 5.136 [5.126, 5.171] | 0.005338 | 0.008182 / 0.008265 | 0.000251 | 320 |
| anisotropic-clean | 8/8 | 10.666 | 5.517 [5.489, 5.548] | 0.001495 | 0.008560 / 0.009301 | 0.000119 | 320 |
| anisotropic-noisy | 0/8 | 49.108 | 44.468 [44.165, 44.785] | 0.003483 | 0.016227 / 0.016316 | 0.000383 | 320 |
| lamino-clean | 8/8 | 19.360 | 12.384 [10.574, 14.160] | 0.085450 | 0.009568 / 0.009920 | 0.000134 | 320 |
| lamino-noisy | 8/8 | 22.982 | 15.211 [12.424, 18.839] | 0.084129 | 0.009974 / 0.009980 | 0.000246 | 320 |

Cold time includes process startup/imports, loading/transfers, setup, compilation and every independent verification. Each warm call begins again with zero voxels and nominal poses. GPU work was serial and exclusive, with numerical test jobs completed before timing. GPU memory is sampled per PID at requested 10 ms intervals over all eight calls; short peaks may be missed. The original timing baseline is retained rather than rerun concurrently.

Gates are rotation RMSE ≤0.01°, shift-vector RMSE ≤0.05 native pixels, and full-volume relative L2 ≤0.10 for parallel/anisotropic or ≤0.20 for laminography. One shared rigid object frame is applied to both volume and poses. There is no crop, attenuation fit, restricted object basis or per-view registration. Raw records retain every quality result, outer history and actual joint linear residual.

## Comparison with the recorded stacked solve

| Cell | Stacked / eliminated cold outers | Stacked / eliminated rotation ° | Accepted cold speedup | Accepted warm speedup | GPU memory fraction |
|---|---:|---:|---:|---:|---:|
| parallel-clean | 6 / 4 | 0.006376 / 0.004062 | 1.089× | 1.225× | 1.143 |
| parallel-noisy | 6 / 4 | 0.007589 / 0.008182 | 1.116× | 1.224× | 1.143 |
| anisotropic-clean | 14 / 5 | 0.008294 / 0.008560 | 1.502× | 1.979× | 1.143 |
| anisotropic-noisy | 64 / 64 | 0.016243 / 0.016227 | undefined | undefined | 1.143 |
| lamino-clean | 45 / 13 | 0.009772 / 0.009568 | 2.476× | 3.542× | 1.143 |
| lamino-noisy | 55 / 17 | 0.009877 / 0.009974 | 2.495× | 3.069× | 1.143 |

## Decision and scope

Exact pose elimination alone does not establish complete recovery. Stop this variant without tuning its passing cells or changing defaults. The whole-pilot successful-recovery baseline and 20× denominator remain unavailable; successful per-cell timings do not fill the missing coverage. This modest-motion pilot does not establish 99% recovery at the much larger ±10-pixel/±3° limits.

This change addresses alignment only. The 27-cell reconstruction scores remain those in the frozen matrix; the [spectral conditioning follow-up](system-matrix-spectral-2026-10-04.md) was rejected after broad regressions. No alignment result is substituted for reconstruction performance. The next ablations in the [research plan](research-test-plan-2026-10-04.md) separate voxel increment damping, actual inner accuracy, true image regularization and cached startup; their effects are not inferred from this run.

## Provenance and validation

Frozen runtime/driver SHA-256: `5ff8791482a48008c71a01ae8a0a8a6b57e03b6e9b403eb95eb6b00254696415`. Source stayed unchanged and every fixture array matches the original coupled comparison. The source archive includes implementation and tests. Validation passed 38 focused CPU checks, four CUDA physical checks, and 613 CPU tests (seven CUDA-dependent skips), plus lint, formatting, configured type checks and all three import contracts. Numerical checks cover weighted tilted anisotropic geometry, frozen pose DOFs, cached/streamed columns, nonzero pose smoothness, constrained Huber-TV, resume and FP32 preservation under global JAX x64 mode.

Artifacts: [all calls and histories](../bench/reference/public-free-voxel-v1-schur.json.gz), [derived comparisons](../bench/reference/public-free-voxel-v1-schur-comparison.json), [launch metadata and validation logs](../bench/reference/public-free-voxel-v1-schur-source.json), [source archive](../bench/reference/public-free-voxel-v1-schur-source.tar.gz).
