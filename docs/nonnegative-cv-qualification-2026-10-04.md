# Nonnegative regularization: dense qualification and rejected matrix-free solver

A nonnegative physical-gradient estimator with held-out parameter selection improves all 24 small controls against the earlier data-selected, unconstrained Krylov control. Its projected Newton-CG implementation reproduces only **22/24** dense reference solutions and is stopped. No public solver, default, fixture, acceptance gate or workflow score changes.

Nonnegativity accounts for a substantial part of the improvement. Against fully solved unregularized nonnegative least squares, the selected prior improves **19/24** image errors and adds small bias in five clean matched-data controls. The public alignment path already enforces positivity. These reconstruction controls therefore do not establish improved joint recovery or resolve the noisy anisotropic rotation failure.

## Estimator and independent checks

The dense reference solves `min(x >= 0) ||A x - b||² + lambda² ||D x||²`. A is the independently integrated FP64 trilinear operator; D takes adjacent differences divided by physical voxel spacing, with free boundaries. The 8×7×6 grid, five irregular views, shifted 9×7 detector, three geometries, two objects, matched/independent data and clean/1% noise conditions are unchanged from the [earlier gradient qualification](hybrid-krylov-qualification-2026-10-04.md#subsequent-physical-gradient-gcv-screen).

Five ray folds use `(u + 2*v + 3*view) % 5`, retaining training rays in every view. Each fold evaluates the same 13 relative weights, `10**linspace(-4, 2, 13)`. Weights are scaled by the ratio of squared Frobenius norms of the training projection operator and physical difference operator. The lowest mean held-out squared error selects the weight; the selected estimator then fits all measurements. Neither truth images nor their errors enter this selection. There are **65 validation fits and one final fit per dataset**.

The reference uses [SciPy NNLS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html). A separate [bounded-variable least-squares implementation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html) checks every selected fit, including relative image agreement within 1e-4 and objective agreement within 1e-8. All fits satisfy the 1e-8 scaled KKT check. Independent overdetermined, underdetermined and rank-deficient algebra controls also pass.

Two unsuccessful checker runs are retained. Exact comparisons with zero incorrectly rejected independent-solver roundoff on both sides of an active bound. The corrected checker uses a 64-machine-epsilon bound tolerance without clipping either evaluated solution; the estimator, validation folds, weights and all image criteria are unchanged. The diagnosed two-solver image difference is 6.9e-15.

This experiment is an inference about constrained regularization in TomoJAX, not a reproduction of [Hansen et al.’s nonnegative smoothing study](https://arxiv.org/abs/1309.4498), which studies electrochemical impedance data and different parameter-choice rules. That work motivates independently checking the constraints and smoothing together; it does not validate tomography performance.

## Complete small-control image results

All errors are relative full-volume L2. “Krylov” is the unchanged earlier GCV-stopped unconstrained control; “NNLS” is full-data unregularized nonnegative least squares; “selected” is the dense constrained prior selected from held-out rays. These small controls have no public-workflow acceptance score.

| Geometry / object | Data | Noise RMS | Krylov L2 | NNLS L2 | Selected L2 | Selected relative weight |
|---|---|---:|---:|---:|---:|---:|
| lamino / smooth | matched | 0.00 | 0.234718 | 9.05049e-05 | 0.000457048 | 0.0001 |
| lamino / smooth | matched | 0.01 | 0.277252 | 0.0604692 | 0.0387611 | 0.0316228 |
| lamino / smooth | independent | 0.00 | 0.397741 | 0.282959 | 0.269084 | 0.1 |
| lamino / smooth | independent | 0.01 | 0.421847 | 0.281785 | 0.268932 | 0.1 |
| lamino / sharp | matched | 0.00 | 0.206765 | 1.29456e-15 | 1.58331e-07 | 0.0001 |
| lamino / sharp | matched | 0.01 | 0.261837 | 0.0413094 | 0.0413093 | 0.0001 |
| lamino / sharp | independent | 0.00 | 1.11073 | 0.714836 | 0.500852 | 0.316228 |
| lamino / sharp | independent | 0.01 | 1.11489 | 0.713503 | 0.499343 | 0.316228 |
| anisotropic / smooth | matched | 0.00 | 0.0876576 | 0.0116377 | 0.012082 | 0.0001 |
| anisotropic / smooth | matched | 0.01 | 0.142134 | 0.0810235 | 0.0418388 | 0.1 |
| anisotropic / smooth | independent | 0.00 | 0.488193 | 0.314712 | 0.260598 | 0.316228 |
| anisotropic / smooth | independent | 0.01 | 0.472901 | 0.316899 | 0.261453 | 0.316228 |
| anisotropic / sharp | matched | 0.00 | 0.165374 | 1.19562e-15 | 5.85714e-07 | 0.0001 |
| anisotropic / sharp | matched | 0.01 | 0.202377 | 0.0474953 | 0.0354667 | 0.0316228 |
| anisotropic / sharp | independent | 0.00 | 1.36601 | 0.534153 | 0.384005 | 0.316228 |
| anisotropic / sharp | independent | 0.01 | 1.40308 | 0.534739 | 0.38641 | 0.316228 |
| parallel / smooth | matched | 0.00 | 0.0959468 | 0.0196517 | 0.0132263 | 0.0001 |
| parallel / smooth | matched | 0.01 | 0.138667 | 0.206634 | 0.0562641 | 0.0316228 |
| parallel / smooth | independent | 0.00 | 0.287124 | 0.298412 | 0.263602 | 0.1 |
| parallel / smooth | independent | 0.01 | 0.307993 | 0.304711 | 0.267427 | 0.1 |
| parallel / sharp | matched | 0.00 | 0.24129 | 9.30376e-15 | 3.64998e-06 | 0.0001 |
| parallel / sharp | matched | 0.01 | 0.265048 | 0.213241 | 0.0829227 | 0.01 |
| parallel / sharp | independent | 0.00 | 0.883286 | 0.656215 | 0.455662 | 0.316228 |
| parallel / sharp | independent | 0.01 | 0.884932 | 0.66224 | 0.456102 | 0.316228 |

## Why the matrix-free implementation is stopped

The prototype uses projected Newton steps, an active free-voxel set, diagonally preconditioned CG and a projected Armijo line search. The inner relative tolerance tightens with the outer projected-gradient residual. It starts at zero and keeps only a fixed number of image/data vectors. Independent callbacks supply the same objective as the dense reference; no retained image or projection Krylov basis is used.

The frozen conformance budget is 128 outer steps, 512 inner CG steps and relative KKT tolerance 1e-12. Relative image agreement with the dense reference must be at most 1e-4. Two clean smooth controls exhaust the outer budget with image disagreements of 0.00712 and 0.01005. Even the converged smooth tilted clean control needs more than 50,000 forward calls. These are numerical-conformance costs, **not time to an accepted reconstruction**; no public early-quality stopping measurement was made.

| Geometry / object | Data | Noise RMS | Outer steps | Forward / adjoint calls | Relative dense-image difference | Conformance |
|---|---|---:|---:|---:|---:|---|
| lamino / smooth | matched | 0.00 | 112 | 50814 / 50610 | 7.38567e-07 | pass |
| lamino / smooth | matched | 0.01 | 14 | 754 / 740 | 4.99916e-13 | pass |
| lamino / smooth | independent | 0.00 | 13 | 343 / 331 | 2.47278e-15 | pass |
| lamino / smooth | independent | 0.01 | 11 | 283 / 273 | 1.14106e-14 | pass |
| lamino / sharp | matched | 0.00 | 18 | 1579 / 1557 | 2.52871e-15 | pass |
| lamino / sharp | matched | 0.01 | 13 | 510 / 496 | 8.84941e-15 | pass |
| lamino / sharp | independent | 0.00 | 9 | 128 / 120 | 2.0706e-14 | pass |
| lamino / sharp | independent | 0.01 | 10 | 136 / 127 | 8.40282e-14 | pass |
| anisotropic / smooth | matched | 0.00 | 128 | 60124 / 59525 | 0.00711907 | iteration limit |
| anisotropic / smooth | matched | 0.01 | 12 | 368 / 356 | 5.44368e-13 | pass |
| anisotropic / smooth | independent | 0.00 | 11 | 153 / 143 | 3.39094e-12 | pass |
| anisotropic / smooth | independent | 0.01 | 10 | 132 / 123 | 4.32588e-14 | pass |
| anisotropic / sharp | matched | 0.00 | 16 | 1331 / 1308 | 7.45166e-11 | pass |
| anisotropic / sharp | matched | 0.01 | 14 | 637 / 615 | 2.25107e-12 | pass |
| anisotropic / sharp | independent | 0.00 | 9 | 124 / 116 | 2.8806e-15 | pass |
| anisotropic / sharp | independent | 0.01 | 9 | 110 / 102 | 2.63372e-12 | pass |
| parallel / smooth | matched | 0.00 | 128 | 59068 / 58384 | 0.0100504 | iteration limit |
| parallel / smooth | matched | 0.01 | 17 | 1211 / 1179 | 2.1165e-12 | pass |
| parallel / smooth | independent | 0.00 | 15 | 551 / 534 | 2.21146e-14 | pass |
| parallel / smooth | independent | 0.01 | 10 | 266 / 257 | 1.07577e-13 | pass |
| parallel / sharp | matched | 0.00 | 21 | 2446 / 2405 | 4.9611e-12 | pass |
| parallel / sharp | matched | 0.01 | 15 | 911 / 883 | 3.37387e-10 | pass |
| parallel / sharp | independent | 0.00 | 9 | 122 / 114 | 5.63399e-14 | pass |
| parallel / sharp | independent | 0.01 | 10 | 143 / 134 | 9.11503e-16 | pass |

An inexact solution can have lower truth error than its dense reference: this happens in the failed anisotropic smooth control. That does not qualify the implementation of the selected estimator, and it is not used to change its numerical stopping criterion. The failed implementation is frozen without increasing budgets or tuning a favorable geometry.

## Cost, scope and retained evidence

The dense qualification function takes 42.43 CPU seconds after imports and peaks at 232.95 MiB host RSS. The separate matrix-free validation function takes 28.54 seconds and peaks at 226.90 MiB host RSS. Those totals include independent validation work and are not solver speed comparisons. NumPy is 2.4.5 and SciPy is 1.17.1. No GPU workload or process GPU-memory measurement is made in this experiment.

At 512³ voxels and 720 views on a 512×512 detector, an explicit FP64 projection matrix alone would require **180 PiB**. The dense reference cannot be a production implementation. Constant-order vector storage in the prototype also does not prove an 8 GiB process-memory result; a qualified GPU implementation would still need full measurement.

All 27 reconstruction and six alignment cells remain in the [raw qualification record](../bench/reference/nonnegative-cv-qualification-2026-10-04.json.gz) as `not_run_failed_matrix_free_qualification`, with null new cold/warm time, quality and GPU-memory fields. Each reconstruction cell retains its existing cold-fastest ASTRA/TIGRE workflow and that same workflow’s memory. The [frozen reconstruction comparison](system-matrix-2026-10-04.md) and [latest public alignment comparison](public-free-voxel-reuse-2026-10-04.md) remain the authoritative performance evidence.

The archive includes all 24 estimator results, all 24 matrix-free trajectories, both failed checker attempts, source and hashes, the predeclared plan and every unperformed workflow cell. Its hashes are in the [archive catalog](../bench/reference/archives.json). No benchmark framework or public API is added. The optimization goal remains open.
