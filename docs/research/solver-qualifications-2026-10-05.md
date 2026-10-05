# Solver qualification follow-up, 2026-10-05

Two more solver variants are stopped before public integration. The augmented nonnegative least-squares solver does not reproduce all selected dense solutions. A range-preserving detector-filter preconditioner passes its algebra checks but worsens image error in seven of 24 fixed-work controls. Neither establishes a reconstruction speedup or improved pose recovery.

The public results remain [26/27 accepted reconstruction pairs](system-matrix-2026-10-04.md) and [5/6 alignment cells passing](public-free-voxel-reuse-2026-10-04.md). No production source, public configuration default, fixture, quality gate or timing denominator changes.

## Augmented nonnegative solve and certificate audit

The previous [nonnegative regularization qualification](nonnegative-cv-qualification-2026-10-04.md) selects a physical-gradient penalty from held-out rays. This follow-up keeps those 24 selected weights and solves the same augmented system `[A; lambda D]` through matrix-free callbacks, using SciPy 1.17.1 trust-region reflective least squares with LSMR. The frozen settings are 128 outer iterations, 512 inner iterations, outer tolerance 1e-12 and LSMR tolerance 1e-14.

The original checker passes **2/24** controls. Its hard active-set gradient test is discontinuous near zero: it treats positive coefficients above a machine-roundoff bound as free, even when their positive gradient and very small value represent an almost-active constraint. [SciPy documents](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html) that this method maintains strictly feasible iterates and uses bound-scaled optimality. The original result is retained; the audit does not rewrite it.

A second run adds a continuous, dimensionless residual: `||u - max(0, u - v)||`, where `u = x / ||x||` and `v = gradient / ||B.T @ target||` for the augmented operator B. It is zero exactly at nonnegative first-order stationarity for feasible x. This is an additional certificate with threshold 1e-8, not an interchangeable normalization of the original gradient check. No solver settings, estimator weights, image-agreement threshold or public acceptance gates change. The certificate is also evaluated on each independent NNLS reference.

The additional certificate, feasibility, successful termination and 1e-4 relative agreement with the dense image pass together in **12/24** controls. Image agreement alone passes 21/24. All three smooth clean matched-data cases still exhaust the outer budget and fail image agreement. The solver is rejected without increasing its budget.

| Geometry | Dense-image difference, smooth clean matched | Forward / adjoint calls | Outer steps |
|---|---:|---:|---:|
| lamino | 0.000367691 | 63865 / 63594 | 128 |
| anisotropic | 0.00671323 | 63892 / 63369 | 128 |
| parallel | 0.0397731 | 64140 / 63557 | 128 |

These calls measure numerical conformance to the selected estimator. They are not time-to-public-quality measurements. The audit takes 52.14 seconds after imports and peaks at 230.34 MiB host RSS, including independent validation. No GPU-memory measurement is made.

## Range-preserving preconditioner

The second experiment solves the original unregularized normal equations by PCG, using the inverse-preconditioner `M = alpha I + A.T F A`. F is a positive detector-plane Laplacian plus a positive low-frequency floor, applied with zero padding in physical detector coordinates. Alpha is 0.001 times an eight-probe estimate of `trace(A.T F A) / voxel_count`, with one fixed seed. The rule depends on geometry, not measurements or truth.

This construction maps `range(A.T)` into itself. A zero-start unregularized solve therefore preserves the intended minimum-Euclidean-norm solution selection in exact arithmetic; a supplied initial null component is retained. Independent overdetermined, underdetermined and rank-deficient algebra controls agree with SVD reference solutions to better than 6e-13 relative and preserve null components to better than 2e-14 absolute.

The first algebra attempt omitted a convergence stop and continued CG after a singular system was solved, producing a large roundoff-driven error. That attempt and its source are retained. The corrected recurrence checks an explicit normal residual when its recursive residual reaches 1e-12 of the initial norm. Physical controls use the same corrected recurrence for both ordinary and preconditioned solves.

The physical screen reuses all 24 controls: 8×7×6 free voxels, five irregular views, a shifted 9×7 detector, three geometries, smooth/sharp objects, matched/independent data and clean/1% RMS noise. Both methods receive at most **256 forward-plus-adjoint applications**. Counts include the candidate's eight setup projections, extra forward/adjoint pair per preconditioner application and any explicit stopping checks. FFT work remains additional overhead; counts are an algorithm-work diagnostic, not a timing comparison.

The candidate improves image error in 17/24 controls but worsens seven; the worst error ratio is **1.526**. Its explicit normal residual is larger in every control at that work budget. A smaller image error under noise or model mismatch can result from incomplete fitting and does not establish faster convergence. This variant is stopped under the predeclared rule against subset-only improvement.

| Geometry / object | Data | Noise RMS | Ordinary L2 | Candidate L2 | Candidate / ordinary |
|---|---|---:|---:|---:|---:|
| lamino / smooth | matched | 0.00 | 0.216943 | 0.326219 | 1.5037 |
| lamino / smooth | matched | 0.01 | 0.299897 | 0.330103 | 1.1007 |
| lamino / smooth | independent | 0.00 | 0.424988 | 0.404958 | 0.9529 |
| lamino / smooth | independent | 0.01 | 0.480672 | 0.407515 | 0.8478 |
| lamino / sharp | matched | 0.00 | 0.194144 | 0.294992 | 1.5195 |
| lamino / sharp | matched | 0.01 | 0.302497 | 0.297464 | 0.9834 |
| lamino / sharp | independent | 0.00 | 1.44112 | 0.601806 | 0.4176 |
| lamino / sharp | independent | 0.01 | 1.42151 | 0.60443 | 0.4252 |
| anisotropic / smooth | matched | 0.00 | 0.0789653 | 0.120501 | 1.5260 |
| anisotropic / smooth | matched | 0.01 | 0.22999 | 0.12508 | 0.5439 |
| anisotropic / smooth | independent | 0.00 | 0.517179 | 0.37046 | 0.7163 |
| anisotropic / smooth | independent | 0.01 | 0.543481 | 0.36991 | 0.6806 |
| anisotropic / sharp | matched | 0.00 | 0.158538 | 0.215939 | 1.3621 |
| anisotropic / sharp | matched | 0.01 | 0.296091 | 0.219575 | 0.7416 |
| anisotropic / sharp | independent | 0.00 | 1.7625 | 0.718469 | 0.4076 |
| anisotropic / sharp | independent | 0.01 | 1.80953 | 0.724978 | 0.4006 |
| parallel / smooth | matched | 0.00 | 0.085967 | 0.106376 | 1.2374 |
| parallel / smooth | matched | 0.01 | 0.255905 | 0.111784 | 0.4368 |
| parallel / smooth | independent | 0.00 | 0.290608 | 0.286795 | 0.9869 |
| parallel / smooth | independent | 0.01 | 0.40215 | 0.288809 | 0.7182 |
| parallel / sharp | matched | 0.00 | 0.223214 | 0.259382 | 1.1620 |
| parallel / sharp | matched | 0.01 | 0.341503 | 0.264737 | 0.7752 |
| parallel / sharp | independent | 0.00 | 1.98728 | 0.559726 | 0.2817 |
| parallel / sharp | independent | 0.01 | 1.88661 | 0.563702 | 0.2988 |

## Scope and retained evidence

The PCG recurrence uses a bounded number of image/data vectors. The tiny diagnostic harness additionally retains iterates for inspection; neither its dense operator nor this retained history is a production implementation or evidence of an 8 GB fit.

The [compressed record](../../bench/reference/solver-qualifications-2026-10-05.json.gz) retains both complete augmented-solver runs, every preconditioner control and image-error history, algebra checks, failed preflight, source and hashes. Third-party SciPy implementation provenance is stored as version, source URL and hash, without redistributing its source text. Hashes are listed in the [archive catalog](../../bench/reference/archives.json).

All 27 reconstruction and six alignment cells remain explicitly listed as `not_run_failed_small_solver_qualifications`, with null new cold/warm time, quality and process GPU memory. Existing fastest-cold ASTRA/TIGRE choices and their same-workflow memory comparators are retained; no new ratio is inferred. No benchmark framework is added. The optimization goal remains open.
