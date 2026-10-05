# Pilot-scale alignment noise diagnostic, 2026-10-04

All six frozen alignment cells now have independently checked FP64 local noise calculations. The anisotropic noisy seed predicts **0.01736°** rotation error with an unknown volume, close to the public pose-eliminated result **0.01623°**. This is evidence that further linear-solver accuracy alone is unlikely to clear that cell. It is not a lower bound for positive, regularized or biased estimators, and it does not establish recovery success or a 99% success rate.

## Results

The calculations are local to the true object and poses. “Expected RMS” means the square root of expected mean-square error under the declared independent Gaussian noise model, after the derivative of the original common-object-frame score. The actual-seed column includes the fixture’s final FP32 data cast. Clean expected noise is zero; their tiny nonzero predicted errors measure that cast.

| Cell | Known-volume expected rotation RMS ° | Free-volume expected rotation RMS ° | Actual-seed free-volume prediction ° | Observed public rotation RMSE ° | Public result |
|---|---:|---:|---:|---:|---|
| parallel-clean | 0 | 0 | 9.907478e-07 | 0.004061666 | pass |
| parallel-noisy | 0.005302978 | 0.01268318 | 0.009232831 | 0.008182024 | pass |
| anisotropic-clean | 0 | 0 | 1.127023e-06 | 0.0085599 | pass |
| anisotropic-noisy | 0.01349963 | 0.022898 | 0.01736177 | 0.01622681 | fail |
| lamino-clean | 0 | 0 | 3.789825e-07 | 0.009568447 | pass |
| lamino-noisy | 0.005539635 | 0.008967707 | 0.007673086 | 0.009974349 | pass |

The observed values come from the [completed public comparison](public-free-voxel-schur-2026-10-04.md), whose cold/warm time, image and pose quality, and peak process GPU memory remain unchanged. This CPU analysis is not another reconstruction or timing baseline. It uses every free voxel and all five independent pose coordinates per view, rather than a fitted object model.

Even the known-volume anisotropic control has expected rotation RMS 0.01350° and predicts 0.01267° for the actual noise sample. This makes a derivative-precision or Krylov-tolerance explanation less convincing. The uncertainty is spread over views: the five largest per-view variance contributions account for only 9.7% of its total. These are conditional local calculations; they do not prove that every estimator must miss 0.01°.

## Independent operator and derivatives

A NumPy/SciPy sparse matrix integrates each trilinear tent basis directly. Rays are split at voxel-centre planes, and two-point Gauss quadrature integrates the polynomial on each segment exactly. This bypasses the JAX/Pallas projector. Every basis column of an independent 3×4×2 control is checked against the existing FP64 piecewise-polynomial oracle, including physical spacing, nonzero origin, detector shifts, axis-aligned rays and oblique poses. Maximum absolute basis disagreement is 2.22e-16. On the actual pilot objects, relative projection disagreement is 3.4–3.7e-16.

Independent FP64 pose columns use central perturbations of 1e-6 and 5e-6 in normalized voxel-displacement units. Their largest relative difference is 1.13e-7. Differentiating the unchanged public common-frame metric at two steps agrees within 2.2e-9. No continuous rigid-frame directions are removed from the likelihood: on a fixed finite voxel basis they need not be exact null directions. Only the existing scoring transformation is differentiated.

## Verified undamped volume elimination

For volume matrix A and pose matrix J, the residual pose columns are P = J − A argmin_Z ||AZ−J||. The reduced pose matrix is S = PᵀP. The local covariance is σ² E S⁻¹ Eᵀ, with E the physical scoring derivative. The actual-seed prediction is E S⁻¹Pᵀnoise. Exact nonlinear pose scoring of the predicted anisotropic perturbation gives 0.01736182°, versus 0.01736177° from its linear metric.

Nonzero volume columns are normalized before the dense normal solve. There is no volume damping, prior, positivity constraint, support mask or eigenvalue truncation. Normal residuals alone are insufficient: a further solve must change every projected pose column by less than 1e-7 relative, and the maximum normal residual must be below 1e-9. Parallel and anisotropic pass both checks after one refinement.

The first tilted Cholesky attempt failed. Structural matching identifies **37 boundary columns supported on 17 rows**, hence 20 structural null directions. A separate SVD verifies full row rank of that 17×37 block; its column-scaled smallest singular value is 4.91e-5. Those columns have exactly zero support outside the 17 rows. Therefore their unconstrained image spans every vector on those rows: dropping those columns and rows before solving preserves the orthogonal residual projection exactly. This would not be a valid shortcut for a positivity-constrained solve.

After that exact elimination, the tilted normal matrix remains extremely ill-conditioned. Three refinement solves still fail the fixed column-stability test and are retained as an unsuccessful attempt. Increasing the work cap allows five solves to meet the original tolerances; neither tolerance is relaxed.

| Geometry | Maximum final normal relative residual | Last projected-column relative change | Volume normal reciprocal-condition estimate | Reduced pose condition |
|---|---:|---:|---:|---:|
| parallel | 5.1e-16 | 5.47e-14 | 1.37e-06 | 8.86e+04 |
| anisotropic | 4.83e-16 | 9.9e-14 | 3e-07 | 1.96e+05 |
| lamino | 7.63e-16 | 1.81e-08 | 3.3e-18 | 3.34e+05 |

Pose conditions refer to the documented normalized displacement coordinates, not arbitrary mixed radians/pixels. No small pose eigenvalues were discarded. Raw results retain every eigenvalue, derivative check, refinement residual, translation prediction and original unsuccessful factorization. These intentionally expensive dense CPU diagnostics used up to 9.08 GiB cumulative host RSS; that number is not process GPU memory or a proposed production method.

## Next test and limits

The next test uses the existing Huber-TV image prior with its existing default weight 0.005 on parallel-clean/noisy, anisotropic-clean/noisy and lamino-clean/noisy. Everything else remains the pose-eliminated public workflow, including all gates. One cold and one warm call per cell form a screening ablation; they are not seven-repeat headline evidence. A partial result stops this fixed-weight variant. Image-prior bias must be counted on clean cells as well as noise reduction on noisy cells.

This diagnostic changes no solver defaults, acceptance gates, fixture arrays or reconstruction-matrix scores. Positivity, image priors, nonlinear initialization error and approximation bias are absent from the local covariance. Agreement with one endpoint does not establish a universal information limit. The whole-pilot successful-recovery baseline and the broader performance goals remain open.

Artifacts: [all results, failed attempts and source](../../bench/reference/pilot-noise-2026-10-04.json.gz), [small pose systems for reanalysis](../../bench/reference/pilot-noise-2026-10-04-systems.tar.gz).
