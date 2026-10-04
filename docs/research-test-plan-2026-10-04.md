# Research-informed test plan, 2026-10-04

The user's [literature review](../reports/Tomographic%20alignment%20arXiv%20review.md)
and topic notes motivate the experiments below. The objective remains a reliable
scientific imaging library across the frozen comparison matrix. Published gains
on bundle adjustment, MRI, or global CT calibration are hypotheses for our
per-view tomography problem, not measured TomoJAX improvements.

The running experiment is immutable: source
`73a6925946726db21367d405b72593a25e9f16952177220322a62c3bf00e1d07`,
exact integration, coupled block PCG, 40 linear iterations, 64 outer iterations
with 20 FISTA refresh iterations, original fixtures/gates, one cold and seven
warm calls. It continues to completion before further GPU work. At plan creation,
parallel-clean, parallel-noisy, and anisotropic-clean passed all eight calls;
seven anisotropic-noisy calls failed rotation. Tilted cells were still pending.
There is no successful whole-pilot timing denominator yet.

Completion update: [all 48 calls finished](public-free-voxel-joint-2026-10-04.md).
Five cells passed all eight calls; anisotropic-noisy failed all eight.
The stacked-PCG variant is frozen and its tuning stops. Diagnostics and exact
pose-block elimination remain separate subsequent experiments.

Diagnostic update: [all six fixed-state checks](alignment-accuracy-2026-10-04.md)
found central-stencil error and inadequate linear accuracy. Plain pose
elimination at the existing 40-iteration budget worsens the tilted residuals,
but a [nonlinear follow-up](shared-conditioning-2026-10-04.md) improves the
tilted poses: the initial residual-only rejection was too strong. An end-to-end
accepted-result test must decide that variant. The existing analytic
Jacobian was [compared separately](public-free-voxel-analytic-2026-10-04.md) with
the same damping, budgets and gates: 40/48 accepted calls, with anisotropic-noisy
failing all eight. That derivative-only variant is stopped. Shared volume
conditioning now precedes any further pose-elimination or tolerance experiment.

The first [independent dense conditioning screen](../bench/reference/shared-conditioning-dense-2026-10-04.json.gz)
rejects a single-centre impulse Fourier approximation: after making its inverse
positive, it still worsens conditioning for parallel and tilted geometry.
An averaged Fourier approximation improves both normal systems in all three
geometries. Eight random data-space adjoint probes give a positive estimate of
that average and improve conditioning for every tested geometry/operator pair
across three seeds. This is a prototype qualification, not a workflow speedup.
Its next test uses the six saved pilot states with eight probes and seed zero,
unchanged damping/stencils, and budgets 40/160/640; setup and FFT application
costs count before any time-to-quality claim. The subsequent reconstruction sweep
has [completed with broad quality regressions](system-matrix-spectral-2026-10-04.md):
the spectral option was withdrawn from the working library. All 54 workers and
their seven warm repetitions finished; only 9/54 passed, versus 49/54 originally.
No per-cell tuning is planned for this rejected variant. Defaults remain unchanged.

After that rejection, prepare the already scheduled pose-elimination ablation:
`gn_joint_solver="pose_eliminated"`, with the same central stencil, 1e-3
pose/volume damping, 40 inner iterations and full-joint residual threshold.
All six public cells remain scheduled. Small pose blocks are factored exactly;
nonzero pose smoothness uses a block-banded factorization. After the reconstruction
sweep completed, 38 focused CPU checks, four CUDA physical checks and the full
613-test CPU selection passed (seven CUDA-dependent skips). Lint, format, type
and import checks passed. The [six-cell public comparison](public-free-voxel-schur-2026-10-04.md)
completed all 48 calls using immutable source
`5ff8791482a48008c71a01ae8a0a8a6b57e03b6e9b403eb95eb6b00254696415`.
Five cells passed all calls; noisy anisotropic failed all eight at about 0.0162°.
Tilted cold time fell about 2.5× and warm time 3.1–3.5×, while process GPU memory
rose from 280 to 320 MiB. This elimination-only variant is frozen and its tuning
stops. It does not establish the missing successful whole-pilot baseline.
The hypothesis concerns the nonlinear recovery trajectory and coupled linear
conditioning; a nearly stationary noisy anisotropic endpoint means success is
uncertain. This ablation does not directly improve the 27 reconstruction cells.

The [pilot-scale FP64 noise diagnostic](pilot-noise-2026-10-04.md) now verifies
all six cells with undamped volume elimination. Tilted geometry required exact
removal of a structurally redundant boundary block and five iterative-refinement
solves; the failed original factorization and inadequate three-solve attempt are
retained. The anisotropic noisy seed predicts 0.01736° rotation error, close to
the observed 0.01623°. Its expected local RMS is 0.02290° with a free volume and
0.01350° even with the volume known. These unconstrained local calculations are
not irreducible limits for positive or regularized estimators. They make further
linear-solver-only tuning a weak next choice. The next existing-prior screen uses
Huber-TV at the library's default weight 0.005 on all six cells, with one cold and
one warm public call each. It holds every solver setting and acceptance gate
fixed, includes clean-data bias, and makes no headline speed or reconstruction
matrix claim.

The installed ASTRA 2.5 API confirms `projector3d.direct_FP` and `direct_BP`
accept preallocated DLPack tensors. Existing `bench/compare_projectors.py`
already includes its direct JAX-device path, and the reconstruction FBP adapter
already uses direct GPU backprojection. The review's interoperability suggestion
therefore does not justify adding another adapter or resetting the comparison.

Current update: the [independent FP64 noise analysis](pilot-noise-2026-10-04.md)
completed all six cells. The resulting [fixed-weight Huber-TV screen](public-free-voxel-tv-2026-10-04.md)
also completed all six cells and failed every rotation gate. That variant is
rejected without changing defaults. The repository/documentation cleanup is
complete. The [cache screen](system-cache-2026-10-04.md) then completed all 135
workers across 33 cells and changed-data controls. It verifies reuse and lower
startup cost in JAX paths, with unchanged quality failures and no benefit to
Fourier paths. This partial improvement stops at the configuration experiment;
no library defaults change. The optimization goal remains open.

The [model-discrepancy controls](model-discrepancy-2026-10-04.md) have now
completed all 27 reconstruction and six alignment cells. Matched-data controls
retain slow tilted recovery, while independent sharp-object data drive large
corrections away from the true sampled image. A separate independent SVD check
shows that the rejected image-space preconditioner changes null-space solution
selection even with consistent noiseless data. The earlier positive-definite
conditioning screen did not check that property. New numerical regression
coverage therefore checks underdetermined solution selection and initialization
before another solver candidate is qualified. The next small prototype should
combine range-preserving filtered-adjoint directions with explicit regularization,
verify the original data objective, and select any regularization from data or
declared noise. This is an unimplemented candidate, not a measured improvement.
All original full-workflow gates and the requirement to test every cell remain.

The [first filtered Krylov/GCV qualification](hybrid-krylov-qualification-2026-10-04.md)
completed 36 small physical runs and 2,304 iterates. Original-objective and
null-space checks passed, but the candidate improved only a subset and its
64-vector image/projection bases would require 77 GiB at the showcase size,
before other allocations. It is rejected without runtime integration or tuning
the passing cases. All 33 full-workflow cells are explicitly recorded as not run
because the prerequisite qualification failed. This closes that particular
prototype, not the broader optimization goal or all regularized methods.

## Scheduled cells and decision rules

Every alignment candidate is checked on **parallel-clean, parallel-noisy,
anisotropic-clean, anisotropic-noisy, lamino-clean, lamino-noisy**. There is no
per-cell parameter tuning. The parallel plateau discussed in the review is
already cleared by the current coupled implementation; the noisy anisotropic
plateau is the first remaining failure. Its actual normal-equation residual
often exceeds its starting residual despite a requested tolerance of `1e-4`.
The 40-iteration cap and conditioning must be distinguished from the requested
tolerance. No current result proves an identifiability or precision floor.

The reconstruction experiments schedule every cell below, all at 180 views:

| Data | Parallel | Shifted odd anisotropic | 30-degree laminography |
|---|---|---|---|
| gaussian-v1 | 64, 128, 256 | 64, 128, 256 | 64, 128, 256 |
| structured-v1 | 64, 128, 256 | 64, 128, 256 | 64, 128, 256 |
| structured-noisy-v1 | 64, 128, 256 | 64, 128, 256 | 64, 128, 256 |

The separate 256/512-cubed, 720-view controls remain additional controls. Existing
acceptance gates, independent verification and failure reporting stay fixed.
Cold time includes setup, transfers, compilation and verification; warm time,
quality and per-process peak GPU memory accompany every cell. Reconstruction
time and memory use the same fastest applicable accepted external workflow as
their denominator. Missing/failed comparisons keep the whole score undefined.
Stop variants that only improve a subset; do not tune their winning cells.

## Ordered experiments

1. **Finish and publish the current coupled comparison.** Preserve all 48 calls
   and its source archive, including failures. This closes the stacked-PCG
   variant before introducing another solver. Only successful cells acquire a
   successful-recovery baseline; do not divide by a previous failed attempt.

2. **Check the remaining accuracy limit before tuning performance.** Use the
   existing independent dense-matrix oracle and exact-projector tests to compare
   central differences, analytic trilinear pose derivatives, and FP64 reference
   perturbations. Include axis-aligned grid knots, oblique/tilted poses, shifted
   detectors, anisotropic pitches, and all pose DOFs. Measure derivative error
   against step size and distinguish discrete basis effects from arithmetic.
   On small dense problems, inspect the reduced Hessian and noise sensitivity
   after separating verified common-object gauge directions. Extend the
   diagnostic to the pilot only with explicitly converged inner solves; report
   uncertainty if those solves are inaccurate. Do not change the volume basis,
   acceptance gauge, or any gate for this diagnostic.

3. **Test exact pose-block elimination as the next solver change.** Following
   [LAP](https://arxiv.org/abs/1705.09992), eliminate the pose variables from the
   linearized system and back-substitute after solving for voxels. First match
   an independent dense joint solve with the *same* damping, active set,
   regularizer, Jacobian and constraints. Then compare against stacked PCG at
   equal operator work across all six cells. Five-by-five blocks apply when
   pose smoothness is zero; nonzero smoothness requires the coupled banded pose
   system, not independent blocks or an unbounded dense matrix at scan scale.
   Keep joint nonlinear acceptance on the constrained volume-and-pose pair.
   This targets the common conditioning/iteration limiter, especially tilted
   and anisotropic scans. It does not directly improve the 27 reconstruction
   timings or establish the large-motion robustness target.

4. **Ablate damping and inner accuracy separately.** After validating elimination,
   compare current voxel increment damping with zero voxel damping while keeping
   the true image regularizer unchanged. Test rank-deficient and constrained
   cases explicitly; Huber-TV does not automatically make every volume system
   positive definite, and the frozen pilot has zero TV weight. Treat
   [Hong et al.](https://openaccess.thecvf.com/content_cvpr_2017/html/Hong_Revisiting_the_Variable_CVPR_2017_paper.html)
   as motivation for this ablation, not a universal ban on damping. Next test a
   single residual-driven inner-accuracy policy, with actual residual and pose
   increment checks and an explicit work cap. Tightening a nominal tolerance
   without achieving it is not an accuracy improvement. This follows the
   inexact-solve issue in [van Leeuwen et al.](https://arxiv.org/abs/1705.08678).
   The unchanged noisy anisotropic result after analytic derivatives, longer
   fixed-state solves and end-to-end elimination makes a purely numerical
   explanation less convincing. Independently verified reduced-Hessian/noise
   sensitivity and a true image-prior experiment must distinguish this from
   regularization bias or weak observability. Neither increment damping nor
   smaller linear residuals alone supplies that evidence. Keep every original
   cell and gate; no inference from the tiny illustrative Hessian replaces a
   pilot-scale calculation.

5. **Unify reconstruction and alignment acceleration.** Test a geometry-derived
   volume preconditioner, or a multilevel correction, on the plain reconstruction
   operator and the pose-eliminated operator. Begin with tilted, irregular,
   shifted and anisotropic geometry; assess all 27 reconstruction cells and all
   six alignment cells together. Check symmetry, positivity, boundary effects,
   DC scaling, quality and memory before timing. A Fourier approximation may
   precondition the exact normal operator; it must not silently replace it.
   A ramp-filtered unmatched backprojector requires a compatible method such as
   [BA-GMRES](https://arxiv.org/abs/2201.07408), not a false claim of a matched
   CGLS adjoint. The target is lower total time and worst-cell time, including
   preconditioner setup, rather than fewer iterations alone.

6. **Separate cached startup and capture-range experiments.** For startup, test
   default versus zero persistent-cache compile threshold in isolated cache
   directories using existing workflows. Record empty-cache first use,
   populated-cache fresh process, and same-process warm calls separately; verify
   actual Pallas cache hits and quality. A warm persistent cache cannot be
   relabeled as empty-cache cold performance. This targets the measured common
   setup/compile cost, not the failed reconstruction's quality. Before updating
   external comparisons, verify ASTRA's direct JAX interface locally and include
   transfers/synchronization consistently. For larger motion, test data-only
   seeding and coarse-to-fine capture followed by coupled refinement on the
   predefined identifiable distribution. The initialization sensitivity in
   [Okunola et al.](https://arxiv.org/abs/2605.06336) is evidence to test switching,
   not a reason to add an unmeasured warmup to every modest-motion case.

## Algebra and measurement safeguards

For pose block `C = J.T W² J + H_pose + lambda_pose I` and cross block `B`,
the volume Schur system is `H_volume - B C^-1 B.T`, with the corresponding
eliminated right-hand side. With pose damping, `I - J C^-1 J.T` is generally
**not an orthogonal projector**. Running ordinary least squares on that matrix
times `A` would square it and solve a different problem. Verify the Schur system
directly, or derive the correct augmented least-squares operator before CGLS.

Common-object gauge directions require a compensating change in both poses and
volume. Derive them for the actual pose convention and geometry, then check
their action on the independent projector. Do not remove every small-eigenvalue
mode: weak observable motion is not gauge. A fixed finite trilinear grid need
not represent every rigidly rotated object exactly. The pilot already uses
`gauge_fix="none"` for detector-frame translations and scores one common object
frame in verification; mean-translation fixing is not active in these runs.

Analytic derivatives should be evaluated in the current trilinear basis before
considering a new basis. [Jiang et al.](https://arxiv.org/abs/2508.13304) and
[Haouchat et al.](https://arxiv.org/abs/2606.21405) motivate that test, but do not
establish our derivative accuracy or justify a claimed FP32 floor. The existing
exact pose-normal kernel is a starting point; no replacement kernel is assumed
necessary before its measured error is known.

Use the existing tests, benchmark drivers and one-off analysis artifacts. This
plan adds no benchmark framework. All implementation changes require a stated
shared limiter, declared cell coverage, and full failure-inclusive results.
