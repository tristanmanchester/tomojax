# Image-model discrepancy and solution selection, 2026-10-04

All 33 diagnostic cells completed. They distinguish three issues that must be
handled before another conditioning-only solver change: incomplete recovery even
with exactly matched data, fitting error caused by a finite image model, and the
choice of unmeasured image components in underdetermined systems. No new accepted
reconstruction, alignment recovery, or speedup is claimed. The frozen matrix
remains at 26/27 accepted pairs; public alignment remains at 5/6 cells.

## Findings and next decision

- Smooth tilted reconstruction retains about 15–16% image error after 64 CGLS
  iterations even on perfectly matched, noiseless data. Starting from the known
  truth and fitting independent smooth measurements moves the image by less than
  1%. Poor finite-budget recovery therefore remains a separate problem from
  the projection model's approximation error in those controls.
- Sharp anisotropic size 64 reconstructs matched data to about 3.7% image error.
  Starting at the exact truth and fitting the independent clean measurements
  instead moves the image to about 30.6% error. The ordinary operator's projection
  discrepancy is sufficient to drive a large unwanted image correction; solving
  those equations more aggressively does not establish scientific accuracy.
- Cubic Joseph interpolation substantially reduces the smooth phantom's forward
  discrepancy, but does not consistently improve sharp-object discrepancies.
  It worsens the model agreement in the exact-trilinear alignment fixtures.
  These projection-only results do not justify a blanket interpolation change.
- A separate independent small-matrix check shows that the rejected image-space
  spectral preconditioner can select a different null-space component even for
  noiseless, exactly consistent data. Better damped condition numbers alone did
  not qualify it for an undamped, underdetermined reconstruction.

The next candidate should combine explicit regularization with a Krylov space
that preserves the intended solution selection. A small prototype will test
filtered-adjoint directions while evaluating the original data residual and
image penalty, rather than assuming that a filtered normal residual measures
the same objective. The filter must cover tilted, shifted, unequal-spacing and
irregular geometry first. Parameter selection must use measured data or declared
noise, never the reference image or the acceptance error. Memory and setup costs
must be counted before a full workflow trial. This is a candidate to qualify,
not an implemented improvement or a reason to restore the rejected preconditioner.

Follow-up: the [first filtered Krylov/GCV prototype](hybrid-krylov-qualification-2026-10-04.md)
passed the algebra checks but failed its small physical quality and storage
qualification. It was stopped before public workflow integration; the frozen
comparisons were not rerun for that rejected candidate.

The research basis is [filtered BA-GMRES](https://arxiv.org/abs/2201.07408) and
[hybrid regularization](https://arxiv.org/abs/2602.17892). Those papers motivate
the experiment; neither establishes its efficacy in TomoJAX. In particular,
filtered objectives, projected regularization and correlated model error need
independent checks. Published two-dimensional hybrid results do not establish
three-dimensional alignment accuracy or bounded-memory performance here.

## Controls and interpretation

Each original reconstruction fixture is unchanged. Three projectors evaluate
the stored reference voxel array: linear Joseph, cubic Joseph, and exact line
integration of the trilinear basis. Their output is compared with the independent
continuous-phantom measurements. Exact integration of a voxel basis is not exact
integration of the underlying continuous phantom.

Two controls then use the public, unregularized linear-Joseph CGLS solver for
64 iterations, with zero tolerance and its existing roundoff checks:

1. **Matched from zero:** replace measurements with that projector's output at
   the stored truth, then reconstruct from zero. This intentionally removes model
   discrepancy and noise. It is an inverse-crime diagnostic, never an independent
   acceptance result. Clean/noisy suite partners share this same matched control.
2. **Independent from truth:** retain the original independent measurements but
   initialize the solver at the exact truth. Its final image error measures how
   far fitting those measurements moves this particular iterate. This is a
   truth-assisted diagnostic, not a reconstruction from available scientific data.

Both controls run twice from their stated initialization in each fresh process.
Sixty-four iterations are a fixed diagnostic budget, not proof of convergence,
an irreducible error, or a replacement for the original 1–256 acceptance search.
The noisy controls include noise as well as model discrepancy. The existing
[noise analysis](pilot-noise-2026-10-04.md) remains the quantitative alignment
noise diagnostic; this experiment does not repeat joint recovery.

An independent FP64 piecewise-polynomial oracle checks 495 selected rays per
cell against exact CUDA trilinear integration, using the same FP32 pose values.
All 33 comparisons meet the fixed 3e-5 relative tolerance. These checks validate
sampled forward calculations, not every ray or any recovery gate.

## Reconstruction model and image errors

All errors are relative L2. J = linear Joseph; C = cubic Joseph; E = exact
trilinear integration. The last two columns are warm control image errors.
The gate column is the unchanged independent-image gate, shown only for context.
No truth-assisted control is scored as an accepted result.

| Cell | J forward | C forward | E forward | Matched from zero | Independent from truth | Original gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gaussian-v1--lamino-64-180 | 0.00312285 | 5.14895e-05 | 0.00386014 | 0.151824 | 0.00804714 | 0.10 |
| structured-noisy-v1--lamino-64-180 | 0.028208 | 0.0289345 | 0.0281032 | 0.168414 | 0.226508 | 0.32 |
| structured-v1--lamino-64-180 | 0.0263883 | 0.027163 | 0.0262762 | 0.168414 | 0.206941 | 0.30 |
| gaussian-v1--lamino-128-180 | 0.000785385 | 1.45447e-05 | 0.000970505 | 0.154212 | 0.00207286 | 0.10 |
| structured-noisy-v1--lamino-128-180 | 0.0164974 | 0.016664 | 0.016473 | 0.172832 | 0.197523 | 0.32 |
| structured-v1--lamino-128-180 | 0.0131267 | 0.0133367 | 0.0130956 | 0.172832 | 0.136622 | 0.30 |
| gaussian-v1--lamino-256-180 | 0.000196792 | 1.42081e-05 | 0.000243144 | 0.157391 | 0.000712356 | 0.10 |
| structured-noisy-v1--lamino-256-180 | 0.0119783 | 0.0120562 | 0.0119687 | 0.177296 | 0.18761 | 0.32 |
| structured-v1--lamino-256-180 | 0.00658731 | 0.00672933 | 0.00656931 | 0.177326 | 0.0768566 | 0.30 |
| gaussian-v1--anisotropic-64-180 | 0.0117398 | 0.00143477 | 0.011954 | 0.00447891 | 0.015379 | 0.03 |
| structured-noisy-v1--anisotropic-64-180 | 0.0416629 | 0.0403339 | 0.0416548 | 0.037324 | 0.314365 | 0.18 |
| structured-v1--anisotropic-64-180 | 0.040464 | 0.0390852 | 0.0404563 | 0.037324 | 0.305526 | 0.15 |
| gaussian-v1--anisotropic-128-180 | 0.0029542 | 0.000149658 | 0.00300669 | 0.00736115 | 0.00374236 | 0.03 |
| structured-noisy-v1--anisotropic-128-180 | 0.0229069 | 0.021823 | 0.0228874 | 0.041169 | 0.237073 | 0.18 |
| structured-v1--anisotropic-128-180 | 0.0206115 | 0.0193975 | 0.0205895 | 0.041169 | 0.204811 | 0.15 |
| gaussian-v1--anisotropic-256-180 | 0.000740108 | 2.01127e-05 | 0.000753142 | 0.00969744 | 0.000926021 | 0.03 |
| structured-noisy-v1--anisotropic-256-180 | 0.0141422 | 0.0138903 | 0.0141355 | 0.0519942 | 0.202551 | 0.18 |
| structured-v1--anisotropic-256-180 | 0.00999331 | 0.00963315 | 0.00998385 | 0.0519942 | 0.128307 | 0.15 |
| gaussian-v1--parallel-64-180 | 0.0019287 | 4.00903e-05 | 0.00232717 | 0.00395638 | 0.00374574 | 0.03 |
| structured-noisy-v1--parallel-64-180 | 0.0247308 | 0.0253501 | 0.0246447 | 0.0347243 | 0.187945 | 0.18 |
| structured-v1--parallel-64-180 | 0.0226191 | 0.0232937 | 0.022529 | 0.0347243 | 0.15944 | 0.15 |
| gaussian-v1--parallel-128-180 | 0.000487304 | 1.42197e-05 | 0.000587012 | 0.0047127 | 0.000963181 | 0.03 |
| structured-noisy-v1--parallel-128-180 | 0.0151265 | 0.0152993 | 0.015091 | 0.0391648 | 0.210376 | 0.18 |
| structured-v1--parallel-128-180 | 0.0113436 | 0.0115721 | 0.0112961 | 0.0391648 | 0.114122 | 0.15 |
| gaussian-v1--parallel-256-180 | 0.00012285 | 1.40221e-05 | 0.000147663 | 0.00836299 | 0.000322685 | 0.03 |
| structured-noisy-v1--parallel-256-180 | 0.011568 | 0.0116291 | 0.0115527 | 0.0530954 | 0.209985 | 0.18 |
| structured-v1--parallel-256-180 | 0.00580408 | 0.00592454 | 0.00577344 | 0.0531723 | 0.0631493 | 0.15 |

## Diagnostic time and memory

Times below are seconds. Fresh total is the entire isolated diagnostic process:
imports, transfers, three projector checks, the FP64 oracle, both first and warm
64-iteration controls, and verification. Each control's first/warm columns time
its public solve, host output, image score and projected residual verification.
The first control call occurs after the projector checks; it is not independent
fresh-process time to an accepted result. Peak memory covers all diagnostic
models and controls in the worker and must not replace production-workflow memory.
Persistent-cache policy is inherited; no uncached-compilation claim is made.

The existing launcher samples process GPU memory at requested 10 ms intervals;
brief peaks can be missed. GPU workers run serially. Small CPU algebra and test
checks overlapped some calls, so these are diagnostic costs, not benchmark wins.
All initial and warm quality values and termination reasons are in the archive.

| Cell | Fresh total s | Matched first / warm s | Truth-start first / warm s | Process peak MiB |
| --- | ---: | ---: | ---: | ---: |
| gaussian-v1--lamino-64-180 | 3.800 | 0.943 / 0.162 | 0.664 / 0.160 | 212 |
| structured-noisy-v1--lamino-64-180 | 3.860 | 0.933 / 0.162 | 0.675 / 0.160 | 212 |
| structured-v1--lamino-64-180 | 3.866 | 0.939 / 0.162 | 0.675 / 0.159 | 212 |
| gaussian-v1--lamino-128-180 | 8.679 | 2.047 / 1.251 | 1.736 / 1.206 | 416 |
| structured-noisy-v1--lamino-128-180 | 8.705 | 2.044 / 1.262 | 1.741 / 1.207 | 416 |
| structured-v1--lamino-128-180 | 8.742 | 2.068 / 1.262 | 1.742 / 1.208 | 416 |
| gaussian-v1--lamino-256-180 | 50.044 | 11.729 / 10.924 | 10.844 / 10.317 | 1168 |
| structured-noisy-v1--lamino-256-180 | 50.334 | 11.773 / 10.974 | 10.932 / 10.360 | 1168 |
| structured-v1--lamino-256-180 | 50.538 | 11.855 / 11.029 | 10.992 / 10.416 | 1168 |
| gaussian-v1--anisotropic-64-180 | 3.859 | 1.002 / 0.082 | 0.666 / 0.079 | 208 |
| structured-noisy-v1--anisotropic-64-180 | 3.864 | 0.992 / 0.081 | 0.675 / 0.076 | 208 |
| structured-v1--anisotropic-64-180 | 3.837 | 0.970 / 0.079 | 0.653 / 0.076 | 208 |
| gaussian-v1--anisotropic-128-180 | 5.858 | 1.439 / 0.549 | 1.102 / 0.524 | 284 |
| structured-noisy-v1--anisotropic-128-180 | 5.765 | 1.407 / 0.533 | 1.089 / 0.513 | 284 |
| structured-v1--anisotropic-128-180 | 5.759 | 1.395 / 0.534 | 1.073 / 0.513 | 284 |
| gaussian-v1--anisotropic-256-180 | 22.092 | 5.142 / 4.271 | 4.763 / 4.194 | 656 |
| structured-noisy-v1--anisotropic-256-180 | 21.992 | 5.154 / 4.213 | 4.703 / 4.133 | 656 |
| structured-v1--anisotropic-256-180 | 22.021 | 5.117 / 4.243 | 4.691 / 4.203 | 656 |
| gaussian-v1--parallel-64-180 | 4.054 | 0.998 / 0.136 | 0.719 / 0.125 | 212 |
| structured-noisy-v1--parallel-64-180 | 3.955 | 0.967 / 0.132 | 0.700 / 0.127 | 212 |
| structured-v1--parallel-64-180 | 4.188 | 1.024 / 0.133 | 0.747 / 0.127 | 212 |
| gaussian-v1--parallel-128-180 | 8.152 | 1.946 / 1.061 | 1.532 / 0.956 | 416 |
| structured-noisy-v1--parallel-128-180 | 8.086 | 1.888 / 1.006 | 1.582 / 0.955 | 416 |
| structured-v1--parallel-128-180 | 7.896 | 1.848 / 1.000 | 1.535 / 0.943 | 416 |
| gaussian-v1--parallel-256-180 | 43.032 | 10.157 / 9.322 | 9.097 / 8.555 | 1168 |
| structured-noisy-v1--parallel-256-180 | 42.952 | 9.869 / 9.106 | 9.281 / 8.782 | 1168 |
| structured-v1--parallel-256-180 | 43.130 | 9.928 / 9.252 | 9.282 / 8.771 | 1168 |

## Alignment forward-model checks

These evaluate the true volume at the true irregular poses. They do not time
the alignment path or estimate poses. Clean exact-model residuals measure FP32
arithmetic/data rounding; noisy residuals also contain the fixed 0.1%-RMS noise.
Exact first/warm timings include projection and its residual verification.
Fresh total includes all three model checks and the independent oracle.

| Cell | J forward L2 | C forward L2 | E forward L2 | Fresh total s | Exact first / warm s | Process peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lamino-clean | 0.00502573 | 0.0191156 | 3.42485e-07 | 1.801 | 0.243 / 0.002 | 208 |
| lamino-noisy | 0.0051267 | 0.0191459 | 0.000997701 | 1.800 | 0.247 / 0.003 | 208 |
| anisotropic-clean | 0.00350141 | 0.0271604 | 4.75281e-07 | 1.796 | 0.243 / 0.002 | 208 |
| anisotropic-noisy | 0.00364637 | 0.0271817 | 0.00100316 | 1.809 | 0.246 / 0.001 | 208 |
| parallel-clean | 0.00278403 | 0.0184459 | 3.08384e-07 | 1.859 | 0.242 / 0.002 | 208 |
| parallel-noisy | 0.00295921 | 0.0184764 | 0.000997702 | 1.844 | 0.259 / 0.002 | 208 |

## Underdetermined solution-selection check

An independent FP64 matrix integrates every tent basis on an 8×7×6 grid with
five irregular views and shifted detector coordinates. Unequal voxel dimensions
and 30° tilt are included. All 336 image coefficients remain free. The experiment
uses noiseless data from an asymmetric image containing smooth and sharp features.

Let A be that matrix and M the positive inverse of the exact Fourier diagonal of
AᵀA, with the rejected method's relative FP32 floor. There are no random-probe
errors in this check. Dense SVD solves compare A⁺y with L(AL)⁺y, where LLᵀ = M.
The latter minimizes the norm in transformed coordinates. Both fit the same data
to machine precision, but can select different components in null(A).

| Geometry | Rank / 336 | Plain image L2 | Preconditioned image L2 | Added null component / truth norm | Relative data difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| lamino | 314 / 336 | 0.155636 | 0.360564 | 0.262594 | 3.02492e-15 |
| anisotropic | 270 / 336 | 0.102361 | 0.241849 | 0.187921 | 2.56294e-15 |
| parallel | 270 / 336 | 0.127 | 0.390042 | 0.306727 | 2.14331e-15 |

This proves that solution selection must be checked when qualifying an image-space
preconditioner. It does not attribute a numerical fraction of each large frozen
failure to null-space contamination. Nor does it claim every nonzero null
component is wrong: a declared image prior can deliberately select one.

Directions of the form Aᵀ F r remain in range(Aᵀ), including for the small
positive detector filter checked here. That range property does not make filtered
least squares equivalent to the original objective on inconsistent data, and
does not by itself prove a good algorithm. New CPU/CUDA CGLS regression tests
compare underdetermined tilted, shifted, unequal-spacing systems with dense
least squares, including ray and linear/cubic Joseph models and nonzero starts.

## Unchanged external comparison

The following historical values come from the frozen matrix. Each row uses its
cold-fastest accepted ASTRA/TIGRE workflow for both time and memory. Those workflows
were not rerun here. Diagnostic control timings above are not comparable accepted
workflow timings and are not divided by these values. The failed anisotropic
sharp cell still has no accepted external or TomoJAX denominator.

| Cell | External workflow | Cold / warm s | Image L2 | Process peak MiB |
| --- | --- | ---: | ---: | ---: |
| gaussian-v1--lamino-64-180 | astra_cgls | 1.655 / 0.674 | 0.0933664 | 158 |
| structured-noisy-v1--lamino-64-180 | astra_cgls | 0.328 / 0.029 | 0.268988 | 158 |
| structured-v1--lamino-64-180 | astra_cgls | 0.338 / 0.029 | 0.268893 | 158 |
| gaussian-v1--lamino-128-180 | astra_cgls | 6.953 / 3.191 | 0.0944359 | 202 |
| structured-noisy-v1--lamino-128-180 | astra_cgls | 0.492 / 0.099 | 0.319828 | 202 |
| structured-v1--lamino-128-180 | astra_fbp_cgls | 0.633 / 0.107 | 0.275349 | 374 |
| gaussian-v1--lamino-256-180 | astra_cgls | 39.277 / 17.920 | 0.0971072 | 524 |
| structured-noisy-v1--lamino-256-180 | astra_cgls | 2.110 / 0.713 | 0.317849 | 524 |
| structured-v1--lamino-256-180 | astra_cgls | 3.068 / 0.988 | 0.254414 | 524 |
| gaussian-v1--anisotropic-64-180 | astra_cgls | 0.334 / 0.030 | 0.0204787 | 150 |
| structured-noisy-v1--anisotropic-64-180 | astra_fbp_cgls | 0.355 / 0.008 | 0.175221 | 182 |
| structured-v1--anisotropic-64-180 | No accepted workflow | — | — | — |
| gaussian-v1--anisotropic-128-180 | astra_fbp3d_cupy | 0.362 / 0.010 | 0.00964915 | 242 |
| structured-noisy-v1--anisotropic-128-180 | astra_fbp3d_cupy | 0.363 / 0.010 | 0.128229 | 242 |
| structured-v1--anisotropic-128-180 | astra_fbp3d_cupy | 0.372 / 0.010 | 0.123795 | 242 |
| gaussian-v1--anisotropic-256-180 | astra_fbp3d_cupy | 0.463 / 0.077 | 0.0136194 | 544 |
| structured-noisy-v1--anisotropic-256-180 | astra_fbp3d_cupy | 0.455 / 0.074 | 0.115666 | 544 |
| structured-v1--anisotropic-256-180 | astra_fbp3d_cupy | 0.465 / 0.076 | 0.0934158 | 544 |
| gaussian-v1--parallel-64-180 | astra_fbp2d | 0.291 / 0.024 | 0.00410799 | 140 |
| structured-noisy-v1--parallel-64-180 | astra_fbp2d | 0.296 / 0.027 | 0.132838 | 140 |
| structured-v1--parallel-64-180 | astra_fbp2d | 0.310 / 0.027 | 0.131569 | 140 |
| gaussian-v1--parallel-128-180 | astra_fbp2d | 0.318 / 0.059 | 0.00104495 | 140 |
| structured-noisy-v1--parallel-128-180 | astra_fbp2d | 0.326 / 0.060 | 0.0993521 | 140 |
| structured-v1--parallel-128-180 | astra_fbp2d | 0.341 / 0.061 | 0.0925228 | 140 |
| gaussian-v1--parallel-256-180 | astra_fbp2d | 0.596 / 0.298 | 0.000285228 | 140 |
| structured-noisy-v1--parallel-256-180 | astra_fbp2d | 0.587 / 0.297 | 0.102397 | 140 |
| structured-v1--parallel-256-180 | astra_fbp2d | 0.607 / 0.301 | 0.0724881 | 140 |

## Evidence and validation

The [complete controls](../bench/reference/model-discrepancy-2026-10-04.json.gz)
retain every result, fixture hash, timing, numerical termination, process-memory
sample count, and the one-off launcher source. The
[small-matrix calculation](../bench/reference/preconditioner-nullspace-2026-10-04.json.gz)
retains its source and independent operator checks. Library/driver source is the
unchanged [pose-elimination snapshot](../bench/reference/public-free-voxel-v1-schur-source.tar.gz),
SHA-256 `5ff8791482a48008c71a01ae8a0a8a6b57e03b6e9b403eb95eb6b00254696415`.
Original fixture generation commands remain in the existing benchmark drivers;
no benchmark framework or production solver variant was added.

The first launcher attempt compared saved FP32 poses against regenerated FP64
poses before the public solver's FP32 cast. All 27 reconstruction checks stopped
at that overly strict setup assertion, before any solve. Those failures and the
original script are retained in `initial_attempt`. Correcting the comparison
to use the same FP32 cast reran all 27 reconstruction cells; the six completed
alignment checks were retained unchanged. No numerical gate was relaxed.

The initial regression-test assertion also used an absolute null-component
tolerance despite the arbitrary data units. It was replaced with a relative
projection norm; the dense-solution comparison remains independently checked.
This changes test scaling, not the solver or the frozen image/pose gates.

Validation: all 22 CPU tests in `tests/test_cgls.py` passed, including the six
new underdetermined cases; all six new CUDA cases passed. Lint and formatting
checks passed for the changed Python files. The documentation check passed
384 local links across 48 documents. Solver arithmetic and defaults are unchanged;
the library change documents initialization semantics, with tests guarding them.
