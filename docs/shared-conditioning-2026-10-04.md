# Shared volume conditioning: diagnostic evidence, 2026-10-04

A positive Fourier preconditioner improves longer linear solves in all three
geometries, but its short-step recovery results are mixed. The subsequent
27-cell reconstruction sweep exposed broad quality regressions. The candidate
has been withdrawn from the working library. The [complete comparison](system-matrix-spectral-2026-10-04.md)
retains all 54 method/cell records and seven warm calls per record, including
failures: 9/54 passed, compared with 49/54 for the original two workflows.
The immutable source is archived. No whole-matrix accepted-result speedup is
established. The noisy anisotropic alignment failure remains unresolved.

## Construction and independent checks

For independent zero-mean data-space probes `z` with covariance identity,
`E[|F A.T z|²] = diag(F A.T A F*)`, with a unitary Fourier transform `F`.
Eight deterministic Rademacher probes estimate this positive normal diagonal.
The calculation streams one backprojected volume at a time and stores a real
half-spectrum. A relative FP32 floor bounds its inverse. Geometry and fixed
weights determine the estimate; measured intensities and truth do not tune it.
The true matched forward/adjoint remain in the solved system.

A single-centre impulse approximation was rejected: its Fourier spectrum can
be negative, and flooring it still worsened the independent parallel and tilted
condition numbers. The averaged approximation and its eight-probe estimate
improved both the reconstruction and pose-eliminated systems across all three
geometries and three probe seeds in the [small dense screen](../bench/reference/shared-conditioning-dense-2026-10-04.json.gz).
Those tiny problems qualify the implementation; they are not speed evidence.

The archived experimental CGLS integration keeps the original least-squares objective,
physical edge regularizer and **unpreconditioned** stopping residual. Residual
replacement recomputes the preconditioned direction before restarting. The
Fourier diagonal of the free-boundary quadratic edge term includes `N-1`
edges per axis, rather than silently imposing a periodic regularizer.

## Fixed-state pilot and a correction to the first screening decision

The [raw pilot diagnostics](../bench/reference/shared-conditioning-pilot-2026-10-04.json.gz)
compare all six saved public terminal states, four solvers and caps of 40, 160
and 640 iterations. Both damping values remain 1e-3; columns use the same
physical central stencil. The positivity active set and linear system are
identical across methods. No public workflow timing is inferred from this test.

The initial [linear-residual screen](alignment-accuracy-2026-10-04.md) was too
strong in rejecting pose elimination. A larger Euclidean normal residual can
coexist with better nonlinear progress. For example, the 40-step plain
pose-eliminated proposal reduced clean laminography rotation from 0.00977°
to 0.00419°, while lowering its actual constrained data loss. Its larger
normal residual did not predict that improvement. Pose elimination remains
eligible for an end-to-end test; residual size alone cannot decide it.

Conversely, the spectral 40-step proposal can reduce loss yet worsen pose
accuracy: parallel-clean moves from 0.00608° to 0.01270° with stacked CG and
0.01329° with pose elimination. This must not be hidden by an iteration-count
or loss-only claim. These are additional diagnostic steps on saved states,
not new recovery runs or a change to the acceptance stopping policy.

| Cell | Plain stacked residual, cap 640 | Spectral stacked residual, cap 640 | Spectral pose-eliminated residual, cap 640 |
|---|---:|---:|---:|
| parallel-clean | 9.27e-5 | 9.55e-5 | 7.94e-5 |
| parallel-noisy | 0.0005955 | 8.25e-05 | 8.165e-05 |
| anisotropic-clean | 0.001617 | 0.0001007 | 9.363e-05 |
| anisotropic-noisy | 0.004331 | 0.0001086 | 5.914e-05 |
| lamino-clean | 0.3483 | 0.02254 | 0.01655 |
| lamino-noisy | 0.4293 | 0.02515 | 0.02744 |

All high-budget proposals at the noisy anisotropic state leave rotation near
0.01617°, compared with 0.01624° before the step. Their nonlinear loss differs
by only about 2e-8 from 0.01330777. This argues against attributing that plateau
solely to the 40-iteration cap. It is not a proof of an irreducible noise floor,
a global optimum, or pilot identifiability. Statistically appropriate volume
regularization and reduced-Hessian/noise diagnostics remain open; gates stay fixed.

## Validation and completed workflow comparison

614 CPU tests passed, 13 skipped and 210 deselected. The CUDA numerical selection
passed 28 tests, covering dense physical solutions, ray and linear/cubic Joseph
models, zero and nonzero starts, damping, physical smoothness, and the Fourier
estimator. Benchmark-driver tests passed 36 cases. Lint, formatting, configured
type checks and all three import contracts passed.

The completed comparison applies the same eight-probe algorithm to ordinary and
multiresolution Joseph CGLS, with their existing budgets and all 27 cells:
`{gaussian-v1, structured-v1, structured-noisy-v1}` ×
`{parallel, shifted odd anisotropic, 30-degree laminography}` ×
`{64,128,256}`, at 180 views. Tilted and anisotropic cells run first.
The [report](system-matrix-spectral-2026-10-04.md) includes every failure, cold
search time, seven warm repetitions, quality values and sampled process memory.
Setup and FFT costs count. It compares against the existing frozen external
workflows, using the same cold-fastest accepted workflow for each cell's time
and memory denominator. No passing-only aggregate establishes the whole-system
goal. The variant is stopped without per-cell tuning.
