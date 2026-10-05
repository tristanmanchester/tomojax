# Read the accuracy and performance evidence

TomoJAX has useful validated paths, but the complete comparison does not yet
support a general speed or memory advantage over ASTRA and TIGRE. Measurements
below use synthetic data on one RTX 4070 Laptop GPU. They do not establish
performance on other accelerators or robustness across real acquisitions.

The [Huber-TV derivative report](research/huber-derivatives-2026-10-04.md) records a
correctness fix with checks on all 33 frozen image inputs. It also retains
failed pose finite-difference controls. These diagnostics do not replace the
complete-workflow timings or establish the gradient cost target.

## Reconstruction

The [latest comparison](research/system-matrix-2026-10-05.md) reruns the 27-cell
matrix: smooth, sharp and noisy objects with parallel, shifted anisotropic and
30° tilted geometry at 64, 128 and 256 nominal grid sizes. Each cell pairs
TomoJAX's fastest accepted workflow with the fastest accepted ASTRA or TIGRE
workflow, reporting quality, failures, fresh-process and warm time and sampled
peak process GPU memory.

26 cells have accepted results on both sides; the sharp anisotropic 64 cell has
none on either side, so the 27-cell aggregate is undefined. Over the 26 pairs,
TomoJAX's cold speedup has a geometric mean of 0.95 (0.40 to 2.49) and its warm
speedup 2.62 (0.75 to 20.5). The weakest cells are small laminography scans,
where JAX start-up and the per-iteration backprojection cost dominate. The
[earlier matrix](research/system-matrix-2026-10-04.md) (cold geometric mean 0.57,
worst 0.21) and its [cold-time profile](research/system-profile-2026-10-04.md)
remain for comparison.

A [spectral conditioning experiment](research/system-matrix-spectral-2026-10-04.md)
regressed broad coverage and was rejected. Its results remain available so that
later work does not silently reuse a failed variant.

The [33-cell model diagnostic](research/model-discrepancy-2026-10-04.md) separates
finite-budget recovery from image-model discrepancy. Independent small systems
also show why a preconditioner can fit the data while changing unmeasured image
features. These truth-assisted controls qualify future solver experiments;
they do not replace the independent acceptance results or their timings.

The subsequent [filtered Krylov/GCV prototype](research/hybrid-krylov-qualification-2026-10-04.md)
failed its small-problem quality and storage qualification. Its algebra is
checked, but it was not integrated or run as a public workflow. The report
records that distinction and the unperformed comparison cells explicitly.

A later [nonnegative held-out regularization screen](research/nonnegative-cv-qualification-2026-10-04.md)
passes its 24-control dense-estimator comparison, but its matrix-free solver
fails two numerical conformance checks. It is stopped before public integration;
the record includes all controls, solver work and unperformed workflow cells.

The [solver follow-up](research/solver-qualifications-2026-10-05.md) also rejects the
augmented nonnegative and range-preserving preconditioned variants. It retains
the stationarity-check audit and every fixed-work quality regression; neither
variant reaches the public workflow comparison.

## Joint volume and pose recovery

The [public free-voxel pilot](research/public-free-voxel-pilot.md) has six clean/noisy
parallel, anisotropic, and tilted cells. It uses modest ±0.25°/±0.5-pixel motion,
independent exact projection data, and fixed image, rotation, and shift gates.
It is not a test of the larger ±3°/±10-pixel capture range in the stretch goal.

| Public path | Result | Record |
| --- | --- | --- |
| Original alternation | 0/6 cells pass | [Baseline](research/public-free-voxel-baseline-2026-10-04.md) |
| Exact projector in the same path | 0/6 cells pass | [Exact rerun](research/public-free-voxel-exact-2026-10-04.md) |
| Coupled volume/pose step | 5/6 cells pass | [Joint solve](research/public-free-voxel-joint-2026-10-04.md) |
| Coupled solve with pose elimination | 5/6 cells pass; faster tilted recovery | [Pose elimination](research/public-free-voxel-schur-2026-10-04.md) |
| Same eliminated solve with reusable compiled objectives | 5/6 cells pass; faster warm calls, unchanged cold startup and memory | [Compiled-objective reuse](research/public-free-voxel-reuse-2026-10-04.md) |
| Same eliminated solve with fixed default-weight Huber-TV | 0/6 cells pass; rejected screen | [TV screen](research/public-free-voxel-tv-2026-10-04.md) |
| Default `tomojax align --mode pose` (coupled solver), fresh CLI process per cell | 5/6 cells pass; rotation RMSE 0.0006–0.008°, 8–18 s | Scored with the pilot's own gates via the public CLI |
| Same eliminated solve with reconstruction batches sized automatically (now the default) | 5/6 cells pass; warm 1.5–5.5 s, 1.9–2.3× faster than one view per batch, same peak memory | [Batching record](../bench/reference/public-free-voxel-batching-2026-10-05.json.gz) (two warm repeats) |

The pilot's measurements integrate the voxel basis exactly, the same model as
the `exact` integrator, so its near-exact clean recoveries partly reflect an
inverse crime. On analytic data from continuous Gaussian objects at 32³, every
solver tried (coupled exact, coupled sampled, alternating) misses the 0.01°
gate, at 0.09–0.5°; started from the true poses, the coupled solver settles at
the same errors, so they are a model-mismatch floor rather than an
optimisation failure. Image quality matches a true-pose reconstruction.

The noisy anisotropic cell still fails the 0.01° rotation gate on the successful
five-cell variant, at about 0.016°. The local noise analysis below predicts about
0.017° from the measurement noise alone, so this gate is probably unreachable for
that cell at this noise level rather than a solver shortfall. There is no complete successful-recovery baseline for a 20×
claim, and these runs do not establish 99% robustness. The coupled solver remains
opt-in. A [local FP64 noise analysis](research/pilot-noise-2026-10-04.md) examines sensitivity;
it is not a universal bound for constrained or regularized estimators.

## Repeated-use startup

The [complete cache screen](research/system-cache-2026-10-04.md) tests all 27 reconstruction
and six alignment cells. Reusing compiled JAX programs reduces some startup and
memory costs. Fourier paths do not benefit, both existing quality failures
remain, and changed input data still triggers some recompilation. This is a
single-repeat diagnostic, not a replacement for the frozen cold comparison.

## Reproduce and interpret results

Use the [benchmark guide](../bench/README.md) for commands, adapters, physical
conventions, and retained source snapshots. [Raw records](../bench/reference)
include failed attempts. Cold timings include fresh-process costs; warm timings
must not be substituted for first use. Sampled PID GPU memory can miss very short
peaks and includes more than array storage.

The [historical performance report](performance.md) retains earlier kernel,
workflow, and showcase measurements. Some earlier external cold timings included
unrelated JAX imports; those tables are explicitly qualified in that report.
Use the corrected frozen matrix for whole-system comparisons.

The README's [synthetic figure](../images/README.md#reproducible-synthetic-reconstruction)
is a matched-model API example. The historical real-data images are qualitative
illustrations. Neither substitutes for the independent acceptance fixtures.
