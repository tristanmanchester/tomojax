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

The [latest comparison](research/system-matrix-2026-10-05-memory.md), after the
memory and kernel changes, has a cold geometric mean of 1.00 (0.44 to 2.81) and
warm 2.78 (0.59 to 20.4) against the fastest accepted ASTRA workflow over the
same 26 cells. The weakest cells remain small laminography scans. The
[morning comparison](research/system-matrix-2026-10-05.md) reran the 27-cell
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

## TV-regularised reconstruction

[`bench/compare_tv.py`](../bench/compare_tv.py) reconstructs a 128³ structured
parallel phantom from 180 views with 3% noise, 50 iterations per library over
each library's own grid of regularisation weights. TomoJAX FISTA-TV reaches a
best relative L2 error of 0.076 in 1.13 s warm. TIGRE's FISTA reaches 0.101 in
9.7 s and its ASD-POCS 0.146 in 5.7 s; larger TIGRE FISTA weights diverge.
Unregularised CGLS is best at 0.138 (10 iterations). TIGRE's timings include
its host transfers. This is one phantom and noise level, not a general ranking.

## Scans larger than device memory

`fbp_host` reconstructed a 1024³ laminography scan (30° tilt, 1024 views of
1024² pixels) from a projection memmap into a volume memmap, 4.3 GB each, in
37 s on the 8 GB laptop GPU, in two x-slabs. Before streaming, laminography
FBP ran out of memory at 768³. Laminography FBP leaves the missing cone empty,
so it is a fast first look or initialiser rather than a converged
reconstruction; see [the reconstruction matrix](research/system-matrix-2026-10-05.md).

FISTA-TV streams large NumPy or memmap projection stacks from host memory one
view batch at a time. On a 512³ laminography scan with 3072 views (3.2 GB of
projections, more than fits beside FISTA's working volumes on the 8 GB GPU),
ten positivity-constrained FISTA-TV iterations peaked at 4.3 GB of GPU memory.
With 768 views, where both fit, streaming took 67.6 s against 65.0 s for
device-resident projections.

CGLS and SPDHG-TV stream the same way. On that 3072-view scan, ten streamed
CGLS iterations took 154 s at 3.2 GB peak, and 192 SPDHG-TV iterations (one
view block each) took 297 s at 4.3 GB, with SPDHG-TV's 3.2 GB dual variable
kept in host memory. Streamed SPDHG-TV is bitwise identical to the
device-resident solver, and streamed CGLS converges to the same solution.

## Iterative solver memory

On a 512³ laminography scan with 768 views of 512² pixels (0.8 GB of
projections) on the 8 GB laptop GPU, with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`:

| Solver | Peak GPU memory before | After |
|---|---|---|
| CGLS | out of memory | 5.6 GB |
| FISTA-TV | 7.2 GB | 4.6 GB |
| SPDHG-TV | out of memory | 4.9 GB |

CGLS keeps about five volumes and three sinograms, FISTA-TV seven volumes and
SPDHG-TV eight volumes plus its dual sinogram; none stores a sinogram-sized
temporary for the data gradient. JAX's default allocator limit is 75% of device
memory, under which this CGLS case still does not fit; the CLI raises the
limit to 90%. Volumes larger than device memory are not yet supported by the
iterative solvers.

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
| `tomojax align --mode pose --ray-integrator exact` (coupled solver), fresh CLI process per cell | 5/6 cells pass; rotation RMSE 0.0005–0.008° (object-frame translations), 0.0000–0.008° (detector frame, now the default), 8–20 s | Scored with the pilot's own gates via the public CLI |
| Same eliminated solve with reconstruction batches sized automatically (now the default) | 5/6 cells pass; warm 1.5–5.5 s, 1.9–2.3× faster than one view per batch, same peak memory | [Batching record](../bench/reference/public-free-voxel-batching-2026-10-05.json.gz) (two warm repeats) |

The pilot's measurements integrate the voxel basis exactly, the same model as
the `exact` integrator, so its near-exact clean recoveries partly reflect an
inverse crime. With the CLI's default Joseph integrator, a different
discretization from the data, every cell fails the 0.01° gate at 0.10–0.26°
on these 32³ objects, while volume errors stay at 0.009–0.085. On analytic data from continuous Gaussian objects at 32³, every
solver tried (coupled exact, coupled sampled, alternating) misses the 0.01°
gate, at 0.09–0.5°; started from the true poses, the coupled solver settles at
the same errors, so they are a model-mismatch floor rather than an
optimisation failure. Image quality matches a true-pose reconstruction.

The noisy anisotropic cell still fails the 0.01° rotation gate on the successful
five-cell variant, at about 0.016°. The local noise analysis below predicts about
0.017° from the measurement noise alone, so this gate is probably unreachable for
that cell at this noise level rather than a solver shortfall. There is no complete successful-recovery baseline for a 20×
claim, and these runs do not establish 99% robustness. The coupled solver is the
default for `tomojax align --mode pose`. A [local FP64 noise analysis](research/pilot-noise-2026-10-04.md) examines sensitivity;
it is not a universal bound for constrained or regularized estimators.

On analytic scans of continuous objects (181 views at 128³, 361 at 256³, 30°
laminography and parallel), fresh CLI processes on the laptop GPU with the
default detector-frame translations:

| Scan | Motion | Time | Rotation RMSE |
|---|---|---:|---:|
| 128³ parallel / laminography | ±0.25°, ±0.5 px | 50 / 43 s | 0.0085 / 0.0026° |
| 128³ parallel / laminography / anisotropic | ±0.5°, ±8 px | 33 / 46 / 34 s | 0.0090 / 0.0029 / 0.022° |
| 256³ laminography, full resolution | ±0.25°, ±0.5 px | 202 s | 0.0030° |
| 256³ laminography, stopped at half resolution | ±0.25°, ±0.5 px | 55 s | 0.0052° |
| 64³ parallel / laminography | ±0.5°, ±15 px (23% of the detector) | 19 / 27 s | 0.031 / 0.013° |

At 64³ the 0.01–0.03° results are the discretisation floor for that size. The
64³ anisotropic scan with ±15 px shifts is not captured (5.0° rotation error)
with either translation frame. On the ±8 px and ±15 px scans, object-frame
translations give the same rotation errors to within 0.003°. An earlier record of 0.0076°
and 0.016° for the ±8 px parallel and anisotropic scans could not be
reproduced: the code at that commit now gives 0.0087° and 0.022°.

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
