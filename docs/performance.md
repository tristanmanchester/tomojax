# Numerical accuracy and CUDA performance

This is the historical measurement archive. Start with the
[measurement guide](measurements.md) for current whole-matrix coverage,
failed cases, and the distinction between user examples and independent evidence.

The [corrected reconstruction matrix](research/system-matrix-2026-10-04.md) has 26/27
accepted TomoJAX/external pairs. The [pose-eliminated public alignment run](research/public-free-voxel-schur-2026-10-04.md)
passes five of six modest-motion cells; noisy anisotropic recovery still fails.
The optimization goal remains open. Kernel, Gaussian showcase, and restricted
object-model timings below do not substitute for those full-workflow gates.

Earlier external cold comparisons imported JAX/Pallas while loading fixtures in
ASTRA/TIGRE workers. Those ratios are unsuitable for judging cold performance.
The corrected protocol isolates solver imports and retains failed budgets.
Historical tables below keep their original source snapshots and must not be
combined with the corrected protocol.

## Test environment and scope

Measured 2026-10-02–04 on an NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB, driver
595.58.03; Linux, Python 3.12.13, JAX/jaxlib 0.11.2, NumPy 2.4.5, ASTRA 2.5.0,
and CERN TIGRE 3.1.3 from revision
`6b0951a8a88aa88a7db5b2a95181a33a95aa8b28`.

Measurements used the working tree based on TomoJAX revision `214365a`, including
the corrections described here. They are not measurements of that clean commit.
Kernel times below are medians of seven synchronized warm calls; the workflow
section specifies its separate sampling protocol. Raw samples,
cold calls, errors, and environment metadata are checked in under
[`bench/reference`](../bench/reference). Full reproduction commands and geometry
conventions are in [the benchmark guide](../bench/README.md).

Cold means a fresh worker process, including imports and fixture loading; OS
file caches and driver caches are not flushed. Thus cold measurements also
reflect normal cache and first-call variation, not a fully cold machine.

The [metrics and benchmark case inventory](research/metrics-and-benchmark-cases.md) covers
the broader measurement space. The workflow measurements below cover an initial
subset of that inventory.

## Forward projection

Device-resident input and output, 60 views, milliseconds; lower is better:

| Geometry | Volume | TomoJAX Pallas | ASTRA CUDA3D | TomoJAX / ASTRA |
|---|---|---:|---:|---:|
| Parallel | 128 × 128 × 128 | 1.46 | 2.83 | 0.52 |
| Laminography, 30° tilt | 128 × 128 × 128 | 4.86 | 2.58 | 1.88 |
| Anisotropic, shifted odd detector | 128 × 125 × 64 | 1.69 | 2.57 | 0.66 |
| Parallel | 256 × 256 × 256 | 17.24 | 9.93 | 1.74 |
| Laminography, 30° tilt | 256 × 256 × 256 | 49.26 | 12.25 | 4.02 |
| Anisotropic, shifted odd detector | 256 × 253 × 128 | 11.43 | 6.55 | 1.74 |

All cases use the same physical phantom and detector for each library. TomoJAX
uses FP32 interpolation with `(16, 4)` tiles and one warp. ASTRA and TomoJAX have
different integration models; neither result is used as the other's ground truth.
For the 128-size cases, TomoJAX's relative L2 error against analytic line integrals
is 0.058%, 0.096%, and 0.301%, respectively; ASTRA's is 0.048%, 0.078%, and 0.295%.
Across all 12 cases (sizes 32–256), Pallas agrees with the JAX reference within
`1.7e-7` relative L2 error. This checks backend agreement, not physical accuracy.

TIGRE's Python interface in this comparison uses host arrays, so its fair timing
comparison includes upload and download for every library:

| Parallel volume, 60 views | TomoJAX Pallas | ASTRA CUDA3D | TIGRE interpolated | TIGRE Siddon |
|---|---:|---:|---:|---:|
| 128³ | 3.73 ms | 4.75 ms | 3.94 ms | 10.55 ms |
| 256³ | 29.00 ms | 18.77 ms | 21.99 ms | 68.42 ms |

The 256³ analytic errors are 0.0147%, 0.0121%, 0.0153%, and 0.0085%, respectively.
The TIGRE adapter does not yet compare tilted or anisotropic scans. The complete
record includes smaller cases and timing variability; these tables retain both
favourable and unfavourable larger cases.

The odd-detector fix is particularly consequential. Requiring tiles to divide
the detector dimensions had reduced some programs to one ray. Masking loads and
stores at detector edges preserves useful tile sizes. In the 128 × 125 × 64
case, the before/after experiment improved from 31.43 ms to 2.46 ms with unchanged
numerical results. Later measurements are shown above; clock variation affects
the exact speedup. The before/after records are retained in
[`correction-comparisons.json`](../bench/reference/correction-comparisons.json).

## Parallel filtered backprojection

An analytic Gaussian, 180 views, detector width `ceil(sqrt(2) * size)`, all
projections already on the GPU:

| Volume | JAX filter + BP | Pallas filter + BP | Speedup | Reconstruction relative L2 error |
|---|---:|---:|---:|---:|
| 32³ | 2.20 ms | 0.58 ms | 3.8× | 0.689% |
| 64³ | 2.90 ms | 1.36 ms | 2.1× | 0.189% |
| 128³ | 14.39 ms | 5.63 ms | 2.6× | 0.0454% |
| 256³ | 148.58 ms | 23.70 ms | 6.3× | 0.0116% |

The two backends produced identical unscaled arrays in these aligned-row cases.
The public `fbp()` call at 256³ took 150.72 ms with JAX and 27.10 ms with Pallas
(5.6× faster), including geometry preparation and angular scaling. This is a
comparison of TomoJAX implementations; it is not an ASTRA/TIGRE FBP speed claim.

The kernel assigns output voxels to GPU lanes, accumulates all views in registers,
and stores the result once. A sinogram transpose makes adjacent axial voxels read
adjacent detector rows. Aligned rows need two detector samples per view;
fractional row coordinates use bilinear interpolation. Partial output blocks are
masked. This replaces the older tiled FBP kernel.

```python
from tomojax.recon import FBPConfig, fbp

# CUDA + built-in ParallelGeometry selects the Pallas kernel automatically.
volume = fbp(geometry, grid, detector, projections)
# Force a backend for validation or reproducible experiments.
reference = fbp(geometry, grid, detector, projections,
                config=FBPConfig(backprojector="jax"))
```

`backprojector="pallas"` requires CUDA inputs and built-in parallel geometry with
no explicit detector grid. It raises on unsupported inputs. Custom geometry and
explicit detector grids use the JAX discrete-adjoint path. The default angular
scale assumes uniformly sampled half-turn data; other angular coverage needs
appropriate weighting. A ramp-filtered adjoint for laminography is an approximate
initializer, not a complete inverse of missing-angle data.

## Complete iterative reconstruction and alignment calls

The second optimization pass measured the **public APIs**, including geometry
setup, norm estimation, compilation, solver work, and Python diagnostics. The
baseline was the working tree before this pass, after the earlier projector and
FBP corrections. These are improvements over TomoJAX's previous implementation,
not comparisons with ASTRA or TIGRE.

Six cases use parallel, 30-degree laminography, and anisotropic/shifted geometry
at nominal sizes 32 and 64, with 12 views. Each run starts a fresh reconstruction
from zero. FISTA uses 10 Huber-TV iterations with supplied L; SPDHG uses 20
iterations with automatic norm estimation. Alignment uses three outer iterations,
four reconstruction iterations per outer, and perturbed initial poses. All use
FP32 and four-view batches. Times are medians of **two repeated full calls** after
one cold call; cold timings and all diagnostics are retained in
[the workflow record](../bench/reference/workflows-rtx4070-laptop.json.gz).

| Public workflow | Before, milliseconds | After, milliseconds | Speedup across valid cases |
|---|---:|---:|---:|
| FISTA | 469–553 | 32–97 | 5.6–14.7× |
| SPDHG | 614–791 | 55–192 | 4.1–11.5× |
| Gauss–Newton joint reconstruction/alignment | 2169–2459 | 865–1009 | 2.4–2.6× |

The anisotropic size-64 alignment failed in the old version and is excluded from
the speedup range. It now completes in about 899 ms, with volume NRMSE 0.346. The
failure came from rounding an all-finite float32 mean below one, not from a
nonfinite reconstruction. On cases that completed before and after, volume NRMSE
changed by less than `6e-5` absolute for alignment and `6e-8` for reconstruction.
These short iteration budgets are performance and numerical regression cases;
they do not establish time to a publication-quality reconstruction. Alignment's
analytic zero-pose truth is approximate for the discretized projector.

At nominal size 128 with 60 views, compute costs become more prominent. In
[the larger cases](../bench/reference/workflows-large-rtx4070-laptop.json.gz), FISTA
improved from 1.50–2.35 s to 0.98–1.84 s (1.28–1.54×), and SPDHG from
2.32–3.78 s to 1.67–3.10 s (1.22–1.39×). These use the same iteration budgets,
not equivalent FISTA/SPDHG convergence targets. Their volume NRMSE values are
unchanged at the reported precision.

A separate [alignment optimizer comparison](../bench/reference/workflows-optimizers-rtx4070-laptop.json.gz)
uses size 32 and the three geometries. Its baseline already includes the gauge
and solver improvements above. Preparing and reusing the L-BFGS objective and
optimizer reduced complete three-outer calls from 43.1–44.0 s to 6.46–6.63 s
(about 6.6×). Each outer allows five optimizer iterations. Later outer alignment
steps take about 0.12 s in the parallel case; first-outer compilation still dominates
the complete call. GD timing was essentially unchanged by this L-BFGS-specific
refactor. FP32 optimizer trajectories are not bitwise reproducible: parallel-case
volume NRMSE ranged from 0.289–0.292 before and 0.290–0.296 after. These are
performance experiments with quality diagnostics, not evidence of improved
alignment accuracy or a universal time-to-quality win.

The changes address general execution costs:

- Public solver loops and their shared power iteration now reuse compiled code,
  with current data, geometry, support, and step sizes passed as array inputs.
- L-BFGS keeps compiled evaluation and line-search updates for the life of one
  alignment problem. Changing volumes and random keys are dynamic inputs; state
  and convergence checks reset for each outer solve.
- Gauge projection no longer recompiles its constraint loop at each invocation.
  Warm alignment substeps in the initial size-32 laminography investigation fell
  from about 360 ms to 26–32 ms; complete-call improvements are smaller because
  objective setup and compilation remain.
- FP32 adjoints accumulate all rays in a batch into one volume. An 18-case sweep
  (sizes 32/64/128, batches 4/16, all three geometries) reduced compiler-reported
  temporary bytes by 2.9–9.3×. Speed ratios ranged from 0.85× to 1.57×; several small
  cases were slower. Larger cases generally improved. Maximum relative difference
  was below `3e-7`. This is temporary-buffer analysis, not measured peak process
  VRAM; see [all samples](../bench/reference/adjoint-stack-rtx4070-laptop.json.gz).
- Implicit alignment solves the reconstruction once and reuses its diagnostics.
  Its measurement-data gradient now uses the implicit adjoint instead of returning
  zero; a dense damped-normal-equation regression checks the result.
- Exact detector-size overrides for 64×64, 96×96, and 80×64 were removed. An explicit
  JAX request stays JAX regardless of problem dimensions. Pallas remains available
  when requested and supported.

## Fixed-quality solver comparisons (2026-10-03)

The first `gaussian-v1` slice measures 180-view scans at nominal size 64.
Acceptance is full-volume relative L2 error at most 3% for parallel/anisotropic
scans and 10% for 30-degree laminography. Independent analytic data and gates
were fixed in [the optimization goal](research/optimization-goal.md) before solver tuning.
These smooth, noise-free phantoms are an initial measurement slice.

Each method starts at zero with no regularization or positivity constraint.
Fixed budgets double from 1 through 256 iterations; each budget is one complete
solve. The first accepted budget is repeated seven times in an isolated worker.
Medians below include the public call, geometry, transfers, layout conversion,
diagnostics, and full-volume verification. They exclude startup and the preceding
budget search, which are reported separately below.

| Geometry | TomoJAX FISTA | TomoJAX CGLS, JAX | TomoJAX CGLS, Pallas | ASTRA CGLS | ASTRA SIRT | TIGRE CGLS |
|---|---:|---:|---:|---:|---:|---:|
| Parallel | 2624 ms (16) | 484 ms (8) | 59 ms (8) | 26 ms (8) | 67 ms (32) | 171 ms (8) |
| Anisotropic/shifted | 2420 ms (32) | 542 ms (16) | 105 ms (16) | 30 ms (16) | 94 ms (64) | Uncovered |
| Laminography | Failed target | 13863 ms (256) | 3393 ms (256) | 677 ms (256) | Failed target | Uncovered |

Parentheses give selected iteration budgets. TomoJAX uses batches of 16 views.
Final Pallas CGLS errors are approximately 1.06%, 2.38%, and 8.62%, respectively.
FISTA's laminography error remains 11.73% at 256 iterations. The TIGRE adapter
does not yet validate tilted/anisotropic geometries; this is uncovered comparison
coverage, not a statement that TIGRE cannot reconstruct those scans.

CGLS reaches the parallel and anisotropic targets about 44x and 23x faster than
TomoJAX FISTA in this slice. ASTRA CGLS remains about 2.2x, 3.5x and 5x faster
than the first TomoJAX Pallas CGLS implementation. Different discretizations and
different achieved errors still matter; the records preserve every failed budget
and the actual errors.

Fresh-process startup plus the entire budget search takes TomoJAX Pallas CGLS
1.56 / 1.68 / 8.98 seconds, versus ASTRA CGLS 0.62 / 0.62 / 1.95 seconds.
Sampled peak worker VRAM is 214 / 212 / 214 MiB for TomoJAX, versus
158 / 150 / 158 MiB for ASTRA. These include runtime/allocator memory across
the search and repeated calls. Sampling at a requested 10 ms interval may miss
short-lived peaks. **Neither the speed nor the memory stretch target is met.**

See the [external and FISTA baseline](../bench/reference/reconstruction-quality-baseline-64.json.gz),
[first CGLS results](../bench/reference/reconstruction-quality-cgls-64.json.gz), and
[reproduction guide](../bench/README.md). Larger, structured, noisy and real-data
cases remain necessary. CGLS supports scalar damping and nonzero starting volumes,
but those options were not used in this unregularized comparison.

A follow-up uses the validated roundoff guard, bulk laminography pose preparation,
and all 180 views in one Pallas batch. It keeps the same data, thresholds and
seven-repeat protocol. [All 12 records pass their quality gates](../bench/reference/reconstruction-quality-cgls-64-128.json.gz):

| Geometry / size | TomoJAX CGLS | ASTRA CGLS | TomoJAX / ASTRA |
|---|---:|---:|---:|
| Parallel / 64 | 42.7 ms | 26.2 ms | 1.63 |
| Anisotropic / 64 | 73.5 ms | 30.0 ms | 2.45 |
| Laminography / 64 | 2394 ms | 677 ms | 3.54 |
| Parallel / 128 | 302 ms | 138 ms | 2.19 |
| Anisotropic / 128 | 464 ms | 109 ms | 4.26 |
| Laminography / 128 | 18345 ms | 3213 ms | 5.71 |

These are repeated complete solves at the first accepted fixed budget, including
verification, not cold startup. At size 128, cold budget searches take
2.11 / 2.51 / 38.48 seconds for TomoJAX and 0.93 / 0.84 / 7.40 seconds for ASTRA.
Sampled worker peaks are 270 / 222 / 270 MiB versus 202 / 188 / 202 MiB.
The ratio above is a remaining performance gap, not a TomoJAX speedup. Reproduce
with `bench/compare_reconstructions.py --sizes 64 128 --methods tomojax_cgls_pallas astra_cgls --batch 180 --repeats 7`.

The roundoff regression uses a damped, inconsistent dense least-squares system.
A normal-residual tolerance below attainable FP32 accuracy previously let a
conjugate recurrence leave a correct minimizer and diverge. CGLS now stops when
an update falls below volume precision, reporting `roundoff_limit` if the
requested tolerance has not been reached. Independent dense solutions and
stationarity checks validate the retained volume.

## Coarse-to-fine CGLS and sharp-phantom follow-up

The explicit `cgls_multires` API preserves physical volume bounds and detector
sample coordinates at every level, including odd dimensions and shifted origins.
The comparison policy was frozen before these retained runs: coarsest maximum
dimension 32, seven eighths of the total budget coarse, the rest fine. No
geometry-specific tuning or threshold changes were applied.

[Gaussian results, seven repeats](../bench/reference/reconstruction-quality-multires-64-128.json.gz):

| Geometry / size | Multiresolution warm solve | ASTRA CGLS warm solve | Multiresolution / ASTRA cold budget search |
|---|---:|---:|---:|
| Parallel / 64 | 31.3 ms | 24.6 ms | 3.60 / 0.59 s |
| Anisotropic / 64 | 34.6 ms | 30.3 ms | 3.62 / 0.59 s |
| Laminography / 64 | 653 ms | 674 ms | 5.03 / 1.93 s |
| Parallel / 128 | 98.3 ms | 136 ms | 4.17 / 0.91 s |
| Anisotropic / 128 | 107 ms | 111 ms | 4.21 / 0.86 s |
| Laminography / 128 | 2705 ms | 3199 ms | 9.97 / 7.33 s |

All gates pass, but repeated selected-budget solves and cold searches give
different conclusions. Additional shape compilation makes multiresolution cold
searches slower in every case. Sampled worker peaks are also higher: 284–406 MiB
versus ASTRA's 150–202 MiB. Close warm differences are not established advantages
under interleaved thermal controls. These results satisfy none of the complete
stretch targets and do not include external direct-reconstruction baselines.

The separately frozen sharp-ellipsoid tests show why the Gaussian results cannot
justify a universal policy. Below are size-64 complete warm solve medians; all
accepted budgets were repeated seven times. A failure remains a failure through
256 total iterations, and its best observed error is reported instead of a time.

| Suite / geometry | Direct TomoJAX CGLS | Multiresolution CGLS | ASTRA CGLS |
|---|---:|---:|---:|
| Sharp / parallel | 41.9 ms | 108.7 ms | 26.7 ms |
| Sharp / anisotropic | FAIL, error 0.1924 | FAIL, error 0.1979 | FAIL, error 0.1881 |
| Sharp / laminography | 89.4 ms | 42.1 ms | 28.9 ms |
| Noisy sharp / parallel | 42.7 ms | 39.1 ms | 25.2 ms |
| Noisy sharp / anisotropic | FAIL, error 0.1926 | FAIL, error 0.1987 | FAIL, error 0.1884 |
| Noisy sharp / laminography | 88.9 ms | 40.3 ms | 28.7 ms |

The sharp parallel case needs 64 total multiresolution iterations versus eight
direct iterations. This reverses the Gaussian speed improvement without changing
the policy. The noisy suite has a separately declared, looser gate, so its times
must not be compared with the clean suite as a noise-performance improvement.
ASTRA remains faster in every accepted sharp/noisy comparison. All three methods
miss the anisotropic gates; the discretization and sampling error need further
investigation, without retroactively loosening acceptance.

See [clean records](../bench/reference/reconstruction-structured-64.json.gz),
[noisy records](../bench/reference/reconstruction-structured-noisy-64.json.gz), and
[fixed case definitions](research/optimization-goal.md). Truth-assisted budget selection
is an offline experiment, not a usable stopping policy for unknown scans.
Multiple noise seeds, dose-dependent noise, larger sharp cases and real data
remain uncovered.

## Sampler and Pallas adjoint follow-up

An 18-case interleaved before/after sampler comparison covers sizes
48, 95, 128, 160, 192, 256 and all three geometries, with 60 views per case.
Each case alternates call order over 21 synchronized samples. Factoring
interpolation, removing redundant masked-load clamps, and pairing forward ray
steps yields 1.04–1.56x speedup. At 256 cubed, tilted projection changes from
47.86 to 42.29 ms (1.13x). The anisotropic size-256 case changes from 10.62 to
8.79 ms (1.21x). Maximum before/after relative error is `6.1e-8`.
This compares with the saved pre-sampler TomoJAX working tree, not ASTRA;
see [all paired samples](../bench/reference/sampler-paired-rtx4070-laptop.json.gz).

The Pallas summed adjoint now accumulates directly into one shared volume.
At 128 cubed with 16 views this removes a 128 MiB compiler-reported temporary
volume stack; the output volume is still required. The three size-128, 16-view
cases improve by 1.40–1.68x. Across the full 18-case sweep the ratio is
0.65–1.90x, so smaller cases can regress. Maximum relative error against the
sum of separate adjoints is `2.2e-7`. Compiler temporary bytes are not process
peak VRAM. See [all adjoint samples](../bench/reference/adjoint-pallas-stack-rtx4070-laptop.json.gz).

## Plane sampling with a gather transpose

Controlled profiles now use the same 180-view, batch-180 CGLS operator path for
forward and adjoint calls. The ray adjoint consumes 74–90% of their combined
cost over sizes 64/128/256 and the three geometries. The optional
`CGLSConfig(projector_model="joseph")` samples voxel-centre planes along the
dominant voxel direction. Its exact discrete transpose gathers detector
contributions into voxels without floating-point atomics. The default remains
the existing trilinear ray model. These models have different interpolation and
boundary conventions; this change is not an algebraic acceleration of the same
forward matrix.

Nine alternating synchronized resident samples, 180 views, milliseconds:

| Geometry / size | Ray forward | Ray adjoint | Joseph forward | Joseph adjoint | Pair speedup |
|---|---:|---:|---:|---:|---:|
| Parallel / 128 | 6.83 | 26.23 | 5.81 | 14.77 | 1.61× |
| Anisotropic / 128 | 4.41 | 25.05 | 3.51 | 13.45 | 1.74× |
| Laminography / 128 | 14.02 | 58.27 | 5.91 | 25.83 | 2.28× |
| Parallel / 256 | 48.75 | 262.38 | 44.20 | 109.72 | 2.02× |
| Anisotropic / 256 | 22.94 | 199.83 | 19.36 | 97.09 | 1.91× |
| Laminography / 256 | 123.51 | 522.72 | 41.86 | 195.54 | 2.72× |

Dynamic pose coefficient preparation is included. Joseph's Gaussian projection
error is slightly lower in all nine cases, including 0.0197% versus 0.0243% for
size-256 laminography. This is an internal TomoJAX operator comparison, not an
ASTRA/TIGRE or complete-workflow speed claim. See [the raw operator comparison](../bench/reference/operator-models-rtx4070-laptop.json.gz)
and [the controlled ray profile](../bench/reference/cgls-operator-profile-rtx4070-laptop.json.gz).

The same frozen Gaussian gates and coarse-to-fine policy give the following
host-to-host complete solves. Warm medians include quality verification; cold
search includes process startup, compilation and all smaller rejected budgets.
Seven repeats, 180 views, batch 180:

| Geometry / size | Direct Joseph (ms) | Coarse-to-fine Joseph (ms) | ASTRA CGLS (ms) | Coarse-to-fine / ASTRA cold search (s) |
|---|---:|---:|---:|---:|
| Parallel / 64 | 29.3 | 20.3 | 26.6 | 2.89 / 0.63 |
| Anisotropic / 64 | 40.6 | 21.4 | 30.3 | 3.09 / 0.62 |
| Lamino / 64 | 1014.0 | 291.2 | 677.9 | 3.71 / 1.96 |
| Parallel / 128 | 188.4 | 67.4 | 136.9 | 3.29 / 0.93 |
| Anisotropic / 128 | 262.5 | 68.0 | 112.9 | 3.35 / 0.87 |
| Lamino / 128 | 7702.8 | 1165.9 | 3212.1 | 5.99 / 7.38 |

[All Gaussian runs](../bench/reference/joseph-gaussian-v1-64-128.json.gz) pass, but
coarse-to-fine process peak memory is still 284–406 MiB versus ASTRA's 150–202
MiB. Most cold searches remain slower. These repeated selected-budget results
exclude offline iteration selection and do not establish the fastest applicable
external workflow. The later direct-FBP comparison below addresses this gap
for the supported baseline adapters.

Sharp and noisy fixtures expose different behavior:

| Suite / case | Direct Joseph (ms) | Coarse-to-fine Joseph (ms) | ASTRA CGLS (ms) |
|---|---:|---:|---:|
| Sharp / parallel-64 | 28.7 | 41.0 | 27.1 |
| Sharp / anisotropic-64 | failed gate | failed gate | failed gate |
| Sharp / lamino-64 | 42.0 | 23.9 | 29.4 |
| Sharp / parallel-128 | 186.8 | 139.1 | 137.9 |
| Sharp / anisotropic-128 | 144.2 | 176.7 | 68.8 |
| Sharp / lamino-128 | 284.8 | 91.7 | 150.7 |
| Noisy / parallel-64 | 28.4 | 27.5 | 26.9 |
| Noisy / anisotropic-64 | failed gate | failed gate | failed gate |
| Noisy / lamino-64 | 41.4 | 23.9 | 29.1 |
| Noisy / parallel-128 | 188.3 | 91.6 | 137.5 |
| Noisy / anisotropic-128 | 144.5 | 175.1 | 68.8 |
| Noisy / lamino-128 | 165.4 | 91.4 | 101.7 |

Both anisotropic size-64 suites fail the unchanged gates in every method through
256 iterations. At size 128, coarse-to-fine Joseph improves tilted cases but
regresses on anisotropic sharp/noisy cases. Whole-suite exit status remains
failure. Raw [sharp](../bench/reference/joseph-structured-v1-64-128.json.gz) and
[noisy](../bench/reference/joseph-structured-noisy-v1-64-128.json.gz) records retain
each error, budget, cold search, timing sample and memory observation. No stretch
target is established by this pass.


## External filtered-backprojection baselines

The frozen quality suites now include CuPy 14.2 GPU Ram-Lak filtering plus
ASTRA 2.5 `direct_BP`, native ASTRA `FBP_CUDA` per axial slice, and TIGRE's
public `fbp`. Physical scaling and orientation were validated against an
off-centre analytic Gaussian at two physical length scales. The 3D workflow
uses detector-area/voxel-volume normalization and half-turn angular weighting;
no reference-derived scale is fitted. Native 2D ASTRA and TIGRE adapters are
currently limited to centred isotropic parallel geometry.

All four direct implementations, including TomoJAX, miss the fixed Gaussian
parallel gate: full-volume errors are 3.064% at size 64 and 3.049% at size 128,
against the unchanged 3% limit. The shifted anisotropic errors are 6.95% and
8.67%; filtered tilted ASTRA BP gives about 56%. Direct solves therefore do
not replace the accepted Gaussian iterative baselines.

The sharp parallel cases are different. Complete host-to-host solves with
quality verification, medians of seven repeats, milliseconds:

| Suite / case | TomoJAX FBP | ASTRA + CuPy FBP3D | ASTRA native FBP2D | TIGRE FBP |
|---|---:|---:|---:|---:|
| Sharp / parallel-64 | 4.32 | 4.00 | 23.19 | 9.94 |
| Sharp / parallel-128 | 18.51 | 22.66 | 63.09 | 47.20 |
| Noisy / parallel-64 | 4.19 | 4.05 | 24.35 | 9.78 |
| Noisy / parallel-128 | 17.49 | 23.18 | 63.87 | 48.62 |
| Noisy / anisotropic-128 | 13.89 | 11.24 | unsupported | unsupported |

These accepted results are substantially faster than the earlier CGLS-only
comparisons. They show why the goal must use the fastest applicable workflow,
not just one competing iterative solver. TomoJAX remains close to the GPU-filtered
ASTRA workflow, without the required speed margin. All remaining sharp/noisy
direct cases fail their gates or are unsupported by the adapters; the clean
anisotropic size-128 error is about 15.14%, just above its 15% gate. No limit
was loosened and no failed solve is counted as a speed win.

See the complete [Gaussian](../bench/reference/direct-gaussian-v1-64-128.json.gz),
[sharp](../bench/reference/direct-structured-v1-64-128.json.gz), and
[noisy](../bench/reference/direct-structured-noisy-v1-64-128.json.gz) records for
cold costs, individual samples, exact errors and process memory. Method
`astra_fbp3d_cupy` is an explicitly composed CuPy/ASTRA workflow, not a native
ASTRA FBP3D algorithm. The [benchmark guide](../bench/README.md#external-direct-reconstruction-baselines)
provides reproduction commands and the adapter limits.

## FBP initialization followed by matched iterations

A composed workflow recomputes FBP from the measured data, then performs CGLS
updates from that volume. It uses the existing public `init_x` interface and
adds no automatic solver default. One update passes the fixed Gaussian parallel
gate at sizes 64 and 128; shifted anisotropic cases require eight. Initializer
work is included in every cold and repeated solve.

| Gaussian case | TomoJAX FBP + Joseph CGLS (ms) | CuPy/ASTRA FBP + ASTRA CGLS (ms) |
|---|---:|---:|
| parallel-64 | 13.2 | 10.7 |
| anisotropic-64 | 28.4 | 18.7 |
| lamino-64 | unsupported_comparison | 635.3 |
| parallel-128 | 63.3 | 64.7 |
| anisotropic-128 | 152.8 | 69.5 |
| lamino-128 | unsupported_comparison | 3095.3 |

These are seven-repeat host-to-host times including quality verification.
ASTRA's native `CGLS3D_CUDA` rejects GPU-linked projection objects in the installed
2.5 build, so its initializer handoff necessarily downloads and uploads host
data through that API; those transfers are included. TomoJAX retains its
initializer on the device. Despite this, the anisotropic comparison still
favours ASTRA, while the parallel workflows are close. Tilted TomoJAX FBP
initialization is explicitly unsupported by its current Pallas FBP API.

[Gaussian](../bench/reference/hybrid-gaussian-v1-64-128.json.gz),
[sharp](../bench/reference/hybrid-structured-v1-64-128.json.gz), and
[noisy](../bench/reference/hybrid-structured-noisy-v1-64-128.json.gz) records retain
all outcomes. Direct FBP remains faster where it already passes; the initializer
is useful when a direct result requires refinement. Fixed thresholds and
iteration-selection rules are unchanged, and this is not a stretch-goal win.

## Independent gVXR material data

The optional `gvxr-materials-v1` generator runs gVXR 2.1.0 in a separate environment
using headless NVIDIA EGL. It renders three disjoint water, aluminium and PMMA
cuboids, with shifted odd detectors, translated poses and parallel or 30-degree
tilted scans. This first mesh fixture uses simple shapes so exact physical-ray
intersections can independently validate the integration. It adds material
attenuation, a synthetic five-energy photon spectrum and seeded Poisson noise;
it is not a calibrated scanner or a scatter model.

The two size-64, 180-view stacks agree with analytic material-weighted box chords
to less than 0.004% relative L2 for both mono and poly data. The independent
check uses gVXR's attenuation coefficients, so it checks geometry, units and
spectral summation rather than independently validating those coefficients.
A fivefold raster refinement retains the original detector centres and reduces
triangle interpolation error; it does not average measurements. Repeating the
parallel generator reproduced every stored array and metadata bit for bit.
The fixed relative error limit remains `1e-4`.

At the preselected 32-iteration budget, unregularized reconstruction returns
large errors on these sharp interfaces. Spectral data add model mismatch. Errors
below use the full voxel-centre 80 keV attenuation reference, without fitted
scaling:

| Scan / channel | TomoJAX ray | TomoJAX Joseph | ASTRA CGLS |
|---|---:|---:|---:|
| Parallel / mono | 36.89% | 36.92% | 36.78% |
| Parallel / poly | 42.31% | 42.30% | 42.12% |
| Parallel / noisy poly | 43.32% | 43.28% | 42.91% |
| Tilted / mono | 40.69% | 40.75% | 40.01% |
| Tilted / poly | 44.56% | 44.49% | 43.68% |
| Tilted / noisy poly | 45.43% | 45.44% | 44.23% |

All 18 solves return finite volumes without numerical breakdown. That is a
solver smoke check, not an accepted image-quality result. Poly errors against
an 80 keV image expose model mismatch: for example, parallel-scan aluminium
mean bias is about +20%, versus about +1–2% for mono data. These diagnostic
results neither change the frozen benchmark thresholds nor establish a speed
improvement. Regularization, detector response, spectral correction and finer
sampling need further study. [Raw diagnostics and fixture metadata](../bench/reference/gvxr-reconstruction-64.json.gz),
[reproducibility check](../bench/reference/gvxr-reproducibility.json.gz), and
[generator instructions](../bench/README.md#independent-material-projections-with-gvxr)
are retained. The generated NPZ files stay in ignored `bench/results/`.

## Correctness changes

- FBP now zero-pads detector rows before applying a discrete ramp filter. This
  removes circular wraparound and preserves the finite ramp's DC coefficient.
  Physical normalization no longer changes reconstructed attenuation when the
  same scan is expressed in different length units. Existing FBP amplitudes can
  therefore change; regenerate quantitative reconstructions after upgrading.
- Explicit backprojection follows the same floating-point sample coordinates
  as forward projection. Reversing the coordinate recurrence had introduced
  drift on long oblique rays. In the 128 × 128 × 5 regression, relative error
  against autodiff fell from `1.08e-3` to about `1.35e-7`. The FP32 implementation
  scatters directly into its accumulator, avoiding a dense temporary volume at
  each ray step. One before/after GPU measurement improved from 7.58 to 4.06 ms;
  the retained seven-repeat benchmark measured 4.20 ms after the fix. CPU timing
  did not improve consistently.
- Integer-slice specialization now rejects tiny tilts and checks accumulated
  detector-spacing drift. A 5-microradian tilt through thin slices previously
  caused a 1.9% discrepancy; the generic path matches the reference in this case.
- Pallas explicit unrolling works on the tested Triton lowering. CPU interpretation
  replays scalar atomic additions because the interpreter cannot correctly model
  masked vector atomics with repeated indices. CUDA retains native masked atomics.
- FBP's memory fallback commits a chunk only after device work completes. An
  asynchronous out-of-memory error retries smaller chunks without losing views
  or reusing a failed accumulator.

Regression tests cover direct linear convolution, physical attenuation, length
rescaling, random detector tails and layouts, long-ray adjoints, FP16/BF16 gather
adjoints, weighted loss gradients, and FISTA/SPDHG convergence. CUDA memory checks
cover the real kernels, not only the CPU interpreter.

Validation after the 2026-10-03 CGLS and shared-adjoint changes:

| Check | Result |
|---|---|
| Full CPU suite | 278 passed, 35 CUDA-only tests skipped |
| Full CUDA suite | 313 passed |
| Compute Sanitizer memcheck | 35 GPU tests passed; zero reported errors |
| Analytic benchmark smoke runs | Forward, FBP, and adjoint accuracy gates passed |
| Static checks | Ruff formatting/lint, basedpyright, and import boundaries passed |
| Distribution | Wheel/sdist build, Twine validation, and installed-wheel smoke passed |
| CLI workflows | Synthetic data, reconstruction, validation, and slice export passed on CPU and CUDA |

JAX's Triton deprecation warnings remain visible during CUDA tests.

After the multiresolution geometry, solver and structured-suite changes, the
full suites pass with **328 CPU tests** (37 CUDA cases skipped) and **364 CUDA
tests** (one explicit-coordinate/Pallas combination correctly skipped).
Compute Sanitizer runs the new solver tests with **11 passed, one skipped and
zero reported errors**, including irregular coarse grids and warm starts.
Ruff formatting/lint, configured basedpyright checks and import boundaries pass.
Direct type-checking of the new numerical module has zero errors and 15 warnings
from JAX's loose annotations and ignored validator return values. Wheel/sdist
build, Twine validation and the installed-wheel smoke also pass. A source archive
for the retained benchmark runs is saved locally under `.artifacts/goal/` using
the reported source hash; subsequent changes only make coordinate tuple types
explicit and strengthen rejection of incompatible legacy checkpoints.

The Joseph and gVXR additions pass **344 CPU tests** (45 CUDA tests skipped)
and **388 CUDA tests** (one unsupported explicit-coordinate combination skipped).
Compute Sanitizer reports **8 Joseph GPU tests passed, zero errors**, including
long oblique planes, odd dimensions, changed pose axes and damped tail-batch
solves against an independent FP64 system. Configured type checking reports
zero errors/warnings; Ruff and import boundaries pass. Wheel/sdist build, Twine
validation and installed-wheel smoke pass, including both new Joseph modules.
The final measured source is archived in `.artifacts/joseph/final-measured-source.tar.gz`.
The CUDA backend still
emits JAX's Pallas/Triton deprecation warnings.

## CGLS precision and residual verification

A new regression reproduced a premature stop: an already-correct axial slice
with amplitude `1e7` dominated the old global volume norm, while an independent
slice with values 0.01–0.12 still needed updates. The old guard stopped after
one update with 32.7% relative error in the weak slice. The corrected Joseph
solve reduces that error below `6e-7` in the CPU reproduction; both ray and
Joseph backends now pass the regression on CPU/CUDA without modifying the
bright slice.

The solver uses voxelwise update checks and a componentwise cancellation
estimate, rather than declaring roundoff from a global volume norm alone.
It recomputes `y - A x` and the normal gradient before reporting convergence
or roundoff, replacing a drifted recursive residual and restarting its direction
when necessary. `normal_residual_is_recomputed` and `residual_recomputations`
make the diagnostic status visible. A fixed-budget termination can still carry
a recursive norm, explicitly marked as such. The requested normal-residual
tolerance is unchanged; `roundoff_limit` does not claim convergence.

A 120-case CUDA stress experiment covers 20 signed datasets, three damping
values, two operator backends and data amplitudes from `1e-3` to `1e3`.
Maximum relative volume error against independently assembled FP64 least-squares
solutions is `5.7e-7`; normalized stationarity error stays below `7.6e-7`.
Existing dense accuracy thresholds remain unchanged. These small-system results
do not prove accuracy for every ill-conditioned scan. Additional operator work
near stopping can affect cold cost and peak memory; earlier timing tables are
retained with their source hashes.

After this correction, the full suites pass with **350 CPU tests** (55 CUDA
tests skipped) and **404 CUDA tests** (one unsupported combination skipped).
Compute Sanitizer passes **50 solver/kernel tests**, with one skipped and zero
reported errors, including the residual-verification branches. Ruff, configured
basedpyright, import contracts and the public import guard pass. The lockfile
check, wheel/sdist build, Twine validation and installed-wheel smoke also pass.
The reproducible [dense stability record](../bench/reference/cgls-stability-rtx4070-laptop.json.gz)
includes every result and stopping diagnostic.

A later full-suite CUDA run exposed an intermittent Joseph/JAX case: atomic
transpose noise could keep the conjugate recurrence moving near the FP32
cancellation floor, avoiding the update-stagnation check until the iteration
limit. In 200 repeated solves of the same independent 60-voxel damped system,
one original solve exceeded the existing `5e-6` normalized-stationarity bound.
The solver now verifies the true residual every 16 iterations once the squared
normal-residual norm falls below `eps32` times its initial value. This preserves
the componentwise cancellation estimate and requested convergence tolerance.
All 200 corrected solves passed, with maximum normalized stationarity
`1.31e-6`, and all stopped at iteration 80 with the honest `roundoff_limit`
diagnostic. The [repeat record](../bench/reference/joseph-atomic-noise-floor-repeat.json.gz)
retains every before/after result. This is evidence for one small noisy-reduction
case, not a universal CGLS accuracy or speed claim. The subsequent full CUDA
suite passes 539 tests with one expected skip; focused CPU solver coverage
passes 38 tests with 13 CUDA skips.

The refreshed seven-repeat Gaussian comparison gives these complete warm
times in milliseconds, including quality verification:

| Case | Joseph CGLS | Multires Joseph | FBP + Joseph | ASTRA CGLS | ASTRA FBP + CGLS |
|---|---:|---:|---:|---:|---:|
| Parallel 64 | 29.5 | 21.9 | 13.3 | 28.4 | 10.9 |
| Anisotropic 64 | 41.1 | 19.1 | 28.0 | 30.5 | 19.1 |
| Laminography 64 | 1017.1 | 297.3 | Unsupported | 677.6 | 641.0 |
| Parallel 128 | 190.0 | 67.9 | 63.9 | 139.3 | 64.0 |
| Anisotropic 128 | 265.6 | 68.4 | 155.9 | 113.8 | 70.9 |
| Laminography 128 | 7787.5 | 1177.9 | Unsupported | 3212.9 | 3107.2 |

The selected budgets are unchanged by the stability correction. No supported
worker has an execution failure, but the sharp and noisy anisotropic-64 cases
still fail the fixed quality gate for all five methods. The earlier accepted
single-pass FBP results remain faster than iterative methods on several sharp
cases; they must still be included when selecting the fastest external workflow.
TomoJAX's sampled process peaks in this refresh are 212–446 MiB, versus
150–296 MiB for ASTRA: the memory target is not met. Cold search times and all
rejected budgets remain in the raw records:
[Gaussian](../bench/reference/stable-gaussian-v1-64-128.json.gz),
[sharp](../bench/reference/stable-structured-v1-64-128.json.gz),
[noisy](../bench/reference/stable-structured-noisy-v1-64-128.json.gz).
The measured source is archived at `.artifacts/stability/measured-source.tar.gz`.

## Retaining FBP filter tails through the output volume

Investigation of Fourier preconditioning exposed a separate FBP error. Although
the raw projections are zero-extended for convolution, the filtered result was
cropped to the acquired detector width before backprojection. Ramp-filter tails
outside that width are nonzero and often negative. Discarding them biases
volume corners outside the detector's inscribed field of view, even when the
object itself fits inside the measured field.

Built-in parallel FBP now retains those tails at every reconstructed voxel.
Symmetric zero extension preserves the acquired detector coordinates, including
offsets and shifted volume origins. An independent NumPy sum of the infinite
discrete convolution checks every voxel on odd/even narrow detectors. The same
correction is applied independently to all external direct adapters and their
FBP initializers, with padding included in timing. No truth values, fitted
amplitude or extra measured rays enter this correction. It assumes zero
unmeasured attenuation; genuinely truncated objects still need a suitable model.
Generic filtered adjoints, including explicit detector grids, retain their
supplied detector support.

The initial experiment reduced size-128 parallel Gaussian error from 3.05% to
0.10%, and anisotropic error from 8.67% to 0.96%, in both TomoJAX and ASTRA.
The full corrected suites pass **355 CPU tests** (63 skipped) and **417 CUDA
tests** (one skipped). Compute Sanitizer passes **29 FBP tests** with zero
reported errors. Static checks, import boundaries, package build, Twine
validation and installed-wheel smoke also pass. Earlier direct/hybrid records
predate this correction and remain historical evidence.

Seven-repeat complete warm timings after the correction (milliseconds):

| Suite / case | TomoJAX FBP | CuPy/ASTRA FBP |
|---|---:|---:|
| Gaussian / parallel 64 | 4.14 | 3.53 |
| Gaussian / anisotropic 64 | 3.83 | 3.20 |
| Gaussian / parallel 128 | 20.51 | 25.60 |
| Gaussian / anisotropic 128 | 13.88 | 13.49 |
| Sharp / parallel 64 | 4.74 | 3.75 |
| Sharp / parallel 128 | 20.07 | 25.52 |
| Sharp / anisotropic 128 | 14.57 | 13.43 |
| Noisy / parallel 64 | 4.55 | 3.77 |
| Noisy / anisotropic 64 | 3.70 | 3.27 |
| Noisy / parallel 128 | 20.14 | 25.19 |
| Noisy / anisotropic 128 | 14.12 | 13.48 |

Every listed result passes the unchanged quality gate. Sharp anisotropic-64
still fails (about 17.6% error versus 15% allowed). The noisy version now passes
at about 17.7%, below its predeclared 18% limit. Tilted filtered ASTRA BP still
fails all three suites; TomoJAX direct Pallas FBP is unsupported for that geometry.
Native ASTRA 2D and TIGRE FBP pass the parallel cases but are slower than the
CuPy/ASTRA workflow in these measurements. Gaussian TomoJAX cold complete calls
take 0.77–0.91 seconds, with sampled worker peaks of 236–414 MiB; the corresponding
CuPy/ASTRA peaks are 174–312 MiB. These results still miss the external speed
and memory targets.

All samples, failures and external native methods are retained:
[Gaussian](../bench/reference/full-support-fbp-gaussian-v1-64-128.json.gz),
[sharp](../bench/reference/full-support-fbp-structured-v1-64-128.json.gz),
[noisy](../bench/reference/full-support-fbp-structured-noisy-v1-64-128.json.gz).
The measured source is archived at `.artifacts/fbp-tails/measured-source.tar.gz`.

Exploratory circulant, zero-padded Fourier and cosine-boundary preconditioners
did not reduce the selected Gaussian quality-gated iteration budgets. Several
aggressive variants made them substantially worse. These prototypes remain
local experiments under `.artifacts/precond/`; no public preconditioner or
performance claim was added.

## Large reconstruction: 720 views

The predefined large check uses the same independent Gaussian phantom and 3%
full-volume error gate, now at 720 views. It is a separate view count from the
180-view standard suite. Initial complete warm times and sampled process peaks:

| Size | Method | Warm ms | Peak MiB | Result |
|---|---|---:|---:|---|
| 256 cubed | TomoJAX FBP | 243.5 | 4254 | Accepted |
| 256 cubed | CuPy/ASTRA FBP | 465.9 | 2820 | Accepted |
| 256 cubed | Native ASTRA 2D | 564.0 | 148 | Accepted |
| 256 cubed | TIGRE FBP | 994.3 | 218 | Accepted |
| 512 cubed | TomoJAX FBP | — | 4232 | Out of memory |
| 512 cubed | CuPy/ASTRA FBP | — | 6928 | Out of memory |
| 512 cubed | Native ASTRA 2D | 3533.1 | 156 | Accepted |
| 512 cubed | TIGRE FBP | 5917.2 | 734 | Accepted |

Both all-view workflows fail while allocating FFT-related buffers, before an
accepted 512-cubed result. TomoJAX's sampled peak does not include the failed
2.83 GiB allocation. Native ASTRA's slice-wise workflow passes with much lower
device memory; its 512-cubed cold complete time is 4.62 seconds. These failures
motivate streaming filtering/backprojection in bounded batches. The raw
[large baseline](../bench/reference/full-support-fbp-gaussian-256-512-720.json.gz)
retains seven repeats, errors and failure logs. Its source archive is
`.artifacts/showcase/baseline-measured-source.tar.gz`.

Fixture generation now uses bounded FP64 coordinate/projection chunks and
casts each accumulated element to FP32 once. This keeps large independent truth
generation practical without changing its equations or component order. All
poses, angles, volume and projection arrays match the previous implementation
bit for bit in 18 cases spanning sizes 33/64/128, 35/180 views, all geometries
and both phantom types. The [parity record](../bench/reference/fixture-chunking-parity.json.gz)
retains each array hash and the before/after generator hashes.

The subsequent streaming implementation filters and backprojects bounded view
batches before moving to the next one, avoiding a full padded/filtered projection
stack. A conservative 512 MiB FFT-workspace estimate selects the batch; input
projections, volume buffers and runtime/compiler allocations are additional
costs. Small stacks remain a single batch. The final shifted batch masks prior
views so that every measurement contributes once. The explicit Pallas helper
also retains filter tails for translated poses.

The CuPy/ASTRA adapter now uploads and filters bounded view batches too, using
ASTRA's `accumulate_BP` on a linked GPU volume. All geometry construction,
linking, synchronization and cleanup remain inside each timed call. Partial
batches match whole-stack results and independent physical-scale Gaussian
truth in dedicated tests; the comparison does not rely on a memory-failing
external baseline.

After streaming changes, the full suites pass **362 CPU tests** (79 skipped)
and **440 CUDA tests** (one skipped). Ruff, configured type checking, import
boundaries, lockfile validation, package build, Twine, installed-wheel smoke
and the CPU CLI workflow pass. Compute Sanitizer passes **49 FBP tests** with
zero reported errors. Final complete streaming timings follow in the
retained measurement records rather than the exploratory resident-kernel times.

The isolated streamed **512-cubed / 720-view** TomoJAX solve now passes in
**4.110 seconds cold and 1.740 seconds warm** (median of seven complete calls).
Full-volume relative L2 error is `1.488e-4` against the unchanged 0.03 limit.
CuPy/ASTRA's equally streamed workflow takes **4.659 seconds cold and
3.513 seconds warm**, with `1.486e-4` error. TomoJAX uses **4250 MiB** sampled
peak process memory, versus **1262 MiB** for CuPy/ASTRA. Thus the Gaussian
reconstruction component of the 8 GB showcase is demonstrated, but the
half-memory target and the full multi-case speed goals remain unmet. No motion
recovery is included in these timings.

At 256 cubed, streaming reduces TomoJAX's sampled peak from 4254 to **1178 MiB**,
with complete warm time **238.6 ms** versus **479.8 ms** for CuPy/ASTRA.
Cold times are 1.173 and 1.209 seconds respectively. The improvement primarily
addresses memory; speed comparisons must use the complete times, not just the
resident prototype's subsecond kernel measurements.

The [streamed large record](../bench/reference/streamed-fbp-gaussian-256-512-720.json.gz)
retains all competing methods and every timing sample.
Native ASTRA 2D takes 4.457 seconds cold and 3.560 seconds warm at 512 cubed,
with only 156 MiB sampled peak; TIGRE takes 7.239 seconds cold and 5.961 seconds
warm, with 734 MiB peak. The native slice-wise memory result is another reason
the relative memory goal remains open. The large measured source is archived
at `.artifacts/showcase/streamed-measured-source.tar.gz`.

The final smaller-case refresh preserves ASTRA's cheaper `direct_BP` path when
the entire stack fits one batch, avoiding unnecessary registered-handle overhead.
Its batch and whole-stack paths pass 20 external-adapter tests. The intermediate
registered-handle-only measurements remain under `streamed-fbp-registered-handles-*`.
Latest complete warm Gaussian times in milliseconds:

| Case | TomoJAX FBP | CuPy/ASTRA FBP |
|---|---:|---:|
| Parallel 64 | 3.77 | 3.56 |
| Anisotropic 64 | 3.95 | 3.23 |
| Parallel 128 | 19.84 | 25.30 |
| Anisotropic 128 | 14.20 | 12.93 |

Acceptance outcomes on the sharp and noisy suites remain unchanged by streaming.
Sharp anisotropic-64 and all tested tilted direct reconstructions still fail
their gates. The latest small records include every external native method:
[Gaussian](../bench/reference/streamed-fbp-gaussian-v1-64-128.json.gz),
[sharp](../bench/reference/streamed-fbp-structured-v1-64-128.json.gz),
[noisy](../bench/reference/streamed-fbp-structured-noisy-v1-64-128.json.gz).
Their source archive is `.artifacts/showcase/fastpath-measured-source.tar.gz`.

## Host-output axial slabs

The public `fbp_host` API accepts NumPy arrays or memmaps and optionally writes
into caller-provided FP32 host storage. It transfers only the detector rows
needed for each axial slab and downloads each completed slab immediately.
Fractional row positions retain both interpolation neighbors. The fixed local
grid and dynamic row phase reuse one compiled shape, including the final partial
slab. Tilted geometry is explicitly unsupported. The existing `fbp` API keeps
its device-returning behavior.

The retained comparison freezes 16 output slices and 32 filtering views per
batch, informed by Gaussian development runs. All slab assembly, uploads,
downloads, output writes and quality verification are timed. Complete large
Gaussian results, in milliseconds, with the unchanged 3% gate:

| Size | Method | Cold ms | Warm ms | Peak MiB |
|---|---|---:|---:|---:|
| 256 cubed | TomoJAX host slabs | 1106.6 | 281.2 | 288 |
| 256 cubed | TomoJAX resident volume | 1177.3 | 241.1 | 1178 |
| 256 cubed | CuPy/ASTRA | 1200.1 | 483.9 | 814 |
| 256 cubed | Native ASTRA 2D | 1202.2 | 568.1 | 148 |
| 256 cubed | TIGRE | 1902.5 | 1004.0 | 218 |
| 512 cubed | TomoJAX host slabs | 3937.2 | 1775.3 | 416 |
| 512 cubed | TomoJAX resident volume | 3142.4 | 1744.6 | 4250 |
| 512 cubed | CuPy/ASTRA | 4636.3 | 3510.7 | 1262 |
| 512 cubed | Native ASTRA 2D | 4547.1 | 3533.0 | 156 |
| 512 cubed | TIGRE | 7330.4 | 5931.6 | 734 |

Every large result is accepted across seven repeats. The slab path's full-volume
relative L2 errors are `2.8584e-4` and `1.4878e-4`. At 512 cubed it reduces
TomoJAX's sampled peak by 90.2%, with a 1.8% warm-time increase. At 256 cubed the
memory reduction is 75.6%, with a 16.6% warm-time increase. This is a useful
memory/runtime tradeoff rather than a universal speed improvement.

Compared with the fastest warm external method here, CuPy/ASTRA, slabs use
35.4% and 33.0% of its sampled memory at 256 and 512 cubed. Warm speedups are
1.72x and 1.98x; cold speedups over the fastest external cold result are only
1.08x and 1.15x. Native ASTRA 2D still uses less memory than either TomoJAX path
and has the fastest external cold result at 512 cubed. These measurements do
not satisfy the overall speed or relative-memory targets.

The [large slab record](../bench/reference/host-fbp-gaussian-256-512-720.json.gz)
retains all samples. Memory is per-process accounting sampled at a requested
10 ms cadence; short allocations may be missed. Source is archived at
`.artifacts/host-fbp/measured-source.tar.gz`, with SHA-256
`cb54455091a7b85c8a0e7bfcfdbf10e9490ec711c16f0d92b1bef9aa870012bf`.

Validation passes **389 CPU tests** (91 skipped), **479 CUDA tests** (one
skipped), and **39 slab tests under Compute Sanitizer** with zero errors.
Checks include independent physical Gaussian attenuation, fractional and
out-of-detector rows, anisotropic spacing, partial slabs, compilation reuse,
memmap output and overlapping-storage rejection. Ruff, configured type checking,
import boundaries, package build, Twine and an installed-wheel smoke check pass.

The completed [180-view Gaussian sweep](../bench/reference/host-fbp-gaussian-v1-64-128-256.json.gz)
also retains small-case regressions. Complete warm milliseconds:

| Case | Host slabs | Resident volume | CuPy/ASTRA |
|---|---:|---:|---:|
| Parallel 64 | 6.30 | 4.66 | 3.76 |
| Anisotropic 64 | 7.10 | 3.49 | 3.25 |
| Parallel 128 | 28.66 | 19.65 | 26.18 |
| Anisotropic 128 | 20.44 | 13.52 | 13.84 |
| Parallel 256 | 193.32 | 154.19 | 324.71 |
| Anisotropic 256 | 126.27 | 96.72 | 106.05 |

All six supported cases pass the original Gaussian gate. Slabs use 224–236 MiB,
but transfer/dispatch overhead outweighs the storage benefit for small scans.
At size 256, slab memory is 236 MiB versus 1178/666 MiB for resident parallel/
anisotropic reconstruction. Tilted TomoJAX FBP remains unsupported and the
approximate CuPy/ASTRA tilted reconstruction fails its gate at every size.

The original [sharp sweep](../bench/reference/host-fbp-structured-v1-64-128-256.json.gz)
was interrupted after **32 of 45 comparisons** when the execution environment
changed. Its retained rows include the failed anisotropic-64 gate and accepted
parallel-256 TomoJAX results, but it is not a completed sweep. No corresponding
new noisy sweep completed. These records retain the same archived source hash;
they must not be combined silently with measurements of subsequent source.
The benchmark runner now supports checked `--resume` for future interruptions,
with atomic result writes and rejection of changed source/settings/environment.

After CUDA access returned, fresh [sharp](../bench/reference/host-fbp-resumable-structured-v1-64-128-256.json.gz)
and [noisy](../bench/reference/host-fbp-resumable-structured-noisy-v1-64-128-256.json.gz)
sweeps completed all 45 comparisons each, including native external methods.
The interrupted evidence remains separate. The sharp sweep has 21 accepted,
6 failed-quality and 18 unsupported comparisons; the noisy sweep has 24 accepted,
3 failed-quality and 18 unsupported comparisons. Size-256 full-volume errors:

| Suite / geometry | TomoJAX host slabs | TomoJAX resident | CuPy/ASTRA | Gate |
|---|---:|---:|---:|---:|
| Sharp / parallel | 7.249% | 7.249% | 7.249% | 15% |
| Sharp / anisotropic | 9.342% | 9.342% | 9.342% | 15% |
| Noisy / parallel | 10.240% | 10.240% | 10.240% | 18% |
| Noisy / anisotropic | 11.564% | 11.564% | 11.567% | 18% |

Sharp anisotropic-64 still fails the 15% gate at about 17.6% error for all three
methods. The noisy counterpart passes its separately predefined 18% gate.
Tilted direct reconstruction remains unsupported in TomoJAX and fails the
external approximate-FBP gate at every size. Passing the larger sharp/noisy
parallel cases does not establish successful tilted reconstruction or motion
recovery. These completed sweeps use source SHA-256
`1fc95afc584e77323d14267a2e809579a785dd23bc3895057744f9d995a70bb9`,
archived at `.artifacts/host-fbp/resumable-measured-source.tar.gz`.

A further host-copy optimization avoids zero-initializing completely measured
slabs and scales downloads directly into the output array. Alternating baseline
and optimized calls at 256/512 cubed produce bit-for-bit identical output and
about 2–3% lower reconstruction time, excluding quality verification, in the
controlled pilot. A fresh isolated complete comparison retains all five methods
and seven repeats in the [copy-optimization record](../bench/reference/host-fbp-copies-gaussian-256-512-720.json.gz).
The slab path takes 1.112 s cold / 0.256 s warm at 256 cubed, and 3.907 s cold /
1.751 s warm at 512 cubed, with unchanged sampled peaks of 288/416 MiB.
CuPy/ASTRA takes 0.482/3.515 s warm; native ASTRA 2D takes 0.569/3.548 s warm.
The larger before/after complete-time difference at size 256 includes ordinary
timing variation and must not all be attributed to the copy change.

This measured source is archived at `.artifacts/slab-tuning/measured-source.tar.gz`
with SHA-256 `d80bce108160f2c0320a2ec02a678bc266f64f13c23dc376af9848d2f0b2ddf0`.
The arithmetic passes 88 FBP tests under Compute Sanitizer with zero errors
and 59 CPU tests (29 CUDA skips). Subsequent extraction of the row-copy helper
passes all 39 host-slab CUDA tests and 27 CPU tests, plus formatting/lint checks.
Filter-batch profiling did not establish a sufficiently broad gain to change
the fixed 32-view policy.

## Fourier-slice reconstruction with host slabs

The opt-in `fourier_reconstruct` API uses six-point Kaiser–Bessel radial
interpolation, linear angular interpolation, a detector-Nyquist disk cutoff and
a padded Cartesian inverse FFT. It preserves physical attenuation without
fitting or clipping. The NumPy implementation is a double-precision reference;
CUDA uses CuPy FFTs and an original FP32 interpolation kernel. Both process
16-slice slabs in these comparisons, with NumPy/memmap input and host output.
Only uniform, unique half-turn built-in parallel scans are supported. Reordered,
reversed and opposing angles, detector offsets, anisotropic spacing and cropped
output grids have separate independent analytic tests. The output domain covers
the acquired detector field as well as the requested ROI before FFT padding.

All four isolated sweeps completed: **174 records**, with seven repeats for each
accepted method. Timings include host transfers, setup, reconstruction, output
writes and the unchanged full-volume FP64 quality check. Sources are frozen at
SHA-256 `f4e139b00bc9a546359e6706985e2bddda860a4f28eb987114d189699ceed256`,
archived in `.artifacts/fourier-recon/measured-source-final.tar.gz`. CuPy is 14.2.0;
the other hardware/software metadata are retained in each record. Fresh-process
cold timing retains normal filesystem/compiler caches, as elsewhere in this report.

The standard 180-view Gaussian results are:

| Case | Fourier cold ms | Fourier warm ms | Peak MiB | Relative L2 | Warm speedup over fastest external |
|---|---:|---:|---:|---:|---:|
| parallel-64-180 | 694.5 | 2.59 | 152 | 0.000115 | 1.39× |
| anisotropic-64-180 | 678.0 | 2.08 | 152 | 0.013551 | 1.54× |
| parallel-128-180 | 1677.1 | 15.98 | 174 | 0.000134 | 1.57× |
| anisotropic-128-180 | 821.9 | 11.63 | 174 | 0.009877 | 1.18× |
| parallel-256-180 | 985.4 | 150.28 | 252 | 0.000149 | 2.16× |
| anisotropic-256-180 | 910.6 | 82.68 | 248 | 0.013707 | 1.24× |

Gaussian warm geometric-mean speedup is **1.48×**, while the cold ratio is
**0.75×**: startup often loses to the fastest external workflow. Sharp/noisy
warm ratios are 1.46×/1.45× over the cases with accepted external comparisons;
they do not count failed or unsupported cases as wins. Raw
[Gaussian](../bench/reference/fourier-gaussian-v1-64-128-256-180.json.gz),
[sharp](../bench/reference/fourier-structured-v1-64-128-256-180.json.gz) and
[noisy](../bench/reference/fourier-structured-noisy-v1-64-128-256-180.json.gz)
records retain every gate, timing sample and failure.

Fourier passes all six supported Gaussian and noisy cases and five of six sharp
cases. Sharp anisotropic-64 error is **0.177969**, failing the unchanged 0.15 gate;
all three compared direct FBP implementations also fail that case. Its noisy
error is **0.178906**, narrowly within the declared 0.18 noisy gate. At size 256,
sharp parallel/anisotropic errors are 0.068552/0.088213 and noisy errors are
0.097765/0.102425. Tilted Fourier reconstruction is unsupported. None of these
results establish motion recovery or universal sharp-data accuracy.

The additional [720-view comparison](../bench/reference/fourier-gaussian-v1-256-512-720.json.gz)
passes all twelve method/case pairs:

| Grid | Method | Cold ms | Warm ms | Peak MiB |
|---|---|---:|---:|---:|
| 256³ | TomoJAX Fourier | 2208.6 | 212.2 | 300 |
| 256³ | TomoJAX host FBP | 1220.6 | 269.6 | 288 |
| 256³ | TomoJAX resident FBP | 1318.7 | 243.1 | 1178 |
| 256³ | CuPy/ASTRA FBP | 1235.5 | 484.0 | 814 |
| 256³ | ASTRA native 2D FBP | 1222.8 | 566.5 | 148 |
| 256³ | TIGRE FBP | 1887.6 | 989.4 | 218 |
| 512³ | TomoJAX Fourier | 3541.7 | 1527.7 | 764 |
| 512³ | TomoJAX host FBP | 3030.5 | 1737.2 | 416 |
| 512³ | TomoJAX resident FBP | 3221.6 | 1742.4 | 4250 |
| 512³ | CuPy/ASTRA FBP | 4658.5 | 3518.6 | 1262 |
| 512³ | ASTRA native 2D FBP | 4453.7 | 3536.0 | 156 |
| 512³ | TIGRE FBP | 7303.1 | 5856.9 | 734 |

Fourier's 512³ error is **0.015018%**, close to host FBP's 0.014878%. Its
**3.54 s cold / 1.53 s warm** is about **1.26× / 2.30×** faster than the respective
fastest accepted external workflow. Sampled peak memory is **764 MiB**, 61% of
the warm-fastest CuPy/ASTRA workflow's 1262 MiB, so the half-memory goal is missed.
Host FBP remains the lower-memory TomoJAX choice at 416 MiB and is faster cold
in this run. Native ASTRA 2D uses just 156 MiB. The 10× mean / 2× every-case goal
is still unmet. All large results concern Gaussian reconstruction alone.

Validation includes 51 Fourier-focused tests on CUDA, 46 of them under Compute
Sanitizer with **zero errors** before the final host-validation additions.
Independent dense Fourier sums check odd/even detectors, signed signals, edge
impulses and Nyquist interpolation. The full CPU suite passed 436 tests with
104 CUDA skips; the subsequent CGLS check has separate focused CPU evidence
above. The final full CUDA suite passed 539 tests with one expected skip.
Formatting/lint, configured type checking, import contracts, package build and
Twine validation pass; the Fourier wheel also passed a clean-environment import,
CLI simulation and dataset-validation smoke test.

### Real inverse FFT follow-up

The CUDA path now stores only half the Cartesian spectrum and uses a real
inverse FFT. It handles self-mapped Nyquist axes explicitly when a coarse output
axis lies inside the measured frequency disk. The NumPy reference retains the
full complex inverse FFT. Angular coordinates remain FP64 until the kernel forms
the interpolation fraction; prematurely rounding a 720-view coordinate to FP32
had caused avoidable asymmetry on signed high-frequency data. FFTs, spectrum
storage and interpolation values remain FP32/complex64 on CUDA.

A controlled 256³/720-view pilot reduced raw reconstruction time from about
127 ms to 105 ms, with a `1.31e-7` relative difference between the accurate full
and half-spectrum implementations on random signed data. This pilot excludes
quality verification and is separate from the complete records below.

All **87 follow-up records completed**, using the same fixed 16-slice policy,
seven repeats, physical quality gates and complete timing scope. Fresh workers
reran both external ASTRA workflows that supplied the previous fastest cold
and warm baselines. Source SHA-256 is
`1f170b37aa5e3a3f21b4428f3e18ce4e27882fb37a027d952068d88bb399f87a`,
archived at `.artifacts/fourier-recon/measured-source-rfft.tar.gz`. The prior
six-method records, including TIGRE and both TomoJAX FBP paths, remain above.

| Grid / views | Method | Cold ms | Warm ms | Peak MiB |
|---|---|---:|---:|---:|
| 256³ / 720 | TomoJAX real Fourier | 1068.7 | 191.1 | 248 |
| 256³ / 720 | CuPy/ASTRA FBP | 1271.7 | 478.7 | 814 |
| 256³ / 720 | ASTRA native 2D FBP | 1193.6 | 566.5 | 148 |
| 512³ / 720 | TomoJAX real Fourier | 3385.9 | 1360.6 | 582 |
| 512³ / 720 | CuPy/ASTRA FBP | 4596.8 | 3512.9 | 1262 |
| 512³ / 720 | ASTRA native 2D FBP | 4518.6 | 3519.9 | 156 |

At 512³, complete warm time falls from 1.528 s to **1.361 s**, and sampled GPU
memory from 764 MiB to **582 MiB**, at unchanged 0.015018% error. Speedup over
the respective fastest external workflow is **1.33× cold / 2.58× warm**. Memory
is **46%** of the warm-fastest CuPy/ASTRA workflow's 1262 MiB, meeting the
half-memory target for this comparison. Native ASTRA 2D still uses just 156 MiB;
this does not meet the memory target against that cold-fastest workflow.

On standard 180-view Gaussian scans, warm time is 2.71/16.66/133.27 ms for
parallel sizes 64/128/256 and 2.05/10.40/76.18 ms for anisotropic sizes.
Sampled peaks are 146–222 MiB. Warm geometric-mean speedup over the rerun
external competitors is **1.53×**, still below 10× and below 2× on several cases.
Small parallel cases show minor timing regressions from the prior implementation;
these are retained, not averaged away as universal wins. The quality outcomes
are unchanged: all supported Gaussian/noisy cases pass, sharp anisotropic-64
fails, and tilted Fourier reconstruction remains unsupported. The raw
[Gaussian](../bench/reference/fourier-rfft-gaussian-v1-64-128-256-180.json.gz),
[sharp](../bench/reference/fourier-rfft-structured-v1-64-128-256-180.json.gz),
[noisy](../bench/reference/fourier-rfft-structured-noisy-v1-64-128-256-180.json.gz) and
[large](../bench/reference/fourier-rfft-gaussian-v1-256-512-720.json.gz) records retain
all results. Cold timings reflect both implementation and ordinary first-call
cache variation and should not all be attributed to the real FFT.

The updated implementation passes **60 Fourier tests under Compute Sanitizer,
with zero errors**, including 720-view signed data, coarse output axes, odd/even
transforms, Nyquist boundaries, physical Gaussians, cropped ROIs and host-storage
validation. CPU/reference/public-surface/benchmark checks pass (66 tests, 16 CUDA
skips before six additional CUDA-only cases), as do lint, configured typechecking,
package build and Twine validation. A separate cubic detector-v interpolation
pilot did not rescue the failed sharp case and worsened several noisy errors,
so it was not promoted; linear detector-v interpolation remains the policy.

## Remaining work

An independent Claude Opus 5.5 High review recommended measuring and replacing
the scattered-atomic adjoint before further multiresolution tuning, adding
external FBP baselines, and testing sharp/noisy data, now measured above. It also identified the
old-checkpoint geometry incompatibility now guarded by a pyramid version marker.
The initial bottleneck estimate used different batches; the controlled profile
and implemented Joseph gather transpose now provide direct evidence above.
The first-order Joseph pose derivatives and fused least-squares operation are
now implemented and measured below; integrating them into a successful joint
alignment workflow remains work in progress. Fused normal-equation accumulation
is implemented and measured below. The initial Fourier-preconditioning pilots above
did not establish a speedup.
The review also called for recomputed-residual checks and localized-update
tests for CGLS. The subsequent stability correction above addresses those
issues, while long-run accuracy on broader ill-conditioned datasets still
needs evidence.

The review also highlights limits of the stretch goal: small-case process memory
is dominated by runtime overhead, and a 10x solve improvement needs algorithmic
work as well as kernel changes. The goal remains unchanged and unmet. Smooth
Gaussian success alone cannot justify an automatic solver policy. Detector
decimation preserves measured-ray coordinates but loses coarse-stage information;
detector-area averaging would require a matching physical measurement model.

The metrics inventory is not fully optimized or fully benchmarked. The current
work does **not** establish that TomoJAX is faster than ASTRA/TIGRE in every metric
or geometry. The initial size-64/128 Gaussian solver comparisons establish a
baseline, with size-64/128 sharp/noisy and independent gVXR material fixtures
revealing limitations. Larger sharp/noisy and real-data comparisons remain outstanding; L-BFGS cold compilation
remains substantial.


More GPUs, larger scans, realistic noisy/truncated data, matched-quality solver comparisons against ASTRA/TIGRE,
and peak memory measurements are needed before broader comparisons. Multi-GPU
and cone/fan-beam geometries are outside the coverage established here.

JAX was updated to 0.11.2 and is constrained below 0.12. Its installed Pallas
Triton backend emits a deprecation warning recommending Mosaic GPU or native
Triton bindings. The kernels work on the tested Ada GPU, but migration remains
necessary before removing the version bound. The JAX reference backend remains
available; an upgrade should pass the numerical and real-CUDA gates before the
bound changes. See [JAX's Pallas documentation](https://docs.jax.dev/en/latest/pallas/)
for the evolving backend APIs.

## Bounded-memory first-order Joseph differentiation

`tomojax.forward.project_joseph` now exposes the plane model with explicit JAX
or CUDA selection. CUDA implements first-order forward and reverse derivatives
for volume and poses, batching, and transposing a linearization. The custom
reverse rule retains the input volume and prepared coefficients, and recomputes
bilinear sample derivatives using five per-ray running sums. It does not store
a ray-by-plane tape. `joseph_l2_value_and_grad` also fuses raw half squared error
with pose-gradient accumulation, then applies the matched gather transpose to
the residual sinogram. It returns matrix-pose gradients; rigid pose parameters
need the corresponding chain rule. The existing alignment pipeline retains its
trilinear ray model, so these measurements do not establish a joint alignment
recovery speedup. CUDA higher derivatives are unsupported; the JAX reference
remains available.

The backprojector now uses nested detector-coordinate loops, preserving the
footprint and accumulation order while removing a dynamic integer division
and remainder from every candidate sample. A preliminary 256-cubed/60-view
comparison reduced parallel backprojection from 38.1 to 28.0 ms and tilted
backprojection from 65.8 to 37.3 ms, with bitwise-identical outputs. These are
internal component measurements, not comparisons against ASTRA/TIGRE.

The retained `joseph-derivatives-v2` sweep uses the existing Gaussian physical
fixtures, 60 views, resident FP32 inputs, seven synchronized samples, and at
least 200 ms of warmup **for each method**. Geometry preparation is included;
imports, transfers, reconstruction and alignment are excluded. General reverse
AD differentiates a plain least-squares composition through `project_joseph`;
the explicit fused operation avoids its separate forward/pose-gradient pass.
All 12 cases pass the numerical gates. Medians in milliseconds:

| Geometry / size | Forward | General reverse: loss + both gradients | Fused loss + both gradients | Fused / forward |
|---|---:|---:|---:|---:|
| Parallel / 32 | 0.068 | 0.219 | 0.193 | 2.84× |
| Anisotropic / 32 | 0.061 | 0.181 | 0.148 | 2.44× |
| Laminography / 32 | 0.070 | 0.238 | 0.207 | 2.97× |
| Parallel / 64 | 0.231 | 1.525 | 1.501 | 6.51× |
| Anisotropic / 64 | 0.154 | 1.245 | 0.853 | 5.55× |
| Laminography / 64 | 0.225 | 1.391 | 1.641 | 7.30× |
| Parallel / 128 | 2.390 | 7.372 | 5.647 | 2.36× |
| Anisotropic / 128 | 1.453 | 4.461 | 3.685 | 2.54× |
| Laminography / 128 | 2.458 | 8.974 | 7.481 | 3.04× |
| Parallel / 256 | 14.694 | 54.313 | 39.868 | 2.71× |
| Anisotropic / 256 | 7.354 | 27.679 | 20.809 | 2.83× |
| Laminography / 256 | 14.466 | 64.431 | 50.106 | 3.46× |

One-warp gather tiles and exact unit-row specialization produce **7/12** cases within
the 3× differentiation target for the fused operation;
it remains unmet for the suite. Small timings still show variability, including
outliers retained in every sample list. The first sweep had no duration-based
warmup and appeared to pass 8/12; its version-1 record is retained. The first
warmed sweep with four-warp gather tiles passed 4/12; the one-warp follow-up
passed 6/12. Both are retained. Runtime/clock state can materially change such
small ratios.

For sizes 32/64 the fused CUDA operation is 16.6–34.2× faster than ordinary
JAX reverse AD for this same plane model, with maximum loss/volume/pose relative
L2 discrepancy `4.4e-7`. At every size, an independent derivative computation
using JAX forward mode agrees with the gradient-direction inner product to
`2.3e-9` after scaling by gradient/direction norms. This complements independent
physical-ray matrices and finite differences in the tests; it is not a
pose-recovery accuracy measurement.

At size 64, parallel/tilted compiler temporary estimates fall from
1,519,792,408 bytes for JAX reverse AD to 1,112,896 bytes for the fused CUDA
operation. Size-256 fused estimates are 9,248,064–17,707,840 bytes, in addition to
input/output storage. **These are compiler estimates, not process peak VRAM.**
Full JAX reverse AD was deliberately not run above size 64; those cases retain
forward-mode directional checks. Fused lowering/compilation took 0.340–0.432 s
on this machine, excluding imports and first execution.

Source hash for the unit-row warmed sweep and memory probes:
`924ed3873d73ea2fd0cab1899c96e2f07f6accb09eadc01cf23ae44b625d25b1`.
See [all current warmed samples and checks](../bench/reference/joseph-derivatives-v2-unitrows-32-64-128-256-60.json.gz),
[the one-warp follow-up](../bench/reference/joseph-derivatives-v2-tile32-32-64-128-256-60.json.gz)
(source `dd733337abc6879cb5ccc9a41bf615cc1520d03cfb42149f0bdd6fe7d2c3fe5e`),
[the four-warp warmed sweep](../bench/reference/joseph-derivatives-v2-32-64-128-256-60.json.gz)
(source `1a73d96e695e6d60d169460f2c8043eced2fe1b1ef436882ab922d831ec2776e`)
and [the initial comparison](../bench/reference/joseph-derivatives-v1-32-64-128-256-60.json.gz)
(source `5604ef48cb8d299bd6086a58b01b424a9a9e5ab18300d9076c1ca41c29408e95`).
The corresponding sources are archived locally. XLA emitted an autotuning
`Delay kernel timed out` diagnostic during each sweep; the reported component
samples use synchronized host wall time, not that internal event timer.

Validation after integration: **571 GPU tests passed, 2 expected skips**;
**52 focused CPU tests passed, 14 CUDA skips**. All 29 Joseph projector and
derivative tests pass under Compute Sanitizer with **zero errors**. Tests cover
signed data, unequal voxel/detector spacing, offsets, all dominant axes, masked
tail tiles, independent physical finite differences, JVP/VJP duality, Jacobians,
batched gradients, and transposed linearizations. Lint/format, configured type
checks and all three import contracts pass.


The unit-row follow-up also measures whole-process memory in **nine isolated
workers**, with per-PID `nvidia-smi` accounting requested every 10 ms. Compilation,
input preparation, runtime/allocator overhead and repeated calls are included.
All workers return finite outputs. This sampling can miss brief allocations;
it is distinct from the compiler estimates above.

| Geometry / size / 60 views | JAX reverse AD peak (MiB) | Fused CUDA peak (MiB) |
|---|---:|---:|
| Parallel / 64 | 2260 | 212 |
| Anisotropic / 64 | 1236 | 212 |
| Laminography / 64 | 2260 | 212 |
| Parallel / 256 | Not run | 398 |
| Anisotropic / 256 | Not run | 270 |
| Laminography / 256 | Not run | 398 |

At size 64 the measured CUDA peak is 9.4–17.2% of the JAX reference peak.
This is an internal derivative comparison, not the memory target against the
fastest accepted external reconstruction workflow. See [the isolated memory
records](../bench/reference/joseph-gradient-memory-v1-unitrows-64-256-60.json.gz).
The earlier one-warp memory probe returned the same peaks.

The final one-warp change passes **73 targeted tests, 2 expected skips under
Compute Sanitizer with zero errors**, plus a separate independent rigid-rotation
finite-difference test under the same sanitizer (1 passed, zero errors). The
built wheel and source distribution pass metadata validation and the installed
wheel CLI smoke test. A tighter per-column gather-bound experiment was rejected:
its extra arithmetic made every tested case slower, despite identical results.
Nsight profiling attributes the large tilted backprojector primarily to compute
rather than DRAM traffic; that guides further work without changing the model.


The unit-row specialization detects an exactly separable detector-to-plane map
with exactly unit row scale. It then computes the two contributing detector-row
weights before iterating over columns. Other spacing, mixed-view and tilted
cases use the general gather. The GPU chooses once per voxel tile from dynamic
coefficients, without a host geometry test or approximate-angle cutoff. The
256-cubed anisotropic fused operation falls from 25.4 to 20.8 ms in the retained
sweeps; small-case ratios still fluctuate. All **83 targeted tests pass, with
2 expected skips and zero Compute Sanitizer errors**, including reversed rows,
fractional offsets, near-unit spacing, long axial stacks, dynamic poses and
batched mixed dispatch. A separate FP32 atomic-transpose pilot passed independent
physical-matrix checks, but was slower in all three size-256 cases and was not
promoted.


## Cubic plane interpolation and pose-recovery controls

The optional `interpolation="cubic"` uses Keys convolution with parameter -1/2
and a 4-by-4 transverse stencil. Forward projection, the gather transpose,
volume/pose derivatives and CGLS use the same model. Negative weights are
preserved. CGLS uses `abs(A).T` in its componentwise roundoff estimate; signed
backprojection of absolute residuals is not a valid bound for this model.
The default remains bilinear interpolation. Cubic does not remove dominant-axis
switches, finite-support error, noise sensitivity, or missing-angle information.

The first cubic component sweep retained a failed size-256 anisotropic
directional check (`5.84e-5` versus the unchanged `2e-5` gate). Its expanded
reference polynomial canceled near the cubic kernel's outer roots, and the
forward summation order differed from CUDA. Stable cell-fraction weights and
matching forward accumulation remove this avoidable discrepancy. JAX still
differentiates the reference automatically, while separate FP64 physical-ray
matrices and finite differences validate the model independently. The gather
uses a factored outer-lobe polynomial; a high-precision polynomial regression
checks weights close to both roots. A near-zero-residual derivative test guards
against amplification of forward rounding differences.

The corrected cubic sweep passes **12/12 numerical cases**, with maximum full
size-32 gradient/loss relative discrepancy `1.53e-6` and maximum all-size
norm-scaled directional discrepancy `7.78e-9`. Only **2/12** cases meet the
3x fused-derivative cost target. Fused lowering/compilation takes 0.445–0.535 s.
Ordinary JAX reverse AD is limited to size 32 for cubic; larger shapes retain
ordinary-JAX forward-mode checks to avoid its large sample tape. At size 256
and 60 views, fused loss plus both gradients takes 147.9 ms parallel, 93.3 ms
anisotropic and 181.1 ms tilted, or 3.12x, 4.15x and 3.86x forward. These are
resident-input components, not complete reconstructions or alignment speedups.
The unchanged 200 ms warmup/seven-sample policy still exhibits small-case
variability; all samples remain in the records.

See the [corrected cubic component record](../bench/reference/joseph-derivatives-v3-cubic-stable-32-64-128-256-60.json.gz),
source `dd045b16a8acb247e175a6a94fa727833cac40461848bd5b3fa680fd9ce53ce1`,
and the [initial cubic record with its failed gate](../bench/reference/joseph-derivatives-v3-cubic-32-64-128-256-60.json.gz),
source `63e47b59faa17bb7ab74719f4e1ff68e36d6eebd88e85f84b124fe120e4e5ac9`.
The [linear control](../bench/reference/joseph-derivatives-v3-linear-32-64-128-256-60.json.gz)
uses that initial source too. Both source snapshots are archived locally.

Before the stability follow-ups, the full GPU suite passed **615 tests, 2 expected
skips**, and **68 focused CPU tests passed with 31 CUDA skips**. The final
weights and absolute-transpose changes pass **117 focused tests, 2 expected
skips under Compute Sanitizer with zero errors**. Wheel/sdist metadata and
installed-wheel CLI checks passed for the initial cubic implementation; those
checks do not yet cover the later stability follow-ups.

The subsequent normal-equation build covers those follow-ups: the full GPU
suite passed **646 tests with 2 expected skips**, and **86 focused CPU tests
passed with 43 CUDA skips**. Lint, formatting, configured type checks and public
import contracts passed. Wheel/sdist metadata validation, installed-wheel CLI
checks and an isolated installed-wheel cubic/normal-equation API smoke passed.

`bench/pose_recovery.py` records a separate known-volume experiment. The target
is an independent analytic integral of two asymmetric continuous Gaussians,
with no fitted amplitude. It starts with independent +/-3-degree rotation and
+/-10-native-pixel translation errors, uses integer FFT correlation followed by
damped per-view Gauss–Newton, and retains cold plus three complete repetitions.
The time includes host transfers, correlation, compilation and output transfer;
fixture generation and error diagnostics are excluded. Translations use detector/
world x,z, explicitly different from the legacy pipeline's object-frame shifts.

The volume has a sampled physical margin of one quarter the native extent on
each side, retaining the Gaussian tails without changing detector rays or the
object's physical scale. Thus **64/128/256-pixel detector cases use
96/192/384-cubed volumes**, respectively. The gate requires both <=0.01-degree
rotation-vector error and <=0.05-pixel translation-vector error for >=99% of
views in every run. With 30 views, that requires all 30. Error rotations use
FP64 rotation vectors: FP32 arccos(trace) would erase errors below about .02
degrees.

| Detector width / volume width | Parallel noiseless angular RMSE | Tilted noiseless angular RMSE | Parallel noisy angular RMSE | Tilted noisy angular RMSE |
|---|---:|---:|---:|---:|
| 64 / 96 | 0.01125° | 0.00880° | 0.06055° | 0.06914° |
| 128 / 192 | 0.00136° | 0.00064° | 0.02770° | 0.02680° |
| 256 / 384 | 0.00024° | 0.00016° | 0.01381° | 0.01575° |

Cubic passes four of six noiseless cases (both geometries at native 128/256);
none of six noisy cases pass. Noise sigma is 0.005 times clean-projection RMS.
The linear noiseless control passes none of six: its angular RMSE at native
256 remains 0.0650° parallel and 0.0551° tilted. Every failed case is retained.
The cold cubic noisy workflow takes 1.61–6.73 s across these cases; repeated
size-256 cases take about 5.18 s parallel and 3.81 s tilted. These timings use
the explicit Jacobian implementation and are not a successful joint-alignment
baseline.

See the [cubic noiseless](../bench/reference/known-volume-pose-v1-cubic-noiseless-64-128-256-30.json.gz),
[cubic noisy](../bench/reference/known-volume-pose-v1-cubic-noisy-64-128-256-30.json.gz)
and [linear noiseless](../bench/reference/known-volume-pose-v1-linear-noiseless-64-128-256-30.json.gz)
records, all with source `63e47b59faa17bb7ab74719f4e1ff68e36d6eebd88e85f84b124fe120e4e5ac9`.
The subsequent numerical changes affect the reference, gather and CGLS bound;
the CUDA projection and pose-JVP used by these recovery runs are unchanged.

Exploratory controls also fit the exact analytic forward model and estimate
local noise sensitivity from its independent finite-difference Jacobian. At
native size 256, the expected angular RMS from that local linearized noise
model is about 0.0154° parallel and 0.0141° tilted, consistent with observed
errors. This is a diagnostic, not a proof about all estimators. Higher matrix
product precision was required in the analytic control: its fused
projection/Jacobian calculation otherwise changed projections by about
2.9e-4 relative to the separately compiled prediction. These controls do not
license dropping difficult noisy cases or claiming the recovery goal is met.

## Fused pose normal equations

`joseph_pose_normal_equations` accumulates each view's raw least-squares loss,
gradient and Gauss–Newton matrix directly from ray derivatives. It also returns
the residual image for line search. The caller supplies up to 16 pose-matrix
tangent directions, so parameter units and translation conventions remain
explicit. CUDA retains tile reductions and residual images instead of a full
detector-by-parameter Jacobian or per-plane differentiation tape. Damping,
constraints and gauge handling belong to the caller. The JAX implementation
provides an independently differentiated reference.

All **24 component cases** pass the fixed `2e-4` gates against the explicit
CUDA Jacobian and ordinary-JAX directional checks. Sizes are 32/64/128/256,
60 views, parallel/anisotropic/tilted geometries and both interpolations. Each
method receives at least 200 ms warmup and seven synchronized samples. The
geometric-mean component speedup over forming the explicit Jacobian is **4.47x
linear** and **4.87x cubic**. This comparison excludes input transfers and the
remaining recovery workflow.

| Model / geometry, size 256 | Explicit Jacobian + reduction | Fused equations | Compiler temporary storage, explicit / fused |
|---|---:|---:|---:|
| Linear parallel | 105.44 ms | 17.06 ms | 157.29 / 23.61 MB |
| Linear anisotropic | 48.18 ms | 8.61 ms | 82.08 / 12.33 MB |
| Linear tilted | 101.68 ms | 17.21 ms | 157.29 / 23.61 MB |
| Cubic parallel | 361.54 ms | 62.03 ms | 157.29 / 23.61 MB |
| Cubic anisotropic | 160.11 ms | 28.42 ms | 82.08 / 12.33 MB |
| Cubic tilted | 346.51 ms | 60.07 ms | 157.29 / 23.61 MB |

Compiler estimates are not peak process VRAM. The maximum full-output relative
discrepancy is `6.97e-6`; the independent directional/reference checks reach
`7.11e-5`. Both stay within the declared gates. The new API's 25 tests passed
Compute Sanitizer with zero errors; six workflow/fixture tests also passed
with zero sanitizer errors. See the
[linear component record](../bench/reference/joseph-pose-normals-v1-linear-32-64-128-256-60.json.gz)
and [cubic component record](../bench/reference/joseph-pose-normals-v1-cubic-32-64-128-256-60.json.gz).

Recovery version 2 compares explicit and fused equations with identical
initialization, damping, step clipping and candidate steps. It reuses the
current residual instead of projecting the zero-step candidate again. All
four six-case runs completed, retaining accuracy failures. Both implementations
pass **4/6 noiseless and 0/6 noisy cases**. Warm noisy recovery is 1.32–2.64x
faster with fused equations; at native 256 it drops from 5.82 to 2.76 seconds
parallel and from 4.52 to 2.36 seconds tilted. Cold times drop from 7.26 to
4.55 seconds and 5.62 to 3.91 seconds respectively. Each record includes cold
and three warm runs, so these workflow measurements have fewer repetitions
than the seven-sample headline policy. Angular errors remain above target on
this weak, noisy two-Gaussian fixture.

See version-2 [fused noisy](../bench/reference/known-volume-pose-v2-cubic-fused-noisy-64-128-256-30.json.gz),
[explicit noisy](../bench/reference/known-volume-pose-v2-cubic-explicit-noisy-64-128-256-30.json.gz),
[fused noiseless](../bench/reference/known-volume-pose-v2-cubic-fused-noiseless-64-128-256-30.json.gz)
and [explicit noiseless](../bench/reference/known-volume-pose-v2-cubic-explicit-noiseless-64-128-256-30.json.gz)
records. All component and version-2 workflow records use source
`e25849e0e0722e98bb58e5c83233cee82411d70613120dbe7862a81f21b74bf4`,
archived locally. These known-volume results establish neither a successful
joint alignment baseline nor the 20x joint workflow target.

### Streamed line search and informative-object recovery

Version 3 evaluates the four line-search candidates sequentially. It retains
one candidate image plus selected per-view parameters and losses, preserving
the previous ordering and first-candidate tie rule. An exploratory 30-view
tilted comparison selected identical parameters at all four sizes. At native
256 (384-cubed padded volume), compiler temporary storage fell from 55.17 to
23.66 MB, with comparable latency. This is a compiler-storage diagnostic,
not a process-memory or complete-workflow speed claim. Nine fixture/workflow
tests passed, including CPU/JAX and CUDA agreement and ties; the initial
streaming implementation passed Compute Sanitizer with zero errors. The
subsequent helper refactor also passed all nine tests, and its CPU run passed
six tests with three CUDA skips.

A fixed nine-Gaussian object adds asymmetry and spatially separated features
to the known-volume test. The record contains every center, width and amplitude.
It uses detector-sized volumes, no padding, true independent rotations within
±3 degrees and translations within ±10 pixels, and nominal initialization.
Noise remains 0.005 times clean-projection RMS; both original per-view accuracy
gates remain unchanged. This is an additional development fixture, not a
replacement for the weak two-Gaussian object or an independent held-out study.

Across parallel and tilted scans and seeds 9345/128904/61937, both explicit and
fused methods pass **12/18 cases** in the cold run and all seven warm repeats.
Every 128/256 case passes, while every 64 case fails the angular gate. For the
passing fused cases, maximum per-view angular error across all runs is 0.00419
degrees at size 128 and 0.00187 degrees at size 256. Warm recovery takes
96.9–160.3 ms at size 128 and 839.1–1107.7 ms at size 256. On these twelve
successful cases, fused versus explicit warm speedup is 1.34–2.66x, geometrically
2.04x; cold speedup averages only 1.12x geometrically and includes regressions.
Repeated calls within a case reuse compiled functions; each case clears the
JAX compilation cache. Neither process startup nor fixture creation is timed.

See [fused richer-object records](../bench/reference/known-volume-pose-v3-rich-cubic-fused-noisy-64-128-256-30.json.gz)
and [explicit richer-object records](../bench/reference/known-volume-pose-v3-rich-cubic-explicit-noisy-64-128-256-30.json.gz),
source `c821b964fb78b524bc69bd55972914b78c13225968200b97de8671621f573c32`.
The complete source snapshot is archived locally. Known-volume recovery still
does not establish joint reconstruction/alignment performance or broad
robustness on other objects and noise distributions.

The same version-3 sweep also completed the original weak-object controls with
seven warm repeats. The [noisy control](../bench/reference/known-volume-pose-v3-cubic-fused-noisy-64-128-256-30.json.gz)
still passes 0/6 cases and the
[noiseless control](../bench/reference/known-volume-pose-v3-cubic-fused-noiseless-64-128-256-30.json.gz)
passes 4/6. All four version-3 runs verified that the archived source hash was
unchanged from start to finish.

## Quadratic smoothness in CGLS

`CGLSConfig.gradient_damping` adds squared adjacent-voxel differences divided
by physical voxel spacing to the CGLS objective. Its square is the penalty
weight, alongside the existing squared scalar damping weight. Free boundaries
omit edges outside the volume, so constants have zero gradient penalty. The
matched normal operator and its absolute-weight roundoff bound are included in
the true-residual convergence checks. The zero default adds no gradient-penalty
operator work. This is an optional prior, not evidence of better pose recovery;
it can bias object shape and geometry estimates.

Explicit FP64 edge matrices check the energy, normal operator, cancellation
bound, anisotropic spacing and singleton axes. Dense augmented least-squares
systems check zero/warm starts, both penalties and all three ray/linear-Joseph/
cubic-Joseph choices on JAX and CUDA. Multiresolution also reaches the same
physical fine-grid regularized solution.

The focused CUDA run passed **49 tests, one expected skip under Compute Sanitizer
with zero errors**. The later multiresolution regression is included in the
full run: **669 GPU tests passed with two expected skips**. Focused CPU checks
passed **24 tests with eight CUDA skips**. Lint/format now cover all benchmark
scripts in CI; configured type checks, public import boundaries and lockfile
validation pass. Wheel/sdist metadata, installed-wheel CLI and an isolated
installed-wheel cubic/normal-equation/regularized-CGLS API check also pass.
The validated source snapshot is
`ea6e7d5cbee4dfd648d7308edbb67729a3199e1ae76bfde7afa8fee6aa063a02`.

Exploratory joint voxel/pose runs use independent noisy nine-Gaussian data,
centroid-based initialization and gauge-aware diagnostics. They have not yet
met the pose gate: clipping an unconstrained reconstruction produced residual
oscillation, while unconstrained continuation increasingly fitted noise.
Quadratic regularization stabilized volume quality but retained an incorrect
pose solution. These trials are diagnostics, not a successful-recovery time
baseline or the 20x joint-alignment target.

## Cubic unit-row adjoint specialization

The exact unit-row gather specialization now also supports cubic interpolation
and absolute weights. It computes vertical interpolation weights once per view
and voxel, outside the horizontal footprint loop, preserving their signs and
accumulation order. Dispatch tests prepared coefficients exactly: fractional
row offsets are supported, while tiny tilts and non-unit row scales retain the
general path. Independent physical-ray matrices cover both interpolation
models, reversed rows, mixed axes, fractional offsets, long rows and dynamic
transitions between specialized and general geometries.

An exploratory paired 12-case comparison gives 1.41–2.22x faster cubic adjoints
on parallel/anisotropic cases, with bit-for-bit matching outputs. The two
largest tilted controls regress about 2.3%; those measurements are retained.
The formal [cubic derivative sweep](../bench/reference/joseph-derivatives-v3-cubic-unitrows-32-64-128-256-60.json.gz)
passes **12/12 numerical checks and 9/12 derivative-cost gates**, up from 2/12
cost gates previously. Projection plus volume/pose gradients takes 2.48x
projection cost at size 256 parallel and 2.60x anisotropic. Tilted ratios at
sizes 32/128/256 remain 6.17/3.74/3.88x. The
[linear control](../bench/reference/joseph-derivatives-v3-linear-cubic-unitrows-control-32-64-128-256-60.json.gz)
passes 12/12 numerical and 7/12 cost gates. All synchronized samples and
compilation costs remain in the records; these are resident component
measurements, not complete reconstruction or process-memory comparisons.

Maximum cubic full-gradient relative discrepancy is `1.49e-6`, and maximum
norm-scaled directional discrepancy is `5.84e-9`. Focused production checks
passed **104 tests under Compute Sanitizer with zero errors**; the full CUDA
suite passed **690 tests with two expected skips**. Type/import checks and
format validation pass. Wheel/sdist metadata, installed-wheel CLI and isolated
CPU API checks pass. Both formal sweeps and the full suite verified unchanged
source `f9cbfddacd6145692c16f52e6937e75b5cf8b36146397a47e1b7a85fd26050a8`,
archived locally with the measured scripts.

## Restricted joint recovery controls

An exploratory unknown-volume workflow starts with measured centroids and
nominal rotations, reconstructs with 24 cubic CGLS iterations, detects Gaussian
features, fits per-view ellipses, jointly refines shared feature centers and
view poses, then reconstructs with 16 regularized CGLS iterations. Truth enters
only the final diagnostics, which remove one common rigid object frame and
sample continuous truth in that frame without amplitude fitting. This method
assumes isolated smooth features; it is not a general alignment algorithm.

The frozen development sweep uses the fixed nine-Gaussian object, sizes
64/128/256, 180 views, parallel/tilted scans, seeds 9345/128904/61937, noise
0.005 times clean RMS and true perturbations within ±3 degrees / ±10 pixels.
It passes **6/18 cases**: one at 64, all three at 128 and two at 256, all parallel.
Every tilted case fails. The accepted cases satisfy both the 99% per-view pose
gate and volume-error gate. Size-128 cases take 9.04–9.20 seconds cold and
3.74–3.83 seconds warm including quality verification, with 1.62–1.71% relative
volume error. There is only one warm repeat; fixture generation and process
startup are excluded. These are exploratory timings, not headline benchmarks.

The retained local record is `.artifacts/pose-recovery/feature-suite-v1.json`;
its scripts and production source are archived with combined hash
`2e0d256f7b7fa79f8813c338dfae576251de53e9392e0aa7f16f39e9b063831a`.
FP64 ellipse fitting does not resolve a tilted failure. A separate full-image
fit of unknown 3D Gaussian amplitudes, centers, covariances and all poses passes
one size-128 tilted case: all 180 views meet the pose gate, maximum angular
error is 0.00603 degrees and model-volume error is `4.95e-5`. That control assumes
the entire object is a Gaussian mixture. It motivates further work but does
not establish general voxel recovery, robustness on other objects, or the 20x
joint-recovery target.

The subsequent frozen full-image Gaussian-model sweep completes all 18 cases,
with **18/18 passing in both a cold and one warm run**. Initial object parameters
come from detected features in a blind 24-step voxel reconstruction; all Gaussian
centers, amplitudes and full covariances are then fitted jointly with every view
pose. The resulting volume is sampled from that fitted model. The same sizes,
180 views, three seeds, noise, perturbation ranges, pose and volume gates apply.
At least 179/180 views pass in every run; the largest individual angular error
is 0.01076 degrees in a size-64 tilted scan, so this is not an all-views claim.
Maximum model-volume relative error across the sweep is `9.75e-5`.

| Size | Parallel cold / warm | Tilted cold / warm |
|---|---:|---:|
| 64 | 4.68–6.32 / 0.61–0.66 s | 4.94–5.02 / 0.77–0.85 s |
| 128 | 7.09–7.21 / 3.08–3.15 s | 8.37–8.51 / 4.44–4.51 s |
| 256 | 24.61–24.69 / 17.83–17.85 s | 34.06–34.72 / 27.19–28.02 s |

Times include gauge and image-quality verification but exclude fixture generation
and process startup. A single warm repeat is insufficient for a headline speed
claim. These results establish a successful **restricted development baseline**,
not general voxel alignment, real-data robustness, or the 20x target. The local
record `.artifacts/pose-recovery/gaussian-joint-suite-v1.json` and its complete
source archive use hash
`398c52b7ae4dafebd238aa9b0d620c1d2efa5da09d83ebc82649dc88432a6f60`,
verified unchanged throughout the sweep.

An optimized prototype caps the initial voxel grid and fitted detector subset
at size 64, uses eight initial CGLS steps, and evaluates up to 180 views per
normal-equation batch. Only the large image Jacobian and its Gram matrix use
FP32; predictions, residuals, model/pose chain rules and the shared Schur solve
retain FP64. A CUDA kernel samples the final fitted Gaussian model. These
changes preserve the explicit Gaussian-mixture assumption and use no true
object parameters in reconstruction or alignment.

The frozen `gaussian-fast-suite-v1` sweep passes all 18 development cases plus
the **512³ / 720-view parallel motion-recovery case**, in a cold and one warm
run each. The large case takes **7.41 seconds cold** for recovery and independent
verification, **7.91 seconds including process startup**, and **1.99 seconds
warm**. Sampled process GPU memory peaks at **1688 MiB**. All 720 views pass:
maximum angular error is 0.00719 degrees, maximum translation-vector error is
0.00845 pixels, and model-volume relative error is **0.00419%**. Fixture generation
is excluded. One warm repeat is preliminary evidence, not a repeated performance
estimate or a general motion-recovery claim. The archived source hash is
`9790b0b28c270a6b48350b85610dabbbd24b7567686e0631dd9a64c4af244b70`.

This sweep uses an independent bounded FP64 CUDA truth evaluator. It agrees
with both retained FP64 host implementations, including nontrivial rigid gauges,
partial slabs and blocks; its focused Compute Sanitizer check reports zero
errors. It shares no model-sampling kernel with the fitted reconstruction.
Its verification cost differs from the earlier host check, so the two sweeps'
complete times must not be used to claim a solver speedup.

The subsequent `gaussian-paired-v2` comparison reruns both solvers with that
same verifier, one cold run and seven warm repeats in isolated processes. All
**288 runs across 18 paired cases pass**. Warm geometric-mean speedup is
**10.19×**; cold speedup including startup is **2.13×**. Neither establishes the
20× target. Per-case warm medians, with ranges across the three fixed seeds:

| Size / geometry | Baseline warm | Optimized warm | Baseline / optimized peak GPU memory |
|---|---:|---:|---:|
| 64 / parallel | 0.614–0.629 s | 0.245–0.250 s | 416 / 1186 MiB |
| 64 / tilted | 0.744–0.809 s | 0.304–0.307 s | 416 / 1186 MiB |
| 128 / parallel | 2.569–2.622 s | 0.255–0.261 s | 672 / 1196 MiB |
| 128 / tilted | 3.893–3.934 s | 0.374–0.403 s | 672 / 1196 MiB |
| 256 / parallel | 15.099–15.133 s | 0.348–0.356 s | 2212 / 1248 MiB |
| 256 / tilted | 24.445–25.218 s | 0.596–0.605 s | 2212 / 1248 MiB |

The larger image-normal batches increase GPU memory at smaller sizes. These
are sampled per-process peaks over the complete worker, including compilation
and repeats; short-lived peaks may be missed.

The repeated **512³ / 720-view** showcase passes all eight runs as well. Cold
recovery plus verification is **6.33 s**, or **6.83 s including startup**. The
seven-run warm median is **1.949 s**, with samples from 1.934 to 1.986 s. All
720 poses pass on every run; volume error and the 1688 MiB GPU peak agree with
the preliminary result. The complete sweep has 296 accepted runs. Synthetic
fixture generation takes 138.65 s and is excluded from these recovery times;
its peak host memory is 2896 MiB after the streaming change. The entire worker
peaks at 4909 MiB of host memory. Both timings and memory scopes are retained
separately in the raw records.

This remains a restricted positive-Gaussian-mixture development result, with
three perturbation/noise seeds and one object specification. It does not
establish general voxel recovery or held-out robustness. The local record and
source archive are `.artifacts/pose-recovery/gaussian-paired-v2{,-source.tar.gz}`;
combined source hash
`a4117d89032061b3389268a100e8f36b7dbf9dde5f3af8f49c73317005708a8e`
was verified unchanged throughout the sweep.

## Bounded benchmark quality verification

The shared reconstruction benchmark now computes its physical full-volume L2
metric in bounded FP64 chunks. It retains every voxel and the same acceptance
thresholds, and rejects mismatched output shapes rather than broadcasting them.
Signed, noncontiguous, mapped and large-dynamic-range inputs agree with the
previous full-array calculation to `1e-14` relative in the checks. Nonfinite
values in later chunks are also rejected.

In a seven-repeat CPU-only probe, verification takes 4.52/27.07/214.11 ms at
128/256/512 cubed, versus 7.86/79.14/640.80 ms for full-array temporaries. This
improves the measurement harness equally for every reconstruction method; it is
not a TomoJAX solver speedup. Existing complete-workflow records retain their
original verification costs. Fresh comparative runs must use the updated check
for both TomoJAX and competitors. The focused benchmark tests pass **26 cases
with 17 CUDA-only skips** on CPU. The archived source hash is
`f4bd941eb9594383a1effec23ec0802dac1ea1714cc0c8d2ea655973ebfb53e6`.

## Fourier host-copy overlap

Large CUDA Fourier calls now prepare the next input slab and write the previous
output slab while the calling thread reconstructs the current slab. The host
queues are bounded; small jobs and the NumPy reference remain sequential. No FFT,
interpolation, physical normalization or public configuration changed.

The prototype passes the existing 60 numerical checks with bit-for-bit identical
output. In a paired seven-repeat probe, 512³ / 720-view reconstruction decreases
from 730.53 to 478.32 ms; including the same bounded host quality check, 954.52
to 702.16 ms. At 256³ / 720 views the corresponding verified medians are 139.31
and 103.80 ms. Thread overhead slows the smallest case, so production enables
overlap only for multiple-slab CUDA jobs with at least 64 MiB of estimated input
and output host work. These prototype figures are not fresh external comparisons.

Production validation passes **702 GPU tests with two expected skips**, and all
**64 focused Fourier tests under Compute Sanitizer with zero errors**. New checks
cover partial slabs, strided input/output, mapped output and a late input error
after completed output slabs. Types, formatting, import contracts and public
import checks pass. The frozen source hash is
`9c760dd51607fd2f50ea593e5ea44df55bd6709725abb532b1f256d2db6b19a2`;
the complete source archive is retained locally. The wheel and source package
pass metadata checks; isolated installed-wheel CLI and physical API smoke checks
also pass on CPU.

All four refreshed comparison sweeps completed with unchanged source. They use
the same bounded quality check for every method, include process startup and
fixture loading in cold time, and retain seven warm samples. Large-case results:

| Case | Method | Cold complete (ms) | Warm complete (ms) | Peak GPU (MiB) |
|---|---|---:|---:|---:|
| 256³ / 720 | TomoJAX Fourier | 2098.1 | 104.1 | 248 |
| 256³ / 720 | CuPy/ASTRA | 1186.9 | 428.8 | 814 |
| 256³ / 720 | ASTRA native 2D | 1166.8 | 527.6 | 148 |
| 256³ / 720 | TIGRE FBP | 1914.6 | 954.7 | 218 |
| 512³ / 720 | TomoJAX Fourier | 2673.1 | 679.2 | 582 |
| 512³ / 720 | CuPy/ASTRA | 4161.3 | 3078.0 | 1262 |
| 512³ / 720 | ASTRA native 2D | 4112.5 | 3101.3 | 156 |
| 512³ / 720 | TIGRE FBP | 6766.3 | 5451.2 | 734 |

TomoJAX passes the same independent physical gate at both sizes; 512³ error
remains **0.015018%**. Warm speedups over the fastest accepted external workflow
are **4.12× / 4.53×**, with 30.5% / 46.1% of its sampled GPU memory. Native
ASTRA 2D uses substantially less memory, and TomoJAX's cold 256³ result is slower
than the fastest competitor. Earlier records use their original quality-check
costs and must not be mixed into these new ratios.

On the standard 180-view Gaussian suite, TomoJAX's parallel warm times are
3.39/11.31/69.22 ms at sizes 64/128/256, and anisotropic times are
2.08/9.91/47.05 ms. All six pass, but warm geometric-mean speedup is **1.70×**
and cold geometric-mean speedup is **0.89×**. The >=2× per-case and 10× overall
targets remain unmet. All six noisy cases pass; sharp anisotropic-64 still
fails at 17.80% error against its unchanged 15% gate. Unsupported competitor
geometries and all failures remain in the records.

See the complete [large](../bench/reference/fourier-host-pipeline-v1-gaussian-256-512-720.json.gz),
[Gaussian](../bench/reference/fourier-host-pipeline-v1-gaussian-64-128-256-180.json.gz),
[sharp](../bench/reference/fourier-host-pipeline-v1-sharp-64-128-256-180.json.gz) and
[noisy](../bench/reference/fourier-host-pipeline-v1-noisy-64-128-256-180.json.gz) results.

## Fourier asynchronous transfers and bounded stream reuse

Large CUDA reconstructions now overlap upload, computation and download using
pinned host buffers and at most two pending slabs. Device-specific queues are
reused across calls; each call still owns its mutable working buffers. Small
immutable geometry arrays have a bounded cache. Fractional host-row preparation
reuses temporary arrays while preserving its FP32 arithmetic, including signed
zeros. The FFT padding, interpolation and physical normalization are unchanged.

An initial asynchronous version created new streams on every call. Its outputs
passed, but CuPy retained separate allocator arenas, causing repeated-call GPU
memory growth. All 80 comparison records from that version remain retained.
A fresh paired diagnostic confirms that queue reuse holds allocator storage
flat across eight calls; measured 512³ process peak falls from 3892 to 678 MiB.
The new memory regression test fails on the original behavior and passes with
queue reuse. Concurrent pipelined calls retain independent outputs, and error
paths drain outstanding transfers.

Final production validation passes **709 GPU tests, two expected skips**, and
**71 focused Fourier tests under Compute Sanitizer with zero errors**. Types,
lint, formatting, import contracts, package metadata and isolated installed
CPU API/CLI checks pass. Source and benchmark hash:
`aade9e680807570dfeb1cfc1841f7bf405a0d40d1754ca078a61e1a60827a14c`.
All 80 refreshed external records completed with that source unchanged: 60
accepted, two failed quality gates and 18 uncovered adapter comparisons.

Cold times include process startup and fixture loading. Each warm result is
the median of seven complete solves with the common independent FP64 verifier.
All methods use the same host data and physical gates; no fitted amplitude or
clipping is applied. The default 16-slice results are:

| Case | Method | Cold complete (ms) | Warm complete (ms) | Peak GPU (MiB) |
|---|---|---:|---:|---:|
| 256³ / 720 | TomoJAX Fourier | 885.3 | 88.0 | 284 |
| 256³ / 720 | CuPy/ASTRA | 1098.0 | 432.1 | 814 |
| 256³ / 720 | ASTRA native 2D | 1082.4 | 512.2 | 148 |
| 256³ / 720 | TIGRE FBP | 1788.8 | 939.5 | 218 |
| 512³ / 720 | TomoJAX Fourier | 2509.1 | 517.8 | 678 |
| 512³ / 720 | CuPy/ASTRA | 4103.0 | 3084.8 | 1262 |
| 512³ / 720 | ASTRA native 2D | 4089.2 | 3142.8 | 156 |
| 512³ / 720 | TIGRE FBP | 6614.1 | 5412.1 | 734 |

The two large warm speedups are **4.91× / 5.96×**. GPU memory is 34.9% / 53.7%
of the warm-fastest external workflow, so default 512³ misses the half-memory
target. The independent 512³ error remains 0.015018%. Native ASTRA 2D uses much
less GPU memory. Earlier timing ratios remain tied to their own source and
measurement records.

The six standard Gaussian cases all pass. Warm parallel times at sizes
64/128/256 are 3.07/10.95/71.81 ms; anisotropic times are 1.37/7.92/41.55 ms.
The geometric-mean warm speedup is **1.92×**, and cold speedup **0.88×**.
Three of six cases reach 2× warm speedup; the overall targets remain open.
All six noisy cases pass. Sharp anisotropic-64 still fails at 17.80% error
against the unchanged 15% gate.

See the complete [large](../bench/reference/fourier-async-pipeline-v3-gaussian-256-512-720.json.gz),
[Gaussian](../bench/reference/fourier-async-pipeline-v3-gaussian-64-128-256-180.json.gz),
[sharp](../bench/reference/fourier-async-pipeline-v3-sharp-64-128-256-180.json.gz) and
[noisy](../bench/reference/fourier-async-pipeline-v3-noisy-64-128-256-180.json.gz) records.
The additional historical file named `fourier-async-pipeline-v3-eight-slices`
actually used 16 slices, as recorded in its per-run metadata: `--batch` controls
iterative view batching. It is retained as a repeat, not an eight-slice result.
The subsequent explicit `--fourier-slices` benchmark option has separate
argument, worker-propagation and resume-compatibility checks.

The [explicit eight-slice comparison](../bench/reference/fourier-explicit-eight-slices-v1-256-512-720.json.gz)
also completes all eight method/case records, with every quality gate passing:

| Case | Fourier cold (ms) | Fourier warm (ms) | Fourier peak (MiB) | Fastest external warm (ms) | Its peak (MiB) |
|---|---:|---:|---:|---:|---:|
| 256³ / 720 | 775.6 | 87.2 | 220 | CuPy/ASTRA: 431.2 | 814 |
| 512³ / 720 | 2600.9 | 615.0 | 430 | Native ASTRA 2D: 3072.2 | 156 |

Use `FourierConfig(slices_per_batch=8, backend="cupy")` for this configuration.
At 512³ it trades speed for lower memory. CuPy/ASTRA takes 3077.5 ms and uses
1262 MiB in this same sweep; native ASTRA 2D is narrowly faster, so the strict
half-memory target against the fastest method remains unmet. The default stays
at 16 slices. This follow-up changes only benchmark configurability, with
27 benchmark-contract tests passing and unchanged library code. Its source and
benchmark hash is
`59c1e6c00ae6bb035cb3f2b2e9205c7a60b07d1c7cdbaa4bc04ce8db9b2a0838`.

## Free-voxel joint recovery development controls

A separate prototype now fits independent nonnegative voxels and all five pose
parameters per view. Its only object regularization is the squared physical
Laplacian; it does not fit a Gaussian-mixture model. Initialization uses measured
centroids, nominal rotations and 24 cubic CGLS steps. A fixed policy of 12 joint
Gauss–Newton updates, 80 inner iterations and curvature weight 3 passes three
blind 128³ / 180-view development controls:

| Geometry / seed | Pose views passing both gates | Angular RMSE (degrees) | Maximum angular error (degrees) | Volume relative L2 |
|---|---:|---:|---:|---:|
| Parallel / 9345 | 180/180 | 0.00393 | 0.00950 | 0.410% |
| Parallel / 128904 | 180/180 | 0.00349 | 0.00778 | 0.407% |
| Laminography / 9345 | 180/180 | 0.00348 | 0.00883 | 1.924% |

These use the existing nine-Gaussian measurement object, 0.5% RMS noise and
nominal initialization with unknown perturbations up to ±3° / ±10 pixels.
Truth enters only verification after each update, with one shared rigid gauge.
The local record is `.artifacts/pose-recovery/positive-curvature-blind-v1.json`.
The 73–82 second exploratory runs include repeated truth diagnostics and are
not formal workflow timing results. Three development cases do not establish
held-out robustness, arbitrary-object accuracy or the 20× joint speed target.
Earlier failed priors and optimization controls remain recorded. Preconditioning
and a common final-verification benchmark are under development.

The subsequent native-grid extension is mixed. At size 64, the same fixed
12-update / 80-inner-iteration policy with the predeclared physical scaling of
the curvature weight passes only 49.4% of parallel views and 96.1% of tilted
views for seed 9345. Doubling the internal grid improves the parallel result
to 82.8%, which still fails the 99% gate. These failures are retained.

At size 256, parallel recovery passes all 180 views in both cold and warm runs,
with 0.189% volume error and maximum angular error 0.00506°. Complete recovery
plus the common independent final verifier takes 595.5 s cold and 608.2 s warm,
excluding fixture generation and reported process startup. Tilted recovery
takes 669.3 / 667.7 s and yields 3.92% volume error, but fails the pose gate at
177/180 views in both runs. Sampled peak process GPU memory is 2230 MiB for
both geometries. These slow development results motivate preconditioning and
multiresolution work; they do not meet the joint-workflow performance target.
The completed local record is `.artifacts/pose-recovery/voxel-resolution-controls-v1.json`.

A coarse-to-fine follow-up caps the joint solve at 128 voxels per axis, adds
an affine-mode preconditioner, and uses 40 inner iterations. It preserves
measured-ray coordinates when subsampling the detector, then performs eight
fixed-pose curvature-regularized iterations against the full native data.
The object still consists of independent nonnegative voxels. At 256³ with
180 views and seed 9345, both geometries pass all 180 pose checks:

| Geometry | Cold complete (s) | Warm complete (s) | Maximum angular error | Volume relative L2 | Sampled GPU peak (MiB) |
|---|---:|---:|---:|---:|---:|
| Parallel | 53.25 | 43.01 | 0.00831° | 0.245% | 2336 |
| Laminography | 57.93 | 47.58 | 0.00953° | 3.283% | 2336 |

These include the same independent final verifier and exclude fixture
generation and separately recorded process startup. Parallel's one warm
sample is 14.14× faster than the accepted native-grid baseline. The tilted
native baseline fails acceptance, so it cannot supply a successful-workflow
speedup. These are exploratory comparisons, with more seeds and repeated
measurements still required. The frozen follow-up record is
`.artifacts/pose-recovery/resolution-followups-v1.json`, source hash
`773289eb2fbf3a12065251a20874a00217845c8cbb02a3a5705fce0b5019cea7`.

The same follow-up retains unsuccessful size-64 padding, oversampling and
prior-strength controls. Noiseless oversampled parallel recovery passes;
the corresponding noisy control fails. A separate compact affine-mode
implementation removes a size-256 allocation failure by constructing volume
modes on demand, but its 20-inner-iteration tilted control still fails the
pose gate. None of these experimental solvers is a public default.

The completed three-seed extension (9345, 128904, 61937) runs both 20- and
40-inner-iteration budgets at 256³ / 180 views. Each case has one cold and one
warm call, both checked against the same independent physical reference:

| Inner iterations | Geometry | Accepted seeds, cold and warm | Warm complete range (s) | Volume relative L2 range |
|---|---|---:|---:|---:|
| 20 | Parallel | 3/3 | 27.29–27.54 | 0.244–0.245% |
| 20 | Laminography | 0/3 | 29.94–30.13 | 3.708–4.218% |
| 40 | Parallel | 3/3 | 43.30–43.52 | 0.244–0.245% |
| 40 | Laminography | 3/3 | 47.56–47.63 | 3.070–3.401% |

At 20 inner iterations the tilted cases pass only 92.2–96.1% of poses; all
three fail the 99% criterion despite acceptable volume errors. At 40, all
three tilted cases pass every pose. The parallel seed-9345 warm control is
22.29× faster than its recorded accepted native baseline, but this is one
warm sample, not a repeated performance result. The remaining seeds lack
measured accepted native baselines. The local record is
`.artifacts/pose-recovery/multires-budget-followups-v1.json`, source hash
`bd1707732013ebec4461d529066ccefbc608de3b3befff1ce05ad076219e09de`.

For size 64, a separate oversampled-voxel experiment penalizes the squared
physical gradient of the Laplacian. An independent dense edge-incidence
calculation checks the resulting coupled system. Of three predeclared prior
weights (12, 48, 192 at the 128³ reference geometry), weight 48 passes all 180
parallel poses for seed 9345 in both calls, with 0.189% volume error and
12.49 s warm complete time. The other parallel weights and all three tilted
controls fail. This selected development setting still needs other seeds and
objects; it does not establish general low-resolution recovery.

## Restricted joint fitting with normalized physical units

A further experimental positive-Gaussian-mixture fitter replaces the initial
iterative reconstruction with a filtered cubic Joseph transpose and normalizes
all physical lengths and line integrals together before fitting. It restores
physical units before sampling the final volume; amplitudes remain densities.
An independent analytic-ray check verifies the unit change over five scales.

All 18 development cases (three sizes, two geometries, three seeds) pass one
cold and one warm call. Tilted fits take eight iterations at each size,
compared with 8 / 12 / 19 without normalization. Warm complete times, including
the independent final verification, span:

| Size / 180 views | Parallel (s) | Laminography (s) |
|---|---:|---:|
| 64³ | 0.102–0.115 | 0.153–0.161 |
| 128³ | 0.119–0.127 | 0.169–0.174 |
| 256³ | 0.207–0.215 | 0.252–0.257 |

These controls are exploratory and assume an unknown positive Gaussian-mixture
object. They do not establish a repeated 20× speedup or arbitrary-voxel
performance. The local record is
`.artifacts/pose-recovery/normalized-fbp-followups-v1.json`, source hash
`802c76935aa95cd24a1bcb9e1c669ccec2d6d0cde81178904b1dac51f393fd4a`.

## Views shared among several GPUs (2026-10-08)

`devices=` gives each GPU a share of the views and the whole volume, and sums
their backprojections once per projector call. On four H100 80GB GPUs in one
machine (Modal), 20 non-negative FISTA iterations on the binned FIPS walnut
(three orbits, every fourth view: 900 views of 486 × 384 pixels, 501³ voxels)
took, on the second call of each:

| GPUs | Time (s) | Speedup | Volume vs one GPU (relative L2) |
|---:|---:|---:|---:|
| 1 | 24.8 | 1.00 | — |
| 2 | 13.6 | 1.83 | 2.9e-7 |
| 4 | 7.4 | 3.34 | 2.9e-7 |

First calls, which compile, took 27.4, 15.8 and 9.5 s. The difference from one
GPU is the order of the cross-device sum. Each GPU still runs the solver's
volume-sized updates and the power iteration's setup, which do not shrink with
more GPUs, and the 0.5 GB volume sum crosses NVLink once per iteration. ASTRA
and TIGRE were not run on this machine, so these numbers are not a comparison
with them.
