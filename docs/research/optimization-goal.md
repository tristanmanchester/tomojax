# TomoJAX stretch performance goal

Accepted 2026-10-03. These are targets, not measured capabilities or delivery
forecasts. The goal remains open until reproducible evidence meets every target.

| Outcome | Target |
|---|---|
| Complete reconstruction, equal accepted quality | 10x geometric-mean speedup over the fastest applicable ASTRA/TIGRE workflow; at least 2x in every supported comparison case |
| Forward projection and matched adjoint | 5–10x faster at matched physical accuracy, including large tilted and irregular volumes |
| Peak process GPU memory | At most half the fastest accepted competing workflow's peak, including runtime/allocator overhead |
| Joint reconstruction/alignment | 20x faster than the recorded TomoJAX baseline through successful recovery |
| Observable pose recovery | Translation RMSE at most 0.05 native detector pixels; rotation-matrix angular RMSE at most 0.01 degrees |
| Recovery robustness | At least 99% success on a declared identifiable noisy distribution including initialization errors up to ±10 pixels and ±3 degrees |
| Differentiation | Forward plus volume/pose gradients at most 3x forward cost; differentiation memory bounded as iteration count increases |
| Startup | Less than 1 second additional startup for cached standard workloads; less than 5 seconds compilation for new supported shapes |
| 8 GB GPU demonstration | 512 cubed voxels, 720 views: accepted reconstruction under 10 seconds, including motion recovery under 30 seconds |

## Rules for evidence

- Fix case definitions, masks, units, noise seeds, acceptance thresholds, and
  iteration/time limits before tuning on a suite version. Changing a definition
  creates a new suite version; retain unsuccessful and superseded results.
- The primary metric is synchronized elapsed time to an accepted result. Include
  setup, host/device transfers, compilation, stopping checks, and verification.
  Report fresh-process cold and repeated calls separately. A warm kernel result
  cannot satisfy the complete-workflow target.
- Use independent analytic or independently simulated data and volume truth;
  do not generate the comparison data with a competing reconstruction operator.
  Add structured phantoms, real data, noise, and held-out scans before declaring
  broad success. Smooth Gaussian cases are only the first measurement slice.
- Report time-to-quality and algorithm-matched comparisons separately. Compare
  the fastest accepted applicable external method, with its settings documented.
  Expose differences in integration model, regularization, constraints, and
  discretization instead of matching raw iteration counts.
- A timeout, nonfinite result, OOM, or missed quality threshold is a failure.
  An unavailable comparison is uncovered. Neither can count as a speed win.
- Alignment needs an alignment baseline, gauge-aware observable geometry errors,
  an identifiability check, and image quality. Existing short-iteration alignment
  records do not yet establish a successful-recovery time baseline.
- Sample peak process VRAM in isolated workers. Report polling cadence and its
  limitations; compiler buffer estimates are a separate diagnostic.
  Select each cell's competing workflow by its fastest accepted fresh-process
  time, then use that same workflow for the memory comparison. A different
  warm-fastest or lower-memory competitor is not an interchangeable denominator.
- Report every case and timing sample, including regressions, and machine/software
  metadata. Use at least seven repeats for headline timings and retain variability.
  Avoid simultaneous GPU workloads during comparisons.

## Initial reconstruction comparison: gaussian-v1

This initial suite is a deliberately limited measurement slice, not full evidence
for the goal. It uses `bench/compare_projectors.py:make_case`: two asymmetric
Gaussian ellipsoids, independent analytic line integrals, FP32 measured data,
parallel / 30-degree laminography / anisotropic shifted odd-detector geometries,
sizes 64, 128, 256, and 180 views. The full-volume relative L2 target is **0.03**
for parallel and anisotropic cases and **0.10** for laminography. No amplitude
fitting, cropping, or post-hoc threshold relaxation is permitted. Check quality
at fixed budgets 1, 2, 4, 8, 16, 32, 64, 128, 256 iterations. A method that misses
the target through 256 iterations fails this slice. Report actual effective data
passes where an iteration is a subset update.

Use the same host measurements, zero initialization, physical geometry, and full
volume evaluation for all methods. Start with unregularized least squares to
isolate operator/solver differences. Additional noisy/regularized and structured
phantom suites must receive their own versioned definitions before optimization.

The first large reconstruction check uses the same `gaussian-v1` parallel
phantom and 0.03 full-volume relative L2 gate at 256/512 cubed with 720 uniform
half-turn views. Compare corrected single-pass TomoJAX Pallas, CuPy/ASTRA,
native ASTRA 2D and TIGRE FBP, with seven repeats, cold complete time and sampled
process memory. This additional view count is recorded explicitly and is not
the original 180-view standard suite. It addresses only the reconstruction
part of the 8 GB demonstration; motion recovery and broader image quality
remain separate requirements.

## Status

### Summary, 2026-10-05

- Reconstruction: over 26 paired cells, TomoJAX's fastest accepted workflow is
  2.6x faster warm than the fastest accepted ASTRA or TIGRE workflow (geometric
  mean; worst 0.75x) and 0.95x cold (worst 0.40x). The 10x target is unmet. See
  [the latest matrix](system-matrix-2026-10-05.md).
- FBP is exact for any circular parallel scan, including tilted half turns.
  FISTA-TV and SPDHG-TV now run on the batched Joseph/Pallas operators
  (20-120x faster than before).
- Joint recovery: the default `tomojax align --mode pose` recovers 5/6 pilot
  cells (rotation 0.0006-0.008 deg), but the pilot integrates the voxel basis
  exactly, the solver's own model. On analytic data from continuous objects,
  every solver settles at a 0.09-0.5 deg discretisation floor at 32 cubed, even
  when started from the true poses; the 0.01 deg gate is not a meaningful target
  for real data at this scale. Reconstructions match true-pose ones. +/-1 deg / +/-2 px motion is recovered in all three
  tested geometries; +/-3 deg / +/-10 px is not, so the 99% robustness target
  is unmet.
- Per-iteration cost on tilted scans remains 2-3x ASTRA's: the matched Joseph
  backprojection is limited by its enumeration loop, and three rewrites
  (compact tiles, exact row intervals, a CUDA fixed-point splat) did not help.

The detailed record below is chronological and retains every experiment.

The [Huber-TV derivative correction](huber-derivatives-2026-10-04.md) removes
NaNs in flat-region and zero-dual derivatives. All 33 frozen image inputs retain
bitwise-identical updates with finite corrected derivatives. This is a
correctness fix, not a new workflow speedup; the zero-TV comparison baselines
below remain unchanged. Sampling-boundary pose finite-difference failures are
retained and documented.

The [2026-10-04 corrected whole-matrix report](system-matrix-2026-10-04.md)
retains all 297 method records, including 25 quality failures and 72 unsupported
adapters. Accepted comparisons cover 26/27 cells; the sharp anisotropic 64 cell
has no accepted method from either library. The overall geometric mean is
undefined, and the worst measured accepted-pair cold ratio is 0.21×.
The [whole-matrix diagnostic profiles](system-profile-2026-10-04.md) separate
compilation from CUDA execution without substituting instrumented timings for
the score. They do not yet justify kernel-specific tuning.

The [public free-voxel baseline attempt](public-free-voxel-baseline-2026-10-04.md)
completed all 48 calls across six irregular-angle clean/noisy cells. All missed
the rotation gate, so successful-recovery time and the 20× denominator remain
undefined. Its frozen source retains the subsequently discovered step-bound
inflation bug. Independent known-volume diagnostics also found ray-integration
bias in all three geometries; oracle-volume successes cannot replace joint recovery.

The [exact-projector public rerun](public-free-voxel-exact-2026-10-04.md)
completed another 48 calls with identical fixture arrays and fixed gates. All
six cells still missed rotation accuracy; successful-recovery time remains
undefined. The reconstruction matrix and its scores are unchanged.

The subsequent [coupled free-voxel experiment](public-free-voxel-joint-2026-10-04.md)
completed all 48 calls: five cells passed all repeats, while anisotropic-noisy
failed rotation in all eight calls (about 0.016 degrees against 0.01). It supplies
accepted timings for those five cells, not a complete successful six-cell
baseline or a 20x speedup denominator. The stacked-PCG variant is not being
tuned further or made the default. The [research-informed test plan](research-test-plan-2026-10-04.md)
separates accuracy limitations and targets shared reconstruction/alignment
acceleration across both matrices. The [accuracy diagnostics](alignment-accuracy-2026-10-04.md)
find central-stencil error and inadequate inner solves. Plain pose elimination
with the existing volume preconditioner does not improve the tilted linear
residuals at the fixed 40-iteration cap, but a
[nonlinear follow-up](shared-conditioning-2026-10-04.md) improves those pose
steps. Residual norms alone cannot decide its end-to-end value. The subsequent
[analytic-Jacobian comparison](public-free-voxel-analytic-2026-10-04.md) completed
all 48 calls with the same 5/6 acceptance pattern; noisy anisotropic still fails
at about 0.016 degrees. That derivative-only variant is also stopped. Shared
volume conditioning was then tested across all 27 reconstruction cells. The
[complete spectral comparison](system-matrix-spectral-2026-10-04.md) passed only
9/54 method/cell combinations, compared with 49/54 for the original two workflows.
All workers and seven warm calls per worker completed. The candidate is rejected
and withdrawn; its time, quality, memory and source remain archived. The separate
[exact pose-elimination comparison](public-free-voxel-schur-2026-10-04.md) also
completed all 48 calls with unchanged damping and budgets. It recovers five cells,
cuts accepted tilted cold time about 2.5×, and raises peak memory from 280 to
320 MiB. Noisy anisotropic still fails all eight calls at about 0.0162°, so the
complete recovery baseline remains missing. That elimination-only variant is
frozen without further tuning. No whole-matrix goal score improves from these
partial or rejected candidates.

The subsequent [pilot-scale noise diagnostic](pilot-noise-2026-10-04.md)
verifies undamped volume elimination for all three geometries, including an exact
structural reduction and iterative refinement for laminography. The actual
anisotropic noise predicts 0.01736° rotation error, close to the observed
0.01623°. This is a local unconstrained calculation at truth, not a successful
recovery, a universal noise floor, or evidence of 99% robustness. It motivates
testing an image prior before further solver-tolerance tuning; no goal score
or acceptance gate changes.

The [default-weight TV screen](public-free-voxel-tv-2026-10-04.md) completed all
six cells and failed all rotation gates; that variant was rejected. The
[whole-system cache screen](system-cache-2026-10-04.md) completed 135 workers
across the 27 reconstruction and six alignment cells plus changed-data controls.
Cache reuse helps JAX startup but leaves both quality failures and Fourier
performance unchanged. The configuration-only result does not close a goal
or justify further tuning of that subset.

The [33-cell model-discrepancy diagnostic](model-discrepancy-2026-10-04.md)
has completed. It distinguishes finite-budget recovery, projection-model error,
and null-space selection by preconditioning. Its matched-data and truth-start
controls are not accepted reconstructions and change no timing denominator or
gate. The next solver must pass underdetermined solution-selection checks as
well as the existing quality and workflow comparisons. No stretch target is
closed by these diagnostics.

The [first regularized filtered-Krylov prototype](hybrid-krylov-qualification-2026-10-04.md)
subsequently failed its small physical quality/storage qualification despite
passing independent algebra checks. That variant is stopped before integration;
no public workflow was timed and no acceptance result changed.

A subsequent physical-gradient GCV screen, recorded in the same report, improves
20 of 24 small controls but strongly worsens independent sharp tilted data. It
is also rejected before integration. Comparing only with fully converged least
squares would hide those failures; early-stopped controls remain in the record.

The [compiled-objective refactor](public-free-voxel-reuse-2026-10-04.md) completes
paired six-cell public runs, one cold and seven warm calls per revision and cell.
It removes warm backend compilation and improves all six warm medians, while
cold time and process GPU memory remain essentially unchanged. Both revisions
recover five of six cells; noisy anisotropic still fails every call. The 27
reconstruction cells have unchanged source and are not retimed for this refactor.
This bounded line is closed, with no new whole-matrix or successful joint target.

The [nonnegative held-out regularization qualification](nonnegative-cv-qualification-2026-10-04.md)
improves all 24 small controls against the earlier unconstrained Krylov baseline.
Its projected Newton-CG implementation matches only 22/24 dense solutions and
requires extreme operator counts in smooth clean controls. That implementation
is stopped before integration; neither workflow matrix is rerun or rescored.
The public alignment path already enforces positivity, so the small estimator
result does not establish improved pose recovery.

The [subsequent solver qualifications](solver-qualifications-2026-10-05.md)
reject an augmented nonnegative solver and a range-preserving detector-filter
preconditioner. An additional continuous stationarity check clarifies the former's
near-bound failures, but three dense-image disagreements remain. The latter
preserves null-space selection while worsening seven of 24 fixed-work image
controls. Both variants stop before public integration; the workflow scores and
successful-recovery baseline remain unchanged.

The goal applies to the complete scientific workflow and the whole frozen
comparison matrix. The scheduled reconstruction matrix is every combination of
`gaussian-v1`, `structured-v1`, `structured-noisy-v1`; parallel, anisotropic
shifted odd-detector, and 30-degree laminography geometry; and sizes 64, 128,
256 with 180 views. Keep failures and uncovered comparisons in the scorecard.
The separate 256/512-cubed, 720-view Gaussian controls do not replace these
27 cells. A geometric mean over only passing cells cannot establish the goal.

All Gaussian-mixture fitting results below are restricted development controls.
They count toward neither the 20x joint-recovery target nor the 99% robustness
target. The required successful baseline must execute the public alignment
workflow with independent free voxels, physical observable pose errors and
accepted image quality. That successful baseline is still missing. Component
speedups and independent voxel prototypes outside the public path do not fill
this gap.

Further optimization must follow whole-workflow profiles and extend to tilted,
irregular, sharp and noisy cases. Per-phantom dispatch, truth-dependent tuning,
and further tile/slab/copy micro-tuning without a measured shared bottleneck are
outside this workstream. Partial gains remain useful historical measurements,
but do not justify continuing a line that leaves the failing cells behind.

### Coarse-to-fine comparison policy, frozen before retained comparisons

The additional method `tomojax_multires_cgls_pallas` uses a coarse grid whose
largest dimension is at most 32: factor `ceil(max(nx, ny, nz) / 32)`, followed by
factor 1. At each total budget B >= 2 it assigns `floor(7*B/8)` iterations to
the coarse level and the remainder to the fine level. Budget 1 and grids already
at most 32 use a single fine solve. The same policy applies to all geometries,
including cases where coarse initialization loses time or quality. Each candidate
starts from zero; all preprocessing, interpolation and both solves are timed.
This policy was informed by exploratory Gaussian runs and is therefore a trained
policy on gaussian-v1, not independent evidence of generalization.

The held-out `structured-v1` and `structured-noisy-v1` suites retain the same
scan geometries, sizes, views and budgets but use exact line integrals through
five piecewise-constant ellipsoids. Truth is their density at voxel centres.
Centres and radii below are fractions of each grid's physical extent; angles
rotate local ellipsoid axes about world z. Inclusion amplitudes add:

| Amplitude | Centre | Radii | Angle |
|---|---|---|---|
| 1.0 | (0, 0, 0) | (0.28, 0.26, 0.30) | 0 degrees |
| -0.55 | (-0.10, 0.04, 0.03) | (0.09, 0.12, 0.10) | 23 degrees |
| 0.75 | (0.12, -0.08, -0.09) | (0.065, 0.055, 0.08) | -17 degrees |
| 1.0 | (0.09, 0.10, 0.13) | (0.03, 0.025, 0.035) | 0 degrees |
| -0.4 | (-0.04, -0.13, -0.12) | (0.045, 0.03, 0.05) | 31 degrees |

Full-volume relative L2 thresholds are 0.15 for parallel/anisotropic and 0.30
for laminography. The noisy version adds independent zero-mean Gaussian noise
with standard deviation 1% of the clean projection RMS, NumPy PCG64 seed 7019;
thresholds are 0.18 and 0.32 respectively. No clipping, fitted scaling, masks or
post-hoc adjustment is allowed. These are development gates, not claims of
clinical or application-specific adequacy. Report failure without relaxing them.

### Current evidence

The additional opt-in Fourier-slice reconstruction completes the 512-cubed /
720-view Gaussian case in 2.67 seconds cold and 0.679 seconds warm at 0.015018%
error, with 582 MiB sampled GPU memory. This is 1.54x cold / 4.53x warm versus
the respective fastest accepted external workflow. Standard 180-view Gaussian
warm speedup averages only 1.70x geometrically, with cold speedup 0.89x.
All 174 original, 87 real-FFT and 80 host-pipeline comparison records completed;
the sharp anisotropic-64 quality failure is retained. Large Fourier
memory is 46% of the warm-fastest competitor's peak, but small cases and the
cold-fastest native ASTRA 2D comparison still miss the overall memory target.
It does not provide motion recovery. These results and the independently
checked NumPy/CUDA API are documented in the performance report.

The Gaussian reconstruction component of the 8 GB showcase is demonstrated:
512 cubed / 720 views with host-output slabs in 3.91 seconds cold and 1.75 seconds
warm, with 0.015% full-volume error and 416 MiB sampled GPU memory. Motion recovery
is not included. The overall stretch targets remain unmet. Slabs use less than
half the warm-fastest CuPy/ASTRA workflow's 1262 MiB, but native ASTRA 2D uses
156 MiB and is the fastest external cold workflow. Warm and cold slab speedups
are only about 2.01x and 1.18x respectively in the latest copy-optimized record.
The resident-volume alternative has comparable warm time but uses 4250 MiB.
Evidence and limitations are in
[the performance report](../performance.md). Sampler validation and the initial
size-64/128 Gaussian, sharp and noisy CGLS comparisons are complete, including the
fixed multiresolution policy and opt-in Joseph plane model. Controlled profiling
confirmed that the ray adjoint dominates operator cost; the implemented plane
forward/gather-transpose pair improves that cost without reaching the external
speed targets. Some Gaussian and sharp/noisy multiresolution warm solves are
faster than ASTRA CGLS. Cold search, memory, and unsuccessful anisotropic cases
still prevent broad claims. Independent gVXR material/noise fixtures now provide
additional correctness diagnostics, not replacement quality gates.

Direct FBP comparisons now include native ASTRA/TIGRE and a GPU-filtered
CuPy/ASTRA workflow. Retaining filtered tails through the full output volume
corrects a shared cropping bias and improves Gaussian direct-reconstruction
quality without relaxing its gate. This correction is included in both
TomoJAX and the external adapters. Sharp/noisy direct-FBP comparisons now extend
through size 256, with the anisotropic-64 sharp failure and tilted limitations
retained. Differentiated workflows, pose-recovery distributions and the motion-corrected 512-cubed
demonstration remain open. Initial Fourier-preconditioning pilots did not
improve selected quality budgets. Per-view normal-equation accumulation from the independent Claude Opus 5.5
High review is now implemented and numerically validated; successful joint
alignment tests remain outstanding.

The first-order Joseph CUDA derivative API and fused raw least-squares
operation are now implemented. All 12 component cases pass numerical checks;
the retained sweep with at least 200 ms warmup per method meets the 3x
forward-cost target in **7/12** cases after switching the gather transpose to
one-warp tiles and specializing exact unit detector-row mappings. It does not establish the differentiation
target across the suite, a successful joint alignment baseline, or peak-process
memory target against external reconstruction libraries. Compiler temporary estimates at size 64 fall from about
1.52 GB for reference reverse AD to 1.11 MB for fused CUDA on parallel/tilted
cases; these are separate from process VRAM. Nine isolated workers measured
212 MiB for fused CUDA versus 1236–2260 MiB for JAX reverse AD at size 64, and
270–398 MiB for fused CUDA at size 256, all with 60 views. These are internal
derivative comparisons, not complete reconstruction/alignment results. See the
[differentiation measurements](../performance.md#bounded-memory-first-order-joseph-differentiation)
for source hashes, all samples, accuracy and limitations.


Optional cubic plane interpolation now has matched CUDA derivatives and CGLS.
The corrected component sweep passes all 12 numerical gates. Extending the exact
unit-row adjoint specialization to cubic improves the 3x derivative-cost target
from 2/12 to 9/12 cases; the linear control remains at 7/12. Three tilted cubic
cases still exceed the target. Known-volume recovery now has a
retained analytic-data baseline with fixed perturbations, physical margins and
per-view error gates. Cubic passes 4/6 noiseless cases and 0/6 noisy cases; the
linear noiseless control passes 0/6. Native 128/256 detector cases use padded
192/384-cubed volumes, not 128/256-cubed reconstruction volumes. None of these
results establishes successful joint reconstruction/alignment or the 20x
workflow target. See [cubic and pose controls](../performance.md#cubic-plane-interpolation-and-pose-recovery-controls).

The fused per-view pose-normal-equation API passes all 24 numerical component
cases. Geometric-mean speedups over forming the explicit Jacobian are 4.47x
linear and 4.87x cubic, with about 6.7x lower compiler temporary storage at
size 256. Complete known-volume noisy recovery is 1.32–2.64x faster, but it
still misses the angular accuracy gate on all six weak-object noisy cases.
These component and known-volume results do not meet the joint recovery goal.

An additional fixed nine-Gaussian known-volume fixture passes 12/18 cases in
both explicit and fused methods, including all 128/256 cases and none at 64.
It uses nominal initialization, true perturbations up to ±3 degrees / ±10 pixels,
three seeds, and cold plus seven warm runs without changing either accuracy
gate. Fused warm recovery averages 2.04x faster on the successful cases, with
96.9–160.3 ms at size 128 and 839.1–1107.7 ms at size 256. Earlier weak-object
failures remain. This development fixture supplies useful known-volume evidence;
it does not prove 99% robustness for joint alignment.

A restricted unknown-volume prototype detects Gaussian features in an initial
reconstruction, jointly fits their shared centers and view poses, then performs
a voxel reconstruction. Its frozen 18-case development sweep passes 6/18 cases,
all parallel; every tilted case fails. A subsequent coupled Gaussian-object
control passes one tilted case, but assumes the entire object is a Gaussian
mixture. These preliminary results do not establish general joint recovery or
the 20x target. See the [joint controls](../performance.md#restricted-joint-recovery-controls).

The full-image Gaussian-object control has since completed the same 18-case
development matrix: all cases pass both cold and one warm run, including at
least 179/180 views passing each pose gate in every run. This supplies a
successful restricted baseline with an explicit Gaussian-mixture object prior.
It does not establish general-volume recovery, independent held-out robustness,
or a 20x successful-workflow speedup. The broader goal remains open.

An optimized version of this restricted model also passes the 512³ / 720-view
motion-recovery showcase: 7.91 seconds cold including process startup, recovery
and independent verification; 1.99 seconds warm; 1688 MiB sampled process GPU
memory. All 720 views pass the pose gates, with 0.00419% model-volume error.
This is one cold and one warm run, excludes synthetic fixture generation, and
assumes the object is a positive Gaussian mixture. All 18 smaller development
cases pass as well.

The completed same-verifier paired comparison now includes one cold and seven
warm runs per solver/case: all 288 smaller-case runs pass, with **10.19x warm**
and **2.13x cold** geometric-mean speedups. The repeated restricted 512³ showcase
also passes all eight runs: **6.83 s cold including startup**, **1.949 s warm
median**, and the same 1688 MiB GPU peak. Its synthetic fixture takes 138.65 s
to generate separately; that cost is excluded from recovery timings. Larger
normal-equation batches increase GPU memory for the smaller cases. These
measurements strengthen the restricted result, while general-volume recovery,
held-out robustness and the overall 20x target remain unresolved.

An independent-voxel prototype also passes three blind 128³ / 180-view controls:
two parallel seeds and one laminography seed, with all 180 views meeting both
pose gates and 0.41% / 1.92% reconstructed-volume errors. It uses nonnegativity
and a squared physical Laplacian prior, without a fitted Gaussian object model.
These remain development tests on the fixed nine-Gaussian measurement object;
general-object robustness and a repeated successful-workflow speed comparison
are not established. See the [free-voxel controls](../performance.md#free-voxel-joint-recovery-development-controls).

Extending that free-voxel baseline exposes remaining gaps: the native size-64
controls fail the pose gate, and size-256 laminography reaches only 177/180
accepted views. Size-256 parallel passes both cold and warm checks at 0.189%
volume error, but takes about ten minutes per complete recovery. Faster and
more broadly successful free-voxel recovery remains active work.

A subsequent coarse-to-fine control passes all 180 views at size 256 for both
geometries: 43.01 s parallel and 47.58 s tilted warm complete recovery, with
0.245% and 3.283% volume error. Parallel is 14.14× faster than the accepted
native baseline in these single warm samples. The tilted baseline failed
acceptance, so no successful-workflow ratio is claimed for it. These remain
development controls on one object and seed, pending broader repeated checks.

The completed asynchronous Fourier comparison improves default 512³ / 720-view
reconstruction to **2.51 s cold / 0.518 s warm**, with unchanged 0.015018% error.
Its warm speedup is **5.96×**, but 678 MiB is 53.7% of the warm-fastest external
workflow's memory, missing the half-memory goal in this configuration. The
standard six-case Gaussian suite reaches **1.92× warm / 0.88× cold** geometric
mean speedup; three cases meet 2× warm. These results do not meet the overall
10× goal. Reused CUDA streams fix a repeated-call allocator growth regression,
with 709 full GPU tests passing and 71 focused tests clean under memory checking.
