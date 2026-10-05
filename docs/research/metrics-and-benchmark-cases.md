# TomoJAX metrics and benchmark cases

This is an improvement inventory for reconstruction, alignment, numerical
kernels, acquisition correction, and the complete data workflow. It describes
what we can measure and improve, which cases those measurements should cover,
and which capabilities exist in the current code. FBP means filtered
backprojection.

The main outcome is **elapsed time to an accepted reconstruction or alignment**,
subject to accuracy, memory, and reliability requirements. Faster kernels,
fewer iterations, or lower training loss are useful only when the final result
still meets those requirements.

This is a measurement specification, not a claim that every measurement below
is already implemented or every case is supported. The coverage section identifies
existing evidence. Continuous parameters create infinitely many combinations;
the case axes below define the space to sample, rather than an exhaustive list
of every possible scan.

## Methods and workflows in the current code

| Method or workflow | Variants to evaluate | Scope |
|---|---|---|
| FBP | Ramp, Shepp–Logan, Hann; automatic, JAX, explicit Pallas backprojection | Public reconstruction API; CUDA specialization for built-in parallel geometry |
| General filtered adjoint | Custom posed parallel rays, explicit detector grids, laminography | Existing FBP fallback; approximate reconstruction for incomplete-angle data |
| FISTA | TV, Huber-TV, zero regularization; constraints; streamed or batched gradients | Public reconstruction API |
| CGLS | Matched JAX/Pallas operators, scalar damping, nonzero starts, normal-residual stopping | Public Python API; unconstrained least squares; not a differentiable reconstruction layer |
| SPDHG | TV, Huber-TV, stochastic blocks, automatic or supplied step sizes | Public reconstruction API |
| Multiresolution FISTA | Coarse-to-fine factors and iteration budgets | Existing module helper; not a separate top-level solver entrypoint |
| Differentiable FISTA core | JAX/Pallas forward and explicit volume-adjoint choices, weighted data, support masks | Internal reconstruction core used by alignment workflows |
| Differentiable reconstruction layer | Unrolled and implicit differentiation, CG adjoint solve | Internal bilevel alignment machinery; Huber-TV or zero regularization required by the layer |
| Pose alignment | Five per-view degrees of freedom, subsets, per-view/polynomial/spline models | Public alignment API |
| Setup calibration | Detector-u centre/COR, detector roll, axis direction/laminography tilt | Public schedules and expert controls |
| Combined setup and pose | Staged correction, active/frozen parameters, explicit gauge policy | Existing workflow; identifiability must be assessed |
| Detector-v reference shift | Expert parameter selection | Physically ambiguous with sample elevation; not a reliable unrestricted recovery target |
| Proposal scoring and initialization | Phase correlation, pose candidates, translation seeding | Existing primitives and stage machinery |
| Nuisance estimation | Per-view gain/offset; constant plus vertical-gradient backgrounds | Existing primitives, not a complete automatic joint nuisance/alignment workflow |
| IO and preprocessing | NeXus/HDF5, TIFF, flat/dark correction, selection/rejection, crop, transmission/log output | Public workflow |

Sources: [reconstruction API](../../src/tomojax/recon/api.py),
[multiresolution helper](../../src/tomojax/recon/multires.py),
[FISTA core](../../src/tomojax/recon/fista_tv_core.py),
[alignment configuration](../../src/tomojax/align/_config.py), and
[reconstruction layer](../../src/tomojax/align/_objectives/recon_layer.py).

## Measurement definitions

Use a fixed evaluation region, physical units, masks, and data domain for each
case. Save these definitions with the result.

- **Volume relative L2 error:** `||x - x_true||₂ / ||x_true||₂`. Also report
  absolute RMSE/MAE/max error, because relative error is undefined at zero truth.
- **Projection residual:** `||W(Ax - y)||₂ / ||Wy||₂`, with the exact weight/mask
  convention recorded. Report unweighted residual too. A square-root statistical
  weight and a raw statistical weight must not be confused.
- **Time to target:** first synchronized elapsed time at which every specified
  quality requirement is met. A timeout or unsuccessful run is a failure, not
  an unusually fast result or a missing sample.
- **Effective data passes:** total views actually processed divided by the
  number of views in the dataset. Record forward and adjoint work separately;
  one SPDHG block update is not one FISTA full-data iteration.
- **Pose errors:** translations in native detector pixels and physical units;
  rotations in degrees or radians, explicitly labelled. Report each DOF and
  an aggregate rotation-matrix angular error. Do not add pixels and degrees.
- **Gauge-aware errors:** compare truth and estimates under the same declared
  gauge. Report observable geometry error separately from raw parameter error.
  Do not conceal calibration bias with an unconstrained post-hoc registration.
- **Image metrics:** specify data range for PSNR/SSIM, foreground/background
  masks, and whether evaluation is 3D or on named slices. Projection-domain SSIM
  used as an optimizer loss is not automatically a reconstruction SSIM metric.
- **Success rate:** successful cases divided by all scheduled cases, with
  failures, timeouts, invalid inputs, unsupported routes, and fallbacks counted
  separately. Show distributions across phantoms, initializations, and seeds.

## Runtime and resource metrics

Apply these to every solver, alignment workflow, and kernel where meaningful.

| ID | Metrics to collect | Improvement criterion |
|---|---|---|
| P01 | Complete API runtime; complete CLI runtime including read, preprocessing, compute, and write | Lower at accepted final quality |
| P02 | Time to target reconstruction error, pose error, and held-out residual | Lower; count runs that never reach target |
| P03 | First-process startup, import, device initialization, tracing, compilation, first execution | Lower; distinguish these from warm calls |
| P04 | Warm median, p90/p95 latency, minimum, standard deviation/CV, confidence interval | Lower latency and variability; retain samples |
| P05 | Time per iteration, view, batch, level, alignment outer loop, and schedule stage | Lower for equivalent work |
| P06 | Projections/s, rays/s, voxel-view contributions/s, effective data passes/s | Higher; state denominator and algorithm |
| P07 | Peak device memory, host resident memory, persistent state, temporary workspace | Lower at fixed workload |
| P08 | Largest volume, detector, and view stack completing under a fixed memory budget | Higher feasible workload |
| P09 | Host-to-device and device-to-host time/bytes; synchronization and Python dispatch overhead | Lower; preserve resident and transfer-inclusive measurements |
| P10 | Allocation count/bytes, intermediate-array volume, padding waste, buffer reuse | Less unnecessary work |
| P11 | Recompilations, distinct executables, executable/cache size, cache hit rate | Lower avoidable specialization cost |
| P12 | Scaling with volume dimensions, detector dimensions, views, active parameters, folds, and batches | Better scaling at equivalent quality |
| P13 | GPU bandwidth, occupancy, register pressure/spills, instruction count, launch count, atomic contention | Diagnostics of a measured bottleneck; no universal best value |
| P14 | Energy per accepted result and average/peak power where telemetry is available | Lower energy at equal result quality |
| P15 | Logging, callbacks, checkpointing, quality diagnostics, and progress-report overhead | Lower overhead with required observability retained |

## Reconstruction quality metrics shared by all methods

| ID | Metrics to collect | Improvement criterion |
|---|---|---|
| R01 | Volume relative L2 error, RMSE, MAE, maximum absolute error | Lower against independent truth |
| R02 | Foreground, background, centre, edge, ROI, and per-slice errors | Lower without hiding difficult regions in an average |
| R03 | Attenuation bias, fitted diagnostic gain/offset, material-region mean error, contrast recovery | Less physical bias; report unfitted errors as primary |
| R04 | PSNR, SSIM, multiscale SSIM | Higher with fixed definitions and data range |
| R05 | Edge-position error, edge-spread width, point-spread width, directional resolution/MTF | Better spatial fidelity without excessive noise |
| R06 | Fourier ring/shell correlation and directional resolution estimates | Better reproducible resolution with a justified independent-reference or split-data protocol |
| R07 | Noise standard deviation, noise power spectrum, SNR, contrast-to-noise ratio | Better task performance at matched resolution/dose; not smoothing alone |
| R08 | Ring, streak, cupping, shading, ringing/overshoot, background leakage, axial smear scores | Lower on explicitly defined artifact regions/tests |
| R09 | Weighted/unweighted training residual, held-out projection error, per-view residual tails | Lower predictive error without fitting noise or corrupt views |
| R10 | Feature size, position, thickness, spacing, particle count, segmentation overlap/surface error | Better downstream scientific measurements |
| R11 | Negative-voxel fraction, lower/upper-bound violation, support leakage, integral/mass bias | Satisfy declared physical constraints and applicable invariants |
| R12 | Error and variance across noise seeds, missing-view patterns, phantoms, hardware, precision | Lower variability and fewer severe failures |
| R13 | Sensitivity to regularization, initial volume, stopping tolerance, sampling step, geometry error | Wider useful operating range |
| R14 | Quality at fixed runtime/memory/dose and runtime at fixed quality | Improve the tradeoff, not a single isolated score |

For real scans without truth, use repeat scans, independent calibration objects,
held-out views, task-specific measurements, and residual inspection. Sharpness
alone can reward noise; data consistency alone can reward incorrect geometry.
Synthetic solver tests using their own forward model establish convergence but
do not independently establish physical reconstruction accuracy.

## FBP metrics

| ID | Metrics to collect | Improvement criterion |
|---|---|---|
| F01 | Filtering time, backprojection time, transpose/copy time, final scaling time | Lower total reconstruction time |
| F02 | Padded FFT length, filter-cache hits, workspace, repeated filter construction cost | Lower cost while preserving linear convolution |
| F03 | Error against direct spatial ramp convolution, detector-edge wraparound, DC/low-frequency bias | Lower numerical/filter error |
| F04 | Attenuation scale and invariance under consistent changes of length units | Correct physical normalization |
| F05 | Angular quadrature and normalization error across scan coverage and view counts | Correct weighting; default FBP is not a nonuniform-angle quadrature implementation |
| F06 | Filter-specific noise, resolution, ringing, truncation bias | Better declared noise/resolution tradeoff for ramp, Shepp–Logan, Hann |
| F07 | JAX/Pallas agreement for integer rows, fractional rows, offsets, shifted origins, partial blocks | Lower discrepancy; correct specialization selection |
| F08 | Error of generic filtered adjoint versus direct parallel FBP and independent truth | Understand discretization differences; do not require unlike operators to be identical |
| F09 | Initialization benefit: iterations and total time saved when FBP seeds an iterative method | Better final solve including initialization cost |
| F10 | OOM fallback success, retry count, retry time, skipped/double-counted views | Reliable completion with exact view accounting |

## FISTA metrics

| ID | Metrics to collect | Improvement criterion |
|---|---|---|
| I01 | Data loss, regularizer value, total objective, relative iterate change | Reach a defined solution tolerance; lower objective alone is not better image quality |
| I02 | Proximal-gradient mapping or projected stationarity residual; convergence rate | Lower appropriate optimality residual |
| I03 | Iterations, data passes, forward/adjoint calls, time to target | Lower equivalent work and elapsed time |
| I04 | Power-method/norm-estimation time and error; supplied/estimated Lipschitz constant | Stable useful step sizes at lower setup cost |
| I05 | TV proximal-solve time, inner iterations, proximal accuracy | Lower cost without degrading outer convergence |
| I06 | Huber smoothing bias, gradient cost, sensitivity to delta/lambda | Better quality/runtime tradeoff with the regularizer declared |
| I07 | Momentum overshoot, oscillation, divergence, early-stop error | Fewer failed or prematurely stopped runs; FISTA need not decrease loss every iteration |
| I08 | Streamed versus batched gradient time/memory and result agreement | Lower memory or time with matched results |
| I09 | Initial-volume/warm-start benefit and support/positivity/bound handling | Faster valid convergence, unchanged constraint semantics |
| I10 | Internal core versus public FISTA result/objective differences | Explain differing regularizers/settings before claiming a backend speedup |

## SPDHG metrics

| ID | Metrics to collect | Improvement criterion |
|---|---|---|
| S01 | Full-data primal objective and residual at common evaluation points | Better solution quality at matched elapsed time/data passes |
| S02 | Primal and dual update residuals, feasibility, justified primal-dual gap | Lower optimality error where the objective/constraints admit a valid gap calculation |
| S03 | Stochastic block updates, sampled views, effective data passes, time to target | Lower work/time; do not compare raw iteration counts to FISTA |
| S04 | Seed-to-seed objective, reconstruction error, runtime, success-rate distribution | Lower variance and higher reliability |
| S05 | Block-size and selection-probability sensitivity, block coverage/frequency | Stable convergence across schedules |
| S06 | Operator-norm estimation, tau/sigma/theta stability margin, setup overhead | Reliable step sizes with lower total cost |
| S07 | Data-dual, TV-dual, accumulator, and extrapolated-volume memory | Lower memory at matched results |
| S08 | Minibatch objective-estimator bias/variance and logging overhead | Useful monitoring at lower cost |
| S09 | Warm-start benefit and consistency under view reordering | Better convergence with reproducibility understood |

SPDHG's current loss history contains a minibatch objective estimator at logged
steps and zero placeholders at unlogged steps. Those zeros are not successful
zero-loss iterations. Compute a full objective separately for solver comparisons.
Sources: [SPDHG implementation](../../src/tomojax/recon/spdhg_tv.py) and
[FISTA implementation](../../src/tomojax/recon/fista_tv.py).

## Multiresolution metrics

| ID | Metrics to collect | Improvement criterion |
|---|---|---|
| M01 | Runtime/memory/iterations per level; total pyramid and resampling overhead | Lower time to the same finest-resolution target |
| M02 | Quality after each level and after final refinement | Preserve or improve final detail |
| M03 | Binning/upsampling amplitude error, aliasing, detector-centre shift, FOV/origin drift | Preserve physical coordinates, especially odd sizes |
| M04 | Capture-range gain versus single-resolution solve | Recover larger initial errors reliably |
| M05 | Coarse-level budget usefulness, switching criterion, final verification cost | Avoid wasted levels or premature refinement |
| M06 | Resume consistency at level/stage boundaries | Match an uninterrupted solve within declared tolerance |

## Alignment accuracy and robustness metrics

Alignment requires both image-quality and geometry-quality outcomes. A visually
better reconstruction is insufficient evidence of correct calibration.

| ID | Metrics to collect | Improvement criterion |
|---|---|---|
| A01 | Per-view RMSE/MAE/median/p95/max for `alpha`, `beta`, `phi`, `dx`, `dz` | Lower gauge-consistent error against truth |
| A02 | Rotation-matrix angular error; transformed landmark displacement | Lower geometric error, avoiding angle-wrap artifacts |
| A03 | Detector-u/COR error, detector-v error where identifiable, detector-roll error | Lower error in native pixels/physical units/degrees |
| A04 | Axis-direction angular error and recovered laminography tilt | Lower physically meaningful direction error |
| A05 | Effective ray-origin/direction error, detector-coordinate error, landmark reprojection error | Better observable geometry even when parameterizations differ |
| A06 | Corrected-volume metrics R01–R14; improvement over unaligned and true-geometry reconstructions | Better reconstruction with the remaining geometry penalty quantified |
| A07 | Training/held-out residual, fold-to-fold spread, train-validation gap | Better generalization and less compensation for model error |
| A08 | Gauge-invariant error, gauge constraint residual, parameter rank/conditioning | Recover observable parameters; expose ambiguity rather than hide it |
| A09 | Cross-talk between pose/setup/nuisance estimates; parameter covariance or sensitivity | Less incorrect attribution of one error source to another |
| A10 | Capture range and success rate versus initial translation, rotation, setup error | Larger reliable basin of convergence |
| A11 | Catastrophic-failure rate and residual count of badly aligned views | Fewer severe failures, not only better mean performance |
| A12 | False correction on already aligned data; degradation relative to no correction | Near-zero unnecessary motion and preserved image quality |
| A13 | Trajectory error, derivative/curvature error, smoothness bias, jump preservation | Recover real motion without fitting noise or smoothing away events |
| A14 | Stability across seeds, view order, initial reconstruction, bounds, priors, loss, and pyramid | Consistent usable solutions |
| A15 | Repeat-scan/calibration-object consistency and downstream scientific measurement error | Better real-data outcomes where absolute truth is unavailable |
| A16 | Uncertainty interval coverage and calibration, if uncertainty estimation is added | Honest confidence; uncertainty estimation is not currently a general reported output |

For A08–A09, compare both canonicalized parameters and predicted rays. Setup/pose
gauge equivalence can make raw parameter RMSE misleading. Unidentifiable
detector-v/elevation cases should be scored on observable geometry, successful
ambiguity detection, or recovery under an explicitly supplied anchor/prior.

## Alignment optimization and differentiation metrics

| ID | Metrics to collect | Improvement criterion |
|---|---|---|
| O01 | Total alignment time split into initialization, reconstruction, pose, setup, proposal scoring, verification, IO | Lower complete time to accepted geometry and volume |
| O02 | Outer loops, inner reconstruction iterations, candidate evaluations, loss/gradient/Jacobian/HVP calls | Less work for equivalent accepted results |
| O03 | Accepted/rejected steps, line-search evaluations, fallback steps, post-constraint rejections | Better progress per unit cost; rejection rate alone is not a quality target |
| O04 | Loss reduction per second/data pass, gradient norm, parameter-step norm | Efficient stable convergence in normalized coordinates |
| O05 | Gauss–Newton/LM assembly and solve time, Jacobian memory, damping, condition/rank diagnostics | Stable useful steps at lower cost |
| O06 | L-BFGS history memory, function evaluations, termination reason, failed search rate | Lower time to valid convergence |
| O07 | Proposal cost, selected-candidate quality/rank, downstream benefit | Good initialization whose later savings exceed scoring cost |
| O08 | Early-stop false positives/negatives, extra work after convergence | Stop at the required quality with less wasted work |
| O09 | Actual backend and precision by stage, fallback count/reason, fraction of work accelerated | Verify what ran; avoid crediting a requested backend that was bypassed |
| O10 | Loss/mask/feature preparation cost and cache reuse | Less repeated target-dependent work |
| D01 | Geometry, detector-coordinate, and volume gradient error against finite differences/independent derivatives | Correct gradients across all active DOFs and units |
| D02 | JVP/VJP duality and directional-derivative error; Hessian-vector consistency | Correct differentiated operators |
| D03 | Unrolled-gradient time/memory/error versus inner iteration count | Better derivative quality/cost tradeoff |
| D04 | Implicit-adjoint CG iterations/time/residual, damping sensitivity, inner stationarity | Accurate useful hypergradients at lower memory/time |
| D05 | Implicit versus unrolled gradient agreement under appropriate convergence assumptions | Diagnose differences; they need not match for an unconverged finite-step inner solve |
| D06 | Fold reconstruction time, reuse/recomputation, validation leakage, held-out prediction error | Faster valid bilevel optimization |
| D07 | Checkpoint/rematerialization cost, saved intermediates, recompilation across stages | Lower derivative memory with acceptable recomputation |

The current fast and reference alignment profiles change regularization and
precision as well as backend policy. Compare both matched numerical settings
and complete profiles at matched achieved quality. Some geometry-gradient paths
explicitly require JAX even when Pallas is requested. Value-only proposal speed
is not a measurement of pose-gradient or complete alignment speed.
Sources: [profiles](../../src/tomojax/align/_profiles.py),
[objective backend routing](../../src/tomojax/align/_objectives/fixed_volume.py),
[optimizer diagnostics](../../src/tomojax/align/optimizers.py), and
[quality policies](../../src/tomojax/align/_quality_policy.py).

## Projector and kernel metrics

| ID | Metrics to collect | Improvement criterion |
|---|---|---|
| K01 | Forward projection latency/throughput and analytic line-integral error | Faster physically accurate projection |
| K02 | Matched-adjoint latency/throughput, VJP discrepancy, inner-product identity error | Faster numerically correct adjoint |
| K03 | Single view, stacked views, summed backprojection, residual SSE, fused weighted loss/gradient | Measure each operation and its complete caller separately |
| K04 | Ray-step and truncation sensitivity, quadrature convergence, boundary/zero-padding error | Controlled integration error |
| K05 | JAX versus Pallas error; FP32 versus FP16/BF16 error and optimization impact | Lower cost within a declared scientific error budget |
| K06 | Generic/integer-slice eligibility, wrong-specialization rate, fallback correctness | No unsafe shortcuts for tiny tilts or accumulated spacing drift |
| K07 | Tail-tile correctness/cost, tile utilization, output-layout/transposition overhead | Efficient odd dimensions without out-of-bounds access |
| K08 | Inline/cached traversal preparation time, state memory, reuse break-even point | Improve the actual repeated-use workload |
| K09 | Tile/warp/unroll tuning, instruction and memory efficiency, atomic collision cost | Lower measured runtime with unchanged semantics |
| K10 | Repeat-run atomic variation, nonfinite outputs, memory-sanitizer errors | Bounded reproducibility error and zero memory errors |
| K11 | CPU interpretation versus real CUDA agreement and actual backend execution | Validate both; interpreter success alone is not GPU validation |
| K12 | Performance and accuracy across supported JAX/backend versions and GPU architectures | Wider verified coverage with fewer regressions |

## Acquisition correction and motion initialization metrics

| ID | Metrics to collect | Improvement criterion |
|---|---|---|
| N01 | Flat/dark-corrected transmission/attenuation error, clipping bias, invalid-pixel fraction | Correct physical preprocessing with explicit invalid-data handling |
| N02 | Gain/offset/background parameter error and post-correction held-out residual | Recover acquisition effects without absorbing geometry or sample signal |
| N03 | Mask coverage, empty/degenerate-mask handling, weighted-fit stability | Reliable fits under missing data |
| N04 | Bad-view/pixel detection precision, recall, false rejection, missed corruption | Better decisions against labelled synthetic/real defects |
| N05 | Residual stripes/rings, hot/dead pixels, zingers, intensity drift, blur | Less contamination without removing real structure |
| N06 | Phase-correlation shift error, sign/axis correctness, wraparound behavior, subpixel bias | Better initialization, including even/odd detector shapes |
| N07 | Nuisance/initialization runtime and resulting alignment capture/time-to-target | End-to-end benefit exceeds preprocessing cost |
| N08 | Robustness under model mismatch and wrong noise-domain assumptions | Detect or tolerate mismatch; do not interpret every residual reduction as truth recovery |

The simulator supplies projection-domain artifact injections. Its current
Poisson option is not by itself a complete photon-counting acquisition model
with incident counts, flat fields, and logarithmic transformation. Physically
specified low-dose benchmarks need that data-generation protocol recorded.
Sources: [artifact implementation](../../src/tomojax/datasets/_impl/artefacts.py),
[preprocessing configuration](../../src/tomojax/io/_preprocess_impl/config.py),
[nuisance API](../../src/tomojax/nuisance/api.py), and
[motion API](../../src/tomojax/motion/api.py).

## Workflow and software reliability metrics

| ID | Metrics to collect | Improvement criterion |
|---|---|---|
| W01 | NeXus/HDF5/TIFF read/write throughput, dataset startup latency, output size/compression | Lower complete workflow cost |
| W02 | Streaming/chunking peak memory and copy count | Larger feasible datasets without excessive copies |
| W03 | Projection/volume/geometry round-trip error, axis order, units, origin, angle convention | Correct persisted scientific data |
| W04 | Checkpoint time/size, resume time, uninterrupted/resumed result discrepancy | Reliable recovery with lower overhead |
| W05 | OOM/nonfinite retry success, fallback provenance, invalid-input detection | Correct completion or useful explicit failure |
| W06 | Crash, timeout, NaN/Inf, silent-corruption, and unsupported-route rates | Fewer failures; unsupported cases must not count as completed cases |
| W07 | CLI/API agreement, saved metadata completeness, exact experiment replay | Reproducible results and traceable backend/geometry choices |
| W08 | Installation/build success, import time, supported Python/JAX/CUDA combinations | Reliable usable distributions |
| W09 | Numerical test coverage by case/DOF/backend, sanitizer coverage, regression escape rate | Better meaningful coverage; line coverage alone is insufficient |
| W10 | Benchmark duration, flakiness, sensitivity to known regressions, source/result reproducibility | A sustainable measurement system that catches real problems |

## Geometry and acquisition case axes

Each metric should be attached to an explicit case. The following axes combine
with the method-specific settings; they are not all valid for every method.

| Axis | Cases to include | Current scope or caveat |
|---|---|---|
| Ray geometry | Parallel CT; parallel laminography; arbitrary rigid parallel-ray poses/axis direction | Existing geometry/model families |
| Tilt | Zero, tiny nonzero, moderate, severe; different supported axes/directions | Include specialization boundaries and poorly conditioned laminography |
| Volume shape | Cubes, thin slabs, elongated samples, non-cubic grids, single-slice limits | Existing array shapes; validate method-specific degeneracies |
| Volume scale | Tiny correctness cases, moderate scans, large scans, near-VRAM-limit scans | Use physical extent and resolution separately |
| Detector shape | Square/rectangular, odd/even/prime sizes, partial tiles, narrow rows/columns | Essential for masking, padding, and scaling correctness |
| Voxel sampling | Isotropic and anisotropic voxels; varying physical spacing | Preserve units when changing resolution |
| Detector sampling | Matched/mismatched pixel spacing; integer/fractional z alignment | Include accumulated near-integer spacing errors |
| Grid placement | Centred, explicit origin, shifted volume centre, displaced ROI | Must preserve the same physical model |
| Detector calibration | Positive/negative u/v offsets, subpixel shifts, detector roll | Explicit grids may change available acceleration routes |
| Views | Few, moderate, dense; uneven batch tails; one-view forward/adjoint limits | Identifiability and reconstruction quality depend on coverage |
| Angular coverage | Half turn, full turn, limited-angle/missing wedge, sparse views | Default FBP weights assume a uniform half turn |
| Angular sampling | Uniform, nonuniform, clustered, gaps, duplicates, reordered/reversed views | Some are stress or rejection cases; nonuniform FBP weighting needs additional work |
| Angle calibration and conventions | Angle sign, zero offset, scale, degrees/radians, pose composition order, axis swaps | Geometry state represents angle offset/scale; they are not additional active DOFs in the ten-parameter alignment registry |
| FOV | Fully covered, truncated, off-centre sample, cropped detector, partial ROI, support touching boundaries | Track ROI and unobserved regions explicitly |
| Signal | Smooth, sharp boundaries, fine texture, sparse objects, dense structures, weak/high contrast | Prevent tuning only to smooth Gaussian phantoms |
| Noise | Noiseless; Gaussian; Poisson with declared domain; mixed/heteroscedastic noise | A stress model is not automatically a physical acquisition model |
| Detector defects | Dead/hot pixels, zingers, stripes, dropped views, blur | Existing synthetic artifact controls |
| Intensity drift | Constant/per-view gain/offset, linear/sinusoidal drift, vertical-gradient background | Simulator and nuisance models provide different subsets |
| Gross corruption | Bad-view bursts, saturation, NaN/Inf, angle mismatch, malformed metadata | Separate recovery benchmarks from expected rejection tests |
| Weights and masks | Uniform/nonuniform view weights, zero-weight views, partially/fully masked data, support masks | Use the granularity supported by each solver/loss; reject unsupported shapes explicitly |
| Input provenance | Analytic projections, independent-library projections, own-model simulation, repeat scans, real calibrated datasets | Report which form of truth or validation is available |
| Repetition | Multiple phantoms, noise seeds, motion seeds, initializations, acquisition repeats | Include uncertainty and failure distributions |

Existing phantoms include Shepp–Logan, cube, rotated cube, sphere, blobs,
random cubes/spheres, and a laminography disk. The new benchmark also generates
analytic Gaussian ellipsoids. Edge/point/resolution patterns and calibrated
real-data collections are useful additions, not all existing built-in generators.
See [datasets API](../../src/tomojax/datasets/api.py) and
[simulation configuration](../../src/tomojax/datasets/_impl/simulate.py).

## Reconstruction settings to cross with the cases

- FBP: all three filters; CPU/JAX CUDA/Pallas routes where supported; explicit
  detector grid versus implicit canonical grid; integer/fractional rows;
  angular scale; detector padding; generic fallback and OOM retries.
- FISTA: TV/Huber-TV/zero regularization; lambda and Huber delta; zero/FBP/supplied
  initialization; auto/supplied Lipschitz constant; norm-estimation iterations;
  TV proximal iterations; fixed budget/early stopping; streaming/batching;
  positivity, support, lower/upper bounds; single/multiple resolution levels.
- SPDHG: TV/Huber-TV/zero regularization; lambda/delta; seed; block size;
  auto/supplied tau/sigma; theta; logged-step interval; zero/FBP/supplied start;
  positivity/support; view order and incomplete final blocks.
- All applicable kernels: integration step/maximum steps; checkpointing; unroll;
  precision; device; resident versus transferred inputs; cold versus warm call.
- Multiresolution: level factors, odd-size scaling, per-level iteration/quality
  budget, interpolation and initialization, final full-resolution verification.

## Alignment cases and settings

### Parameters and error patterns

The current DOF registry has five pose parameters and five setup parameters:

| Scope | Parameters | Cases |
|---|---|---|
| Pose rotation | `alpha`, `beta`, `phi` | Each alone, pairs, all rotations, angle wrapping, wrong nominal angle/sign |
| Pose translation | `dx`, `dz` | Each alone, coupled translation, subpixel and large shifts |
| Full pose | All five pose DOFs | Combined rotation/translation; active/frozen subsets; bound hits |
| Detector centre | `det_u_px`, `det_v_px` | COR/u recovery; v as anchored or ambiguity-detection experiment |
| Detector roll | `detector_roll_deg` | Positive/negative roll; roll plus COR; roll versus apparent sample rotation |
| Rotation axis | `axis_rot_x_deg`, `axis_rot_y_deg` | Individual/coupled axis error, laminography tilt, gauge diagnostics |
| Mixed ownership | Pose plus setup plus nuisance perturbations | Correct separation, compensation/cross-talk, physically supplied priors |

Angle-suffixed setup names have radian-valued internal state and degree-facing
display/configuration conventions; record conversions rather than assuming the
name determines storage units. Source: [DOF registry](../../src/tomojax/align/_model/dof_specs.py).

Cross those parameters with these motion/error patterns:

- No error, to detect harmful false correction.
- Constant global offset; slowly varying drift; linear/quadratic trajectories.
- Smooth periodic motion, wobble, and correlated drift.
- Independent per-view jitter and high-frequency motion.
- Abrupt single jumps, piecewise drift, bursts, and isolated bad views.
- Small, medium, and large initial error, in declared pixels/degrees/world units.
- Symmetric, repetitive, low-texture, sparse, and asymmetric objects.
- Good/poor initial reconstruction; correct/incorrect support; truncation.
- Geometry error alone, nuisance error alone, and combined errors.
- Observable parameters, nearly dependent parameters, and deliberate gauge ambiguity.

Abrupt jumps, object-frame drift, and unrestricted mixed nuisance/geometry
recovery remain research or diagnostic cases, not guaranteed public workflows.
The existing object-motion trace stores `tx_obj_px`, `ty_obj_px`, `tz_obj_px`,
and `rot_obj_z_deg`, with a `tx_rmse_px` helper. Evaluate those quantities when
that frame/truth is available; do not equate them directly with the five pose
parameters without the appropriate transformation.

### Motion models and schedules

Evaluate per-view parameters, polynomial trajectories at different degrees,
and spline trajectories at different degrees/knot spacings. Record both the
number of free coefficients and the expanded per-view parameter count.

The complete public schedule preset list is:

`lightning_pose`, `tortoise_pose`, `pose_only`, `pose_phi_only`,
`pose_dx_dz_after_phi`, `cor`, `cor_then_pose`, `detector_roll`,
`axis_direction`, `lamino_tilt`, `setup_safe`.

Also distinguish direct active-DOF selection, custom schedules, and
`expert_coupled`. Cross them with single/multiresolution operation,
fast/reference profiles, reconstruction method/budget, bounds/frozen DOFs,
translation seeding, masks, observer stops, and checkpoint/resume boundaries.
Gauge policies are `reject`, `anchor_mean`, `prior_required`, and `diagnose_only`;
gauge fixing includes mean translation or none where feasible.

Stage roles are proposal, setup, reconstruction, refine, and verify. Internal
quality policies also distinguish proposal, fast, refine, verify, final, and
reference. Do not compare a cheap proposal with a verified final reconstruction
as if they were equivalent work.
Source: [schedule definitions and validation](../../src/tomojax/align/_model/schedules.py).

### Objective and optimizer combinations

| Objective | Existing executable optimizer choices | Important distinction |
|---|---|---|
| Fixed volume | Gradient descent, Gauss–Newton, L-BFGS | Measures pose fitting given an estimated/fixed volume |
| Bilevel cross-validation | Validation Levenberg–Marquardt | Includes reconstruction on training folds and prediction on validation views |
| All-data bilevel | Validation Levenberg–Marquardt | Expert objective using all data; not an independent held-out score |

Adam appears in a type declaration but is rejected by the current schedule
validator for these objectives; it is not a supported executable schedule choice.
GN weighting and setup validation-LM currently support the least-squares-like
losses `l2`, `l2_otsu`, `pwls`, and `edge_l2`. Other losses must use a supported
route or be classified as rejected/fallback cases. The current fold builder uses
interleaved folds; other fold constructions would be additions.

### Complete registered alignment loss list

| Family | Registered names |
|---|---|
| Least squares | `l2`, `l2_otsu`, `pwls`, `edge_l2` |
| Robust | `charbonnier`, `huber`, `cauchy`, `welsch`, `student_t`, `barron`, `correntropy` |
| Correlation | `zncc`, `phasecorr`, `fft_mag` |
| Structural similarity | `ssim`, `ms_ssim`, `ssim_otsu` |
| Foreground overlap | `tversky` |
| Gradient/edge | `grad_l1`, `ngf`, `grad_orient`, `chamfer_edge` |
| Information | `mi`, `nmi`, `renyi_mi` |
| Distribution/descriptor | `swd`, `mind` |
| Count likelihood | `poisson` |

These are 28 registered losses, not 28 proven equally suitable choices for every
scan. Compare loss preparation/evaluation cost, derivative correctness,
capture range, noise/outlier tolerance, intensity-change tolerance, and final
geometry/volume accuracy. Include per-resolution loss schedules. Likelihood
losses must receive data in an appropriate domain. Different losses have
different scales; compare common held-out/physical metrics rather than raw loss
values across families.
Sources: [loss registry](../../src/tomojax/align/_objectives/loss_specs.py) and
[loss adapters and optimizer support](../../src/tomojax/align/_objectives/loss_adapters.py).

## Execution cases

| Axis | Cases |
|---|---|
| Backend | CPU JAX, CUDA JAX, real CUDA Pallas, Pallas CPU interpretation; requested versus actual backend |
| Precision | FP32, FP16/BF16 gathers where supported, automatic precision and its resolved choice |
| Kernel mode | Generic versus integer-slice; single versus stack; inline/cached traversal; supported detector layouts |
| Tuning | Tile shape, warps, unroll, ray steps, batch size, checkpointing, supported state reuse |
| Reuse | First process, first shape, warm same-shape call, changing shapes, changing values, cached geometry/filter |
| Placement | Device resident, host-to-host, realistic IO-to-output pipeline |
| Memory | Comfortable fit, constrained fit, OOM/retry, many sequential scans, long repeated runs |
| Hardware | Different CPU/thread settings, GPU architectures/VRAM/power limits; record driver and clocks |
| Software | Supported Python/JAX/plugin versions, optional library availability, backend migration candidates |
| External comparison | Matched physical geometry, array conventions, precision, integration model, data-transfer scope, and quality target |

Multi-GPU, distributed, ROCm/TPU acceleration, and a replacement GPU compiler
backend require implementation/validation before being listed as supported
performance cases. A CPU interpreter does not establish those capabilities.

## Existing measurements and gaps

| Area | Existing evidence or diagnostics | What the inventory adds |
|---|---|---|
| Forward projection | Analytic Gaussian comparisons, JAX/Pallas error, warm/cold/transfer timings; ASTRA and limited TIGRE adapters | More phantoms, high-contrast/real data, hardware and memory scaling |
| Matched adjoint | Long-ray VJP/dot-product checks, precision regressions, synchronized single-view and batched benchmarks, compiler temporary-buffer measurements | More hardware and measured peak process memory |
| FBP | Analytic amplitude/filter tests, unit rescaling, JAX/Pallas timing and accuracy, OOM regressions | Noise/resolution tradeoffs, truncation, broader geometry and external reconstruction comparisons |
| FISTA/SPDHG | Convergence regressions; complete public-call before/after timings at nominal sizes 32, 64, and 128 across parallel, tilted, and anisotropic scans | External matched-quality comparisons and stochastic robustness benchmarks |
| CGLS | Dense least-squares regressions, JAX/CUDA, damping and roundoff checks; size-64/128 external fixed-quality comparisons, isolated process memory sampling | Larger scans, structured/noisy data, more hardware, differentiated solve |
| Internal FISTA core | Stochastic projection loss/gradient and reconstruction diagnostic script | Systematic differentiated-solver and full alignment comparisons |
| Alignment | Complete-call GN and L-BFGS speedups on synthetic cases, GD timings, outer-step diagnostics, implicit-data-gradient checks, bounded and smooth-motion tests | Broader pose-truth/capture-range studies, setup and bilevel performance, realistic quality-versus-time comparisons |
| IO/nuisance/motion | Contract and focused numerical tests, preprocessing provenance/warnings | Full throughput, scale, uncertainty, and combined-error evaluation |

Relevant existing locations:

- [Projector benchmark](../../bench/compare_projectors.py)
- [FBP benchmark](../../bench/reconstruction_benchmark.py)
- [Adjoint benchmark](../../bench/adjoint_benchmark.py)
- [Public reconstruction/alignment workflow benchmark](../../bench/workflow_benchmark.py)
- [Fixed-quality external solver comparison](../../bench/compare_reconstructions.py)
- [Internal FISTA diagnostic](../../bench/fista_projection_benchmark.py)
- [Reconstruction convergence tests](../../tests/test_reconstruction_convergence.py)
- [Alignment result schema](../../src/tomojax/align/_results.py)
- [Alignment loop timing](../../src/tomojax/align/_pose/_pose_loop.py)
- [Measured performance report](../performance.md)

## Future capability benchmarks

These would require additional implementation; they are not alternative methods
already available through the current reconstruction API:

- SIRT/SART, LSQR, ADMM or other additional iterative solvers.
- Fan-beam/cone-beam/helical geometry and FDK where applicable.
- Alternative forward models such as Siddon or distance-driven projectors.
- Multi-GPU/distributed reconstruction and verified out-of-core execution.
- Nonrigid or within-exposure motion and a complete joint acquisition model.
- General uncertainty estimation and calibrated confidence intervals.
- Polychromatic/beam-hardening, scatter, or other additional physical models.

Each extension inherits runtime, numerical, quality, memory, and reliability
metrics from this inventory, plus model-specific validation. An ASTRA/TIGRE
feature is not implied to exist in TomoJAX merely because it is benchmarkable.

## Turning the inventory into repeatable experiments

The geometry package already provides `MetricSpec`, `ObjectiveCard`, and
`CandidateScore` metadata containers. They can describe metric direction,
validation splits, and candidate results; they do not implement the evaluators
in this inventory. See [calibration objective metadata](../../src/tomojax/geometry/_calibration/objectives.py).

Store one record per metric, method/configuration, case, seed, backend, and
hardware/software environment. Include source revision and dirty-state identity,
units and evaluation masks, physical geometry, actual backend/precision,
initialization, stopping criteria, compilation policy, raw timing samples,
peak-memory definition, quality values, and an explicit success/failure status.

Use four levels of coverage:

1. Small deterministic correctness cases in CPU CI, including boundary and
   invalid-input tests.
2. Real GPU numerical tests and kernel memory checks for each supported route.
3. A representative performance matrix spanning sizes, geometries, noise,
   motion patterns, and algorithms, with paired before/after measurements.
4. Difficult combinations and independent real-data validation: tilted thin
   samples, odd detectors, anisotropy, truncation, motion plus drift, and
   memory-constrained scans.

Cover individual axes and important interactions rather than attempting the
entire Cartesian product. Keep held-out phantoms/seeds so tuning cannot optimize
only the benchmark fixtures. For alignment, the first full scorecard should
include total time to target, peak VRAM, per-DOF error, observable ray error,
reconstruction error, held-out residual, capture range, and failure rate for
pose-only, setup-only, and combined correction in both CT and laminography.
