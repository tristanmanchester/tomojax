# ASTRA Comparison Benchmark

The ASTRA comparison benchmark is for parallel-beam TomoJAX performance and quality checks. It
does not use FDK or cone-beam geometry.

## Modes

- `quick`: local smoke/debug evidence only. Not valid for performance claims.
- `guard`: optimization guard for committed changes. Not publication evidence.
- `publication`: broader evidence for the recorded machine, GPU, driver, Python, and TomoJAX commit.

Every tracked run should record branch, commit, timestamp, note, environment metadata, and an
evidence class. Dirty-worktree runs are exploratory only and should not be compared as real results.

## Timing Policy

JAX dispatch is asynchronous, so benchmark timing must synchronize returned JAX arrays at the timing
boundary. Library functions such as `fbp()` should not call `block_until_ready()` internally because
that breaks JAX transformability and makes the function hostile to `grad`/`jit` use.

Each case reports:

- cold/first-call wall time, including setup, compilation, and cached-call construction where
  applicable
- warm steady-state timing summaries: median, mean, min, and max
- GPU memory peak and peak delta when NVML process memory is available

Cold and warm timings answer different questions. Warm timing is the optimization guard for repeated
workloads. Cold timing shows first-use cost and should be included when making broader benchmark
claims.

## Quality Policy

Each case reports:

- Pallas forward vs TomoJAX JAX forward relative L2 and max absolute error
- ASTRA `parallel3d` forward vs TomoJAX JAX forward relative L2 and max absolute error
- the exact timed FBP path: public `fbp()` or the specialized Pallas parallel-z helper
- specialized/direct TomoJAX FBP vs TomoJAX generic-adjoint FBP relative L2 and max
  absolute error
- specialized/direct TomoJAX FBP MSE/RMSE/PSNR against the phantom
- ASTRA slice-wise `FBP_CUDA` MSE/RMSE/PSNR against the phantom

Direct voxel-domain FBP and generic ray-walking adjoint FBP are not required to match at machine
precision for every discretization. Their difference must be measured and bounded explicitly so a
speedup cannot silently change semantics.

The specialized Pallas FBP helper is not the default public `fbp()` differentiable path. It targets
regular parallel z-axis geometry with detector rows aligned to volume z and integer detector-v
mapping. Benchmark claims that use this helper must be labeled as specialized-helper/backend
results, not as evidence that all public FBP calls have the same performance characteristics.

## Sanity Guards

Pallas changed-input sanity is enabled by default. It must show that changing the input volume or
pose stack changes the output and that the Pallas result still matches the JAX reference within the
declared tolerance.

The optional alignment smoke is a workflow and differentiability guard. It can show that the
configured alignment path still produces finite gradients and a loss drop, but it is not a
pose-recovery accuracy claim.

## Entry Points

The repo-owned entry points are:

- `tomojax-astra-parallel-bench`
- `tomojax-benchmark-suite`
- `tomojax-alignment-smoke-bench`
- `tomojax-pallas-sanity`

Machine-local wrappers may set environment variables, manage result directories, and enforce clean
worktrees, but benchmark semantics should live in the repo-owned Python modules.
