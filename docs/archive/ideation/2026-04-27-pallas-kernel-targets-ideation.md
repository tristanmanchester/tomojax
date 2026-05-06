---
date: 2026-04-27
topic: pallas-kernel-targets
focus: "Best TomoJAX codebase targets to rewrite as JAX Pallas kernels for real performance improvements"
mode: repo-grounded
---

# Ideation: Pallas Kernel Targets

## Grounding Context

TomoJAX is a differentiable JAX CT reconstruction and alignment codebase. The dominant repeated work is projection and backprojection over many views in FBP, FISTA-TV, SPDHG-TV, fixed-volume alignment, validation residuals, and calibration-style objectives.

The strongest local hot-path evidence is in `src/tomojax/core/projector.py`:

- `forward_project_view_T` ray-marches a detector image for one pose.
- `_trilinear_gather` performs 8 clipped volume reads and interpolation weights per ray step.
- `_backproject_view_accum_T` and `_trilinear_scatter_add` implement the explicit adjoint by scatter-like accumulation through `jax.linear_transpose`.
- `sum_backproject_views_T` scans over views to keep memory bounded while accumulating one volume.

The main consumers are:

- `src/tomojax/recon/fbp.py`: FFT filtering followed by repeated backprojection.
- `src/tomojax/recon/fista_tv_core.py`: `_projection_loss_and_explicit_grad` does `project -> residual -> sum_backproject_views_T` every FISTA iteration.
- `src/tomojax/recon/spdhg_tv.py`: SPDHG block updates and norm estimation repeatedly apply forward and adjoint operators.
- `src/tomojax/align/objectives.py`: fixed-volume alignment scores pose chunks via `forward_project_view_T`.
- `src/tomojax/align/validation_residuals.py`: validation normals build residual chunks, linearize them, and accumulate `loss`, `J^T r`, and `J^T J`.

Existing validation hooks are good enough to support kernel experiments:

- `tests/test_projector.py` covers path length, non-cubic rotated geometry, origin handling, gather dtype forwarding, explicit adjoint parity, and laminography cases.
- `tests/test_projector_precision.py` covers bf16 gather tolerance.
- `tests/test_fbp_batching.py`, `tests/test_recon.py`, `tests/test_spdhg.py`, and `tests/test_tv_ops.py` cover downstream effects.
- Benchmark profiles include `bench/profiles/screen_speed_parallel_fbp_128.yaml`, `bench/profiles/screen_memory_parallel_fista_128.yaml`, `bench/profiles/canary_iterative_parallel_160.yaml`, and `bench/profiles/canary_align_parallel_3d_128_noisy.yaml`.

Prior repository learning matters: `docs/solutions/architecture-patterns/reuse-align-multires-for-geometry-calibration-2026-04-25.md` says previous performance failures were not just missing low-level kernels; they were also objective-shape and memory-contract failures. Pallas work should preserve the streamed/chunked contracts and avoid fold-wide prediction, residual, or Jacobian materialization.

External grounding: the current JAX Pallas docs describe Pallas as an experimental custom-kernel layer requiring explicit memory/block scheduling. On GPU, the serious path is Mosaic GPU, with Hopper-and-newer support called out in the quickstart; the Triton backend exists but is best-effort. Pallas is strongest where manual tiling/fusion reduces HBM pressure or exposes schedules XLA cannot infer. Tomography libraries such as ASTRA, TIGRE, and TomocuPy treat forward/backprojection as core GPU acceleration primitives, which matches this codebase's hot paths.

## Ranked Ideas

### 1. Backend-Gated Ray-Major Forward Projector Fast Path

**Description:** Add an experimental Pallas implementation of `forward_project_view_T` where each program owns a tile of detector rays for one view, computes traversal locally, performs trilinear volume gathers, accumulates in registers, and writes one detector tile. Keep the current JAX implementation as the default oracle and fallback.

**Warrant:** `direct:` `forward_project_view_T` is called by FISTA, SPDHG, alignment objectives, validation residuals, simulation, and loss benchmarks. Its inner loop calls `_trilinear_gather`, which performs 8 clipped volume reads plus interpolation math at every ray step inside `lax.scan`.

**Rationale:** This is the cleanest first Pallas target because the output ownership model is natural: one ray/detector pixel produces one scalar. It avoids the hardest backprojection race problem while testing whether explicit detector tiling, fixed loop structure, and local traversal state beat the current `scan` plus `vmap` lowering. If it wins, the benefit compounds across nearly every reconstruction and alignment workflow.

**Downsides:** Forward-only speedup does not accelerate FBP backprojection-heavy work by itself. It also needs backend gating because Pallas GPU support is hardware-sensitive. A custom VJP story is required before replacing differentiable paths broadly; until then, use it only where the forward image or scalar loss path is enough.

**Confidence:** 86%

**Complexity:** Medium

**Status:** Unexplored

**Validation:** Compare against current `forward_project_view_T` on `tests/test_projector.py` and `tests/test_projector_precision.py`; benchmark with `screen_memory_parallel_fista_128`, `canary_iterative_parallel_160`, and `canary_align_parallel_3d_128_noisy`. Record compile time separately from warm-run time.

### 2. Voxel-Driven Backprojection and Chunked Sum-Backprojection

**Description:** Build a voxel-owned Pallas backprojection path for `backproject_view_T` and eventually `sum_backproject_views_T`: each program owns an output voxel tile, maps voxels into detector coordinates for one or more views, gathers detector values, accumulates locally, and writes each voxel once.

**Warrant:** `direct:` `_backproject_view_accum_T` currently uses `_trilinear_scatter_add`, implemented via `jax.linear_transpose` of `_trilinear_gather`; `sum_backproject_views_T` scans this over views. These functions are central to FBP, FISTA explicit gradients, SPDHG updates, and norm estimation.

**Rationale:** Backprojection is probably the highest-upside target after the forward projector. The important reframing is not "write the current scatter in Pallas"; it is "make the output voxel own the computation." That avoids the worst atomic/scatter risk and matches Pallas's preference for disjoint output blocks. A chunked version could preserve TomoJAX's existing `views_per_batch` memory contract while reducing repeated per-view volume accumulation overhead.

**Downsides:** Geometry is more subtle than forward ray marching, and exact adjoint parity is non-negotiable. A voxel-driven formulation may have less straightforward memory coalescing over detector reads and may need different kernels for parallel CT, laminography, and pose-aware cases.

**Confidence:** 79%

**Complexity:** High

**Status:** Unexplored

**Validation:** Pass `<Ax, y> ~= <x, A^T y>` tests in `tests/test_projector.py`, FBP batch equivalence in `tests/test_fbp_batching.py`, and quality checks in `tests/test_recon.py` / `tests/test_spdhg.py`. Benchmark `screen_speed_parallel_fbp_128`, `screen_memory_parallel_fista_128`, and `canary_iterative_parallel_160`.

### 3. Fused Project-Residual-Loss Kernel

**Description:** Add a fused Pallas path for callers that only need scalar or per-view objective values: ray-march, load measured detector data, apply masks/weights, compute residuals, and reduce loss without materializing the predicted projection chunk.

**Warrant:** `direct:` `fista_tv_core._projection_loss`, `align.objectives.project_and_score_stack`, and `validation_residuals.residual_chunk` all follow the pattern `pred = project(...)`, then subtract targets, weight/mask, and reduce or reshape residuals.

**Rationale:** This targets memory traffic rather than just arithmetic. In alignment and validation, the projected image is often an intermediate, not the product. A fused path follows the same idea as online accumulation kernels: keep the large transient in registers/local reductions and write only compact outputs. It also aligns with the prior repository lesson to avoid fold-wide prediction materialization.

**Downsides:** It is consumer-specific and less reusable than a standalone projector. It must support the loss-adapter surface carefully; start with weighted L2 or a single Gauss-Newton-compatible loss before attempting robust/masked variants.

**Confidence:** 77%

**Complexity:** Medium

**Status:** Unexplored

**Validation:** First target weighted L2. Compare scalar losses against `_projection_loss` and `project_and_score_stack`; benchmark objective-evaluation time and peak memory on `canary_align_parallel_3d_128_noisy` and `screen_memory_parallel_fista_128`.

### 4. Validation Normal-Equation Streaming Kernel

**Description:** Rewrite the validation normal accumulation in `accumulate_validation_normals` so a Pallas kernel streams residuals and directly accumulates `loss`, `J^T r`, `J^T J`, and residual counts for a small active parameter dimension.

**Warrant:** `direct:` `src/tomojax/align/validation_residuals.py` currently builds residual chunks, calls `jax.linearize`, vmaps columns of the linearized residual, and reduces into a small gradient/Hessian output.

**Rationale:** This is a strong "many residuals, tiny output" Pallas shape. It is narrower than replacing the full differentiable projector, but directly targets a path the prior solution identified as architecturally important: streamed validation residual/JVP accumulation with no large Jacobian materialization. The active dimension is small, so direct accumulation could cut residual materialization and repeated column passes.

**Downsides:** It requires careful derivative math. A naive implementation can easily become a bespoke alignment kernel that is hard to maintain. It should be limited to one loss adapter and one active-parameter shape first.

**Confidence:** 71%

**Complexity:** High

**Status:** Unexplored

**Validation:** Compare `loss`, `grad`, `hess`, and `residual_count` against `accumulate_validation_normals` on fixed seeds and active dimensions. Measure validation fold runtime and memory, not only kernel microbenchmarks.

### 5. FISTA Data-Gradient / Normal-Equation Chunk Kernel

**Description:** Fuse `_projection_loss_and_explicit_grad` for one weighted-L2 chunk: compute `A x`, residual, loss, and `A^T residual` in a Pallas-backed path so FISTA does not materialize `pred` and `grad_resid` as separate detector arrays.

**Warrant:** `direct:` `src/tomojax/recon/fista_tv_core.py` currently computes `pred = vm_project(T_chunk)`, builds `raw_resid`, `weighted_resid`, `grad_resid`, then calls `sum_backproject_views_T` for every chunk inside every FISTA iteration.

**Rationale:** This is the highest-leverage iterative-reconstruction target after the base projector/backprojector. It directly attacks the `project -> residual -> backproject` sequence that dominates data-gradient work. It can also become the reusable normal-equation action for future bilevel or learned objectives.

**Downsides:** It combines the hard parts of both forward and backprojection. If implemented as ray-driven scatter, it may inherit atomic/write-conflict problems. It is better as a second-wave target after at least one standalone forward or voxel-backprojection kernel has been proven.

**Confidence:** 68%

**Complexity:** High

**Status:** Unexplored

**Validation:** Compare `(loss, grad)` against `_projection_loss_and_explicit_grad`, then compare FISTA loss history and reconstruction quality. Benchmark `screen_memory_parallel_fista_128` and `canary_iterative_parallel_160` with fixed `views_per_batch`.

### 6. SPDHG Data-Block Update Kernel

**Description:** Specialize the SPDHG data step in `spdhg_tv.one_step`: compute `pred = A x_bar`, update the detector-space dual `y_data`, form `delta_y`, and apply `A^T delta_y` for the selected block in one fused or paired Pallas path.

**Warrant:** `direct:` `src/tomojax/recon/spdhg_tv.py` performs a tightly coupled data dual update followed immediately by `sum_backproject_views_T` inside each stochastic block update.

**Rationale:** SPDHG has a natural fixed block shape (`views_per_batch`), so it is a good candidate for chunk-specialized kernels. If TomoJAX uses SPDHG heavily, this could reduce detector intermediate churn and operator dispatch overhead per stochastic step.

**Downsides:** It is algorithm-specific, and the current benchmark profiles appear more FBP/FISTA/alignment-oriented than SPDHG-first. Its value depends on SPDHG being important in actual workloads.

**Confidence:** 58%

**Complexity:** High

**Status:** Unexplored

**Validation:** Compare `x`, `y_data`, `s`, and loss behavior for fixed seeds; use `tests/test_spdhg.py` and a SPDHG-specific benchmark profile before promoting this above FISTA/alignment targets.

### 7. Projector Backend Interface with Golden-Adjoint Benchmark Harness

**Description:** Add a narrow experimental backend dispatch surface for `jax` vs `pallas` implementations of forward projection, backprojection, sum-backprojection, and fused loss/gradient kernels, plus a benchmark harness that records backend, device, compile time, warm-run time, memory, and numerical deltas.

**Warrant:** `external:` JAX Pallas is experimental, Mosaic GPU support is hardware-specific, and the Triton GPU backend is best-effort. `direct:` this repo already has tests and benchmark profiles that can serve as gates.

**Rationale:** This is not a performance kernel, but it is the enabling target that makes the real kernels safe. It prevents Pallas conditionals from leaking through FBP, FISTA, SPDHG, alignment, and validation code. It also makes "real performance improvement" measurable rather than anecdotal.

**Downsides:** It does not improve runtime by itself. If built too broadly before a real kernel exists, it becomes premature abstraction. Keep it minimal: enough dispatch and measurement to support one forward-projector experiment.

**Confidence:** 82%

**Complexity:** Medium

**Status:** Unexplored

**Validation:** Default remains current JAX behavior. Pallas only activates for explicitly supported devices/shapes. Every Pallas path must report parity and benchmark deltas against the current implementation on named profiles.

## Rejection Summary

| # | Idea | Reason Rejected |
|---|------|-----------------|
| 1 | Detector-tile trilinear interpolation microkernel | Duplicates the stronger forward-projector target; too narrow to prove end-to-end improvement alone. |
| 2 | Explicit ray-driven scatter backprojection | Reconsidered; see `docs/plans/2026-04-29-002-feat-pallas-backprojection-plan.md` for ray-owned atomic-scatter prototype as first candidate. |
| 3 | Two-phase tiled ray backprojection | Plausible fallback, but more complex and less clean than voxel-driven first. |
| 4 | FBP-only filtered-backprojection kernel | Valuable only through backprojection; covered by the more general voxel/chunked backprojection idea. |
| 5 | Pose-derivative projection kernel | Too ambitious before projection/residual kernels are proven; derivative surface is high risk. |
| 6 | Generic projector-consumer microkernels | Good strategic direction, but too broad and not actionable enough as a first target. |
| 7 | Traversal-state prelude/cache kernel | Interesting implementation detail, but likely secondary unless profiling shows traversal setup dominates. |
| 8 | Geometry-specialized/generated kernel family | Better as rollout/tuning mechanism after a concrete kernel wins. |
| 9 | TV stencil/update kernel | Second-tier; XLA likely fuses simple stencil/elementwise code adequately until projector/backprojector costs fall. |
| 10 | FFT filtering or phase-correlation kernels | Poor Pallas target; vendor FFT paths are likely already the right implementation. |
| 11 | Generic alignment loss kernels | Less central than fused projection/residual paths and often reducible to standard elementwise/reduction operations. |
| 12 | Standalone preprocessing, masks, scalar weights, dense linear algebra | Too far from the measured hot path or already library/XLA-covered. |

## Recommended Next Brainstorm Seed

Start with **Backend-Gated Ray-Major Forward Projector Fast Path**. It has the cleanest output ownership, the broadest downstream reach, and the least risk of scatter/adjoint correctness problems. The brainstorm should define the first supported shape/device contract, fallback behavior, benchmark gates, and how to keep the current JAX projector as the oracle.

## Sources

- JAX Pallas Quickstart: https://docs.jax.dev/en/latest/pallas/quickstart.html
- JAX Pallas design notes: https://docs.jax.dev/en/latest/pallas/design/design.html
- JAX Pallas GPU reference: https://docs.jax.dev/en/latest/pallas/gpu/reference.html
- JAX Pallas pipelining: https://docs.jax.dev/en/latest/pallas/pipelining.html
- JAX Pallas changelog: https://docs.jax.dev/en/latest/pallas/CHANGELOG.html
- MaxText Pallas optimization guide: https://maxtext.readthedocs.io/en/latest/guides/optimization/pallas_kernels_performance.html
- ASTRA BP_CUDA: https://astra-toolbox.com/docs/algs/BP_CUDA.html
- ASTRA FP_CUDA: https://sourceforge.net/p/astra-toolbox/wiki/FP_CUDA/
- TIGRE: https://github.com/CERN/TIGRE
- TomocuPy paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC9814072/
- TomocuPy performance docs: https://tomocupy.readthedocs.io/en/latest/performance.html
- CUDA CT projection/backprojection paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC4664243/

