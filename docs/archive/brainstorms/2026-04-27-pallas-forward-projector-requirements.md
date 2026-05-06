---
date: 2026-04-27
topic: pallas-forward-projector
---

# Pallas Forward Projector Spike

## Problem Frame

TomoJAX's forward projector is a shared hot path across reconstruction, alignment, validation, simulation, and loss benchmarking. The current implementation in `src/tomojax/core/projector.py` expresses one view as JAX traversal setup plus a `lax.scan` over samples, with `_trilinear_gather` doing eight clipped volume reads at every ray step.

The goal is not to replace the projector wholesale. The goal is to run a small, backend-gated Pallas spike that answers one question quickly: can a ray-major Pallas kernel beat the current JAX/XLA implementation on realistic forward-projection shapes while preserving numerical parity and keeping the existing JAX path as the oracle and fallback?

This is a technical feature brainstorm. Implementation details are intentionally included where they define the scope and success criteria of the experiment.

---

## Actors

- A1. TomoJAX developer: implements and evaluates the Pallas spike.
- A2. TomoJAX benchmark harness: runs cold/warm timings, memory measurement, and quality checks.
- A3. Existing JAX projector: remains the correctness oracle and fallback backend.
- A4. Pallas runtime/backend: executes only on explicitly supported device/backend combinations.

---

## Key Flows

- F1. Forward-kernel parity check
  - **Trigger:** A developer enables the experimental Pallas forward projector for a supported test shape.
  - **Actors:** A1, A3, A4
  - **Steps:** Build a small grid/detector/pose/volume fixture; run current `forward_project_view_T`; run the Pallas path; compare shape, dtype, finite values, and numerical delta.
  - **Outcome:** The Pallas path either matches within tolerance or is rejected before benchmark claims.
  - **Covered by:** R1, R2, R5, R9

- F2. Quick microbenchmark
  - **Trigger:** A parity-passing Pallas path exists for one static shape family.
  - **Actors:** A1, A2, A3, A4
  - **Steps:** Warm current JAX and Pallas callables; time several blocked forward calls; capture compile/cold time separately from warm time; report speed ratio and max error.
  - **Outcome:** A fast local answer: continue, tune, or stop the Pallas spike.
  - **Covered by:** R6, R7, R8, R10

- F3. Profile-level benchmark
  - **Trigger:** The microbenchmark shows a plausible warm-run win.
  - **Actors:** A1, A2, A3, A4
  - **Steps:** Run the existing benchmark profiles with the current JAX backend and then with the Pallas backend; record runtime, memory, backend metadata, compile behavior, and output quality.
  - **Outcome:** The experiment can claim or reject real workflow impact, not just isolated kernel throughput.
  - **Covered by:** R7, R8, R11, R12

---

## Approaches Considered

**Approach A: Ray-major forward projector only**

Add a Pallas implementation for `forward_project_view_T` only. Each program owns a detector tile, computes ray entry/exit and voxel-space stepping, performs trilinear gathers, accumulates locally, and writes one projection tile.

- Pros: clean output ownership, no scatter races, broad downstream reach, easiest first correctness target.
- Cons: does not accelerate FBP's backprojection-dominated path by itself; differentiable replacement needs later custom derivative work.
- Best suited for: first spike.

**Approach B: Fused project-residual-loss kernel**

Skip producing a detector image for callers that only need scalar losses. Fuse ray marching, target load, residual weighting, and reduction.

- Pros: stronger memory-traffic story for alignment and validation objectives.
- Cons: narrower consumer-specific surface; depends on loss-adapter scope.
- Best suited for: second spike if forward-only speed is plausible but workflow impact is limited by materialized predictions.

**Approach C: Projector backend interface first**

Build backend selection and benchmark plumbing before writing a serious kernel.

- Pros: safer rollout, clean fallback, reusable measurement harness.
- Cons: no performance answer until a kernel exists; risk of premature abstraction.
- Best suited for: a minimal wrapper around Approach A, not a standalone phase.

**Recommendation:** Use Approach A with the smallest useful slice of Approach C. Do not start with fused losses or custom VJPs. A forward-only kernel is the cleanest way to answer whether Pallas is worth carrying in this repo at all.

---

## Requirements

**Experiment scope**

- R1. The spike must add an experimental Pallas forward-projector path for the same conceptual operation as `forward_project_view_T`: one pose, one volume, one detector image.
- R2. The existing JAX implementation must remain the default behavior and the correctness oracle.
- R3. The Pallas path must be opt-in through an explicit backend or experimental flag; unsupported devices, shapes, or configurations must fall back cleanly.
- R4. The first supported target may be intentionally narrow: GPU backend, fixed static grid/detector shapes, `float32` accumulation, and `gather_dtype` support limited to `fp32` plus one lower-precision mode if straightforward.

**Kernel behavior**

- R5. The kernel must preserve the current geometric contract: `T` is `world_from_object`, rays travel along world `+y`, and sampling happens in object-frame voxel coordinates.
- R6. The first kernel should own detector tiles, not individual Python-level rays; the tile shape must be explicit and benchmark-tunable.
- R7. The kernel should compute cheap traversal state locally inside the kernel unless an early benchmark shows precomputed traversal arrays are faster.
- R8. The first version should not attempt to support custom autodiff. If used inside differentiated paths, it must either stay behind non-differentiated benchmark calls or explicitly fall back to the JAX projector.

**Correctness gates**

- R9. The Pallas output must match the existing forward projector within documented tolerances for representative parallel-beam cases before any performance claim is accepted.
- R10. Correctness checks must include at least: uniform-volume path length, localized voxel origin behavior, non-cubic rotated volume behavior, finite values, shape/dtype parity, and `fp32` baseline parity.
- R11. Lower-precision gather behavior must be compared against the existing `gather_dtype` tolerances before being counted as supported.

**Benchmarking**

- R12. The quick benchmark must separate first-call compile/cold time from steady-state warm time using explicit `block_until_ready`.
- R13. The quick benchmark must compare current JAX and Pallas on the same generated fixture, same pose, same volume, same detector grid, same `n_steps`, and same dtype mode.
- R14. The first benchmark sizes should include a tiny correctness size, a smoke size around `32^3` or `64^3`, and one profile-like size derived from existing `128` or `160` profiles.
- R15. Profile-level benchmarking must use existing benchmark profiles before claiming real workflow impact: `bench/profiles/screen_memory_parallel_fista_128.yaml`, `bench/profiles/canary_iterative_parallel_160.yaml`, and `bench/profiles/canary_align_parallel_3d_128_noisy.yaml`.
- R16. Benchmark output must report backend/device, supported/fallback status, compile or first-call time, warm-run mean, max absolute error, relative error, and peak memory where available.

**Decision gates**

- R17. Continue past the spike only if Pallas shows a clear warm-run win on at least one profile-like forward-projection benchmark and no correctness regressions.
- R18. Treat microbenchmark-only wins as insufficient unless at least one downstream profile moves or the result clearly justifies a fused residual benchmark next.
- R19. Stop or defer Pallas if the implementation requires broad backend-specific complexity before showing a measurable win.

---

## Acceptance Examples

- AE1. **Covers R1, R2, R9, R10.** Given a `16^3` uniform volume and an aligned detector, when the JAX and Pallas forward paths run for one view, both return the expected path length image within the existing projector tolerance.
- AE2. **Covers R3, R4.** Given CPU, Apple Silicon, or an unsupported GPU backend, when the experimental backend is requested, TomoJAX falls back to the existing JAX projector and records that fallback rather than failing at import time.
- AE3. **Covers R12, R13, R16.** Given a profile-like `128` shape, when the quick benchmark runs, it prints or writes first-call time, warm-call time, max error, relative error, backend/device, and whether the Pallas path actually ran.
- AE4. **Covers R17, R18.** Given a Pallas microbenchmark win but no improvement in FISTA or alignment profiles, when deciding next steps, the result is treated as inconclusive rather than accepted as a real performance improvement.

---

## Success Criteria

- A developer can answer within one short spike whether a Pallas forward projector is worth deeper investment.
- The experiment never compromises the current projector API or default behavior.
- Correctness is measured against the current JAX implementation before any speed claim.
- Benchmarks distinguish compile/cold cost from steady-state warm runtime.
- The handoff to planning is concrete enough that planning does not need to invent the supported shape contract, fallback behavior, or benchmark gates.

---

## Scope Boundaries

- Do not replace `backproject_view_T`, `_trilinear_scatter_add`, or `sum_backproject_views_T` in this spike.
- Do not implement voxel-driven backprojection in this spike.
- Do not implement a fused residual/loss kernel until the standalone forward projector has been measured.
- Do not implement custom VJP/custom JVP in the first pass.
- Do not make Pallas the default backend.
- Do not claim workflow speedup from a microbenchmark alone.
- Do not optimize FFT filtering, TV stencils, phase correlation, or generic loss functions as part of this work.
- Do not attempt broad geometry support before proving one common profile-like shape.

---

## Key Decisions

- Start with a forward-only Pallas kernel because it has clean output ownership and avoids scatter races.
- Keep the current JAX projector as oracle and fallback because Pallas is experimental and backend-sensitive.
- Benchmark first with a dedicated quick harness, then only use existing profile-level benchmarks if the quick result is promising.
- Record compile/cold time separately because Pallas compile cost can erase practical value in iterative or interactive workflows.
- Treat backend support as a runtime capability check, not as a packaging assumption.

---

## Dependencies / Assumptions

- Pallas is available through the existing `jax>=0.10,<0.11` dependency.
- Current JAX documentation says Pallas is experimental; Mosaic GPU is the serious GPU path and currently targets Hopper-and-newer NVIDIA GPUs, while the Triton backend is best-effort.
- A local development machine may not have a supported GPU, so the implementation and benchmark path must tolerate fallback.
- Existing benchmark profiles and tests are adequate first gates; a dedicated forward-only microbenchmark is still needed for a faster feedback loop.
- The first kernel can ignore differentiability as long as differentiated callers fall back to the current JAX path.

---

## Outstanding Questions

### Resolve Before Planning

- None.

### Deferred to Planning

- [Affects R4][Technical] Which exact backend/capability predicate should enable Pallas: platform string, GPU compute capability, successful smoke compile, or an explicit environment override?
- [Affects R6][Technical] What detector tile shape should the first kernel use, and should tile shape be static config or hardcoded for the first spike?
- [Affects R7][Technical] Is local traversal-state recomputation faster than loading precomputed per-ray traversal arrays for the target shapes?
- [Affects R11][Technical] Should `bf16`/`fp16` gather be supported in the first kernel or deferred until `fp32` speed is proven?
- [Affects R15][Technical] Should the existing `bench/fitness.py` profile runner grow a `projector_backend` config field, or should the first profile run use an environment variable to avoid wider config churn?

---

## Next Steps

-> `/ce-plan` for structured implementation planning.

