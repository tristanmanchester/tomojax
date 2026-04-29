---
title: feat: Design Pallas Backprojection Prototype
type: feat
status: planned
date: 2026-04-29
origin: docs/plans/2026-04-29-001-feat-pallas-v2-optimization-plan.md
---

# feat: Design Pallas Backprojection Prototype

## Summary

Design the first opt-in Pallas backprojection experiment as a narrow,
non-default adjoint prototype. The goal is not to replace
`backproject_view_T` immediately; the goal is to test whether Pallas can
accelerate the scatter-heavy adjoint used by FBP, FISTA-TV, SPDHG, and data
term gradients while preserving the existing explicit VJP contract.

The first viable implementation candidate is a ray-owned Pallas kernel that
reuses the forward traversal semantics and atomically scatters detector samples
into the volume with trilinear weights. Pallas exposes Triton atomic operations
through `jax.experimental.pallas.triton.atomic_add`, so a bounded prototype is
technically feasible. It should still be gated carefully because atomics,
output initialization, and update collisions are the hard part of this kernel.

## Current Backprojection Contract

- `src/tomojax/core/projector.py` owns the JAX oracle.
- `_projector_traversal_state(...)` is shared by forward and adjoint passes.
- `backproject_view_T(...)` calls `_backproject_view_accum_T(...)`.
- `_backproject_view_accum_T(...)` scans backward over the same ray samples,
  multiplies each detector value by `active * step_size`, and calls
  `_trilinear_scatter_add(...)`.
- `_trilinear_scatter_add(...)` is implemented as the linear transpose of
  `_trilinear_gather(...)`, so it is intentionally the explicit discrete
  adjoint of the forward projector.
- `sum_backproject_views_T(...)` sums per-view adjoints and is used by
  reconstruction and optimization paths.

Existing consumers:

- `src/tomojax/recon/fbp.py`: filtered backprojection calls
  `backproject_view_T` through `_bp_one` and chunked scans.
- `src/tomojax/recon/fista_tv.py`: data-term gradients call
  `backproject_view_T` or `sum_backproject_views_T`.
- `src/tomojax/recon/fista_tv_core.py` and `src/tomojax/recon/spdhg_tv.py`:
  iterative reconstruction hot paths call `sum_backproject_views_T`.
- Alignment objectives and tests rely on the same forward/adjoint consistency.

## Non-Goals

- Do not replace `backproject_view_T` or `sum_backproject_views_T` by default.
- Do not use Pallas backprojection in differentiated callers.
- Do not claim reconstruction speedup from a single-view microbenchmark.
- Do not support arbitrary calibrated detector grids in v1.
- Do not implement voxel-owned backprojection first unless the ray-owned
  atomic scatter prototype fails specifically because of atomic contention.
- Do not support lower-precision accumulation in v1. Use fp32 image input,
  fp32 atomic accumulation, and fp32 output.

## Candidate A: Ray-Owned Atomic Scatter

### Surface

Add a new experimental module or extend `src/tomojax/core/pallas_projector.py`
with:

```python
def backproject_view_T_pallas(
    T: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    image: jnp.ndarray,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    unroll: int | None = None,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    interpret: bool = False,
    tile_shape: tuple[int, int] = (8, 16),
    num_warps: int = 4,
    kernel_variant: str = "auto",
    layout_variant: str = "detector_vu",
) -> jnp.ndarray:
    ...
```

Initial supported subset:

- `image.dtype` castable to fp32, output fp32.
- `gather_dtype="fp32"` only for v1 backprojection, despite forward Pallas
  supporting lower-precision gather. This avoids mixing precision questions
  with atomic-add questions.
- Canonical detector grid only: `None` or `get_detector_grid_device(detector)`.
- Real Pallas only on GPU; CPU tests use `interpret=True`.
- Single view only. Multi-view summation is a benchmark wrapper, not the first
  kernel.

### Kernel Algorithm

Use the same detector-tiled ownership pattern as the forward projector:

```text
grid = (ceil(nv / tile_v), ceil(nu / tile_u))
```

Each program owns one detector tile. For every detector pixel:

1. Compute the same ray state as `forward_project_view_T_pallas`, or consume a
   cached traversal state in a later candidate.
2. Load the detector image scalar `ray_val`.
3. Iterate over the same effective global `n_steps`.
4. Apply `active = step_idx < n_steps_ray`.
5. Compute the eight trilinear neighbor indices and weights using the same
   floor, clip, and explicit in-bounds semantics as `_trilinear_gather`.
6. Atomically add `ray_val * active * step_size * weight` into the flat output
   volume for each in-bounds neighbor.

The kernel must match this flat layout:

```text
flat_index = ix * (ny * nz) + iy * nz + iz
```

Output initialization must be explicit. Preferred implementation:

- Pass `jnp.zeros((nx * ny * nz,), dtype=jnp.float32)` as an aliased input.
- Use `pallas_call(..., input_output_aliases={input_index: 0})` so atomics
  update a zeroed output buffer.
- Return the flat output reshaped to `(nx, ny, nz)`.

Do not rely on uninitialized output refs being zero.

### Variant Policy

Reuse the existing forward Pallas support predicate where possible:

- `kernel_variant="auto"` may select `z_integer4` for the interpolation
  weights if the same predicate is valid for the adjoint.
- The first implementation may also force `kernel_variant="generic"` if
  `z_integer4` scatter complicates correctness.
- Any explicit unsupported fast path should raise
  `PallasProjectorUnsupported`; benchmark wrappers record fallback metadata.

## Candidate B: Voxel-Owned Gather From Detector

This is not the first prototype, but it is the fallback direction if atomics are
too slow.

Each Pallas program owns a volume tile and computes which detector samples
contribute to each voxel. This can avoid atomics but is substantially more
geometry-specific and harder to match the exact discrete adjoint because the
current operator is defined by ray sample positions and trilinear scatter,
not by analytic voxel-to-detector integration.

Use this only after Candidate A proves atomics are the bottleneck and after a
separate derivation confirms exact or acceptable adjoint semantics.

## Tests

Add `tests/test_projector_pallas_backprojection.py` or extend
`tests/test_projector_pallas.py` with CPU `interpret=True` tests:

- Uniform aligned case: Pallas backprojection shape and values match JAX.
- Random small parallel case: Pallas matches `backproject_view_T`.
- Rotated non-cubic case: Pallas matches JAX within fp32 tolerance.
- Remainder detector tile case, e.g. `nu=7`, `nv=5`.
- Explicit traversal controls: `step_size=0.5`, explicit positive `n_steps`.
- Canonical detector grid accepted.
- Noncanonical detector grid rejected with controlled unsupported error.
- Invalid `gather_dtype` or non-fp32 mode rejected in v1.
- Adjoint smoke: `<A x, y>` equals `<x, A_T_pallas y>` for a small fixture
  against `forward_project_view_T`.

Do not route reconstruction tests through Pallas in v1.

## Benchmark Design

Create a dedicated benchmark rather than overloading the forward benchmark:

- `src/tomojax/bench/backprojector.py`
- `bench/backprojector.py`
- `tests/test_bench_backprojector.py`

Modes:

- `jax_single`: existing `backproject_view_T`.
- `pallas_single`: new `backproject_view_T_pallas`.
- `jax_sum`: existing `sum_backproject_views_T`.
- `pallas_loop_sum`: Python/JAX loop over single-view Pallas, only as a
  dispatch/control mode.

Suites:

- `quick`: one high-ray `128^3` single-view case.
- `confirm`: `profile-128`, `noncubic-align-128`, `high-ray-count-128`.
- `stress`: mirror forward `stress` shapes.
- `sum`: multi-view `64`, `128`, and high-ray `128` residual-like shapes.

Metrics:

- Actual backend/mode.
- Pallas eligibility and fallback reason.
- Requested and actual Pallas variant metadata.
- Max absolute error and relative norm error versus JAX backprojection.
- Adjoint relative error for one small case.
- Median warm time and speedup versus JAX.

## Acceptance Gates

Candidate A is accepted only if all are true on vivobook:

- CPU `interpret=True` tests pass.
- GPU benchmark rows are Pallas-eligible and parity-clean.
- `quick` high-ray single-view backprojection is at least `1.20x` faster than
  JAX.
- `confirm` geomean is at least `1.05x`.
- No confirm case is below `0.90x`.
- Multi-view sum benchmark shows either a credible win or a clear reason why a
  batched backprojection kernel is needed next.

If Candidate A is parity-clean but slower, document whether the bottleneck is
atomic contention, output zeroing, or traversal recomputation before moving to
Candidate B.

## Risks

- Atomic contention can dominate because many rays update the same voxels.
- Floating-point atomic order can introduce nondeterministic last-bit
  differences; tolerances must be honest and stricter than reconstruction
  quality would require.
- Pallas output aliasing mistakes can turn stale memory into apparent signal.
- `z_integer4` forward speedups may not transfer to scatter if the bottleneck
  is atomics rather than neighbor count.
- Multi-view reconstruction workflows may need a view-batched or chunked
  Pallas backprojection strategy even if single-view wins.

## Next Step

Implement Candidate A only as an opt-in experimental function plus a
single-view benchmark. Do not integrate it into FBP, FISTA-TV, SPDHG, or
alignment code until the benchmark gate clears.
