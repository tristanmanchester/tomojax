# Huber-TV derivative correction, 2026-10-04

Huber-TV updates now have finite forward and reverse derivatives in flat image
regions and at zero dual variables. The previous implementation differentiated
`sqrt(0)` before masking it, producing NaNs even though the enclosing Huber
formula is smooth there. Masking the squared norm before the square root fixes
that derivative without changing the ordinary update values.

Seventeen regression cases check Hessian products against an independently
assembled voxel-edge matrix, piecewise-constant finite differences, the exact
zero-dual proximal derivative, and measured-data/initialization gradients through
the real unrolled and implicit reconstruction layer. The layer controls cover
parallel, anisotropic and tilted geometry. All six layer controls fail with the
old implementation and pass with the correction.

The focused CPU and CUDA selections each passed 22 tests. The full CPU suite
passed **649 tests**, with seven CUDA-dependent skips and 218 GPU deselections;
coverage was **72.81%**. These results establish this correction, not general
pose-recovery accuracy or the differentiation cost/memory targets.

## Every frozen image input

The table covers the input volumes of all 27 reconstruction cells and all six
free-voxel alignment cells. Every new JVP and VJP is finite, every ordinary
Huber update is bitwise equal to the previous update, and the largest relative
Hessian symmetry discrepancy is below 5.71e-8. The old JVP is nonfinite in every
cell; counts below are voxel entries in the first call.

These are **derivative diagnostics**, not reconstruction or alignment runs.
Cold time includes fresh worker startup, import, transfer, compilation, old/new
comparison and verification; warm time is the second diagnostic call. Peak GPU
memory includes the process, sampled at a requested 10 ms interval, which can
miss short allocations. CPU tests overlapped these diagnostics. Times include
both implementations and cannot be used as a solver speedup or ASTRA/TIGRE
comparison. The original full-workflow comparison and its competing-workflow
memory denominators remain in the [frozen matrix](system-matrix-2026-10-04.md).

| Image input | Cold diagnostic s | Warm diagnostic s | Peak process MiB | Old nonfinite entries | New JVP/VJP |
| --- | ---: | ---: | ---: | ---: | --- |
| lamino-clean | 2.359 | 0.0017 | 144 | 24216 | finite / finite |
| lamino-noisy | 0.874 | 0.0039 | 144 | 24216 | finite / finite |
| gaussian-v1--lamino-64-180 | 1.011 | 0.0073 | 156 | 53 | finite / finite |
| structured-noisy-v1--lamino-64-180 | 0.953 | 0.0068 | 156 | 262114 | finite / finite |
| structured-v1--lamino-64-180 | 0.981 | 0.0057 | 156 | 262114 | finite / finite |
| gaussian-v1--lamino-128-180 | 1.103 | 0.0279 | 270 | 595 | finite / finite |
| structured-noisy-v1--lamino-128-180 | 1.086 | 0.0281 | 270 | 2097121 | finite / finite |
| structured-v1--lamino-128-180 | 1.073 | 0.0260 | 270 | 2097121 | finite / finite |
| gaussian-v1--lamino-256-180 | 1.371 | 0.2566 | 1678 | 6012 | finite / finite |
| structured-noisy-v1--lamino-256-180 | 1.395 | 0.2496 | 1678 | 16777197 | finite / finite |
| structured-v1--lamino-256-180 | 1.386 | 0.2596 | 1678 | 16777197 | finite / finite |
| anisotropic-clean | 0.914 | 0.0013 | 144 | 11056 | finite / finite |
| anisotropic-noisy | 0.954 | 0.0035 | 144 | 11056 | finite / finite |
| gaussian-v1--anisotropic-64-180 | 1.019 | 0.0052 | 156 | 17 | finite / finite |
| structured-noisy-v1--anisotropic-64-180 | 1.033 | 0.0052 | 156 | 124893 | finite / finite |
| structured-v1--anisotropic-64-180 | 1.025 | 0.0052 | 156 | 124893 | finite / finite |
| gaussian-v1--anisotropic-128-180 | 1.066 | 0.0161 | 206 | 231 | finite / finite |
| structured-noisy-v1--anisotropic-128-180 | 1.090 | 0.0162 | 206 | 1023969 | finite / finite |
| structured-v1--anisotropic-128-180 | 1.072 | 0.0152 | 206 | 1023969 | finite / finite |
| gaussian-v1--anisotropic-256-180 | 1.261 | 0.1317 | 654 | 2516 | finite / finite |
| structured-noisy-v1--anisotropic-256-180 | 1.269 | 0.1327 | 654 | 8290262 | finite / finite |
| structured-v1--anisotropic-256-180 | 1.297 | 0.1309 | 654 | 8290262 | finite / finite |
| parallel-clean | 0.922 | 0.0029 | 144 | 24216 | finite / finite |
| parallel-noisy | 0.921 | 0.0030 | 144 | 24216 | finite / finite |
| gaussian-v1--parallel-64-180 | 1.057 | 0.0046 | 156 | 53 | finite / finite |
| structured-noisy-v1--parallel-64-180 | 1.045 | 0.0070 | 156 | 262114 | finite / finite |
| structured-v1--parallel-64-180 | 1.054 | 0.0064 | 156 | 262114 | finite / finite |
| gaussian-v1--parallel-128-180 | 1.218 | 0.0263 | 270 | 595 | finite / finite |
| structured-noisy-v1--parallel-128-180 | 1.250 | 0.0429 | 270 | 2097121 | finite / finite |
| structured-v1--parallel-128-180 | 1.209 | 0.0264 | 270 | 2097121 | finite / finite |
| gaussian-v1--parallel-256-180 | 1.502 | 0.2673 | 1678 | 6012 | finite / finite |
| structured-noisy-v1--parallel-256-180 | 1.484 | 0.2717 | 1678 | 16777197 | finite / finite |
| structured-v1--parallel-256-180 | 1.511 | 0.2667 | 1678 | 16777197 | finite / finite |

## Pose finite differences: retained failure and follow-up

An additional FP32 directional check through four regularized unrolled steps
failed its original finite-difference tolerance. All computed derivatives were
finite. At the nominal tilted pose, the sampled integrator's AD directional
value was -4.60618e-4, while the central difference at step 0.001 was -5.98142e-4.
The forward and backward slopes were -4.62867e-4 and -7.33417e-4 respectively.
Their separation persists as the step shrinks, consistent with a sampling or
interpolation boundary; a central slope need not equal the selected local AD
branch there.

The complete follow-up retains both sampled and exact integration at nominal
and randomly offset poses for every geometry, with five step sizes each. The
offset tilted sampled check agrees at step 0.01 (-6.51695e-4 AD versus -6.51716e-4
finite difference), as does nominal exact integration (-9.58770e-5 versus
-9.59262e-5). Smaller steps also show FP32 cancellation, and larger steps can
cross boundaries in other controls. These observations do not convert the
original failure into a pass or prove differentiability at every pose. No
projector, pose-update setting or tolerance was changed to hide it.

The 33 frozen full workflows use zero TV weight and were not rerun for this
fix. Their **26/27 accepted reconstruction comparisons** and **5/6 successful
alignment cells** remain unchanged; the stretch goal remains open.

## Reproduce and inspect

Run `JAX_PLATFORMS=cpu uv run --no-sync pytest -q tests/test_huber_differentiation.py`
for the permanent regressions; use a CUDA-enabled environment without the CPU
platform override for the same accelerator checks. The
[compressed evidence](../bench/reference/huber-derivatives-2026-10-04.json.gz)
contains every cell and repeat, original and follow-up pose failures, source
texts/hashes, and test logs. Original fixture hashes are retained; the full
input fixtures remain local experiment artifacts. Hashes of both compressed and
original evidence bytes are in the [archive catalog](../bench/reference/archives.json).
