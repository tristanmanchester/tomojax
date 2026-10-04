# Reusing compiled free-voxel alignment objectives

Reusing the coupled objective removes repeated compilation and reduces warm alignment time in all six pilot cells. It does not improve cold startup or memory, and noisy anisotropic alignment still fails the fixed rotation gate. This bounded refactor is complete; further tuning of this line stops here. The whole-system optimization goal remains open.

## Complete workflow comparison

Both revisions use public `tomojax.align.align` with free voxels, exact projection, central pose differences and pose elimination. Each cell runs in an isolated process with one cold call and seven warm calls. Cold time includes process startup, imports, setup, transfers, compilation and verification. Warm time is the median of seven verified calls. NVIDIA process memory is sampled every 10 ms across the complete worker, so short-lived peaks can be missed.

All six fixture hashes and solver configurations match. The objects, irregular views, positivity constraint, iteration limits and acceptance gates are unchanged from the [public Schur pilot](public-free-voxel-schur-2026-10-04.md). Failed-attempt timings remain in the table but are not accepted-result speedups.

| Cell | Accepted calls, before / after | Cold s, before / after | Warm median s, before / after | Peak process GPU MiB, before / after |
|---|---:|---:|---:|---:|
| parallel-clean | 8/8 / 8/8 | 9.914 / 9.988 | 5.015 / 2.925 | 320 / 320 |
| parallel-noisy | 8/8 / 8/8 | 10.115 / 10.166 | 5.129 / 2.938 | 320 / 320 |
| anisotropic-clean | 8/8 / 8/8 | 10.468 / 10.639 | 5.547 / 3.391 | 320 / 320 |
| anisotropic-noisy | 0/8 / 0/8 | 49.225 / 49.752 | 44.322 / 42.500 | 320 / 320 |
| lamino-clean | 8/8 / 8/8 | 18.289 / 19.134 | 14.198 / 12.007 | 320 / 320 |
| lamino-noisy | 8/8 / 8/8 | 20.251 / 20.123 | 15.047 / 12.050 | 320 / 320 |

The five accepted cells have warm speedups of 1.18–1.75×. Noisy anisotropic attempts fall from 44.32 to 42.50 seconds, but none succeeds. Cold differences range from −0.13 to +0.85 seconds and do not establish a startup improvement. Tilted runs reach the unchanged gates at different outer iterations across repetitions; the complete trajectories are retained in the raw archive.

Quality below is the worst value across each revision’s eight calls, after one common rigid object-frame registration. The fixed gates are 0.01° rotation RMSE, 0.05 px translation-vector RMSE and relative volume L2 of 0.1 for parallel/anisotropic or 0.2 for laminography. Each quality column shows before / after.

| Cell | Rotation RMSE ° | Translation-vector RMSE px | Volume relative L2 |
|---|---:|---:|---:|
| parallel-clean | 0.004042 / 0.003951 | 0.000113 / 0.000114 | 0.002325 / 0.002329 |
| parallel-noisy | 0.008383 / 0.008378 | 0.000251 / 0.000251 | 0.005343 / 0.005343 |
| anisotropic-clean | 0.009292 / 0.009385 | 0.000119 / 0.000119 | 0.001507 / 0.001496 |
| anisotropic-noisy | 0.016394 / 0.016345 | 0.000384 / 0.000386 | 0.003484 / 0.003485 |
| lamino-clean | 0.009958 / 0.009881 | 0.000144 / 0.000146 | 0.087984 / 0.085975 |
| lamino-noisy | 0.009969 / 0.009997 | 0.000245 / 0.000250 | 0.084297 / 0.084907 |

## What changed and what was checked

The old factory created new JIT functions that captured each scan’s measurements. The new module-level compiled functions receive measurements, poses, masks, weights and detector coordinates as array arguments. Only geometry and algorithm choices form the static specification. The linear equations, damping, finite-difference steps, positivity handling and stopping rules are unchanged.

A separate cProfile check on parallel-noisy records 83 backend compilation calls on the first invocation and **zero on the second**; both recoveries pass. Earlier node-level profiles measured 2.13–2.38 seconds of compilation on warm calls. These diagnostic traces are separate from the uninstrumented table above.

Independent dense-model tests vary measurements, nominal poses, support, active parameters and weights at identical shapes. They exercise both joint solvers, cached and streamed pose columns, CPU and CUDA. The complete final source passes 653 CPU tests and 221 CUDA tests (seven and one skips respectively), plus formatting, lint, types and import boundaries.

## Scope and provenance

All 27 frozen reconstruction cells remain controls: 97 reconstruction/core/geometry/forward/backend source files and all 18 benchmark modules are byte-identical. They were not retimed for this alignment-only change. Their existing [per-cell cold/warm time, quality, memory and fastest applicable ASTRA/TIGRE comparisons](system-matrix-2026-10-04.md) remain the reconstruction evidence. No new reconstruction score is inferred from source identity.

The joint comparison is against the recorded public TomoJAX workflow; it does not assert an equivalent ASTRA/TIGRE joint-recovery baseline. Only five of six cells succeed, so there is still no successful whole-pilot denominator for the 20× target, and no evidence here for 99% large-motion recovery.

Baseline is release commit `776a8c9556b2161a6b4b6a0cea5a8249371cc68a`. The source-tree digests are:

- Baseline: `b7a71ca8077909dc344d971d73795204baf5f5b576c6554161e77d1fff7a90ea`.
- Candidate: `af08ea5e3525fd6e1045ec32d941440161c2de880b9c8b7613c65df6e1146b00`.

The environment’s installed distribution metadata still reports 0.2.0 because the development environment was not resynced; workers explicitly import the hashed frozen source snapshots. Source identity, not that stale distribution field, identifies the tested revisions.

The [compressed raw archive](../bench/reference/public-free-voxel-reuse-2026-10-04.json.gz) contains both complete manifests, all 96 call histories, fixture hashes, configurations, source snapshots for changed modules, regression tests and the compilation diagnostic. [The archive catalog](../bench/reference/archives.json) records uncompressed and compressed SHA-256 hashes.
