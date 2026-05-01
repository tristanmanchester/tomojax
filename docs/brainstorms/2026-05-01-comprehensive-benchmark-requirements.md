---
date: 2026-05-01
topic: comprehensive-benchmark-suite
---

# Comprehensive TomoJAX Benchmark Suite

## Problem Frame

Recent TomoJAX optimization work produced large speedups in FBP and Pallas forward
projection. The benchmark harness now needs to support two different jobs:

- a fast optimization guard that agents can run after each committed change
- an optional publication-evidence suite that gives broader confidence before making
  stronger performance claims

The benchmark must remain commit-stamped, clean-worktree enforced, and hard to game.
It should make real speedups visible while catching broken differentiability, stale/cached
outputs, quality regressions, and alignment workflow breakage.

## Key Decisions

- Keep the existing tracked 128³ ASTRA comparison as the default optimization guard.
- Add a separate suite runner instead of making the default command unexpectedly slow.
- Include a Pallas changed-input sanity probe so cached setup cannot be confused with cached
  projection output.
- Keep the full five-DOF 24³ alignment smoke optional, not part of every hot-loop run.
- Use bundle directories for suite runs so publication artifacts do not become a flat-file mess.

## Benchmark Modes

### Quick

Purpose: local smoke/debug.

- One 64³ parallel-beam case.
- One warmup and one measured repeat.
- Includes Pallas changed-input sanity by default.

### Guard

Purpose: agent optimization guard.

- Headline 128³ parallel-beam case.
- One smaller 64³ shape sanity case.
- Pallas changed-input sanity by default.
- Optional full-pose alignment smoke via an explicit flag.

### Publication

Purpose: robust evidence before making performance claims.

- 64³, 128³, and 192³ scaling cases.
- More warmups and repeats than guard mode.
- Pallas changed-input sanity by default.
- Optional full-pose alignment smoke once per suite.
- Produces a bundle directory with per-case JSON, Markdown, timing CSVs, quality CSVs,
  logs, aggregate `cases.csv`, `suite.json`, and `summary.md`.

## Required Metrics

Each case reports:

- TomoJAX JAX forward median/mean/min/max runtime
- TomoJAX Pallas forward median/mean/min/max runtime
- ASTRA `parallel3d` forward median/mean/min/max runtime
- TomoJAX FBP median/mean/min/max runtime
- ASTRA slice-wise `FBP_CUDA` median/mean/min/max runtime
- GPU memory peak and peak delta where NVML is available
- Pallas-vs-JAX forward relative L2 and max error
- ASTRA-vs-JAX forward relative L2 and max error
- TomoJAX FBP MSE/RMSE/PSNR against the phantom
- ASTRA FBP MSE/RMSE/PSNR against the phantom

## Guardrails

- Refuse dirty TomoJAX worktrees by default.
- Record branch, commit, note, timestamp, and environment metadata.
- Treat dirty exploratory runs as invalid for tracked comparisons.
- Do not use FDK; the benchmark is parallel beam.
- Do not use ASTRA inside TomoJAX code paths.
- Do not claim pose recovery accuracy from the alignment smoke; use it as a workflow guard.
- Do not claim publication-grade evidence from quick or guard mode.

## Acceptance Examples

- A developer can run `~/tomojax-bench/run_tomojax_benchmark_suite.sh` and get a
  commit-stamped guard bundle under `results/`.
- A developer can run `TOMOJAX_BENCH_MODE=publication
  ~/tomojax-bench/run_tomojax_benchmark_suite.sh` and get a multi-case publication bundle.
- Pallas changed-input sanity fails if the output does not change when the input volume or pose
  changes.
- Optional alignment smoke records wall time, recon time, align time, loss drop, and MSE
  improvement in the suite summary.

## Scope Boundaries

- Non-cubic and detector-mismatch matrices are deferred until ASTRA comparison semantics are
  defined for those cases.
- Pose-recovery metrics are deferred until gauge handling and comparison conventions are
  explicitly designed.
- Publication mode is evidence for the tested laptop/GPU environment, not a universal hardware
  claim.
