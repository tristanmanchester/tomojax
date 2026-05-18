# Changelog

## Unreleased
- Reduce the repository to a publishable product spine: public CLI/API, minimal examples, and focused product tests only.
- Remove benchmark harnesses, historical/v1-parity gates, one-off runners, oversized development logs, and diagnostic artifact builders from the shipped tree; retain them in a separate development archive.
- Remove `tomojax.bench` and `tomojax.verify` entirely from the shipped tree.
- Laminography geometry now aligns the rotation axis with the volume z-axis, so lamino reconstructions produce z-stacks with square x–y slices; regenerate datasets/recons if you relied on the old orientation.
- Recon CLI now crops to detector FOV by default (`--roi auto`); pass `--roi off` to keep legacy behavior.
- Normalize NX volume IO: write volumes on disk in `zyx` order with `@volume_axes_order` metadata, transpose on load, and warn (silence via `TOMOJAX_AXES_SILENCE`).
- Add CLI `--volume-axes` override for `recon`/`align` and update NX data wrangler to tag volumes.
- Fix CUDA “invalid image” faults on Turing GPUs by replacing the projector’s small GEMM with element-wise transforms, ensuring SPDHG/FWD projections compile cleanly on RTX 6000/8000 while keeping mixed-precision gather heuristics.

## 0.2.0 — v2 at repo root
- Promote the v2 implementation to the primary package at `tomojax`.
- Move v2 code under `src/tomojax`, tests under `tests`, docs under `docs`.
- Add root `pyproject.toml` (src/ layout) and update `pixi.toml` tasks.
- Remove legacy root examples and experimental folders.
