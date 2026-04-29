---
title: FISTA Square Artifacts From Uncentered Laminography Projections
date: 2026-04-29
category: logic-errors
module: real-laminography-alignment
problem_type: logic_error
component: tooling
symptoms:
  - "FISTA checkpoint slices show large square or blocky low-frequency artifacts"
  - "Coarse pyramid z-stack previews look like repeated square fields rather than the TEM grid"
  - "FBP baseline looks plausible while FISTA alignment previews regress badly"
root_cause: config_error
resolution_type: config_change
severity: high
tags: [fista-tv, laminography, preprocessing, background-correction, checkpoint-previews, real-data]
---

# FISTA Square Artifacts From Uncentered Laminography Projections

## Problem

A cropped real laminography TEM-grid run produced FISTA checkpoint previews with
large square, blocky artifacts. The FBP baseline still showed the grid structure,
so the first suspicion was that the preview renderer, slab z mapping, or checkpoint
resampling was corrupting the images.

The failing run used the correct centered slab geometry but launched FISTA/alignment
with projection background correction disabled.

## Symptoms

- FISTA checkpoint slices showed a dominant square or box-like structure instead of
  the TEM grid.
- Level 4 z-stack tiles repeated the same coarse content across adjacent global-z
  labels.
- The baseline `00_baseline/orthos.png` looked plausible, while
  `01_setup_geometry/01_cor/timeline_z/...` previews looked wrong.
- The run was numerically alive: losses decreased and geometry estimates moved
  gradually, so this was not an obvious optimizer crash or NaN failure.

## What Didn't Work

- Treating the issue as a slab-centering mistake did not explain the square artifact.
  The corrected run manifest had `slab_center_global_z=148`,
  `preview_global_z=148`, and `preview_local_z=48`, so the preview plane was centered.
- Treating the issue as PNG display scaling was incomplete. A controlled render from
  the raw checkpoint showed that level-native checkpoints were coarse
  (`32x32x12` at level 8 and `64x64x24` at level 4), but the same low-frequency
  square structure remained after proper linear upsampling.
- Treating the repeated z-stack labels as the whole bug was also incomplete. Coarse
  pyramid previews naturally repeat nearby global-z labels because several full-res
  global z positions map to the same coarse slice. That explains some preview
  blockiness, not the strong square DC artifact.

## Solution

Run real laminography FISTA/alignment on zero-centered projection residuals. For
these cropped and flat/dark-corrected real datasets, keep edge-median subtraction
enabled unless the input has explicitly been converted to a zero-background
attenuation residual.

Bad launch:

```bash
uv run python scripts/real_laminography/run_real_lamino_native_setup_pose_256.py \
  --input data/cropped_lamino_256/k11-67208_tem_grid_highview_crop256_corrected_512views_runner.nxs \
  --slab-center-z 148 \
  --preview-z 148 \
  --slab-nz 96 \
  --projection-background none \
  --no-recon-positivity
```

Corrected launch:

```bash
uv run python scripts/real_laminography/run_real_lamino_native_setup_pose_256.py \
  --input data/cropped_lamino_256/k11-67208_tem_grid_highview_crop256_corrected_512views_runner.nxs \
  --slab-center-z 148 \
  --preview-z 148 \
  --slab-nz 96 \
  --projection-background edge_median \
  --background-edge-px 16 \
  --no-recon-positivity
```

The corrected run preserved the same centered slab and geometry convention but
changed the projection preprocessing:

```json
{
  "projection_preprocessing": {
    "background_mode": "edge_median",
    "background_edge_px": 16,
    "alignment_and_fista_use": "background_corrected_projections"
  }
}
```

The important diagnostic is the manifest's working projection statistics. In the
bad run, raw and working projections were identical, with view medians around
`0.20-0.25`. In the corrected run, edge offsets around `0.20-0.25` were removed
and working view medians were near zero:

```json
{
  "working_projection_stats": {
    "percentiles": [-0.085, -0.071, -0.057, -0.000086, 0.068, 0.078, 0.090],
    "view_median_percentiles": [-0.0049, -0.0035, -0.00024, 0.0038, 0.0089]
  }
}
```

## Why This Works

Flat/dark correction makes detector response comparable across views, but it does
not guarantee that the data passed to iterative reconstruction is zero-centered.
The real cropped 67208 projections still had a positive edge/DC offset. With
`projection_background=none`, FISTA-TV treated that offset as object signal and
reconstructed it as a smooth, low-frequency square volume. At coarse pyramid
levels this appeared as a very obvious blocky square in checkpoint previews.

`edge_median` estimates a per-view background level from detector edges and
subtracts it before the alignment/FISTA path. That leaves the reconstruction
objective dominated by sample contrast instead of detector/background offset.

The baseline FBP can still look plausible because it is rendered from raw
projections for comparison, while alignment/FISTA uses the working projection
stack. This difference is intentional, but it means a plausible baseline does not
prove the FISTA input is correctly centered.

## Prevention

- For real laminography FISTA/alignment runs, inspect both
  `raw_projection_stats` and `working_projection_stats` in `run_manifest.json`
  before trusting checkpoint previews.
- Treat `projection_background=none` as unsafe for real cropped data unless a
  separate preprocessing step proves the working projection medians are already
  near zero.
- Add a runner guard or warning when `projection_background=none` and per-view
  medians are not near zero.
- Keep in mind that coarse checkpoint previews are level-native. Level 8 previews
  are expected to be blocky because the checkpoint volume is `32x32x12`; that is
  separate from a square DC artifact.

Concrete regression coverage should create a positive-offset projection stack,
run the runner preprocessing path with `edge_median`, and assert that the
manifested working view-median percentiles are near zero. A complementary test
should verify that `projection_background=none` records unchanged working stats
and emits a warning for real-data FISTA/alignment runs.

## Related Issues

- The corrected run used the same centered slab mapping:
  `slab_center_global_z=148`, `preview_global_z=148`, `preview_local_z=48`.
- The preview renderer also needs careful interpretation for pyramid checkpoints:
  repeated global-z labels can occur when multiple full-resolution z positions map
  to the same coarse checkpoint slice.
