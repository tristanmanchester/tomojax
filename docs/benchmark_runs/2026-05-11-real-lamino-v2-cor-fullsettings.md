# 2026-05-11 Real Laminography v2 COR Full-Settings Diagnostic

Reference target report:
`runs/real_lamino_native_setup_pose_256_k11_54014-edge-20260427-153525/real_mvp_report/real_mvp_summary.json`

## Purpose

Diagnose whether the apparent v2 COR-only gap was a real preprocessing/geometry
problem or an artifact of the previous smoke run. The smoke run used a
48-slice slab and 3 FISTA iterations, so its COR-only loss was not comparable to
the v1 reference COR-only FISTA loss.

## Result

- v1 COR-only reference loss: 6804.66845703125.
- v2 COR-only full-settings loss after streaming FISTA fix: 6740.04248046875.
- Volume shape: `[256, 256, 96]`.
- Peak sampled memory during resumed streaming COR-only FISTA: 807 MiB.
- Peak sampled memory during the preceding full setup pass before the all-view
  FISTA OOM: about 1955 MiB.

## Diagnosis

The COR-only quality gap was not caused by detector flips/transposes,
laminography tilt convention, preprocessing, volume cropping, or output scaling.
With full slab/reconstruction settings, v2 COR-only is comparable to and
slightly better than the v1 COR-only reference.

The failure source was memory policy in the v2 real runner: `views_per_batch=0`
was passed to FISTA as `None`, which means all views in one batched
projector/adjoint chunk. The fair run OOMed in COR-only FISTA while trying to
allocate 4.84 GiB. The runner now normalizes the runtime default to
`views_per_batch=1`, preserving a streaming FISTA path unless the user
explicitly requests batching.

## Artifacts

- Failed all-view run directory:
  `runs/real_lamino_v2_cor_mvp_fullsettings_20260511/`
- Resumed streaming report:
  `runs/real_lamino_v2_cor_mvp_fullsettings_20260511/v2_cor_mvp_report/real_mvp_summary.json`
- Streaming GPU trace:
  `runs/real_lamino_v2_cor_mvp_fullsettings_20260511/gpu_memory_resume_cor_only.csv`

This closes the COR-only comparability diagnosis. The remaining MVP blocker is
not COR-only reconstruction quality; it is making the full staged v2 path
improve over the now-valid v2 COR-only comparator.
