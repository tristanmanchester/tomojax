# Real Laminography v2 Production MVP Baseline - 2026-05-12

## Status

This is the morning-facing productionization baseline for the current TomoJAX v2
real-laminography MVP. It promotes the completed v1-parity k11 run as the
strongest current evidence, while keeping the distinction between a clean future
`real_lamino_mvp` profile and the existing strict `v1_parity_audit` path.

Current conclusion: TomoJAX v2 reproduces the known-working v1 staged
real-laminography workflow on the k11 reference scan closely enough to serve as
a real-data MVP/demo candidate. This does not prove general production readiness
for arbitrary scans, all synthetic stopped-reconstruction cases, nuisance
fitting, object drift, or all Pallas fast paths.

## Evidence Run

- Current commit while writing this report: `4f005f6`.
- v2 run:
  `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512`.
- v2 summary:
  `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512/v2_cor_mvp_report/real_mvp_summary.json`.
- v2 parity audit:
  `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512/v2_cor_mvp_report/real_mvp_v1_parity_audit.json`.
- v1 reference run:
  `runs/real_lamino_native_setup_pose_256_k11_54014-edge-20260427-153525`.
- Input dataset:
  `/home/tristan/projects/tomojax/runs/real-lamo-256/k11-54014_corrected_log_256cube.nxs`.

The production-named profile now exists. The clean MVP command is:

```bash
NVLIB=$(find "$PWD/.venv/lib/python3.12/site-packages/nvidia" -type d -name lib | paste -sd: -)
env LD_LIBRARY_PATH="$NVLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  UV_CACHE_DIR=.uv-cache JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run python scripts/real_laminography/run_real_lamino_v2_cor_mvp.py \
    --input /home/tristan/projects/tomojax/runs/real-lamo-256/k11-54014_corrected_log_256cube.nxs \
    --out runs/real_lamino_v2_production_mvp_k11_54014_20260512 \
    --reference-report runs/real_lamino_native_setup_pose_256_k11_54014-edge-20260427-153525/real_mvp_report/real_mvp_summary.json \
    --profile real_lamino_mvp \
    --overwrite
```

The strict audit command is the same except for `--profile v1_parity_audit`.
The old `--v1-parity-real-lamino` flag remains as an alias for the audit
profile.

## Metrics

| Quantity | v1 reference | v2 parity run |
| --- | ---: | ---: |
| COR-only final loss | `6804.668457` | `6740.051270` |
| Final selected loss | `6438.161133` | `6378.633301` |
| Improvement over COR-only | `366.507324` | `361.417969` |
| Relative improvement over COR-only | `5.386%` | `5.362%` |
| Final volume shape | `256 x 256 x 96` | `256 x 256 x 96` |
| Backend device | `cuda:0` | `cuda:0` |

The v2 final loss is lower than the v1 reference on the reported loss scale
(`6378.6333` versus `6438.1611`) and improves over the v2 COR-only comparator by
`361.418` loss units.

## Stage Path

The completed v2 run executed the full staged workflow:

1. `00_baseline`
2. `01_setup_geometry/01_cor`
3. `06_cor_only_fista`
4. `01_setup_geometry/02_detector_roll`
5. `01_setup_geometry/03_axis_direction`
6. `02_pose_phi`
7. `03_pose_dx_dz`
8. `04_pose_polish`
9. `05_final`

The public-facing `real_lamino_mvp` profile should present this as baseline,
setup geometry, pose, polish, final FISTA, and COR-only comparator without
requiring users to know the debugging history.

## Stage Loss Summary

| Stage | Rows | First loss_before | Last loss_after |
| --- | ---: | ---: | ---: |
| `01_setup_geometry/01_cor` | 23 | `38.290432` | `451.304749` |
| `01_setup_geometry/02_detector_roll` | 20 | `35.004150` | `450.303162` |
| `01_setup_geometry/03_axis_direction` | 21 | `34.985558` | `445.347534` |
| `02_pose_phi` | 6 | `129.643066` | `1846.505981` |
| `03_pose_dx_dz` | 7 | `129.635193` | `1824.277466` |
| `04_pose_polish` | 6 | `477.496918` | `1783.243652` |
| `05_final` | 1 | `10744.597656` | `6378.633301` |

Losses are level-dependent, so coarse and fine rows are not directly comparable
inside this table. The key gate is the final FISTA comparison against COR-only.

## Setup And Pose

Winning v2 setup estimates:

- `det_u_px = -3.725865`
- `detector_roll_deg = 0.137386`
- `axis_rot_x_deg = 0.460339`
- `axis_rot_y_deg = 0.007741`

Winning v2 pose summary from `run_manifest.json`:

- `dx`: std `2.404333 px`, min `-7.762053`, max `7.736288`
- `dz`: std `1.578230 px`, min `-4.979375`, max `5.915191`
- `phi`: mean `-0.191225 deg`, std `0.032677 deg`
- `alpha`: std `0.280306 deg`
- `beta`: std `0.318627 deg`

Report-quality note: this was initially only present in `run_manifest.json`.
The report builder now also copies `final_pose_summary` into
`real_mvp_summary.json`, and the winning report was regenerated in place.

## Parity Audit

The strict v1 parity contract passed: schedules, bounds, reconstruction
settings, pose model, preprocessing, and final-candidate policy matched the
expected v1-derived contract.

Parity table status counts:

- `matched`: 85 rows
- `missing_v2_row`: 1 row

The remaining row-shape mismatch is
`01_setup_geometry/03_axis_direction`, level `8`, iteration `7`. It was
diagnosed as early-stop row-count sensitivity: v2 stopped one row earlier after
near-identical tiny improvements. Pose/final reconstruction losses are on the
v1 scale, and `pose_loss_scale_failures` is empty.

## Artifacts To Inspect

Primary report artifacts:

- Summary markdown:
  `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512/v2_cor_mvp_report/real_mvp_summary.md`
- Residual trace CSV:
  `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512/v2_cor_mvp_report/real_mvp_residual_trace.csv`
- Geometry trace JSON:
  `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512/v2_cor_mvp_report/real_mvp_geometry_trace.json`
- V1 parity table:
  `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512/v2_cor_mvp_report/real_mvp_v1_parity_table.csv`

Publication PNGs:

- Before central slice:
  `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512/v2_cor_mvp_report/publication/before_xy_aligned_xy_global_z209.png`
- COR-only central slice:
  `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512/v2_cor_mvp_report/publication/cor_only_xy_aligned_xy_global_z209.png`
- Full final central slice:
  `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512/v2_cor_mvp_report/publication/full_xy_aligned_xy_global_z209.png`
- Full final orthos:
  `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512/v2_cor_mvp_report/publication/full_orthos.png`
- Full-vs-baseline delta:
  `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512/v2_cor_mvp_report/publication/full_delta_xy_delta_xy_global_z209.png`

Contact sheets for stage-by-stage central slice, orthos, z-stack, and direct
v1-vs-v2 final comparison still need to be generated or linked into the report.

## Runtime And Memory

- Started: `2026-05-12T11:02:45`
- Completed: `2026-05-12T17:13:58`
- Wall time: `6:11:13`
- Sampled peak GPU memory from `gpu_memory.csv`: `5967 MiB`
- GPU: `cuda:0`
- JAX preallocation was disabled.

This run is not yet a fast production workflow. It is a reproducible real-data
MVP result with honest provenance.

## Original Plan Status

| Phase | Current status | Notes |
| --- | --- | --- |
| Phase 0 baseline and benchmark contract | `partial` | Real k11 report and parity artifacts exist; broad benchmark contract is still incomplete. |
| Phase 1 geometry graph, parameters, gauges | `implemented_for_real_mvp` | Setup and pose state, bounds, gauge-like mean translation handling, and JSON/CSV artifacts exist for this path. Full general geometry graph is still partial. |
| Phase 2 JAX reference forward model and residual loss | `implemented_for_real_mvp` | Tomography/laminography projectors and level losses are working in the real run. Robust pseudo-Huber default remains broader-plan work. |
| Phase 3 FISTA / Huber-TV preview reconstruction | `implemented_for_real_mvp` | Huber-TV/FISTA and measured-L fallback are required for the winning run. |
| Phase 4 pose-only 5x5 LM | `implemented_for_real_mvp` | Phi, dx/dz, and 5DOF polish run and improve the final result in the v1-parity path. |
| Phase 5 setup-only LM | `implemented_for_real_mvp` | COR/det_u, detector roll, and axis direction stages run on the reference data. |
| Phase 6 joint setup+pose Schur LM | `partial` | Synthetic Schur diagnostics exist, but the real MVP path is still staged rather than a single default joint Schur solve. |
| Phase 7 alternating solver and continuation | `partial` | Alternating infrastructure and artifacts exist; real MVP uses a v1-like staged continuation. |
| Phase 8 nuisance and weak DOF handling | `diagnostic_only` | Nuisance/weak-mode work is not a real-MVP success claim. |
| Phase 9 Pallas fast paths | `partial` | Pallas fast paths exist and rigid detector-grid folding was added, but parity mode intentionally preserves measured-FISTA fallback where required. |
| Phase 10 experimental modules | `out_of_scope_for_mvp` | Not needed for the overnight real-data MVP. |

## What Worked

- The k11 v2 real-laminography path improves final reconstruction quality over
  COR-only on real data.
- V2 matches or beats the v1 reference final reported loss for this run.
- The stage path includes COR/det_u, detector roll, axis direction, phi, dx/dz,
  5DOF polish, and final FISTA.
- The phi level-2 loss explosion was rooted to the calibrated-grid Huber-FISTA
  core fallback and fixed for parity by restoring measured-L public FISTA.
- Fail-closed checks prevent non-finite pose-stage volumes from being promoted.

## What Remains Unproven Or Messy

- A clean `real_lamino_mvp` CLI/profile now exists and resolves to the winning
  v1-derived settings, but it has not been rerun under a production-named output
  directory.
- `v1_parity_audit` remains available and should stay separate from the
  user-facing demo profile.
- `run_real_lamino_v2_cor_mvp.py` still mixes profile selection, execution,
  validation, reporting, and parity table generation.
- `real_mvp_summary.json` omits `final_pose_summary` even though the run
  manifest records it.
- Stage contact sheets and v1-vs-v2 final comparison sheets need to be generated
  or made easy to find.
- Synthetic tomography MVP gates for `synth128_setup_global_tomo` and
  `synth128_pose_random_extreme` still need bounded artifact-producing reports.
- This report does not claim that nuisance fitting, object drift, weak-view
  handling, all five original synthetic cases, or arbitrary scan production use
  are solved.

## Next Actions

1. Split the real-runner profile/report/parity responsibilities enough that the
   working MVP is not buried as an accidental flag combination.
2. Add focused tests for phi level-2 loss scale, final
   candidate selection, fail-closed output, and portable artifact paths.
3. Produce the bounded synthetic tomography MVP report.
