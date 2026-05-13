# 2026-05-13 Real Laminography Product Workflow

This report records the current production-facing real-laminography evidence
for TomoJAX v2. It uses the retained k11 256-detector laminography case as a
real-data workflow check, but reports it through the newer staged report
contract rather than development-era naming.

## Command

The report was regenerated locally from the completed full-resolution staged
run:

```bash
uv run python scripts/real_laminography/summarize_real_lamino_report.py \
  --run-dir runs/real_lamino_v2_full_mvp_full256_multires_oneouter_40iter_spline_all_20260512 \
  --require-success
```

The product-facing command for future runs is:

```bash
uv run python scripts/real_laminography/run_real_lamino_staged.py \
  --input /path/to/scan.nxs \
  --out runs/real_lamino_staged_run \
  --profile staged-lamino \
  --overwrite
```

## Result

| Metric | Value |
|---|---:|
| Final staged FISTA loss | `6517.55712890625` |
| COR-only comparator FISTA loss | `7411.73046875` |
| Absolute improvement | `894.17333984375` |
| Relative improvement | `0.12064299202646986` |
| Same output volume shape | `true` |
| Report success | `true` |

The success criterion is real reconstruction quality:

```text
final staged reconstruction loss < COR-only reconstruction loss
at the same output volume shape
```

Truth metrics are not applicable to this real dataset and are not used for the
pass decision.

## Staged Path

The completed staged path is:

1. `00_baseline`
2. `01_setup_geometry/01_cor`
3. `01_setup_geometry/02_detector_roll`
4. `01_setup_geometry/03_axis_direction`
5. `02_pose_phi`
6. `03_pose_dx_dz`
7. `04_pose_polish`
8. `05_final`
9. `06_cor_only_fista`

The full staged path and COR-only comparator both ran at 40 effective FISTA
iterations for the final comparison.

## Artifacts

Generated report artifacts:

- `runs/real_lamino_v2_full_mvp_full256_multires_oneouter_40iter_spline_all_20260512/real_lamino_report/real_lamino_summary.json`
- `runs/real_lamino_v2_full_mvp_full256_multires_oneouter_40iter_spline_all_20260512/real_lamino_report/real_lamino_summary.md`
- `runs/real_lamino_v2_full_mvp_full256_multires_oneouter_40iter_spline_all_20260512/real_lamino_report/real_lamino_residual_trace.csv`
- `runs/real_lamino_v2_full_mvp_full256_multires_oneouter_40iter_spline_all_20260512/real_lamino_report/real_lamino_geometry_trace.json`
- `runs/real_lamino_v2_full_mvp_full256_multires_oneouter_40iter_spline_all_20260512/real_lamino_report/publication/`

The publication directory contains before, COR-only, final, and final-delta
orthos/XY images copied from the staged run.

## Method Boundaries

The generated report records:

- no COR grid search added,
- no sinogram or correlation method added,
- no sharpness or autofocus method added,
- no benchmark-only knobs promoted,
- COR-only retained only as the comparator artifact.

This is a real-data staged reconstruction workflow check, not evidence that
the generic truth-free stopped-alignment blocker has been solved.

