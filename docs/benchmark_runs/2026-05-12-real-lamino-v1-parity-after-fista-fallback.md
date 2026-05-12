# 2026-05-12 Real Lamino V1-Parity Full Gate After FISTA Fallback

Run: `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512`
Report: `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512/v2_cor_mvp_report/real_mvp_summary.json`
Parity table: `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512/v2_cor_mvp_report/real_mvp_v1_parity_table.csv`
Reference: `runs/real_lamino_native_setup_pose_256_k11_54014-edge-20260427-153525`

## Result

The full v2 `--v1-parity-real-lamino` run completed on CUDA after the measured-L
FISTA fallback fix. The real reconstruction gate passed: full staged final FISTA
loss was `6378.6333`, COR-only FISTA loss was `6740.0513`, an improvement of
`361.4180` (`5.36%`).

The pose-stage loss-scale regression is fixed. The parity audit reported no
`pose_loss_scale_failures`; phi, dx/dz, polish, and final FISTA rows are all on
v1 scale.

## Stage Loss Evidence

- Phi v2: level 4 `129.6431 -> 129.6281`, `129.1895 -> 129.1895`; level 2
  `481.8929 -> 481.8202`, `478.6499 -> 478.6471`; level 1
  `1857.3625 -> 1857.2839`, `1846.6211 -> 1846.5060`.
- dx/dz v2: level 4 `129.6352 -> 129.4725`, `128.9657 -> 128.9415`,
  `128.9002 -> 128.8972`; level 2 `479.6337 -> 479.3388`,
  `475.7672 -> 475.7672`; level 1 `1836.0453 -> 1835.8522`,
  `1824.2775 -> 1824.2775`.
- 5DOF polish v2: level 2 `477.4969 -> 476.4218`, `471.9253 -> 471.5915`,
  `470.3536 -> 470.2352`; level 1 `1806.1754 -> 1803.7443`,
  `1788.7288 -> 1788.4834`, `1783.3057 -> 1783.2437`.
- Final FISTA v2: `10744.5977 -> 6378.6333`; v1 reference final FISTA:
  `10745.2734 -> 6438.1611`.

## Remaining Audit Gap

This run is valid functional evidence, but the parity-report shape still has two
known reporting gaps before the audit can be treated as strict pass/fail:

- `06_cor_only_fista` emits spurious `missing_v2_row` entries because the table
  compares v1 setup-iteration rows against a v2 final FISTA-only stage.
- One setup row, `01_setup_geometry/03_axis_direction` level 8 iteration 7, is
  marked `missing_v2_row` because v2 early-stopped that level. The audit does
  not currently fail missing-row statuses even though the parity prompt asks for
  the same stage/level/iteration structure.

