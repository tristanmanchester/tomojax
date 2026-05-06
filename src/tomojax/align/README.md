# tomojax.align

## Purpose

`tomojax.align` owns alignment orchestration: alternating reconstruction and
geometry updates, continuation policy, observability reporting, gauge
canonicalisation, and solver traces.

The current package still contains transitional pre-v2 staged alignment code.
This README and `api.py` define the public boundary while later milestones
delete or migrate internal owners.

## Public API

- `AlignConfig`
- `AlignmentSmokeReport`
- `JointSchurDiagnostics`
- `JointSchurLMConfig`
- `JointSchurLMResult`
- `PoseOnlyLMConfig`
- `PoseOnlyLMResult`
- `SetupOnlyLMConfig`
- `SetupOnlyLMResult`
- `align`
- `align_multires`
- `run_alignment_smoke`
- `schur_step_from_jacobian`
- `solve_joint_schur_lm`
- `solve_pose_only_lm`
- `solve_setup_only_lm`

## Dependencies

Allowed dependencies:

- `tomojax.core`
- `tomojax.geometry`
- `tomojax.motion`
- `tomojax.forward`
- `tomojax.recon`
- `tomojax.verify`
- `tomojax.backends`
- `tomojax.io`

Forbidden dependencies:

- private implementation files from other deep modules
- generic utility modules
- new legacy compatibility layers

## Invariants

- Default geometry optimisation must be gradient-first LM/GN, not grid search.
- Geometry updates must emit artifact/provenance data.
- Public imports should come through this package facade.
- `run_alignment_smoke` is a tiny v2 wiring check. It is not the final
  optimiser.
- `solve_pose_only_lm` currently optimises `phi_residual_rad`, `dx_px`, and
  `dz_px`; `alpha_rad` and `beta_rad` are frozen until the reference projector
  supports out-of-plane pose effects.
- `solve_setup_only_lm` currently optimises `theta_offset_rad`, `det_u_px`, and
  active `det_v_px`; roll, axis rotation, and theta scale remain frozen until
  the reference projector models them.
- `solve_joint_schur_lm` is the first reference Schur setup+pose slice for the
  supported setup and pose DOFs. It is not yet the final trust-region engine.

## Tests

- Existing alignment tests cover transitional behavior.
- `tests/test_v2_module_skeleton.py` verifies the v2 facade exists and imports.
