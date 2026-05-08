# 2026-05-08 Phase 8 Pose-Only Trust Diagnostic

Ran direct true-volume Schur probes on the existing
`synth128_pose_random_extreme` 128^3 sidecar after the fixed-truth sigma policy
change. These probes bypass `align-auto` and call `solve_joint_schur_lm`
directly to isolate pose-only solver behavior.

## All-5 Pose Trust Probe

Configuration:

```text
active_setup_parameters = ()
active_pose_dofs = alpha_rad,beta_rad,phi_residual_rad,dx_px,dz_px
max_iterations = 4
sigma = 1.0
residual_filters = raw
```

| Pose trust radius | Final loss | alpha/beta RMSE rad | phi RMSE rad | dx RMSE px | dz RMSE px |
|---:|---:|---:|---:|---:|---:|
| none | 0.931687 | 0.127884 | 0.138258 | 4.791669 | 4.891444 |
| 50 | 0.931687 | 0.127884 | 0.138258 | 4.791669 | 4.891444 |
| 20 | 1.130748 | 0.114371 | 0.134623 | 5.621214 | 5.845314 |
| 10 | 1.397434 | 0.095919 | 0.126344 | 6.700839 | 7.063507 |
| 5 | 1.736992 | 0.073701 | 0.115317 | 8.131236 | 8.451736 |
| 2 | 2.198980 | 0.040044 | 0.103513 | 10.068182 | 10.300804 |

Interpretation: relaxing the global pose trust cap recovers translations better
but makes angular pose recovery worse. A blanket no-trust all-5 policy should
not be promoted.

## Phi/Translation-Only Probe

Configuration:

```text
active_setup_parameters = ()
active_pose_dofs = phi_residual_rad,dx_px,dz_px
pose_trust_radius = none
sigma = 1.0
residual_filters = raw
```

| Iterations | Final loss | alpha/beta RMSE rad | phi RMSE rad | dx RMSE px | dz RMSE px |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.988331 | 0.028422 | 0.138090 | 4.942043 | 5.523209 |
| 8 | 0.174174 | 0.028422 | 0.139292 | 1.081214 | 1.379676 |
| 12 | 0.058544 | 0.028422 | 0.104954 | 0.435001 | 0.135811 |

Interpretation: with alpha/beta frozen and trust disabled, translations are
recoverable from the true volume with enough iterations. Phi remains poor and
alpha/beta remain at their initial zero-pose error. The next implementation
slice should be an angular pose observability/acceptance policy or a staged
pose solver with separate angular validation, not a simple increase in global
iterations or trust radius.
