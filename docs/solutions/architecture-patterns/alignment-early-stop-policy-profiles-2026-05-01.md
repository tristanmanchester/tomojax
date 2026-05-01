---
title: Alignment Early Stop Policy Profiles
date: 2026-05-01
category: architecture-patterns
module: TomoJAX alignment
problem_type: architecture_pattern
component: tooling
severity: medium
applies_when:
  - "Changing alignment early stopping for setup geometry or pose stages"
  - "Interpreting real-data laminography loss curves that move without visible preview changes"
  - "Adding diagnostics to alignment stage summaries, checkpoints, or runner manifests"
tags: [tomojax, alignment, early-stopping, validation-lm, diagnostics, real-data]
---

# Alignment Early Stop Policy Profiles

## Context

Alignment early stopping is part of the unified setup+pose alignment contract.
It should not be a private runner heuristic, and it should not interpret every
loss decrease the same way across setup geometry, pose cleanup, and final polish.

The practical failure mode from the April 2026 real laminography runs was:

```text
total level loss keeps drifting down
but accepted geometry updates are microscopic
and previews show no meaningful change
```

For setup geometry stages, especially detector roll and late refreshes, comparing
one outer iteration's `geometry_loss_after` to the next outer iteration's
`geometry_loss_after` is misleading. That comparison includes reconstruction
refresh drift. The meaningful setup evidence is the accepted validation-LM step
within the same objective context:

```text
geometry_loss_before -> geometry_loss_after
geometry_accepted
geometry_step_norm
optimizer_selected_scale
optimizer_condition_number
```

## Policy Shape

TomoJAX exposes two intended profiles:

- `compute_saving`: the default. Stop marginal stages when accepted gain and
  active-DOF movement are both too small for the configured patience.
- `robust`: conservative opt-in. Use it for final real-data runs or cases where
  a false stop is more expensive than extra GPU time.

`--no-early-stop` still disables alignment early stopping entirely. Legacy
threshold flags such as `--early-stop-rel` and `--early-stop-patience` remain
valid and override the selected profile's gain/patience thresholds.

## Setup Stages

Setup geometry uses stopped train-fold reconstruction and validation-LM. The
early-stop policy should treat accepted validation-LM step evidence as primary:

- accepted relative improvement from `geometry_loss_before` to
  `geometry_loss_after`,
- whitened/native setup step size,
- optimizer acceptance,
- selected LM scale,
- condition number.

Cross-outer `geometry_loss_after` drift is useful telemetry. It can reveal
reconstruction refresh effects or real loss movement, but it should not keep a
stage alive by itself.

## Pose Stages

Pose stages still have a useful fixed-volume pre/post step loss contract, so
same-outer `rel_impr` remains meaningful. The policy also checks active movement
evidence:

- rotation movement for `alpha`, `beta`, or `phi`,
- translation movement for `dx` or `dz`,
- optimizer acceptance where available.

This makes light polish stages less likely to run full iteration budgets when
both loss gain and active movement are tiny.

## Monitoring Guidance

When monitoring a real run, report the total loss, but do not stop there. Also
report:

- `accepted_rel_impr`,
- `geometry_step_norm` or pose movement stats,
- `optimizer_selected_scale`,
- `optimizer_condition_number`,
- `early_stop_profile`,
- `early_stop_decision`,
- `early_stop_reason`,
- early-stop streak counters.

This distinction matters because "loss still drifting" and "accepted geometry
step is too small to matter" are different diagnoses.

## Related

- `docs/solutions/architecture-patterns/reuse-align-multires-for-geometry-calibration-2026-04-25.md`
- `docs/brainstorms/geometry-calibration-solver-requirements.md`
