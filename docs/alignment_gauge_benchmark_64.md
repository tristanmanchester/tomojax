# Alignment Gauge Fixing 64^3 Benchmark

This note records CPU validation runs for explicit alignment gauge fixing. The
goal is to check behavior and interpretation of saved alignment traces, not to
make a general performance claim.

The comparison uses the new `mean_translation` gauge mode against the explicit
opt-out mode, `none`.

## Environment

- Date: 2026-04-19
- Backend: JAX CPU
- Device: `cpu:0`
- JAX: `0.10.0`
- Volume shape: `64 x 64 x 64`
- Detector shape: `64 x 64`
- Views: `12`
- Active alignment DOFs: `dx,dz`
- Alignment loss: `l2`
- Raw local artifacts: `runs/gauge_benchmark_64/`

## Dataset

The benchmark generated a `64^3` Shepp phantom and a misaligned projection stack
with deterministic translation-only motion. The motion contains both residual
per-view variation and a deliberately nonzero global detector shift:

| Quantity | Value |
| --- | ---: |
| True `dx` mean | `1.2500 px` |
| True `dz` mean | `-0.8000 px` |
| True `dx` residual span | `0.6929 px` |
| True `dz` residual span | `0.5000 px` |

The nonzero mean is intentional. It separates trace-shape recovery from global
translation recovery and makes the effect of gauge fixing visible.

## Fixed-Volume Optimizer Comparison

These runs used the ground-truth volume as `init_x`, one outer alignment update,
translation-only active DOFs, and no reconstruction ambiguity from alternating
updates. This setup makes the global shift observable in the fixed-volume loss,
so absolute loss and absolute translation RMSE are not the fairest comparison of
trace shape under gauge fixing. The relative and gauge-fixed RMSE columns are
the relevant trace-shape checks.

| Optimizer | Gauge | Wall time | Final loss | `dx` mean | `dz` mean | Abs trans RMSE | Relative trans RMSE | Gauge-fixed trans RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GD | `mean_translation` | `2.94s` | `110578.49` | `-1.86e-09` | `6.21e-10` | `1.0654` | `0.1322` | `0.2601` |
| GD | `none` | `1.55s` | `93422.13` | `0.0769` | `-0.0979` | `0.9840` | `0.1322` | `0.2601` |
| GN | `mean_translation` | `2.59s` | `107090.75` | `3.73e-09` | `-7.45e-09` | `1.0646` | `0.3972` | `0.2532` |
| GN | `none` | `1.89s` | `10831.30` | `0.7861` | `-0.7026` | `0.3800` | `0.3972` | `0.2532` |
| L-BFGS | `mean_translation` | `10.76s` | `59192.19` | `1.99e-08` | `4.97e-09` | `1.2714` | `0.5904` | `1.0151` |
| L-BFGS | `none` | `8.94s` | `2033.69` | `0.8651` | `-0.7329` | `0.4069` | `0.2877` | `0.4226` |

For GD and GN, the relative-motion and gauge-fixed trace-shape metrics match
between `mean_translation` and `none` up to numerical precision where the
accepted update shape is the same. The saved means differ by design.

The lower fixed-volume losses for `none` are expected in this synthetic setup:
the reference volume is fixed in the ground-truth frame, so the global detector
translation is physically observable by the fixed-volume objective. In joint
reconstruction and alignment, that same global shift can be gauge ambiguous and
less useful as a per-view trace component.

## Smooth Pose Model

These runs repeated the fixed-volume GD comparison with
`pose_model="polynomial"`, so accepted expanded per-view parameters are
gauge-fixed and then refit back into motion coefficients.

| Pose model | Gauge | Wall time | Final loss | `dx` mean | `dz` mean | Abs trans RMSE | Relative trans RMSE | Gauge-fixed trans RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Polynomial GD | `mean_translation` | `2.62s` | `107582.66` | `-6.21e-10` | `7.45e-09` | `1.0609` | `0.1446` | `0.2202` |
| Polynomial GD | `none` | `1.52s` | `78355.84` | `0.1816` | `-0.1850` | `0.8930` | `0.1476` | `0.2742` |

This exercises the coefficient-refit path. The gauge-fixed polynomial output
kept zero-mean saved translations after refitting.

## Alternating Reconstruction Smoke

These runs checked the end-to-end alternating path on a `64^3` volume with a
small amount of reconstruction work (`outer_iters=1`, `recon_iters=2`,
`recon_L=5000`). They are runtime smoke tests, not convergence benchmarks.

| Optimizer | Gauge | Wall time | Final loss | `dx` mean | `dz` mean | Abs trans RMSE | Relative trans RMSE | Gauge-fixed trans RMSE | Result |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GN | `mean_translation` | `4.13s` | `6842941.00` | `0.0000` | `0.0000` | `1.0685` | `0.1467` | `0.2842` | No accepted pose movement |
| GN | `none` | `2.37s` | `6842941.00` | `0.0000` | `0.0000` | `1.0685` | `0.1467` | `0.2842` | No accepted pose movement |

The alternating smoke validates that the configured gauge mode flows through the
normal alignment workflow and metadata path on a `64^3` case. Because this tiny
run did not accept a nonzero pose update, it does not demonstrate the opt-out
behavior as clearly as the fixed-volume comparisons above.

## Gauge Metadata Checks

Every `mean_translation` run recorded post-gauge means in the alignment stats:

| Run | Post-gauge `dx` mean | Post-gauge `dz` mean |
| --- | ---: | ---: |
| Fixed GD | `0.00e+00` | `0.00e+00` |
| Fixed GN | `0.00e+00` | `-1.74e-08` |
| Fixed L-BFGS | `0.00e+00` | `-4.97e-09` |
| Polynomial GD | `-6.21e-10` | `0.00e+00` |
| Alternating GN | `0.00e+00` | `0.00e+00` |

Runs with `gauge_fix="none"` did not record post-gauge means, and their saved
fixed-volume traces retained nonzero translation means where the optimizer moved
the global shift into per-view parameters.

## Test Results

The focused gauge/alignment/metadata test set passed after the implementation:

```text
89 passed in 211.81s
```

The full repository test suite was rerun after the `64^3` comparisons:

```text
uv run pytest -q tests
413 passed, 1 skipped in 419.12s (0:06:59)
```

## Interpretation

`mean_translation` does what it is intended to do: saved active `dx,dz` traces
are centered at numerical zero after initialization and accepted alignment
updates. This makes the trace easier to read as residual per-view motion rather
than a mixture of residual motion and an arbitrary global detector shift.

The explicit `none` mode preserves historical unconstrained behavior. In the
fixed-volume comparisons, `none` retained nonzero translation means after
accepted updates, which is the expected opt-out behavior.

Absolute pose RMSE and fixed-volume objective values can be worse with gauge
fixing when the synthetic target includes a real global translation and the
reference volume is held fixed. That does not contradict the purpose of the
runtime gauge constraint: it removes a global ambiguity from saved per-view
alignment traces in joint reconstruction/alignment workflows. Gauge-aware
benchmark metrics remain separate scoring tools against known truth.

Single-run CPU timings are included only to make the comparisons reproducible.
They should not be treated as GPU or production performance guidance.
