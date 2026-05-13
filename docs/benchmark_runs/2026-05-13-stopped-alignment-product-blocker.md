# 2026-05-13 Stopped-Alignment Product Blocker

This report records the current truth-free stopped-alignment status for
detector-center / detector-u recovery. It is intentionally separate from the
oracle fixed-volume synthetic gates.

## Product Boundary

Production reconstruction and alignment must eventually work from:

```text
projections + approximate metadata -> preview reconstruction -> geometry update -> final reconstruction
```

The fixed-truth geometry gates prove that the Schur geometry solver has the
right signal when the volume is already in the correct object frame. They do
not prove the production stopped-reconstruction handoff is solved.

## Current Best Evidence

The latest completed truth-free scout/tangent stopped gate is:

```bash
uv run python tools/run_rich_phantom_v1_parity_gate.py \
  --out-dir .artifacts/product_truth_20260513/stopped_detu_scout_tangent/run \
  --views 128 \
  --profile reference \
  --mode stopped_multires \
  --preview-volume-support scout_soft \
  --preview-support-outside-weight 0.05 \
  --preview-low-frequency-anchor-weight 0.02 \
  --preview-det-u-gauge-mode-weight 0.05
```

It completed on CUDA without OOM as part of the product-truth runner.

## Results

| Level | Classification | Final det_u RMSE px | Volume NMSE | Schur accepted |
|---|---|---:|---:|---|
| `32^3` | `independent_projection_losses_consistent` | `0.369589` | `0.768313` | `true` |
| `64^3` | `reconstruction_absorbed_geometry` | `0.665706` | `0.073194` | `true` |
| `128^3` | `reconstruction_absorbed_geometry` | `1.704632` | `0.120314` | `true` |

Against the previous stopped baseline:

| Level | Baseline det_u RMSE px | Latest scout/tangent det_u RMSE px | Baseline volume NMSE | Latest scout/tangent volume NMSE |
|---|---:|---:|---:|---:|
| `32^3` | `1.607467` | `0.369589` | `0.740777` | `0.768313` |
| `64^3` | `1.675375` | `0.665706` | `0.512812` | `0.073194` |
| `128^3` | `2.954166` | `1.704632` | `0.502960` | `0.120314` |

Against the previous scout/tangent stopped artifact:

| Level | Previous scout/tangent det_u RMSE px | Latest scout/tangent det_u RMSE px | Previous volume NMSE | Latest volume NMSE |
|---|---:|---:|---:|---:|
| `32^3` | `0.297959` | `0.369589` | `0.769341` | `0.768313` |
| `64^3` | `0.904070` | `0.665706` | `0.203639` | `0.073194` |
| `128^3` | `1.924456` | `1.704632` | `0.218229` | `0.120314` |

## Interpretation

The scout-support and tangent-gauge direction is useful, not solved:

- It materially improves the 64^3 and 128^3 stopped trajectory compared with
  both the original rich-phantom stopped baseline and the previous
  scout/tangent artifact.
- It improves volume NMSE sharply at 64^3 and 128^3.
- It still does not pass detector-u recovery at production scale.
- The final `128^3` result is still classified
  `reconstruction_absorbed_geometry`, meaning the preview volume can still
  represent enough detector-center error to bias the geometry handoff.

The current production blocker is therefore still:

```text
truth-free preview reconstruction can absorb detector-center error into the
volume before geometry recovery has enough object-frame gauge information
```

## What This Means For Production

The public production claim should be:

```text
TomoJAX v2 has production-shaped real-data staged reconstruction and strong
fixed-volume/oracle geometry recovery, but truth-free stopped detector-center
recovery remains an active research blocker.
```

Do not label this path as a production pass yet. Do not hide this behind
volume-NMSE improvements or Schur acceptance, because both can improve while
detector-u remains wrong.

## Next Engineering Direction

The next work should strengthen the constrained alignment-volume handoff:

1. keep the two-volume framing: constrained alignment volume for geometry,
   freer final reconstruction after geometry is solved;
2. improve scout support / low-frequency anchor so it defines object frame
   without using truth support;
3. apply tangent-gauge projection during FISTA updates, not only as a
   diagnostic post-refresh operation;
4. keep running the 64^3 and 128^3 stopped det_u gates until detector-u RMSE
   reaches `< 0.5 px` first, then `< 0.2 px`;
5. avoid new geometry DOFs, nuisance terms, Pallas optimization, or extra
   benchmark cases until this handoff improves.
