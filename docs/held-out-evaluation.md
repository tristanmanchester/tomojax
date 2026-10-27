# Held-out evaluation

`tomojax.evaluation.Split` freezes training and held-out observations before
fitting. It does not select an iteration, regularisation weight or projector.
The caller supplies predictions from the **same discretisation** used in the
fit. Scores have no fitted intensity scale or background offset.

Choose the split for what the model shares between observations:

- `Split.interleaved(shape, stride=5, offset=2)` holds out individual views for
  selecting reconstruction iterations or regularisation. Repeat at several
  strides; sparse angular sampling can change which solution is preferred.
- `Split.angular(shape, angles, start=60, width=30, guard=5)` holds out a
  contiguous angular interval, repeated every 360 degrees. This tests global
  geometry or a smooth motion model that predicts withheld angles.
- `Split.detector(shape, start=32, width=32, period=128, guard=8, offsets=...)`
  holds out detector-column blocks within **every** view. Offsets are frozen
  integer pixel offsets, one per view or a common offset. Use this split for
  independent per-view poses: poses at completely unseen views have no fit
  evidence. Column blocks cover all detector rows.

`shape` is `(views, rows, columns)` and angles are degrees. `split.train` and
`split.test` are read-only boolean arrays broadcastable to that shape. Neither
mask includes guard observations. The JSON `split.specification` and SHA256
`split.fingerprint` describe the split; record them with source IDs, selected
frame IDs, geometry, corrections and the fitting command. The fingerprint is
not a data checksum. Freeze the split before inspecting its scores.

## Prevent leakage

Every fitted operation must exclude test and guard observations: reconstruction,
pose initialisation, motion fitting, data-derived weights and nuisance fitting.
Masking only the last residual is insufficient. Do not estimate an amplitude or
offset from the held-out observations. Independent flats and darks may calibrate
both partitions; sample-dependent ring correction or angular smoothing must not
mix them. Freeze a native-coordinate split before preprocessing. Guard bands
must cover local preprocessing support and allowed motion; global Fourier
filters cannot be certified by a finite guard band.

The masks do **not** enforce this policy inside `tj.align` or `tj.reconstruct`.
For interleaved/angular splits, fit only the selected training views. Detector
splits require a solver whose image and pose objectives both accept observation
weights. Replacing excluded observations by zero without also zeroing their
likelihood weights is not exclusion. Test this contract by changing all
excluded observations and checking that the complete fit remains unchanged.

## Score matched predictions

```python
import numpy as np
from tomojax.evaluation import Split

split = Split.interleaved(observations.shape, stride=5, offset=2)
# Fit using only split.train. Compute prediction with that fit's projector,
# geometry, intensity convention and any training-only nuisance parameters.
score = split.score(prediction, observations)
print(score.mse, score.relative_l2, score.count)
```

`weights=` is nonnegative **precision**, not square-root precision. The score
reports weighted squared error, weighted target energy, precision sum, and the
number of positive-weight test observations. Zero-weight observations contribute
nothing. Positive-weight nonfinite test values are errors; unscored values are
not used. Empty batches have zero additive totals and NaN mean/relative error.
Relative error is NaN when target energy is zero.

Large scans can be scored in view batches:

```python
from tomojax.evaluation import Score

total = Score(0, 0, 0, 0)
for start in range(0, observations.shape[0], 32):
    views = slice(start, start + 32)
    total += split.score(prediction[views], observations[views], views=views)
print(total.mse)
```

Sum the statistics before forming ratios; do not average per-batch MSEs when
batch sizes or precisions differ. Cross-validation scores used to choose a model
are selection scores, not an independent final test of that selected model.
