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

## Stopping-rule limits

Minimising held-out projection MSE does not guarantee the smallest image error.
The following CGLS checks hold out every fifth view, offset two, and compare
iterations 1, 2, 4, 8, 16, 32 and 64. The oracle is the best **tested** iteration
for the stated image metric, not an optimum over all iterations. Geometry and
intensity scale are fixed; no image registration or fitted rescaling is used.

| Scan and image metric | MSE-selected iteration / error | Oracle iteration / error | Error ratio |
| --- | ---: | ---: | ---: |
| 2DeteCT 00001 mode 1, 128²; mode-2 reference agreement | 16 / 0.22469 | 8 / 0.15918 | 1.4115 |
| Walnut 1, all three orbits, detector binning 8, 64³; authors' AGD reference agreement | 64 / 0.28250 | 8 / 0.20272 | 1.3935 |
| I13 159815, row 1080, every eighth sample view, 256²; injected Gaussian incremental-response error | 16 / 0.56073 | 32 / 0.25989 | 2.1576 |

Errors are relative image L2 norms. The I13 metric compares the difference of
two reconstructions with an analytic attenuation addition, not with the unknown
whole specimen. Its binomial thinning uses rounded photon counts inferred from
ADU values and retains original read noise; it is an approximate known-truth
control. CGLS depends on its right-hand side, so this difference is not a linear
reconstruction of the added signal. Walnut uses all 1200 views from each orbit
before splitting and exact cell-overlap averaging of the published reference.

Measured 2026-10-27 with numerical source `20fe1a9c` and evaluation `09437d17`.
Commands are `OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
uv run --no-sync python shared/agent02/` followed by
`stopping_stack_kill.py`, `stopping_walnut_protocol.py` or
`stopping_i13_increment.py`, with `LAB_ARTIFACTS` set to an output directory.
The radiometric-convention comparison uses `stopping_author_protocol.py`.
Those scripts pin the stated numerical source. Selected-iteration cold/warm
times are respectively 0.220/0.220, 12.183/12.218 and 0.153/0.145 seconds;
peak process GPU memory is 208, 1172 and 220 MiB, and peak host RSS is 959,
2266 and 1005 MiB. Times include reconstruction and matched prediction/scoring;
I13 includes both reconstructions needed for its difference metric. Cold means
the first run of that candidate; these are not fresh-process compilation times.

All three exceed the tested 1.10 oracle-gap limit. The 2DeteCT check also
depends on radiometric convention: using the authors' plain flat/dark correction
and crop centre gives a gap of 1.4960 on the same candidate iterations, rather
than fixing the failure. Shared calibration noise, truncation and model mismatch
can invalidate an independent-noise interpretation of view cross-validation.
Use image or known-truth checks alongside prediction scores; TomoJAX does not
adopt these failed MSE selectors as automatic stopping rules.


### Discrete image-space validation

On 2DeteCT 00001 at 64², the criterion
`||x||² - 2 <x, A_test⁺ y_test>` also missed the image-reference oracle.
The full-rank held-out projector's numerical left inverse selected 16 CGLS
iterations (relative image error 0.215930), versus 8 (0.127258), a 1.69679 gap.
Its normal-matrix eigenvalue ratio was 3.80e-5 and numerical left-inverse
residual 3.14e-15. Replacing low-dose validation observations with real mode-2
high-dose observations selected the same iteration and gap. Accurate inversion
and higher-dose validation did not remove the failure. These tests do not
separate physical-model error, shared calibration and reference-reconstruction bias.
Reference agreement is not physical accuracy.

Measured 2026-10-27 on `20fe1a9c`, using the same interleaved split and tested
iterations, authors' plain flat/dark correction and crop centre, and no fitted
scale or registration. Commands use the environment above with
`stopping_exact_image_risk.py` and `stopping_exact_image_risk_dose.py`.
Dense setup and all candidates took 50.07/51.20 s; the selected candidate's
first/repeat reconstruction and scoring took 0.6498/0.6494 and
0.6496/0.6496 s. Peak GPU memory was 532/528 MiB and host RSS 6877/6700 MiB.
Dense basis construction is a diagnostic, not a scalable stopping implementation;
neither criterion is an automatic TomoJAX selector.

### Ridge regularization

The same failure occurs without iterative stopping. On 2DeteCT 00001 at 32²,
unconstrained ridge solutions minimize `||A_train x - y_train||² + λ ||x||²`.
Testing λ divided by the largest training normal-matrix eigenvalue at
`0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, .01, .1, 1` gives an oracle at `.01`
(relative reference error 0.0807442). Prediction MSE and discrete image-space
risk, with either low-dose or mode-2 validation, select `.0001` (0.2644305):
an oracle gap of 3.27492. The numerical inverse residual is 1.03e-15.
This is reference agreement, not physical accuracy; exact algebra does not
remove representation mismatch or shared-calibration effects.

Measured 2026-10-28 on `20fe1a9c`, with the same split, radiometric convention
and fixed geometry, using `ridge_lambda_kill.py` in the environment above.
Dense setup and all candidates took 16.73 s, with peak GPU memory 314 MiB
and host RSS 4360 MiB. TomoJAX does not adopt this diagnostic as a λ selector.

### SIRT stopping

On 2DeteCT 00001 at 128², projection MSE selected 256 SIRT updates rather
than the reference-agreement oracle at 128. Unconstrained SIRT had errors
0.1762805 versus 0.1397493 (gap 1.26140); nonnegative SIRT had errors
0.1352244 versus 0.1088912 (gap 1.24183). Both exceed 1.10. The candidate
updates were powers of two from 1 through 1024, without acceleration, using
900 views, the same interleaved split and authors' radiometric convention.

Measured 2026-10-28 on `20fe1a9c` with `stopping_sirt_family.py` and
`stopping_sirt_supplement.py`. Selected-candidate first/repeat times were
11.173/11.169 and 11.130/11.129 s. Completed-candidate times sum to 357.80 s;
this is not end-to-end wall time. Peak sampled GPU memory was 164 MiB.
The initial process timed out after 21 of 22 configurations, losing its host
peak measurement; the final configuration completed separately with host RSS
829 MiB. That host peak does not characterize the complete candidate sweep.

### Fine-grid image validation

A different diagnostic uses `||x||² - 2 <x, z_test>`, where `z_test` is a
1024² FDK reconstruction of held-out views averaged into the candidate's cells.
The 956-column fan detector is symmetrically zero-extended to 1912 columns
before filtering to retain filtered tails; this assumes zero attenuation beyond
the measured panel and does not recover genuinely truncated observations.
On 2DeteCT 00001 and 00067 at 32², this criterion selected the ridge oracle
λ ratio `.01` in both cases, with reference errors 0.0807442 and 0.2970754.
Projection-MSE oracle gaps were 3.27492 and 1.83881 on the same paths.

At 128², using all 3600 views, 2880 training and 720 held out at stride five,
offset two, the criterion gave the following CGLS reference-agreement results.
Candidates were powers of two from 1 through 64; every tested oracle was at eight
updates, not an endpoint. Slice 05521 is out of distribution in the dataset split.

| Slice | Image-risk updates / error | Oracle error | Image-risk gap | Projection-MSE gap |
| --- | ---: | ---: | ---: | ---: |
| 00001 | 8 / 0.138736 | 0.138736 | 1.00000 | 1.47721 |
| 00067 | 8 / 0.350894 | 0.350894 | 1.00000 | 1.21979 |
| 00100 | 16 / 0.167781 | 0.164503 | 1.01993 | 1.36731 |
| 05521 | 8 / 0.138079 | 0.138079 | 1.00000 | 1.27097 |

These are reference-agreement checks, not physical accuracy or a general stopping
rule. Shared calibration and physical-model errors can bias this image proxy.

Measured 2026-10-28 on `20fe1a9c` with `ridge_fine_proxy.py` and
`stopping_fine_full.py` (00001/00067) and `stopping_fine_fresh.py`
(00100/05521), using fixed geometry and no fitted rescaling.
Ridge setup and all candidates took 15.26/15.41 s, with GPU peaks 314/314 MiB
and host RSS 4340/4334 MiB. CGLS selected-candidate first/repeat times were
1.566/1.565 and 1.572/1.574 s, with complete-sweep GPU peaks 224/202 MiB and
host RSS 989/880 MiB. CPU validation-proxy first/repeat times were separately
1.275/0.451 and 1.078/0.490 s. For 00100/05521, selected-candidate times were
2.960/2.960 and 1.568/1.569 s, GPU peaks 202/202 MiB, host RSS 889/880 MiB;
CPU proxy times were 0.858/0.356 and 0.870/0.350 s. A 128² analytic sharp-disk
control had error
0.1742 with 180 held-out angles versus 0.01693 with 720; a genuinely truncated
Gaussian control had error 0.1591 despite zero extension. These failures prevent
using this diagnostic as an automatic TomoJAX selector.
