# Align a scan

`tj.align` estimates geometry errors and sample movement while it reconstructs. You choose
which errors it estimates with `fit=`. Start with the cheapest model that matches the error you
suspect, look at the result, and add parameters only while artefacts remain: an unnecessary
parameter can cost accuracy, while a missing one leaves an error the fit cannot represent.

Before you align, correct the projections to line integrals and check the declared geometry
against the instrument, as the [real scan guide](real-laminography.md) describes. Reconstruct once
without alignment so that you have an image to compare with.

## Choose what to fit

| What you see | Likely error | Fit |
| --- | --- | --- |
| Point-like features drawn as rings or arcs, the same in every slice | The rotation centre | `fit="centre"` |
| Doubled or smeared edges that vary from feature to feature | Drift or sudden slips of the sample between views | `fit="translations"` (the default) |
| Blur that grows away from the detector's centre row, remaining after translations | The detector roll, or rotation-axis roll about the beam | `fit=("translations", "roll")` |
| Doubled or streaked edges that grow with distance from the rotation axis, remaining after translations | The laminography or axis tilt | `fit=("translations", "tilt", "roll")` |
| Blur that remains after translations and the setup angles | The sample rotating or wobbling between views | `fit=("translations", "rotations")` |

```python
import tomojax as tj

scan = tj.load("corrected.nxs")
result = tj.align(scan)                                  # translations
result = tj.align(scan, fit="centre")                    # the rotation centre only
result = tj.align(scan, fit=("translations", "tilt", "roll"))
tj.save("aligned.nxs", result)
recon = tj.reconstruct(result.scan, "cgls")              # carries the corrections
```

The command line uses the same names:

```bash
tomojax align corrected.nxs -o aligned.nxs --fit translations tilt roll
```

`fit` is one name or a tuple of distinct names; the order does not affect the model. Parameters
you leave out stay at their input values. All combinations use the same joint Gauss–Newton solve
for parallel, laminography and cone data. `grid=` chooses the reconstruction region and sampling;
it does not change the meaning or units of the corrections.

| Name | Fitted parameters | Units and frame |
| --- | --- | --- |
| `centre` | One detector-u centre, plus detector-v for cone data | Physical detector-centre lengths, in the same units as `Detector.center` |
| `tilt` | One axis or laminography tilt correction shared by every view | Degrees; positive tilt leans the rotation axis towards the beam |
| `roll` | One detector roll correction shared by every view | Degrees about the beam |
| `translations` | Two translations per view; three for cone data | Physical lab-frame lengths: across the beam and vertically, plus along the beam for cone |
| `rotations` | Three rotations per view | Sample-frame rotations in radians |

The figure slots below are being remade with a spark-plug phantom. Earlier watch figures and
timings used the retired alignment routes and a profiled tilt search; they are not qualification
results for this solver. Numerical checks and commands for the new implementation are described
in [the benchmark guide](../bench/README.md#alignment-fit-settings).

## The rotation centre

A wrong rotation centre turns every small, bright feature into a ring. The rings are the same size
everywhere in the slice, because every view is displaced by the same amount.

<!-- Figure: nominal and fitted centre, plus differences from true-geometry reconstructions. -->

`fit="centre"` estimates one number for parallel and laminography data. Cone data need two
detector-frame offsets, because changing a cone detector centre is not the same as translating
the object. Fit the centre first on a new instrument or after a change to the detector or stage.

`tj.estimate_center(scan)` is also available as a projection-only preview for suitable axial
parallel scans with opposing views. It has its own coverage and motion assumptions; it does not
run the image-and-geometry solve or estimate tilt, roll or drift.

## Drift and slips

When the sample moves between views, the features are doubled or smeared by different amounts,
depending on when each view was taken. The translation fit follows smooth drift and sudden slips
of the sample together.

<!-- Figure: drift and slips, before and after fit="translations", including the recovered path. -->

The default, `fit="translations"`, also absorbs a centre error in parallel and laminography scans,
so you do not need to fit the centre separately before it. If you request `"centre"` and
`"translations"` together, a zero-mean translation convention separates their shared offset;
the data cannot distinguish those two descriptions of the same parallel-ray geometry.

## The setup angles

A wrong laminography tilt, axis tilt or detector roll misplaces each feature by an amount that
depends on its position. Per-view translations cannot represent it: the error stays after
`fit="translations"`.

<!-- Figure: 0.75-degree parallel-beam roll, with off-centre rows and matching difference images. -->

Roll is especially visible in parallel-beam data. A 0.75° roll about the beam can blur slices away
from the detector's centre row, even when the centre slice looks good. Inspect slices at several
heights. Small laminography or axis-tilt errors can be much less apparent in the image.

Adding `"tilt"` or `"roll"` estimates one angle for the whole scan. These are shared columns in
the same linear system as the image and per-view corrections. They are not an outer search over
repeated alignments. Fitting global angles still needs structure and angular coverage that make
them observable; an angle weakly constrained by the data is not an instrument calibration.

## Sample rotation

If the sample itself tips or turns slightly between views, translations remove most of the blur
but leave a residual at the finest features. Fitting rotations estimates three more parameters for
every view.

<!-- Figure: random per-view rotations, comparing translations with translations plus rotations. -->

Fit rotations only when the residual is visible. Their size prior is specified in angular units,
so changing the voxel size or binning does not make a rotation cheaper. The data must justify
rotations against that prior. This regularization does not resolve a wrong matching orientation:
on repetitive structures, such as a TEM grid, per-view rotations can lock onto a different copy of
the pattern. Check the fitted angles, and compare the image with the translation fit.

## Check the result

1. Compare the aligned reconstruction with your unaligned one and with the previous setting's
   result. An extra parameter should remove a visible artefact; if it does not, use the simpler fit.
2. Inspect the estimated values. `result.info["calibration"]` holds the fitted global values under
   the names used in `fit=`, and `result.poses` holds the per-view correction table. The returned
   `result.scan` carries both, so use it for subsequent reconstruction and saving.
3. Check that the global values are physically plausible: a tilt correction of several degrees, or
   per-view rotations of more than a fraction of a degree, can indicate a wrong model. Image
   sharpness and a lower training residual alone do not establish accurate poses.
4. Compare poses only after removing one common rigid object transform. Moving the whole object
   and moving every pose back predicts the same data. A centre offset and a nonzero mean
   parallel-ray translation also share a gauge; include the detector centre when scoring them.

`calibration` reports final global values: `centre` is a scalar for parallel data and a `(u, v)`
pair for cone data; `tilt` is the signed lab inclination `atan2(axis_y, axis_z)` in degrees; `roll`
is the detector angle from u towards v in degrees. Only requested globals appear. For a segmented
cone scan whose starting calibrations differ, their final values are listed in segment order.
`info["corrections"]` records their fitted increments, with centre increments in native detector
pixels and angular increments in degrees. Per-view tables use `(alpha, beta, phi, x, z, y)`;
rotations are radians and translations are physical lengths.

The alignment image uses a bounded working grid unless you supply `grid=`. The returned scan
retains the input detector sampling. Reconstruct that scan at the resolution you need; the
alignment volume is not automatically a final-resolution reconstruction.

## How the figures were made

The earlier draft used a watch mesh rendered with gVXR: 35° laminography, 360 of 720 views,
192 × 192 detector pixels of 0.27 mm, 10⁴ photons per pixel, reconstructed with FBP on a 192³ grid.
Errors were applied to the declared geometry (centre and tilt) or rendered into the data (drift,
slips and rotations). Pose errors were landmark RMS in native 0.135 mm pixels after one rigid
gauge, and timings were for one RTX 4070 Laptop GPU. Those figures are awaiting replacement and
requalification with this solver; their numerical results are not reused here.
