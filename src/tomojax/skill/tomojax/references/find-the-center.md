# Find the rotation center

1. Inspect the geometry, detector pixel size and angle coverage. Correct raw frames before fitting.
2. For a parallel scan with opposing projections, use `align` with `method="opposite-views"`. For a geometry fit, use `describe` to check the alignment settings.
3. Use a `center_sweep` preview on the input scan. Compare the same slice across nearby signed horizontal offsets in detector pixels. Look for sharp single edges and record the range the images resolve.
4. Report the fitted center with its units and sign convention. Pass the alignment's scan member to `reconstruct`.
