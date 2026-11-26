# Check a reconstruction

1. Use `view` on the reconstruction to inspect all three planes. Check an off-center slice and a profile across a sharp edge. Report where you see blur, double edges, rings or streaks.
2. Use statistics to check the range and non-finite count. For a reference on the same grid, inspect a difference image. For a simulation, compare the volume with its phantom.
3. To compare solver settings, use `evaluate` on the source acquisition with the same held-out split for each candidate. Inspect slices for the leading settings at the same intensity limits. Report the score alongside the visible noise and resolution.

Use a handle member, session array, `.npy` file or explicit `file.h5::/dataset` for an array. For saved TomoJAX volumes, pass the path directly to `view` or as the `volume` in `project`. The volume axes are `(z, y, x)`. Explicit datasets use their stored axis order.
