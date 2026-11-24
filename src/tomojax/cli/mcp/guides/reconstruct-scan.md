# Reconstruct a scan

1. For a file, use `inspect` to check projection shape, angle coverage, detector spacing and geometry, then `open` it. For a live scan, start from its handle.
2. Use `correct` on raw frames to obtain line integrals. Check the centre with `align` if it is uncertain. Pass the fitted scan member to reconstruction.
3. Call `reconstruct` with `dry_run=true` to check the method, grid and placement. For a raw acquisition larger than memory, check `reconstruct_slabs` with a destination path.
4. Run the reconstruction and inspect its three planes. Use `view` to examine off-centre structures. Save the result with `save` and report the path, method, grid and visible artefacts.
