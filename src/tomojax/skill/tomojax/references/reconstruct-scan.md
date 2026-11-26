# Reconstruct a scan

1. For a file, use `inspect` to check projection shape, angle coverage, detector spacing and geometry, then `open` it. For a live scan, start from its handle.
2. Use `correct` on raw frames to obtain line integrals; include `{"type": "Rings"}` if detector rings appear (it takes no width parameter). Check the center with `align` if it is uncertain. Pass the fitted scan member to reconstruction.
3. Call `reconstruct` with `dry_run=true` to check the method, grid and placement. For a raw acquisition larger than memory, check `reconstruct_slabs` with a destination path.
4. Run the reconstruction and inspect its three planes. Use `view` to examine off-center structures. Save the result with `save` and report the path, method, grid and visible artifacts.

Workflow `source` fields also accept acquisition paths. To supply measured arrays, open the scan and select `.projections`, or open with `kind=frames` and select `.counts` for raw counts. Use `inspect` with `preview=true` for a first look or a `correction_preview` with `preview` to examine corrected rows.
