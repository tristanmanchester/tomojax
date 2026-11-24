# Reconstruct a laboratory CT acquisition

1. Inspect the acquisition directory. Open the acquisition metadata file and check its angle range, pixel size, source distances and detector dimensions.
2. Identify flats, darks and exposure times. Open raw counts as frames. Record any assumed calibration value and use a correction preview to compare raw and corrected rows before applying corrections.
3. Check the rotation centre. Fit detector roll when the data supports it. Inspect the aligned scan and retain its geometry for reconstruction.
4. Check the reconstruction plan with `dry_run`, then reconstruct. Inspect central and off-centre slices for doubled edges, rings and streaks. Save the volume and manifest. Use `export` for a TIFF stack or raw volume.
