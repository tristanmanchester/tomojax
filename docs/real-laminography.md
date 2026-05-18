# Real Laminography Workflow

The package-facing real-data path starts with inspection and validation, then
uses the public alignment command for detector-centre/COR correction and final
reconstruction.

```bash
uv run tomojax inspect /path/to/scan.nxs
uv run tomojax validate /path/to/scan.nxs
uv run tomojax align --data /path/to/scan.nxs \
  --mode cor \
  --out aligned.nxs
```

When data arrive as TIFF projection stacks, ingest them into the standard
contract before reconstruction or alignment:

```bash
uv run tomojax ingest ./projections \
  --angles angles.csv \
  --du 0.65 \
  --dv 0.65 \
  --out scan.nxs
```

The product claim is the CLI/API path above.
