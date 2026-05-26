# Synthetic Tomography Workflow

Generate a synthetic projection dataset with `tomojax simulate`, then
reconstruct it with `tomojax recon`.

```bash
uv run tomojax simulate \
  --out synthetic_scan.nxs \
  --nx 64 --ny 64 --nz 64 \
  --nu 64 --nv 64 \
  --n-views 64 \
  --phantom random_shapes

uv run tomojax recon --data synthetic_scan.nxs --out synthetic_recon.nxs
```

For a Python equivalent, see
[`examples/simulate_and_reconstruct.py`](../examples/simulate_and_reconstruct.py).
