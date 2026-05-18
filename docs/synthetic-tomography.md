# Synthetic Tomography Workflow

The product-facing synthetic workflow is small and reproducible: generate a
standard projection dataset with `tomojax simulate`, then reconstruct it with
`tomojax recon`.

```bash
uv run tomojax simulate \
  --out synthetic_scan.nxs \
  --nx 64 --ny 64 --nz 64 \
  --nu 64 --nv 64 \
  --n-views 64 \
  --phantom random_shapes

uv run tomojax recon --data synthetic_scan.nxs --out synthetic_recon.nxs
```

The equivalent public-Python path is kept in
[`examples/simulate_and_reconstruct.py`](../examples/simulate_and_reconstruct.py).
It imports only from package facades such as `tomojax.geometry`,
`tomojax.forward`, and `tomojax.recon`.

Supported synthetic workflows should use the CLI command or public Python
facades shown here.
