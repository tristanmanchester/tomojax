# tomojax.datasets

`tomojax.datasets` generates deterministic synthetic scans for the
`tomojax simulate` workflow.

## Public API

- `SimConfig`, `SimulationArtefacts`, `SimulatedData`
- `simulate`, `simulate_to_file`, `make_phantom`
- phantoms: `shepp_logan_3d`, `cube`, `sphere`, `blobs`, `random_cubes_spheres`,
  `rotated_centered_cube` and `lamino_disk`

Import from `tomojax.datasets`, not from its implementation modules.

## Calibrated count simulation

`DetectorModel.from_flats(flats, darks)` estimates static response maps, effective
counts per detected photon and read noise from repeated reference images. Arrays
are `(frame, row, column)` in recorded count units. With only one dark image,
provide an independently known `read_noise`; zero is an explicit assumption.

```python
from tomojax.datasets import DetectorModel

model = DetectorModel.from_flats(flats, darks)
measurement = model.simulate(
    line_integrals, flat_frames=100, dark_frames=20, seed=7,
    gain_change=sample_gain_ratio, offset_change=sample_offset_counts,
)
```

`gain_change` is the sample/calibration sensitivity ratio; `offset_change` is the
sample minus calibration dark offset in counts. Both can be per-pixel fields or
vary by view. The model draws sample counts and finite flat/dark averages before
normalizing and taking the logarithm. Static sensitivity shared by sample and
flats cancels. Reference noise remains shared across all views, producing a
different persistent error in each detector row.

The result retains raw counts, flat/dark references and its model/seed.
`expected` is noiseless sample counts corrected with the sampled references;
`ideal` uses noiseless references and retains any imposed detector changes.
Neither is the mathematical expectation of logarithmically transformed noisy
counts. Compare these with the input line integrals to separate systematic
response changes, reference error and sample noise.

This independent-pixel noise model does not model spatially correlated noise or
temporal illumination drift, and matching flat noise does not establish realistic
sample-ring statistics. Use independent projection data to test reconstruction
accuracy; the detector model does not prescribe a projector.
