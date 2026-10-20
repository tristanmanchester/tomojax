# tomojax.datasets

`tomojax.datasets` generates deterministic synthetic scans for the
`tomojax simulate` workflow.

## Public API

- `SimConfig`, `SimulationArtefacts`, `SimulatedData`
- `simulate`, `simulate_to_file`, `make_phantom`
- phantoms: `shepp_logan_3d`, `cube`, `sphere`, `blobs`, `random_cubes_spheres`,
  `rotated_centered_cube` and `lamino_disk`

Import from `tomojax.datasets`, not from its implementation modules.
