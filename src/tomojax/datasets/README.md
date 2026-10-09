# tomojax.datasets

`tomojax.datasets` provides deterministic synthetic data generation for the
`tomojax simulate` workflow.

## Public API

- `SimConfig`
- `SimulatedData`
- `SimulationArtefacts`
- `make_phantom`
- `simulate`
- `simulate_to_file`
- simple phantom helpers such as `shepp_logan_3d`, `cube`, `sphere`, `blobs`,
  `random_cubes_spheres`, `rotated_centered_cube` and `lamino_disk`

`tomojax.datasets.api` adds `SimMetadata` and `validate_simulation_artefacts`.

## Dependency policy

Import from `tomojax.datasets` or `tomojax.datasets.api`, not internal
data-generation helpers.
