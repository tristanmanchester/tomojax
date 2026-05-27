# tomojax.datasets

`tomojax.datasets` provides deterministic synthetic data generation for the
`tomojax simulate` workflow.

## Public API

- `SimConfig`
- `SimMetadata`
- `SimulatedData`
- `SimulationArtefacts`
- `apply_simulation_artefacts`
- `make_phantom`
- `simulate`
- `simulate_to_file`
- `validate_simulation_artefacts`
- simple phantom helpers such as `shepp_logan_3d`, `cube`, `sphere`, `blobs`,
  `random_cubes_spheres`, and `lamino_disk`
- sidecar metadata loaders for generated datasets

## Dependency policy

Import from `tomojax.datasets`, not internal data-generation helpers.
