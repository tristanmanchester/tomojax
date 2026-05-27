# tomojax.cli

`tomojax.cli` implements the `tomojax` console script with commands: `inspect`,
`validate`, `preprocess`, `ingest`, `convert`, `recon`, `align`, and
`simulate`.

`align` defaults to 5-DOF per-projection pose correction. Mixed setup and pose
correction requires an explicit gauge policy.
