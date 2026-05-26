# tomojax.cli

`tomojax.cli` owns the grouped `tomojax` console script. The installed command
exposes only product workflows: `inspect`, `validate`, `preprocess`, `ingest`,
`convert`, `recon`, `align`, and `simulate`.

The `align` command defaults to 5-DOF per-projection pose correction. COR and
mixed setup and pose correction remain public modes, but mixed correction must
carry an explicit gauge policy.

Benchmark and diagnostic dispatch has been removed from the product CLI.
One-off runners and historical `tomojax dev ...` commands are archived outside
the publishable package.
