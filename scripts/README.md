# Scripts

This directory contains developer and data-preparation utilities that are not
installed as package entry points.

## Evidence generation

- `generate_alignment_before_after_128.py`: synthetic alignment evidence
  generator used by docs and tests.
- `alignment_visuals.py`: visualization helper for the evidence generator.

## Data preparation

- `nexus_data_wrangler.py`: practical NeXus/HDF5 conversion utility for raw
  laminography-style data. This script has focused contract tests.

## Manual experiments

- `exp_spdhg_bench.py` and `exp_spdhg_report.py`: manual SPDHG experiment driver
  and report helper.
- `perf_harness.py`: quick local CLI performance sweep helper.

## Real laminography diagnostics

Real-data diagnostics live under `scripts/real_laminography/`. They are manual tools
for local/laptop runs and are intentionally not package entry points.
