# Retained measurement evidence

Start with the [measurement guide](../../docs/measurements.md) for conclusions,
failed cases, and timing scope. This directory retains evidence, not library code.

Bulk raw JSON records are stored as **lossless `.json.gz` archives**. Summaries,
comparisons, source identifiers, and audits remain readable JSON. The
[archive catalog](archives.json) records original filenames, byte/line counts,
and SHA-256 hashes of both original and compressed bytes. No measurements or
failed attempts were removed during compression.

The retained spectral-withdrawal patch is also compressed as `.diff.gz`, with
its original bytes and hashes preserved in the same catalog.

The first archive cleanup converted 115 raw JSON files containing 2,004,829
lines. Original copies also remain in the local ignored `.artifacts/` directory;
that local backup is not required to read the retained archives.

## Read an archive

Python can load the records directly:

```python
import gzip
import json

with gzip.open("bench/reference/public-free-voxel-v1-schur.json.gz", "rt") as stream:
    result = json.load(stream)
```

To reproduce a historical script that expects the original filename, decompress
without removing the archive:

```bash
gzip -dk bench/reference/public-free-voxel-v1-schur.json.gz
```

Run those commands from the repository root. Historical reproduction commands
still write ordinary JSON; their output format has not changed. Compress new raw
records before retaining them here, and update the catalog and report links.
The ignore rules keep regenerated bulk JSON out of the source change list.
Small summaries and source metadata are explicitly exempted.

Source `.tar.gz` archives are separate from measurement JSON. They preserve the
implementation used for a particular run and must not be replaced by whatever
happens to be in the current working tree.
