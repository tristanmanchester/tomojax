# Known Limitations

Current production-shaped coverage is intentionally narrower than the full v2
research plan. The detailed status table is in
[`support-matrix.md`](support-matrix.md).

- Real laminography is validated on the retained k11 staged workflow evidence,
  not on a broad multi-scan corpus.
- The original synthetic setup-global and pose-random `128^3` gates currently
  prove oracle/fixed-volume geometry recovery. They are strong solver evidence,
  but they are not truth-free stopped-reconstruction production passes.
- Truth-free stopped detector-centre recovery remains a research blocker: a
  wrong-geometry preview reconstruction can absorb the detector-centre error
  into the volume.
- Laminography axis/roll/theta recovery, object-frame drift, nuisance fitting,
  and combined bad-view/jump handling remain diagnostic or research surfaces
  until their dedicated reports contain production evidence.
- End-to-end Pallas defaults remain diagnostic. Reference/JAX paths are the
  current correctness baseline for publication-facing claims.
- JAX GPU selection currently requires exporting the venv NVIDIA library path
  before launching CUDA runs on the laptop.
