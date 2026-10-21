# Beam hardening bench

Scores beam hardening corrections on reconstruction error against references
the correction never sees: simulated truth, a second scan of the same object
at a harder spectrum, and scans that have no beam hardening. It shares no code
with the corrections it scores; each method is a function in `methods.py`
that takes raw `tj.Frames` and the settings a scanner records.

One command per family, from the repository root:

```bash
python bench/beam_hardening/run.py simulation --split dev --methods none,oracle,exact
python bench/beam_hardening/run.py lamino --split dev --methods none,oracle,exact
python bench/beam_hardening/run.py 2detect --split dev --methods none
python bench/beam_hardening/run.py walnut --methods none
python bench/beam_hardening/run.py mono --methods none
```

Each writes a JSON report (`--output`, by default `beam_hardening.json` in
`$LAB_ARTIFACTS` or the current directory) with every case's metrics and a
summary per method, and one figure per case in `<output>-figures/`: every
method beside the reference on the reference's grey window, with radial
profiles (walnut: each image times its best scale; mono: each curve's
deviation from a straight line).

## Families

**simulation**: 24 random phantoms of z-aligned cylinders in three families
(`cases.py`): one material; a low-Z matrix with two to four dense rods and
perhaps a void; a tube wall around a filling with one dense rod. Tube voltage
40-225 kV from a tungsten anode (SpekPy or xpecgen), window, filters,
scintillator (CsI, Gd2O2S, CdWO4), dose and detector blur are drawn at
random, and so are three settings the scanner does not record: the effective
anode angle, extra aluminium filtration and the scintillator's thickness.
`simulate.py` renders each case with gVXR in its own environment (set
`GVXR_PYTHON`; default `~/Projects/tomojax/.artifacts/gvxr-env/bin/python`),
checks gVXR's path lengths against exact ray-cylinder chords and its spectral
integration against an independent sum over energies, and the bench deletes
the case once scored. Cone beam, 720 views, 400 × 24 detector pixels of 0.2 mm,
magnification 2, 400 × 400 × 28 voxels of 0.1 mm. Scores are in `score.py`.

**lamino**: 12 plates, 1-4 mm thick, alone or with dense rods through them,
in parallel-beam laminography with the rotation axis leaning 30-40° towards
the beam; 720 views of 200 × 150 pixels and 200 × 200 × 50 voxels of 0.1 mm.
Filtered backprojection is not quantitative in laminography, so the bench
reconstructs these with 50 CGLS iterations, and scores the plate's central
slab clear of its faces. Laminography never measures the frequencies within
the tilt angle of the axis, which a plate is made of: the `exact` control's
attenuation is 1-7 % off (3-5 % at 400 iterations). The `exact` control
therefore always runs there, and every method's mu bias and cupping are
scored against its image (the values against the truth are kept under
`vs_truth`); line-integral scores are unaffected.

**2detect**: 2DeteCT slices (`detect2.py`, `detect2_score.py`). The 60 kV
unfiltered scan (mode 3) is corrected and compared with the 90 kV
Thoraeus-filtered high-dose scan (mode 2) of the same slice, on the published
segmentation's tube wall and filler and on its objects split into light
(food) and dense (lava stone) ones. Slices 2831-5000 and 5871 onwards were
taken on a replacement detector (the paper's Table 2), whose 60:90 ratios
differ from the first detector's, so statistics across slices mix the two.

**walnut** and **mono**: do-no-harm checks (`donoharm.py`).

## Held-out sets

Simulated cases (cone or laminography) with titanium or above 150 kV
(`--split heldout`) and the 2DeteCT out-of-distribution slices (5521 onwards;
5920, from the noise study, is excluded) are held out: corrections are not
tuned on them, and they are scored once per frozen method.

## Controls

`none` reconstructs without correction. `oracle` applies the exact
single-material curve, from the simulated spectrum, of the material carrying
most attenuation: exact for one material, and what a correctly calibrated
single-material correction gives for several. `exact` reconstructs the
noiseless true line integrals: the floor of every metric.

The method under test, `spectrum`, applies
`tomojax.corrections.BeamHardening.from_spectrum` for the recorded tube
settings and the sample's named material.
