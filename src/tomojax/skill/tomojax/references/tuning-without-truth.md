# Tune a reconstruction without ground truth

On a real scan there is no phantom to compare against. Tune on what the user cares about instead: a feature you have agreed with them, scored on the data itself, with held-out projections as the check that the volume still explains the measurements.

## Steps

1. **Agree the feature.** Reconstruct a first look (FBP is quick), then mark the feature on a slice with `view` and `marks` and ask the user whether that is what they mean. See [pointing-and-features.md](pointing-and-features.md).
2. **Make the feature and its surroundings.** Turn the confirmed mark into a feature handle with the `feature` tool: a `seeded` recipe from the mark's position for a region of similar values, or a shape recipe from the mark's corners or center. Make a surrounding band with `{"type": "band", "feature": FEATURE_ID, "inner": 1, "outer": 4}`, which leaves one voxel between the two so partial-volume edge voxels count in neither. The surroundings should be one material: around a thin shell, a band reaches both the air outside and the material inside, so keep the side you care about, for example by intersecting the band with a seeded region of the inside in `python` and registering the result with `register_feature`.
3. **Sweep candidates in one `python` call.** Reconstruct each candidate on the resident scan, register it, and score it on the feature (block below). One call keeps the data on the GPU and returns a small table instead of many tool results. Start from TomoJAX's own choice and step a decade either side: left unset, FISTA estimates `tv_weight` from detector noise and interpolation error, stops at the estimated noise level (up to 200 updates), and reports `resolved_tv_weight`, `effective_iterations`, and `early_stop` in `info`. A good weight can lie on either side of that estimate. For unpenalized `cgls` or full-data `sirt`, omitting `iterations` stops at the estimated projection discrepancy up to 50 updates (with momentum on full-data `sirt`); on real scans the 50-update cap is often reached (`termination="iteration_limit"`). Do not run unpenalized CGLS or SIRT for hundreds of updates without checking held-out projections and slices, because both fit noise past an early optimum.
4. **Check against the measurements.** Run `evaluate` on the two or three best settings. A candidate that scores well on the feature but fits held-out projections worse is smoothing away real structure.
5. **Show the user.** View the best two or three side by side on the same slice with the feature outlined (`view` with `features`), and say which you recommend and why.
6. **Run the full volume** with the chosen settings, then `save`.

## Sweep and score

Set `scan_id`, `feature_id` and `surround_id` to your handles first. The scores come from `scripts/feature_scores.py` in this Skill, which the worker can load from the installed package.

```python
import runpy
from importlib.resources import files

script = files("tomojax") / "skill" / "tomojax" / "scripts" / "feature_scores.py"
scores = runpy.run_path(str(script))["scores"]
scan = handles[scan_id]
feature, surround = handles[feature_id].mask, handles[surround_id].mask
rows = []
for weight in (None, 1e-5, 1e-4, 1e-3):
    recon = tj.reconstruct(scan, "fista", tv_weight=weight, nonnegative=True)
    volume = np.asarray(recon.volume)
    weight = recon.info["resolved_tv_weight"]
    rows.append({"tv_weight": weight, "handle": register(recon), **scores(volume, feature, surround)})
rows
```

Each row has the candidate's handle, so `view` can show it with `features=[feature_id]`. Every candidate stays in memory until closed: once you have chosen, close the others with `control` and `action="close"`, which matters for large volumes.

## Check against held-out projections

`evaluate` reconstructs each candidate without every fifth view (by default) and scores how well it predicts the views it did not see. Give the scan and the candidates, each tagged with its operation:

```json
{
  "source": "@scan-1a2b3c4d",
  "candidates": [
    {"type": "reconstruct", "method": "fista", "tv_weight": 1e-5, "nonnegative": true},
    {"type": "reconstruct", "method": "fista", "nonnegative": true},
    {"type": "reconstruct", "method": "fbp"}
  ]
}
```

Each entry of the result's `scores` repeats its `candidate` and gives `relative_l2`, the held-out error relative to the measured projections: lower is better. Compare candidates only within one call, which uses one split for all of them. Describe `evaluate` for the split options.

## Read the scores

| Score | Meaning | Better |
| --- | --- | --- |
| `contrast_to_noise` | Difference of the mean inside the feature and in its surroundings, over their pooled standard deviation | Higher |
| `noise` | Standard deviation inside the feature, in the volume's units | Lower, if the feature should be uniform |
| `edge_width` | 10-90% rise across the feature boundary, in voxels | Lower |

Both use the feature's interior, two voxels in from its edge, so blurred edges do not count as noise. For a feature too thin to have an interior, such as a shell a few voxels thick, pass `margin=0` to `scores`. `edge_width` assumes the feature is thicker than the blur: across a thin shell it measures the shell's own profile, so judge the edges of a thin feature by eye and by the held-out score instead.

Stronger regularization lowers noise and raises contrast-to-noise until it starts blurring edges, which widens `edge_width` and worsens the held-out score. Choose the setting where contrast-to-noise has stopped improving much and the edge is still sharp. Compare candidates only on the same masks and slices.
