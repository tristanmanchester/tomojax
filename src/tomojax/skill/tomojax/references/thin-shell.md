# Worked example: a thin bright shell

The user asks you to segment "the bright thin shell" of a sample and tune the reconstruction for it. The shell is a few voxels thick and touches two materials: air outside and a core inside. Both matter: a region grown from a seed leaks along partial-volume voxels into the core, and a single band around the shell mixes air with core.

## 1. Measure, then mark

Find the shell in the data first. A profile along one row through the middle of the volume shows it as a narrow peak between the air and the core:

```json
{"source": "@recon-…", "mode": "profile", "roi": [[64, 65], [64, 65], [0, 128]]}
```

Take the index of the peak as the shell's position on that row, and read the core and air levels from the same profile. Then mark it with `view` (`"at": [64, 64, x]`, or a pixel of the image you are looking at) and check the report: the value should be near the peak and its `percentile` high. Show the user and ask; never ask about a mark whose value is near the core or air level.

## 2. Segment by threshold, and check it is stable

Set `recon_id` to the volume, `shell_at` to the confirmed shell position (the profile's peak), and `core_at` and `air_at` to positions well inside the core and in the air. The core and air levels are medians of a small neighborhood; the shell level is the peak value itself, since a neighborhood around a shell one or two voxels thick is mostly core and air. The block keeps the bright voxels connected to the shell above a threshold between the core and shell levels, and checks the mask hardly changes when the threshold moves:

```python
from scipy import ndimage

volume = np.asarray(handles[recon_id].volume)


def level(at):
    """The median of the 3 x 3 x 3 neighborhood around a position."""
    z, y, x = (round(v) for v in at)
    return float(np.median(volume[z - 1 : z + 2, y - 1 : y + 2, x - 1 : x + 2]))


seed = tuple(round(v) for v in shell_at)
core, shell = level(core_at), float(volume[seed])
masks = {}
for fraction in (0.4, 0.5, 0.6):
    labels, _ = ndimage.label(volume > core + fraction * (shell - core))
    masks[fraction] = labels == labels[seed]


def dice(a, b):
    return float(2 * (a & b).sum() / (a.sum() + b.sum()))


shell_id = register_feature(masks[0.5], like=recon_id, name="shell")
{
    "core": core,
    "shell": shell,
    "voxels": int(masks[0.5].sum()),
    "stability": [dice(masks[0.4], masks[0.5]), dice(masks[0.5], masks[0.6])],
}
```

Overlaps above about 0.95 mean the boundary is set by the data, not by your threshold. A lower overlap means the threshold decides the boundary: the shell and core levels are too close for the noise, or the shell is so thin that a higher threshold breaks it apart (an overlap near 0). Lower the noise first (step 4), or ask the user where the boundary should be. Show the mask with `view` and `features=[shell_id]` before going on.

## 3. Score the shell against each side

Score the shell against the core and against the air separately, with surroundings that leave one voxel gap on each side. `margin=0` because a shell this thin has no interior; `edge_width` measures the shell's own profile here, so judge its edges by eye.

```python
import runpy
from importlib.resources import files

script = files("tomojax") / "skill" / "tomojax" / "scripts" / "feature_scores.py"
scores = runpy.run_path(str(script))["scores"]
air = level(air_at)
shell_mask = handles[shell_id].mask
around = ndimage.binary_dilation(shell_mask, iterations=4)
around &= ~ndimage.binary_dilation(shell_mask)
core_side = around & (volume > (air + core) / 2)
air_side = around & ~core_side
{
    "against_core": scores(volume, shell_mask, core_side, margin=0),
    "against_air": scores(volume, shell_mask, air_side, margin=0),
}
```

## 4. Tune and check

Sweep candidates as in [tuning-without-truth.md](tuning-without-truth.md), scoring each against the core side (usually the harder contrast) on these masks, check the leaders with `evaluate`, and show the best two side by side with the shell outlined: `view` with `source` as a list, the same `limits` and `features=[shell_id]`. A mask cut from one reconstruction favors that reconstruction when you score others on it (in one test, the first look scored above a candidate that fitted the held-out projections twice as well), so let `evaluate` decide between close candidates, and recompute the threshold mask on the chosen reconstruction before you report its size or export it.
