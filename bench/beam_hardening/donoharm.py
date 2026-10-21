"""Do-no-harm checks: the 40 kV walnut, and monochromatic synchrotron scans.

``walnut``: FIPS Walnut 1, orbit 2, binned 2 x 2, reconstructed by FDK after
the method. Its relative L2 error inside the walnut against the authors'
three-orbit reference, after the one scale factor that minimises it
(``scaled_in_walnut``), must not exceed the uncorrected one: the reference is
itself in the beam's polychromatic units, and a gauge is a choice of unit.
The raw error and the factor (``scale``) are reported beside it.

``mono``: scans whose metadata records a monochromator (APS 32-ID at 8 keV,
Diamond I12 at 54.3 keV) have no beam hardening, so a method's pointwise
curve must be linear within noise. For each method that returns corrected
line integrals, ``nonlinearity`` is the largest deviation of the curve from
its best line through the origin over the 1st to 99th percentile of the
measured line integrals, relative to the curve's value at the 99th; ``noise``
is half the largest difference between the curves fitted to the even and to
the odd views. The check passes when the nonlinearity is at most three times
the noise, or 0.5 %.
"""

from __future__ import annotations

from pathlib import Path
import sys

import figures as plots
import methods
import numpy as np

WALNUT = Path("~/data/walnuts/Walnut1").expanduser()
MONO = {
    "aps-32id": {
        "path": Path(
            "/home/tristan/PycharmProjects/tomocupy/tomocupy-stream/tests/data/test_data.h5"
        ),
        "metadata": {"kv": None, "monochromatic_kev": 8.0, "material": "H2O"},
        "center": -15.1,
    },
    "i12": {
        "path": Path("~/data/diamond-3339629/derived/bin4_every2nd_view.nxs").expanduser(),
        "metadata": {"kv": None, "monochromatic_kev": 54.3, "material": "H2O"},
        "center": -34.5,
    },
}


def _peak_gb() -> float:
    """Peak resident memory of the process so far, in GB."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def walnut(names: list[str], figures: Path | None = None) -> list[dict]:
    """Relative L2 error inside the walnut for every method, and its central slice."""
    import tomojax as tj

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import walnut as bench_walnut

    data, vectors = bench_walnut.load_orbits(WALNUT, [2], 1, binning=2)
    geometry = {
        "type": "cone_vec",
        "DetectorRowCount": data.shape[0],
        "DetectorColCount": data.shape[2],
        "Vectors": vectors,
    }
    scan = tj.Scan.from_astra(data, geometry, bench_walnut.volume_geometry(10))
    del data
    counts = np.exp(-np.asarray(scan.projections))
    # Unit flats and zero darks give back exactly these line integrals.
    frames = tj.Frames(
        counts=counts,
        geometry=scan.geometry,
        flats=np.ones((1, *counts.shape[1:]), np.float32),
        darks=np.zeros((1, *counts.shape[1:]), np.float32),
    )
    del scan
    truth = bench_walnut.reference(WALNUT)
    metadata = {
        "kv": 40.0,
        "anode": "W",
        "window": ["Be", 0.3],  # the FleX-ray tube's window, recorded with 2DeteCT
        "filters": [],
        "scintillator": ["CsI", None],
        "material": "H2O",
    }
    record: dict = {"case": "walnut1-orbit2-bin2"}
    images = {}
    for name in names:
        try:
            result = methods.METHODS[name](frames, metadata, None)
        except Exception as error:  # truth-only controls, or a failing method
            record[name] = {"error": f"{type(error).__name__}: {error}"}
            continue
        volume = np.asarray(result.volume)
        record[name] = bench_walnut.compare(volume, truth)
        inside = truth > 0.1 * float(np.percentile(truth, 99.9))
        v, t = volume[inside].astype(np.float64), truth[inside].astype(np.float64)
        scale = (v @ t) / (v @ v)
        record[name]["scaled_in_walnut"] = float(np.linalg.norm(v * scale - t) / np.linalg.norm(t))
        record[name]["scale"] = float(scale)
        record[name] |= {"peak_rss_gb": _peak_gb()}
        images[name] = scale * volume[:, :, volume.shape[2] // 2]
        del result
    if figures is not None:
        figures.mkdir(parents=True, exist_ok=True)
        middle = truth[:, :, truth.shape[2] // 2]
        centre = ((middle.shape[0] - 1) / 2, (middle.shape[1] - 1) / 2)
        title = "Walnut 1, central slice; each image times its best scale to the reference"
        plots.save(figures / "walnut1.png", middle, images, centre, 0.1, title)
    return [record]


def curve(
    measured: np.ndarray, corrected: np.ndarray, bins: int = 40
) -> tuple[np.ndarray, np.ndarray, float]:
    """Binned deviation of ``corrected`` from its best line through the origin, as above.

    A pointwise curve needs few samples: at most four million pixels, evenly strided.
    """
    stride = max(1, np.size(measured) // 4_000_000)
    q = np.ravel(measured)[::stride].astype(np.float64)
    p = np.ravel(corrected)[::stride].astype(np.float64)
    low, high = np.percentile(q[q > 0], [1, 99])
    keep = (q >= low) & (q <= high)
    q, p = q[keep], p[keep]
    slope = (q @ p) / (q @ q)
    which = np.minimum(((q - low) / (high - low) * bins).astype(int), bins - 1)
    count = np.maximum(np.bincount(which, minlength=bins), 1)
    deviation = (np.bincount(which, p - slope * q, bins) / count) / (slope * high)
    centres = np.bincount(which, q, bins) / count
    return centres, deviation, float(slope)


def mono(names: list[str], figures: Path | None = None) -> list[dict]:
    """Linearity of every method's curve on each monochromatic scan, drawn as curves."""
    import h5py

    import tomojax as tj

    records = []
    for key, spec in MONO.items():
        with h5py.File(spec["path"]) as f:
            data = f["exchange/data"] if "exchange" in f else f["entry/instrument/detector/data"]
            nv, nu = data.shape[1:]
        # Pixel units: neither file records a pixel pitch.
        detector = tj.Detector(nu=nu, nv=nv, du=1.0, dv=1.0, center=(spec["center"], 0.0))
        frames = tj.load_frames(spec["path"], detector=detector)
        measured = np.asarray(frames.corrected().projections)
        record: dict = {"case": key}
        for name in names:
            parts = []
            for subset in (slice(None), slice(0, None, 2), slice(1, None, 2)):
                part = frames.selected(subset)
                try:
                    result = methods.METHODS[name](part, spec["metadata"], None)
                except Exception as error:  # truth-only controls, or a failing method
                    record[name] = {"error": f"{type(error).__name__}: {error}"}
                    break
                if result.projections is None:
                    record[name] = {"error": "no corrected line integrals"}
                    break
                parts.append(curve(measured[subset], result.projections))
                del result
            if len(parts) < 3:
                continue
            nonlinearity = float(np.abs(parts[0][1]).max())
            noise = float(np.abs(parts[1][1] - parts[2][1]).max() / 2)
            if figures is not None:
                figures.mkdir(parents=True, exist_ok=True)
                lines = {v: p[:2] for v, p in zip(("all", "even", "odd"), parts, strict=True)}
                limit = max(3 * noise, 0.005)
                plots.curves(figures / f"{key}-{name}.png", lines, limit, f"{key}: {name}")
            record[name] = {
                "nonlinearity": nonlinearity,
                "noise": noise,
                "slope": parts[0][2],
                "linear": nonlinearity <= max(3 * noise, 0.005),
                "peak_rss_gb": _peak_gb(),
            }
        records.append(record)
        print(record, flush=True)
    return records
