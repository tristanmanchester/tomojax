r"""Align every walnut of the FIPS collection, fetching only the files it uses.

For each walnut of Der Sarkissian et al.'s collection (42 walnuts, Zenodo
records 2686726 onwards, CC BY 4.0; see ``bench/walnut.py``) this reads, by
HTTP range requests into the walnut's 6 GB zip, every Nth projection of the
three orbits, their dark and flat fields, both geometry records and the
published reference reconstruction, about 1.6 GB, and decodes them in
memory. It then aligns the three orbits from the scanner's original geometry
with ``tj.align``, compares the recovered heights with the authors'
correction, reconstructs (non-negative least squares) with the original,
aligned and corrected geometries against the reference, and appends one JSON
line per walnut (a failed walnut's line holds the error, and a rerun tries it
again), with each reconstruction's three central slices (float16) and the
three pose tables (``walnutNN_slices.npz``, about 5 MB) to draw figures from.
Nothing else is written, and the next walnut downloads while this one is
aligned; a rerun skips walnuts already in the results.

    python bench/walnut_collection.py --out .artifacts/walnut-collection
    python bench/walnut_collection.py --out results/ --walnuts 1-10 --previews
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import http.client
import io
import json
import logging
from pathlib import Path
import threading
import time
import traceback
from typing import Any
import urllib.error
import urllib.request
import zipfile
import zlib

import imageio.v3 as iio
import numpy as np
from walnut import COLS, ROWS, VIEWS, compare, volume_geometry
from walnut_alignment import _orbit_one_fixed

import tomojax as tj

RECORDS = (2686726, 2686971, 2687387, 2687635, 2687897, 2688112)
WORKERS = 16  # parallel range requests; Zenodo drops connections at 32
REQUESTS_PER_SECOND = 1.5  # all threads together; about 10 minutes a walnut


def walnut_urls() -> dict[int, str]:
    """Each walnut's zip on Zenodo, by walnut number."""
    urls = {}
    for record in RECORDS:
        listing = json.loads(_get(f"https://zenodo.org/api/records/{record}"))
        for item in listing["files"]:
            name = item["key"]
            if name.startswith("Walnut") and name.endswith(".zip"):
                urls[int(name[6:-4])] = item["links"]["self"]
    return dict(sorted(urls.items()))


class _Pace:
    """At most ``rate`` requests a second across every thread: Zenodo answers 429 beyond."""

    def __init__(self, rate: float) -> None:
        self.interval, self.next, self.lock = 1.0 / rate, 0.0, threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next)
            self.next = start + self.interval
        time.sleep(max(0.0, start - now))


_PACE = _Pace(REQUESTS_PER_SECOND)


def _get(url: str, first: int | None = None, last: int | None = None) -> bytes:
    """``url``'s bytes (``first`` to ``last`` inclusive), paced and retried with back-off."""
    headers = {} if first is None else {"Range": f"bytes={first}-{last}"}
    for attempt in range(10):
        _PACE.wait()
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=120
            ) as r:
                return r.read()
        except urllib.error.HTTPError as error:
            if attempt == 9 or error.code not in {429, 500, 502, 503, 504}:
                raise
            # Too many requests or the server busy: wait as long as it asks.
            time.sleep(float(error.headers.get("Retry-After") or 2 ** min(attempt, 6)))
        except (OSError, http.client.HTTPException):  # dropped connections and timeouts
            if attempt == 9:
                raise
            time.sleep(2 ** min(attempt, 6))
    raise AssertionError("unreachable")


class _Remote(io.RawIOBase):
    """A remote file read by range requests, enough for ``zipfile`` to parse its index."""

    def __init__(self, url: str) -> None:
        self.url, self.pos = url, 0
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=120) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = (offset, self.pos + offset, self.size + offset)[whence]
        return self.pos

    def readinto(self, buffer: Any) -> int:
        end = min(self.size, self.pos + len(buffer))
        if end <= self.pos:
            return 0
        data = _get(self.url, self.pos, end - 1)
        buffer[: len(data)] = data
        self.pos += len(data)
        return len(data)


class RemoteZip:
    """The members of a zip on a web server, fetched in parallel without the rest."""

    def __init__(self, url: str) -> None:
        self.url = url
        index = zipfile.ZipFile(io.BufferedReader(_Remote(url), buffer_size=1 << 16))
        self.members = {info.filename: info for info in index.infolist()}

    def read(self, name: str) -> bytes:
        """Member ``name``'s bytes, checked against the index's CRC."""
        info = self.members[name]
        # The local header (30 bytes, then name and extra field) precedes the data;
        # its extra field can differ from the index's, so over-read and parse it.
        start = info.header_offset
        blob = _get(self.url, start, start + 30 + len(name.encode()) + 1024 + info.compress_size)
        skip = 30 + int.from_bytes(blob[26:28], "little") + int.from_bytes(blob[28:30], "little")
        body = blob[skip : skip + info.compress_size]
        data = zlib.decompress(body, -15) if info.compress_type == zipfile.ZIP_DEFLATED else body
        if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
            raise OSError(f"{name}: corrupt download")
        return data

    def read_all(self, names: list[str]) -> list[bytes]:
        """Members ``names``, fetched in parallel, in order."""
        with ThreadPoolExecutor(WORKERS) as pool:
            return list(pool.map(self.read, names))


def _image(data: bytes) -> np.ndarray:
    # The scanner reads out in portrait mode: flip rows, then transpose (bench/walnut.py).
    return np.transpose(np.flipud(np.asarray(iio.imread(data, extension=".tif"), np.float32)))


def fetch(n: int, url: str, *, every: int, reference: bool) -> dict[str, Any]:
    """Walnut ``n``'s absorption data, both geometry records and the reference, in memory."""
    archive = RemoteZip(url)
    root = f"Walnut{n}"
    indices = range(VIEWS, 0, -every)  # read in reverse, as the authors' scripts do
    data = np.empty((ROWS, 3 * len(indices), COLS), np.float32)
    vectors: dict[str, list[np.ndarray]] = {"original": [], "corrected": []}
    for orbit in (1, 2, 3):
        folder = f"{root}/Projections/tubeV{orbit}"
        for name, rows in vectors.items():
            text = archive.read(f"{folder}/scan_geom_{name}.geom").decode()
            rows.append(np.loadtxt(io.StringIO(text))[0:VIEWS:every])
        dark, *flats = (_image(b) for b in archive.read_all(
            [f"{folder}/di000000.tif", f"{folder}/io000000.tif", f"{folder}/io000001.tif"]
        ))  # fmt: skip
        flat = np.mean(flats, axis=0) - dark
        names = [f"{folder}/scan_{index:06d}.tif" for index in indices]
        start = (orbit - 1) * len(indices)
        for view, raw in enumerate(archive.read_all(names)):
            image = (_image(raw) - dark) / flat
            data[:, start + view, :] = -np.log(np.clip(image, 1e-6, None))
    truth = None
    if reference:
        prefix = f"{root}/Reconstructions/full_AGD_50_"
        names = sorted(name for name in archive.members if name.startswith(prefix))
        zyx = np.stack([np.asarray(iio.imread(b, extension=".tiff"), np.float32)
                        for b in archive.read_all(names)])  # fmt: skip
        truth = np.transpose(zyx, (2, 1, 0))
    return {
        "data": data,
        "vectors": {name: np.concatenate(v) for name, v in vectors.items()},
        "reference": truth,
    }


def evaluate(n: int, fetched: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Align walnut ``n`` from its original geometry; compare heights and reconstructions."""
    proj = {"type": "cone_vec", "DetectorRowCount": ROWS, "DetectorColCount": COLS}

    def scan(name: str) -> tj.Scan:
        astra = proj | {"Vectors": fetched["vectors"][name]}
        return tj.Scan.from_astra(fetched["data"], astra, volume_geometry()).binned(args.bin)

    original, corrected = scan("original"), scan("corrected")
    start = time.perf_counter()
    result = tj.align(original, levels=args.levels)
    summary: dict[str, Any] = {"walnut": n, "align_seconds": time.perf_counter() - start}
    aligned = _orbit_one_fixed(result.scan, original)
    recorded = np.asarray(original.poses)[:, 4]
    per = len(original.angles) // 3
    heights = {
        "tomojax": (np.asarray(aligned.poses)[:, 4] - recorded).reshape(3, per).mean(axis=1),
        "authors": (np.asarray(corrected.poses)[:, 4] - recorded).reshape(3, per).mean(axis=1),
    }
    summary["orbit_heights_mm"] = {k: [round(float(h), 4) for h in v] for k, v in heights.items()}
    options = {"iterations": args.iterations, "tv_weight": 0.0, "nonnegative": True}
    volumes = {}
    for name, item in (("original", original), ("aligned", aligned), ("corrected", corrected)):
        start = time.perf_counter()
        volumes[name] = np.asarray(tj.reconstruct(item, "fista", **options).volume)
        summary[f"{name}_seconds"] = time.perf_counter() - start
    truth = fetched["reference"]
    if truth is not None:
        summary["error_in_walnut"] = {
            name: round(compare(volume, truth)["relative_l2_in_walnut"], 4)
            for name, volume in volumes.items()
        }
    # Against the authors' geometry reconstructed the same way: what alignment leaves.
    summary["aligned_vs_corrected"] = round(
        compare(volumes["aligned"], volumes["corrected"])["relative_l2_in_walnut"], 4
    )
    # The data barely fix the object's sideways position in a cone beam (moving
    # it needs the along-beam dy, which alignment leaves alone), so alignment
    # may place it a voxel or so from the authors'. The same comparisons after
    # moving the aligned volume onto the corrected one separate that from focus.
    shift, registered = _registered(volumes["aligned"], volumes["corrected"])
    summary["aligned_shift_voxels"] = [round(float(s), 2) for s in shift]
    summary["aligned_vs_corrected_registered"] = round(
        compare(registered, volumes["corrected"])["relative_l2_in_walnut"], 4
    )
    if truth is not None:
        summary["aligned_error_in_walnut_registered"] = round(
            compare(registered, truth)["relative_l2_in_walnut"], 4
        )
        volumes["reference"] = truth
    poses = {f"{name}_poses": np.asarray(item.poses) for name, item in
             (("original", original), ("aligned", aligned), ("corrected", corrected))}  # fmt: skip
    _save_slices(args.out / f"walnut{n:02d}_slices.npz", volumes, poses)
    if args.previews:
        _preview(args.out / f"walnut{n:02d}.png", volumes)
    return summary


def _slices(volume: np.ndarray) -> dict[str, np.ndarray]:
    """The three central orthogonal slices of an ``(x, y, z)`` volume."""
    x, y, z = (size // 2 for size in volume.shape)
    return {"xz": volume[:, y, :], "yz": volume[x, :, :], "xy": volume[:, :, z]}


def _registered(volume: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The shift (voxels) best moving ``volume`` onto ``target``, and the moved volume.

    The peak of their cross-correlation, refined to a fraction of a voxel by a
    parabola through its neighbours along each axis.
    """
    from scipy import fft, ndimage

    cross = fft.rfftn(target) * np.conj(fft.rfftn(volume))
    correlation = fft.irfftn(cross, s=volume.shape, workers=-1)
    peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    shift = np.zeros(3)
    for axis, (index, size) in enumerate(zip(peak, correlation.shape, strict=True)):
        around = [list(peak) for _ in range(3)]
        for k, step in enumerate((-1, 0, 1)):
            around[k][axis] = (index + step) % size
        below, at, above = (correlation[tuple(p)] for p in around)
        curvature = below - 2 * at + above
        fraction = 0.5 * (below - above) / curvature if curvature < 0 else 0.0
        shift[axis] = (index + size // 2) % size - size // 2 + fraction
    return shift, ndimage.shift(volume, shift, order=3, mode="constant")


def _save_slices(path: Path, volumes: dict[str, np.ndarray], poses: dict[str, np.ndarray]) -> None:
    """Each volume's central slices, as float16, and the poses, to draw figures from later."""
    arrays = {
        f"{name}_{plane}": image.astype(np.float16)
        for name, volume in volumes.items()
        for plane, image in _slices(volume).items()
    }
    np.savez_compressed(path, **arrays, **poses)


def _preview(path: Path, volumes: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    peak = float(np.percentile(volumes["corrected"], 99.9))
    fig, axes = plt.subplots(2, len(volumes), figsize=(3.3 * len(volumes), 6.6), squeeze=False)
    for col, (name, volume) in enumerate(volumes.items()):
        slices = _slices(volume)
        for row, plane in enumerate(("xz", "xy")):
            ax = axes[row, col]
            ax.imshow(np.rot90(slices[plane]), cmap="gray", vmin=0, vmax=peak)
            ax.set_xticks([]), ax.set_yticks([])
        axes[0, col].set_title(name if name == "reference" else f"{name} geometry")
    axes[0, 0].set_ylabel("vertical slice")
    axes[1, 0].set_ylabel("axial slice")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def _walnuts(text: str) -> list[int]:
    chosen: list[int] = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        chosen.extend(range(int(first), int(last or first) + 1))
    return chosen


def main() -> None:
    """Run the collection, resuming after the walnuts already recorded."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True, help="Directory for results.jsonl")
    parser.add_argument("--walnuts", default="1-42", help="Walnut numbers, e.g. 1-10,17")
    parser.add_argument("--every", type=int, default=4, help="Use every Nth view")
    parser.add_argument("--bin", type=int, default=2, help="Average N x N detector pixels")
    parser.add_argument("--levels", default="4,2", help="Alignment resolution levels")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--no-reference",
        dest="reference",
        action="store_false",
        help="Skip the published reference (saves about 0.5 GB a walnut)",
    )
    parser.add_argument("--previews", action="store_true", help="Write a slice PNG per walnut")
    args = parser.parse_args()
    args.levels = tuple(int(f) for f in args.levels.split(","))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # The scanner's TIFFs carry a malformed colour-map tag, which tifffile skips.
    logging.getLogger("tifffile").setLevel(logging.CRITICAL)
    args.out.mkdir(parents=True, exist_ok=True)
    results = args.out / "results.jsonl"
    done = set()
    if results.exists():
        records = [json.loads(line) for line in results.read_text().splitlines() if line]
        done = {record["walnut"] for record in records if "error" not in record}
    urls = walnut_urls()
    todo = [n for n in _walnuts(args.walnuts) if n in urls and n not in done]
    logging.info("%d walnuts to do, %d already recorded", len(todo), len(done))

    def download(n: int) -> dict[str, Any]:
        start = time.perf_counter()
        fetched = fetch(n, urls[n], every=args.every, reference=args.reference)
        fetched["seconds"] = time.perf_counter() - start
        return fetched

    with ThreadPoolExecutor(1) as prefetch:
        pending: Future[dict[str, Any]] | None = (
            prefetch.submit(download, todo[0]) if todo else None
        )
        for i, n in enumerate(todo):
            assert pending is not None
            try:
                fetched = pending.result()
            except Exception:  # record the failure, carry on with the next walnut
                fetched = None
                error = traceback.format_exc(limit=3)
            pending = prefetch.submit(download, todo[i + 1]) if i + 1 < len(todo) else None
            if fetched is None:
                summary = {"walnut": n, "error": error}
            else:
                try:
                    summary = {"download_seconds": fetched["seconds"], **evaluate(n, fetched, args)}
                except Exception:
                    summary = {"walnut": n, "error": traceback.format_exc(limit=3)}
            del fetched
            with results.open("a") as f:
                f.write(json.dumps(summary) + "\n")
            shown = ("orbit_heights_mm", "error_in_walnut", "aligned_vs_corrected", "error")
            logging.info(
                "walnut %d: %s", n, json.dumps({k: summary[k] for k in shown if k in summary})
            )


if __name__ == "__main__":
    main()
