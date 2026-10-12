"""Lazy acquisitions stay separate from caller-owned mapped correction output."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import tifffile

import tomojax as tj


def _tiff(path: Path, data: np.ndarray) -> np.memmap:
    tifffile.imwrite(path, data, photometric="minisblack", contiguous=True)
    return tifffile.memmap(path, mode="r+")


def _hdf(path: Path, data: np.ndarray) -> np.memmap:
    with h5py.File(path, "w") as file:
        dataset = file.create_dataset("exchange/data", data=data)
        file.create_dataset("exchange/theta", data=np.arange(len(data)) * 30.0)
        offset = dataset.id.get_offset()
    assert offset is not None
    return np.memmap(path, mode="r+", dtype=data.dtype, shape=data.shape, offset=offset)


@pytest.mark.parametrize("file_format", ["tiff", "hdf"])
@pytest.mark.parametrize("role", ["counts", "flats", "darks"])
@pytest.mark.parametrize("link", ["original", "symlink", "hardlink"])
@pytest.mark.parametrize("buffer_view", [False, True])
def test_lazy_input_file_alias_is_rejected_before_reading_or_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_format: str,
    role: str,
    link: str,
    buffer_view: bool,
) -> None:
    data = np.arange(4 * 5 * 7, dtype=np.float32).reshape(4, 5, 7) + 1000
    path = tmp_path / ("raw.tif" if file_format == "tiff" else "raw.h5")
    out = (_tiff if file_format == "tiff" else _hdf)(path, data)
    input_path = path
    if link != "original":
        input_path = tmp_path / ("alias" + path.suffix)
        if link == "symlink":
            input_path.symlink_to(path)
        else:
            input_path.hardlink_to(path)
    frames = tj.load_frames(input_path, angles=np.arange(4) * 30.0, white_level=2000)
    source = frames.counts
    if role != "counts":
        frames = replace(frames, counts=data.copy(), **{role: source})
    # Selections/crops keep file identity even when the mapped output is a view.
    frames = frames.selected([0, 1, 2]).cropped(slice(1, 4), slice(2, 6))
    if buffer_view:
        out = np.frombuffer(memoryview(out), dtype=out.dtype).reshape(out.shape)
    target = out[:3, 1:4, 2:6]
    before = path.read_bytes()

    def read(*args: object) -> None:
        raise AssertionError("file aliases must fail before any acquisition pixels are read")

    monkeypatch.setattr(type(source), "__getitem__", read)
    with pytest.raises(ValueError, match="separate .*files"):
        frames.corrected(out=target)
    assert path.read_bytes() == before


@pytest.mark.parametrize("file_format", ["tiff", "hdf"])
def test_separate_mapped_output_preserves_lazy_input_and_exact_results(
    tmp_path: Path, file_format: str
) -> None:
    data = np.arange(4 * 5 * 7, dtype=np.float32).reshape(4, 5, 7) + 1000
    path = tmp_path / ("raw.tif" if file_format == "tiff" else "raw.h5")
    source_map = (_tiff if file_format == "tiff" else _hdf)(path, data)
    frames = tj.load_frames(path, angles=np.arange(4) * 30.0, white_level=2000)
    expected = frames.corrected(batch_views=2)
    before = path.read_bytes()
    out = np.memmap(tmp_path / "corrected.bin", mode="w+", dtype=np.float32, shape=data.shape)
    actual = frames.corrected(batch_views=2, out=out)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    assert actual.projections is out
    assert actual.corrections == expected.corrections
    np.testing.assert_array_equal(source_map, data)
    assert path.read_bytes() == before


@pytest.mark.parametrize("layout", ["linked", "virtual", "nested_virtual", "external_raw"])
def test_indirect_hdf_storage_cannot_be_used_as_correction_output(
    tmp_path: Path, layout: str
) -> None:
    data = np.arange(4 * 5 * 7, dtype=np.float32).reshape(4, 5, 7) + 1000
    original = tmp_path / "source.h5"
    if layout == "external_raw":
        raw = tmp_path / "pixels.raw"
        out = np.memmap(raw, mode="w+", shape=data.shape, dtype=data.dtype)
        out[:] = data
        out.flush()
        with h5py.File(original, "w") as file:
            file.create_dataset(
                "exchange/data",
                shape=data.shape,
                dtype=data.dtype,
                external=[(str(raw), 0, h5py.h5f.UNLIMITED)],
            )
            file.create_dataset("exchange/theta", data=np.arange(4) * 30.0)
        path, mapped = original, raw
    else:
        out = _hdf(original, data)
        path, mapped = tmp_path / "wrapper.h5", original
        virtual_source = original
        if layout == "nested_virtual":
            virtual_source = tmp_path / "intermediate.h5"
            with h5py.File(virtual_source, "w") as file:
                virtual = h5py.VirtualLayout(shape=data.shape, dtype=data.dtype)
                virtual[:] = h5py.VirtualSource(original.name, "exchange/data", shape=data.shape)
                file.create_virtual_dataset("exchange/data", virtual)
        with h5py.File(path, "w") as file:
            if layout == "linked":
                file["exchange/data"] = h5py.ExternalLink(original.name, "/exchange/data")
            else:
                virtual = h5py.VirtualLayout(shape=data.shape, dtype=data.dtype)
                virtual[:] = h5py.VirtualSource(
                    virtual_source.name, "exchange/data", shape=data.shape
                )
                file.create_virtual_dataset("exchange/data", virtual)
            file.create_dataset("exchange/theta", data=np.arange(4) * 30.0)
    frames = tj.load_frames(path, white_level=2000)
    np.testing.assert_array_equal(np.asarray(frames.counts), data)
    before = mapped.read_bytes()
    with pytest.raises(ValueError, match="separate .*files"):
        frames.corrected(out=out)
    assert mapped.read_bytes() == before


@pytest.mark.parametrize("missing_source", [False, True])
def test_virtual_sources_allow_independent_output_including_native_fill(
    tmp_path: Path, missing_source: bool
) -> None:
    data = np.arange(4 * 5 * 7, dtype=np.float32).reshape(4, 5, 7) + 1000
    original = tmp_path / "source.h5"
    if not missing_source:
        _hdf(original, data)
    with h5py.File(tmp_path / "views.h5", "w") as file:
        virtual = h5py.VirtualLayout(shape=data.shape, dtype=data.dtype)
        virtual[:] = h5py.VirtualSource(original.name, "exchange/data", shape=data.shape)
        file.create_virtual_dataset("exchange/data", virtual, fillvalue=1200)
        file.create_dataset("exchange/theta", data=np.arange(4) * 30.0)
    frames = tj.load_frames(tmp_path / "views.h5", white_level=2000)
    expected = frames.corrected(batch_views=2)
    out = np.memmap(tmp_path / "corrected.bin", mode="w+", dtype=np.float32, shape=data.shape)
    actual = frames.corrected(out=out, batch_views=2)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    assert actual.corrections == expected.corrections


def test_printf_virtual_files_remain_physical_inputs(tmp_path: Path) -> None:
    data = np.arange(4 * 5 * 7, dtype=np.float32).reshape(4, 5, 7) + 1000
    maps = [_hdf(tmp_path / f"counts{i}%.h5", data + i) for i in range(4)]
    with h5py.File(tmp_path / "views.h5", "w") as file:
        virtual = h5py.VirtualLayout(shape=data.shape, maxshape=(None, 5, 7), dtype=data.dtype)
        source = h5py.VirtualSource(
            str(tmp_path / "counts%b%%.h5"), "exchange/data", shape=(1, 5, 7)
        )
        virtual[: h5py.h5s.UNLIMITED] = source
        file.create_virtual_dataset("exchange/data", virtual)
        file.create_dataset("exchange/theta", data=np.arange(4) * 30.0)
    frames = tj.load_frames(tmp_path / "views.h5", white_level=2000)
    np.testing.assert_array_equal(
        np.asarray(frames.counts), np.stack([data[0] + i for i in range(4)])
    )
    before = (tmp_path / "counts2%.h5").read_bytes()
    with pytest.raises(ValueError, match="separate .*files"):
        frames.corrected(out=maps[2])
    assert (tmp_path / "counts2%.h5").read_bytes() == before
    out = np.memmap(tmp_path / "corrected.bin", mode="w+", dtype=np.float32, shape=data.shape)
    np.testing.assert_array_equal(
        frames.corrected(out=out).projections, frames.corrected().projections
    )


@pytest.mark.parametrize("layout", ["virtual_prefix", "virtual_absolute_fallback", "raw_cwd"])
def test_hdf_source_search_rules_do_not_hide_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    data = np.arange(4 * 5 * 7, dtype=np.float32).reshape(4, 5, 7) + 1000
    folder = tmp_path / "calibration"
    folder.mkdir()
    path = tmp_path / "views.h5"
    if layout == "raw_cwd":
        raw = folder / "pixels.raw"
        out = np.memmap(raw, mode="w+", dtype=data.dtype, shape=data.shape)
        out[:] = data
        out.flush()
        monkeypatch.chdir(folder)
        with h5py.File(path, "w") as file:
            file.create_dataset(
                "exchange/data",
                shape=data.shape,
                dtype=data.dtype,
                external=[(raw.name, 0, h5py.h5f.UNLIMITED)],
            )
            file.create_dataset("exchange/theta", data=np.arange(4) * 30.0)
    else:
        original = (folder if layout == "virtual_prefix" else tmp_path) / "source.h5"
        out = _hdf(original, data)
        name = (
            original.name if layout == "virtual_prefix" else str(tmp_path / "old" / original.name)
        )
        if layout == "virtual_prefix":
            monkeypatch.setenv("HDF5_VDS_PREFIX", str(tmp_path / "missing") + ":" + str(folder))
        with h5py.File(path, "w") as file:
            virtual = h5py.VirtualLayout(shape=data.shape, dtype=data.dtype)
            virtual[:] = h5py.VirtualSource(name, "exchange/data", shape=data.shape)
            file.create_virtual_dataset("exchange/data", virtual)
            file.create_dataset("exchange/theta", data=np.arange(4) * 30.0)
    frames = tj.load_frames(path, white_level=2000)
    np.testing.assert_array_equal(np.asarray(frames.counts), data)
    before = out.copy()
    with pytest.raises(ValueError, match="separate .*files"):
        frames.corrected(out=out)
    np.testing.assert_array_equal(out, before)


@pytest.mark.parametrize("absolute_pattern", [False, True])
def test_printf_fallback_blocks_remain_inputs_but_shadowed_files_do_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, absolute_pattern: bool
) -> None:
    data = np.arange(4 * 5 * 7, dtype=np.float32).reshape(4, 5, 7) + 1000
    priority = tmp_path / "priority%b[raw]"
    priority.mkdir()
    maps = []
    for index in range(4):
        folder = priority if index < 2 else tmp_path
        maps.append(_hdf(folder / f"counts{index}.h5", data + index))
    shadowed = _hdf(tmp_path / "counts0.h5", data + 100)
    if absolute_pattern:
        # This existing absolute directory supplies only the first two blocks.
        pattern = str(priority / "counts%b.h5").replace("priority%b", "priority%%b")
    else:
        monkeypatch.setenv("HDF5_VDS_PREFIX", str(priority))
        pattern = "counts%b.h5"
    with h5py.File(tmp_path / "views.h5", "w") as file:
        virtual = h5py.VirtualLayout(shape=data.shape, maxshape=(None, 5, 7), dtype=data.dtype)
        virtual[: h5py.h5s.UNLIMITED] = h5py.VirtualSource(
            pattern, "exchange/data", shape=(1, 5, 7)
        )
        file.create_virtual_dataset("exchange/data", virtual)
        file.create_dataset("exchange/theta", data=np.arange(4) * 30.0)
    frames = tj.load_frames(tmp_path / "views.h5", white_level=2000)
    expected_counts = np.stack([data[0] + i for i in range(4)])
    np.testing.assert_array_equal(np.asarray(frames.counts), expected_counts)
    original = maps[2].copy()
    with pytest.raises(ValueError, match="separate .*files"):
        frames.corrected(out=maps[2])
    np.testing.assert_array_equal(maps[2], original)
    expected = frames.corrected()
    # The lower-priority duplicate is independent output, not an acquired input.
    actual = frames.corrected(out=shadowed)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    np.testing.assert_array_equal(np.asarray(frames.counts), expected_counts)


@pytest.mark.parametrize("repeated", [False, True])
def test_non_native_printf_spellings_are_independent_files(tmp_path: Path, repeated: bool) -> None:
    data = np.arange(4 * 5 * 7, dtype=np.float32).reshape(4, 5, 7) + 1000
    for index in range(4):
        name = f"counts{index}-{index}.h5" if repeated else f"counts{index}.h5"
        _hdf(tmp_path / name, data + index)
    name = "counts0-1.h5" if repeated else "counts00.h5"
    independent = _hdf(tmp_path / name, data + 100)
    pattern = "counts%b-%b.h5" if repeated else "counts%b.h5"
    with h5py.File(tmp_path / "views.h5", "w") as file:
        virtual = h5py.VirtualLayout(shape=data.shape, maxshape=(None, 5, 7), dtype=data.dtype)
        virtual[: h5py.h5s.UNLIMITED] = h5py.VirtualSource(
            pattern, "exchange/data", shape=(1, 5, 7)
        )
        file.create_virtual_dataset("exchange/data", virtual)
        file.create_dataset("exchange/theta", data=np.arange(4) * 30.0)
    frames = tj.load_frames(tmp_path / "views.h5", white_level=2000)
    expected_counts = np.stack([data[0] + i for i in range(4)])
    np.testing.assert_array_equal(np.asarray(frames.counts), expected_counts)
    expected = frames.corrected()
    np.testing.assert_array_equal(
        frames.corrected(out=independent).projections, expected.projections
    )
    np.testing.assert_array_equal(np.asarray(frames.counts), expected_counts)
