"""Host-volume ownership and local huge-page allocation policy."""

import importlib
import mmap
import sys

import numpy as np
import pytest

implementation = importlib.import_module("tomojax.recon._host_arrays")


def test_hugepage_capacity_counts_all_suitable_buddy_orders(monkeypatch):
    monkeypatch.setattr(implementation.sys, "platform", "linux")
    monkeypatch.setattr(implementation.os, "sysconf", lambda _: 4096)

    def read(path):
        if str(path).endswith("hpage_pmd_size"):
            return "2097152\n"
        if str(path).endswith(("enabled", "defrag")):
            return "always [madvise] never\n"
        return (
            "Node 0, zone Normal 100 0 0 0 0 0 0 0 200 3 2\n"
            "Node 1, zone Normal 0 0 0 0 0 0 0 0 0 1 0\n"
        )

    monkeypatch.setattr(implementation.Path, "read_text", read)
    assert implementation._hugepage_capacity() == 8 * 2097152


@pytest.mark.parametrize("inactive", ["enabled", "defrag"])
def test_inactive_direct_hugepage_compaction_keeps_numpy(monkeypatch, inactive):
    monkeypatch.setattr(implementation.sys, "platform", "linux")

    def read(path):
        if str(path).endswith(inactive):
            return "always madvise [never]"
        if str(path).endswith("enabled"):
            return "always [madvise] never"
        pytest.fail("inactive THP queried buddy capacity")

    monkeypatch.setattr(implementation.Path, "read_text", read)
    assert implementation._hugepage_capacity() is None


@pytest.mark.parametrize("failure", [OSError, ValueError])
def test_unavailable_capacity_retains_numpy_allocation(monkeypatch, failure):
    monkeypatch.setattr(implementation.sys, "platform", "linux")

    def unavailable(_):
        raise failure("unavailable kernel metadata")

    monkeypatch.setattr(implementation.Path, "read_text", unavailable)
    assert implementation._hugepage_capacity() is None
    result = implementation._empty_host_output((3, 5, 2))
    assert result.flags.owndata


def test_nonlinux_does_not_read_linux_metadata(monkeypatch):
    monkeypatch.setattr(implementation.sys, "platform", "darwin")

    def unexpected(_):
        pytest.fail("non-Linux output consulted Linux metadata")

    monkeypatch.setattr(implementation.Path, "read_text", unexpected)
    assert implementation._hugepage_capacity() is None


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("pressure", [False, True])
@pytest.mark.skipif(sys.platform != "linux", reason="Linux anonymous-page advice")
def test_default_output_order_dtype_and_owned_lifetime(monkeypatch, order, pressure):
    shape = (7, 5, 3)
    size = np.prod(shape) * 4
    monkeypatch.setattr(implementation, "_hugepage_capacity", lambda: size - int(pressure))
    output = implementation._empty_host_output(shape, order=order)
    assert output.shape == shape and output.dtype == np.float32
    assert output.flags.c_contiguous if order == "C" else output.flags.f_contiguous
    expected = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    output[:] = expected
    np.testing.assert_array_equal(output, expected)
    if pressure:
        storage = output.base
        assert isinstance(storage, mmap.mmap) and not storage.closed
        del output
        storage.close()
        assert storage.closed
    else:
        assert output.flags.owndata


def test_caller_output_never_queries_or_changes_page_policy(monkeypatch):
    def unexpected():
        pytest.fail("caller output consulted the allocation policy")

    monkeypatch.setattr(implementation, "_hugepage_capacity", unexpected)
    projections = np.ones((11, 3, 17), dtype=np.float32)
    output = np.empty((7, 5, 3), dtype=np.float32, order="F")
    assert implementation.validate_host_arrays(projections, output, output.shape) is output
    assert output.flags.f_contiguous and output.flags.owndata


@pytest.mark.skipif(sys.platform != "linux", reason="Linux anonymous-page advice")
def test_fourier_physical_rows_remain_exact_under_page_pressure(monkeypatch):
    from tomojax.geometry import Detector, Grid, ParallelGeometry
    from tomojax.recon import fourier_reconstruct

    grid = Grid(17, 19, 5, 1.3, 1.7, 0.8)
    detector = Detector(35, 7, 0.9, 0.8, (0.7, 0.1))
    angles = np.array([-37, -20, 1, 13, 37, 72, 108, 132])
    projections = np.sin(np.arange(angles.size * 7 * 35)).reshape(angles.size, 7, 35)
    geometry = ParallelGeometry(grid, detector, angles)
    monkeypatch.setattr(implementation, "_hugepage_capacity", lambda: None)
    expected = fourier_reconstruct(geometry, grid, detector, projections)
    monkeypatch.setattr(implementation, "_hugepage_capacity", lambda: 0)
    actual = fourier_reconstruct(geometry, grid, detector, projections)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert actual[:, :, 0].flags.c_contiguous
