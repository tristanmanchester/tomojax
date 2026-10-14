"""Raw correction reuses device summaries without changing correction order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj
from tomojax.corrections import BeamHardening, RejectViews, Step, Stripes


def _frames() -> tj.Frames:
    counts = np.full((13, 3, 7), 500, np.uint16)
    counts[[3, 8]] = 5
    geometry = tj.ParallelGeometry(
        tj.Grid(7, 7, 3, 1.0, 1.0, 1.0),
        tj.Detector(7, 3, 1.0, 1.0),
        np.arange(13) * 10.0,
    )
    return tj.Frames(counts, geometry, white_level=1000.0)


@pytest.mark.parametrize(
    "steps",
    [
        (RejectViews(),),
        (BeamHardening((1.0, 0.1)), RejectViews()),
        (RejectViews(), BeamHardening((1.0, 0.1)), RejectViews()),
        (Stripes(), RejectViews()),
    ],
)
@pytest.mark.parametrize("batch", [4, 13])
def test_raw_selection_uses_the_statistic_at_its_place_in_the_pipeline(steps, batch) -> None:
    frames = _frames()
    expected = frames.corrected(batch_views=batch).corrected(*steps, batch_views=batch)
    actual = frames.corrected(*steps, batch_views=batch)
    np.testing.assert_array_equal(actual.angles, expected.angles)
    np.testing.assert_allclose(actual.projections, expected.projections, rtol=2e-6, atol=2e-7)
    assert actual.corrections == expected.corrections


def test_first_view_selector_does_not_retransfer_corrected_batches(monkeypatch) -> None:
    frames = _frames()
    transfer_shapes = []
    asarray = jnp.asarray

    def record_transfer(value, *args, **kwargs):
        if isinstance(value, np.ndarray) and value.ndim == 3 and value.dtype == np.float32:
            transfer_shapes.append(value.shape)
        return asarray(value, *args, **kwargs)

    monkeypatch.setattr(jnp, "asarray", record_transfer)
    actual = frames.corrected(RejectViews(), batch_views=4)
    assert actual.corrections[-1].found["rejected"] == [3, 8]
    assert transfer_shapes == []


@dataclass(frozen=True)
class _RangeSelector(Step):
    domain: ClassVar = "line integrals"
    selects_views: ClassVar = True

    def statistic(self, batch):
        return jnp.stack([jnp.min(batch, axis=(1, 2)), jnp.max(batch, axis=(1, 2))], axis=1)

    def select(self, statistics, views):
        keep = statistics[:, 1] - statistics[:, 0] < 0.5
        return keep, {"rejected": views[~keep].tolist()}


def test_device_summary_reuse_supports_custom_multicolumn_statistics() -> None:
    frames = _frames()
    frames.counts[5, 0, 0] = 5
    expected = frames.corrected().corrected(_RangeSelector(), batch_views=4)
    actual = frames.corrected(_RangeSelector(), batch_views=4)
    np.testing.assert_array_equal(actual.angles, expected.angles)
    np.testing.assert_array_equal(actual.projections, expected.projections)
    assert actual.corrections == expected.corrections
    assert actual.corrections[-1].found["rejected"] == [5]


def test_device_summary_preserves_the_callers_float64_context() -> None:
    import jax

    frames = _frames()
    expected = frames.corrected(batch_views=4).projections[:, 0, 0].astype(np.float64) + 2**-30

    @dataclass(frozen=True)
    class WideSummary(Step):
        domain: ClassVar = "line integrals"
        selects_views: ClassVar = True

        def statistic(self, batch):
            return batch[:, 0, 0].astype(jnp.float64) + jnp.float64(2**-30)

        def select(self, statistics, views):
            assert statistics.dtype == np.float64
            np.testing.assert_array_equal(statistics, expected)
            return np.ones(len(views), bool), {}

    with jax.enable_x64():
        actual = frames.corrected(WideSummary(), batch_views=4)
    np.testing.assert_array_equal(actual.angles, frames.geometry.angles)


def test_device_summary_preserves_the_callers_mesh_context() -> None:
    import jax
    from jax.sharding import Mesh, PartitionSpec

    @dataclass(frozen=True)
    class MeshSummary(Step):
        domain: ClassVar = "line integrals"
        selects_views: ClassVar = True

        def statistic(self, batch):
            values = jax.lax.with_sharding_constraint(batch, PartitionSpec("replica", None, None))
            return jnp.mean(values, axis=(1, 2))

        def select(self, statistics, views):
            np.testing.assert_allclose(statistics[[0, 3]], -np.log([0.5, 0.005]), rtol=2e-6)
            return np.ones(len(views), bool), {}

    frames = _frames()
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("replica",))
    with jax.set_mesh(mesh):
        actual = frames.corrected(MeshSummary(), batch_views=4)
    np.testing.assert_array_equal(actual.angles, frames.geometry.angles)


def test_device_summary_honours_disabled_jit_on_the_caller() -> None:
    import jax

    @dataclass(frozen=True)
    class EagerSummary(Step):
        domain: ClassVar = "line integrals"
        selects_views: ClassVar = True

        def statistic(self, batch):
            assert jax.config.jax_disable_jit
            return jnp.mean(batch, axis=(1, 2))

        def select(self, statistics, views):
            assert len(statistics) == len(views)
            return np.ones(len(views), bool), {}

    frames = _frames()
    with jax.disable_jit():
        actual = frames.corrected(EagerSummary(), batch_views=13)
    np.testing.assert_array_equal(actual.angles, frames.geometry.angles)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf])
def test_device_summary_honours_the_callers_runtime_debug_checks(nonfinite: float) -> None:
    import jax

    @dataclass(frozen=True)
    class NonfiniteSummary(Step):
        domain: ClassVar = "line integrals"
        selects_views: ClassVar = True

        def statistic(self, batch):
            return batch[:, 0, 0] * nonfinite

        def select(self, statistics, views):
            raise AssertionError("runtime debug checks must precede view selection")

    context = jax.debug_nans if np.isnan(nonfinite) else jax.debug_infs
    with context(True), pytest.raises(FloatingPointError):
        _frames().corrected(NonfiniteSummary(), batch_views=13)


@pytest.mark.parametrize("budget", ["tight", "zero", "unknown"])
def test_cuda_summaries_keep_serial_dispatch_for_insufficient_allocator_budget(
    monkeypatch, budget: str
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import jax

    if jnp.asarray(0).__dlpack_device__()[0] != 2:
        pytest.skip("requires CUDA allocator metadata")
    (device,) = jnp.asarray(0).devices()
    metadata = (
        None if budget == "unknown"
        else {"bytes_limit": 1, "bytes_in_use": int(budget == "zero"), "free": 2**40}
    )  # fmt: skip
    monkeypatch.setattr(type(device), "memory_stats", lambda _device: metadata)
    if budget != "unknown":
        monkeypatch.setattr(
            jax, "device_get_memory_info", lambda _device: {"free": 2**40}, raising=False
        )
    submit = ThreadPoolExecutor.submit

    def check_submit(executor, function, *args, **kwargs):
        assert not args or not isinstance(args[0], jax.Array), "device summary must stay serial"
        return submit(executor, function, *args, **kwargs)

    monkeypatch.setattr(ThreadPoolExecutor, "submit", check_submit)
    actual = _frames().corrected(RejectViews(), batch_views=4)
    kept = np.delete(np.arange(13), [3, 8])
    np.testing.assert_array_equal(actual.angles, _frames().geometry.angles[kept])
    assert actual.corrections[-1].found["rejected"] == [3, 8]
