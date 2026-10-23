"""Independent recurrences for the memory-bounded ASTRA benchmark adapters."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from _astra_cone import Operators, cgls, native, ordered_subsets, volume_geometry
from cone_matrix import image_difference


class DenseOperators:
    cp = np
    shape = (1, 3, 5)

    def __init__(self):
        rng = np.random.default_rng(15)
        self.matrices = [rng.uniform(0, 1, (7, 15)).astype(np.float32) for _ in range(3)]
        for matrix in self.matrices:
            matrix[:, -1] = 0
        self.data = [rng.uniform(0.1, 1, (1, 7, 1)).astype(np.float32) for _ in range(3)]
        self.parts = [(i, None, data.shape) for i, data in enumerate(self.data)]

    def measured(self, part):
        return self.data[part[0]].copy()

    def upload(self, value):
        return value.copy()

    def download(self, value, *, out=None):
        if out is None:
            return value.copy()
        out[:] = value
        return out

    def forward(self, value, part):
        return (self.matrices[part[0]] @ value.ravel()).reshape(part[2])

    def back(self, value, part, output):
        output[:] = (self.matrices[part[0]].T @ value.ravel()).reshape(self.shape)


@pytest.mark.parametrize("method", ["ossart", "osem"])
@pytest.mark.parametrize("policy", ["resident", "host", "recompute"])
def test_subset_normalization_storage_preserves_dense_updates(method, policy):
    ops = DenseOperators()
    expected = np.full(15, 0.01, np.float64)
    for _ in range(2):
        for raw_matrix, data in zip(ops.matrices, ops.data, strict=True):
            matrix = raw_matrix.astype(np.float64)
            predicted, norm = matrix @ expected, matrix.sum(axis=0)
            residual = data.ravel() - predicted
            if method == "ossart":
                update = matrix.T @ (residual / matrix.sum(axis=1))
                expected = np.maximum(expected + update / np.where(norm > 0, norm, 1), 0)
            else:
                update = matrix.T @ (data.ravel() / predicted)
                expected *= np.divide(update, norm, out=np.ones_like(norm), where=norm > 0)
    if method == "osem":
        expected[-1] = 0
    initial = np.full(ops.shape, 0.01, np.float32)
    actual = ordered_subsets(ops, initial, method=method, iterations=2, norms=policy)
    np.testing.assert_allclose(actual.ravel(), expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("resident", [False, True])
def test_bounded_cgls_matches_dense_least_squares(resident):
    ops = DenseOperators()
    matrix = np.concatenate(ops.matrices).astype(np.float64)
    data = np.concatenate([d.ravel() for d in ops.data]).astype(np.float64)
    expected = np.linalg.lstsq(matrix, data, rcond=None)[0]
    actual = cgls(ops, None, iterations=20, resident_residual=resident).ravel()
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-6)


def test_full_image_difference_scores_noncontiguous_planes():
    truth = np.arange(60, dtype=np.float32).reshape(3, 4, 5).transpose(2, 0, 1) + 1
    result = image_difference(truth * 1.25, truth)
    assert result["relative_l2"] == pytest.approx(0.25)
    assert result["max_abs"] == 15


@pytest.mark.gpu
@pytest.mark.parametrize("iterations", [1, 4])
@pytest.mark.parametrize("resident", [False, True])
def test_bounded_cgls_matches_native_astra(iterations, resident):
    astra = pytest.importorskip("astra")
    cp = pytest.importorskip("cupy")
    import tomojax as tj

    rng = np.random.default_rng(531)
    pg = astra.geom_2vec(
        astra.create_proj_geom(
            "cone", 0.8, 1.1, 21, 27, np.linspace(0, 2 * np.pi, 23, endpoint=False), 35, 20
        )
    )
    vectors = pg["Vectors"]
    vg = volume_geometry(tj.Grid(17, 15, 13, 0.8, 0.9, 0.9, vol_center=(0.2, 0.7, -0.2)))
    data = rng.uniform(0.01, 1, (23, 21, 27)).astype(np.float32)
    initial = rng.uniform(0.001, 0.02, (13, 15, 17)).astype(np.float32)
    saved_initial = initial.copy()
    expected = native(data, vectors, vg, method="cgls", iterations=iterations, initial=initial)
    np.testing.assert_array_equal(initial, saved_initial)
    pool_bytes = cp.get_default_memory_pool().total_bytes()
    for batch in (7, 23):
        with Operators(
            data, vectors, vg, [slice(i, i + batch) for i in range(0, 23, batch)]
        ) as ops:
            actual = cgls(ops, initial, iterations=iterations, resident_residual=resident)
            assert ops.work["forward_views"] == 23 * (iterations + 1)
            assert ops.work["backprojected_views"] == 23 * (iterations + 1)
            uploads = 1 + iterations * (1 if resident else 2)
            downloads = iterations if resident else 2 * iterations + 1
            assert ops.work["host_to_device_bytes"] == initial.nbytes + uploads * data.nbytes
            assert ops.work["device_to_host_bytes"] == initial.nbytes + downloads * data.nbytes
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-6)
    assert cp.get_default_memory_pool().total_bytes() == pool_bytes
