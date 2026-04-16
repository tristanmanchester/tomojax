from __future__ import annotations

from tomojax.utils import memory as memory_utils


def test_memory_helpers_cover_batch_estimation_and_backend_defaults(monkeypatch):
    monkeypatch.setenv("TOMOJAX_MAX_VIEWS_PER_BATCH", "3")

    capped = memory_utils.estimate_views_per_batch(
        n_views=10,
        grid_nxyz=(16, 16, 16),
        det_nuv=(8, 8),
        free_bytes_override=10**9,
    )
    fallback = memory_utils.estimate_views_per_batch(
        n_views=10,
        grid_nxyz=(16, 16, 16),
        det_nuv=(8, 8),
        free_bytes_override=0,
    )

    assert 1 <= capped <= 3
    assert fallback == 8
    assert memory_utils._bytes_per("bf16") == 2
    assert memory_utils._bytes_per("fp32") == 4

    monkeypatch.setattr(memory_utils, "_current_backend", lambda: "cpu")
    assert memory_utils.default_gather_dtype() == "fp32"
