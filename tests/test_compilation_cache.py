import sys
from types import SimpleNamespace

import pytest

from tomojax.core.compilation_cache import default_cache_dir


def test_cache_isolates_python_compression_formats(monkeypatch, tmp_path):
    monkeypatch.setenv("TOMOJAX_JAX_CACHE_DIR", str(tmp_path))
    assert default_cache_dir() == tmp_path / sys.implementation.cache_tag
    monkeypatch.delenv("TOMOJAX_JAX_CACHE_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_cache_dir() == tmp_path / "tomojax/jax_cache" / sys.implementation.cache_tag


class _Config:
    """JAX's global configuration, recorded instead of changed for the whole process."""

    def __init__(self, cache_dir=None):
        self.jax_compilation_cache_dir = cache_dir
        self.updates = {}

    def update(self, name, value):
        self.updates[name] = value


def _enable(monkeypatch, config):
    from tomojax.core import compilation_cache

    monkeypatch.setattr(compilation_cache, "jax", SimpleNamespace(config=config))
    compilation_cache.enable_persistent_compilation_cache.cache_clear()
    try:
        compilation_cache.enable_persistent_compilation_cache()
    finally:
        compilation_cache.enable_persistent_compilation_cache.cache_clear()
    return config.updates


def test_cache_is_enabled_for_every_program(monkeypatch, tmp_path):
    monkeypatch.delenv("TOMOJAX_JAX_CACHE", raising=False)
    monkeypatch.setenv("TOMOJAX_JAX_CACHE_DIR", str(tmp_path / "cache"))
    updates = _enable(monkeypatch, _Config())
    expected = tmp_path / "cache" / sys.implementation.cache_tag
    assert expected.is_dir()
    assert updates == {
        "jax_compilation_cache_dir": str(expected),
        "jax_persistent_cache_min_entry_size_bytes": -1,
        "jax_persistent_cache_min_compile_time_secs": 0,
        "jax_persistent_cache_enable_xla_caches": "xla_gpu_per_fusion_autotune_cache_dir",
    }


@pytest.mark.parametrize("value", ["off", "0", "False", " no "])
def test_cache_opt_out_changes_nothing(monkeypatch, tmp_path, value):
    monkeypatch.setenv("TOMOJAX_JAX_CACHE", value)
    monkeypatch.setenv("TOMOJAX_JAX_CACHE_DIR", str(tmp_path / "cache"))
    assert _enable(monkeypatch, _Config()) == {}
    assert not (tmp_path / "cache").exists()


def test_cache_leaves_a_configured_directory_alone(monkeypatch, tmp_path):
    monkeypatch.delenv("TOMOJAX_JAX_CACHE", raising=False)
    monkeypatch.setenv("TOMOJAX_JAX_CACHE_DIR", str(tmp_path / "cache"))
    assert _enable(monkeypatch, _Config(str(tmp_path / "theirs"))) == {}
    assert not (tmp_path / "cache").exists()


def test_cache_that_cannot_be_created_changes_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv("TOMOJAX_JAX_CACHE", raising=False)
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setenv("TOMOJAX_JAX_CACHE_DIR", str(blocker / "cache"))
    assert _enable(monkeypatch, _Config()) == {}
