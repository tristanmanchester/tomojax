import sys

from tomojax.core.compilation_cache import default_cache_dir


def test_cache_isolates_python_compression_formats(monkeypatch, tmp_path):
    monkeypatch.setenv("TOMOJAX_JAX_CACHE_DIR", str(tmp_path))
    assert default_cache_dir() == tmp_path / sys.implementation.cache_tag
    monkeypatch.delenv("TOMOJAX_JAX_CACHE_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_cache_dir() == tmp_path / "tomojax/jax_cache" / sys.implementation.cache_tag
