from __future__ import annotations

import logging
import sys
import types

from tomojax.utils import logging as logging_utils


def test_logging_helpers_cover_progress_and_duration(monkeypatch):
    backend_logger = logging.getLogger("jax._src.xla_bridge")
    backend_logger.setLevel(logging.INFO)
    monkeypatch.delenv("TOMOJAX_BACKEND_LOG", raising=False)
    logging_utils.setup_logging("warning")
    assert backend_logger.level == logging.WARNING

    monkeypatch.setenv("TOMOJAX_PROGRESS", "0")
    assert list(logging_utils.progress_iter([1, 2, 3], desc="plain")) == [1, 2, 3]

    fake_tqdm = types.ModuleType("tqdm")
    fake_tqdm.tqdm = lambda iterable, **kwargs: iterable
    monkeypatch.setitem(sys.modules, "tqdm", fake_tqdm)
    monkeypatch.setenv("TOMOJAX_PROGRESS", "1")
    assert list(logging_utils.progress_iter([1, 2], total=2, desc="bar")) == [1, 2]

    assert logging_utils.format_duration(None) == "-"
    assert logging_utils.format_duration(0.0005) == "500µs"
    assert logging_utils.format_duration(0.25) == "0.25s"
    assert logging_utils.format_duration(65.0) == "1m05.0s"
