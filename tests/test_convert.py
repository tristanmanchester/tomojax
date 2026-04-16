from __future__ import annotations

from tomojax.cli import convert as convert_cli


def test_convert_main_parses_paths_and_calls_convert(monkeypatch):
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        convert_cli,
        "convert",
        lambda in_path, out_path: captured.update({"in_path": in_path, "out_path": out_path}),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["convert", "--in", "input.npz", "--out", "output.nxs"],
    )

    convert_cli.main()

    assert captured == {"in_path": "input.npz", "out_path": "output.nxs"}
