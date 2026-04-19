from __future__ import annotations

import pytest

import tomojax.cli.align as align_cli
from tomojax.cli.config import parse_args_with_config
import tomojax.cli.recon as recon_cli


def test_recon_config_supplies_required_args_and_typed_values(tmp_path):
    config_path = tmp_path / "recon.toml"
    config_path.write_text(
        "\n".join(
            [
                'data = "input.nxs"',
                'out = "runs/recon.nxs"',
                'algo = "spdhg"',
                "lambda_tv = 0.01",
                'views_per_batch = "auto"',
                "grid = [8, 9, 10]",
                "checkpoint_projector = false",
                "progress = true",
            ]
        ),
        encoding="utf-8",
    )

    parser = recon_cli._build_parser()
    args, metadata = parse_args_with_config(
        parser,
        ["--config", str(config_path)],
        required=("data", "out"),
    )

    assert args.config == str(config_path)
    assert args.data == "input.nxs"
    assert args.out == "runs/recon.nxs"
    assert args.algo == "spdhg"
    assert args.lambda_tv == pytest.approx(0.01)
    assert args.views_per_batch == "auto"
    assert args.grid == [8, 9, 10]
    assert args.checkpoint_projector is False
    assert args.progress is True
    assert metadata["config_path"] == str(config_path)
    assert metadata["config_file_values"]["algo"] == "spdhg"
    assert metadata["effective_options"]["out"] == "runs/recon.nxs"


def test_align_cli_overrides_config_scalars_lists_booleans_and_append_values(tmp_path):
    config_path = tmp_path / "align.toml"
    config_path.write_text(
        "\n".join(
            [
                'data = "input.nxs"',
                'out = "runs/align.nxs"',
                "lambda_tv = 0.02",
                "levels = [4, 2, 1]",
                "checkpoint_projector = false",
                'loss_param = ["delta=1.0"]',
            ]
        ),
        encoding="utf-8",
    )

    parser = align_cli._build_parser()
    args, metadata = parse_args_with_config(
        parser,
        [
            "--config",
            str(config_path),
            "--lambda-tv",
            "0.03",
            "--levels",
            "2",
            "1",
            "--checkpoint-projector",
            "--loss-param",
            "eps=0.001",
        ],
        required=("data", "out"),
    )

    assert args.data == "input.nxs"
    assert args.out == "runs/align.nxs"
    assert args.lambda_tv == pytest.approx(0.03)
    assert args.levels == [2, 1]
    assert args.checkpoint_projector is True
    assert args.loss_param == ["eps=0.001"]
    assert metadata["config_file_values"]["levels"] == [4, 2, 1]
    assert "lambda_tv" in metadata["explicit_cli_keys"]
    assert "levels" in metadata["explicit_cli_keys"]
    assert "checkpoint_projector" in metadata["explicit_cli_keys"]
    assert "loss_param" in metadata["explicit_cli_keys"]
    assert metadata["effective_options"]["levels"] == [2, 1]


def test_config_unknown_key_reports_useful_error(tmp_path, capsys):
    config_path = tmp_path / "bad.toml"
    config_path.write_text(
        "\n".join(
            [
                'data = "input.nxs"',
                'out = "runs/recon.nxs"',
                "not_a_real_option = true",
            ]
        ),
        encoding="utf-8",
    )

    parser = recon_cli._build_parser()
    with pytest.raises(SystemExit):
        parse_args_with_config(parser, ["--config", str(config_path)], required=("data", "out"))

    captured = capsys.readouterr()
    assert "unknown config key" in captured.err
    assert "not_a_real_option" in captured.err
    assert "lambda_tv" in captured.err


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ('data = "input.nxs"\nout = "runs/recon.nxs"\nalgo = "not-real"\n', "invalid choice"),
        ('data = "input.nxs"\nout = "runs/recon.nxs"\ngrid = [8, 9]\n', "expected 3 values"),
    ],
)
def test_config_invalid_values_report_parser_errors(tmp_path, capsys, body, expected):
    config_path = tmp_path / "bad_value.toml"
    config_path.write_text(body, encoding="utf-8")

    parser = recon_cli._build_parser()
    with pytest.raises(SystemExit):
        parse_args_with_config(parser, ["--config", str(config_path)], required=("data", "out"))

    captured = capsys.readouterr()
    assert expected in captured.err


def test_yaml_config_is_rejected_without_runtime_dependency(tmp_path, capsys):
    config_path = tmp_path / "recon.yaml"
    config_path.write_text('data: "input.nxs"\nout: "runs/recon.nxs"\n', encoding="utf-8")

    parser = recon_cli._build_parser()
    with pytest.raises(SystemExit):
        parse_args_with_config(parser, ["--config", str(config_path)], required=("data", "out"))

    captured = capsys.readouterr()
    assert "YAML config files are not supported" in captured.err
    assert "TOML" in captured.err


def test_missing_required_args_still_fails_without_cli_or_config(capsys):
    parser = recon_cli._build_parser()
    with pytest.raises(SystemExit):
        parse_args_with_config(parser, [], required=("data", "out"))

    captured = capsys.readouterr()
    assert "the following arguments are required: --data, --out" in captured.err
