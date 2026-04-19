from __future__ import annotations

import pytest

import tomojax.cli.align as align_cli
from tomojax.align.dofs import normalize_bounds
from tomojax.align.losses import loss_spec_name, resolve_loss_for_level
from tomojax.align.pipeline import AlignConfig
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


def test_recon_config_accepts_fista_constraints_and_cli_can_disable_positivity(tmp_path):
    config_path = tmp_path / "recon.toml"
    config_path.write_text(
        "\n".join(
            [
                'data = "input.nxs"',
                'out = "runs/recon.nxs"',
                'algo = "fista"',
                "positivity = true",
                "lower_bound = 0.0",
                "upper_bound = 1.0",
            ]
        ),
        encoding="utf-8",
    )

    parser = recon_cli._build_parser()
    args, metadata = parse_args_with_config(
        parser,
        ["--config", str(config_path), "--no-positivity"],
        required=("data", "out"),
    )

    assert args.algo == "fista"
    assert args.positivity is False
    assert args.lower_bound == pytest.approx(0.0)
    assert args.upper_bound == pytest.approx(1.0)
    assert metadata["config_file_values"]["positivity"] is True
    assert metadata["config_file_values"]["lower_bound"] == pytest.approx(0.0)
    assert metadata["config_file_values"]["upper_bound"] == pytest.approx(1.0)
    assert "positivity" in metadata["explicit_cli_keys"]
    assert metadata["effective_options"]["positivity"] is False


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
                'checkpoint = "runs/align.ckpt.npz"',
                "checkpoint_every = 2",
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
            "--checkpoint-every",
            "3",
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
    assert args.checkpoint == "runs/align.ckpt.npz"
    assert args.checkpoint_every == 3
    assert args.loss_param == ["eps=0.001"]
    assert metadata["config_file_values"]["levels"] == [4, 2, 1]
    assert metadata["config_file_values"]["checkpoint"] == "runs/align.ckpt.npz"
    assert metadata["config_file_values"]["checkpoint_every"] == 2
    assert "lambda_tv" in metadata["explicit_cli_keys"]
    assert "levels" in metadata["explicit_cli_keys"]
    assert "checkpoint_projector" in metadata["explicit_cli_keys"]
    assert "checkpoint_every" in metadata["explicit_cli_keys"]
    assert "loss_param" in metadata["explicit_cli_keys"]
    assert metadata["effective_options"]["levels"] == [2, 1]


def test_align_config_toml_accepts_loss_schedule_string(tmp_path):
    config_path = tmp_path / "align.toml"
    config_path.write_text(
        "\n".join(
            [
                'data = "input.nxs"',
                'out = "runs/align.nxs"',
                'loss = "l2"',
                'loss_schedule = "4:phasecorr,2:ssim,1:l2_otsu"',
            ]
        ),
        encoding="utf-8",
    )

    parser = align_cli._build_parser()
    args, metadata = parse_args_with_config(
        parser,
        ["--config", str(config_path)],
        required=("data", "out"),
    )
    loss_config, loss_params = align_cli._parse_loss_config(args, parser)

    assert loss_params == {}
    assert metadata["config_file_values"]["loss_schedule"] == "4:phasecorr,2:ssim,1:l2_otsu"
    assert loss_spec_name(resolve_loss_for_level(loss_config, 4)) == "phasecorr"
    assert loss_spec_name(resolve_loss_for_level(loss_config, 2)) == "ssim"
    assert loss_spec_name(resolve_loss_for_level(loss_config, 1)) == "l2_otsu"
    assert loss_spec_name(resolve_loss_for_level(loss_config, 8)) == "l2"


def test_align_config_toml_accepts_loss_schedule_mapping(tmp_path):
    config_path = tmp_path / "align.toml"
    config_path.write_text(
        "\n".join(
            [
                'data = "input.nxs"',
                'out = "runs/align.nxs"',
                'loss_schedule = { "4" = "phasecorr", "2" = "ssim" }',
            ]
        ),
        encoding="utf-8",
    )

    parser = align_cli._build_parser()
    args, _ = parse_args_with_config(
        parser,
        ["--config", str(config_path)],
        required=("data", "out"),
    )
    loss_config, _ = align_cli._parse_loss_config(args, parser)

    assert loss_spec_name(resolve_loss_for_level(loss_config, 4)) == "phasecorr"
    assert loss_spec_name(resolve_loss_for_level(loss_config, 2)) == "ssim"
    assert loss_spec_name(resolve_loss_for_level(loss_config, 1)) == "l2_otsu"


def test_align_cli_loss_schedule_overrides_config(tmp_path):
    config_path = tmp_path / "align.toml"
    config_path.write_text(
        "\n".join(
            [
                'data = "input.nxs"',
                'out = "runs/align.nxs"',
                'loss = "l2"',
                'loss_schedule = "4:phasecorr"',
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
            "--loss-schedule",
            "2:ssim",
        ],
        required=("data", "out"),
    )
    loss_config, _ = align_cli._parse_loss_config(args, parser)

    assert "loss_schedule" in metadata["explicit_cli_keys"]
    assert metadata["effective_options"]["loss_schedule"] == "2:ssim"
    assert loss_spec_name(resolve_loss_for_level(loss_config, 2)) == "ssim"
    assert loss_spec_name(resolve_loss_for_level(loss_config, 4)) == "l2"


def test_align_cli_rejects_invalid_loss_schedule_before_main_work(capsys):
    parser = align_cli._build_parser()
    args, _ = parse_args_with_config(
        parser,
        [
            "--data",
            "input.nxs",
            "--out",
            "runs/align.nxs",
            "--loss-schedule",
            "4phasecorr",
        ],
        required=("data", "out"),
    )

    with pytest.raises(SystemExit):
        align_cli._parse_loss_config(args, parser)

    captured = capsys.readouterr()
    assert "LEVEL:LOSS" in captured.err


def test_align_cli_dof_options_parse_and_normalize_named_dofs():
    parser = align_cli._build_parser()
    args, _ = parse_args_with_config(
        parser,
        [
            "--data",
            "input.nxs",
            "--out",
            "runs/align.nxs",
            "--optimise-dofs",
            "dx,dz",
            "--freeze-dofs",
            "phi",
        ],
        required=("data", "out"),
    )

    optimise_dofs, freeze_dofs = align_cli._parse_dof_args(args, parser)

    assert optimise_dofs == ("dx", "dz")
    assert freeze_dofs == ("phi",)


def test_align_config_toml_accepts_dof_arrays(tmp_path):
    config_path = tmp_path / "align.toml"
    config_path.write_text(
        "\n".join(
            [
                'data = "input.nxs"',
                'out = "runs/align.nxs"',
                'optimise_dofs = ["dx", "dz"]',
                'freeze_dofs = ["phi"]',
            ]
        ),
        encoding="utf-8",
    )

    parser = align_cli._build_parser()
    args, metadata = parse_args_with_config(
        parser,
        ["--config", str(config_path)],
        required=("data", "out"),
    )
    optimise_dofs, freeze_dofs = align_cli._parse_dof_args(args, parser)

    assert metadata["config_file_values"]["optimise_dofs"] == ["dx", "dz"]
    assert metadata["config_file_values"]["freeze_dofs"] == ["phi"]
    assert optimise_dofs == ("dx", "dz")
    assert freeze_dofs == ("phi",)


def test_align_cli_rejects_unknown_dof_name(capsys):
    parser = align_cli._build_parser()
    args, _ = parse_args_with_config(
        parser,
        [
            "--data",
            "input.nxs",
            "--out",
            "runs/align.nxs",
            "--optimise-dofs",
            "theta",
        ],
        required=("data", "out"),
    )

    with pytest.raises(SystemExit):
        align_cli._parse_dof_args(args, parser)

    captured = capsys.readouterr()
    assert "Unknown alignment DOF" in captured.err
    assert "theta" in captured.err
    assert "alpha" in captured.err


def test_normalize_bounds_accepts_cli_string_in_canonical_order():
    bounds = normalize_bounds(
        "dx=-20:20,dz=-10:10,alpha=-0.05:0.05",
        option_name="--bounds",
    )

    assert bounds == (
        ("alpha", -0.05, 0.05),
        ("dx", -20.0, 20.0),
        ("dz", -10.0, 10.0),
    )


def test_normalize_bounds_accepts_toml_style_mapping():
    bounds = normalize_bounds(
        {"dz": [-10, 10], "alpha": [-0.05, 0.05]},
        option_name="bounds",
    )

    assert bounds == (
        ("alpha", -0.05, 0.05),
        ("dz", -10.0, 10.0),
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("theta=-1:1", "Unknown alignment DOF"),
        ("dx=-1:1,dx=-2:2", "Duplicate alignment bounds"),
        ("dx=1:1", "lower bound 1 must be less than upper bound 1"),
        ("dx=-inf:1", "must be finite"),
        ("dx=left:1", "is not numeric"),
    ],
)
def test_normalize_bounds_rejects_invalid_values(raw, expected):
    with pytest.raises(ValueError, match=expected):
        normalize_bounds(raw, option_name="--bounds")


def test_align_cli_bounds_option_parses_named_bounds():
    parser = align_cli._build_parser()
    args, metadata = parse_args_with_config(
        parser,
        [
            "--data",
            "input.nxs",
            "--out",
            "runs/align.nxs",
            "--bounds",
            "dx=-20:20,dz=-20:20,alpha=-0.05:0.05",
        ],
        required=("data", "out"),
    )

    assert args.bounds == (
        ("alpha", -0.05, 0.05),
        ("dx", -20.0, 20.0),
        ("dz", -20.0, 20.0),
    )
    assert "bounds" in metadata["explicit_cli_keys"]


def test_align_config_toml_accepts_bounds_mapping(tmp_path):
    config_path = tmp_path / "align.toml"
    config_path.write_text(
        "\n".join(
            [
                'data = "input.nxs"',
                'out = "runs/align.nxs"',
                "bounds = { dx = [-20, 20], alpha = [-0.05, 0.05] }",
            ]
        ),
        encoding="utf-8",
    )

    parser = align_cli._build_parser()
    args, metadata = parse_args_with_config(
        parser,
        ["--config", str(config_path)],
        required=("data", "out"),
    )

    assert args.bounds == (
        ("alpha", -0.05, 0.05),
        ("dx", -20.0, 20.0),
    )
    assert metadata["config_file_values"]["bounds"] == (
        ("alpha", -0.05, 0.05),
        ("dx", -20.0, 20.0),
    )


def test_align_config_toml_accepts_and_cli_overrides_pose_model_options(tmp_path):
    config_path = tmp_path / "align.toml"
    config_path.write_text(
        "\n".join(
            [
                'data = "input.nxs"',
                'out = "runs/align.nxs"',
                'pose_model = "spline"',
                "knot_spacing = 6",
                "degree = 3",
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
            "--pose-model",
            "polynomial",
            "--degree",
            "2",
        ],
        required=("data", "out"),
    )

    assert args.pose_model == "polynomial"
    assert args.knot_spacing == 6
    assert args.degree == 2
    assert metadata["config_file_values"]["pose_model"] == "spline"
    assert metadata["config_file_values"]["knot_spacing"] == 6
    assert metadata["config_file_values"]["degree"] == 3
    assert "pose_model" in metadata["explicit_cli_keys"]
    assert "degree" in metadata["explicit_cli_keys"]


def test_align_config_toml_accepts_spdhg_recon_options(tmp_path):
    config_path = tmp_path / "align.toml"
    config_path.write_text(
        "\n".join(
            [
                'data = "input.nxs"',
                'out = "runs/align.nxs"',
                'recon_algo = "spdhg"',
                "views_per_batch = 2",
                "spdhg_seed = 7",
                "recon_positivity = false",
            ]
        ),
        encoding="utf-8",
    )

    parser = align_cli._build_parser()
    args, metadata = parse_args_with_config(
        parser,
        ["--config", str(config_path)],
        required=("data", "out"),
    )

    assert args.recon_algo == "spdhg"
    assert args.views_per_batch == 2
    assert args.spdhg_seed == 7
    assert args.recon_positivity is False
    assert metadata["config_file_values"]["recon_algo"] == "spdhg"
    assert metadata["config_file_values"]["views_per_batch"] == 2
    assert metadata["config_file_values"]["spdhg_seed"] == 7
    assert metadata["config_file_values"]["recon_positivity"] is False


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"recon_algo": "not-a-solver"}, "recon_algo must be one of"),
        ({"pose_model": "not-a-model"}, "pose_model must be one of"),
        ({"pose_model": "polynomial", "degree": -1}, "degree must be >= 0"),
        ({"pose_model": "spline", "knot_spacing": 0}, "knot_spacing must be >= 1"),
        ({"pose_model": "spline", "degree": 4}, "degree must be one of"),
    ],
)
def test_align_config_rejects_invalid_pose_model_options(kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        AlignConfig(**kwargs)


def test_align_cli_rejects_invalid_bounds_before_main_work(capsys):
    parser = align_cli._build_parser()

    with pytest.raises(SystemExit):
        parse_args_with_config(
            parser,
            [
                "--data",
                "input.nxs",
                "--out",
                "runs/align.nxs",
                "--bounds",
                "dx=20:-20",
            ],
            required=("data", "out"),
        )

    captured = capsys.readouterr()
    assert "Invalid alignment bounds" in captured.err
    assert "dx" in captured.err
    assert "lower bound 20 must be less than upper bound -20" in captured.err


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
