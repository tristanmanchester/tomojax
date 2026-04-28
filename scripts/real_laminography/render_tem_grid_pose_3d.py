"""Render 3D diagnostics for per-view TEM-grid pose corrections."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw


def _read_params(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float64)
    if data.shape == ():
        data = np.asarray([data])
    return {name: np.asarray(data[name], dtype=np.float64) for name in data.dtype.names or ()}


def _build_pose_geometry(
    params: dict[str, np.ndarray],
    *,
    radius: float,
    correction_scale: float,
) -> dict[str, np.ndarray]:
    view = params["view"]
    n = int(view.size)
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radial = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
    tangent = np.stack([-np.sin(theta), np.cos(theta), np.zeros_like(theta)], axis=1)
    vertical = np.zeros_like(radial)
    vertical[:, 2] = 1.0

    dx = params["dx"]
    dz = params["dz"]
    phi_deg = params["phi_deg"]
    nominal = radius * radial
    corrected = nominal + correction_scale * dx[:, None] * radial + correction_scale * dz[:, None] * vertical

    return {
        "view": view,
        "theta_deg": np.rad2deg(theta),
        "nominal": nominal,
        "corrected": corrected,
        "displacement": corrected - nominal,
        "radial": radial,
        "tangent": tangent,
        "dx": dx,
        "dz": dz,
        "phi_deg": phi_deg,
    }


def _color_map(values: np.ndarray) -> list[tuple[int, int, int]]:
    arr = np.asarray(values, dtype=np.float64)
    max_abs = float(np.nanmax(np.abs(arr))) if arr.size else 1.0
    if max_abs <= 0.0 or not np.isfinite(max_abs):
        max_abs = 1.0
    norm = np.clip((arr / max_abs + 1.0) * 0.5, 0.0, 1.0)
    colors: list[tuple[int, int, int]] = []
    for t in norm:
        if t < 0.5:
            u = t / 0.5
            r = int(round(49 + u * (245 - 49)))
            g = int(round(130 + u * (245 - 130)))
            b = int(round(189 + u * (245 - 189)))
        else:
            u = (t - 0.5) / 0.5
            r = int(round(245 + u * (202 - 245)))
            g = int(round(245 + u * (52 - 245)))
            b = int(round(245 + u * (47 - 245)))
        colors.append((r, g, b))
    return colors


def _project(
    points: np.ndarray,
    *,
    width: int,
    height: int,
    scale: float,
    yaw_deg: float,
    pitch_deg: float,
) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    ry = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rotated = np.asarray(points, dtype=np.float64) @ ry.T @ rx.T
    projected = rotated[:, :2] * scale
    projected[:, 0] += width / 2.0
    projected[:, 1] = height / 2.0 - projected[:, 1]
    return projected


def _draw_static_pose_png(geom: dict[str, np.ndarray], path: Path, *, yaw_deg: float, pitch_deg: float) -> None:
    width, height = 1400, 960
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    nominal = geom["nominal"]
    corrected = geom["corrected"]
    phi_deg = geom["phi_deg"]
    all_points = np.concatenate([nominal, corrected], axis=0)
    extent = float(np.nanmax(np.linalg.norm(all_points[:, :2], axis=1)))
    z_extent = float(np.nanmax(np.abs(all_points[:, 2])))
    scale = min(width, height) * 0.36 / max(extent, z_extent, 1.0)
    nominal_2d = _project(nominal, width=width, height=height, scale=scale, yaw_deg=yaw_deg, pitch_deg=pitch_deg)
    corrected_2d = _project(corrected, width=width, height=height, scale=scale, yaw_deg=yaw_deg, pitch_deg=pitch_deg)
    colors = _color_map(phi_deg)

    draw.text((34, 24), "Per-projection pose correction in acquisition space", fill=(20, 20, 20))
    draw.text(
        (34, 48),
        f"Grey ring = nominal view orbit; colored points/arrows = corrected poses; color = phi; camera = yaw {yaw_deg:g}, pitch {pitch_deg:g}",
        fill=(80, 80, 80),
    )
    draw.line([tuple(p) for p in nominal_2d] + [tuple(nominal_2d[0])], fill=(190, 190, 190), width=2)
    for idx in range(0, nominal.shape[0], 8):
        draw.line([tuple(nominal_2d[idx]), tuple(corrected_2d[idx])], fill=(80, 80, 80), width=1)
    for idx, point in enumerate(corrected_2d):
        color = colors[idx]
        r = 4
        draw.ellipse((point[0] - r, point[1] - r, point[0] + r, point[1] + r), fill=color, outline=(30, 30, 30))

    legend_x, legend_y = width - 270, 40
    draw.text((legend_x, legend_y), "phi_deg", fill=(20, 20, 20))
    for i, t in enumerate(np.linspace(0.0, 1.0, 120)):
        value = -1.0 + 2.0 * t
        color = _color_map(np.asarray([value]))[0]
        draw.line([(legend_x + i, legend_y + 24), (legend_x + i, legend_y + 42)], fill=color)
    max_phi = float(np.nanmax(np.abs(phi_deg)))
    draw.text((legend_x, legend_y + 46), f"-{max_phi:.3f} deg", fill=(70, 70, 70))
    draw.text((legend_x + 78, legend_y + 46), "0", fill=(70, 70, 70))
    draw.text((legend_x + 112, legend_y + 46), f"+{max_phi:.3f} deg", fill=(70, 70, 70))
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.asarray(image))


def _write_pose_csv(geom: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "view",
                "theta_deg",
                "dx",
                "dz",
                "phi_deg",
                "nominal_x",
                "nominal_y",
                "nominal_z",
                "corrected_x",
                "corrected_y",
                "corrected_z",
                "disp_x",
                "disp_y",
                "disp_z",
                "disp_norm",
            ],
        )
        writer.writeheader()
        for idx in range(geom["view"].size):
            disp = geom["displacement"][idx]
            writer.writerow(
                {
                    "view": int(geom["view"][idx]),
                    "theta_deg": float(geom["theta_deg"][idx]),
                    "dx": float(geom["dx"][idx]),
                    "dz": float(geom["dz"][idx]),
                    "phi_deg": float(geom["phi_deg"][idx]),
                    "nominal_x": float(geom["nominal"][idx, 0]),
                    "nominal_y": float(geom["nominal"][idx, 1]),
                    "nominal_z": float(geom["nominal"][idx, 2]),
                    "corrected_x": float(geom["corrected"][idx, 0]),
                    "corrected_y": float(geom["corrected"][idx, 1]),
                    "corrected_z": float(geom["corrected"][idx, 2]),
                    "disp_x": float(disp[0]),
                    "disp_y": float(disp[1]),
                    "disp_z": float(disp[2]),
                    "disp_norm": float(np.linalg.norm(disp)),
                }
            )


def _write_interactive_html(geom: dict[str, np.ndarray], path: Path) -> None:
    nominal = geom["nominal"]
    corrected = geom["corrected"]
    disp = geom["displacement"]
    view = geom["view"]
    dx = geom["dx"]
    dz = geom["dz"]
    phi = geom["phi_deg"]

    arrow_x: list[float | None] = []
    arrow_y: list[float | None] = []
    arrow_z: list[float | None] = []
    for idx in range(0, view.size, 4):
        arrow_x += [float(nominal[idx, 0]), float(corrected[idx, 0]), None]
        arrow_y += [float(nominal[idx, 1]), float(corrected[idx, 1]), None]
        arrow_z += [float(nominal[idx, 2]), float(corrected[idx, 2]), None]

    payload = {
        "view": view.tolist(),
        "dx": dx.tolist(),
        "dz": dz.tolist(),
        "phi": phi.tolist(),
        "dispNorm": np.linalg.norm(disp, axis=1).tolist(),
        "nominal": nominal.tolist(),
        "corrected": corrected.tolist(),
        "arrows": {"x": arrow_x, "y": arrow_y, "z": arrow_z},
    }
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>TEM-grid per-projection pose correction</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #202124; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    .plot {{ width: 100%; height: 680px; border: 1px solid #ddd; }}
    p {{ max-width: 1100px; line-height: 1.4; color: #4b5563; }}
  </style>
</head>
<body>
  <h1>TEM-grid per-projection pose correction</h1>
  <p>
    Left: nominal projection views on a circular acquisition orbit, with corrected positions offset by dx
    radially and dz vertically. Grey spokes are correction vectors every 4 views. Color encodes per-view phi.
    Right: the same correction field as a pose-space trajectory: view index vs dx vs dz, color encoded by phi.
  </p>
  <div class="grid">
    <div id="orbit" class="plot"></div>
    <div id="trajectory" class="plot"></div>
  </div>
  <script>
    const data = {json.dumps(payload)};
    const nominal = {{
      type: "scatter3d",
      mode: "lines",
      name: "nominal orbit",
      x: data.nominal.map(p => p[0]),
      y: data.nominal.map(p => p[1]),
      z: data.nominal.map(p => p[2]),
      line: {{color: "rgba(150,150,150,0.65)", width: 4}}
    }};
    const corrected = {{
      type: "scatter3d",
      mode: "markers+lines",
      name: "corrected projection pose",
      x: data.corrected.map(p => p[0]),
      y: data.corrected.map(p => p[1]),
      z: data.corrected.map(p => p[2]),
      marker: {{
        size: 4,
        color: data.phi,
        colorscale: "RdBu",
        reversescale: true,
        colorbar: {{title: "phi deg"}}
      }},
      line: {{color: "rgba(40,90,160,0.4)", width: 2}},
      text: data.view.map((v, i) => `view ${{v}}<br>dx=${{data.dx[i].toFixed(3)}} px<br>dz=${{data.dz[i].toFixed(3)}} px<br>phi=${{data.phi[i].toFixed(4)}} deg`),
      hoverinfo: "text"
    }};
    const arrows = {{
      type: "scatter3d",
      mode: "lines",
      name: "correction vectors",
      x: data.arrows.x,
      y: data.arrows.y,
      z: data.arrows.z,
      line: {{color: "rgba(40,40,40,0.4)", width: 2}}
    }};
    Plotly.newPlot("orbit", [nominal, arrows, corrected], {{
      title: "Spatial orbit view",
      scene: {{
        xaxis: {{title: "orbit x + dx radial"}},
        yaxis: {{title: "orbit y + dx radial"}},
        zaxis: {{title: "dz"}},
        aspectmode: "data"
      }},
      margin: {{l: 0, r: 0, b: 0, t: 40}}
    }});
    const trajectory = {{
      type: "scatter3d",
      mode: "markers+lines",
      name: "pose-space trajectory",
      x: data.view,
      y: data.dx,
      z: data.dz,
      marker: {{
        size: 4,
        color: data.phi,
        colorscale: "RdBu",
        reversescale: true,
        colorbar: {{title: "phi deg"}}
      }},
      line: {{color: "rgba(80,80,80,0.45)", width: 2}},
      text: data.view.map((v, i) => `view ${{v}}<br>dx=${{data.dx[i].toFixed(3)}} px<br>dz=${{data.dz[i].toFixed(3)}} px<br>phi=${{data.phi[i].toFixed(4)}} deg`),
      hoverinfo: "text"
    }};
    Plotly.newPlot("trajectory", [trajectory], {{
      title: "Pose-space trajectory",
      scene: {{
        xaxis: {{title: "view index"}},
        yaxis: {{title: "dx px"}},
        zaxis: {{title: "dz px"}},
        aspectmode: "cube"
      }},
      margin: {{l: 0, r: 0, b: 0, t: 40}}
    }});
  </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def render(args: argparse.Namespace) -> dict[str, Any]:
    params = _read_params(Path(args.params))
    geom = _build_pose_geometry(params, radius=float(args.radius), correction_scale=float(args.correction_scale))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "projection_pose_corrections_3d.png"
    html_path = out_dir / "projection_pose_corrections_3d.html"
    csv_path = out_dir / "projection_pose_corrections_3d.csv"
    manifest_path = out_dir / "projection_pose_corrections_3d_manifest.json"
    _draw_static_pose_png(geom, png_path, yaw_deg=float(args.yaw_deg), pitch_deg=float(args.pitch_deg))
    _write_interactive_html(geom, html_path)
    _write_pose_csv(geom, csv_path)
    manifest: dict[str, Any] = {
        "params": str(Path(args.params).resolve()),
        "n_views": int(geom["view"].size),
        "radius": float(args.radius),
        "correction_scale": float(args.correction_scale),
        "static_camera": {
            "yaw_deg": float(args.yaw_deg),
            "pitch_deg": float(args.pitch_deg),
        },
        "coordinate_model": {
            "nominal": "views placed uniformly on a circular acquisition orbit",
            "dx": "rendered as radial detector-u displacement from the nominal orbit",
            "dz": "rendered as vertical displacement",
            "phi_deg": "rendered as point color",
        },
        "summary": {
            "dx_min": float(np.nanmin(geom["dx"])),
            "dx_max": float(np.nanmax(geom["dx"])),
            "dz_min": float(np.nanmin(geom["dz"])),
            "dz_max": float(np.nanmax(geom["dz"])),
            "phi_deg_min": float(np.nanmin(geom["phi_deg"])),
            "phi_deg_max": float(np.nanmax(geom["phi_deg"])),
            "disp_norm_max_scaled": float(np.nanmax(np.linalg.norm(geom["displacement"], axis=1))),
        },
        "outputs": {
            "static_png": str(png_path.resolve()),
            "interactive_html": str(html_path.resolve()),
            "csv": str(csv_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", required=True, help="params.csv from the final alignment stage")
    parser.add_argument("--out", required=True)
    parser.add_argument("--radius", type=float, default=80.0)
    parser.add_argument("--correction-scale", type=float, default=2.0)
    parser.add_argument("--yaw-deg", type=float, default=-42.0)
    parser.add_argument("--pitch-deg", type=float, default=46.0)
    manifest = render(parser.parse_args())
    print(json.dumps(manifest["outputs"], indent=2))


if __name__ == "__main__":
    main()
