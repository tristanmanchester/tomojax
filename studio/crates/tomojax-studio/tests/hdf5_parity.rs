//! Studio's native NXtomo reader against TomoJAX's own: files written by `tj.save`
//! must give Studio the shapes, axes, spacings and units `tj.load` and `tj.inspect`
//! give Python. Run explicitly with the locked environment's TOMOJAX_TEST_PYTHON.

use std::path::Path;
use std::process::Command;

use serde_json::Value;
use studio_core::protocol::{ArrayDescriptor, Coordinates, Storage};
use tomojax_studio::metadata;

/// Writes the corpus into the directory given, and beside it `expected.json`: for each
/// file, what TomoJAX reads back from it.
const CORPUS: &str = r#"
import json, sys
from pathlib import Path

import numpy as np
import tomojax as tj
from tomojax.geometry import grid_volume_origin

out = Path(sys.argv[1])
rng = np.random.default_rng(7)
# Anisotropic voxels and pixels, an off-center volume and a shifted detector.
grid = tj.Grid(6, 5, 4, 0.5, 0.75, 1.25, vol_center=(0.3, -0.2, 0.1))
placed = tj.Grid(7, 5, 3, 0.4, 0.6, 1.5, vol_origin=(-1.0, 2.0, 0.5))
detector = tj.Detector(9, 7, 0.6, 1.1, (0.4, -0.35))
angles = np.array([0.0, 17.5, 61.0, 133.25, 270.0])
beam = tj.ConeBeam(40.0, 64.0, axis_offset=0.2)


def scan(geometry):
    views, rows, columns = len(geometry.angles), geometry.detector.nv, geometry.detector.nu
    return tj.Scan(rng.random((views, rows, columns), dtype=np.float32), geometry)


cone = scan(tj.ConeGeometry(placed, detector, angles, beam, tilt=5.0, tilt_about="z"))
poses = np.c_[rng.normal(0, 1, (5, 3)), rng.normal(0, 0.3, (5, 3))]
line = tj.ParallelGeometry(
    tj.Grid(8, 6, 1, 0.4, 0.4, 0.9), tj.Detector(11, 1, 0.5, 0.9, (0.3, 0.0)), angles
)
cases = {
    "parallel": ("scan", scan(tj.ParallelGeometry(grid, detector, angles))),
    "cone": ("scan", cone),
    "laminography": (
        "scan",
        scan(tj.LaminographyGeometry(grid, detector, angles, tilt=25.0, tilt_about="z")),
    ),
    "posed": ("scan", scan(tj.ParallelGeometry(grid, detector, angles)).with_poses(poses)),
    "alignment": ("alignment", tj.Alignment(
        cone.with_poses(poses), {}, rng.random(grid.shape, dtype=np.float32), grid
    )),
    "line": ("scan", scan(line)),
    "simulation": (
        "simulation",
        tj.simulate("sphere", geometry="lamino", grid=placed, detector=detector, n_views=6),
    ),
    "reconstruction": (
        "reconstruction",
        tj.Reconstruction(
            rng.random((3, 5, 7), dtype=np.float32), cone.geometry, "fbp", {}, cone,
            length_unit="mm",
        ),
    ),
}
expected = {}
for name, (kind, item) in cases.items():
    path = out / f"{name}.nxs"
    tj.save(path, item, **({"include_scan": True} if kind == "reconstruction" else {}))
    volume = None
    if kind == "simulation":
        simulation = tj.load_simulation(path)
        loaded, volume, volume_grid, unit = (
            simulation.scan, simulation.phantom, simulation.scan.grid, None
        )
    elif kind == "alignment":
        aligned = tj.load_alignment(path)
        loaded, volume, volume_grid, unit = aligned.scan, aligned.image, aligned.grid, None
    elif kind == "reconstruction":
        recon = tj.load_reconstruction(path)
        loaded, volume, volume_grid, unit = tj.load(path), recon.volume, recon.grid, recon.length_unit
    else:
        loaded, unit = tj.load(path), None
    expected[name] = {
        "path": str(path),
        "kind": kind,
        "inspection": tj.inspect(path),
        "projections": list(loaded.projections.shape),
        "pitch": [loaded.detector.dv, loaded.detector.du],
        "center": [loaded.detector.center[1], loaded.detector.center[0]],
        "length_unit": unit,
        "posed": loaded.poses is not None,
        "volume": None if volume is None else {
            "shape": list(np.shape(volume)),
            "origin": list(reversed(grid_volume_origin(volume_grid))),
            "spacing": [volume_grid.vz, volume_grid.vy, volume_grid.vx],
        },
    }
(out / "expected.json").write_text(json.dumps(expected))
"#;

fn corpus(directory: &Path) -> serde_json::Map<String, Value> {
    let python = std::env::var_os("TOMOJAX_TEST_PYTHON")
        .expect("set TOMOJAX_TEST_PYTHON to the locked Python executable");
    let status = Command::new(python)
        .args(["-c", CORPUS])
        .arg(directory)
        .env("JAX_PLATFORMS", "cpu")
        .status()
        .unwrap();
    assert!(status.success(), "the corpus script failed");
    let text = std::fs::read_to_string(directory.join("expected.json")).unwrap();
    match serde_json::from_str(&text).unwrap() {
        Value::Object(cases) => cases,
        other => panic!("expected one entry per file, got {other}"),
    }
}

fn close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "{what}: {actual} against TomoJAX's {expected}"
    );
}

fn uniform(array: &ArrayDescriptor, axis: usize) -> (f64, f64) {
    match array.axes[axis].coordinates {
        Coordinates::Uniform { origin, spacing } => (origin, spacing),
        ref other => panic!("axis {axis} is not uniform: {other:?}"),
    }
}

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

fn sizes(value: &Value) -> Vec<usize> {
    numbers(value).into_iter().map(|v| v as usize).collect()
}

#[test]
#[ignore = "writes its corpus with TomoJAX; set TOMOJAX_TEST_PYTHON and run --ignored"]
fn studio_reads_saved_nxtomo_as_tomojax_loads_it() {
    let directory = tempfile::tempdir().unwrap();
    let cases = corpus(directory.path());
    assert_eq!(cases.len(), 8);
    for (name, case) in &cases {
        let path = Path::new(case["path"].as_str().unwrap());
        let inspection = &case["inspection"];
        let (kind, arrays, report) = metadata::quick_metadata(path)
            .unwrap()
            .unwrap_or_else(|| panic!("{name}: Studio did not recognize the file"));
        assert_eq!(kind, case["kind"].as_str().unwrap(), "{name}: kind");
        for (key, fields) in [
            ("projection", &["shape", "dtype", "path"][..]),
            ("angles", &["count", "units", "path"]),
            ("detector_metadata", &["nu", "nv", "du", "dv", "center"]),
        ] {
            for field in fields {
                assert_eq!(
                    report[key][field], inspection[key][field],
                    "{name}: {key}.{field}"
                );
            }
        }
        assert_eq!(report["geometry"]["type"], inspection["geometry"]["type"]);
        assert_eq!(report["alignment"]["found"], case["posed"], "{name}: poses");
        assert_eq!(report["volume"]["shape"], inspection["volume"]["shape"]);

        // Studio and the worker name an unrecorded length unit `scan_length`.
        let unit = case["length_unit"].as_str().unwrap_or("scan_length");
        let projections = &arrays["projections"];
        assert_eq!(projections.shape, sizes(&case["projections"]), "{name}");
        let names: Vec<_> = projections.axes.iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["view", "v", "u"], "{name}");
        let view = &projections.axes[0];
        assert_eq!(view.unit, "degree", "{name}: angles");
        assert_eq!(
            view.coordinates,
            Coordinates::Coordinates {
                values: Storage::Hdf5 {
                    path: path.to_string_lossy().into_owned(),
                    dataset: inspection["angles"]["path"].as_str().unwrap().into(),
                }
            },
            "{name}: angles"
        );
        let (pitch, center) = (numbers(&case["pitch"]), numbers(&case["center"]));
        for axis in [1, 2] {
            let (origin, spacing) = uniform(projections, axis);
            let pixels = projections.shape[axis] as f64;
            assert_eq!(projections.axes[axis].unit, unit, "{name}: detector unit");
            close(
                spacing,
                pitch[axis - 1],
                &format!("{name}: detector pitch {axis}"),
            );
            // The middle pixel sits at the detector center.
            let middle = origin + (pixels - 1.0) / 2.0 * spacing;
            close(
                middle,
                center[axis - 1],
                &format!("{name}: detector center {axis}"),
            );
        }

        let volume_name = match case["kind"].as_str().unwrap() {
            "simulation" => Some("phantom"),
            "alignment" => Some("image"),
            "reconstruction" => Some("volume"),
            _ => None,
        };
        assert_eq!(
            arrays.len(),
            1 + usize::from(volume_name.is_some()),
            "{name}"
        );
        if let Some(volume_name) = volume_name {
            let expected = &case["volume"];
            let volume = &arrays[volume_name];
            assert_eq!(volume.shape, sizes(&expected["shape"]), "{name}");
            assert_eq!(volume.unit, format!("1/{unit}"), "{name}: attenuation unit");
            let (origins, spacings) = (numbers(&expected["origin"]), numbers(&expected["spacing"]));
            for (axis, label) in ["z", "y", "x"].into_iter().enumerate() {
                assert_eq!(volume.axes[axis].name, label, "{name}");
                assert_eq!(volume.axes[axis].unit, unit, "{name}: voxel unit");
                let (origin, spacing) = uniform(volume, axis);
                close(
                    origin,
                    origins[axis],
                    &format!("{name}: first {label} voxel"),
                );
                close(spacing, spacings[axis], &format!("{name}: {label} pitch"));
            }
        }

        // Given the worker's own inspection instead of its header-only one, Studio
        // registers the same arrays.
        let audited = metadata::inspection_arrays(inspection, path).unwrap();
        assert_eq!(audited, arrays, "{name}: the audited registration differs");
    }
}
