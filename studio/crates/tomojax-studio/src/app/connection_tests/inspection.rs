//! The acquisition pane must be usable while the inspector is still reading the stack.
use super::*;

#[test]
#[ignore = "uses the real executor and a named project behind slowlink"]
fn tiff_projection_precedes_numerical_inspection() {
    let Ok(root) = std::env::var("TOMOJAX_CONNECTION_FIXTURE") else {
        isolated_scene(
            "app::connection_tests::inspection::tiff_projection_precedes_numerical_inspection",
            "",
        );
        return;
    };
    let root = Path::new(&root);
    let source = root.join("scan");
    fs::create_dir(&source).unwrap();
    let python = std::env::var("TOMOJAX_TEST_PYTHON").unwrap();
    assert!(Command::new(&python).args(["-c", "import sys,numpy as np,tifffile as tf; from pathlib import Path; d=Path(sys.argv[1]); a=np.arange(120,dtype=np.uint16).reshape(10,12); tf.imwrite(d/'part_0001.tif',a); tf.imwrite(d/'part_0002.tif',a+200)"]).arg(&source).status().unwrap().success());
    fs::write(source.join("part.xtekct"), "[XTekCT]\nDetectorOffsetX=0\nDetectorOffsetY=0\nCentreOfRotationTop=0\nCentreOfRotationBottom=0\nName=part\nSrcToObject=12.0\nSrcToDetector=18.0\nDetectorPixelSizeX=0.1\nDetectorPixelSizeY=0.1\nWhiteLevel=60000\nInitialAngle=0\nAngularStep=90\n").unwrap();
    fs::write(
        root.join("sitecustomize.py"),
        r#"import os, time
from pathlib import Path
from tomojax.io import _inspection
root = Path(os.environ["TOMOJAX_CONNECTION_FIXTURE"])
original = _inspection.projection_stats
def held(data):
    (root / "reading").touch()
    deadline = time.monotonic() + 35
    while not (root / "release").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    return original(data)
_inspection.projection_stats = held
"#,
    )
    .unwrap();
    let wrapped = root.join("inspection-python");
    executable(
        &wrapped,
        "#!/bin/sh\nexport PYTHONPATH=\"$TOMOJAX_CONNECTION_FIXTURE:${PYTHONPATH:-}\"\nexec \"$TOMOJAX_TEST_PYTHON\" \"$@\"\n",
    );
    let mut state = AppState::init();
    let start = import_named(
        &mut state,
        root,
        &root.join("Review"),
        &source,
        wrapped.to_str().unwrap(),
    );
    wait(
        &mut state,
        "first lossless sample while statistics are blocked",
        |s| root.join("reading").exists() && s.imaging.plane(4).is_some_and(|p| p.lossless),
    );
    let asset = state.model.active_asset.clone().unwrap();
    let plane = state.imaging.plane(4).unwrap();
    assert_eq!(plane.domain, "counts");
    assert_eq!(plane.probe([0., 0.]), Some(([0, 0], 119.)));
    assert!(
        state
            .project
            .as_ref()
            .unwrap()
            .document
            .asset(&asset)
            .unwrap()
            .inspection
            .is_null()
    );
    eprintln!(
        "First registered Nikon sample and exact probe in {:.3} s",
        start.elapsed().as_secs_f64()
    );
    fs::write(root.join("release"), "").unwrap();
    wait(&mut state, "finished scientific report", |s| {
        s.project
            .as_ref()
            .unwrap()
            .document
            .asset(&asset)
            .unwrap()
            .inspection
            .pointer("/projection/stats/max")
            .and_then(Value::as_f64)
            == Some(319.)
    });
    wait(
        &mut state,
        "first projection still registered after the audit",
        |s| s.imaging.plane(4).is_some_and(|p| p.lossless),
    );
    assert_eq!(
        state.imaging.plane(4).unwrap().probe([0., 0.]),
        Some(([0, 0], 119.))
    );
}
