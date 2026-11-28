//! A detector impulse between overview samples must reach both the GPU and the probe.
use super::*;

const EDGE: usize = 2560;
const IMPULSE: [usize; 2] = [1279, 1283];

fn resolution_target(path: &Path) {
    let mut values = vec![0f32; EDGE * EDGE];
    for (i, v) in values.iter_mut().enumerate() {
        let y = i / EDGE;
        let x = i % EDGE;
        let radius = ((y as f32 - IMPULSE[0] as f32).powi(2)
            + (x as f32 - IMPULSE[1] as f32).powi(2))
        .sqrt();
        // Rings locate the isolated impulse. Alternating bars expose the change in sampling.
        *v = if (38. ..40.).contains(&radius)
            || (170. ..174.).contains(&radius)
            || (750. ..755.).contains(&radius)
        {
            0.65
        } else if (95..205).contains(&x.abs_diff(IMPULSE[1])) && y.abs_diff(IMPULSE[0]) < 48 {
            if (x / 2).is_multiple_of(2) {
                0.85
            } else {
                0.15
            }
        } else {
            0.05
        };
    }
    values[IMPULSE[0] * EDGE + IMPULSE[1]] = 1.;
    let file = hdf5::File::create(path).unwrap();
    file.create_group("entry")
        .unwrap()
        .new_dataset::<f32>()
        .shape([1, EDGE, EDGE])
        .create("projections")
        .unwrap()
        .write_raw(&values)
        .unwrap();
}

#[test]
#[ignore = "uses the native worker, slowlink and an isolated connection catalog"]
fn zoom_recovers_the_skipped_detector_sample() {
    let Ok(root) = std::env::var("TOMOJAX_CONNECTION_FIXTURE") else {
        for scene in ["local-fit", "local-4x", "connection-fit", "connection-4x"] {
            isolated_scene(
                "app::connection_tests::zoom::zoom_recovers_the_skipped_detector_sample",
                scene,
            );
        }
        return;
    };
    let root = Path::new(&root);
    let scene = std::env::var("TOMOJAX_CONNECTION_SCENE").unwrap();
    let mut state = open_target(root, scene.starts_with("connection"));
    let mut tx = Tx::default();
    let uv = [
        (IMPULSE[1] as f32 + 0.5) / EDGE as f32,
        (IMPULSE[0] as f32 + 0.5) / EDGE as f32,
    ];
    let pane = if std::env::var_os("TOMOJAX_CONNECTION_CAPTURES").is_some() {
        900.
    } else {
        420.
    };
    let stride = if pane > 512. { 3 } else { 5 };
    state.render_panes(&[(9, pane)]);
    wait(&mut state, "overview at pane resolution", |s| {
        s.imaging
            .plane(4)
            .is_some_and(|p| p.lossless && p.stride == stride)
    });
    let base = state.imaging.plane(4).unwrap();
    let (index, value) = base.probe(uv).unwrap();
    assert_ne!(index, IMPULSE);
    assert_eq!(value, 0.05);
    state.ui.probe = Some((9, uv));
    if scene.ends_with("4x") {
        let started = Instant::now();
        state.dispatch(UiAction::Zoom(4.), &mut tx);
        state.render_panes(&[(9, pane)]);
        wait(&mut state, "lossless zoom", |s| {
            s.imaging
                .plane(4)
                .is_some_and(|p| p.detail.as_ref().is_some_and(|d| d.lossless))
        });
        let base = state.imaging.plane(4).unwrap();
        let detail = base.detail.as_ref().unwrap();
        assert_eq!(detail.region, [[960, 1600], [960, 1600]]);
        assert_eq!((detail.stride, detail.shape), (1, [640, 640]));
        assert_eq!(base.probe(uv), Some((IMPULSE, 1.)));
        state.ui.probe = Some((9, uv.map(|v| (v - 0.5) * 4. + 0.5)));
        assert!(
            state
                .probe_text(9, PlaneKind::Projection)
                .unwrap()
                .contains("v[1279]  u[1283]   1.00000")
        );
        eprintln!(
            "{scene}: zoom to lossless in {:.3}s",
            started.elapsed().as_secs_f64()
        );
    }
    if std::env::var_os("TOMOJAX_CONNECTION_CAPTURES").is_some() {
        snap(state, &scene);
    }
}

fn open_target(root: &Path, connected: bool) -> AppState {
    let source = root.join("Detector resolution target.nxs");
    resolution_target(&source);
    let mut state = AppState::init();
    let project = root.join("Detector detail");
    if connected {
        let python = std::env::var("TOMOJAX_TEST_PYTHON").unwrap();
        import_named(&mut state, root, &project, &source, &python);
    } else {
        state.open_project(&project, true).unwrap();
        state.import_paths(vec![source]);
    }
    wait(&mut state, "lossless overview", |s| {
        s.imaging.plane(4).is_some_and(|p| p.lossless)
    });
    state.dispatch(
        UiAction::SetMainView(PlaneKind::Projection),
        &mut Tx::default(),
    );
    state
}
