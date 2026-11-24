use super::*;

/// Freeze an observed application state with the same renderer as the window.
/// Only tests opt in to evidence capture; ordinary --snap still opens a saved
/// project without starting a numerical worker.
pub(crate) fn capture_native(mut state: AppState, path: &Path) {
    // Match the ordinary actions a reviewer would take before inspecting the
    // result. Numerical values and observed worker state remain untouched.
    state.ui.jobs_open = false;
    if let Some(control) = state
        .model
        .validation
        .issues
        .first()
        .and_then(|issue| issue.control.as_ref())
    {
        state.ui.advanced_open = true;
        if control == "corrections-json" {
            crate::ui::corrections::reveal_json(&mut state.ui);
        }
    }
    let mut spec = crate::app_spec();
    spec.setup = None;
    spec.poll = Some(|state, _, tx| {
        if let Some(control) = state
            .model
            .validation
            .issues
            .first()
            .and_then(|issue| issue.control.as_ref())
        {
            tx.focus(control.clone(), true);
            tx.scroll_into_view(format!("validation-field-{control}"));
        } else if state
            .model
            .preview
            .as_ref()
            .is_some_and(|preview| preview.count.is_some())
        {
            tx.scroll_into_view("preview-index");
        }
        false
    });
    spec.on_key = None;
    let app = rune::App::new(spec).init(move || state).compositor(
        crate::volume::ScientificCompositor::new(AppState::render_items),
    );
    let tree = path.with_extension("tree.json");
    let args = rune::tooling::Args::parse_from(vec![
        "--snap".into(),
        path.display().to_string(),
        "--tree".into(),
        tree.display().to_string(),
        "--size".into(),
        "1440x900".into(),
        "--scale".into(),
        "1".into(),
        "--theme".into(),
        "light".into(),
        "--at".into(),
        "1".into(),
    ]);
    rune::tooling::headless(app, args);
    assert!(path.is_file(), "native evidence was not rendered");
}
use crate::project::JobStatus;

#[test]
fn preview_job_controls_target_the_worker_job_instead_of_a_local_keep_copy() {
    let mut state = AppState::init();
    state.model.jobs = vec![
        JobView {
            id: "keep-copy".into(),
            status: crate::ui::model::JobStatus::Running,
            ..Default::default()
        },
        JobView {
            id: "device-run".into(),
            status: crate::ui::model::JobStatus::Paused,
            ..Default::default()
        },
    ];
    state.running = Some(Running {
        job: "device-run".into(),
        handles: vec![],
        products: vec![],
        next_input: 0,
    });
    assert_eq!(state.preview_blocker().unwrap().id, "device-run");
    state.running = None;
    assert!(
        state.preview_blocker().is_none(),
        "a local copy does not hold the numerical device"
    );
}

#[test]
fn kept_preview_elapsed_identifies_save_time() {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    let mut project = Project::open(directory.path(), true).unwrap();
    let mut job = crate::project::Job::new("slice", json!({}), vec![]);
    job.started = Some(1000);
    job.finished = Some(1120);
    job.status = JobStatus::Completed;
    let mut kept = job.clone();
    kept.id = "kept-preview".into();
    kept.kept_preview = Some(crate::project::KeptPreviewContext {
        request: "preview-1".into(),
        revision: 2,
    });
    project.document.jobs = vec![job, kept];
    state.project = Some(project);
    state.refresh_model();
    assert_eq!(state.model.jobs[0].elapsed.as_deref(), Some("0.1 s"));
    assert_eq!(state.model.jobs[1].elapsed.as_deref(), Some("save 0.1 s"));
}

fn registered_projection_fixture() -> (tempfile::TempDir, AppState) {
    use std::io::Write;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("projections.npy");
    let mut file = std::fs::File::create(&path).unwrap();
    let mut header = "{'descr': '<f4', 'fortran_order': False, 'shape': (4, 5, 8), }".to_owned();
    while !(10 + header.len() + 1).is_multiple_of(64) {
        header.push(' ');
    }
    header.push('\n');
    file.write_all(b"\x93NUMPY\x01\x00").unwrap();
    file.write_all(&(header.len() as u16).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    for _ in 0..4 {
        for row in 0..5 {
            for column in 0..8 {
                let value: f32 = if row == 2 { column as f32 + 1. } else { 0. };
                file.write_all(&value.to_le_bytes()).unwrap();
            }
        }
    }
    drop(file);
    let descriptor = serde_json::from_value(json!({
        "shape":[4,5,8], "dtype":"<f4", "unit":"1", "domain":"line_integral",
        "storage":{"type":"npy","path":path},
        "axes":[
            {"name":"view","frame":"acquisition","unit":"degree","coordinates":{"type":"uniform","origin":0.,"spacing":45.}},
            {"name":"v","frame":"detector","unit":"mm","coordinates":{"type":"uniform","origin":-2.,"spacing":1.}},
            {"name":"u","frame":"detector","unit":"mm","coordinates":{"type":"uniform","origin":-3.5,"spacing":1.}}
        ],"fixed":[],"columns":[]
    })).unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(directory.path(), true).unwrap());
    let mut asset = Asset::input(path, false);
    asset.arrays.insert("projections".into(), descriptor);
    state.model.active_asset = Some(asset.id.clone());
    state.project.as_mut().unwrap().document.assets.push(asset);
    state.ui.main_view = PlaneKind::Projection;
    state.refresh_images();
    drive(&mut state, |state| {
        state.imaging.plane(4).is_some() && state.imaging.plane(5).is_some()
    });
    (directory, state)
}

#[test]
fn default_sinogram_uses_the_central_row_and_preserves_an_explicit_row() {
    let (_directory, mut state) = registered_projection_fixture();
    assert_eq!(state.ui.display.sinogram_row, 2);
    assert_eq!(state.imaging.plane(5).unwrap().index, 2);
    assert!(state.imaging.plane(5).unwrap().statistics.max > 0.);
    state.ui.display.sinogram_row = 0;
    state.refresh_images();
    drive(&mut state, |state| {
        state.imaging.plane(5).is_some_and(|plane| plane.index == 0)
    });
    assert_eq!(state.imaging.plane(5).unwrap().statistics.max, 0.);
    let context = state.capture_workspace();
    state.restore_workspace(&context);
    state.refresh_images();
    assert_eq!(state.ui.display.sinogram_row, 0);
}

#[test]
fn worker_disconnect_unbinds_temporary_preview_files() {
    let (_directory, mut state) = registered_projection_fixture();
    let array = state.project.as_ref().unwrap().document.assets[0]
        .projections()
        .unwrap()
        .clone();
    state.set_operation("correction_preview", None);
    state.ui.workspace = Workspace::Prepare;
    state.preview_context.active = Some(crate::previews::PreviewKind::Corrections);
    state.preview_context.arrays.insert("after".into(), array);
    state.preview_context.signature = "old-worker-session".into();
    state.model.preview = Some(PreviewView {
        operation: "correction_preview".into(),
        has_image: true,
        can_keep: true,
        ..Default::default()
    });
    state.refresh_images();
    state.engine_event(EngineEvent::Disconnected {
        code: Some(1),
        reason: "Test worker exited".into(),
        unfinished: vec![],
    });
    assert!(state.preview_context.arrays.is_empty());
    assert!(state.preview_context.signature.is_empty());
    assert!(state.model.preview.is_none());
    assert!(
        state.imaging.plane(4).is_none(),
        "navigation must not reuse a descriptor from the closed session"
    );
}

#[test]
fn window_fields_show_the_rendered_range_without_overwriting_unfinished_edits() {
    let (_directory, mut state) = registered_projection_fixture();
    let current = state.imaging.plane(4).unwrap().statistics.window;
    assert_eq!(
        state.ui.display.window_low.text.parse::<f32>().unwrap(),
        current[0]
    );
    assert_eq!(
        state.ui.display.window_high.text.parse::<f32>().unwrap(),
        current[1]
    );
    state.ui.display.window_low.replace_text("-");
    state.ui.display.window_high.replace_text("2.5");
    state.refresh_plane_model();
    assert_eq!(state.ui.display.window_low.text, "-");
    assert_eq!(state.ui.display.window_high.text, "2.5");
    assert!(state.applied_window.is_none());
}

/// A z-ramp float32 volume of `depth` × 4 × 4 samples, registered as a result.
fn ramp_volume(directory: &Path, name: &str, depth: usize) -> Asset {
    use std::io::Write;
    let path = directory.join(format!("{name}.npy"));
    let mut file = std::fs::File::create(&path).unwrap();
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({depth}, 4, 4), }}");
    while !(10 + header.len() + 1).is_multiple_of(64) {
        header.push(' ');
    }
    header.push('\n');
    file.write_all(b"\x93NUMPY\x01\x00").unwrap();
    file.write_all(&(header.len() as u16).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    for z in 0..depth {
        for _ in 0..16 {
            file.write_all(&(z as f32).to_le_bytes()).unwrap();
        }
    }
    drop(file);
    let axis = |name: &str| json!({"name":name,"frame":"sample","unit":"mm","coordinates":{"type":"uniform","origin":0.,"spacing":1.}});
    let descriptor = serde_json::from_value(json!({
        "shape":[depth,4,4], "dtype":"<f4", "unit":"1", "domain":"attenuation",
        "storage":{"type":"npy","path":path},
        "axes":[axis("z"), axis("y"), axis("x")],"fixed":[],"columns":[]
    }))
    .unwrap();
    let mut asset = Asset::input(path, false);
    asset.arrays.insert("volume".into(), descriptor);
    asset
}

#[test]
fn compared_volumes_of_different_depth_show_the_same_slice() {
    let directory = tempfile::tempdir().unwrap();
    let left = ramp_volume(directory.path(), "deep", 6);
    let right = ramp_volume(directory.path(), "shallow", 3);
    let mut state = AppState::init();
    state.project = Some(Project::open(directory.path(), true).unwrap());
    state.model.compare_left = Some(left.id.clone());
    state.model.compare_right = Some(right.id.clone());
    state.project.as_mut().unwrap().document.assets = vec![left, right];
    state.ui.workspace = Workspace::Compare;
    state.ui.display.indices[0] = 5;
    state.refresh_images();
    drive(&mut state, |state| {
        state.imaging.plane(6).is_some() && state.imaging.plane(7).is_some()
    });
    // The shared index stops at the shallower volume's last slice on both sides.
    for slot in [6, 7] {
        let plane = state.imaging.plane(slot).unwrap();
        assert_eq!(plane.index, 2, "slot {slot}");
        assert_eq!(plane.statistics.max, 2.);
    }
    assert_eq!(state.model.comparison_index, 2);
    assert_eq!(state.model.comparison_count, 3);
    assert_eq!(
        state.ui.display.indices[0], 2,
        "the next step starts from the shown slice"
    );
    // Outside Compare, the shorter compared volume no longer limits the slice.
    state.ui.workspace = Workspace::Explore;
    state.ui.display.indices[0] = 4;
    state.refresh_images();
    assert_eq!(state.ui.display.indices[0], 4);
}

#[test]
fn multi_input_jobs_take_inputs_in_project_order_not_id_order() {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(directory.path(), true).unwrap());
    let mut first = ramp_volume(directory.path(), "imported", 2);
    first.id = "scan-z".into();
    let mut second = ramp_volume(directory.path(), "simulated", 2);
    second.id = "result-a".into();
    state.project.as_mut().unwrap().document.assets = vec![first, second];
    state.model.selected_assets = ["result-a".to_owned(), "scan-z".to_owned()].into();
    assert_eq!(state.selected_inputs(), ["scan-z", "result-a"]);
}

#[test]
fn region_statistics_use_only_samples_whose_centres_are_inside_the_box() {
    let directory = tempfile::tempdir().unwrap();
    let volume = ramp_volume(directory.path(), "volume", 3);
    let mut state = AppState::init();
    state.project = Some(Project::open(directory.path(), true).unwrap());
    state.model.active_asset = Some(volume.id.clone());
    state.project.as_mut().unwrap().document.assets = vec![volume];
    state.ui.main_view = PlaneKind::Axial;
    state.refresh_images();
    drive(&mut state, |state| state.imaging.plane(1).is_some());
    state.ui.inspect_tool = InspectTool::Region;
    // Samples 0.2–1.8 on a 4 × 4 slice hold one sample centre, (1, 1).
    let uv = |pixel: f32| (pixel + 0.5) / 4.;
    state
        .measure(PlaneKind::Axial, [uv(0.2); 2], [uv(1.8); 2], true)
        .unwrap();
    let region = state.model.measurements.last().unwrap();
    assert!(region.detail.starts_with("1 samples"), "{}", region.detail);
    assert!(
        state
            .measure(PlaneKind::Axial, [uv(0.2); 2], [uv(0.8); 2], true)
            .is_err()
    );
    // A box in the outer half of the first sample holds no sample centre.
    assert!(
        state
            .measure(PlaneKind::Axial, [uv(-0.45); 2], [uv(-0.1); 2], true)
            .is_err()
    );
}

#[test]
fn switching_projects_saves_the_latest_workspace_or_stays() {
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("first");
    let second = directory.path().join("second");
    let mut state = AppState::init();
    state.open_project(&first, true).unwrap();
    state.ui.display.indices = [7, 8, 9];
    state.open_project(&second, true).unwrap();
    let saved = Project::open(&first, false).unwrap();
    assert_eq!(saved.document.workspace.indices, [7, 8, 9]);
    drop(saved);

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let read_only = std::fs::Permissions::from_mode(0o555);
        std::fs::set_permissions(&second, read_only).unwrap();
        let error = state.open_project(&first, false).unwrap_err();
        std::fs::set_permissions(&second, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(error.contains("stays open"), "{error}");
        assert_eq!(
            state.project.as_ref().unwrap().root,
            std::fs::canonicalize(&second).unwrap()
        );
    }
}

#[test]
fn a_finished_job_activates_its_result_only_while_its_inputs_are_selected() {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(&directory.path().join("project"), true).unwrap());
    let first = ramp_volume(directory.path(), "first", 2);
    let second = ramp_volume(directory.path(), "second", 2);
    let project = state.project.as_mut().unwrap();
    project.document.assets = vec![first.clone(), second.clone()];
    let mut jobs = Vec::new();
    for _ in 0..2 {
        let job = project::Job::new(
            "reconstruct",
            json!({}),
            vec![project.capture(&first.id).unwrap()],
        );
        jobs.push(job.id.clone());
        project.document.jobs.push(job);
    }
    // The user has moved on to the second scan: its draft keeps its input.
    state.model.selected_assets = [second.id.clone()].into();
    state.model.active_asset = Some(second.id.clone());
    state.complete_asset(&jobs[0], ramp_volume(directory.path(), "result-a", 2));
    assert_eq!(
        state.model.active_asset.as_deref(),
        Some(second.id.as_str())
    );
    // While the job's own input is still selected, its result opens.
    state.model.selected_assets = [first.id.clone()].into();
    state.model.active_asset = Some(first.id.clone());
    let result = ramp_volume(directory.path(), "result-b", 2);
    let result_id = result.id.clone();
    state.complete_asset(&jobs[1], result);
    assert_eq!(
        state.model.active_asset.as_deref(),
        Some(result_id.as_str())
    );
}

#[test]
fn header_metadata_keeps_worker_registered_arrays() {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(&directory.path().join("project"), true).unwrap());
    let mut result = ramp_volume(directory.path(), "corrected", 2);
    let mut authoritative = result.arrays["volume"].clone();
    authoritative.domain = "line_integrals".into();
    result
        .arrays
        .insert("projections".into(), authoritative.clone());
    let id = result.id.clone();
    state.project.as_mut().unwrap().document.assets.push(result);
    let mut header = authoritative.clone();
    header.domain = "stored_measurements".into();
    let mut arrays = std::collections::BTreeMap::new();
    arrays.insert("projections".to_owned(), header);
    state.local_event(LocalEvent::Metadata {
        asset: id.clone(),
        result: Ok(Some(("scan".into(), arrays, json!({})))),
    });
    let asset = state.project.as_ref().unwrap().document.asset(&id).unwrap();
    assert_eq!(asset.arrays["projections"].domain, "line_integrals");

    // A worker result keeps the kind the worker gave it.
    let project = state.project.as_mut().unwrap();
    let asset = project.document.asset_mut(&id).unwrap();
    asset.kind = "scan".into();
    asset.parent = Some("source-scan".into());
    state.local_event(LocalEvent::Metadata {
        asset: id.clone(),
        result: Ok(Some((
            "reconstruction".into(),
            Default::default(),
            json!({}),
        ))),
    });
    let asset = state.project.as_ref().unwrap().document.asset(&id).unwrap();
    assert_eq!(asset.kind, "scan");
}

#[test]
fn projects_are_not_switched_while_a_kept_preview_is_being_copied() {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state
        .open_project(&directory.path().join("first"), true)
        .unwrap();
    let mut job = project::Job::new("slice", json!({}), vec![]);
    job.status = JobStatus::Running;
    state.project.as_mut().unwrap().document.jobs.push(job);
    let error = state
        .open_project(&directory.path().join("second"), true)
        .unwrap_err();
    assert!(error.contains("active work"), "{error}");
}

#[test]
fn exporting_a_standalone_reconstruction_leaves_out_the_scan_it_lacks() {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(&directory.path().join("project"), true).unwrap());
    let mut volume = ramp_volume(directory.path(), "recon", 2);
    volume.kind = "reconstruction".into();
    volume.inspection = json!({"projection": {"found": false}});
    let id = volume.id.clone();
    let project = state.project.as_mut().unwrap();
    project.document.assets.push(volume);
    let input = project.capture(&id).unwrap();
    state.queue_export(input, directory.path().join("export.nxs"), "nxs");
    let job = state
        .project
        .as_ref()
        .unwrap()
        .document
        .jobs
        .last()
        .unwrap();
    assert_eq!(job.parameters["include_scan"], false);
}

#[test]
fn projection_arrays_come_from_a_selected_result_or_the_geometry_scan() {
    let directory = tempfile::tempdir().unwrap();
    let saved = ramp_volume(directory.path(), "forward", 2).arrays["volume"].clone();
    let selected = Product::Projections {
        projections: saved.clone(),
    };
    let scan = |handle: &str| Product::Scan {
        handle: handle.into(),
        projections: saved.clone(),
    };
    let (first, second) = (scan("first"), scan("second"));
    let chosen = |products: &[&Product]| {
        jobs::array_member(products.iter().copied(), "projections").map(|source| source.storage)
    };
    // A selected result wins wherever it sits in project order.
    for products in [[&selected, &first], [&first, &selected]] {
        assert_eq!(chosen(&products), Some(saved.storage.clone()));
    }
    // Otherwise the projections belong to the scan that supplies the geometry.
    assert!(matches!(
        chosen(&[&first, &second]),
        Some(crate::protocol::Storage::Member { handle, .. }) if handle == "first"
    ));

    // Geometry: forward projection into another selected scan, whichever comes
    // first; back projection from the scan whose projections it reads.
    let reconstruction = Product::Reconstruction {
        handle: "reconstruction".into(),
        volume: saved.clone(),
        scan: Some(crate::protocol::Storage::Member {
            handle: "reconstruction".into(),
            member: "scan".into(),
        }),
        method: "fbp".into(),
    };
    assert_eq!(
        jobs::projection_geometry(&[&reconstruction, &second], "volume"),
        Some(1)
    );
    assert_eq!(
        jobs::projection_geometry(&[&reconstruction], "volume"),
        Some(0)
    );
    assert_eq!(
        jobs::projection_geometry(&[&first, &second], "projections"),
        Some(0)
    );
    assert_eq!(
        jobs::projection_geometry(&[&selected, &second], "projections"),
        Some(1)
    );

    // Forward projection keeps the volume's own registered grid.
    let mut volume = saved.clone();
    for (axis, origin) in volume.axes.iter_mut().zip([-1.5, 2., 0.25]) {
        axis.coordinates = crate::protocol::Coordinates::Uniform {
            origin,
            spacing: 0.5,
        };
    }
    let volume = Product::Volume { volume };
    let mut operation = serde_json::Map::new();
    operation.insert("grid".into(), Value::Null);
    assert!(
        jobs::fill_projection_array(&mut operation, [&first, &volume].into_iter(), "volume")
            .unwrap()
    );
    assert_eq!(
        operation["grid"],
        json!({"nx":4,"ny":4,"nz":2,"vx":0.5,"vy":0.5,"vz":0.5,"vol_origin":[0.25,2.,-1.5]})
    );

    // A volume in another length unit than the scan's geometry is refused.
    let mut micrometres = saved.clone();
    for axis in &mut micrometres.axes {
        axis.unit = "um".into();
    }
    let mut detector = saved.clone();
    detector.axes[1].unit = "mm".into();
    let scan = Product::Scan {
        handle: "scan".into(),
        projections: detector,
    };
    let volume = Product::Volume {
        volume: micrometres,
    };
    let error = jobs::fill_projection_array(
        &mut serde_json::Map::new(),
        [&scan, &volume].into_iter(),
        "volume",
    )
    .unwrap_err();
    assert!(error.contains("um") && error.contains("mm"), "{error}");
}

fn drive(state: &mut AppState, condition: impl Fn(&AppState) -> bool) {
    let limit = Instant::now() + Duration::from_secs(45);
    while !condition(state) {
        assert!(
            Instant::now() < limit,
            "workflow timed out: {:?}; jobs={:?}",
            state.model.error,
            state.model.jobs
        );
        state.poll();
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
#[ignore = "requires the installed TomoJAX environment; exercised by the real workflow gate"]
fn real_application_submits_captured_jobs_persists_results_and_reopens_context() {
    let python = std::env::var_os("TOMOJAX_TEST_PYTHON")
        .expect("set TOMOJAX_TEST_PYTHON to the installed Python");
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(directory.path(), true).unwrap());
    state.startup_input = None;
    state.auto_demo = false;
    state.started = true;
    state.engine = Some(Engine::spawn(PathBuf::from(python), std::sync::Arc::new(|| {})).unwrap());
    drive(&mut state, |s| s.model.backend.ready);
    state.set_operation("simulate", Some(&json!({"size":16,"n_views":24,"seed":3})));
    state.queue_operation("simulate").unwrap();
    drive(&mut state, |s| {
        s.project
            .as_ref()
            .unwrap()
            .document
            .jobs
            .iter()
            .all(|j| !j.status.active())
            && s.metadata_pending == 0
    });
    let project = state.project.as_ref().unwrap();
    assert_eq!(
        project.document.jobs[0].status,
        JobStatus::Completed,
        "{:?}",
        project.document.jobs[0].error
    );
    assert_eq!(project.document.assets.len(), 1);
    assert_eq!(project.document.assets[0].name, "Shepp phantom");
    assert!(project.document.assets[0].volume().is_some());
    assert!(project.document.assets[0].projections().is_some());
    assert!(project.resolve(&project.document.assets[0].path).is_file());

    state.set_operation("reconstruct", Some(&json!({"method":"fourier"})));
    state.queue_operation("reconstruct").unwrap();
    // A later edit must have no effect on the queued command or resulting solver.
    state.set_operation(
        "reconstruct",
        Some(&json!({"method":"cgls","iterations":7})),
    );
    assert_eq!(
        state.project.as_ref().unwrap().document.jobs[1].parameters["method"],
        "fourier"
    );
    drive(&mut state, |s| {
        s.project
            .as_ref()
            .unwrap()
            .document
            .jobs
            .iter()
            .all(|j| !j.status.active())
            && s.metadata_pending == 0
    });
    let project = state.project.as_ref().unwrap();
    assert_eq!(
        project.document.jobs[1].status,
        JobStatus::Completed,
        "{:?}",
        project.document.jobs[1].error
    );
    assert_eq!(
        project.document.assets[1].method.as_deref(),
        Some("fourier")
    );
    assert_eq!(project.document.assets[1].name, "Shepp phantom · Fourier");
    assert_eq!(state.ui.main_view, PlaneKind::Axial);
    let volume = project.document.assets[1].volume().unwrap();
    assert_eq!(volume.shape, [16, 16, 16]);
    assert!(
        matches!(&volume.storage,crate::protocol::Storage::Hdf5{path,..} if !Path::new(path).is_absolute())
    );
    let source_id = project.document.assets[0].id.clone();
    let volume_id = project.document.assets[1].id.clone();
    // These products have no savable reconstruction handle. Their pixels and
    // coordinate vectors must survive the worker session as registered arrays.
    for (operation, parameters) in [
        ("slice", json!({"axis":"y","index":8,"method":"fourier"})),
        (
            "centre_sweep",
            json!({"axis":"z","index":8,"centres":[-1.0,0.0,1.0],"method":"fourier"}),
        ),
        ("region_preview", json!({})),
    ] {
        let preview_source = if operation == "region_preview" {
            state.set_operation("simulate",Some(&json!({"phantom":"lamino_disk","size":16,"n_views":24,"geometry":"lamino","tilt":36.0,"seed":5})));
            state.queue_operation("simulate").unwrap();
            drive(&mut state, |s| {
                s.project
                    .as_ref()
                    .unwrap()
                    .document
                    .jobs
                    .iter()
                    .all(|job| !job.status.active())
                    && s.metadata_pending == 0
            });
            let project = state.project.as_ref().unwrap();
            let job = project.document.jobs.last().unwrap();
            assert_eq!(
                job.status,
                JobStatus::Completed,
                "laminography source: {:?}",
                job.error
            );
            job.result.clone().unwrap()
        } else {
            source_id.clone()
        };
        state.model.active_asset = Some(preview_source.clone());
        state.model.selected_assets = [preview_source].into_iter().collect();
        state.set_operation(operation, Some(&parameters));
        let jobs_before = state.project.as_ref().unwrap().document.jobs.len();
        state.queue_operation(operation).unwrap();
        drive(&mut state, |s| {
            s.model
                .preview
                .as_ref()
                .is_some_and(|preview| preview.can_keep)
        });
        assert_eq!(
            state.project.as_ref().unwrap().document.jobs.len(),
            jobs_before,
            "interactive edits must not create durable jobs"
        );
        state.keep_interactive().unwrap();
        drive(&mut state, |s| {
            s.project
                .as_ref()
                .unwrap()
                .document
                .jobs
                .iter()
                .all(|job| !job.status.active())
        });
        let project = state.project.as_ref().unwrap();
        let job = project.document.jobs.last().unwrap();
        assert_eq!(
            job.status,
            JobStatus::Completed,
            "{operation}: {:?}",
            job.error
        );
        let asset = project
            .document
            .asset(job.result.as_ref().unwrap())
            .unwrap();
        assert!(project.resolve(&asset.path).is_file());
        for array in asset.arrays.values() {
            assert!(
                matches!(&array.storage,crate::protocol::Storage::Npy{path,artifact:None} if !Path::new(path).is_absolute()&&project.resolve(Path::new(path)).is_file())
            );
        }
        let expected = if operation == "slice" {
            PlaneKind::Coronal
        } else {
            PlaneKind::Axial
        };
        assert_eq!(state.ui.main_view, expected);
        drive(&mut state, |s| s.imaging.plane(expected.mark()).is_some());
        if operation == "centre_sweep" {
            assert_eq!(state.slice_count(0), 3);
            state.ui.display.indices[0] = 2;
            state.refresh_images();
            drive(&mut state, |s| {
                s.imaging.plane(1).is_some_and(|plane| plane.index == 2)
            });
            assert!(
                state
                    .imaging
                    .plane(1)
                    .unwrap()
                    .fixed
                    .iter()
                    .any(|fixed| fixed.name == "centre" && fixed.coordinate == 1.0)
            );
            let input = state
                .project
                .as_ref()
                .unwrap()
                .capture(state.model.active_asset.as_ref().unwrap())
                .unwrap();
            state.queue_export(input, directory.path().join("registered-export"), "array");
            drive(&mut state, |s| {
                s.project
                    .as_ref()
                    .unwrap()
                    .document
                    .jobs
                    .iter()
                    .all(|job| !job.status.active())
            });
            assert_eq!(
                state
                    .project
                    .as_ref()
                    .unwrap()
                    .document
                    .jobs
                    .last()
                    .unwrap()
                    .status,
                JobStatus::Completed
            );
            assert!(
                directory
                    .path()
                    .join("registered-export/arrays.json")
                    .is_file()
            );
        }
    }
    state.model.active_asset = Some(volume_id.clone());
    state.model.selected_assets = [volume_id].into_iter().collect();
    state.set_operation(
        "reconstruct",
        Some(&json!({"method":"cgls","iterations":7})),
    );
    state.ui.display.indices = [3, 4, 5];
    state.ui.display.window_low.replace_text("0.05");
    state.ui.display.window_high.replace_text("1.2");
    state
        .ui
        .parameter_edits
        .get_mut("iterations")
        .unwrap()
        .replace_text("unfinished draft");
    state.shutdown();
    drop(state);
    let reopened = Project::open(directory.path(), false).unwrap();
    assert_eq!(reopened.document.workspace.indices, [3, 4, 5]);
    assert_eq!(
        reopened.document.workspace.draft_edits["iterations"],
        "unfinished draft"
    );
    assert_eq!(reopened.document.assets.len(), 6);
    assert!(
        reopened
            .document
            .jobs
            .iter()
            .all(|j| j.status == JobStatus::Completed)
    );
}

#[test]
#[ignore = "real worker and native preview interaction; set TOMOJAX_TEST_PYTHON"]
fn real_preview_slider_replaces_pixels_without_creating_durable_jobs() {
    let python = std::env::var_os("TOMOJAX_TEST_PYTHON").expect("set TOMOJAX_TEST_PYTHON");
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(directory.path(), true).unwrap());
    state.project.as_mut().unwrap().document.name = "Interactive preview verification".into();
    state.startup_input = None;
    state.auto_demo = false;
    state.started = true;
    state.engine = Some(Engine::spawn(PathBuf::from(python), std::sync::Arc::new(|| {})).unwrap());
    drive(&mut state, |state| state.model.backend.ready);
    state.set_operation("simulate", Some(&json!({"size":64,"n_views":96,"seed":3})));
    state.queue_operation("simulate").unwrap();
    drive(&mut state, |state| {
        state
            .project
            .as_ref()
            .unwrap()
            .document
            .jobs
            .iter()
            .all(|job| !job.status.active())
            && state.metadata_pending == 0
    });
    assert_eq!(
        state.project.as_ref().unwrap().document.jobs[0].status,
        JobStatus::Completed
    );
    let source = state.model.active_asset.clone();
    let jobs = state.project.as_ref().unwrap().document.jobs.len();
    state.set_operation(
        "slice",
        Some(&json!({"axis":"z","index":32,"method":"fourier"})),
    );
    state.request_interactive("slice").unwrap();
    drive(&mut state, |state| {
        state
            .model
            .preview
            .as_ref()
            .is_some_and(|preview| preview.can_keep && preview.count == Some(64))
            && state.imaging.plane(1).is_some()
    });
    // A normal text edit does not invoke a slider action or wait for poll().
    // Keep must still refuse the result for the value that was just replaced.
    state
        .ui
        .parameter_edits
        .get_mut("index")
        .unwrap()
        .replace_text("33");
    assert!(state.keep_interactive().is_err());
    assert_eq!(state.project.as_ref().unwrap().document.jobs.len(), jobs);
    assert!(!state.model.preview.as_ref().unwrap().has_image);
    state
        .ui
        .parameter_edits
        .get_mut("index")
        .unwrap()
        .replace_text("32");
    drive(&mut state, |state| {
        state
            .model
            .preview
            .as_ref()
            .is_some_and(|preview| preview.can_keep && preview.index == 32)
            && state.imaging.plane(1).is_some()
    });
    state.set_operation(
        "centre_sweep",
        Some(&json!({"axis":"z","index":32,"centres":[0.0],"method":"fourier"})),
    );
    drive(&mut state, |state| {
        state
            .model
            .preview
            .as_ref()
            .is_some_and(|preview| preview.can_keep && preview.operation == "centre_sweep")
    });
    state.set_operation(
        "slice",
        Some(&json!({"axis":"z","index":32,"method":"fourier"})),
    );
    state.model.active_asset = None;
    state.model.selected_assets.clear();
    assert!(
        state.keep_interactive().is_err(),
        "a cached result from the previous source must not be kept after switching control kinds"
    );
    assert_eq!(state.project.as_ref().unwrap().document.jobs.len(), jobs);
    state.model.active_asset = source.clone();
    state.model.selected_assets = source.iter().cloned().collect();
    drive(&mut state, |state| {
        state.model.preview.as_ref().is_some_and(|preview| {
            preview.can_keep && preview.operation == "slice" && preview.count == Some(64)
        }) && state.imaging.plane(1).is_some()
    });
    let first_request = state
        .previews
        .current(crate::previews::PreviewKind::Slice)
        .unwrap()
        .request
        .clone();
    let mut runtime = rune::Runtime::new(
        state,
        crate::ui::view,
        (1440., 900.),
        rune::Env::new(false, 1.),
    );
    runtime.frame(0.);
    runtime.update(|state, tx| {
        state
            .ui
            .parameter_edits
            .get_mut("index")
            .unwrap()
            .replace_text("unfinished");
        state.dispatch(UiAction::RefreshPreview, tx);
    });
    assert!(!runtime.state.model.preview.as_ref().unwrap().has_image);
    assert!(
        runtime.state.imaging.plane(1).is_none(),
        "Update with invalid text must unbind the previous result even when its intent signature was reset"
    );
    runtime.update(|state, tx| {
        state
            .ui
            .parameter_edits
            .get_mut("index")
            .unwrap()
            .replace_text("32");
        state.dispatch(UiAction::RefreshPreview, tx);
    });
    drive(&mut runtime.state, |state| {
        state
            .model
            .preview
            .as_ref()
            .is_some_and(|preview| preview.can_keep && preview.count == Some(64))
            && state.imaging.plane(1).is_some()
    });
    runtime.invalidate();
    runtime.frame(1. / 60.);
    let first_view_slider = runtime
        .elements()
        .into_iter()
        .find(|element| element.key.as_deref() == Some("preview-index"))
        .expect("slice slider before scrolling");
    assert!(
        first_view_slider.visible[3] > 0.,
        "interactive control hidden below the form: {first_view_slider:?}"
    );
    runtime.update(|_, tx| tx.scroll_into_view("preview-index"));
    runtime.frame(1. / 60.);
    runtime.frame(1. / 60.);
    let slider = runtime
        .elements()
        .into_iter()
        .find(|element| element.key.as_deref() == Some("preview-index"))
        .expect("slice slider");
    assert!(slider.focusable && slider.visible[3] > 0., "{slider:?}");
    let x = slider.rect[0] + slider.rect[2] * 0.25;
    let y = slider.rect[1] + slider.rect[3] * 0.5;
    runtime.pointer_move(x, y);
    runtime.pointer_down(x, y);
    runtime.pointer_move(slider.rect[0] + slider.rect[2] * 0.75, y);
    runtime.pointer_up(slider.rect[0] + slider.rect[2] * 0.75, y);
    runtime.frame(1. / 60.);
    let expected = runtime.state.ui.parameter_edits["index"]
        .text
        .parse::<usize>()
        .unwrap();
    assert_ne!(
        expected, 32,
        "the native slider did not update the worker parameter"
    );
    assert!(
        !runtime.state.model.preview.as_ref().unwrap().has_image,
        "old pixels must disappear as soon as a newer edit exists"
    );
    assert!(
        runtime.state.imaging.plane(1).is_none(),
        "a stale plane remained registered to the latest control value"
    );
    let mut state = std::mem::replace(&mut runtime.state, AppState::init());
    drop(runtime);
    drive(&mut state, |state| {
        state
            .model
            .preview
            .as_ref()
            .is_some_and(|preview| preview.can_keep && preview.index == expected)
            && state.imaging.plane(1).is_some()
    });
    let latest = state
        .previews
        .current(crate::previews::PreviewKind::Slice)
        .unwrap();
    assert_ne!(latest.request, first_request);
    assert_eq!(latest.parameters["index"], json!(expected));
    assert_eq!(state.model.active_asset, source);
    assert_eq!(
        state.project.as_ref().unwrap().document.jobs.len(),
        jobs,
        "dragging must not queue durable jobs"
    );
    assert!(
        state
            .imaging
            .plane(1)
            .unwrap()
            .fixed
            .iter()
            .any(|fixed| fixed.name == "z")
    );
    if let Some(evidence) = std::env::var_os("TOMOJAX_PREVIEW_UI_EVIDENCE") {
        let evidence = PathBuf::from(evidence);
        std::fs::create_dir_all(&evidence).unwrap();
        std::fs::write(evidence.join("preview.json"),serde_json::to_vec_pretty(&json!({
            "operation":latest.operation,"parameters":latest.parameters,"request":latest.request,"revision":latest.revision,
            "output":latest.output,"input":latest.input,"durable_jobs_before":jobs,"durable_jobs_after":state.project.as_ref().unwrap().document.jobs.len(),
            "snapshot_note":"Native slider input followed by an actual current worker result. Temporary artifacts belong to the test session."
        })).unwrap()).unwrap();
        capture_native(state, &evidence.join("preview.png"));
    } else {
        state.shutdown();
    }
}

#[test]
fn source_copy_never_recurses_into_its_project_or_overwrites_existing_data() {
    let cancelled = std::sync::atomic::AtomicBool::new(false);
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("acquisition");
    std::fs::create_dir(&source).unwrap();
    std::fs::write(source.join("view.tif"), b"pixels").unwrap();
    assert!(stage_copy_input(&source, &source.join("project").join("copied"), &cancelled).is_err());
    let destination = directory.path().join("managed");
    stage_copy_input(&source, &destination, &cancelled).unwrap();
    assert_eq!(
        std::fs::read(destination.join("view.tif")).unwrap(),
        b"pixels"
    );
    assert!(stage_copy_input(&source, &destination, &cancelled).is_err());
    assert_eq!(std::fs::read(source.join("view.tif")).unwrap(), b"pixels");
}

#[test]
fn copying_a_parameter_file_brings_its_frames_and_a_container_comes_alone() {
    let directory = tempfile::tempdir().unwrap();
    let nikon = directory.path().join("nikon-scan");
    let hdf5 = directory.path().join("beamline");
    for folder in [&nikon, &hdf5] {
        std::fs::create_dir(folder).unwrap();
    }
    std::fs::write(nikon.join("scan.xtekct"), b"[XTekCT]").unwrap();
    std::fs::write(nikon.join("scan_0001.tif"), b"pixels").unwrap();
    std::fs::write(hdf5.join("scan.nxs"), b"container").unwrap();
    std::fs::write(hdf5.join("unrelated.nxs"), b"other scan").unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(&directory.path().join("project"), true).unwrap());
    state.ui.import_copy = true;
    state.import_paths(vec![nikon.join("scan.xtekct"), hdf5.join("scan.nxs")]);
    drive(&mut state, |state| {
        state.copying == 0 && state.project.as_ref().unwrap().document.assets.len() == 2
    });
    let project = state.project.as_ref().unwrap();
    let copied: Vec<_> = project
        .document
        .assets
        .iter()
        .map(|asset| project.resolve(&asset.path))
        .collect();
    let xtekct = copied.iter().find(|p| p.ends_with("scan.xtekct")).unwrap();
    assert!(xtekct.starts_with(&project.root));
    assert_eq!(
        std::fs::read(xtekct.with_file_name("scan_0001.tif")).unwrap(),
        b"pixels"
    );
    let nxs = copied.iter().find(|p| p.ends_with("scan.nxs")).unwrap();
    assert!(nxs.starts_with(&project.root));
    assert!(!nxs.with_file_name("unrelated.nxs").exists());
}

#[test]
fn hdf5_copies_only_files_that_hold_all_their_data() {
    let directory = tempfile::tempdir().unwrap();
    let raw = directory.path().join("raw");
    std::fs::create_dir_all(&raw).unwrap();
    let pixels = raw.join("pixels.h5");
    hdf5::File::create(&pixels)
        .unwrap()
        .new_dataset_builder()
        .with_data(&[1.0f32, 2., 3.])
        .create("data")
        .unwrap();
    // An NXtomo master that reaches its pixels through an external link.
    hdf5::File::create(raw.join("linked.nxs"))
        .unwrap()
        .link_external("pixels.h5", "/data", "data")
        .unwrap();
    // A virtual dataset assembled from the detector file.
    hdf5::File::create(raw.join("virtual.nxs"))
        .unwrap()
        .new_dataset::<f32>()
        .shape(3)
        .with_dcpl(|plist| plist.virtual_map("pixels.h5", "/data", 3, .., 3, ..))
        .create("data")
        .unwrap();
    // Raw pixels kept in a sidecar file.
    std::fs::write(raw.join("pixels.raw"), [0u8; 12]).unwrap();
    hdf5::File::create(raw.join("external.h5"))
        .unwrap()
        .new_dataset::<f32>()
        .shape(3)
        .external("pixels.raw", 0, 12)
        .create("data")
        .unwrap();

    let mut state = AppState::init();
    state.project = Some(Project::open(&directory.path().join("project"), true).unwrap());
    state.ui.import_copy = true;
    state.import_paths(vec![pixels.clone()]);
    drive(&mut state, |state| {
        state.copying == 0 && !state.project.as_ref().unwrap().document.assets.is_empty()
    });
    let project = state.project.as_ref().unwrap();
    assert!(
        project
            .resolve(&project.document.assets[0].path)
            .starts_with(&project.root)
    );

    for refused in ["linked.nxs", "virtual.nxs", "external.h5"] {
        state.model.error = None;
        state.import_paths(vec![raw.join(refused)]);
        drive(&mut state, |state| state.copying == 0);
        assert_eq!(state.project.as_ref().unwrap().document.assets.len(), 1);
        assert!(
            state
                .model
                .error
                .as_ref()
                .is_some_and(|e| e.contains("can't be copied")),
            "{refused}: {:?}",
            state.model.error
        );
    }
}

#[test]
fn npz_is_rejected_before_a_worker_or_copy_task_is_started() {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(directory.path(), true).unwrap());
    state.import_paths(vec![directory.path().join("untrusted.NPZ")]);
    assert!(state.project.as_ref().unwrap().document.assets.is_empty());
    assert_eq!(state.copying, 0);
    assert!(state.pending.is_empty());
    assert!(
        state
            .model
            .error
            .as_ref()
            .is_some_and(|e| e.contains("NPZ"))
    );
}

#[test]
fn changed_input_is_rejected_before_a_computed_result_is_accepted() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("acquisition.bin");
    std::fs::write(&source, b"original data").unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(&directory.path().join("project"), true).unwrap());
    let project = state.project.as_mut().unwrap();
    let asset = Asset::input(source.clone(), false);
    let asset_id = asset.id.clone();
    project.document.assets.push(asset);
    let mut job = project::Job::new(
        "validate",
        json!({}),
        vec![project.capture(&asset_id).unwrap()],
    );
    job.status = JobStatus::Running;
    let id = job.id.clone();
    project.document.jobs.push(job);
    std::fs::write(&source, b"changed after numerical execution").unwrap();
    state.result(
        Pending::Execute(id.clone()),
        ResultValue::Output(Box::new(Output {
            value: Product::Validated,
            info: crate::protocol::Info {
                summary: Default::default(),
                arrays: Default::default(),
            },
            manifest: crate::protocol::Provenance {
                operation: "validate".into(),
                parameters: json!({}),
                inputs: vec![],
            },
        })),
    );
    drive(&mut state, |s| {
        s.project.as_ref().unwrap().document.jobs[0].status == JobStatus::Failed
    });
    let project = state.project.as_ref().unwrap();
    assert!(
        project.document.jobs[0]
            .error
            .as_ref()
            .unwrap()
            .contains("changed after submission")
    );
    assert!(project.document.jobs[0].result.is_none());
    assert_eq!(project.document.assets.len(), 1);
    state.shutdown();
}

#[test]
fn reopening_uses_the_saved_view_or_centres_the_newest_result() {
    let (directory, mut original) = registered_projection_fixture();
    let document = &mut original.project.as_mut().unwrap().document;
    let scan = document.assets[0].clone();
    let mut result = scan.clone();
    result.id = "volume-result".into();
    result.parent = Some(scan.id.clone());
    let mut volume = result.arrays.remove("projections").unwrap();
    for (axis, name) in volume.axes.iter_mut().zip(["z", "y", "x"]) {
        axis.name = name.into();
        axis.frame = "volume".into();
        axis.unit = "mm".into();
    }
    result.arrays.insert("volume".into(), volume);
    document.assets.push(result.clone());
    document.assets[0].created = result.created + 1;
    document.assets[0].provenance = Some(crate::protocol::Provenance {
        operation: "open".into(),
        parameters: json!({}),
        inputs: vec![],
    });
    original.shutdown();
    drop(original);
    for active in [None, Some("removed"), Some(&scan.id)] {
        let mut project = Project::open(directory.path(), false).unwrap();
        project.document.workspace.active = active.map(str::to_owned);
        project.document.workspace.view = "Sinogram".into();
        project.document.workspace.sinogram = 1;
        project.changed().unwrap();
        project.flush().unwrap();
        drop(project);
        let mut reopened = AppState::init();
        reopened.open_project(directory.path(), false).unwrap();
        if active == Some(scan.id.as_str()) {
            assert_eq!(reopened.model.active_asset.as_ref(), Some(&scan.id));
            assert_eq!(reopened.ui.main_view, PlaneKind::Sinogram);
            assert_eq!(reopened.ui.display.sinogram_row, 1);
        } else {
            assert_eq!(reopened.model.active_asset.as_ref(), Some(&result.id));
            assert!(reopened.model.selected_assets.contains(&result.id));
            assert_eq!(reopened.ui.main_view, PlaneKind::Axial);
            assert_eq!(reopened.ui.display.indices, [2, 2, 4]);
        }
        reopened.shutdown();
    }
}
