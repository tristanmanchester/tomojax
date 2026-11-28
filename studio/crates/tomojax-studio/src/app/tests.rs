use super::*;

/// Freeze an observed application state with the same renderer as the window.
/// Only tests opt in to evidence capture; ordinary --snap still opens a saved
/// project without starting a numerical worker.
pub(crate) fn capture_native(state: AppState, path: &Path) {
    capture_native_viewport(state, path, "1440x900", "1");
}

pub(crate) fn capture_native_viewport(mut state: AppState, path: &Path, size: &str, scale: &str) {
    // Match the ordinary actions a reviewer would take before inspecting the
    // result. Numerical values and observed worker state remain untouched.
    state.ui.activity.close();
    render_native_viewport(state, path, size, scale);
}

pub(crate) fn render_native_viewport(mut state: AppState, path: &Path, size: &str, scale: &str) {
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
        crate::volume::ScientificCompositor::new(AppState::render_panes),
    );
    let tree = path.with_extension("tree.json");
    let args = rune::tooling::Args::parse_from(vec![
        "--snap".into(),
        path.display().to_string(),
        "--tree".into(),
        tree.display().to_string(),
        "--size".into(),
        size.into(),
        "--scale".into(),
        scale.into(),
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
    state.model.jobs[1].can_pause = true;
    assert_eq!(state.preview_blocker().unwrap().id, "device-run");
    state.model.jobs[1].can_pause = false;
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
        state.imaging.plane(4).is_some_and(|p| p.lossless)
            && state.imaging.plane(5).is_some_and(|p| p.lossless)
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
        state
            .imaging
            .plane(5)
            .is_some_and(|plane| plane.index == 0 && plane.lossless)
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
    state.preview_context.shown = super::previewing::Shown::Result {
        request: "preview-1".into(),
        arrays: [("after".into(), array)].into(),
    };
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
    assert!(state.preview_context.shown.arrays().is_none());
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
        state.window.field(0).text,
        crate::ui::compact_number(f64::from(current[0]))
    );
    assert_eq!(
        state.window.field(1).text,
        crate::ui::compact_number(f64::from(current[1]))
    );
    state.window.field_mut(0).replace_text("-");
    state.window.field_mut(1).replace_text("2.5");
    state.refresh_plane_model();
    assert_eq!(state.window.field(0).text, "-");
    assert_eq!(state.window.field(1).text, "2.5");
    assert!(state.window.applied().is_none());
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
    // The shared slice count is known from the registrations before any image loads.
    assert_eq!(state.compared_count(0), Some(3));
    state.ui.display.indices[0] = 5;
    state.refresh_images();
    drive(&mut state, |state| {
        state.imaging.plane(6).is_some_and(|p| p.lossless)
            && state.imaging.plane(7).is_some_and(|p| p.lossless)
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
fn region_statistics_use_only_samples_whose_centers_are_inside_the_box() {
    let directory = tempfile::tempdir().unwrap();
    let volume = ramp_volume(directory.path(), "volume", 3);
    let mut state = AppState::init();
    state.project = Some(Project::open(directory.path(), true).unwrap());
    state.model.active_asset = Some(volume.id.clone());
    state.project.as_mut().unwrap().document.assets = vec![volume];
    state.ui.main_view = PlaneKind::Axial;
    state.refresh_images();
    drive(&mut state, |state| {
        state.imaging.plane(1).is_some_and(|p| p.lossless)
    });
    state.ui.inspect_tool = InspectTool::Region;
    // Samples 0.2–1.8 on a 4 × 4 slice hold one sample center, (1, 1).
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
    // A box in the outer half of the first sample holds no sample center.
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
    // A corrections result, registered by the worker.
    let mut result = ramp_volume(directory.path(), "corrected", 2);
    result.parent = Some("source-scan".into());
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
        inspect_if_unknown: true,
        intent: 0,
        asset: id.clone(),
        result: Ok(Some(("scan".into(), arrays, json!({})))),
    });
    let asset = state.project.as_ref().unwrap().document.asset(&id).unwrap();
    assert_eq!(asset.arrays["projections"].domain, "line_integrals");

    // A worker result keeps the kind the worker gave it.
    let project = state.project.as_mut().unwrap();
    let asset = project.document.asset_mut(&id).unwrap();
    asset.kind = "scan".into();
    state.local_event(LocalEvent::Metadata {
        inspect_if_unknown: true,
        intent: 0,
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
    state
        .queue_export(input, directory.path().join("export.nxs"), "nxs")
        .unwrap();
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
fn corrections_edited_as_json_survive_reopening() {
    let mut state = AppState::init();
    state.set_operation("correct", None);
    let edit = |state: &mut AppState, text: &str| {
        state
            .ui
            .parameter_edits
            .get_mut("corrections")
            .unwrap()
            .replace_text(text);
    };
    edit(&mut state, r#"[{"type":"Zingers","threshold":0.1}]"#);
    crate::ui::corrections::reset(&mut state.ui, &state.catalog);
    state.ui.corrections.json_mode = true;
    edit(&mut state, r#"[{"type":"Zingers","threshold":0.3}]"#);
    let context = state.capture_workspace();
    state.restore_workspace(&context);
    assert!(state.ui.corrections.json_mode);
    let parameters = crate::ui::corrections::parameters_detailed(&state.ui).unwrap();
    assert_eq!(parameters[0]["threshold"], 0.3);
}

#[test]
fn a_back_projection_belongs_to_the_projections_it_used() {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(&directory.path().join("project"), true).unwrap());
    let mut scan = ramp_volume(directory.path(), "scan", 2);
    let mut projected = ramp_volume(directory.path(), "projections", 2);
    scan.kind = "scan".into();
    projected.kind = "projections".into();
    projected
        .arrays
        .insert("projections".into(), scan.arrays["volume"].clone());
    let project = state.project.as_mut().unwrap();
    project.document.assets = vec![scan.clone(), projected.clone()];
    let mut job = project::Job::new(
        "backproject",
        json!({}),
        vec![
            project.capture(&scan.id).unwrap(),
            project.capture(&projected.id).unwrap(),
        ],
    );
    job.script = Some(project::script::Script::build(&job, false, String::new(), None).unwrap());
    let id = job.id.clone();
    project.document.jobs.push(job);
    let row=serde_json::from_value(json!({"directory":project.job_directory(&id),"state":"completed","launcher":"process","detail":"","cancel_requested":false,"receipt":{
        "type":"arrays","product":"volume","arrays":{"volume":scan.arrays["volume"]},"manifest":{"operation":"backproject","parameters":{},"inputs":[]}}})).unwrap();
    let asset = project
        .apply_job(&id, 0, &row, None)
        .unwrap()
        .asset
        .unwrap();
    assert_eq!(
        project.document.asset(&asset).unwrap().parent.as_ref(),
        Some(&projected.id)
    );
}

#[test]
fn export_offers_what_each_kind_of_result_can_be_written_as() {
    let mut state = AppState::init();
    state.model.active_asset = Some("a".into());
    for (kind, selected, expected) in [
        (Export::Dataset, "ImageJ TIFF", "ImageJ TIFF"),
        (Export::Dataset, "OME-Zarr 0.5", "nxs"),
        (Export::Volume, "OME-Zarr 0.5", "OME-Zarr 0.5"),
        (Export::Volume, "array", "nxs"),
        (Export::Array, "ImageJ TIFF", "array"),
        (Export::Unavailable, "nxs", ""),
    ] {
        state.model.assets = vec![AssetView {
            id: "a".into(),
            export: kind,
            ..Default::default()
        }];
        state.ui.export_format = selected.into();
        assert_eq!(state.active_export_format(), expected);
    }
}

#[test]
fn importing_a_scan_clears_the_previous_scans_pixels() {
    let directory = tempfile::tempdir().unwrap();
    let volume = ramp_volume(directory.path(), "volume", 3);
    let mut state = AppState::init();
    state.project = Some(Project::open(&directory.path().join("project"), true).unwrap());
    state.model.active_asset = Some(volume.id.clone());
    state.project.as_mut().unwrap().document.assets = vec![volume];
    state.ui.main_view = PlaneKind::Axial;
    state.refresh_images();
    drive(&mut state, |state| {
        state.imaging.plane(1).is_some_and(|p| p.lossless)
    });
    let tiffs = directory.path().join("tiffs");
    std::fs::create_dir(&tiffs).unwrap();
    std::fs::write(tiffs.join("view_0001.tif"), b"not yet inspected").unwrap();
    state.import_paths(vec![tiffs]);
    assert!(state.imaging.plane(1).is_none());
}

#[test]
fn a_copied_import_opens_with_its_own_kind_and_a_fresh_window() {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(&directory.path().join("project"), true).unwrap());
    // The previous asset: a reconstruction viewed through a narrow window.
    let mut previous = ramp_volume(directory.path(), "previous", 2);
    previous.kind = "reconstruction".into();
    let previous_id = previous.id.clone();
    state
        .project
        .as_mut()
        .unwrap()
        .document
        .assets
        .push(previous);
    state.model.active_asset = Some(previous_id);
    state.set_operation("open", None);
    assert_eq!(state.ui.parameter_choices["kind"], "reconstruction");
    state.window.restore(Some([0., 0.01]));
    let copy = directory.path().join("copied.nxs");
    std::fs::write(&copy, b"copied scan").unwrap();
    let mut scan = Asset::input(copy.clone(), true);
    scan.kind = "scan".into();
    let id = scan.id.clone();
    let selection = state.selected_inputs();
    state.local_event(LocalEvent::Copied {
        asset: scan,
        result: Ok(copy),
        selection,
    });
    assert_eq!(state.model.active_asset.as_deref(), Some(id.as_str()));
    assert_eq!(state.window.applied(), None);
    assert_eq!(state.ui.parameter_choices["kind"], "scan");

    // A copy that finishes after the user moved on is added, not opened.
    let started = state.selected_inputs();
    let other = directory.path().join("other.nxs");
    std::fs::write(&other, b"another scan").unwrap();
    state.model.selected_assets.clear();
    state.model.active_asset = None;
    state.local_event(LocalEvent::Copied {
        asset: Asset::input(other.clone(), true),
        result: Ok(other),
        selection: started,
    });
    assert_eq!(state.model.active_asset, None);
    assert_eq!(state.project.as_ref().unwrap().document.assets.len(), 3);
}

#[test]
fn a_read_updates_the_calibration_kind_and_opens_a_volume_on_its_slices() {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(&directory.path().join("project"), true).unwrap());
    let volume = ramp_volume(directory.path(), "volume", 4);
    let arrays = volume.arrays.clone();
    let mut imported = Asset::at(volume.path.clone(), false, false);
    imported.kind = "scan".into();
    let id = imported.id.clone();
    state
        .project
        .as_mut()
        .unwrap()
        .document
        .assets
        .push(imported);
    state.model.selected_assets = [id.clone()].into();
    state.activate(id.clone());
    state.set_operation("open", None);
    assert_eq!(state.ui.parameter_choices["kind"], "scan");
    // The read finds a reconstruction holding only a volume.
    state.record_read(&id, Ok(("reconstruction".into(), arrays)), json!({}));
    assert_eq!(state.ui.parameter_choices["kind"], "reconstruction");
    assert_ne!(state.ui.main_view, PlaneKind::Projection);
    assert_eq!(state.ui.display.indices[0], 2);
}

#[test]
fn compare_reopens_on_its_plane_and_mode() {
    let mut state = AppState::init();
    state.ui.compare_plane = PlaneKind::Coronal;
    state.ui.compare_mode = CompareMode::Difference;
    let context = state.capture_workspace();
    state.ui.compare_plane = PlaneKind::Axial;
    state.ui.compare_mode = CompareMode::Pair;
    state.restore_workspace(&context);
    assert_eq!(state.ui.compare_plane, PlaneKind::Coronal);
    assert_eq!(state.ui.compare_mode, CompareMode::Difference);
}

#[test]
fn a_reconnected_result_keeps_its_registration_on_the_new_file() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("project");
    let mut result = ramp_volume(directory.path(), "result", 2);
    result.parent = Some("source".into());
    // A companion coordinate file beside the result.
    let axis = directory.path().join("result-axis0.npy");
    result.arrays.get_mut("volume").unwrap().axes[0].coordinates =
        crate::protocol::Coordinates::Coordinates {
            values: crate::protocol::Storage::Npy {
                path: axis.to_string_lossy().into(),
                artifact: None,
            },
        };
    let elsewhere = directory.path().join("elsewhere");
    let moved = elsewhere.join("result.npy");
    result.relink(moved.clone(), &root);
    let npy = |path: std::path::PathBuf| crate::protocol::Storage::Npy {
        path: path.to_string_lossy().into(),
        artifact: None,
    };
    let volume = &result.arrays["volume"];
    assert_eq!(volume.storage, npy(moved.clone()));
    // Reconnected inside the project, its paths stay relative so the project can move.
    let mut inside = result.clone();
    inside.relink(root.join("derived/result.npy"), &root);
    assert_eq!(
        inside.arrays["volume"].storage,
        npy("derived/result.npy".into())
    );
    // Whether a source is a folder is recorded at import, whatever its name.
    let folder = directory.path().join("sample.raw");
    std::fs::create_dir(&folder).unwrap();
    assert!(Asset::input(folder, false).folder);
    assert!(matches!(
        &volume.axes[0].coordinates,
        crate::protocol::Coordinates::Coordinates { values } if *values == npy(elsewhere.join("result-axis0.npy"))
    ));
    // An import's registration is read again from the new file.
    let mut import = ramp_volume(directory.path(), "import", 2);
    import.relink(moved, &root);
    assert!(import.arrays.is_empty());
}

#[test]
fn activating_the_active_asset_keeps_its_calibration_draft() {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(&directory.path().join("project"), true).unwrap());
    let scan = ramp_volume(directory.path(), "scan", 2);
    let id = scan.id.clone();
    state.project.as_mut().unwrap().document.assets.push(scan);
    state.activate(id.clone());
    state.set_operation("open", None);
    state
        .ui
        .parameter_edits
        .get_mut("pixel_size")
        .unwrap()
        .replace_text("0.65");
    state.activate(id);
    assert_eq!(state.ui.parameter_edits["pixel_size"].text, "0.65");
}

#[test]
fn slice_shortcuts_step_the_plane_on_screen() {
    let mut state = AppState::init();
    state.ui.main_view = PlaneKind::Axial;
    assert_eq!(state.shown_axis(), Some(0));
    state.ui.workspace = Workspace::Compare;
    state.ui.compare_plane = PlaneKind::Coronal;
    assert_eq!(state.shown_axis(), Some(1));
}

#[test]
fn the_zoom_readout_is_the_zoom_the_3d_camera_shows() {
    let mut state = AppState::init();
    state.ui.main_view = PlaneKind::Volume;
    state.ui.display.zoom = 10.;
    assert_eq!(state.shown_zoom(), crate::volume::ZOOM[1]);
    state.ui.main_view = PlaneKind::Axial;
    assert_eq!(state.shown_zoom(), 10.);
}

#[test]
fn view_undo_stays_with_its_project() {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state
        .open_project(&directory.path().join("first"), true)
        .unwrap();
    let context = state.capture_workspace();
    state.view_history.push(context);
    state
        .open_project(&directory.path().join("second"), true)
        .unwrap();
    assert!(state.view_history.is_empty());
}

#[test]
fn projection_roles_name_the_geometry_scan_and_the_array_input() {
    use project::script::ProjectionRoles;
    let directory = tempfile::tempdir().unwrap();
    let saved = ramp_volume(directory.path(), "forward", 2).arrays["volume"].clone();
    let input = |kind: &str, keys: &[&str]| {
        let mut asset = Asset::input("input.nxs".into(), false);
        asset.kind = kind.into();
        for key in keys {
            asset.arrays.insert((*key).into(), saved.clone());
        }
        project::CapturedInput::from_asset(&asset, directory.path())
    };
    let selected = input("projections", &["projections"]);
    let first = input("scan", &["projections"]);
    let second = first.clone();
    let reconstruction = input("reconstruction", &["volume", "projections"]);
    let simulation = input("simulation", &["projections", "phantom"]);
    let roles = |op: &str, products: &[&project::CapturedInput]| {
        let inputs: Vec<_> = products.iter().map(|i| (*i).clone()).collect();
        let r = ProjectionRoles::choose(op, &inputs, &json!({})).unwrap();
        (r.geometry, r.array)
    };
    // Back projection: a selected projection result wherever it sits, else the
    // projections of the scan that supplies the geometry.
    assert_eq!(roles("backproject", &[&selected, &first]), (1, Some(0)));
    assert_eq!(roles("backproject", &[&first, &selected]), (0, Some(1)));
    assert_eq!(roles("backproject", &[&first, &second]), (0, Some(0)));
    // Forward projection: a reconstruction into another selected scan, in either
    // order; into its own scan when it is alone; a reconstruction before a
    // simulation's ground-truth phantom.
    assert_eq!(roles("project", &[&reconstruction, &second]), (1, Some(0)));
    assert_eq!(roles("project", &[&second, &reconstruction]), (0, Some(1)));
    assert_eq!(roles("project", &[&reconstruction]), (0, Some(0)));
    assert_eq!(
        roles("project", &[&simulation, &reconstruction]),
        (0, Some(1))
    );
    // An array given by reference is used as given and belongs to no input.
    let referenced = ProjectionRoles::choose(
        "backproject",
        std::slice::from_ref(&first),
        &json!({"projections": {"type": "npy", "path": "/residuals.npy"}}),
    )
    .unwrap();
    assert_eq!(
        (referenced.geometry, referenced.array, referenced.parent()),
        (0, None, None)
    );

    // Forward projection keeps the volume's own registered grid.
    let mut volume = saved.clone();
    for (axis, origin) in volume.axes.iter_mut().zip([-1.5, 2., 0.25]) {
        axis.coordinates = crate::protocol::Coordinates::Uniform {
            origin,
            spacing: 0.5,
        };
    }
    let mut v = input("volume", &[]);
    v.arrays.insert("volume".into(), volume);
    let mut operation = json!({});
    ProjectionRoles::choose("project", &[first, v], &json!({"grid":null}))
        .unwrap()
        .apply(&mut operation, |_| unreachable!());
    assert_eq!(
        operation["volume"],
        serde_json::to_value(&saved.storage).unwrap()
    );
    assert_eq!(
        operation["grid"],
        json!({"nx":4,"ny":4,"nz":2,"vx":0.5,"vy":0.5,"vz":0.5,"vol_origin":[0.25,2.,-1.5]})
    );
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
    state.demo = crate::app::Demo::Off;
    state.engine = crate::app::session::Session::local(
        Engine::spawn_through(PathBuf::from(python), &[], std::sync::Arc::new(|| {})).unwrap(),
    );
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
    assert!(
        project
            .resolve(&project.document.assets[0].path)
            .lexical()
            .is_file()
    );

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
            "center_sweep",
            json!({"axis":"z","index":8,"centers":[-1.0,0.0,1.0],"method":"fourier"}),
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
        assert!(project.resolve(&asset.path).lexical().is_file());
        for array in asset.arrays.values() {
            assert!(
                matches!(&array.storage,crate::protocol::Storage::Npy{path,artifact:None} if !Path::new(path).is_absolute()&&project.resolve_storage(Path::new(path)).is_file())
            );
        }
        let expected = if operation == "slice" {
            PlaneKind::Coronal
        } else {
            PlaneKind::Axial
        };
        assert_eq!(state.ui.main_view, expected);
        drive(&mut state, |s| s.imaging.plane(expected.mark()).is_some());
        if operation == "center_sweep" {
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
                    .any(|fixed| fixed.name == "center" && fixed.coordinate == 1.0)
            );
            let input = state
                .project
                .as_ref()
                .unwrap()
                .capture(state.model.active_asset.as_ref().unwrap())
                .unwrap();
            state
                .queue_export(input, directory.path().join("registered-export"), "array")
                .unwrap();
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
    state.window.field_mut(0).replace_text("0.05");
    state.window.field_mut(1).replace_text("1.2");
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
    state.demo = crate::app::Demo::Off;
    state.engine = crate::app::session::Session::local(
        Engine::spawn_through(PathBuf::from(python), &[], std::sync::Arc::new(|| {})).unwrap(),
    );
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
        "center_sweep",
        Some(&json!({"axis":"z","index":32,"centers":[0.0],"method":"fourier"})),
    );
    drive(&mut state, |state| {
        state
            .model
            .preview
            .as_ref()
            .is_some_and(|preview| preview.can_keep && preview.operation == "center_sweep")
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
        .map(|asset| project.resolve(&asset.path).lexical().to_owned())
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
            .lexical()
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
fn reopening_uses_the_saved_view_or_centers_the_newest_result() {
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

#[test]
fn the_palette_opens_every_operation_in_its_own_workspace() {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state
        .open_project(&directory.path().join("project"), true)
        .unwrap();
    state.refresh_model();
    let align = state
        .model
        .commands
        .iter()
        .find(|command| command.id == "operation:align")
        .expect("Alignment is a command");
    assert_eq!((align.title.as_str(), align.enabled), ("Alignment", true));
    let mut runtime = rune::Runtime::new(
        state,
        crate::ui::view,
        (1440., 900.),
        rune::Env::new(false, 1.),
    );
    runtime.update(|state, tx| {
        state.dispatch(UiAction::RunCommand("operation:align".into()), tx);
    });
    assert_eq!(runtime.state.ui.workspace, Workspace::Prepare);
    assert_eq!(runtime.state.ui.operation, "align");
    runtime.update(|state, tx| {
        state.dispatch(UiAction::RunCommand("operation:slice".into()), tx);
    });
    assert_eq!(runtime.state.ui.workspace, Workspace::Reconstruct);
    assert_eq!(runtime.state.ui.operation, "slice");
    // Each processing workspace returns to the operation it last showed.
    runtime.update(|state, tx| state.dispatch(UiAction::SetWorkspace(Workspace::Prepare), tx));
    assert_eq!(runtime.state.ui.operation, "align");

    runtime
        .state
        .ui
        .parameter_edits
        .get_mut("iterations")
        .expect("Alignment has an iteration limit")
        .replace_text("21");
    // Reopening it from the palette keeps the settings being edited, and shows them.
    runtime.update(|state, tx| state.dispatch(UiAction::SetWorkspace(Workspace::Explore), tx));
    runtime.state.ui.inspector_open = false;
    runtime.update(|state, tx| {
        state.dispatch(UiAction::RunCommand("operation:align".into()), tx);
    });
    assert_eq!(runtime.state.ui.parameter_edits["iterations"].text, "21");
    assert!(runtime.state.ui.inspector_open);

    // The simulation shortcut keeps a simulation draft too.
    runtime.update(|state, tx| {
        state.dispatch(UiAction::RunCommand("operation:simulate".into()), tx);
    });
    runtime
        .state
        .ui
        .parameter_edits
        .get_mut("size")
        .expect("Simulation has a size")
        .replace_text("37");
    runtime.update(|state, tx| state.dispatch(UiAction::SetWorkspace(Workspace::Explore), tx));
    runtime.update(|state, tx| state.dispatch(UiAction::RunCommand("simulate".into()), tx));
    assert_eq!(runtime.state.ui.operation, "simulate");
    assert_eq!(runtime.state.ui.parameter_edits["size"].text, "37");
}

#[test]
fn activity_closes_after_the_work_it_opened_for_succeeds_and_shows_a_failure() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("project");
    let open = |create: bool| {
        let mut state = AppState::init();
        state.open_project(&path, create).unwrap();
        state
    };
    // Queues a job, opens Activity for it, then ends the job with `status`.
    let run = |state: &mut AppState, status: JobStatus, close_early: bool| {
        let job = project::Job::new("simulate", serde_json::json!({}), vec![]);
        let id = job.id.clone();
        state.project.as_mut().unwrap().document.jobs.push(job);
        state.ui.activity.follow(vec![id.clone()]);
        state.refresh_model();
        assert!(
            state.ui.activity.is_open(),
            "Activity stays open while its job is queued"
        );
        if close_early {
            state.ui.activity.close();
        }
        let job = state
            .project
            .as_mut()
            .unwrap()
            .document
            .jobs
            .iter_mut()
            .find(|job| job.id == id)
            .unwrap();
        job.status = status;
        if status == JobStatus::Failed {
            job.error = Some("Reconstruction ran out of GPU memory.".into());
        }
        state.refresh_model();
        id
    };
    // Renders `state` and checks the failed job is listed with its error.
    let shows_failure = |state: AppState, failed: &str| {
        let mut runtime = rune::Runtime::new(
            state,
            crate::ui::view,
            (1440., 900.),
            rune::Env::new(false, 1.),
        );
        runtime.frame(0.);
        runtime.frame(1. / 60.);
        let elements = runtime.elements();
        elements
            .iter()
            .any(|element| element.key.as_deref() == Some(&format!("job-{failed}")))
            && elements.iter().any(|element| {
                element.text.as_deref() == Some("Reconstruction ran out of GPU memory.")
            })
    };

    let mut state = open(true);
    run(&mut state, JobStatus::Completed, false);
    assert!(!state.ui.activity.is_open(), "closes once its job succeeds");
    let failed = run(&mut state, JobStatus::Failed, false);
    assert!(state.ui.activity.is_open(), "a failure keeps Activity open");
    assert!(shows_failure(state, &failed));

    // Activity the user opened is theirs: success never closes it, a failure is listed.
    let mut state = open(false);
    state.ui.activity.show(Vec::new());
    run(&mut state, JobStatus::Completed, false);
    assert!(
        state.ui.activity.is_open(),
        "the user's Activity stays open"
    );
    let failed = run(&mut state, JobStatus::Failed, false);
    assert!(shows_failure(state, &failed));

    // A job already running when the user opens Activity stays listed when it fails,
    // even when the failure arrives before Activity refreshes.
    let mut state = open(false);
    let mut job = project::Job::new("simulate", serde_json::json!({}), vec![]);
    job.status = JobStatus::Running;
    let running = job.id.clone();
    state.project.as_mut().unwrap().document.jobs.push(job);
    state.refresh_model();
    let mut runtime = rune::Runtime::new(
        state,
        crate::ui::view,
        (1440., 900.),
        rune::Env::new(false, 1.),
    );
    runtime.frame(0.);
    runtime.frame(1. / 60.);
    let button = runtime
        .elements()
        .into_iter()
        .find(|element| element.key.as_deref() == Some("toggle-jobs"))
        .expect("Activity button");
    let (x, y) = (
        button.rect[0] + button.rect[2] / 2.,
        button.rect[1] + button.rect[3] / 2.,
    );
    runtime.pointer_move(x, y);
    runtime.pointer_down(x, y);
    runtime.pointer_up(x, y);
    runtime.frame(1. / 60.);
    assert!(runtime.state.ui.activity.is_open());
    runtime.update(|state, _| {
        let job = state
            .project
            .as_mut()
            .unwrap()
            .document
            .jobs
            .last_mut()
            .unwrap();
        job.status = JobStatus::Failed;
        job.error = Some("Reconstruction ran out of GPU memory.".into());
        state.refresh_model();
    });
    assert!(shows_failure(runtime.state, &running));

    // Closing Activity early leaves it closed when the job later fails.
    let mut state = open(false);
    run(&mut state, JobStatus::Failed, true);
    assert!(
        !state.ui.activity.is_open(),
        "stays closed after the user closed it"
    );
}

#[test]
fn activity_lists_every_failure_it_opened_for() {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state
        .open_project(&directory.path().join("project"), true)
        .unwrap();
    let mut ids = Vec::new();
    for n in 0..6 {
        let mut job = project::Job::new("simulate", serde_json::json!({}), vec![]);
        job.status = JobStatus::Failed;
        job.error = Some(format!("Failure {n}"));
        ids.push(job.id.clone());
        state.project.as_mut().unwrap().document.jobs.push(job);
    }
    state.ui.activity.follow(ids);
    state.refresh_model();
    let mut runtime = rune::Runtime::new(
        state,
        crate::ui::view,
        (1440., 900.),
        rune::Env::new(false, 1.),
    );
    runtime.frame(0.);
    runtime.frame(1. / 60.);
    let elements = runtime.elements();
    for n in 0..6 {
        let error = format!("Failure {n}");
        assert!(
            elements
                .iter()
                .any(|e| e.text.as_deref() == Some(error.as_str())),
            "{error} is listed"
        );
    }
}

#[test]
fn demo_projects_are_numbered_so_earlier_demos_are_kept() {
    let folder = crate::preferences::Preferences::data_dir().join("Demo projects");
    let _ = std::fs::remove_dir_all(&folder);
    let mut state = AppState::init();
    // A scan passed at launch waits for the user's own project.
    let scratch = tempfile::tempdir().unwrap();
    let scan = scratch.path().join("scan.nxs");
    std::fs::write(&scan, b"").unwrap();
    state.startup_input = Some(scan.clone());
    state.open_demo_project().unwrap();
    assert_eq!(
        state.project.as_ref().unwrap().root,
        std::fs::canonicalize(folder.join("Demo project")).unwrap()
    );
    assert!(state.project.as_ref().unwrap().document.assets.is_empty());
    assert_eq!(state.startup_input.as_ref(), Some(&scan));
    state.open_demo_project().unwrap();
    assert_eq!(
        state.project.as_ref().unwrap().root,
        std::fs::canonicalize(folder.join("Demo project 2")).unwrap()
    );
    state
        .open_project(&scratch.path().join("mine"), true)
        .unwrap();
    assert!(state.startup_input.is_none());
    assert_eq!(state.project.as_ref().unwrap().document.assets.len(), 1);
    drop(state);
    std::fs::remove_dir_all(&folder).unwrap();
}

#[test]
fn window_fields_round_for_reading_but_keep_narrow_windows_applicable() {
    use super::window::window_text;
    assert_eq!(window_text(-0.020069934, 0.9726638), ["-0.0201", "0.9727"]);
    // Four decimals would read 1 and 1: an empty window the fields could not apply.
    assert_eq!(window_text(1.0, 1.00002), ["1", "1.00002"]);
    assert_eq!(window_text(2e-5, 9e-5), ["2.000e-5", "9.000e-5"]);
}

#[test]
fn applying_the_window_the_fields_show_keeps_it_exact() {
    let exact = [-0.020069934_f32, 0.9726638];
    let mut state = AppState::init();
    state.window.restore(Some(exact));
    state.window.follow(Some(exact));
    assert_eq!(state.window.field(0).text, "-0.0201");
    let mut runtime = rune::Runtime::new(
        state,
        crate::ui::view,
        (1440., 900.),
        rune::Env::new(false, 1.),
    );
    runtime.update(|state, tx| state.dispatch(UiAction::ApplyWindow, tx));
    assert_eq!(runtime.state.window.applied(), Some(exact));
    // An edited field applies what was typed.
    runtime.state.window.field_mut(1).replace_text("0.5");
    runtime.update(|state, tx| state.dispatch(UiAction::ApplyWindow, tx));
    assert_eq!(runtime.state.window.applied(), Some([-0.020069934, 0.5]));

    // Editing one limit keeps the other exact: 1.000051 reads as 1.0001, but an upper
    // limit of 1.00008 still lies above it.
    let committed = [1.000051, 1.5];
    let mut state = AppState::init();
    state.window.restore(Some(committed));
    state.window.follow(Some(committed));
    assert_eq!(state.window.field(0).text, "1.0001");
    state.window.field_mut(1).replace_text("1.00008");
    state.window.follow(Some(committed)); // a viewer refresh while the edit is pending
    assert_eq!(state.window.field(1).text, "1.00008");
    assert_eq!(state.window.parse(), Ok(Some([1.000051, 1.00008])));

    // Reopening the project restores the pending edit with the exact untouched limit.
    let saved = state.capture_workspace();
    let mut reopened = AppState::init();
    reopened.restore_workspace(&saved);
    assert_eq!(reopened.window.field(1).text, "1.00008");
    assert_eq!(reopened.window.parse(), Ok(Some([1.000051, 1.00008])));
}

#[test]
fn a_new_project_lives_in_the_folder_the_save_dialog_names() {
    let directory = tempfile::tempdir().unwrap();
    // A folder named like the manifest is still the new project's own folder.
    let named = directory.path().join(project::MANIFEST);
    let mut state = AppState::init();
    state.open_project(&named, true).unwrap();
    assert_eq!(
        state.project.as_ref().unwrap().root,
        std::fs::canonicalize(&named).unwrap()
    );
    assert!(named.join(project::MANIFEST).is_file());
    assert!(
        !directory.path().join(".tomojax.lock").exists(),
        "the parent is untouched"
    );

    drop(state);
    // It reopens as itself, and so does its manifest file.
    for path in [named.clone(), named.join(project::MANIFEST)] {
        let mut state = AppState::init();
        state.open_project(&path, false).unwrap();
        assert_eq!(
            state.project.as_ref().unwrap().root,
            std::fs::canonicalize(&named).unwrap(),
            "{}",
            path.display()
        );
    }
}
