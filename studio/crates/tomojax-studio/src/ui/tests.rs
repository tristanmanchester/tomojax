//! Exercise native layout and event routing with the real application state.
use rune::{Env, Runtime, TextEdit, runtime::ElementInfo};
use tempfile::TempDir;

use super::{model::*, view};
use crate::{
    AppState,
    project::{Asset, Project},
};

fn runtime(state: AppState, size: (f32, f32)) -> Runtime<AppState> {
    let mut runtime = Runtime::new(state, view, size, Env::new(false, 1.));
    runtime.frame(0.);
    runtime.frame(1. / 60.);
    runtime
}

fn find(runtime: &Runtime<AppState>, key: &str) -> ElementInfo {
    runtime
        .elements()
        .into_iter()
        .find(|element| element.key.as_deref() == Some(key))
        .unwrap_or_else(|| panic!("missing native element: {key}"))
}

fn click(runtime: &mut Runtime<AppState>, key: &str) {
    let rect = find(runtime, key).rect;
    let (x, y) = (rect[0] + rect[2] / 2., rect[1] + rect[3] / 2.);
    runtime.pointer_move(x, y);
    runtime.pointer_down(x, y);
    runtime.pointer_up(x, y);
    runtime.frame(1. / 60.);
}

fn project_state() -> (TempDir, AppState) {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(directory.path(), true).unwrap());
    state.model.project = Some(ProjectView {
        name: "Layout test project".into(),
        path: directory.path().display().to_string(),
        ..Default::default()
    });
    state.model.error = None;
    (directory, state)
}

/// Test presentation metadata only: no fixture image enters production code.
fn viewing_state() -> (TempDir, AppState) {
    let (directory, mut state) = project_state();
    let path = directory.path().join("specimen.nxs");
    std::fs::write(&path, []).unwrap();
    let mut asset = Asset::input(path.clone(), false);
    asset.name = "Registered test volume".into();
    state.model.active_asset = Some(asset.id.clone());
    state.model.selected_assets.insert(asset.id.clone());
    state.model.assets.push(AssetView {
        id: asset.id.clone(),
        name: asset.name.clone(),
        kind: AssetKind::Volume,
        path: path.display().to_string(),
        shape: vec![32, 40, 48],
        ..Default::default()
    });
    state.project.as_mut().unwrap().document.assets.push(asset);
    for (kind, count, axes) in [
        (PlaneKind::Axial, 32, ["y", "x"]),
        (PlaneKind::Coronal, 40, ["z", "x"]),
        (PlaneKind::Sagittal, 48, ["z", "y"]),
        (PlaneKind::Sinogram, 40, ["view", "u"]),
        (PlaneKind::Volume, 1, ["", " "]),
    ] {
        state.model.planes.insert(
            kind,
            PlaneView {
                available: true,
                pixel_shape: [40, 48],
                spacing: [1., 1.],
                axes: axes.map(str::to_owned),
                count,
                index: count / 2,
                ..Default::default()
            },
        );
    }
    state.ui.display.indices = [16, 20, 24];
    (directory, state)
}

#[test]
fn minimum_window_keeps_canvas_inputs_and_export_action_usable() {
    let (_directory, state) = viewing_state();
    let runtime = runtime(state, (1180., 780.));
    let canvas = find(&runtime, "canvas-area-8");
    assert!(canvas.rect[2] > 500. && canvas.rect[3] > 270., "{canvas:?}");
    let inspector = find(&runtime, "inspector").rect;
    let low = find(&runtime, "window-low").rect;
    let high = find(&runtime, "window-high").rect;
    for rect in [low, high] {
        assert!(
            rect[2] > 70. && rect[3] >= 20.,
            "unusable numeric input: {rect:?}"
        );
        assert!(rect[0] >= inspector[0] && rect[0] + rect[2] <= inspector[0] + inspector[2]);
    }
    assert!(low[0] + low[2] <= high[0], "window inputs overlap");
    let export = find(&runtime, "export-result");
    assert!(
        export.visible[3] >= 39. && export.visible[1] + export.visible[3] <= 780.,
        "export footer is hidden"
    );
    for key in [
        "colormap-gray",
        "colormap-bone",
        "colormap-viridis",
        "colormap-magma",
    ] {
        assert!(find(&runtime, key).rect[2] > 40., "{key} is compressed");
    }
}

#[test]
fn long_processing_forms_scroll_without_shrinking_the_queue_action() {
    let (_directory, mut state) = viewing_state();
    state.ui.workspace = Workspace::Reconstruct;
    state.ui.advanced_open = true;
    state.model.backend.ready = true;
    let runtime = runtime(state, (1180., 780.));
    let scroll = find(&runtime, "inspector-scroll").rect;
    let queue = find(&runtime, "queue-operation");
    assert!(scroll[3] > 350.);
    assert!(queue.focusable && queue.visible[3] >= 39.);
    assert!(
        queue.rect[1] >= scroll[1] + scroll[3],
        "queue action overlaps its scroll region"
    );
    let iterations = find(&runtime, "parameter-iterations");
    assert!(iterations.rect[2] > 180. && iterations.rect[3] >= 20.);
}

#[test]
fn selecting_a_linked_view_promotes_it_through_the_application_action() {
    let (_directory, state) = viewing_state();
    let mut runtime = runtime(state, (1540., 980.));
    click(&mut runtime, "thumbnail-Axial");
    assert_eq!(runtime.state.ui.main_view, PlaneKind::Axial);
    assert!(runtime.state.model.error.is_none());
}

#[test]
fn command_search_runs_the_visible_action_and_closes_the_palette() {
    let mut state = AppState::init();
    state.ui.palette_open = true;
    state.ui.command_query = TextEdit::new("Fit image");
    state.ui.display.zoom = 3.;
    let mut runtime = runtime(state, (1180., 780.));
    assert!(
        find(&runtime, "command-search").focused,
        "palette must focus search on open"
    );
    let options = runtime
        .elements()
        .into_iter()
        .filter(|e| e.role == "option")
        .collect::<Vec<_>>();
    assert_eq!(options.len(), 1);
    assert_eq!(options[0].label.as_deref(), Some("Fit image"));
    click(&mut runtime, "command-fit");
    assert_eq!(runtime.state.ui.display.zoom, 1.);
    assert!(!runtime.state.ui.palette_open);
}

#[test]
fn dropping_on_welcome_explains_the_required_project() {
    let mut runtime = runtime(AppState::init(), (1180., 780.));
    runtime.drop_files(vec!["/tmp/specimen.nxs".into()], Some((400., 300.)));
    runtime.frame(1. / 60.);
    assert!(
        runtime
            .state
            .model
            .error
            .as_deref()
            .is_some_and(|e| e.contains("Create or open a project"))
    );
    assert!(find(&runtime, "error-banner").visible[3] > 0.);
}

#[test]
fn dropping_multiple_files_keeps_each_path_and_original() {
    let (directory, state) = project_state();
    let paths = ["first specimen; (a).nxs", "second specimen [b].nxs"]
        .map(|name| directory.path().join(name));
    for path in &paths {
        std::fs::write(path, b"test source retained").unwrap();
    }
    let mut runtime = runtime(state, (1180., 780.));
    runtime.drop_files(paths.to_vec(), Some((500., 300.)));
    runtime.frame(1. / 60.);
    let project = runtime.state.project.as_ref().unwrap();
    assert_eq!(project.document.assets.len(), 2);
    for path in paths {
        assert!(
            project
                .document
                .assets
                .iter()
                .any(|asset| project.resolve(&asset.path) == path)
        );
        assert_eq!(std::fs::read(path).unwrap(), b"test source retained");
    }
    assert_eq!(runtime.state.ui.workspace, Workspace::Prepare);
    assert_eq!(runtime.state.ui.operation, "open");
}

#[test]
fn comparison_uses_each_result_grid_when_the_active_asset_has_no_plane() {
    let (_directory, mut state) = viewing_state();
    state.ui.workspace = Workspace::Compare;
    let mut second = state.model.assets[0].clone();
    second.id = "second-result".into();
    second.name = "Other registered result".into();
    state.model.compare_left = Some(state.model.assets[0].id.clone());
    state.model.compare_right = Some(second.id.clone());
    state.model.assets.push(second);
    state.model.planes.clear();
    state.model.comparison_count = 32;
    state.model.comparison_index = 16;
    state.model.comparison_planes = [
        Some(PlaneView {
            available: true,
            pixel_shape: [20, 40],
            spacing: [2., 1.],
            axes: ["y".into(), "x".into()],
            index: 16,
            count: 32,
            ..Default::default()
        }),
        Some(PlaneView {
            available: true,
            pixel_shape: [40, 20],
            spacing: [1., 3.],
            axes: ["y".into(), "x".into()],
            index: 16,
            count: 48,
            ..Default::default()
        }),
    ];
    let runtime = runtime(state, (1180., 780.));
    for (mark, aspect) in [(6, 1.), (7, 1.5)] {
        let canvas = find(&runtime, &format!("scientific-canvas-{mark}"));
        assert!(
            (canvas.rect[2] / canvas.rect[3] - aspect).abs() < 0.01,
            "{canvas:?}"
        );
        assert!(canvas.visible[2] > 200. && canvas.visible[3] > 150.);
    }
    assert!(
        find(&runtime, "image-index").focusable,
        "comparison slice navigation must remain available"
    );
}

#[test]
fn fixed_region_planes_do_not_claim_a_one_voxel_volume() {
    let (_directory, mut state) = viewing_state();
    state.model.assets[0].kind = AssetKind::Other;
    state.model.assets[0].shape = vec![12, 18];
    state.ui.main_view = PlaneKind::Axial;
    state.model.planes.remove(&PlaneKind::Volume);
    for plane in state.model.planes.values_mut() {
        plane.count = 1;
        plane.index = 0;
    }
    let runtime = runtime(state, (1180., 780.));
    let text = runtime
        .elements()
        .into_iter()
        .filter_map(|element| element.text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("12 × 18"));
    assert!(
        !text.contains("1 × 1 × 1"),
        "fixed previews are not a reconstructed volume"
    );
    assert!(!text.contains("Drag to place the linked crosshair"));
}

#[test]
fn derived_sidebar_label_names_the_operation_even_for_simulated_inputs() {
    let (_directory, mut state) = viewing_state();
    state.model.assets[0].simulated = true;
    state.model.assets[0].operation = Some("reconstruct".into());
    let runtime = runtime(state, (1440., 900.));
    let text = runtime
        .elements()
        .into_iter()
        .filter_map(|element| element.text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("Reconstruction"));
    assert!(!text.contains("Simulated volume"));
}

#[test]
fn welcome_and_empty_states_show_only_the_project_actions() {
    let welcome = runtime(AppState::init(), (1440., 900.));
    let text = welcome
        .elements()
        .into_iter()
        .filter_map(|element| element.text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("Create a project"));
    assert!(text.contains("Open project"));
    assert!(!text.contains("Projects contain"));
    assert!(!text.contains("make the data your own"));
    assert!(find(&welcome, "new-project").focusable);
    let (_directory, state) = project_state();
    let empty = runtime(state, (1440., 900.));
    let text = empty
        .elements()
        .into_iter()
        .filter_map(|element| element.text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("No scans in this project"));
    assert!(!text.contains("referenced in place"));
    assert!(find(&empty, "empty-add-scans").focusable);
    assert!(find(&empty, "empty-add-folder").focusable);
}

#[test]
fn native_job_record_keeps_nested_progress_units_plots_and_critical_notes() {
    use crate::protocol::Progress;
    use serde_json::json;
    let (_directory, mut state) = project_state();
    let mut job = crate::project::Job::new("align", json!({}), vec![]);
    for (task, parent, name, total, unit) in [
        (1, None, "align", Some(2), "levels"),
        (2, Some(1), "level", Some(12), "steps"),
        (3, Some(2), "step", Some(3), "iterations"),
    ] {
        job.telemetry.apply(
            "run-1",
            &Progress::Started {
                schema: 2,
                task,
                time: 0.,
                parent,
                name: name.into(),
                label: name.into(),
                total,
                unit: unit.into(),
            },
        );
    }
    for step in 1..=2 {
        job.telemetry.apply(
            "run-1",
            &Progress::Advanced {
                schema: 2,
                task: 2,
                time: step as f64,
                done: step,
                total: Some(12),
                eta: Some(30.),
                metrics: [
                    ("objective".into(), json!(1. / step as f64)),
                    ("pose_step_rms".into(), json!(0.01 * step as f64)),
                ]
                .into_iter()
                .collect(),
            },
        );
    }
    job.telemetry.apply(
        "run-1",
        &Progress::Paused {
            schema: 2,
            task: 2,
            time: 2.,
        },
    );
    job.telemetry.apply(
        "run-1",
        &Progress::Note {
            schema: 2,
            task: Some(2),
            time: 2.,
            level: "critical".into(),
            message: "Recorded numerical warning".into(),
        },
    );
    state.model.jobs.push(JobView {
        id: job.id.clone(),
        name: "Alignment".into(),
        operation: "align".into(),
        status: JobStatus::Paused,
        ..Default::default()
    });
    state.ui.selected_job = Some(job.id.clone());
    state.project.as_mut().unwrap().document.jobs.push(job);
    state.ui.workspace = Workspace::Jobs;
    let runtime = runtime(state, (1440., 900.));
    let elements = runtime.elements();
    let text = elements
        .iter()
        .filter_map(|element| element.text.as_deref())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(text.contains("2 / 12 steps · paused"), "{text}");
    assert!(text.contains("ETA 30.0 s of active work · paused"));
    assert!(text.contains("Recorded numerical warning"));
    for task in 1..=3 {
        find(&runtime, &format!("task-run-1-{task}"));
    }
    for (metric, title) in [
        ("objective", "Alignment objective"),
        ("pose_step_rms", "Accepted pose step"),
    ] {
        let plot = find(&runtime, &format!("metric-plot-{metric}"));
        let unit = &crate::progress::metric_registry()[metric].unit;
        assert_eq!(
            plot.label.as_deref(),
            Some(format!("{title} against step number, {unit}").as_str())
        );
    }
}

#[test]
fn job_controls_stay_visible_above_a_long_expanded_history() {
    use crate::protocol::Progress;
    use serde_json::json;
    let (_directory, mut state) = project_state();
    let mut job = crate::project::Job::new("align", json!({}), vec![]);
    for task in 1..=40 {
        job.telemetry.apply(
            "run",
            &Progress::Started {
                schema: 2,
                task,
                time: task as f64,
                parent: (task > 1).then_some(1),
                name: if task == 1 { "align" } else { "step" }.into(),
                label: format!("task {task}"),
                total: Some(40),
                unit: "steps".into(),
            },
        );
    }
    let id = job.id.clone();
    state.model.jobs.push(JobView {
        id: id.clone(),
        name: "Alignment".into(),
        operation: "align".into(),
        status: JobStatus::Paused,
        can_pause: true,
        ..Default::default()
    });
    state.project.as_mut().unwrap().document.jobs.push(job);
    state.ui.workspace = Workspace::Jobs;
    state.ui.selected_job = Some(id.clone());
    let mut runtime = runtime(state, (1440., 900.));
    for key in [format!("resume-job-{id}"), format!("cancel-job-{id}")] {
        let button = find(&runtime, &key);
        assert!(
            button.visible[3] >= 29. && button.rect[1] < 450.,
            "{button:?}"
        );
    }
    click(&mut runtime, &format!("job-details-{id}"));
    assert!(runtime.state.ui.selected_job.is_none());
}

#[test]
fn processing_menu_routes_each_operation_to_one_workspace() {
    let (_dir, state) = project_state();
    let mut rt = runtime(state, (1540., 980.));
    let mut seen = std::collections::BTreeSet::new();
    for (tab, mandatory, excluded) in [
        ("prepare", "correct", "reconstruct"),
        ("reconstruct", "slice", "open"),
    ] {
        click(&mut rt, &format!("workspace-{tab}"));
        click(&mut rt, "operation-selector");
        let keys = rt
            .elements()
            .into_iter()
            .filter(|e| !e.exiting)
            .filter_map(|e| e.key)
            .collect::<Vec<_>>();
        assert!(keys.contains(&format!("operation-{mandatory}")));
        assert!(
            !keys.contains(&format!("operation-{excluded}")),
            "{tab}: {keys:?}"
        );
        for key in keys
            .iter()
            .filter_map(|k| k.strip_prefix("operation-"))
            .filter(|key| !matches!(*key, "menu" | "selector"))
        {
            assert!(seen.insert(key.to_owned()), "duplicate operation {key}");
        }
    }
    assert_eq!(
        seen.len(),
        crate::operations::GROUPS
            .iter()
            .map(|g| g.operations.len())
            .sum::<usize>()
    );
}

#[test]
fn filter_picker_and_solver_fields_follow_the_submitted_method() {
    let (_dir, mut state) = project_state();
    state.set_workspace(Workspace::Reconstruct);
    state.set_operation(
        "reconstruct",
        Some(&serde_json::json!({"method":"fbp","filter":"hann","iterations":12})),
    );
    let mut rt = runtime(state, (1540., 980.));
    click(&mut rt, "parameter-filter");
    find(&rt, "parameter-filter-shepp-logan");
    assert!(
        !rt.elements()
            .iter()
            .any(|e| e.key.as_deref() == Some("parameter-iterations"))
    );
    let parameters =
        |state: &AppState| crate::forms::parameters(&state.ui, &state.catalog).unwrap();
    assert_eq!(parameters(&rt.state)["filter"], "hann");
    assert!(parameters(&rt.state).get("iterations").is_none());
    rt.state
        .ui
        .parameter_choices
        .insert("method".into(), "fista".into());
    rt.invalidate();
    rt.frame(0.1);
    find(&rt, "parameter-iterations");
    find(&rt, "parameter-tv_weight");
    assert!(
        !rt.elements()
            .iter()
            .any(|e| !e.exiting && e.key.as_deref() == Some("parameter-filter"))
    );
    assert_eq!(parameters(&rt.state)["iterations"], 12);
    assert!(parameters(&rt.state).get("filter").is_none());
}

#[test]
fn navigator_rename_survives_reopen_without_changing_the_job_or_provenance() {
    let (dir, mut state) = viewing_state();
    let doc = &mut state.project.as_mut().unwrap().document;
    let asset = &mut doc.assets[0];
    let id = asset.id.clone();
    let provenance = crate::protocol::Provenance {
        operation: "simulate".into(),
        parameters: serde_json::json!({"phantom":"cube"}),
        inputs: vec![],
    };
    asset.provenance = Some(provenance);
    let before = serde_json::to_value(&asset.provenance).unwrap();
    let mut job =
        crate::project::Job::new("simulate", serde_json::json!({"phantom":"cube"}), vec![]);
    job.result = Some(id.clone());
    job.status = crate::project::JobStatus::Completed;
    doc.jobs.push(job);
    let jobs = serde_json::to_value(&doc.jobs).unwrap();
    state.model.assets[0].operation = Some("simulate".into());
    let mut rt = runtime(state, (1180., 780.));
    click(&mut rt, "rename-result");
    rt.state
        .ui
        .rename_draft
        .replace_text(" Calibration sphere ");
    rt.frame(0.1);
    click(&mut rt, "save-result-name");
    rt.state.shutdown();
    drop(rt);
    let project = Project::open(dir.path(), false).unwrap();
    let result = project.document.asset(&id).unwrap();
    assert_eq!(result.name, "Calibration sphere");
    assert_eq!(serde_json::to_value(&result.provenance).unwrap(), before);
    assert_eq!(serde_json::to_value(&project.document.jobs).unwrap(), jobs);
}

#[test]
fn contextual_tools_and_named_toolbar_controls_fit_the_workspace() {
    for (kind, control) in [
        (PlaneKind::Volume, "clip-2"),
        (PlaneKind::Axial, "tool-distance"),
    ] {
        let (_dir, mut state) = viewing_state();
        state.ui.main_view = kind;
        let rt = runtime(state, (1180., 780.));
        let element = find(&rt, control);
        assert!(element.visible[3] >= 24.);
        for (key, name) in [
            ("toggle-inspector", "Inspector"),
            ("toggle-jobs", "Activity"),
            ("shortcut-settings", "Shortcuts"),
        ] {
            let e = find(&rt, key);
            assert_eq!(e.label.as_deref(), Some(name));
            assert!(e.visible[0] + e.visible[2] <= 1180. && e.visible[3] >= 32.);
        }
        find(&rt, "export-result");
    }
}

#[test]
fn explorer_labels_physical_pitch_and_keeps_one_save_status() {
    let (_dir, mut state) = viewing_state();
    state.model.assets[0].voxel_pitch = Some([0.5, 0.75, 2.]);
    state.model.assets[0].units = "mm".into();
    let rt = runtime(state, (1540., 980.));
    let elements = rt.elements();
    assert!(
        elements
            .iter()
            .filter_map(|e| e.text.as_deref())
            .any(|text| text.contains("voxels of 0.5 × 0.75 × 2 mm"))
    );
    let saved = elements
        .iter()
        .filter(|e| e.text.as_deref() == Some("Project saved"))
        .collect::<Vec<_>>();
    assert_eq!(saved.len(), 1);
    assert!(saved[0].rect[1] >= 950.);
    assert!(
        elements
            .iter()
            .any(|e| e.text.as_deref() == Some("Explore") && e.path.contains("inspector"))
    );
}

#[test]
fn job_row_shows_one_status_and_preserves_stage_detail() {
    let (_dir, mut state) = project_state();
    state.ui.workspace = Workspace::Jobs;
    state.model.jobs.push(JobView {
        id: "result-job".into(),
        name: "Specimen · FBP".into(),
        operation: "Reconstruction".into(),
        status: JobStatus::Completed,
        stage: "Complete".into(),
        ..Default::default()
    });
    let mut rt = runtime(state, (1540., 980.));
    assert_eq!(
        rt.elements()
            .iter()
            .filter(|e| e.text.as_deref() == Some("Complete"))
            .count(),
        1
    );
    rt.update(|s, _| s.model.jobs[0].stage = "Plan ready".into());
    rt.frame(0.1);
    assert!(
        rt.elements()
            .iter()
            .any(|e| e.text.as_deref() == Some("Plan ready"))
    );
}

fn comparison_state(note: Option<&str>) -> (TempDir, AppState) {
    let (directory, mut state) = viewing_state();
    state.ui.workspace = Workspace::Compare;
    let mut second = state.model.assets[0].clone();
    second.id = "second-result".into();
    second.name = "Other registered result".into();
    state.model.compare_left = Some(state.model.assets[0].id.clone());
    state.model.compare_right = Some(second.id.clone());
    state.model.assets.push(second);
    state.model.comparison_note = note.map(str::to_owned);
    (directory, state)
}

#[test]
fn compare_switches_planes_and_modes_and_offers_no_export() {
    let (_directory, state) = comparison_state(None);
    let rt = runtime(state, (1540., 980.));
    for key in [
        "compare-plane-axial",
        "compare-plane-coronal",
        "compare-plane-sagittal",
        "compare-mode-side-by-side",
        "compare-mode-split",
        "compare-mode-difference",
        "swap-comparison",
    ] {
        assert!(find(&rt, key).focusable, "{key}");
    }
    assert!(
        rt.elements()
            .iter()
            .all(|e| e.key.as_deref() != Some("export-result"))
    );
}

#[test]
fn split_and_difference_need_one_shared_grid() {
    let (_directory, state) = comparison_state(Some(
        "Grids differ: slices are matched by index, not position",
    ));
    let rt = runtime(state, (1540., 980.));
    assert!(find(&rt, "compare-mode-side-by-side").focusable);
    for key in ["compare-mode-split", "compare-mode-difference"] {
        assert!(!find(&rt, key).focusable, "{key}");
    }
}

#[test]
fn result_picker_lists_volumes_in_place_and_closes_on_choice() {
    let (_directory, state) = comparison_state(None);
    let mut rt = runtime(state, (1540., 980.));
    click(&mut rt, "compare-right");
    assert_eq!(rt.state.ui.compare_menu, Some(1));
    rt.frame(0.5); // let the list finish appearing before pressing an option
    let left_id = rt.state.model.compare_left.clone().unwrap();
    click(&mut rt, &format!("compare-right-{left_id}"));
    assert_eq!(
        rt.state.model.compare_right.as_deref(),
        Some(left_id.as_str())
    );
    assert_eq!(rt.state.ui.compare_menu, None);
}

#[test]
fn child_results_drop_the_parent_prefix_and_custom_names_stay_whole() {
    let (_directory, mut state) = viewing_state();
    let parent = state.model.assets[0].clone();
    for (id, name) in [
        ("derived", format!("{} · FBP", parent.name)),
        ("custom", "Best slice set".into()),
    ] {
        state.model.assets.push(AssetView {
            id: id.into(),
            name,
            kind: AssetKind::Volume,
            parent: Some(parent.id.clone()),
            operation: Some("reconstruct".into()),
            ..Default::default()
        });
    }
    let rt = runtime(state, (1540., 980.));
    let texts: Vec<String> = rt.elements().into_iter().filter_map(|e| e.text).collect();
    assert!(texts.iter().any(|t| t == "FBP"));
    assert!(texts.iter().any(|t| t == "Best slice set"));
}

#[test]
fn the_top_bar_names_the_project_only_while_the_navigator_is_hidden() {
    for tree_open in [true, false] {
        let (_directory, mut state) = viewing_state();
        state.ui.tree_open = tree_open;
        let name = state.model.project.as_ref().unwrap().name.clone();
        let rt = runtime(state, (1540., 980.));
        let shown = rt
            .elements()
            .iter()
            .filter(|e| e.text.as_deref() == Some(name.as_str()))
            .count();
        assert_eq!(shown, 1, "tree_open={tree_open}");
    }
}

#[test]
fn removing_an_asset_needs_confirmation_and_leaves_its_file_on_disk() {
    let (_directory, state) = viewing_state();
    let id = state.model.assets[0].id.clone();
    let path = state
        .project
        .as_ref()
        .unwrap()
        .resolve(&state.project.as_ref().unwrap().document.assets[0].path);
    let mut rt = runtime(state, (1540., 980.));
    click(&mut rt, "remove-asset");
    assert_eq!(rt.state.ui.confirm_remove.as_deref(), Some(id.as_str()));
    assert!(
        rt.state
            .project
            .as_ref()
            .unwrap()
            .document
            .asset(&id)
            .is_some()
    );
    rt.frame(0.5);
    click(&mut rt, "confirm-remove-asset");
    assert!(
        rt.state
            .project
            .as_ref()
            .unwrap()
            .document
            .asset(&id)
            .is_none()
    );
    assert!(path.exists());
}

#[test]
fn an_asset_used_by_a_queued_job_cannot_be_removed() {
    let (_directory, mut state) = viewing_state();
    let id = state.model.assets[0].id.clone();
    let project = state.project.as_mut().unwrap();
    let input = project.capture(&id).unwrap();
    project.document.jobs.push(crate::project::Job::new(
        "reconstruct",
        serde_json::json!({}),
        vec![input],
    ));
    state.ui.confirm_remove = Some(id.clone());
    let mut rt = runtime(state, (1540., 980.));
    click(&mut rt, "confirm-remove-asset");
    assert!(
        rt.state
            .project
            .as_ref()
            .unwrap()
            .document
            .asset(&id)
            .is_some()
    );
    assert!(
        rt.state
            .model
            .error
            .as_deref()
            .is_some_and(|e| e.contains("Cancel the job first"))
    );
}

#[test]
fn the_active_asset_stays_in_the_selection() {
    let (directory, state) = viewing_state();
    let mut rt = runtime(state, (1180., 780.));
    let first = rt.state.model.active_asset.clone().unwrap();
    let path = directory.path().join("second.nxs");
    std::fs::write(&path, []).unwrap();
    let second = Asset::input(path, false);
    let second_id = second.id.clone();
    rt.update(|s, _| s.project.as_mut().unwrap().document.assets.push(second));
    let select = |rt: &mut Runtime<AppState>, id: &str| {
        let id = id.to_owned();
        rt.update(move |s, tx| s.dispatch(UiAction::SelectAsset { id, extend: true }, tx));
    };
    select(&mut rt, &second_id);
    assert_eq!(
        rt.state.model.active_asset.as_deref(),
        Some(second_id.as_str())
    );
    // Taking the active asset out of the selection activates what remains.
    select(&mut rt, &second_id);
    assert_eq!(rt.state.model.selected_assets.len(), 1);
    assert_eq!(rt.state.model.active_asset.as_deref(), Some(first.as_str()));
    // Deselecting the last one leaves nothing to process.
    select(&mut rt, &first);
    assert!(rt.state.model.selected_assets.is_empty());
    assert_eq!(rt.state.model.active_asset, None);
    // Removing the active asset from the project activates what stays selected.
    select(&mut rt, &first);
    select(&mut rt, &second_id);
    let removed = second_id.clone();
    rt.update(move |s, tx| s.dispatch(UiAction::RemoveAsset(removed), tx));
    assert_eq!(rt.state.model.active_asset.as_deref(), Some(first.as_str()));
}
