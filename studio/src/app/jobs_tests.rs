use super::*;
use crate::project::CapturedInput;
use crate::protocol::{ArrayDescriptor, Info, Provenance};
use npyz::WriterBuilder;

fn source(state: &mut AppState, directory: &Path) -> CapturedInput {
    let input = directory.join("input.nxs");
    std::fs::write(&input, b"captured source metadata").unwrap();
    let asset = Asset::input(input, false);
    let id = asset.id.clone();
    state.project = Some(Project::open(&directory.join("project"), true).unwrap());
    state.project.as_mut().unwrap().document.assets.push(asset);
    state.project.as_ref().unwrap().capture(&id).unwrap()
}

fn plane(path: &Path) -> ArrayDescriptor {
    serde_json::from_value(json!({
        "storage":{"type":"npy","path":path,"artifact":"plane-token"},"shape":[2,2],"dtype":"<f4",
        "axes":[
            {"name":"y","frame":"volume","unit":"mm","coordinates":{"type":"uniform","origin":1.,"spacing":2.}},
            {"name":"x","frame":"volume","unit":"mm","coordinates":{"type":"uniform","origin":-1.,"spacing":0.5}}
        ],"unit":"1/mm","domain":"attenuation",
        "fixed":[{"name":"z","frame":"volume","unit":"mm","index":3,"coordinate":7.}]
    })).unwrap()
}

#[test]
fn durable_commands_request_checkpoints_only_for_a_connected_consumer() {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    let input = source(&mut state, directory.path());
    let mut job = Job::new("reconstruct", json!({"method":"cgls"}), vec![input]);
    state.running = Some(Running {
        job: job.id.clone(),
        handles: vec!["scan".into()],
        products: vec![Product::Scan {
            handle: "scan".into(),
            projections: plane(Path::new("unused.npy")),
        }],
        next_input: 1,
    });
    assert_eq!(state.job_command(&job).unwrap()["previews"], true);
    state.live.set_enabled(false);
    assert_eq!(state.job_command(&job).unwrap()["previews"], false);
    state.live.set_enabled(true);
    job.operation = "align".into();
    assert_eq!(state.job_command(&job).unwrap()["previews"], true);
    for operation in ["correct", "simulate", "crop"] {
        job.operation = operation.into();
        assert_eq!(
            state.job_command(&job).unwrap()["previews"],
            false,
            "{operation} has no live checkpoint consumer"
        );
    }
}

#[test]
fn explicit_keep_copies_registered_arrays_and_captured_provenance_without_recomputing() {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state.started = true;
    state.demo = crate::app::Demo::Off;
    let input = source(&mut state, directory.path());
    let path = directory.path().join("preview.npy");
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&[2, 2])
        .writer(std::fs::File::create(&path).unwrap())
        .begin_nd()
        .unwrap();
    for value in [1f32, 2., 3., 4.] {
        writer.push(&value).unwrap();
    }
    writer.finish().unwrap();
    let parameters = json!({"method":"fourier","axis":"z","index":3});
    let manifest = Provenance {
        operation: "slice".into(),
        parameters: parameters.clone(),
        inputs: vec![Provenance {
            operation: "open".into(),
            parameters: json!({"path":input.path}),
            inputs: vec![],
        }],
    };
    let mut telemetry = crate::progress::JobProgress::default();
    for record in [
        json!({"event":"started","schema":2,"task":1,"time":0.,"parent":null,"name":"slice","label":"Slice preview","total":1,"unit":"steps"}),
        json!({"event":"advanced","schema":2,"task":1,"time":0.1,"done":1,"total":1,"metrics":{"loss":2.5},"eta":null}),
        json!({"event":"note","schema":2,"task":1,"time":0.2,"level":"warning","message":"Recorded worker preview warning"}),
    ] {
        telemetry.apply("slice-request", &serde_json::from_value(record).unwrap());
    }
    let kept = crate::previews::KeptPreview {
        input,
        operation: "slice".into(),
        parameters: parameters.clone(),
        request: "slice-request".into(),
        revision: 8,
        telemetry,
        output: Box::new(Output {
            value: Product::Plane {
                plane: plane(&path),
            },
            info: Info {
                summary: Default::default(),
                arrays: Default::default(),
            },
            manifest: manifest.clone(),
        }),
    };
    assert!(state.project.as_ref().unwrap().document.jobs.is_empty());
    assert!(state.engine.is_none());
    state.model.status = "3 s active work remaining".into();
    let id = state.keep_preview(kept).unwrap();
    let until = Instant::now() + Duration::from_secs(5);
    loop {
        state.poll();
        let job = state
            .project
            .as_ref()
            .unwrap()
            .document
            .jobs
            .iter()
            .find(|job| job.id == id)
            .unwrap();
        if !job.status.active() {
            break;
        }
        assert!(
            Instant::now() < until,
            "Keep did not finish: {:?}",
            state.model.error
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    let project = state.project.as_ref().unwrap();
    let job = &project.document.jobs[0];
    assert_eq!(job.status, JobStatus::Completed, "{:?}", job.error);
    assert_eq!(job.parameters, parameters);
    assert_eq!(job.kept_preview.as_ref().unwrap().request, "slice-request");
    assert_eq!(job.kept_preview.as_ref().unwrap().revision, 8);
    assert_eq!(
        job.telemetry.notes[0].message,
        "Recorded worker preview warning"
    );
    let loss = &job.telemetry.task("slice-request", 1).unwrap().metrics["loss"];
    assert_eq!(loss.unit.as_deref(), Some("objective units"));
    assert_eq!(loss.samples[0].value, json!(2.5));
    let asset = project
        .document
        .asset(job.result.as_ref().unwrap())
        .unwrap();
    assert_eq!(
        serde_json::to_value(asset.provenance.as_ref().unwrap()).unwrap(),
        serde_json::to_value(manifest).unwrap()
    );
    let durable = &asset.arrays["plane"];
    assert_eq!(durable.fixed[0].coordinate, 7.);
    let Storage::Npy {
        path: stored,
        artifact,
    } = &durable.storage
    else {
        panic!("Keep did not persist NPY")
    };
    assert!(artifact.is_none());
    assert_eq!(
        std::fs::read(project.resolve(Path::new(stored))).unwrap(),
        std::fs::read(&path).unwrap()
    );
    assert!(
        state.engine.is_none(),
        "Keep must not resubmit numerical work"
    );
    assert_eq!(state.model.status, "Slice preview complete");
}

#[test]
fn background_result_does_not_retarget_an_active_interactive_preview() {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    let input = source(&mut state, directory.path());
    let source_id = input.asset.clone();
    state.model.active_asset = Some(source_id.clone());
    state.model.selected_assets.insert(source_id.clone());
    state.ui.workspace = Workspace::Reconstruct;
    state.preview_context.active = Some(crate::previews::PreviewKind::Slice);
    let preview = plane(Path::new("live-preview.npy"));
    state
        .preview_context
        .arrays
        .insert("plane".into(), preview.clone());
    state.ui.display.indices = [7, 8, 9];
    state.ui.display.sinogram_row = 1;
    state.centre_sinogram = false;
    state.applied_window = Some([-3., 8.]);
    state.ui.display.window_low.replace_text("-3");
    state.ui.display.window_high.replace_text("8");
    let job = Job::new("reconstruct", json!({"method":"cgls"}), vec![input]);
    let id = job.id.clone();
    state.project.as_mut().unwrap().document.jobs.push(job);
    let mut result = Asset::input("derived/completed.nxs".into(), true);
    result.kind = "reconstruction".into();
    let result_id = result.id.clone();
    state.complete_asset(&id, result);
    assert_eq!(
        state.model.active_asset.as_deref(),
        Some(source_id.as_str())
    );
    assert_eq!(
        state
            .model
            .selected_assets
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        [source_id.as_str()]
    );
    assert_eq!(state.ui.display.indices, [7, 8, 9]);
    assert_eq!(state.ui.display.sinogram_row, 1);
    assert!(!state.centre_sinogram);
    assert_eq!(state.applied_window, Some([-3., 8.]));
    assert_eq!(state.ui.display.window_low.text, "-3");
    assert_eq!(
        serde_json::to_value(&state.preview_context.arrays["plane"]).unwrap(),
        serde_json::to_value(preview).unwrap()
    );
    let project = state.project.as_ref().unwrap();
    assert_eq!(project.document.jobs[0].status, JobStatus::Completed);
    assert_eq!(
        project.document.jobs[0].result.as_deref(),
        Some(result_id.as_str())
    );
    assert!(project.document.asset(&result_id).is_some());
}

#[test]
fn selected_processing_result_uses_a_central_sinogram_and_its_own_window() {
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state.started = true;
    state.demo = crate::app::Demo::Off;
    let input = source(&mut state, directory.path());
    // The input stays selected, as it is when the job is submitted.
    state.model.active_asset = Some(input.asset.clone());
    let job = Job::new("correct", json!({}), vec![input]);
    let id = job.id.clone();
    let path = state
        .project
        .as_ref()
        .unwrap()
        .root
        .join("derived/corrected.npy");
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&[2, 5, 4])
        .writer(std::fs::File::create(&path).unwrap())
        .begin_nd()
        .unwrap();
    for _ in 0..2 {
        for row in 0..5 {
            for column in 0..4 {
                writer
                    .push(&(if row == 2 { column as f32 + 1. } else { 0. }))
                    .unwrap();
            }
        }
    }
    writer.finish().unwrap();
    let mut projections = plane(&path);
    projections.shape = vec![2, 5, 4];
    projections.storage = Storage::Npy {
        path: path.to_string_lossy().into(),
        artifact: None,
    };
    projections.axes[0].name = "v".into();
    projections.axes[1].name = "u".into();
    projections.axes[0].frame = "detector".into();
    projections.axes[1].frame = "detector".into();
    projections.axes.insert(0,serde_json::from_value(json!({"name":"view","frame":"acquisition","unit":"degree","coordinates":{"type":"uniform","origin":0.,"spacing":90.}})).unwrap());
    projections.fixed.clear();
    projections.domain = "line_integrals".into();
    projections.unit = "1".into();
    let mut result = Asset::input(state.project.as_ref().unwrap().stored_path(&path), true);
    result.kind = "projections".into();
    result.arrays.insert("projections".into(), projections);
    state.project.as_mut().unwrap().document.jobs.push(job);
    state.centre_sinogram = false;
    state.ui.display.sinogram_row = 0;
    state.applied_window = Some([-100., 100.]);
    state.ui.display.window_low.replace_text("-100");
    state.ui.display.window_high.replace_text("100");
    state.window_prefill = None;
    state.complete_asset(&id, result);
    let until = Instant::now() + Duration::from_secs(3);
    while state.imaging.plane(5).is_none() || state.imaging.plane(4).is_none() {
        state.poll();
        assert!(Instant::now() < until, "new result did not load");
        std::thread::yield_now();
    }
    assert_eq!(state.imaging.plane(5).unwrap().index, 2);
    assert!(state.imaging.plane(5).unwrap().statistics.max > 0.);
    assert!(state.applied_window.is_none());
    let window = state.imaging.plane(4).unwrap().statistics.window;
    assert_eq!(
        state.ui.display.window_low.text,
        crate::ui::compact_number(f64::from(window[0]))
    );
    assert_eq!(
        state.ui.display.window_high.text,
        crate::ui::compact_number(f64::from(window[1]))
    );
}

#[test]
#[ignore = "real alignment checkpoint through AppState and the native Jobs UI"]
fn real_alignment_displays_registered_slices_pose_rows_and_metric_history() {
    let python = std::env::var_os("TOMOJAX_TEST_PYTHON").expect("set TOMOJAX_TEST_PYTHON");
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(directory.path(), true).unwrap());
    state.project.as_mut().unwrap().document.name = "Alignment checkpoint verification".into();
    state.started = true;
    state.demo = crate::app::Demo::Off;
    state.startup_input = None;
    state.ui.workspace = Workspace::Jobs;
    state.engine = Some(Engine::spawn(PathBuf::from(python), std::sync::Arc::new(|| {})).unwrap());
    let drive = |state: &mut AppState, done: &dyn Fn(&AppState) -> bool| {
        let until = Instant::now() + Duration::from_secs(120);
        while !done(state) {
            state.poll();
            assert!(
                Instant::now() < until,
                "real app timed out: {:?}; {:?}",
                state.model.error,
                state.model.jobs
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    };
    drive(&mut state, &|state| state.model.backend.ready);
    state.set_operation(
        "simulate",
        Some(&json!({"phantom":"shepp","size":12,"n_views":24,"seed":3})),
    );
    state.queue_operation("simulate").unwrap();
    drive(&mut state, &|state| {
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
        JobStatus::Completed,
        "{:?}",
        state.project.as_ref().unwrap().document.jobs[0].error
    );
    state.set_operation(
        "align",
        Some(&json!({
            "method":"joint","fit":"translations",
            "grid":{"nx":12,"ny":12,"nz":12,"vx":1.,"vy":1.,"vz":1.},
            "config":{"type":"AlignConfig","levels":[1],"iterations":2,"initial_image_iterations":2,
                "image_iterations":2,"refine_iterations":1,"tolerance":0.,"min_iterations":1}
        })),
    );
    state.queue_operation("align").unwrap();
    let id = state
        .project
        .as_ref()
        .unwrap()
        .document
        .jobs
        .last()
        .unwrap()
        .id
        .clone();
    drive(&mut state, &|state| {
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
    let job = state
        .project
        .as_ref()
        .unwrap()
        .document
        .jobs
        .iter()
        .find(|job| job.id == id)
        .unwrap();
    assert_eq!(job.status, JobStatus::Completed, "{:?}", job.error);
    drive(&mut state, &|state| {
        state
            .live
            .frame()
            .is_some_and(|frame| frame.info.job == id && frame.poses.is_some())
            && !state.live.loading()
    });
    drive(&mut state, &|state| {
        state
            .engine
            .as_ref()
            .is_some_and(|engine| engine.pending_count() == 0)
    });
    assert!(
        state.model.error.is_none(),
        "completed alignment cleanup failed: {:?}",
        state.model.error
    );
    let job = state
        .project
        .as_ref()
        .unwrap()
        .document
        .jobs
        .iter()
        .find(|job| job.id == id)
        .unwrap();
    for metric in ["objective", "pose_step_rms"] {
        assert!(
            job.telemetry.tasks.iter().any(|task| task
                .metrics
                .get(metric)
                .is_some_and(|history| !history.samples.is_empty())),
            "the real app lost {metric} history"
        );
    }
    state.ui.workspace = Workspace::Jobs;
    state.ui.activity = crate::ui::model::Activity::Closed;
    state.ui.selected_job = Some(id.clone());
    state.refresh_model();
    for mark in [40, 41, 42] {
        assert!(state.render_items().iter().any(|item|matches!(item,crate::volume::RenderItem::Plane {mark:actual,..} if *actual==mark)),
            "missing registered live scalar texture for mark {mark}");
    }
    let mut runtime = rune::Runtime::new(
        state,
        crate::ui::view,
        (1440., 900.),
        rune::Env::new(false, 1.),
    );
    runtime.frame(0.);
    runtime.frame(1. / 60.);
    let elements = runtime.elements();
    for key in [
        "live-slice-z",
        "live-slice-y",
        "live-slice-x",
        "live-window-z",
        "live-window-y",
        "live-window-x",
        "pose-table",
        "metric-plot-objective",
        "metric-plot-pose_step_rms",
    ] {
        assert!(
            elements
                .iter()
                .any(|element| element.key.as_deref() == Some(key)),
            "missing native live element {key}"
        );
    }
    // A finished job leads with its results; its internal steps open as the worker trace.
    let has_task_rows = |elements: &[rune::runtime::ElementInfo]| {
        elements.iter().any(|element| {
            element
                .key
                .as_deref()
                .is_some_and(|key| key.starts_with("task-"))
        })
    };
    assert!(!has_task_rows(&elements));
    runtime.state.ui.worker_trace = Some(id.clone());
    runtime.frame(1. / 60.);
    assert!(has_task_rows(&runtime.elements()));
    let poses = runtime.state.live.frame().unwrap().poses.as_ref().unwrap();
    assert_eq!(poses.shape, [24, 6]);
    assert_eq!(
        poses
            .columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        ["alpha", "beta", "phi", "x", "z", "y"]
    );
    if let Some(evidence) = std::env::var_os("TOMOJAX_LIVE_UI_EVIDENCE") {
        let evidence = PathBuf::from(evidence);
        std::fs::create_dir_all(&evidence).unwrap();
        // The normal regression above proves a durable completed alignment.
        // Evidence additionally observes and pauses a real warm solve; no UI
        // status, scalar array, task or metric is manufactured for the capture.
        let source_id = runtime
            .state
            .project
            .as_ref()
            .unwrap()
            .document
            .jobs
            .iter()
            .find(|job| job.id == id)
            .unwrap()
            .inputs[0]
            .asset
            .clone();
        runtime.state.model.active_asset = Some(source_id.clone());
        runtime.state.model.selected_assets.clear();
        runtime.state.model.selected_assets.insert(source_id);
        runtime.state.set_operation("align",Some(&json!({
            "method":"joint","fit":"translations",
            "grid":{"nx":12,"ny":12,"nz":12,"vx":1.,"vy":1.,"vz":1.},
            "config":{"type":"AlignConfig","levels":[1],"iterations":100,"initial_image_iterations":2,
                "image_iterations":2,"refine_iterations":1,"tolerance":0.,"min_iterations":100}
        })));
        runtime.state.queue_operation("align").unwrap();
        let capture_id = runtime
            .state
            .project
            .as_ref()
            .unwrap()
            .document
            .jobs
            .last()
            .unwrap()
            .id
            .clone();
        runtime.state.ui.selected_job = None;
        let until = Instant::now() + Duration::from_secs(90);
        let mut checkpoints = std::collections::BTreeSet::new();
        let mut pause_requested = false;
        loop {
            runtime.state.poll();
            let status = runtime
                .state
                .project
                .as_ref()
                .unwrap()
                .document
                .jobs
                .iter()
                .find(|job| job.id == capture_id)
                .unwrap()
                .status;
            if let Some(frame) = runtime.state.live.frame()
                && frame.info.job == capture_id
                && frame.poses.is_some()
            {
                checkpoints.insert(frame.info.time.to_bits());
            }
            if checkpoints.len() >= 2 && status == JobStatus::Running {
                runtime.state.control_job(&capture_id, "pause").unwrap();
                pause_requested = true;
                break;
            }
            if !status.active() {
                break;
            }
            assert!(
                Instant::now() < until,
                "warm live capture did not reach two pose checkpoints"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
        if pause_requested {
            drive(&mut runtime.state, &|state| {
                state
                    .project
                    .as_ref()
                    .unwrap()
                    .document
                    .jobs
                    .iter()
                    .find(|job| job.id == capture_id)
                    .is_some_and(|job| job.status == JobStatus::Paused || !job.status.active())
            });
        }
        drive(&mut runtime.state, &|state| {
            !state.live.loading()
                && state
                    .live
                    .frame()
                    .is_some_and(|frame| frame.info.job == capture_id && frame.poses.is_some())
        });
        assert!(
            runtime.state.model.error.is_none(),
            "starting another job released a live artifact twice: {:?}",
            runtime.state.model.error
        );
        runtime.state.refresh_model();
        runtime.frame(1.);
        runtime.frame(2.);
        let job = runtime
            .state
            .project
            .as_ref()
            .unwrap()
            .document
            .jobs
            .iter()
            .find(|job| job.id == capture_id)
            .unwrap();
        assert!(
            matches!(job.status, JobStatus::Paused | JobStatus::Completed),
            "{:?}",
            job.error
        );
        if job.status == JobStatus::Paused {
            let elements = runtime.elements();
            for key in [
                format!("resume-job-{capture_id}"),
                format!("cancel-job-{capture_id}"),
                "metric-plot-objective".into(),
                "metric-plot-pose_step_rms".into(),
            ] {
                let element = elements
                    .iter()
                    .find(|element| element.key.as_deref() == Some(key.as_str()))
                    .unwrap_or_else(|| panic!("the paused native UI lost {key}"));
                assert!(
                    element.rect[1] >= 0.
                        && element.rect[1] + element.rect[3] <= 900.
                        && element.visible[2] > 0.
                        && element.visible[3] + 1. >= element.rect[3],
                    "the paused job's {key} is clipped below the usable viewport: {element:?}"
                );
            }
        }
        eprintln!(
            "Native live capture observed {:?}, {} registered pose checkpoints, pause requested: {}",
            job.status,
            checkpoints.len(),
            pause_requested
        );
        std::fs::write(evidence.join("live-capture-state.json"),serde_json::to_vec_pretty(&json!({
            "observed_status":job.status,"pose_checkpoints":checkpoints.len(),"pause_requested":pause_requested,
            "job":capture_id,"note":"Observed application state from a real worker; no replay or fabricated progress."
        })).unwrap()).unwrap();
        std::fs::write(
            evidence.join("observed-job.json"),
            serde_json::to_vec_pretty(job).unwrap(),
        )
        .unwrap();
        let planes = runtime.state.live.frame().unwrap().planes.iter().map(|(axis, plane)| {
            (axis, json!({
                "shape":plane.shape,"source_shape":plane.source_shape,
                "axes":plane.axes.iter().map(|axis| json!({"name":axis.name,"frame":axis.frame,
                    "unit":axis.unit,"origin":axis.origin,"spacing":axis.spacing})).collect::<Vec<_>>(),
                "fixed":plane.fixed,"window":plane.statistics.window,"values":plane.values.as_ref(),
            }))
        }).collect::<std::collections::BTreeMap<_,_>>();
        std::fs::write(
            evidence.join("observed-planes.json"),
            serde_json::to_vec_pretty(&planes).unwrap(),
        )
        .unwrap();
        crate::app::tests::capture_native(runtime.state, &evidence.join("live-progress.png"));
    } else {
        runtime.state.shutdown();
    }
}
