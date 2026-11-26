use super::*;
use crate::engine::Engine;
use crate::project::{Asset, Project};
use crate::protocol::{Info, Output, Provenance};
use crate::ui::model::ProjectView;
use rune::{Env, Runtime};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

fn plan() -> Product {
    serde_json::from_value(json!({
        "type":"reconstruction_plan", "method":"fourier", "reasons":["Recorded worker reason"],
        "iterations":null, "grid":{"nx":12,"ny":10,"nz":8,"vx":0.1,"vy":0.2,"vz":0.3},
        "placement":{"devices":["cpu:0"],"strategy":"resident","host_bytes":4096,"device_bytes":8192,"snapshot":[]}
    })).unwrap()
}

#[test]
fn edits_debounce_and_clear_a_plan_before_the_next_worker_revision_exists() {
    let mut controller = Controller::default();
    let start = Instant::now();
    assert!(controller.observe(json!({"iterations":"3"}), start));
    assert!(!controller.settled(start + Duration::from_millis(349)));
    assert!(controller.settled(start + DEBOUNCE));
    controller.view.plan = Some(plan());
    controller.submitted("old-plan".into(), Work::Plan);
    let request = controller.requests["old-plan"];
    assert!(controller.observe(json!({"iterations":"4"}), start + DEBOUNCE));
    assert!(controller.view.plan.is_none());
    assert!(!controller.current("old-plan", request, false));
    assert!(!controller.settled(start + DEBOUNCE + Duration::from_millis(349)));
}

#[test]
fn already_piped_result_is_discarded_when_the_text_changed_before_poll() {
    let mut state = AppState::init();
    state.ui.workspace = Workspace::Reconstruct;
    state.set_operation("reconstruct", None);
    state
        .validation
        .observe(state.validation_signature(), Instant::now());
    state
        .validation
        .submitted("already-piped".into(), Work::Plan);
    state
        .ui
        .parameter_edits
        .get_mut("iterations")
        .unwrap()
        .replace_text("19");
    let event = EngineEvent::Reply {
        response: Response::Result {
            request: "already-piped".into(),
            channel: None,
            result: ResultValue::Output(Box::new(Output {
                value: plan(),
                info: Info {
                    summary: BTreeMap::new(),
                    arrays: BTreeMap::new(),
                },
                manifest: Provenance {
                    operation: "plan".into(),
                    parameters: json!({}),
                    inputs: vec![],
                },
            })),
        },
        stale: false,
    };
    assert!(state.validation_event(&event));
    assert!(state.model.validation.plan.is_none());
    assert_eq!(state.model.validation.phase, Phase::Editing);
    assert!(state.validation.requests.is_empty());
}

#[test]
fn changing_project_keeps_ownership_of_an_old_validation_rejection() {
    let mut state = AppState::init();
    state
        .validation
        .observe(state.validation_signature(), Instant::now());
    state
        .validation
        .submitted("old-project-check".into(), Work::Validate);
    state.validation.reset();
    assert!(state.validation_event(&EngineEvent::Reply {
        response: Response::ProtocolError {
            kind: "invalid_request".into(),
            message: "Rejected old project input".into(),
            request: Some("old-project-check".into()),
        },
        stale: false,
    }));
    assert!(state.model.error.is_none());
    assert!(state.model.validation.issues.is_empty());
    assert!(state.validation.requests.is_empty());
}

#[test]
fn a_relinked_asset_cannot_reuse_a_plan_from_its_previous_source() {
    let directory = tempfile::tempdir().unwrap();
    let first = directory.path().join("original.nxs");
    let replacement = directory.path().join("replacement.nxs");
    std::fs::write(&first, b"first captured source").unwrap();
    std::fs::write(&replacement, b"different bytes and metadata").unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(directory.path(), true).unwrap());
    state.ui.workspace = Workspace::Reconstruct;
    state.set_operation("reconstruct", None);
    let asset = Asset::input(first, false);
    let id = asset.id.clone();
    state.project.as_mut().unwrap().document.assets.push(asset);
    state.model.active_asset = Some(id.clone());
    state.model.selected_assets.insert(id.clone());
    state.poll_form_context();
    let original = state.current_form_input().unwrap();
    state
        .validation
        .observe(state.validation_signature(), Instant::now());
    state.validation.input_context = Some(json!(original));
    state
        .validation
        .submitted("original-plan".into(), Work::Plan);
    let reply = |id: &str| EngineEvent::Reply {
        response: Response::Result {
            request: id.into(),
            channel: None,
            result: ResultValue::Output(Box::new(Output {
                value: plan(),
                info: Info {
                    summary: BTreeMap::new(),
                    arrays: BTreeMap::new(),
                },
                manifest: Provenance {
                    operation: "plan".into(),
                    parameters: json!({}),
                    inputs: vec![],
                },
            })),
        },
        stale: false,
    };
    assert!(state.validation_event(&reply("original-plan")));
    assert_eq!(state.model.validation.plan_inputs, Some(json!(original)));
    state
        .validation
        .submitted("old-source-plan".into(), Work::Plan);
    state
        .project
        .as_mut()
        .unwrap()
        .document
        .asset_mut(&id)
        .unwrap()
        .path = replacement;
    state.poll_form_context();
    let changed = state.current_form_input().unwrap();
    assert_eq!(changed.asset, original.asset);
    assert_ne!(changed.path, original.path);
    assert!(state.validation_event(&reply("old-source-plan")));
    assert!(state.model.validation.plan.is_none());
    assert!(state.model.validation.plan_inputs.is_none());
}

#[test]
fn nested_error_reveals_its_actual_editor_and_the_review_action_focuses_it() {
    let mut state = AppState::init();
    state.model.project = Some(ProjectView {
        name: "Validation test".into(),
        ..Default::default()
    });
    state.ui.workspace = Workspace::Reconstruct;
    state.set_operation(
        "reconstruct",
        Some(&json!({"config":{"type":"CGLSConfig","interpolation":"cubic"}})),
    );
    state.validation_error(DraftError::new(
        "config.interpolation",
        "Cubic interpolation is unavailable for this geometry.",
    ));
    state.model.validation = state.validation.view.clone();
    assert!(state.ui.advanced_open);
    let mut runtime = Runtime::new(state, crate::ui::view, (1180., 780.), Env::new(false, 1.));
    runtime.frame(0.);
    runtime.frame(1. / 60.);
    let review = runtime
        .elements()
        .into_iter()
        .find(|element| element.key.as_deref() == Some("review-error-config.interpolation"))
        .unwrap();
    let (x, y) = (
        review.rect[0] + review.rect[2] / 2.,
        review.rect[1] + review.rect[3] / 2.,
    );
    runtime.pointer_move(x, y);
    runtime.pointer_down(x, y);
    runtime.pointer_up(x, y);
    runtime.frame(2. / 60.);
    runtime.frame(3. / 60.);
    let field = runtime
        .elements()
        .into_iter()
        .find(|element| element.key.as_deref() == Some("parameter-config"))
        .unwrap();
    assert!(
        field.visible[2] > 80. && field.visible[3] > 10.,
        "{field:?}"
    );
    let message = runtime
        .elements()
        .into_iter()
        .find(|element| element.key.as_deref() == Some("validation-message-config.interpolation"))
        .expect("path-specific validation message");
    assert!(
        message
            .text
            .as_ref()
            .unwrap()
            .contains("config.interpolation: Cubic interpolation")
    );
    assert!(
        message.visible[3] >= message.rect[3] - 1.,
        "the error text must be fully visible with its editor: {message:?}"
    );
}

fn drive(state: &mut AppState, condition: impl Fn(&AppState) -> bool) {
    let until = Instant::now() + Duration::from_secs(50);
    while !condition(state) {
        assert!(
            Instant::now() < until,
            "validation timed out: {:?}; error={:?}; jobs={:?}",
            state.model.validation,
            state.model.error,
            state.model.jobs
        );
        state.poll();
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
#[ignore = "requires the real TomoJAX worker; run in the release ignored gate"]
fn real_worker_nested_errors_plan_refresh_and_already_piped_reply_rejection() {
    let python =
        PathBuf::from(std::env::var_os("TOMOJAX_TEST_PYTHON").expect("set TOMOJAX_TEST_PYTHON"));
    let directory = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(directory.path(), true).unwrap());
    state.project.as_mut().unwrap().document.name = "Cone geometry validation".into();
    state.ui.workspace = Workspace::Explore;
    state.startup_input = None;
    state.demo = crate::app::Demo::Off;
    state.engine = Some(Engine::spawn(python, Arc::new(|| {})).unwrap());
    drive(&mut state, |state| {
        state.model.backend.ready && state.engine.as_ref().unwrap().catalog().is_some()
    });
    state.set_operation(
        "simulate",
        Some(&json!({"size":6,"n_views":12,"geometry":"cone","rotation":360,"seed":9})),
    );
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
    assert_eq!(state.project.as_ref().unwrap().document.assets.len(), 1);
    state.ui.workspace = Workspace::Reconstruct;
    state.set_operation("reconstruct", Some(&json!({"method":"cgls","config":{"type":"CGLSConfig","backend":"jax","interpolation":"cubic","iterations":0}})));
    drive(&mut state, |state| {
        state.model.validation.phase == Phase::Invalid
    });
    let issue = &state.model.validation.issues[0];
    assert_eq!(issue.path, "config.interpolation");
    assert_eq!(issue.control.as_deref(), Some("parameter-config"));
    let reconstruction_error = json!({
        "parameters": forms::parameters(&state.ui, &state.catalog).unwrap(),
        "path": issue.path, "message": issue.message, "control": issue.control,
    });
    assert!(state.ui.advanced_open);
    assert!(state.model.validation.plan.is_none());
    let queue_error = state
        .queue_operation("reconstruct")
        .expect_err("known-invalid settings were queued");
    assert!(
        queue_error.contains("config.interpolation"),
        "{queue_error}"
    );
    assert_eq!(
        state.project.as_ref().unwrap().document.jobs.len(),
        1,
        "preflight must not create durable jobs"
    );

    state
        .ui
        .parameter_edits
        .get_mut("config")
        .unwrap()
        .replace_text(
            r#"{"type":"CGLSConfig","backend":"jax","interpolation":"linear","iterations":2}"#,
        );
    assert!(
        state.current_validation_error("reconstruct").is_none(),
        "corrected text inherited the old queue blocker"
    );
    state.poll();
    drive(&mut state, |state| {
        matches!(
            state.model.validation.plan,
            Some(Product::ReconstructionPlan {
                iterations: Some(2),
                ..
            })
        )
    });
    let Product::ReconstructionPlan {
        method,
        grid,
        placement,
        ..
    } = state.model.validation.plan.as_ref().unwrap()
    else {
        unreachable!()
    };
    assert_eq!(method, "cgls");
    assert!(grid.nx > 0 && grid.ny > 0 && grid.nz > 0);
    assert!(placement.host_bytes > 0 || placement.device_bytes > 0);
    let reconstruction_plan = json!({
        "parameters": forms::parameters(&state.ui, &state.catalog).unwrap(),
        "product": state.model.validation.plan,
    });

    state
        .ui
        .parameter_edits
        .get_mut("config")
        .unwrap()
        .replace_text(
            r#"{"type":"CGLSConfig","backend":"jax","interpolation":"cubic","iterations":2}"#,
        );
    state.poll();
    drive(&mut state, |state| state.validation.in_flight.is_some());
    let obsolete = state.validation.in_flight.clone().unwrap();
    state
        .ui
        .parameter_edits
        .get_mut("config")
        .unwrap()
        .replace_text(
            r#"{"type":"CGLSConfig","backend":"jax","interpolation":"linear","iterations":3}"#,
        );
    // Drain the real terminal before the normal form poll sees the edit.
    let until = Instant::now() + Duration::from_secs(15);
    let mut terminal = false;
    while !terminal {
        assert!(Instant::now() < until, "old validation did not terminate");
        for event in state.engine.as_mut().unwrap().poll() {
            terminal |= matches!(&event, EngineEvent::Reply { response, .. } if response.request() == Some(obsolete.as_str()) && response.terminal());
            state.engine_event(event);
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        state.model.validation.issues.is_empty(),
        "obsolete cubic error must not mark the edited linear field"
    );
    assert!(state.model.validation.plan.is_none());
    drive(&mut state, |state| {
        matches!(
            state.model.validation.plan,
            Some(Product::ReconstructionPlan {
                iterations: Some(3),
                ..
            })
        )
    });

    state.ui.workspace = Workspace::Prepare;
    state.set_operation(
        "correct",
        Some(&json!({"corrections":[
            {"type":"Rings","views":null}, {"type":"Zingers","threshold":-1.0}
        ]})),
    );
    state.poll();
    drive(&mut state, |state| {
        state.model.validation.phase == Phase::Invalid
    });
    let issue = &state.model.validation.issues[0];
    assert_eq!(issue.path, "corrections[1].threshold");
    assert_eq!(
        issue.control.as_deref(),
        Some("correction-Zingers.threshold")
    );
    let correction_error = json!({
        "parameters": forms::parameters(&state.ui, &state.catalog).unwrap(),
        "path": issue.path, "message": issue.message, "control": issue.control,
    });
    assert_eq!(state.project.as_ref().unwrap().document.jobs.len(), 1);
    if let Some(directory) = std::env::var_os("TOMOJAX_VALIDATION_EVIDENCE") {
        let directory = PathBuf::from(directory);
        std::fs::create_dir_all(&directory).unwrap();
        state.ui.workspace = Workspace::Reconstruct;
        state.set_operation("reconstruct", Some(&reconstruction_error["parameters"]));
        state.poll();
        drive(&mut state, |state| {
            state.model.validation.phase == Phase::Invalid
        });
        assert_eq!(
            state.model.validation.issues[0].path,
            "config.interpolation"
        );
        let project = state.project.as_ref().unwrap();
        let asset = &project.document.assets[0];
        let dataset = directory.join("validation-cone.nxs");
        std::fs::copy(project.resolve(&asset.path), &dataset).unwrap();
        let evidence = json!({
            "description":"Captured from the official worker during the real validation regression; suitable for explicitly labeled UI replay.",
            "input":{"path":dataset,"kind":asset.kind,"name":"Validation cone fixture","geometry":"cone","size":6,"views":12},
            "reconstruction_error":reconstruction_error,
            "reconstruction_plan":reconstruction_plan,
            "correction_error":correction_error,
        });
        std::fs::write(
            directory.join("validation.json"),
            serde_json::to_vec_pretty(&evidence).unwrap(),
        )
        .unwrap();
        crate::app::tests::capture_native(state, &directory.join("validation.png"));
    } else {
        state.shutdown();
    }
}

#[test]
fn planning_support_comes_from_the_public_union() {
    let catalog = SchemaCatalog::new(
        serde_json::from_str(include_str!(
            "../../../../../../docs/protocol/operations.schema.json"
        ))
        .unwrap(),
    )
    .unwrap();
    assert!(accepts_operation(&catalog, "Validate", "correct"));
    assert!(accepts_operation(&catalog, "Validate", "open"));
    assert!(accepts_operation(&catalog, "Plan", "reconstruct"));
    assert!(!accepts_operation(&catalog, "Plan", "correct"));
}
