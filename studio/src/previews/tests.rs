use super::*;

fn input() -> CapturedInput {
    serde_json::from_value(
        json!({"asset":"scan-a","name":"Acquisition","path":"/source.nxs","kind":"scan"}),
    )
    .unwrap()
}

fn admitted(previews: &mut Previews, kind: PreviewKind, id: &str) {
    let intent = previews
        .slots
        .get_mut(&kind)
        .unwrap()
        .intent
        .as_mut()
        .unwrap();
    intent.request = Some(id.into());
    previews.pending.insert(
        id.into(),
        Pending::Preview(Box::new(Submission {
            kind,
            generation: intent.generation,
            input: intent.input.clone(),
            parameters: intent.parameters.clone(),
            revision: None,
            telemetry: JobProgress::default(),
        })),
    );
}

fn plane(request: &str, revision: u64, index: usize) -> Response {
    serde_json::from_value(json!({"type":"result","request":request,"channel":{"name":"preview.slice","revision":revision},
        "result":{"type":"output","value":{"type":"plane","plane":{
            "storage":{"type":"npy","path":"/temporary.npy","artifact":format!("artifact-{request}")},"shape":[2,3],"dtype":"float32",
            "axes":[{"name":"y","frame":"volume","unit":"mm","coordinates":{"type":"uniform","origin":1.,"spacing":0.5}},
                {"name":"x","frame":"volume","unit":"mm","coordinates":{"type":"uniform","origin":2.,"spacing":0.25}}],
            "unit":"1/mm","domain":"attenuation","fixed":[{"name":"z","frame":"volume","unit":"mm","index":index,"coordinate":index as f64}]}},
            "info":{"summary":{},"arrays":{}},"manifest":{"operation":"slice","parameters":{"index":index},"inputs":[]}}})).unwrap()
}

#[test]
fn an_already_piped_terminal_cannot_become_the_image_for_a_newer_intent() {
    let mut previews = Previews::default();
    previews
        .request(PreviewKind::Slice, input(), json!({"axis":"z","index":0}))
        .unwrap();
    admitted(&mut previews, PreviewKind::Slice, "old");
    previews
        .request(PreviewKind::Slice, input(), json!({"axis":"z","index":1}))
        .unwrap();
    assert!(previews.consume(plane("old", 1, 0), false).is_none());
    assert!(previews.current(PreviewKind::Slice).is_none());
    assert_eq!(
        previews.status(PreviewKind::Slice).phase,
        PreviewPhase::Debouncing
    );
    assert!(previews.take_for_keep(PreviewKind::Slice).is_err());
    assert!(previews.retired[0].artifacts.contains("artifact-old"));
    admitted(&mut previews, PreviewKind::Slice, "new");
    previews.consume(plane("new", 2, 1), false);
    let kept = previews.take_for_keep(PreviewKind::Slice).unwrap();
    assert_eq!(kept.parameters["index"], 1);
    assert_eq!(kept.revision, 2);
    assert_eq!(kept.output.manifest.parameters["index"], 1);
    assert_eq!(
        previews.retired.len(),
        1,
        "Keep transfers the current owner; it must not also retire it"
    );
}

#[test]
fn a_stale_admitted_revision_is_retired_without_becoming_keepable() {
    let mut previews = Previews::default();
    previews
        .request(PreviewKind::Slice, input(), json!({"axis":"z","index":2}))
        .unwrap();
    admitted(&mut previews, PreviewKind::Slice, "stale");
    previews
        .slots
        .get_mut(&PreviewKind::Slice)
        .unwrap()
        .revision = 5;
    previews.consume(plane("stale", 4, 2), false);
    assert!(previews.current(PreviewKind::Slice).is_none());
    assert_eq!(
        previews.status(PreviewKind::Slice).phase,
        PreviewPhase::Cancelled
    );
    assert!(previews.take_for_keep(PreviewKind::Slice).is_err());
    assert!(previews.retired[0].artifacts.contains("artifact-stale"));
}

#[test]
fn a_restarted_worker_does_not_inherit_the_previous_sessions_channel_revision() {
    let mut previews = Previews::default();
    previews
        .request(PreviewKind::Slice, input(), json!({"index":0}))
        .unwrap();
    admitted(&mut previews, PreviewKind::Slice, "previous-session");
    previews.consume(plane("previous-session", 5, 0), false);
    previews.disconnected();
    assert!(!previews.has_work());
    previews
        .request(PreviewKind::Slice, input(), json!({"index":1}))
        .unwrap();
    admitted(&mut previews, PreviewKind::Slice, "new-session");
    previews.consume(plane("new-session", 1, 1), false);
    let kept = previews.take_for_keep(PreviewKind::Slice).unwrap();
    assert_eq!(kept.revision, 1);
    assert_eq!(kept.parameters["index"], 1);
}

#[test]
fn unchanged_failed_settings_wait_for_an_explicit_retry() {
    let mut previews = Previews::default();
    let parameters = json!({"axis":"z","index":0});
    let generation = previews
        .request(PreviewKind::Slice, input(), parameters.clone())
        .unwrap();
    admitted(&mut previews, PreviewKind::Slice, "invalid");
    let response = serde_json::from_value(json!({
        "type":"error", "request":"invalid", "channel":{"name":"preview.slice","revision":1},
        "error":{"schema":2,"event":"error","kind":"invalid_parameter","message":"Choose a supported interpolation", "path":"config.interpolation"}
    })).unwrap();
    previews.consume(response, false);
    assert_eq!(
        previews.status(PreviewKind::Slice).phase,
        PreviewPhase::Failed
    );
    assert_eq!(
        previews
            .request(PreviewKind::Slice, input(), parameters.clone())
            .unwrap(),
        generation
    );
    assert_eq!(
        previews.status(PreviewKind::Slice).phase,
        PreviewPhase::Failed
    );
    assert_eq!(
        previews.status(PreviewKind::Slice).error_path.as_deref(),
        Some("config.interpolation")
    );
    previews.retry(PreviewKind::Slice);
    assert!(
        !previews.has_priority_work(),
        "An explicit retry with no parseable replacement parameters must not block validation"
    );
    assert!(
        previews
            .request(PreviewKind::Slice, input(), parameters)
            .unwrap()
            > generation
    );
    admitted(&mut previews, PreviewKind::Slice, "retried");
    previews.consume(plane("retried", 2, 0), false);
    assert!(previews.take_for_keep(PreviewKind::Slice).is_ok());
}

#[test]
fn keeping_a_preview_transfers_its_actual_worker_notes_and_metrics() {
    let mut previews = Previews::default();
    previews
        .request(PreviewKind::Slice, input(), json!({"index":1}))
        .unwrap();
    admitted(&mut previews, PreviewKind::Slice, "computed");
    for record in [
        json!({"event":"started","schema":2,"task":1,"time":0.,"parent":null,"name":"slice","label":"Slice","total":1,"unit":"planes"}),
        json!({"event":"advanced","schema":2,"task":1,"time":0.1,"done":1,"total":1,"metrics":{"loss":2.5},"eta":null}),
        json!({"event":"note","schema":2,"task":1,"time":0.2,"level":"warning","message":"Recorded acquisition warning"}),
        json!({"event":"finished","schema":2,"task":1,"time":0.3,"status":"done"}),
    ] {
        let response = serde_json::from_value(json!({"type":"progress","request":"computed", "channel":{"name":"preview.slice","revision":1},"record":record})).unwrap();
        previews.consume(response, false);
    }
    previews.consume(plane("computed", 1, 1), false);
    let kept = previews.take_for_keep(PreviewKind::Slice).unwrap();
    assert_eq!(
        kept.telemetry.notes[0].message,
        "Recorded acquisition warning"
    );
    assert_eq!(kept.telemetry.notes[0].request, "computed");
    assert_eq!(
        kept.telemetry.tasks[0].state,
        crate::progress::TaskState::Done
    );
    assert_eq!(
        kept.telemetry.tasks[0].metrics["loss"].samples[0].value,
        json!(2.5)
    );
    assert_eq!(
        kept.telemetry.tasks[0].metrics["loss"].unit.as_deref(),
        Some("objective units")
    );
}

#[test]
fn explicit_form_inputs_remain_warm_while_only_two_idle_sources_are_cached() {
    let mut previews = Previews::default();
    let inputs: Vec<_> = (0..5)
        .map(|index| {
            let mut input = input();
            input.asset = format!("scan-{index}");
            input
        })
        .collect();
    previews.retain_inputs(&inputs).unwrap();
    let first = previews.ensure_input(inputs[0].clone()).unwrap();
    assert_eq!(first, previews.ensure_input(inputs[0].clone()).unwrap());
    assert!(previews.sources.trim(&BTreeSet::new()).is_empty());
    assert_eq!(previews.sources.entries.len(), 5);
    previews.retain_inputs(&[]).unwrap();
    assert_eq!(previews.sources.trim(&BTreeSet::new()).len(), 3);
    assert_eq!(previews.sources.entries.len(), 2);
    let mut changed = inputs[0].clone();
    changed.path = "/data/other-scan.nxs".into();
    assert_ne!(first, previews.ensure_input(changed).unwrap());
}
