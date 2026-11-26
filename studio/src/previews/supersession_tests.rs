//! Random slider edits against a fake worker channel that admits, rejects and answers
//! requests in any order, checked after every step: a superseded answer never replaces
//! the preview of newer settings.
use super::*;
use proptest::prelude::*;

#[derive(Clone, Debug)]
enum Step {
    /// The user moves the slice slider to this index.
    Edit(u64),
    /// The frame poll sends the current settings on the channel.
    Submit,
    /// The worker admits a sent request, taking the channel's next revision.
    Admit(usize),
    /// The worker refuses a sent request without taking a revision.
    Reject(usize),
    /// The worker answers an admitted request.
    Answer(usize),
}

fn step() -> impl Strategy<Value = Step> {
    prop_oneof![
        3 => (0..4u64).prop_map(Step::Edit),
        3 => Just(Step::Submit),
        3 => (0..4usize).prop_map(Step::Admit),
        1 => (0..4usize).prop_map(Step::Reject),
        3 => (0..4usize).prop_map(Step::Answer),
    ]
}

fn input() -> CapturedInput {
    serde_json::from_value(
        json!({"asset":"scan-a","name":"Acquisition","path":"/source.nxs","kind":"scan"}),
    )
    .unwrap()
}

fn answer(request: &str, revision: u64, index: u64) -> Response {
    serde_json::from_value(json!({"type":"result","request":request,"channel":{"name":"preview.slice","revision":revision},
        "result":{"type":"output","value":{"type":"plane","plane":{
            "storage":{"type":"npy","path":"/temporary.npy","artifact":format!("artifact-{request}")},"shape":[2,3],"dtype":"float32",
            "axes":[{"name":"y","frame":"volume","unit":"mm","coordinates":{"type":"uniform","origin":0.,"spacing":1.}},
                {"name":"x","frame":"volume","unit":"mm","coordinates":{"type":"uniform","origin":0.,"spacing":1.}}],
            "unit":"1/mm","domain":"attenuation","fixed":[{"name":"z","frame":"volume","unit":"mm","index":index,"coordinate":index as f64}]}},
            "info":{"summary":{},"arrays":{}},"manifest":{"operation":"slice","parameters":{"index":index},"inputs":[]}}}))
    .unwrap()
}

/// What the worker holds: requests sent but not yet admitted, admitted ones with their
/// revision, and the channel's latest admitted revision.
#[derive(Default)]
struct Worker {
    sent: Vec<(String, u64)>,
    admitted: Vec<(String, u64, u64)>,
    revision: u64,
    next_id: usize,
}

fn run(steps: Vec<Step>) {
    let kind = PreviewKind::Slice;
    let mut previews = Previews::default();
    let mut worker = Worker::default();
    let mut latest = None;
    let mut published: Option<(String, u64)> = None;
    for step in steps {
        match step {
            Step::Edit(index) => {
                previews
                    .request(kind, input(), json!({"axis":"z","index":index}))
                    .unwrap();
                latest = Some(index);
            }
            Step::Submit => {
                // What `Previews::poll` records when the engine accepts the request.
                if let Some(slot) = previews.slots.get_mut(&kind)
                    && let Some(intent) = slot.intent.as_mut()
                    && intent.request.is_none()
                {
                    worker.next_id += 1;
                    let id = format!("request-{}", worker.next_id);
                    intent.request = Some(id.clone());
                    previews.pending.insert(
                        id.clone(),
                        Pending::Preview(Box::new(Submission {
                            kind,
                            generation: intent.generation,
                            input: intent.input.clone(),
                            parameters: Value::Object(intent.parameters.clone()),
                            revision: None,
                            telemetry: JobProgress::default(),
                        })),
                    );
                    worker
                        .sent
                        .push((id, intent.parameters["index"].as_u64().unwrap()));
                }
            }
            Step::Admit(pick) if !worker.sent.is_empty() => {
                let (id, index) = worker.sent.remove(pick % worker.sent.len());
                worker.revision += 1;
                worker.admitted.push((id, index, worker.revision));
            }
            Step::Reject(pick) if !worker.sent.is_empty() => {
                let (id, _) = worker.sent.remove(pick % worker.sent.len());
                let refusal = serde_json::from_value(json!({"type":"protocol_error",
                    "kind":"busy","message":"The channel is full","request":id}))
                .unwrap();
                previews.consume(refusal, false);
            }
            Step::Answer(pick) if !worker.admitted.is_empty() => {
                let (id, index, revision) = worker.admitted.remove(pick % worker.admitted.len());
                // The engine marks an answer stale once a newer request was admitted.
                previews.consume(answer(&id, revision, index), revision < worker.revision);
            }
            Step::Admit(_) | Step::Reject(_) | Step::Answer(_) => {}
        }

        let current = previews
            .current(kind)
            .map(|current| (current.request.clone(), current.revision));
        if current != published {
            let (request, revision) = current.clone().expect("a published preview stays");
            let shown = previews.current(kind).unwrap();
            assert_eq!(
                shown.parameters["index"].as_u64(),
                latest,
                "{request} was published for older settings"
            );
            assert!(
                published
                    .as_ref()
                    .is_none_or(|(_, previous)| revision > *previous),
                "{request} at revision {revision} replaced revision {published:?}"
            );
        }
        published = current;
        // Only a preview of the latest settings can be kept.
        if previews.status(kind).phase == PreviewPhase::Ready {
            let shown = previews
                .current(kind)
                .expect("a ready preview has an image");
            assert_eq!(shown.parameters["index"].as_u64(), latest);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 512,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn superseded_answers_never_replace_newer_previews(
        steps in proptest::collection::vec(step(), 1..40)
    ) {
        run(steps);
    }
}
