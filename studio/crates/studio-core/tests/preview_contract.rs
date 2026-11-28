//! Exercises the real interactive owner against the unchanged official worker.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use studio_core::engine::{Engine, EngineEvent};
use studio_core::previews::{PreviewKind, PreviewPhase, Previews};
use studio_core::project::{Asset, CapturedInput};
use studio_core::protocol::{ErrorKind, Output, Product, Response, ResultValue, Storage};

struct Client {
    engine: Engine,
    previews: Previews,
    terminals: HashMap<String, Response>,
    published: BTreeMap<PreviewKind, Vec<String>>,
    superseded: Vec<String>,
    pinned: BTreeSet<String>,
    defer_terminal: Option<String>,
    deferred: Option<Response>,
}

impl Client {
    fn start() -> Self {
        let python = std::env::var_os("TOMOJAX_TEST_PYTHON").expect("set TOMOJAX_TEST_PYTHON");
        let mut client = Self {
            engine: Engine::spawn_through(PathBuf::from(python), &[], Arc::new(|| {})).unwrap(),
            previews: Previews::default(),
            terminals: HashMap::new(),
            published: BTreeMap::new(),
            superseded: Vec::new(),
            pinned: BTreeSet::new(),
            defer_terminal: None,
            deferred: None,
        };
        client.until(|client| client.engine.catalog().is_some());
        client
    }

    fn pump(&mut self) {
        for event in self.engine.poll() {
            match event {
                EngineEvent::Reply { response, stale } => {
                    if response.terminal() && response.request() == self.defer_terminal.as_deref() {
                        self.deferred = Some(response);
                        continue;
                    }
                    if let Response::Error { request, error, .. } = &response
                        && matches!(error.kind, ErrorKind::Superseded)
                    {
                        self.superseded.push(request.clone());
                    }
                    if let Some(response) = self.previews.consume(response, stale)
                        && response.terminal()
                    {
                        self.terminals
                            .insert(response.request().unwrap().into(), response);
                    }
                }
                EngineEvent::Disconnected { reason, .. } => {
                    panic!("unexpected disconnect: {reason}")
                }
                EngineEvent::Diagnostic(message) => eprintln!("worker: {message}"),
                EngineEvent::Ready(_) => {}
            }
        }
        self.previews.poll(&mut self.engine);
        let pinned = &self.pinned;
        self.previews
            .collect_retired(&mut self.engine, |tokens| !tokens.is_disjoint(pinned));
        for kind in [PreviewKind::Slice, PreviewKind::Centers] {
            if let Some(current) = self.previews.current(kind) {
                let history = self.published.entry(kind).or_default();
                if history.last() != Some(&current.request) {
                    history.push(current.request.clone());
                }
            }
        }
    }

    fn until(&mut self, condition: impl Fn(&Self) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while !condition(self) {
            self.pump();
            assert!(
                Instant::now() < deadline,
                "worker timeout; slice={:?}",
                self.previews.status(PreviewKind::Slice)
            );
            thread::sleep(Duration::from_millis(2));
        }
    }

    fn terminal(&mut self, command: Value) -> Response {
        let id = self.engine.request(command, None).unwrap();
        self.until(|client| client.terminals.contains_key(&id));
        self.terminals.remove(&id).unwrap()
    }

    fn output(&mut self, command: Value) -> Box<Output> {
        match self.terminal(command) {
            Response::Result {
                result: ResultValue::Output(output),
                ..
            } => output,
            other => panic!("expected output: {other:?}"),
        }
    }

    fn status(&mut self) -> BTreeMap<String, String> {
        match self.terminal(json!({"type":"status"})) {
            Response::Result {
                result: ResultValue::Status { requests, .. },
                ..
            } => requests,
            other => panic!("expected status: {other:?}"),
        }
    }
}

#[test]
#[ignore = "real worker superseding and artifact ownership; set TOMOJAX_TEST_PYTHON"]
fn interactive_channels_supersede_queued_work_keep_only_latest_and_retain_open_source() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("input.nxs");
    let mut client = Client::start();
    let simulation=client.output(json!({"type":"run","operation":{"type":"simulate","phantom":"sphere","size":16,"n_views":32}}));
    client.output(json!({"type":"run","operation":{"type":"save","source":{"type":"member","handle":simulation.value.handle().unwrap(),"member":"self"},"path":path,"include_scan":true}}));
    let mut asset = Asset::input(path.clone(), false);
    asset.kind = "simulation".into();
    let input = CapturedInput::from_asset(&asset, directory.path());
    client
        .previews
        .retain_inputs(std::slice::from_ref(&input))
        .unwrap();
    client.until(|client| client.previews.source_ref(&input).unwrap().is_some());
    let source = client.previews.source_ref(&input).unwrap().unwrap();
    let holding=client.engine.request(json!({"type":"run","operation":{"type":"reconstruct","source":source,"method":"cgls","iterations":10000,"config":{"type":"CGLSConfig","tolerance":0.0,"absolute_tolerance":0.0}}}),None).unwrap();
    client.terminal(json!({"type":"control","target":holding,"action":"pause"}));
    let deadline = Instant::now() + Duration::from_secs(60);
    while client.status().get(&holding).map(String::as_str) != Some("paused") {
        assert!(Instant::now() < deadline);
    }

    let mut dispatched = Vec::new();
    for index in [1, 2, 5] {
        client
            .previews
            .request(
                PreviewKind::Slice,
                input.clone(),
                json!({"axis":"z","index":index,"method":"fourier"}),
            )
            .unwrap();
        client.until(|client| client.previews.status(PreviewKind::Slice).request.is_some());
        dispatched.push(client.previews.status(PreviewKind::Slice).request.unwrap());
    }
    client
        .previews
        .request(
            PreviewKind::Centers,
            input.clone(),
            json!({"axis":"z","index":5,"centers":[-1.0,0.0,0.0,1.0],"method":"fourier"}),
        )
        .unwrap();
    client.until(|client| {
        client
            .previews
            .status(PreviewKind::Centers)
            .request
            .is_some()
            && client.previews.status(PreviewKind::Slice).blocked_by_paused
    });
    assert!(client.previews.current(PreviewKind::Slice).is_none());
    assert_eq!(
        client.status()[&holding],
        "paused",
        "interactive work must not resume or cancel unrelated paused work"
    );
    assert_eq!(
        client.previews.source_ref(&input).unwrap(),
        Some(source.clone())
    );

    client.terminal(json!({"type":"control","target":holding,"action":"cancel"}));
    client.until(|client| {
        client.previews.status(PreviewKind::Slice).phase == PreviewPhase::Ready
            && client.previews.status(PreviewKind::Centers).phase == PreviewPhase::Ready
    });
    let current = client.previews.current(PreviewKind::Slice).unwrap();
    assert_eq!(current.parameters["index"], 5);
    assert_eq!(
        current.output.value.primary_array().unwrap().fixed[0].index,
        5
    );
    assert_eq!(
        client.published[&PreviewKind::Slice],
        vec![dispatched[2].clone()]
    );
    assert!(
        dispatched[..2]
            .iter()
            .all(|request| client.superseded.contains(request))
    );
    assert_eq!(client.previews.source_ref(&input).unwrap(), Some(source));
    assert!(current.revision >= 3);
    let Product::Sweep { planes } = &client
        .previews
        .current(PreviewKind::Centers)
        .unwrap()
        .output
        .value
    else {
        panic!("expected center sweep")
    };
    assert_eq!(planes.shape, [4, 16, 16]);
    assert_eq!(planes.axes[0].name, "center");

    // The actual worker may finish just before the pointer changes again. Hold
    // that real terminal in the client event queue, edit locally before the next
    // admission, and verify it cannot become the new image or be kept.
    client
        .previews
        .request(
            PreviewKind::Slice,
            input.clone(),
            json!({"axis":"z","index":6,"method":"fourier"}),
        )
        .unwrap();
    client.until(|client| client.previews.status(PreviewKind::Slice).request.is_some());
    client.defer_terminal = client.previews.status(PreviewKind::Slice).request;
    client.until(|client| client.deferred.is_some());
    let held = client.deferred.take().unwrap();
    let Response::Result {
        result: ResultValue::Output(output),
        ..
    } = &held
    else {
        panic!("expected completed real slice before local edit: {held:?}");
    };
    let Storage::Npy {
        path: stale_path, ..
    } = &output.value.primary_array().unwrap().storage
    else {
        panic!("expected temporary plane storage");
    };
    let stale_path = PathBuf::from(stale_path);
    assert!(stale_path.is_file());
    client.defer_terminal = None;
    client
        .previews
        .request(
            PreviewKind::Slice,
            input,
            json!({"axis":"z","index":7,"method":"fourier"}),
        )
        .unwrap();
    assert!(client.previews.consume(held, false).is_none());
    assert_eq!(
        client
            .previews
            .current(PreviewKind::Slice)
            .unwrap()
            .parameters["index"],
        5
    );
    assert!(client.previews.take_for_keep(PreviewKind::Slice).is_err());
    client.until(|client| client.previews.status(PreviewKind::Slice).phase == PreviewPhase::Ready);
    assert_eq!(
        client
            .previews
            .current(PreviewKind::Slice)
            .unwrap()
            .parameters["index"],
        7
    );
    client.until(|_| !stale_path.exists());

    let kept = client.previews.take_for_keep(PreviewKind::Slice).unwrap();
    assert_eq!(kept.parameters["index"], 7);
    assert_eq!(kept.output.manifest.operation, "slice");
    let warm_input = kept.input.clone();
    let plane = kept.output.value.primary_array().unwrap();
    let Storage::Npy {
        path: plane_path,
        artifact: Some(token),
    } = &plane.storage
    else {
        panic!("expected leased npy")
    };
    let saved = PathBuf::from(plane_path);
    let token = token.clone();
    client.pinned.insert(token.clone());
    client
        .previews
        .retire_resources(BTreeSet::from([token]), None);
    client.status();
    assert!(
        saved.is_file(),
        "a retired preview must outlive an active native reader"
    );
    client.pinned.clear();
    client.until(|_| !saved.exists());
    // Completed center-sweep pixels remain owned independently of their source.
    // Moving away from that form must put its open scan under the idle limit.
    client.previews.retain_inputs(&[]).unwrap();
    for index in 0..3 {
        let mut other = warm_input.clone();
        other.asset = format!("another-import-{index}");
        client.previews.ensure_input(other).unwrap();
    }
    client.pump();
    assert!(client.previews.source_ref(&warm_input).unwrap().is_none());
    assert!(client.previews.current(PreviewKind::Centers).is_some());
    client.previews.reset();
    client.until(|client| !client.previews.has_work());
    let ending = client.engine.drain_shutdown(Duration::from_secs(5));
    assert!(ending.iter().any(|event|matches!(event,EngineEvent::Disconnected {code:Some(0),unfinished,..} if unfinished.is_empty())));
    assert!(
        Path::new(&path).is_file(),
        "caller-owned input must survive shutdown"
    );
}
