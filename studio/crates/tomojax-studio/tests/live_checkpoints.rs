//! The official numerical worker supplies the images, poses, registration and metrics.
//! No native numerical approximation participates in this regression.
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use studio_core::engine::{Engine, EngineEvent};
use studio_core::progress::JobProgress;
use studio_core::protocol::{
    ArrayDescriptor, Coordinates, Output, Progress, Response, ResultValue, Storage,
};
use tomojax_studio::live::LiveCheckpoints;

struct Client {
    engine: Engine,
    live: LiveCheckpoints,
    terminals: BTreeMap<String, Response>,
    progress: BTreeMap<String, JobProgress>,
    arrays: BTreeMap<String, BTreeMap<String, ArrayDescriptor>>,
    commands: BTreeMap<String, Value>,
    records: BTreeMap<String, Vec<Value>>,
    paths: BTreeMap<String, PathBuf>,
    released: BTreeSet<String>,
    cleanup: BTreeSet<String>,
    closed: BTreeSet<String>,
}
impl Drop for Client {
    fn drop(&mut self) {
        // A failed assertion must still stop file readers before Engine's drop
        // shuts down the process that owns their backing artifacts.
        self.live.shutdown();
    }
}
impl Client {
    fn new() -> Self {
        let python = std::env::var_os("TOMOJAX_TEST_PYTHON").expect("set TOMOJAX_TEST_PYTHON");
        let mut client = Self {
            engine: Engine::spawn(Path::new(&python), Arc::new(|| {})).unwrap(),
            live: LiveCheckpoints::new(),
            terminals: BTreeMap::new(),
            progress: BTreeMap::new(),
            arrays: BTreeMap::new(),
            commands: BTreeMap::new(),
            records: BTreeMap::new(),
            paths: BTreeMap::new(),
            released: BTreeSet::new(),
            cleanup: BTreeSet::new(),
            closed: BTreeSet::new(),
        };
        let until = Instant::now() + Duration::from_secs(45);
        while client.engine.catalog().is_none() {
            client.pump();
            assert!(Instant::now() < until, "worker startup timed out");
            std::thread::sleep(Duration::from_millis(2));
        }
        client
    }
    fn send(&mut self, command: Value) -> String {
        let request = self.engine.request(command.clone(), None).unwrap();
        self.commands.insert(request.clone(), command);
        request
    }
    fn send_cleanup(&mut self, command: Value) {
        let request = self.send(command);
        self.cleanup.insert(request);
    }
    fn pump(&mut self) {
        for event in self.engine.poll() {
            match event {
                EngineEvent::Reply { response, stale } => {
                    assert!(!stale);
                    if let Some(request) = response.request() {
                        self.records
                            .entry(request.into())
                            .or_default()
                            .push(serde_json::to_value(&response).unwrap());
                    }
                    if response.terminal()
                        && response
                            .request()
                            .is_some_and(|request| self.cleanup.remove(request))
                    {
                        assert!(
                            matches!(
                                response,
                                Response::Result {
                                    result: ResultValue::Ack,
                                    ..
                                }
                            ),
                            "worker rejected artifact/handle cleanup: {response:?}"
                        );
                        continue;
                    }
                    if let Response::Result {
                        request,
                        result: ResultValue::Output(output),
                        ..
                    } = &response
                        && let Some(handle) = output.value.handle()
                    {
                        self.live.bind_result(request, handle);
                    }
                    if let Response::Progress {
                        request, record, ..
                    } = &response
                    {
                        let telemetry = self.progress.entry(request.clone()).or_default();
                        telemetry.apply(request, record);
                        if let Progress::Preview {
                            task, time, arrays, ..
                        } = record
                        {
                            for descriptor in arrays.values() {
                                let coordinates = descriptor.axes.iter().filter_map(|axis| {
                                    match &axis.coordinates {
                                        Coordinates::Coordinates { values } => Some(values),
                                        _ => None,
                                    }
                                });
                                for storage in
                                    std::iter::once(&descriptor.storage).chain(coordinates)
                                {
                                    if let Storage::Npy {
                                        path,
                                        artifact: Some(token),
                                    } = storage
                                    {
                                        assert!(
                                            Path::new(path).is_file(),
                                            "checkpoint published a missing artifact"
                                        );
                                        self.paths.insert(token.clone(), path.into());
                                    }
                                }
                            }
                            self.arrays.insert(request.clone(), arrays.clone());
                            self.live.accept(
                                "job",
                                request,
                                *task,
                                *time,
                                telemetry.task(request, *task).map(|task| {
                                    tomojax_studio::live::CheckpointPosition {
                                        done: task.done,
                                        total: task.total,
                                        unit: task.unit.clone(),
                                    }
                                }),
                                arrays.clone(),
                            );
                        }
                    }
                    if response.terminal() {
                        self.terminals
                            .insert(response.request().unwrap().to_owned(), response);
                    }
                }
                EngineEvent::Diagnostic(message) => eprintln!("worker: {message}"),
                EngineEvent::Disconnected { reason, .. } => panic!("Worker disconnected: {reason}"),
                EngineEvent::Ready(_) => {}
            }
        }
        self.live.poll();
        assert!(self.live.error().is_none(), "{:?}", self.live.error());
        for token in self.live.drain_releases() {
            assert!(
                self.released.insert(token.clone()),
                "artifact released twice: {token}"
            );
            self.send_cleanup(json!({"type":"release","artifact":token}));
        }
        for handle in self.live.drain_handles() {
            assert!(
                self.closed.insert(handle.clone()),
                "result handle closed twice"
            );
            self.send_cleanup(json!({"type":"close","handle":handle}));
        }
    }
    fn output(&mut self, command: Value) -> (String, Box<Output>) {
        let request = self.send(command);
        let until = Instant::now() + Duration::from_secs(120);
        loop {
            self.pump();
            if let Some(response) = self.terminals.remove(&request) {
                match response {
                    Response::Result {
                        result: ResultValue::Output(output),
                        ..
                    } => return (request, output),
                    other => panic!("Expected output, received {other:?}"),
                }
            }
            assert!(Instant::now() < until, "request {request} timed out");
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    fn frame(&mut self, request: &str) {
        let until = Instant::now() + Duration::from_secs(5);
        while self.live.loading()
            || self
                .live
                .frame()
                .is_none_or(|frame| frame.info.request != request)
        {
            self.pump();
            assert!(
                Instant::now() < until,
                "live display did not load {request}"
            );
            std::thread::yield_now();
        }
        let frame = self.live.frame().unwrap();
        assert_eq!(
            frame
                .planes
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["x", "y", "z"])
        );
        for (axis, plane) in &frame.planes {
            assert!(plane.fixed.iter().any(|fixed| &fixed.name == axis));
            assert!(plane.values.iter().any(|value| value.is_finite()));
            assert_eq!(plane.domain, "attenuation");
        }
        for descriptor in self.arrays[request].values() {
            for token in descriptor.artifacts() {
                assert!(!self.released.contains(token));
                assert!(
                    self.paths[token].is_file(),
                    "displayed checkpoint must remain owned"
                );
            }
        }
    }
    fn evidence(&self, request: &str, operation: &str) {
        let Some(directory) = std::env::var_os("TOMOJAX_LIVE_EVIDENCE") else {
            return;
        };
        let directory = PathBuf::from(directory).join(operation);
        std::fs::create_dir_all(&directory).unwrap();
        let mut arrays = self.arrays[request].clone();
        for (key, array) in &mut arrays {
            let copy = |storage: &mut Storage, suffix: &str| {
                let Storage::Npy { path, .. } = storage else {
                    panic!("checkpoint was not NPY")
                };
                let destination = directory.join(format!("{key}{suffix}.npy"));
                std::fs::copy(&*path, &destination).unwrap();
                *storage = Storage::Npy {
                    path: destination.to_string_lossy().into(),
                    artifact: None,
                };
            };
            copy(&mut array.storage, "");
            for (index, axis) in array.axes.iter_mut().enumerate() {
                if let Coordinates::Coordinates { values } = &mut axis.coordinates {
                    copy(values, &format!("-axis{index}"));
                }
            }
        }
        let info = &self.live.frame().unwrap().info;
        std::fs::write(directory.join("checkpoint.json"),serde_json::to_vec_pretty(&json!({
            "operation":operation,"request":request,"job":"job","task":info.task,"time":info.time,"step":info.position.as_ref().map(|p| p.done),
            "arrays":arrays,"telemetry":self.progress[request],"recorded":true,
            "snapshot_note":"Recorded checkpoint from the real TomoJAX worker regression; local array references are copied for replay."
        })).unwrap()).unwrap();
        let mut transcript =
            serde_json::to_string(&json!({"id":request,"command":self.commands[request]})).unwrap();
        transcript.push('\n');
        for response in &self.records[request] {
            transcript.push_str(&serde_json::to_string(response).unwrap());
            transcript.push('\n');
        }
        std::fs::write(directory.join("worker-progress.jsonl"), transcript).unwrap();
    }
}

#[test]
#[ignore = "real worker checkpoints and artifact ownership; run with the locked TomoJAX environment"]
fn real_iterative_and_alignment_checkpoints_are_registered_and_released_after_display() {
    let mut client = Client::new();
    let (_, simulation) = client.output(json!({"type":"run","operation":{
        "type":"simulate","phantom":"shepp","size":12,"n_views":24,"seed":3
    }}));
    let source = serde_json::to_value(simulation.value.scan_source().unwrap()).unwrap();
    let (reconstruct, reconstruction) =
        client.output(json!({"type":"run","previews":true,"operation":{
            "type":"reconstruct","source":source,"method":"cgls","iterations":3,
            "config":{"type":"CGLSConfig","tolerance":0.,"absolute_tolerance":0.}
        }}));
    client.frame(&reconstruct);
    assert!(
        client.progress[&reconstruct]
            .tasks
            .iter()
            .any(|task| task.metrics.contains_key("normal_residual_norm"))
    );
    client.evidence(&reconstruct, "reconstruct");
    // A durable save would close this result next. It owns the live files too:
    // retain native pixels while releasing files first, then close once.
    let reconstructed_plane = client.live.frame().unwrap().planes["z"].clone();
    assert!(
        client
            .live
            .retire_handle(reconstruction.value.handle().unwrap())
    );
    let until = Instant::now() + Duration::from_secs(5);
    loop {
        client.pump();
        if client
            .closed
            .contains(reconstruction.value.handle().unwrap())
            && client.cleanup.is_empty()
        {
            break;
        }
        assert!(
            Instant::now() < until,
            "completed reconstruction handle did not retire"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(
        client.live.frame().unwrap().planes["z"].values,
        reconstructed_plane.values
    );
    let (align, alignment) = client.output(json!({"type":"run","previews":true,"operation":{
        "type":"align","source":source,"method":"joint","fit":"translations",
        "grid":{"nx":12,"ny":12,"nz":12,"vx":1.,"vy":1.,"vz":1.},
        "config":{"type":"AlignConfig","levels":[1],"iterations":2,"initial_image_iterations":2,
            "image_iterations":2,"refine_iterations":1,"tolerance":0.,"min_iterations":1}
    }}));
    client.frame(&align);
    let poses = client
        .live
        .frame()
        .unwrap()
        .poses
        .as_ref()
        .expect("alignment's final checkpoint must include poses");
    assert_eq!(poses.shape, [24, 6]);
    assert_eq!(
        poses
            .columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        ["alpha", "beta", "phi", "x", "z", "y"]
    );
    assert!(
        poses.columns[..3]
            .iter()
            .all(|column| column.unit == "degree")
    );
    assert!(poses.values.iter().all(|value| value.is_finite()));
    let alignment_metrics = client.progress[&align]
        .tasks
        .iter()
        .flat_map(|task| task.metrics.iter())
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        alignment_metrics[&"objective".to_owned()].unit.as_deref(),
        Some("objective units")
    );
    assert_eq!(
        alignment_metrics[&"pose_step_rms".to_owned()]
            .unit
            .as_deref(),
        Some("working pixels")
    );
    assert!(
        client.progress[&align]
            .tree()
            .iter()
            .any(|(depth, _)| *depth > 0)
    );
    client.evidence(&align, "align");
    assert!(client.live.retire_handle(alignment.value.handle().unwrap()));
    let until = Instant::now() + Duration::from_secs(5);
    loop {
        client.pump();
        if client.paths.values().all(|path| !path.exists())
            && client.cleanup.is_empty()
            && client.closed.contains(alignment.value.handle().unwrap())
        {
            break;
        }
        assert!(
            Instant::now() < until,
            "released checkpoint artifacts remained on disk"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(client.released.len(), client.paths.len());
    assert!(
        client.live.frame().unwrap().poses.is_some(),
        "retirement removed owned display data"
    );
    client.live.clear();
    assert!(
        client.live.drain_releases().is_empty(),
        "next job released already-retired files"
    );
    client
        .engine
        .close_handle(simulation.value.handle().unwrap())
        .unwrap();
    for token in client.live.shutdown() {
        client.engine.release(&token).unwrap();
    }
    let ending = client.engine.drain_shutdown(Duration::from_secs(5));
    assert!(ending.iter().any(|event|matches!(event,EngineEvent::Disconnected {code:Some(0),unfinished,..} if unfinished.is_empty())));
}
