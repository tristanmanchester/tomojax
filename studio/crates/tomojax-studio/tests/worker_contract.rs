//! Real process contract, independent of the native window and rendering loop.
//! Run explicitly with the locked environment's TOMOJAX_TEST_PYTHON.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use studio_core::engine::{Engine, EngineEvent};
use studio_core::project;
use studio_core::protocol::{ErrorKind, Output, Product, Progress, Response, ResultValue, Storage};
use tomojax_io::metadata;

struct Client {
    engine: Engine,
    terminals: HashMap<String, Response>,
    paused: HashSet<String>,
    previews: Vec<PathBuf>,
}

impl Client {
    fn start() -> Self {
        let python = std::env::var_os("TOMOJAX_TEST_PYTHON")
            .expect("set TOMOJAX_TEST_PYTHON to the locked Python executable");
        let mut client = Self {
            engine: Engine::spawn_through(Path::new(&python), &[], Arc::new(|| {})).unwrap(),
            terminals: HashMap::new(),
            paused: HashSet::new(),
            previews: Vec::new(),
        };
        let until = Instant::now() + Duration::from_secs(45);
        while client.engine.catalog().is_none() {
            client.pump();
            assert!(
                Instant::now() < until,
                "worker did not describe its parameters"
            );
            thread::sleep(Duration::from_millis(5));
        }
        client
    }

    fn pump(&mut self) {
        for event in self.engine.poll() {
            match event {
                EngineEvent::Reply { response, .. } => {
                    if let Response::Progress {
                        request, record, ..
                    } = &response
                    {
                        match record {
                            Progress::Paused { .. } => {
                                self.paused.insert(request.clone());
                            }
                            Progress::Resumed { .. } => {
                                self.paused.remove(request);
                            }
                            Progress::Preview { arrays, .. } => {
                                for array in arrays.values() {
                                    assert_eq!(array.shape.len(), array.axes.len());
                                    if let Storage::Npy { path, artifact } = &array.storage {
                                        assert!(artifact.is_some());
                                        assert!(Path::new(path).is_file());
                                        self.previews.push(path.into());
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    if response.terminal() {
                        self.terminals
                            .insert(response.request().unwrap().to_owned(), response);
                    }
                }
                EngineEvent::Diagnostic(message) => eprintln!("worker: {message}"),
                EngineEvent::Disconnected { reason, .. } => {
                    panic!("Unexpected worker exit: {reason}")
                }
                EngineEvent::Ready(_) => {}
            }
        }
    }

    fn send(&mut self, command: Value) -> String {
        self.engine.request(command, None).unwrap()
    }

    fn terminal(&mut self, id: &str) -> Response {
        let until = Instant::now() + Duration::from_secs(60);
        loop {
            self.pump();
            if let Some(response) = self.terminals.remove(id) {
                return response;
            }
            assert!(Instant::now() < until, "request {id} did not terminate");
            thread::sleep(Duration::from_millis(5));
        }
    }

    fn output(&mut self, command: Value) -> Output {
        let id = self.send(command);
        match self.terminal(&id) {
            Response::Result {
                result: ResultValue::Output(output),
                ..
            } => *output,
            other => panic!("Expected a published output, got {other:?}"),
        }
    }

    fn save(&mut self, output: &Output, path: &Path) {
        let result=self.output(json!({"type":"run","operation":{"type":"save","source":{
            "type":"member","handle":output.value.handle().unwrap(),"member":"self"},"path":path,"include_scan":true}}));
        let Product::Files { paths } = result.value else {
            panic!("Save did not publish files")
        };
        assert!(paths.iter().any(|name| name.ends_with(".manifest.json")));
        assert!(paths.iter().all(|name| Path::new(name).exists()));
    }
}

#[test]
#[ignore = "real numerical worker; set TOMOJAX_TEST_PYTHON and run --ignored"]
fn project_roundtrip_captured_recipe_and_registered_volume() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("study");
    let mut project = project::Project::open(&root, true).unwrap();
    let mut client = Client::start();
    let simulation=client.output(json!({"type":"run","operation":{"type":"simulate","phantom":"sphere","size":16,"n_views":32}}));
    let source_path = project.root.join("derived/source.nxs");
    client.save(&simulation, &source_path);
    let mut source = project::Asset::input(project.stored_path(&source_path), true);
    source.kind = "simulation".into();
    source.provenance = Some(simulation.manifest.clone());
    let source_id = source.id.clone();
    project.document.assets.push(source);

    let mut draft = json!({"method":"fourier"});
    let mut job = project::Job::new(
        "reconstruct",
        draft.clone(),
        vec![project.capture(&source_id).unwrap()],
    );
    draft["method"] = json!("sirt");
    let job_id = job.id.clone();
    project.document.jobs.push(job.clone());
    project.changed().unwrap();
    project.flush().unwrap();
    assert_eq!(
        project.document.job_mut(&job_id).unwrap().parameters["method"],
        "fourier"
    );

    let reopened_source =
        client.output(json!({"type":"open","kind":job.inputs[0].kind,"path":job.inputs[0].path}));
    let source_ref = serde_json::to_value(reopened_source.value.scan_source().unwrap()).unwrap();
    let invalid=client.send(json!({"type":"run","operation":{"type":"reconstruct","source":source_ref,"method":"invented-method"}}));
    assert!(matches!(
        client.terminal(&invalid),
        Response::ProtocolError { .. }
    ));
    let operation =
        json!({"type":"reconstruct","source":source_ref,"method":job.parameters["method"]});
    let plan = client.output(json!({"type":"plan","operation":operation}));
    assert!(matches!(plan.value,Product::ReconstructionPlan {ref method,..} if method=="fourier"));
    let reconstruction = client.output(json!({"type":"run","operation":operation,"previews":true}));
    let result_path = project.root.join("derived/reconstruction.nxs");
    client.save(&reconstruction, &result_path);

    let original = hdf5::File::open(&source_path)
        .unwrap()
        .dataset("/entry/phantom/data")
        .unwrap()
        .read_raw::<f32>()
        .unwrap();
    let recovered = hdf5::File::open(&result_path)
        .unwrap()
        .dataset("/entry/volume/data")
        .unwrap()
        .read_raw::<f32>()
        .unwrap();
    let squared_error: f64 = original
        .iter()
        .zip(&recovered)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum();
    let energy: f64 = original.iter().map(|value| f64::from(*value).powi(2)).sum();
    assert!(
        (squared_error / energy).sqrt() < 0.4,
        "inverse does not recover independent phantom truth"
    );
    let report = client.output(json!({"type":"inspect","path":result_path}));
    let Product::Inspection { report } = report.value else {
        panic!("Expected inspection")
    };
    let registered = metadata::inspection_arrays(&report, &result_path).unwrap();
    let (kind, quick, header) = metadata::quick_metadata(&result_path).unwrap().unwrap();
    assert_eq!(kind, "reconstruction");
    assert_eq!(quick["volume"], registered["volume"]);
    assert_eq!(header["source"], "header_only");
    assert_eq!(registered["volume"].shape, [16, 16, 16]);
    assert_eq!(
        registered["volume"]
            .axes
            .iter()
            .map(|axis| axis.name.as_str())
            .collect::<Vec<_>>(),
        ["z", "y", "x"]
    );
    let mut result = project::Asset::input(project.stored_path(&result_path), true);
    result.kind = "reconstruction".into();
    result.parent = Some(source_id);
    result.method = Some("fourier".into());
    result.register_read(&kind, quick, &project.root);
    result.provenance = Some(reconstruction.manifest);
    job.status = project::JobStatus::Completed;
    job.result = Some(result.id.clone());
    *project.document.job_mut(&job_id).unwrap() = job;
    project.document.assets.push(result);
    project.changed().unwrap();
    project.flush().unwrap();
    drop(project);

    let reopened = project::Project::open(&root, false).unwrap();
    assert_eq!(reopened.document.assets.len(), 2);
    assert_eq!(reopened.document.jobs[0].parameters["method"], "fourier");
    assert_eq!(
        reopened.document.jobs[0].status,
        project::JobStatus::Completed
    );
    assert!(reopened.document.assets[1].volume().is_some());
    let opened = client.output(json!({"type":"open","kind":"reconstruction","path":result_path}));
    assert!(
        matches!(opened.value,Product::Reconstruction {ref method,scan:Some(_),..} if method=="fourier")
    );
    // Iterative methods publish numerical previews. The direct Fourier path
    // publishes its final reconstruction without promising a progress preview.
    let previewed = client.output(json!({"type":"run","operation":{"type":"reconstruct",
        "source":source_ref,"method":"cgls","iterations":2},"previews":true}));
    assert_eq!(previewed.value.primary_array().unwrap().shape, [16, 16, 16]);
    assert!(
        !client.previews.is_empty(),
        "the real solve must publish registered previews"
    );
    let ending = client.engine.drain_shutdown(Duration::from_secs(5));
    assert!(ending.iter().any(|event|matches!(event,EngineEvent::Disconnected {code:Some(0),unfinished,..} if unfinished.is_empty())));
    assert!(client.previews.iter().all(|path| !path.exists()));
}

#[test]
#[ignore = "real cooperative worker; set TOMOJAX_TEST_PYTHON and run --ignored"]
fn paused_real_work_holds_queue_and_cancellation_has_a_terminal() {
    let mut client = Client::start();
    let simulation=client.output(json!({"type":"run","operation":{"type":"simulate","phantom":"sphere","size":16,"n_views":32}}));
    let source = serde_json::to_value(simulation.value.scan_source().unwrap()).unwrap();
    let holding=client.send(json!({"type":"run","operation":{"type":"reconstruct","source":source,
        "method":"cgls","iterations":10000,"config":{"type":"CGLSConfig","tolerance":0.,"absolute_tolerance":0.}}}));
    let pause = client.engine.control(&holding, "pause").unwrap();
    assert!(matches!(
        client.terminal(&pause),
        Response::Result {
            result: ResultValue::Ack,
            ..
        }
    ));
    let until = Instant::now() + Duration::from_secs(60);
    while !client.paused.contains(&holding) {
        // The initial work boundary precedes a progress task, so an early pause
        // can be authoritative in status without emitting a Paused record.
        let status = client.send(json!({"type":"status"}));
        let Response::Result {
            result: ResultValue::Status { requests, .. },
            ..
        } = client.terminal(&status)
        else {
            panic!("Expected cached status while pausing")
        };
        if requests
            .get(&holding)
            .is_some_and(|state| state == "paused")
        {
            break;
        }
        assert!(
            !client.terminals.contains_key(&holding),
            "the held request finished before reaching its paused work boundary"
        );
        assert!(
            Instant::now() < until,
            "worker did not reach a work boundary"
        );
        thread::sleep(Duration::from_millis(5));
    }
    let queued = client.send(
        json!({"type":"run","operation":{"type":"reconstruct","source":source,"method":"fourier"}}),
    );
    let status = client.send(json!({"type":"status"}));
    let Response::Result {
        result: ResultValue::Status { requests, .. },
        ..
    } = client.terminal(&status)
    else {
        panic!("Expected cached status")
    };
    assert_eq!(requests[&holding], "paused");
    assert_eq!(requests[&queued], "queued");
    for target in [&queued, &holding] {
        let cancellation = client.engine.control(target, "cancel").unwrap();
        assert!(matches!(
            client.terminal(&cancellation),
            Response::Result {
                result: ResultValue::Ack,
                ..
            }
        ));
        assert!(matches!(
            client.terminal(target),
            Response::Error {
                error: studio_core::protocol::WorkerError {
                    kind: ErrorKind::Canceled,
                    ..
                },
                ..
            }
        ));
    }
    let result = client.output(
        json!({"type":"run","operation":{"type":"reconstruct","source":source,"method":"fourier"}}),
    );
    assert_eq!(result.value.primary_array().unwrap().shape, [16, 16, 16]);
    let ending = client.engine.drain_shutdown(Duration::from_secs(5));
    assert!(ending.iter().any(|event|matches!(event,EngineEvent::Disconnected {code:Some(0),unfinished,..} if unfinished.is_empty())));
}
