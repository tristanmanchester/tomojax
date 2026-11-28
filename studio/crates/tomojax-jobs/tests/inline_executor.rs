//! Drive two scripts through one real Python executor. The transport barrier lets the
//! controller pause while open is outstanding even on an instantaneous filesystem.
use serde_json::{Value, json};
use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::{Arc, Condvar, Mutex, MutexGuard, TryLockError},
    thread,
    time::{Duration, Instant},
};
use tomojax_jobs::{
    executor::{ChildExecutor, Executor},
    runner::InlineSession,
    script::Launcher,
    service::{self, Service},
};

struct Session {
    process: Mutex<ChildExecutor>,
    ready: Value,
    // Submitter and runner rendezvous at send(open), before the terminal is read.
    hold: (Mutex<bool>, Condvar),
    opening: (Mutex<bool>, Condvar),
}

struct Lease<'a> {
    owner: &'a Session,
    executor: MutexGuard<'a, ChildExecutor>,
    ready: Option<Value>,
}

impl Executor for Lease<'_> {
    fn send(&mut self, row: &Value) -> io::Result<()> {
        self.executor.send(row)?;
        if row["command"]["type"] == "open" {
            *self.owner.opening.0.lock().unwrap() = true;
            self.owner.opening.1.notify_all();
            let guard = self.owner.hold.0.lock().unwrap();
            let (hold, timeout) = self
                .owner
                .hold
                .1
                .wait_timeout_while(guard, Duration::from_secs(30), |hold| *hold)
                .unwrap();
            assert!(
                !*hold && !timeout.timed_out(),
                "test did not release the open"
            );
        }
        Ok(())
    }
    fn receive(&mut self, timeout: Duration) -> io::Result<Option<Value>> {
        if let Some(ready) = self.ready.take() {
            return Ok(Some(ready));
        }
        self.executor.receive(timeout)
    }
}

impl InlineSession for Session {
    fn acquire(&self, stopping: &dyn Fn() -> bool) -> io::Result<Box<dyn Executor + '_>> {
        loop {
            match self.process.try_lock() {
                Ok(executor) => {
                    return Ok(Box::new(Lease {
                        owner: self,
                        executor,
                        ready: Some(self.ready.clone()),
                    }));
                }
                Err(TryLockError::WouldBlock) if !stopping() => {
                    thread::sleep(Duration::from_millis(10))
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::Interrupted,
                        "stopped waiting for executor",
                    ));
                }
            }
        }
    }
}

fn resource_counts(session: &Session, id: &str) -> Value {
    let mut ex = session.process.lock().unwrap();
    // The probe itself pins one input scope identically before/after each script.
    let code = "import gc\nfrom tomojax.cli._session import Session\nfrom tomojax.core.input_files import _pins\nnext(([len(s.handles),len(s.artifacts),sum(_pins.values())] for s in gc.get_objects() if isinstance(s,Session)))";
    ex.send(&json!({"id":id,"command":{"type":"python","code":code}}))
        .unwrap();
    loop {
        let row = ex.receive(Duration::from_secs(30)).unwrap().unwrap();
        if row["type"] == "progress" {
            continue;
        }
        assert_eq!(row["type"], "result", "{row}");
        let value = row["result"]["value"]["value"].clone();
        assert!(value.is_array(), "{row}");
        return value;
    }
}

fn wait_for(dir: &Path, state: &str) {
    let until = Instant::now() + Duration::from_secs(60);
    loop {
        let row = service::status(dir).unwrap();
        if row["state"] == state {
            return;
        }
        assert!(Instant::now() < until, "{}: {row}", dir.display());
        thread::sleep(Duration::from_millis(15));
    }
}

#[test]
#[ignore = "requires the installed TomoJAX executor"]
fn shared_executor_pause_success_failure_and_cancel_release_every_resource() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path();
    fs::create_dir_all(dir.join("work/session")).unwrap();
    let python = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../.venv/bin/python");
    let mut process = ChildExecutor::start(&python, dir).unwrap();
    let ready = process.receive(Duration::from_secs(60)).unwrap().unwrap();
    let session = Arc::new(Session {
        process: Mutex::new(process),
        ready,
        hold: (Mutex::new(false), Condvar::new()),
        opening: (Mutex::new(false), Condvar::new()),
    });
    let api = Service::new(
        Launcher::Inline,
        python,
        "unused".into(),
        &[],
        session.clone(),
        Arc::new(|_| panic!("test script publishes a file set or a report")),
    )
    .unwrap();
    let baseline = resource_counts(&session, "baseline");
    let seed = dir.join("seed");
    let source = dir.join("scan.nxs");
    api.submit(&seed,&[
        json!({"id":"seed.sim","command":{"type":"run","operation":{"type":"simulate","size":8,"n_views":8}}}),
        json!({"id":"seed.save","command":{"type":"run","operation":{"type":"save","source":{"type":"member","handle":"seed.sim","member":"self"},"path":source,"include_scan":true}}}),
    ],&[]).unwrap();
    wait_for(&seed, "completed");
    assert_eq!(resource_counts(&session, "after-seed"), baseline);
    let script = |id: &str| {
        vec![
            json!({"id":format!("{id}.open"),"command":{"type":"open","path":source,"kind":"scan"}}),
            json!({"id":format!("{id}.plan"),"command":{"type":"plan","operation":{"type":"reconstruct","source":{"type":"member","handle":format!("{id}.open"),"member":"self"},"method":"sirt","iterations":2}}}),
        ]
    };
    *session.hold.0.lock().unwrap() = true;
    let first = dir.join("first");
    api.submit(&first, &script("first"), &[]).unwrap();
    let (started, _) = session
        .opening
        .1
        .wait_timeout_while(
            session.opening.0.lock().unwrap(),
            Duration::from_secs(30),
            |open| !*open,
        )
        .unwrap();
    assert!(*started);
    drop(started);
    service::control(&first, "pause").unwrap();
    let second = dir.join("second");
    api.submit(&second, &script("second"), &[]).unwrap();
    wait_for(&second, "queued");
    *session.hold.0.lock().unwrap() = false;
    session.hold.1.notify_all();
    service::control(&first, "resume").unwrap();
    wait_for(&first, "completed");
    wait_for(&second, "completed");
    assert_eq!(resource_counts(&session, "after-success"), baseline);
    let mut bad = script("failed");
    bad[1]["command"]["operation"]["method"] = json!("no such solver");
    api.submit(&dir.join("failed"), &bad, &[]).unwrap();
    wait_for(&dir.join("failed"), "failed");
    let failed = service::control(&dir.join("failed"), "cancel").unwrap();
    assert_eq!(failed["state"], "failed");
    assert!(
        failed["detail"]
            .as_str()
            .unwrap()
            .contains("no such solver"),
        "{failed}"
    );
    assert_eq!(resource_counts(&session, "after-failure"), baseline);
    *session.hold.0.lock().unwrap() = true;
    *session.opening.0.lock().unwrap() = false;
    let cancel = dir.join("cancelled");
    api.submit(&cancel, &script("cancelled"), &[]).unwrap();
    let (started, _) = session
        .opening
        .1
        .wait_timeout_while(
            session.opening.0.lock().unwrap(),
            Duration::from_secs(30),
            |open| !*open,
        )
        .unwrap();
    assert!(*started);
    drop(started);
    service::control(&cancel, "cancel").unwrap();
    *session.hold.0.lock().unwrap() = false;
    session.hold.1.notify_all();
    wait_for(&cancel, "canceled");
    assert_eq!(resource_counts(&session, "after-cancel"), baseline);
}
