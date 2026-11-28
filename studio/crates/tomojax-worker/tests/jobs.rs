//! The public worker protocol, the production runner, the Python executor, and a batch
//! controller that can deliver actual signals without requiring a cluster allocation.
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::mpsc::{self, Receiver},
    thread,
    time::{Duration, Instant},
};
use tomojax_jobs::log::page;

fn python() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../.venv/bin/python")
}
fn until(mut condition: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(120);
    while !condition() {
        assert!(
            Instant::now() < deadline,
            "job did not reach the expected boundary"
        );
        thread::sleep(Duration::from_millis(10));
    }
}

struct Worker {
    process: Child,
    replies: Receiver<Value>,
    next: u64,
    root: PathBuf,
}
impl Worker {
    fn start(root: &Path, launcher: &str, interpreter: &Path) -> Self {
        fs::create_dir_all(root.join("slurm")).unwrap();
        let tools = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tomojax-jobs/tests/slurm");
        let paths = std::env::join_paths(
            [tools]
                .into_iter()
                .chain(std::env::split_paths(&std::env::var_os("PATH").unwrap())),
        )
        .unwrap();
        let mut process = Command::new(env!("CARGO_BIN_EXE_tomojax-worker"))
            .args(["serve", "--jobs", launcher, "--python"])
            .arg(interpreter)
            .arg("--results")
            .arg(root)
            .env("PATH", paths)
            .env("FAKE_SLURM", root.join("slurm"))
            .env("JAX_PLATFORMS", "cpu")
            .env("OMP_NUM_THREADS", "2")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let stdout = process.stdout.take().unwrap();
        let (send, replies) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Some(row) = line.ok().and_then(|l| serde_json::from_str(&l).ok()) else {
                    break;
                };
                if send.send(row).is_err() {
                    break;
                }
            }
        });
        let worker = Self {
            process,
            replies,
            next: 0,
            root: root.into(),
        };
        assert_eq!(worker.read()["type"], "ready");
        worker
    }
    fn read(&self) -> Value {
        self.replies
            .recv_timeout(Duration::from_secs(120))
            .expect("worker stopped answering")
    }
    fn request(&mut self, cmd: Value) -> Value {
        self.next += 1;
        let id = format!("client-{}", self.next);
        writeln!(
            self.process.stdin.as_mut().unwrap(),
            "{}",
            json!({"id":id,"command":cmd})
        )
        .unwrap();
        loop {
            let reply = self.read();
            if reply["request"] == id && reply["type"] != "progress" {
                assert_eq!(reply["type"], "result", "{reply}");
                return reply["result"].clone();
            }
        }
    }
    fn submit(&mut self, name: &str, commands: Vec<Value>) -> PathBuf {
        let script: Vec<_> = commands
            .into_iter()
            .enumerate()
            .map(|(i, c)| json!({"id":format!("{name}.{i}"),"command":c}))
            .collect();
        let dir = self.root.join("jobs").join(name);
        self.request(json!({"type":"submit","directory":dir,"script":script}));
        dir
    }
    fn job(&mut self, dir: &Path) -> Value {
        self.request(json!({"type":"job","directory":dir,"offset":0,"limit":1}))
    }
    fn finished(&mut self, dir: &Path, wanted: &str) -> Value {
        let mut job = Value::Null;
        until(|| {
            job = self.job(dir);
            assert!(
                job["state"] == wanted
                    || !matches!(
                        job["state"].as_str(),
                        Some("failed" | "canceled" | "interrupted")
                    ),
                "unexpected job state {job}\n{}",
                fs::read_to_string(dir.join("replies.jsonl")).unwrap_or_default()
            );
            job["state"] == wanted && (wanted != "completed" || !dir.join("work").exists())
        });
        job
    }
    fn kill(&mut self) {
        self.process.kill().unwrap();
        self.process.wait().unwrap();
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.process.stdin.take();
        let _ = self.process.wait();
    }
}

fn signal(dir: &Path) {
    let row: Value = serde_json::from_slice(&fs::read(dir.join("runner.json")).unwrap()).unwrap();
    let pid = rustix::process::Pid::from_raw(row["pid"].as_i64().unwrap() as i32).unwrap();
    rustix::process::kill_process(pid, rustix::process::Signal::TERM).unwrap();
}

fn advanced(dir: &Path) {
    let mut offset = 0;
    let mut solve = None;
    until(|| {
        let (rows, next) = page(dir, offset, 1 << 20).unwrap();
        offset = next;
        for row in rows {
            let r = &row["record"];
            if r["event"] == "started" && r["unit"] == "iterations" {
                solve = r["task"].as_u64();
            }
            if r["event"] == "advanced" && solve.is_some() && solve == r["task"].as_u64() {
                return true;
            }
            assert!(
                row["type"] != "end",
                "attempt ended before solve advanced: {row}"
            );
        }
        false
    });
}

fn simulate() -> Value {
    json!({"type":"run","operation":{"type":"simulate","phantom":"shepp","size":16,"n_views":24,"seed":1}})
}

fn fake_slurm_requeues(worker: &mut Worker, op: &Value) {
    let dir = worker.submit("whole", vec![op.clone()]);
    let whole = worker.finished(&dir, "completed");
    let dir = worker.submit("limited", vec![op.clone()]);
    advanced(&dir);
    signal(&dir);
    let resumed = worker.finished(&dir, "completed");
    let compare=Command::new(python()).args(["-c","import sys, numpy as np, tomojax as tj; a,b=(tj.load_reconstruction(p).volume for p in sys.argv[1:]); np.testing.assert_array_equal(a,b)"])
        .arg(whole["receipt"]["path"].as_str().unwrap()).arg(resumed["receipt"]["path"].as_str().unwrap()).env("JAX_PLATFORMS","cpu").status().unwrap();
    assert!(compare.success());
    assert_eq!(
        fs::read_to_string(dir.join("replies.jsonl"))
            .unwrap()
            .matches("\"outcome\":\"stopped\"")
            .count(),
        1
    );
    fs::write(worker.root.join("slurm/fail-requeue"), "").unwrap();
    let dir = worker.submit("requeue-failed", vec![op.clone()]);
    advanced(&dir);
    signal(&dir);
    let failed = worker.finished(&dir, "interrupted");
    assert_eq!(failed["checkpoint"]["usable"], true);
}

#[test]
#[ignore = "runs real CPU reconstructions and resumes a checkpointed solve"]
fn detached_scripts_survive_the_submitter_and_checkpoint_on_signals() {
    let root = tempfile::tempdir().unwrap();
    let mut scan = Value::Null;
    for mode in ["process", "slurm"] {
        let mut submitter = Worker::start(root.path(), mode, &python());
        let dir = submitter.submit(mode, vec![simulate()]);
        submitter.kill();
        let mut observer = Worker::start(root.path(), "inline", &python());
        let done = observer.finished(&dir, "completed");
        scan = done["receipt"]["path"].clone();
        assert!(
            done["receipt"]["arrays"]
                .as_object()
                .is_some_and(|a| !a.is_empty())
        );
    }
    let mut worker = Worker::start(root.path(), "slurm", &python());
    let mut op = json!({"type":"run","operation":{"type":"reconstruct","source":{"type":"path","path":scan},"method":"sirt","iterations":300}});
    let mut inline = Worker::start(root.path(), "inline", &python());
    let running = inline.submit("inline-killed", vec![op.clone()]);
    advanced(&running);
    inline.kill();
    worker.finished(&running, "interrupted");
    fake_slurm_requeues(&mut worker, &op);
    for i in 0..3 {
        let dir = worker.submit(&format!("cancel-{i}"), vec![op.clone()]);
        advanced(&dir);
        worker.request(json!({"type":"job_control","directory":dir,"action":"cancel"}));
        let canceled = worker.finished(&dir, "canceled");
        assert_eq!(canceled["checkpoint"]["usable"], true);
        let path = &canceled["checkpoint"]["path"];
        if i > 0 {
            assert_eq!(path, &op["operation"]["checkpoint"]);
        }
        op["operation"]["checkpoint"] = path.clone();
        assert_eq!(
            fs::read_to_string(dir.join("replies.jsonl"))
                .unwrap()
                .matches("\"type\":\"attempt\"")
                .count(),
            1
        );
    }
}

#[test]
#[ignore = "starts the installed worker and sends real signals"]
fn termination_while_waiting_for_the_gpu_or_executor_closes_a_stopped_attempt() {
    let root = tempfile::tempdir().unwrap();
    let base = root.path();
    let wrapper = base.join("python");
    fs::write(&wrapper,format!("#!/bin/sh\ncase \"$TMPDIR\" in */blocked/work/session) touch \"$TMPDIR/waiting\"; exec sleep 120;; esac\nexec '{}' \"$@\"\n",python().display())).unwrap();
    fs::set_permissions(&wrapper, fs::Permissions::from_mode(0o700)).unwrap();
    let mut worker = Worker::start(base, "slurm", &wrapper);
    let op = json!({"type":"run","operation":{"type":"correct","source":{"type":"path","path":"unused"}}});
    let blocked = worker.submit("blocked", vec![op.clone()]);
    until(|| blocked.join("work/session/waiting").exists());
    signal(&blocked);
    let done = worker.finished(&blocked, "interrupted");
    assert!(done["checkpoint"].is_null());
    drop(worker);
    let machine = tomojax_jobs::locks::Machine::local().unwrap();
    let _busy = machine.take_job(&|| false).unwrap();
    let mut worker = Worker::start(base, "process", &python());
    let queued = worker.submit("locked", vec![op]);
    until(|| queued.join("runner.json").exists());
    assert_eq!(worker.job(&queued)["state"], "queued");
    signal(&queued);
    worker.finished(&queued, "interrupted");
    let log = fs::read_to_string(queued.join("replies.jsonl")).unwrap();
    assert!(log.contains("\"outcome\":\"stopped\""));
}
