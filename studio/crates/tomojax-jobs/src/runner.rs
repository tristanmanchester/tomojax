use crate::{
    directory::read_files,
    executor::{ChildExecutor, Executor, clear_work},
    locks::{Machine, try_lock},
    log::Log,
    publication,
    script::{Claim, Launcher, host, job_id, prepare, read_claim, read_script},
    state::Outcome,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    fs,
    io::{self, ErrorKind},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

const TICK: Duration = Duration::from_millis(100);
pub type Stored<'a> = dyn Fn(&Path) -> io::Result<Value> + 'a;

#[derive(PartialEq, Eq)]
enum Lifetime {
    Shared,
    Owned,
    Draining,
}

struct Attempt<'a> {
    dir: &'a Path,
    stop: &'a AtomicBool,
    executor: &'a mut dyn Executor,
    log: &'a mut Log,
    number: u64,
    serial: u64,
    previews: u64,
    pending: BTreeSet<String>,
    lifetime: Lifetime,
    record_error: Option<io::Error>,
    handles: BTreeSet<String>,
    artifacts: BTreeSet<String>,
}

fn stopped() -> io::Error {
    io::Error::new(ErrorKind::Interrupted, "job stopped")
}

impl Attempt<'_> {
    fn stopping(&self) -> bool {
        self.stop.load(Ordering::Acquire) || self.dir.join("cancel").exists()
    }

    fn internal(&mut self, command: Value) -> io::Result<String> {
        self.serial += 1;
        let id = format!("@{}.{}.{}", job_id(self.dir)?, self.number, self.serial);
        self.executor.send(&json!({"id":id,"command":command}))?;
        self.pending.insert(id.clone());
        Ok(id)
    }

    fn next(&mut self) -> io::Result<Option<Value>> {
        let Some(row) = self.executor.receive(TICK)? else {
            return Ok(None);
        };
        if row.pointer("/record/event") == Some(&json!("preview")) {
            let tokens = publication::tokens(&row["record"]["arrays"]);
            self.artifacts.extend(tokens.iter().cloned());
            self.previews += 1;
            if self.record_error.is_none() {
                self.record_error =
                    publication::preview(self.dir, self.number, self.previews, row).err();
            }
            if self.lifetime == Lifetime::Draining {
                return Ok(None);
            }
            for token in tokens {
                self.internal(json!({"type":"release","artifact":token}))?;
                self.artifacts.remove(&token);
            }
            return Ok(None);
        }
        if matches!(
            row["type"].as_str(),
            Some("result" | "error" | "protocol_error")
        ) && let Some(id) = row["request"].as_str()
        {
            self.pending.remove(id);
        }
        if self.record_error.is_none() {
            self.record_error = self.log.append(&row).err();
        }
        Ok(Some(row))
    }

    fn line(&mut self, line: &Value) -> io::Result<Value> {
        while self.dir.join("paused").try_exists()? {
            if self.stopping() {
                return Err(stopped());
            }
            thread::sleep(TICK);
        }
        if let Some(error) = self.record_error.take() {
            return Err(error);
        }
        if self.stopping() {
            return Err(stopped());
        }
        self.executor.send(line)?;
        self.terminal(&line["id"], true)
    }

    fn terminal(&mut self, id: &Value, control: bool) -> io::Result<Value> {
        let mut cancel_sent = false;
        let mut paused = false;
        loop {
            if control && !cancel_sent {
                let pause = self.dir.join("paused").try_exists()?;
                let action = if self.stopping() || self.record_error.is_some() {
                    cancel_sent = true;
                    Some("cancel")
                } else if pause != paused {
                    paused = pause;
                    Some(if pause { "pause" } else { "resume" })
                } else {
                    None
                };
                if let Some(action) = action {
                    if self.lifetime == Lifetime::Owned && self.stop.load(Ordering::Acquire) {
                        self.lifetime = Lifetime::Draining;
                        self.internal(json!({"type":"shutdown"}))?;
                    } else {
                        self.internal(json!({"type":"control","action":action,"target":id}))?;
                    }
                }
            }
            let Some(row) = self.next()? else {
                continue;
            };
            if &row["request"] != id {
                continue;
            }
            match row["type"].as_str() {
                Some("error" | "protocol_error") => {
                    if let Some(error) = self.record_error.take() {
                        return Err(error);
                    }
                    if row["error"]["kind"] == "canceled" {
                        return Err(stopped());
                    }
                    let message = row["error"]["message"].as_str().or(row["message"].as_str());
                    return Err(io::Error::other(
                        message
                            .map(str::to_owned)
                            .unwrap_or_else(|| row.to_string()),
                    ));
                }
                Some("result") => {
                    let result = &row["result"];
                    if let Some(handle) = result.pointer("/value/handle").and_then(Value::as_str) {
                        self.handles.insert(handle.into());
                    }
                    self.artifacts
                        .extend(publication::tokens(&result["info"]["arrays"]));
                    // Handle-bearing products can own additional artifacts; close retires those.
                    if result.pointer("/value/handle").is_none()
                        && result.pointer("/value/type") != Some(&json!("inspection"))
                    {
                        self.artifacts.extend(publication::tokens(&result["value"]));
                    }
                    if let Some(error) = self.record_error.take() {
                        return Err(error);
                    }
                    return Ok(result.clone());
                }
                _ => {}
            }
        }
    }

    fn execute(&mut self, lines: &[Value], stored: &Stored<'_>) -> io::Result<()> {
        loop {
            if self.stopping() {
                return Err(stopped());
            }
            if let Some(row) = self.next()? {
                if row["type"] != "ready" {
                    return Err(io::Error::other("executor did not send ready"));
                }
                break;
            }
        }
        let mut output = Value::Null;
        for line in lines {
            output = self.line(line)?;
        }
        let command = &lines.last().expect("validated nonempty script")["command"];
        let product = &output["value"];
        if matches!(
            product["type"].as_str(),
            Some("simulation" | "scan" | "alignment" | "reconstruction")
        ) {
            if self.stopping() {
                return Err(stopped());
            }
            let handle = product["handle"]
                .as_str()
                .ok_or_else(|| io::Error::other("dataset missing handle"))?;
            let save=self.internal(json!({"type":"run","operation":{"type":"save","source":{"type":"member","handle":handle,"member":"self"},"path":self.dir.join("work/result/result.nxs"),"include_scan":product["type"]!="reconstruction" || !product["scan"].is_null()}}))?;
            let saved = self.terminal(&json!(save), true)?;
            if saved["value"]["type"] != "files" {
                return Err(io::Error::other("save did not return files"));
            }
        }
        let receipt = publication::receipt(self.dir, command, output, stored)?;
        if self.stopping() {
            return Err(stopped());
        }
        publication::publish(self.dir, &receipt)
    }

    fn release(&mut self) -> io::Result<()> {
        if self.lifetime == Lifetime::Draining {
            while !self.pending.is_empty() {
                self.next()?;
            }
            return Ok(());
        }
        let resources = std::mem::take(&mut self.artifacts)
            .into_iter()
            .map(|artifact| json!({"type":"release","artifact":artifact}))
            .chain(
                std::mem::take(&mut self.handles)
                    .into_iter()
                    .map(|handle| json!({"type":"close","handle":handle})),
            );
        let mut first = None;
        for command in resources {
            let result = self
                .internal(command)
                .and_then(|id| self.terminal(&json!(id), false));
            if let Err(e) = result {
                first.get_or_insert(e);
            }
        }
        while !self.pending.is_empty() {
            self.next()?;
        }
        first.map_or(Ok(()), Err)
    }
}

pub trait InlineSession: Send + Sync {
    /// Wait for exclusive use of the session, interruptibly. The returned stream owns the
    /// lease until drop, and its first record is the ready of the borrowed generation.
    fn acquire(&self, stopping: &dyn Fn() -> bool) -> io::Result<Box<dyn Executor + '_>>;
}

pub fn run_detached(dir: &Path, stored: &Stored<'_>) -> io::Result<i32> {
    let stop = Arc::new(AtomicBool::new(false));
    let signal = signal_hook::flag::register(signal_hook::consts::SIGTERM, Arc::clone(&stop))?;
    let status = run(dir, &stop, None, stored);
    signal_hook::low_level::unregister(signal);
    status
}

pub fn run(
    dir: &Path,
    stop: &AtomicBool,
    inline: Option<&dyn InlineSession>,
    stored: &Stored<'_>,
) -> io::Result<i32> {
    let claim = read_claim(dir)?;
    let Some(_lock) = try_lock(&dir.join("lock"))? else {
        return Err(io::Error::other("another runner owns this job"));
    };
    if dir.join("result/receipt.json").try_exists()? {
        return Ok(0);
    }
    let runner = json!({"host":host(),"pid":std::process::id(),"slurm_job_id":std::env::var("SLURM_JOB_ID").ok(),"started":std::time::SystemTime::now()});
    let mut temp = tempfile::NamedTempFile::new_in(dir)?;
    serde_json::to_writer(&mut temp, &runner)?;
    temp.persist(dir.join("runner.json"))?;
    let stopping = || stop.load(Ordering::Acquire) || dir.join("cancel").exists();
    let machine = (claim.launcher == Launcher::Process)
        .then(|| Machine::local()?.take_job(&stopping))
        .transpose();
    let session = inline.map(|s| s.acquire(&stopping)).transpose();
    let mut log = Log::open(dir)?;
    let number = read_files(dir)?.attempt + 1;
    log.append(&json!({"type":"attempt","number":number}))?;
    let work = (|| {
        if let Err(e) = &machine {
            return Err(io::Error::new(e.kind(), e.to_string()));
        }
        let session = session?;
        if stopping() {
            return Err(stopped());
        }
        clear_work(dir)?;
        fs::create_dir_all(dir.join("work/result"))?;
        fs::create_dir_all(dir.join("work/session"))?;
        run_script(dir, &claim, stop, session, &mut log, number, stored)
    })();
    let cleanup = clear_work(dir);
    let result = work.and(cleanup);
    let (status, outcome) = match &result {
        Ok(()) => (0, Outcome::Completed),
        Err(e)
            if e.kind() == ErrorKind::Interrupted
                || (e.kind() == ErrorKind::BrokenPipe && stop.load(Ordering::Acquire)) =>
        {
            (130, Outcome::Stopped)
        }
        Err(_) => (1, Outcome::Failed),
    };
    log.append(
        &json!({"type":"end","outcome":outcome,"message":result.err().map(|e|e.to_string())}),
    )?;
    drop(machine);
    Ok(status)
}

fn run_script(
    dir: &Path,
    claim: &Claim,
    stop: &AtomicBool,
    session: Option<Box<dyn Executor + '_>>,
    log: &mut Log,
    number: u64,
    stored: &Stored<'_>,
) -> io::Result<()> {
    let mut lines = read_script(dir)?;
    prepare(dir, claim.launcher, &mut lines)?;
    let mut executor: Box<dyn Executor + '_> = match session {
        Some(session) => session,
        None if claim.launcher == Launcher::Inline => {
            return Err(io::Error::other("inline job has no session executor"));
        }
        None => Box::new(ChildExecutor::start(&claim.python, dir)?),
    };
    let mut attempt = Attempt {
        dir,
        stop,
        executor: &mut *executor,
        log,
        number,
        serial: 0,
        previews: 0,
        pending: BTreeSet::new(),
        lifetime: if claim.launcher == Launcher::Inline {
            Lifetime::Shared
        } else {
            Lifetime::Owned
        },
        record_error: None,
        handles: BTreeSet::new(),
        artifacts: BTreeSet::new(),
    };
    let result = attempt.execute(&lines, stored);
    result.and(attempt.release())
}
