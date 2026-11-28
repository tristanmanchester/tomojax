//! One owner orders admissions, executor generations, file leases and outgoing frames.
use crate::{
    Event, Options,
    executor::Executor,
    files, transfer,
    wire::{Input, Outbox, Reply},
};
mod work;
pub use work::Output;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::SyncSender,
};
use std::time::{Duration, Instant};

pub const CAPABILITIES: &[&str] = &[
    "fetch",
    "list",
    "project_results",
    "stored",
    "peek",
    "copy",
    "upload",
    "download",
    "status",
    "control",
    "shutdown",
    "submit",
    "jobs",
    "job",
    "job_control",
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    id: String,
    channel: Option<String>,
    command: Value,
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Side {
    Native,
    Executor,
}

struct Active {
    reply: Reply,
    fetch: bool,
    side: Side,
    sequence: u64,
    cancel: Arc<AtomicBool>,
}

struct Token {
    path: PathBuf,
    owner: String,
    retired: bool,
}

pub struct Session {
    opts: Options,
    pub out: Outbox,
    pub closing: Option<Instant>,
    shutdown: Option<Reply>,
    pub executor: Option<Executor>,
    generation: u64,
    root: Option<PathBuf>,
    detail: Value,
    retry: Instant,
    delay: Duration,
    up: Option<Instant>,
    active: HashMap<String, Active>,
    permanent: HashSet<String>,
    channels: HashMap<String, (Side, u64)>,
    sequence: u64,
    artifacts: HashMap<String, Token>,
    leases: HashMap<u64, Vec<String>>,
    deferred: VecDeque<(Vec<u8>, Vec<String>)>,
    uploads: HashMap<String, transfer::Upload>,
    events: SyncSender<Event>,
    pool: crate::Pool,
    planes: Arc<crate::pixels::PlaneCache>,
    scripts: crate::inline::Queue,
    jobs: Arc<tomojax_jobs::service::Service>,
    job_pool: crate::Pool,
    placement: Option<crate::jobs::Placement>,
    cpu_only: bool,
}

impl Session {
    pub fn new(opts: Options, events: SyncSender<Event>) -> Result<Self> {
        let mut out = Outbox::default();
        let ready = json!({"type":"ready","protocol":1,"progress":2,"host":rustix::system::uname().nodename().to_string_lossy(),"home":std::env::var("HOME")?,"directory":std::env::current_dir()?,"jobs":opts.jobs,"pixel_credit":crate::wire::CREDIT,"capabilities":CAPABILITIES});
        out.send(ready);
        let jobs = crate::jobs::service(&opts, crate::inline::Scripts::new(events.clone()))?;
        let placement = crate::jobs::Placement::new(opts.jobs)?;
        let cpu_only = placement.as_ref().is_some_and(|p| p.cpu_only());
        Ok(Self {
            opts,
            out,
            closing: None,
            shutdown: None,
            executor: None,
            generation: 0,
            root: None,
            detail: Value::Null,
            retry: Instant::now(),
            delay: Duration::from_secs(1),
            up: None,
            active: HashMap::new(),
            permanent: HashSet::new(),
            channels: HashMap::new(),
            sequence: 0,
            artifacts: HashMap::new(),
            leases: HashMap::new(),
            deferred: VecDeque::new(),
            uploads: HashMap::new(),
            jobs,
            job_pool: crate::Pool::new(),
            placement,
            scripts: Default::default(),
            cpu_only,
            events,
            pool: crate::Pool::new(),
            planes: Default::default(),
        })
    }

    fn state(&mut self, state: &str, detail: Value) {
        self.detail =
            json!({"type":"executor","state":state,"detail":detail,"generation":self.generation});
        self.out.send(self.detail.clone());
    }

    pub fn tick(&mut self) -> Result<()> {
        if self.closing.is_none()
            && let Some(placement) = &mut self.placement
        {
            let idle =
                self.scripts.idle() && !self.active.values().any(|a| a.side == Side::Executor);
            match placement.check(idle)? {
                Some(crate::jobs::Move::ToCpu(lease)) => {
                    self.restart(true, "the GPU was given to a job");
                    drop(lease);
                }
                Some(crate::jobs::Move::ToGpu) => self.restart(false, "the GPU is available"),
                None => {}
            }
        }
        if self.closing.is_none() && self.executor.is_none() && Instant::now() >= self.retry {
            self.generation += 1;
            self.state("starting", Value::Null);
            match Executor::start(&self.opts, self.generation, &self.events, self.cpu_only) {
                Ok(child) => self.executor = Some(child),
                Err(e) => self.gone(self.generation, format!("{e:#}")),
            }
        }
        Ok(())
    }

    pub fn gone(&mut self, generation: u64, reason: String) {
        if generation != self.generation || self.detail["state"] == "gone" {
            return;
        }
        self.executor.take();
        self.root = None;
        if self
            .up
            .take()
            .is_some_and(|at| at.elapsed() >= Duration::from_secs(30))
        {
            self.delay = Duration::from_secs(1);
        }
        self.retry = Instant::now() + self.delay;
        self.delay = (self.delay * 2).min(Duration::from_secs(300));
        if self.closing.is_none() {
            self.state("gone", json!(reason));
        }
        let failed = self
            .active
            .iter()
            .filter(|(_, a)| {
                a.side == Side::Executor
                    || self.leases.get(&a.sequence).is_some_and(|l| !l.is_empty())
            })
            .map(|(k, _)| k.clone())
            .collect::<Vec<_>>();
        for id in failed {
            self.cancel(&id, "input", &reason);
        }
        self.scripts = Default::default();
        self.artifacts.clear();
        self.deferred.clear();
    }

    /// Restart after changing GPU ownership. Existing generation leases end before the new start.
    pub fn restart(&mut self, cpu_only: bool, reason: &str) {
        self.gone(self.generation, reason.into());
        self.cpu_only = cpu_only;
        self.retry = Instant::now();
    }

    pub fn script(&mut self, action: crate::inline::Action) {
        use crate::inline::Action;
        if self.closing.is_some() {
            return;
        }
        match action {
            Action::Acquire(ticket, tx) => self.scripts.wait(ticket, tx),
            Action::Release(ticket) => self.scripts.release(ticket),
            Action::Send(ticket, row) => {
                let Some(id) = row["id"].as_str() else {
                    return;
                };
                if let Some(tx) = self.scripts.sender(ticket) {
                    let result =
                        if !self.active.contains_key(id) && self.permanent.insert(id.into()) {
                            self.executor
                                .as_ref()
                                .context("executor gone")
                                .and_then(|e| e.send(crate::wire::line(&row)))
                        } else {
                            Err(anyhow::anyhow!("duplicate script ID"))
                        };
                    if let Err(e) = result {
                        let error = json!({"type":"protocol_error","request":id,"kind":"invalid_request","message":e.to_string()});
                        let _ = tx.send(error);
                    } else {
                        self.scripts.track(id, ticket, tx);
                    }
                }
            }
        }
    }
    pub fn expired(&mut self) {
        self.gone(self.generation, "executor did not shut down".into());
    }

    pub fn diagnostic(&mut self, generation: u64, message: String) {
        if self.generation == generation && self.executor.is_some() {
            if self.root.is_none() {
                self.state("starting", json!(message));
            } else {
                eprintln!("{message}");
            }
        }
    }

    pub fn answer(&mut self, generation: u64, bytes: Vec<u8>) -> Result<()> {
        if self.generation != generation || self.executor.is_none() {
            return Ok(());
        }
        let mut value: Value = serde_json::from_slice(&bytes)?;
        if value["type"] == "ready" {
            ensure!(
                self.root.is_none() && value["protocol"] == 1,
                "unexpected executor ready"
            );
            let path = value["root"]
                .as_str()
                .context("executor did not report its session root")?;
            self.root = Some(
                self.executor
                    .as_ref()
                    .context("executor gone")?
                    .validate_root(path)?,
            );
            self.up = Some(Instant::now());
            self.state("ready", value);
            return Ok(());
        }
        let id = value["request"]
            .as_str()
            .context("executor reply lacks request")?;
        let terminal = matches!(
            value["type"].as_str(),
            Some("result" | "error" | "protocol_error")
        );
        if self.scripts.reply(id, &value, terminal) {
            return Ok(());
        }
        if matches!(value["type"].as_str(), Some("result" | "progress")) {
            self.record_tokens(id, &value);
        }
        let mut changed = false;
        if value["type"] == "protocol_error" {
            self.permanent.remove(id);
            if let Some(active) = self.active.remove(id) {
                self.reject_revision(&active);
            }
        } else if let Some(active) = self.active.get(id) {
            // The executor starts counting at one again after a generation change.
            let channel = active.reply.channel.clone();
            let status = value["result"]["type"] == "status";
            let request = id.to_owned();
            if terminal {
                self.active.remove(&request);
            }
            if status {
                self.add_status(&mut value["result"], &request);
                changed = true;
            }
            if value["channel"] != channel {
                value["channel"] = channel;
                changed = true;
            }
        }
        if changed {
            self.out.send(value);
        } else {
            self.out.queue_record(&value, bytes);
        }
        Ok(())
    }

    fn reject_revision(&mut self, refused: &Active) {
        let Some(name) = refused.reply.channel["name"].as_str() else {
            return;
        };
        if let Some((_, revision)) = self.channels.get_mut(name) {
            *revision -= 1;
            if *revision == 0 {
                self.channels.remove(name);
            }
        }
        for later in self
            .active
            .values_mut()
            .filter(|a| a.sequence > refused.sequence && a.reply.channel["name"] == name)
        {
            if let Some(n) = later.reply.channel["revision"].as_u64() {
                later.reply.channel["revision"] = json!(n - 1);
            }
        }
    }

    fn record_tokens(&mut self, id: &str, value: &Value) {
        match value {
            Value::Object(_) if value["type"] == "npy" => {
                if let (Some(token), Some(path), Some(root)) = (
                    value["artifact"].as_str(),
                    value["path"].as_str(),
                    self.root.as_ref(),
                ) {
                    let path = PathBuf::from(path);
                    if path.parent() == Some(root.as_path()) {
                        self.artifacts
                            .entry(token.to_string())
                            .or_insert_with(|| Token {
                                path,
                                owner: id.into(),
                                retired: false,
                            });
                    }
                }
            }
            Value::Array(items) => {
                for v in items {
                    self.record_tokens(id, v);
                }
            }
            Value::Object(items) => {
                for v in items.values() {
                    self.record_tokens(id, v);
                }
            }
            _ => {}
        }
    }

    fn lease(&self, path: &str, token: Option<&str>) -> Result<Option<String>> {
        let path = files::resolve(path)?;
        if let Some(token) = token {
            let artifact = self.artifacts.get(token).context("unknown artifact")?;
            ensure!(
                !artifact.retired && path == artifact.path,
                "artifact path differs from issued path or has been released"
            );
            Ok(Some(token.into()))
        } else {
            if let Some(root) = &self.root {
                let resolved = path.canonicalize().unwrap_or(path);
                ensure!(
                    !resolved.starts_with(root) && !root.starts_with(resolved),
                    "session files require their issued token"
                );
            }
            Ok(None)
        }
    }

    pub fn reject(&mut self, id: Option<&str>, reason: &str) {
        self.out.send(
            json!({"type":"protocol_error","request":id,"kind":"invalid_request","message":reason}),
        );
    }

    pub fn input(&mut self, input: Input) {
        match input {
            Input::Rejected(e) => self.reject(None, &e),
            Input::Eof => self.stop(),
            Input::Block(id, block) => {
                if !self.uploads.contains_key(&id) {
                    self.reject(Some(&id), "unknown upload");
                } else if let Err(e) = self.block(&id, &block) {
                    self.cancel(&id, "input", &e.to_string());
                }
            }
            Input::Request(bytes, value) => {
                let id = value["id"].as_str().map(str::to_owned);
                let result = if value.as_object().is_some_and(|o| o.len() == 1)
                    && value.get("credit").is_some()
                {
                    value["credit"]
                        .as_u64()
                        .context("invalid credit")
                        .and_then(|n| self.out.credit(n))
                } else if value.as_object().is_some_and(|o| o.len() == 1)
                    && value.get("window").is_some()
                {
                    value["window"]
                        .as_u64()
                        .context("invalid window")
                        .and_then(|n| self.out.window(n))
                } else {
                    serde_json::from_value(value)
                        .map_err(Into::into)
                        .and_then(|r| self.admit(r, bytes))
                };
                if let Err(e) = result {
                    self.reject(id.as_deref(), &e.to_string());
                }
            }
        }
    }

    fn side(&self, kind: &str, cmd: &Value) -> Side {
        let target = cmd["target"].as_str().unwrap_or("");
        match kind {
            "control" => self.active.get(target).map_or(Side::Executor, |a| a.side),
            "status" if self.root.is_some() => Side::Executor,
            t if CAPABILITIES.contains(&t) => Side::Native,
            _ => Side::Executor,
        }
    }

    fn admit(&mut self, r: Request, bytes: Vec<u8>) -> Result<()> {
        let id = &r.id;
        ensure!(
            !id.is_empty()
                && id.len() <= 256
                && r.channel
                    .as_ref()
                    .is_none_or(|c| !c.is_empty() && c.len() <= 256),
            "invalid request or channel ID"
        );
        ensure!(
            !self.permanent.contains(id) && !self.active.contains_key(id),
            "duplicate id"
        );
        let cmd = &r.command;
        let kind = cmd["type"].as_str().context("missing command type")?;
        let side = self.side(kind, cmd);
        let channel = if let Some(name) = &r.channel {
            let (owner, rev) = self.channels.get(name).copied().unwrap_or((side, 0));
            ensure!(owner == side, "channel belongs to the other side");
            json!({"name":name,"revision":rev+1})
        } else {
            Value::Null
        };
        let mut leases = Vec::new();
        let work = if side == Side::Native {
            self.prepare(kind, cmd, &mut leases)?
        } else {
            None
        };
        if kind == "fetch" && channel.is_null() {
            ensure!(
                self.active
                    .values()
                    .filter(|a| a.fetch && a.reply.channel.is_null())
                    .count()
                    < 8,
                "eight unchanneled fetches pending"
            );
        }
        ensure!(self.active.len() < 1024, "too many pending requests");
        ensure!(
            matches!(kind, "shutdown" | "fetch" | "job" | "jobs") || self.permanent.len() < 100000,
            "session id limit reached"
        );
        if let Some(name) = &r.channel {
            self.channels.insert(
                name.clone(),
                (side, channel["revision"].as_u64().unwrap_or(0)),
            );
            if side == Side::Native
                && let Some(old) = self
                    .active
                    .values()
                    .find(|a| a.reply.channel["name"] == *name)
                    .map(|a| a.reply.id.clone())
            {
                self.cancel(&old, "superseded", "superseded");
            }
        }
        self.sequence += 1;
        let sequence = self.sequence;
        let cancel = Arc::new(AtomicBool::new(false));
        let reply = Reply {
            id: r.id.clone(),
            channel,
        };
        self.active.insert(
            r.id.clone(),
            Active {
                fetch: kind == "fetch",
                side,
                sequence,
                cancel: cancel.clone(),
                reply: reply.clone(),
            },
        );
        if !matches!(kind, "fetch" | "job" | "jobs") {
            self.permanent.insert(r.id.clone());
        }
        if !leases.is_empty() {
            self.leases.insert(sequence, leases);
        }
        let result = if side == Side::Executor {
            self.forward(cmd, bytes)
        } else if let Some(work) = work {
            self.schedule(work, r.id.clone(), sequence, cancel)
        } else {
            self.fast(kind, cmd, &reply)
        };
        if let Err(e) = result {
            self.cancel(&r.id, "input", &format!("{e:#}"));
            self.leases.remove(&sequence);
        }
        Ok(())
    }

    fn forward(&mut self, cmd: &Value, bytes: Vec<u8>) -> Result<()> {
        ensure!(
            self.root.is_some(),
            "executor is {}",
            self.detail["state"].as_str().unwrap_or("starting")
        );
        if cmd["type"] == "status" || cmd["type"] == "control" {
            return self
                .executor
                .as_ref()
                .context("executor is gone")?
                .send(bytes);
        }
        let mut wait = Vec::new();
        if cmd["type"] == "close" || cmd["type"] == "release" {
            for (token, artifact) in &mut self.artifacts {
                if cmd["artifact"] == *token || cmd["handle"] == artifact.owner {
                    artifact.retired = true;
                    wait.push(token.clone());
                }
            }
        }
        self.deferred.push_back((bytes, wait));
        self.flush_deferred()
    }

    fn flush_deferred(&mut self) -> Result<()> {
        if !self.scripts.idle() {
            if self.root.is_some()
                && self
                    .active
                    .values()
                    .filter(|a| a.side == Side::Executor)
                    .count()
                    == self.deferred.len()
            {
                self.scripts.grant(&self.detail["detail"]);
            }
            if !self.scripts.idle() {
                return Ok(());
            }
        }
        while let Some((_, wait)) = self.deferred.front() {
            if self
                .leases
                .values()
                .flatten()
                .any(|token| wait.contains(token))
            {
                break;
            }
            let (bytes, _) = self.deferred.pop_front().context("missing request")?;
            self.executor
                .as_ref()
                .context("executor is gone")?
                .send(bytes)?;
        }
        Ok(())
    }

    fn add_status(&self, status: &mut Value, request: &str) {
        status["executor"] = self.detail.clone();
        for (id, active) in &self.active {
            if active.side == Side::Native && id != request {
                status["requests"][id] = json!("running");
            }
        }
    }

    pub fn finish_shutdown(&mut self) -> bool {
        if let Some(reply) = self.shutdown.take() {
            self.out.send(reply.result(json!({"type":"ack"})));
            return false;
        }
        true
    }

    fn fast(&mut self, kind: &str, cmd: &Value, reply: &Reply) -> Result<()> {
        match kind {
            "shutdown" => {
                self.active.remove(&reply.id);
                self.shutdown = Some(reply.clone());
                self.stop();
            }
            "status" => {
                let mut status = json!({"type":"status","requests":{},"handles":{},"allocators":[],"python_namespace_bytes":0});
                self.add_status(&mut status, &reply.id);
                self.finish(reply, status);
            }
            "control" => {
                ensure!(cmd["action"] == "cancel", "file work supports cancel");
                let id = cmd["target"].as_str().context("missing target")?;
                if let Some(a) = self.active.get(id) {
                    a.cancel.store(true, Ordering::Release);
                }
                if self.out.streams.contains_key(id) || self.uploads.contains_key(id) {
                    self.cancel(id, "canceled", "canceled");
                }
                self.finish(reply, json!({"type":"ack"}));
            }
            "upload" => {
                let path =
                    files::resolve(cmd["path"].as_str().context("missing upload destination")?)?;
                let upload = transfer::Upload::start(
                    path,
                    cmd["bytes"].as_u64().context("missing size")?,
                    cmd["sha256"].as_str().context("missing digest")?.into(),
                )?;
                if cmd["notify_ready"] == true && !upload.complete() {
                    self.out
                        .send(json!({"type":"progress","request":reply.id,"stage":"upload_ready"}));
                }
                self.uploads.insert(reply.id.clone(), upload);
                self.block(&reply.id, &[])?;
            }
            _ => bail!("unknown native command"),
        }
        Ok(())
    }

    fn block(&mut self, id: &str, data: &[u8]) -> Result<()> {
        let upload = self.uploads.get_mut(id).context("unknown upload")?;
        upload.write(data)?;
        if upload.complete() {
            self.uploads
                .remove(id)
                .context("missing upload")?
                .publish()?;
            let reply = self
                .active
                .get(id)
                .context("unknown request")?
                .reply
                .clone();
            self.finish(&reply, json!({"type":"ack"}));
        }
        Ok(())
    }

    fn finish(&mut self, r: &Reply, result: Value) {
        self.out.send(r.result(result));
        self.active.remove(&r.id);
    }
    fn cancel(&mut self, id: &str, kind: &str, reason: &str) {
        if let Some(active) = self.active.remove(id) {
            active.cancel.store(true, Ordering::Release);
            self.out.send(active.reply.error(kind, reason));
            if self.out.streams.remove(id).is_some() {
                self.leases.remove(&active.sequence);
            }
        }
        self.uploads.remove(id);
    }

    pub fn progress(&mut self, id: &str, sequence: u64, bytes: u64) {
        if let Some(a) = self.active.get(id)
            && a.sequence == sequence
        {
            self.out.send(json!({"type":"progress","request":id,"channel":a.reply.channel,"record":{"schema":2,"event":"advanced","task":1,"time":0.0,"done":bytes,"total":null,"eta":null,"metrics":{}}}));
        }
    }

    pub fn done(&mut self, id: &str, sequence: u64, result: Result<Output>) {
        if let Some(a) = self.active.get(id)
            && a.sequence == sequence
        {
            let reply = a.reply.clone();
            match result {
                Ok(Output::Result(value)) => self.finish(&reply, value),
                Ok(Output::Stream(p)) => {
                    if a.cancel.load(Ordering::Acquire) {
                        self.cancel(id, "canceled", "canceled");
                    } else {
                        self.out.streams.insert(id.into(), (reply, p));
                        return;
                    }
                }
                Err(e) => {
                    let kind = if a.cancel.load(Ordering::Acquire) {
                        "canceled"
                    } else {
                        "input"
                    };
                    self.cancel(id, kind, &format!("{e:#}"));
                }
            }
        }
        self.leases.remove(&sequence);
    }

    pub fn stream_done(&mut self, id: &str) {
        if let Some(a) = self.active.remove(id) {
            self.leases.remove(&a.sequence);
        }
    }

    pub fn flush(&mut self) {
        if let Err(e) = self.flush_deferred() {
            self.gone(self.generation, e.to_string());
        }
    }

    pub fn stop(&mut self) {
        if self.closing.is_some() {
            return;
        }
        self.jobs.stop_inline();
        self.closing = Some(Instant::now());
        self.scripts = Default::default();
        let local = self
            .active
            .values()
            .filter(|a| a.side == Side::Native)
            .map(|a| a.reply.id.clone())
            .collect::<Vec<_>>();
        for id in local {
            self.cancel(&id, "canceled", "canceled");
        }
        if self.root.is_none() {
            self.executor.take();
        } else if let Some(executor) = &mut self.executor {
            executor.eof();
        }
    }
}
