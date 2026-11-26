//! Asynchronous client for the official TomoJAX worker.
//!
//! The worker owns numerical state and commits. The UI owns this connection and
//! applies correlated replies; cancellation acknowledgment is never a terminal
//! for the target job. Normal shutdown keeps draining through process exit.

use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::protocol::{Ready, Response, ResultValue, SchemaCatalog};

pub type Wake = Arc<dyn Fn() + Send + Sync>;
const REQUEST_BYTES: usize = 1024 * 1024;
const RESPONSE_BYTES: u64 = 16 * 1024 * 1024;
const EXECUTION_LIMIT: usize = 60;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Lifecycle {
    Starting,
    Ready,
    Stopping,
    Exited,
    Failed(String),
}

#[derive(Debug)]
pub enum EngineEvent {
    Ready(Ready),
    Reply {
        response: Response,
        stale: bool,
    },
    Diagnostic(String),
    Disconnected {
        code: Option<i32>,
        reason: String,
        unfinished: Vec<String>,
    },
}

enum Incoming {
    Response(Response),
    Diagnostic(String),
    Fault(String),
    Exited {
        code: Option<i32>,
        description: String,
    },
}
enum Outgoing {
    Line(Vec<u8>),
    Close,
}

#[derive(Debug)]
struct Pending {
    channel: Option<String>,
    revision: Option<u64>,
    execution: bool,
}

/// Tracks observed admissions, not speculative channel replacements.
#[derive(Default)]
struct Requests {
    next: u64,
    pending: HashMap<String, Pending>,
    revisions: BTreeMap<String, u64>,
}

impl Requests {
    fn observe(&mut self, response: &Response) -> Result<bool, String> {
        let Some(id) = response.request() else {
            return Ok(false);
        };
        let Some(pending) = self.pending.get_mut(id) else {
            return Err(format!(
                "Worker replied to unknown or completed request {id}"
            ));
        };
        let mut stale = false;
        if !matches!(response, Response::ProtocolError { .. }) {
            let actual = response.channel();
            if pending.channel.as_deref() != actual.map(|c| c.name.as_str()) {
                return Err(format!("Worker changed the channel for request {id}"));
            }
            if let Some(channel) = actual {
                if channel.revision == 0 {
                    return Err("Worker channel revisions start at one".into());
                }
                if pending
                    .revision
                    .is_some_and(|revision| revision != channel.revision)
                {
                    return Err(format!(
                        "Worker changed the admitted revision for request {id}"
                    ));
                }
                pending.revision = Some(channel.revision);
                let newest = self.revisions.entry(channel.name.clone()).or_default();
                stale = channel.revision < *newest;
                *newest = (*newest).max(channel.revision);
            }
        }
        if response.terminal() {
            self.pending.remove(id);
        }
        Ok(stale)
    }
}

pub struct Engine {
    lifecycle: Lifecycle,
    ready: Option<Ready>,
    catalog: Option<SchemaCatalog>,
    requests: Requests,
    incoming: Option<mpsc::Receiver<Incoming>>,
    outgoing: Option<mpsc::SyncSender<Outgoing>>,
    terminate: mpsc::Sender<()>,
    supervisor: Option<JoinHandle<()>>,
    _workspace: tempfile::TempDir,
}

impl Engine {
    pub fn spawn(python: impl AsRef<Path>, wake: Wake) -> Result<Self, String> {
        let workspace = tempfile::Builder::new()
            .prefix("tomojax-studio-serve-")
            .tempdir()
            .map_err(|error| format!("Could not create the worker's private workspace: {error}"))?;
        let mut child = Command::new(python.as_ref())
            .args(["-u", "-m", "tomojax.cli.main", "serve"])
            .env("TMPDIR", workspace.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                format!(
                    "Could not start TomoJAX with {}: {e}",
                    python.as_ref().display()
                )
            })?;
        let (Some(mut input), Some(output), Some(errors)) =
            (child.stdin.take(), child.stdout.take(), child.stderr.take())
        else {
            let _ = child.kill();
            let _ = child.wait();
            return Err("TomoJAX worker started without its protocol pipes".into());
        };
        let (send, incoming) = mpsc::sync_channel(128);
        let (outgoing, receive) = mpsc::sync_channel(128);
        let (terminate, stopping) = mpsc::channel();
        let writer_send = send.clone();
        let writer_wake = wake.clone();
        let writer_stop = terminate.clone();
        let writer = thread::spawn(move || {
            while let Ok(Outgoing::Line(line)) = receive.recv() {
                if let Err(error) = input.write_all(&line).and_then(|()| input.flush()) {
                    let _ =
                        writer_send.send(Incoming::Fault(format!("Worker input closed: {error}")));
                    writer_wake();
                    let _ = writer_stop.send(());
                    break;
                }
            }
        });
        let reader_send = send.clone();
        let reader_wake = wake.clone();
        let reader_stop = terminate.clone();
        let reader = thread::spawn(move || {
            let mut buffered = BufReader::new(output);
            loop {
                let mut line = Vec::new();
                let message = match buffered
                    .by_ref()
                    .take(RESPONSE_BYTES + 1)
                    .read_until(b'\n', &mut line)
                {
                    Ok(0) => break,
                    Ok(_) if line.len() as u64 > RESPONSE_BYTES || !line.ends_with(b"\n") => {
                        Incoming::Fault(
                            "Worker returned an oversized or incomplete protocol line".into(),
                        )
                    }
                    Ok(_) => match serde_json::from_slice::<Response>(&line) {
                        Ok(response) => Incoming::Response(response),
                        Err(error) => Incoming::Fault(format!("Invalid worker response: {error}")),
                    },
                    Err(error) => Incoming::Fault(format!("Could not read worker output: {error}")),
                };
                let failed = matches!(message, Incoming::Fault(_));
                if reader_send.send(message).is_err() {
                    break;
                }
                reader_wake();
                if failed {
                    let _ = reader_stop.send(());
                    break;
                }
            }
        });
        let stderr_send = send.clone();
        let stderr_wake = wake.clone();
        let stderr = thread::spawn(move || {
            let mut buffered = BufReader::new(errors);
            loop {
                let mut line = Vec::new();
                match buffered.by_ref().take(65536).read_until(b'\n', &mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let message = String::from_utf8_lossy(&line).trim_end().to_owned();
                        if !message.is_empty() {
                            if stderr_send.send(Incoming::Diagnostic(message)).is_err() {
                                break;
                            }
                            stderr_wake();
                        }
                    }
                }
            }
        });
        let writer_close = outgoing.clone();
        let supervisor = thread::spawn(move || {
            let status = loop {
                match child.try_wait() {
                    Ok(Some(status)) => break Some(status),
                    Err(error) => {
                        let _ = send.send(Incoming::Fault(format!(
                            "Could not observe worker exit: {error}"
                        )));
                        let _ = child.kill();
                        break child.wait().ok();
                    }
                    Ok(None) => {}
                }
                match stopping.recv_timeout(Duration::from_millis(25)) {
                    Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                        let _ = child.kill();
                        break child.wait().ok();
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                }
            };
            let _ = writer_close.try_send(Outgoing::Close);
            let _ = writer.join();
            let _ = reader.join();
            let _ = stderr.join();
            let description = status
                .map(|status| status.to_string())
                .unwrap_or_else(|| "exit status unavailable".into());
            let _ = send.send(Incoming::Exited {
                code: status.and_then(|status| status.code()),
                description,
            });
            wake();
        });
        Ok(Self {
            lifecycle: Lifecycle::Starting,
            ready: None,
            catalog: None,
            requests: Requests::default(),
            incoming: Some(incoming),
            outgoing: Some(outgoing),
            terminate,
            supervisor: Some(supervisor),
            _workspace: workspace,
        })
    }

    pub fn lifecycle(&self) -> &Lifecycle {
        &self.lifecycle
    }
    pub fn ready(&self) -> Option<&Ready> {
        self.ready.as_ref()
    }
    pub fn catalog(&self) -> Option<&SchemaCatalog> {
        self.catalog.as_ref()
    }
    pub fn pending_count(&self) -> usize {
        self.requests.pending.len()
    }

    pub fn execution_capacity(&self) -> usize {
        if self.lifecycle != Lifecycle::Ready {
            return 0;
        }
        EXECUTION_LIMIT.saturating_sub(
            self.requests
                .pending
                .values()
                .filter(|request| request.execution)
                .count(),
        )
    }

    /// A replacement on a channel with outstanding work may displace queued work
    /// even when ordinary execution admission is full.
    pub fn channel_capacity(&self, channel: &str) -> usize {
        if self.lifecycle != Lifecycle::Ready {
            return 0;
        }
        self.execution_capacity().max(usize::from(
            self.requests
                .pending
                .values()
                .any(|pending| pending.execution && pending.channel.as_deref() == Some(channel)),
        ))
    }

    /// Queue a complete immutable command. Its terminal, not sending, establishes outcome.
    pub fn request(&mut self, command: Value, channel: Option<&str>) -> Result<String, String> {
        let tag = command
            .get("type")
            .and_then(Value::as_str)
            .ok_or("A command needs its type tag")?;
        let shutdown = tag == "shutdown";
        if self.lifecycle != Lifecycle::Ready
            && !(shutdown && self.lifecycle == Lifecycle::Starting)
        {
            return Err("The TomoJAX worker is not accepting commands".into());
        }
        if let Some(ready) = &self.ready
            && !ready.capabilities.iter().any(|c| c == tag)
        {
            return Err(format!("This TomoJAX build does not support {tag}"));
        }
        if channel.is_some_and(|name| name.is_empty() || name.len() > 256) {
            return Err("A channel name must contain 1–256 UTF-8 bytes".into());
        }
        let execution = !matches!(
            tag,
            "describe" | "devices" | "status" | "control" | "close" | "release" | "shutdown"
        );
        if execution
            && channel.map_or_else(
                || self.execution_capacity(),
                |name| self.channel_capacity(name),
            ) == 0
        {
            return Err("The execution queue is full; wait for a job to finish".into());
        }
        if !shutdown && self.requests.next >= 99000 {
            return Err(
                "Restart the worker before submitting more work; its session ID limit is near"
                    .into(),
            );
        }
        self.requests.next += 1;
        let id = format!("studio-{}", self.requests.next);
        let mut line = serde_json::to_vec(&json!({"id":id,"command":command,"channel":channel}))
            .map_err(|e| format!("Could not encode request: {e}"))?;
        if line.len() > REQUEST_BYTES {
            return Err("The operation exceeds the worker's 1 MiB request limit".into());
        }
        line.push(b'\n');
        self.outgoing
            .as_ref()
            .ok_or("The worker connection is closed")?
            .try_send(Outgoing::Line(line))
            .map_err(|e| match e {
                mpsc::TrySendError::Full(_) => "The worker input queue is full".to_owned(),
                mpsc::TrySendError::Disconnected(_) => "The worker input pipe is closed".to_owned(),
            })?;
        self.requests.pending.insert(
            id.clone(),
            Pending {
                channel: channel.map(str::to_owned),
                revision: None,
                execution,
            },
        );
        if shutdown {
            self.lifecycle = Lifecycle::Stopping;
        }
        Ok(id)
    }

    pub fn control(&mut self, target: &str, action: &str) -> Result<String, String> {
        if !matches!(action, "pause" | "resume" | "cancel") {
            return Err("Control must be pause, resume or cancel".into());
        }
        self.request(
            json!({"type":"control","target":target,"action":action}),
            None,
        )
    }

    pub fn release(&mut self, artifact: &str) -> Result<String, String> {
        self.request(json!({"type":"release","artifact":artifact}), None)
    }

    pub fn close_handle(&mut self, handle: &str) -> Result<String, String> {
        self.request(json!({"type":"close","handle":handle}), None)
    }

    pub fn shutdown(&mut self) -> Result<String, String> {
        self.request(json!({"type":"shutdown"}), None)
    }

    /// Closing-window path: drain real terminals through EOF before giving up on
    /// a noncooperative kernel. The caller applies these events before flushing
    /// the project; unfinished requests must be recorded as interrupted.
    pub fn drain_shutdown(&mut self, grace: Duration) -> Vec<EngineEvent> {
        let mut events = Vec::new();
        if matches!(self.lifecycle, Lifecycle::Ready | Lifecycle::Starting)
            && let Err(error) = self.shutdown()
        {
            events.push(EngineEvent::Diagnostic(error));
        }
        if self.lifecycle == Lifecycle::Exited {
            return events;
        }
        let mut deadline = Instant::now() + grace;
        let mut forced = false;
        loop {
            let next = self.poll();
            let ended = next
                .iter()
                .any(|event| matches!(event, EngineEvent::Disconnected { .. }));
            events.extend(next);
            if ended {
                return events;
            }
            if Instant::now() >= deadline {
                if forced {
                    events.push(EngineEvent::Diagnostic(
                        "The worker is still being reaped after forced shutdown".into(),
                    ));
                    return events;
                }
                self.lifecycle = Lifecycle::Failed("The worker did not reach a shutdown checkpoint before the closing deadline; unfinished work was interrupted".into());
                let _ = self.terminate.send(());
                forced = true;
                deadline = Instant::now() + Duration::from_secs(1);
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    /// Drain without blocking the UI. All terminals survive stale display revisions.
    pub fn poll(&mut self) -> Vec<EngineEvent> {
        let messages: Vec<_> = self
            .incoming
            .as_ref()
            .map(|r| r.try_iter().collect())
            .unwrap_or_default();
        let mut events = Vec::with_capacity(messages.len());
        for message in messages {
            match message {
                Incoming::Response(Response::Ready(ready)) => {
                    if self.ready.is_some() || ready.protocol != 1 || ready.progress != 2 {
                        self.lifecycle =
                            Lifecycle::Failed("Unsupported or repeated worker handshake".into());
                        let _ = self.terminate.send(());
                        continue;
                    }
                    self.ready = Some(ready.clone());
                    if self.lifecycle == Lifecycle::Starting {
                        self.lifecycle = Lifecycle::Ready;
                        if let Err(error) = self.request(json!({"type":"describe"}), None) {
                            self.lifecycle = Lifecycle::Failed(error);
                            let _ = self.terminate.send(());
                        }
                    }
                    events.push(EngineEvent::Ready(ready));
                }
                Incoming::Response(response) => match self.requests.observe(&response) {
                    Ok(stale) => {
                        if let Response::Result {
                            result: ResultValue::Description { parameters },
                            ..
                        } = &response
                        {
                            match SchemaCatalog::new(parameters.clone()) {
                                Ok(catalog) => self.catalog = Some(catalog),
                                Err(error) => {
                                    self.lifecycle = Lifecycle::Failed(error);
                                    let _ = self.terminate.send(());
                                }
                            }
                        }
                        events.push(EngineEvent::Reply { response, stale });
                    }
                    Err(error) => {
                        self.lifecycle = Lifecycle::Failed(error);
                        let _ = self.terminate.send(());
                    }
                },
                Incoming::Diagnostic(message) => events.push(EngineEvent::Diagnostic(message)),
                Incoming::Fault(message) => {
                    self.lifecycle = Lifecycle::Failed(message);
                    let _ = self.terminate.send(());
                }
                Incoming::Exited { code, description } => {
                    let normal = self.lifecycle == Lifecycle::Stopping && code == Some(0);
                    let reason = match &self.lifecycle {
                        Lifecycle::Failed(message) => message.clone(),
                        _ if normal => "TomoJAX shut down cleanly".into(),
                        _ => format!(
                            "TomoJAX disconnected ({description}); check durable outputs before retrying"
                        ),
                    };
                    self.lifecycle = if normal {
                        Lifecycle::Exited
                    } else {
                        Lifecycle::Failed(reason.clone())
                    };
                    let unfinished = self.requests.pending.drain().map(|(id, _)| id).collect();
                    events.push(EngineEvent::Disconnected {
                        code,
                        reason,
                        unfinished,
                    });
                }
            }
        }
        events
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Graceful application close calls shutdown() and keeps polling. A lost
        // window or panic must still reap the child; dropping the receiver first
        // unblocks bounded output writers before joining the supervisor.
        self.incoming.take();
        self.outgoing.take();
        let _ = self.terminate.send(());
        if let Some(supervisor) = self.supervisor.take() {
            let _ = supervisor.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reply(id: &str, revision: u64) -> Response {
        serde_json::from_value(json!({"type":"result","request":id,
            "channel":{"name":"plane","revision":revision},"result":{"type":"ack"}}))
        .unwrap()
    }

    #[test]
    fn rejected_replacement_does_not_advance_channel_and_old_terminals_remain_correlated() {
        let mut requests = Requests::default();
        for id in ["old", "replacement", "new"] {
            requests.pending.insert(
                id.into(),
                Pending {
                    channel: Some("plane".into()),
                    revision: None,
                    execution: true,
                },
            );
        }
        let rejection = Response::ProtocolError {
            kind: "invalid_request".into(),
            message: "invalid parameter".into(),
            request: Some("replacement".into()),
        };
        assert!(!requests.observe(&rejection).unwrap());
        assert!(requests.revisions.is_empty());
        assert!(!requests.observe(&reply("new", 2)).unwrap());
        assert!(requests.observe(&reply("old", 1)).unwrap());
        assert!(requests.pending.is_empty());
        assert!(requests.observe(&reply("old", 1)).is_err());
    }

    #[test]
    fn mismatched_response_channel_cannot_complete_another_job() {
        let mut requests = Requests::default();
        requests.pending.insert(
            "job".into(),
            Pending {
                channel: None,
                revision: None,
                execution: true,
            },
        );
        assert!(requests.observe(&reply("job", 1)).is_err());
        assert!(requests.pending.contains_key("job"));
    }

    #[test]
    fn a_full_execution_queue_still_admits_replacement_on_its_existing_channel() {
        let (outgoing, received) = mpsc::sync_channel(2);
        let (terminate, _) = mpsc::channel();
        let mut engine = Engine {
            lifecycle: Lifecycle::Ready,
            ready: None,
            catalog: None,
            requests: Requests::default(),
            incoming: None,
            outgoing: Some(outgoing),
            terminate,
            supervisor: None,
            _workspace: tempfile::tempdir().unwrap(),
        };
        for index in 0..EXECUTION_LIMIT {
            engine.requests.pending.insert(
                format!("queued-{index}"),
                Pending {
                    channel: (index == 0).then(|| "preview.slice".into()),
                    revision: (index == 0).then_some(1),
                    execution: true,
                },
            );
        }
        let command = json!({"type":"slice","source":{"type":"member","handle":"scan","member":"self"},"axis":"z","index":1});
        assert_eq!(engine.execution_capacity(), 0);
        assert!(engine.request(command.clone(), None).is_err());
        assert!(
            engine
                .request(command.clone(), Some("preview.other"))
                .is_err()
        );
        let replacement = engine.request(command, Some("preview.slice")).unwrap();
        let Outgoing::Line(line) = received.try_recv().unwrap() else {
            panic!("expected replacement request");
        };
        let envelope: Value = serde_json::from_slice(&line).unwrap();
        assert_eq!(envelope["id"], replacement);
        assert_eq!(envelope["channel"], "preview.slice");
        assert_eq!(engine.execution_capacity(), 0);
    }
}
