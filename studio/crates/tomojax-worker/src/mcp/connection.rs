//! One noninteractive worker connection. A dropped transport fails its pending requests.
use anyhow::{Context, Result, anyhow, bail, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::{
    Arc, LazyLock, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::{mpsc, oneshot, watch},
};

pub(super) const PARAMETERS: &[u8] =
    include_bytes!("../../../../../src/tomojax/cli/parameters.json");
static SCHEMA: LazyLock<String> = LazyLock::new(|| format!("{:x}", Sha256::digest(PARAMETERS)));

#[derive(Clone, Default)]
pub struct Inventory {
    pub worker: Value,
    pub executor: Value,
    failure: Option<String>,
}

impl Inventory {
    fn check(&self, needs_executor: bool) -> Result<()> {
        if let Some(message) = &self.failure {
            bail!("{message}");
        }
        let detail = &self.executor["detail"];
        match self.executor["state"].as_str() {
            Some("ready") => ensure!(
                detail["schema"] == SCHEMA.as_str(),
                "Parameter schema mismatch: the MCP server runs TomoJAX {}, the executor runs TomoJAX {}.",
                super::identity().version,
                detail["versions"]["tomojax"].as_str().unwrap_or("unknown")
            ),
            Some("gone") if needs_executor => bail!("TomoJAX executor stopped: {detail}"),
            _ if needs_executor => bail!(
                "TomoJAX is waiting in the queue. {}",
                detail.as_str().unwrap_or("The executor is starting.")
            ),
            _ => {}
        }
        Ok(())
    }
}

pub struct Connections {
    pub name: String,
    current: tokio::sync::Mutex<Option<(Instant, Arc<Session>)>>,
}

impl Connections {
    pub fn new(name: String) -> Self {
        Self {
            name,
            current: Default::default(),
        }
    }

    pub async fn close(&self) {
        if let Some((_, session)) = self.current.lock().await.take() {
            session.stop.send_replace(true);
            let mut state = session.inventory.clone();
            let _ = state.wait_for(|s| s.failure.is_some()).await;
        }
    }

    pub async fn open(&self, needs_executor: bool) -> Result<(Arc<Session>, Inventory)> {
        let mut current = self.current.lock().await;
        let session = match current.as_ref() {
            Some((at, session))
                if session.inventory.borrow().failure.is_none()
                    || at.elapsed() < Duration::from_secs(5) =>
            {
                session.clone()
            }
            _ => {
                let session = Arc::new(Session::start(&self.name)?);
                *current = Some((Instant::now(), session.clone()));
                session
            }
        };
        drop(current);
        let mut state = session.inventory.clone();
        // A normal local executor starts immediately. A scheduler diagnostic is final
        // for this call, so the client can do other work while Slurm is waiting.
        let inventory = state
            .wait_for(|s| {
                s.failure.is_some()
                    || !s.worker.is_null()
                        && (!needs_executor
                            || s.executor["state"] == "ready"
                            || s.executor["state"] == "gone"
                            || s.executor["detail"].is_string()
                            || self.name != "local")
            })
            .await?
            .clone();
        inventory.check(needs_executor)?;
        Ok((session, inventory))
    }
}

#[derive(Debug)]
pub struct WorkerError(pub Value);
impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            self.0["message"]
                .as_str()
                .unwrap_or("Worker request failed"),
        )
    }
}
impl std::error::Error for WorkerError {}
pub struct Reply {
    pub header: Value,
    pub bytes: Vec<u8>,
}
impl Reply {
    pub fn result(&self) -> Result<&Value> {
        if self.header["type"] == "error" {
            return Err(WorkerError(self.header["error"].clone()).into());
        }
        ensure!(
            self.header["type"] == "result",
            "{}",
            self.header["error"]["message"]
                .as_str()
                .or(self.header["message"].as_str())
                .unwrap_or("Worker request failed")
        );
        Ok(&self.header["result"])
    }
}
struct Pending {
    activity: Option<super::activity::Activity>,
    upload_ready: Option<oneshot::Sender<()>>,
    done: oneshot::Sender<Result<Reply>>,
    bytes: Vec<u8>,
}
type PendingCalls = Arc<Mutex<Option<HashMap<String, Pending>>>>;

pub struct Session {
    pub renders: AtomicU64,
    pub ledger: Mutex<super::ledger::Ledger>,
    inventory: watch::Receiver<Inventory>,
    outgoing: mpsc::Sender<Vec<u8>>,
    pending: PendingCalls,
    next: AtomicU64,
    stop: watch::Sender<bool>,
}

impl Session {
    fn start(name: &str) -> Result<Self> {
        let mut child = Command::new(std::env::current_exe()?)
            .args(["connect", name])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()?;
        let input = child.stdin.take().context("missing worker stdin")?;
        let output = child.stdout.take().context("missing worker stdout")?;
        let stderr = child
            .stderr
            .take()
            .context("missing connection diagnostics")?;
        let (state, inventory) = watch::channel(Inventory::default());
        let (outgoing, receive) = mpsc::channel(64);
        let pending: PendingCalls = Arc::new(Mutex::new(Some(HashMap::new())));
        let (stop, mut stopping) = watch::channel(false);
        let calls = pending.clone();
        let name = name.to_owned();
        let credit = outgoing.downgrade();
        tokio::spawn(async move {
            let diag = tokio::spawn(async move {
                let mut lines = BufReader::new(stderr).lines();
                let mut last = String::new();
                while let Ok(Some(line)) = lines.next_line().await {
                    if !line.is_empty() {
                        last = line;
                    }
                }
                last
            });
            let reason = tokio::select! {
                r=drive(input,output,receive,&calls,&state,credit)=>r,
                _=stopping.changed()=>Ok(()),
            };
            let message = disconnect(&mut child, &name, reason, diag).await;
            state.send_modify(|s| s.failure = Some(message.clone()));
            for (_, pending) in calls.lock().unwrap().take().unwrap_or_default() {
                let _ = pending.done.send(Err(anyhow!(message.clone())));
            }
        });
        Ok(Self {
            renders: AtomicU64::new(1),
            ledger: Default::default(),
            inventory,
            outgoing,
            pending,
            next: AtomicU64::new(1),
            stop,
        })
    }

    pub async fn request(&self, command: Value) -> Result<Reply> {
        let cancels = matches!(
            command["type"].as_str(),
            Some("run" | "python" | "open" | "read" | "plan" | "validate" | "statistics")
        );
        let (id, mut recv) = self.begin(command, None).await?;
        if cancels && let Ok(token) = super::tasks::CANCEL.try_with(Clone::clone) {
            tokio::select! {
                biased;
                result = &mut recv => return result.context("Worker disconnected before replying")?,
                _ = token.cancelled() => {let _=self.begin(json!({"type":"control","action":"cancel","target":id}),None).await?;}
            }
        }
        recv.await.context("Worker disconnected before replying")?
    }

    pub async fn upload(&self, command: Value, bytes: Vec<u8>) -> Result<Reply> {
        let (ready, admitted) = oneshot::channel();
        let (id, recv) = self.begin(command, Some(ready)).await?;
        let outgoing = self.outgoing.clone();
        // Finish an admitted upload even if the MCP request is canceled.
        tokio::spawn(async move {
            if admitted.await.is_err() {
                return;
            }
            for chunk in bytes.chunks(crate::wire::BLOCK) {
                let mut frame =
                    crate::wire::line(&json!({"type":"block","request":id,"bytes":chunk.len()}));
                frame.extend_from_slice(chunk);
                if outgoing.send(frame).await.is_err() {
                    break;
                }
            }
        });
        recv.await.context("Worker disconnected during upload")?
    }

    async fn begin(
        &self,
        command: Value,
        upload_ready: Option<oneshot::Sender<()>>,
    ) -> Result<(String, oneshot::Receiver<Result<Reply>>)> {
        let id = self.next.fetch_add(1, Ordering::Relaxed).to_string();
        let mut line = serde_json::to_vec(&json!({"id":id,"command":command}))?;
        line.push(b'\n');
        ensure!(line.len() <= 1 << 20, "Worker request exceeds 1 MiB");
        let (done, recv) = oneshot::channel();
        let permit = self
            .outgoing
            .reserve()
            .await
            .context("Worker connection closed")?;
        self.pending
            .lock()
            .unwrap()
            .as_mut()
            .context("Worker connection closed")?
            .insert(
                id.clone(),
                Pending {
                    activity: super::activity::CURRENT.try_with(Clone::clone).ok(),
                    upload_ready,
                    done,
                    bytes: Vec::new(),
                },
            );
        permit.send(line);
        Ok((id, recv))
    }
}

async fn disconnect(
    child: &mut Child,
    name: &str,
    result: Result<()>,
    diag: tokio::task::JoinHandle<String>,
) -> String {
    let status = match tokio::time::timeout(Duration::from_secs(2), child.wait()).await {
        Ok(Ok(s)) => s.code(),
        _ => {
            let _ = child.kill().await;
            None
        }
    };
    let diagnostic = diag.await.unwrap_or_default();
    if status == Some(3) {
        return format!(
            "{name} needs a login. Ask the user to run `tomojax-worker connect {name} --login` in a terminal, or to connect in Studio; nothing was run."
        );
    }
    if !diagnostic.is_empty() {
        return diagnostic;
    }
    result
        .err()
        .map_or_else(|| format!("{name} disconnected"), |e| e.to_string())
}

fn progress(pending: &PendingCalls, row: &Value) -> Result<()> {
    let owner = row["request"].as_str().context("Missing progress owner")?;
    let mut pending = pending.lock().unwrap();
    if let Some(call) = pending.as_mut().and_then(|calls| calls.get_mut(owner)) {
        if row["stage"] == "upload_ready" {
            if let Some(ready) = call.upload_ready.take() {
                let _ = ready.send(());
            }
        } else if let Some(activity) = &call.activity {
            activity.record(&row["record"]);
        }
    }
    Ok(())
}
async fn drive(
    mut input: ChildStdin,
    output: ChildStdout,
    mut receive: mpsc::Receiver<Vec<u8>>,
    pending: &PendingCalls,
    state: &watch::Sender<Inventory>,
    credit: mpsc::WeakSender<Vec<u8>>,
) -> Result<()> {
    let writer = async {
        while let Some(line) = receive.recv().await {
            input.write_all(&line).await?;
        }
        input.shutdown().await?;
        Ok(())
    };
    let reader = async {
        let mut stream = BufReader::new(output);
        let mut line = String::new();
        loop {
            line.clear();
            if stream.read_line(&mut line).await? == 0 {
                return Ok(());
            }
            let row: Value = serde_json::from_str(&line)?;
            match row["type"].as_str() {
                Some("ready") => {
                    ensure!(
                        row["protocol"] == 1 && row["directory"].is_string(),
                        "Unsupported worker handshake"
                    );
                    state.send_modify(|s| s.worker = row);
                }
                Some("executor") => state.send_modify(|s| s.executor = row),
                Some("progress") => progress(pending, &row)?,
                Some("block") => {
                    let size = row["bytes"]
                        .as_u64()
                        .filter(|n| (1..=65536).contains(n))
                        .context("Invalid worker block size")?
                        as usize;
                    let mut bytes = vec![0; size];
                    stream.read_exact(&mut bytes).await?;
                    {
                        let mut calls = pending.lock().unwrap();
                        let call = calls
                            .as_mut()
                            .context("Worker connection closed")?
                            .get_mut(row["request"].as_str().context("Missing block owner")?)
                            .context("Unknown block owner")?;
                        ensure!(
                            call.bytes.len() + size <= 100 << 20,
                            "Worker payload exceeds 100 MiB"
                        );
                        call.bytes.extend_from_slice(&bytes);
                    }
                    credit
                        .upgrade()
                        .context("Connection closed")?
                        .send(format!("{{\"credit\":{size}}}\n").into_bytes())
                        .await?;
                }
                Some("result" | "error" | "protocol_error") => {
                    let id = row["request"]
                        .as_str()
                        .context("Uncorrelated worker error")?;
                    let call = pending
                        .lock()
                        .unwrap()
                        .as_mut()
                        .context("Worker connection closed")?
                        .remove(id);
                    let Some(call) = call else {
                        // Already queued blocks can outlive a failed write (e.g. a full disk).
                        if row["type"] == "protocol_error" && row["message"] == "unknown upload" {
                            continue;
                        }
                        bail!("Unknown worker request");
                    };
                    let _ = call.done.send(Ok(Reply {
                        header: row,
                        bytes: call.bytes,
                    }));
                }
                _ => bail!("Unsupported worker message"),
            }
        }
    };
    tokio::select! { result=writer => result, result=reader => result }
}
