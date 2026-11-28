//! Blocking I/O runs in bounded pools. The owner retains request admission and file leases.
use super::Session;
use crate::{Event, files, pixels, transfer, wire::Payload};
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

impl Session {
    pub(super) fn schedule(
        &self,
        work: Work,
        id: String,
        sequence: u64,
        cancel: Arc<AtomicBool>,
    ) -> Result<()> {
        let send = self.events.clone();
        let pool = if matches!(work, Work::Job(..)) {
            &self.job_pool
        } else {
            &self.pool
        };
        pool.submit(move || {
            if cancel.load(Ordering::Acquire) {
                let _ = send.send(Event::Done(id, sequence, Err(anyhow::anyhow!("canceled"))));
                return;
            }
            let result = work.run(&cancel, |n| {
                let _ = send.send(Event::Progress(id.clone(), sequence, n));
            });
            let _ = send.send(Event::Done(id, sequence, result));
        })
    }

    pub(super) fn prepare(
        &self,
        kind: &str,
        cmd: &Value,
        leases: &mut Vec<String>,
    ) -> Result<Option<Work>> {
        Ok(Some(match kind {
            "submit" | "jobs" | "job" | "job_control" => {
                Work::Job(serde_json::from_value(cmd.clone())?, Arc::clone(&self.jobs))
            }
            "fetch" => {
                let fetch: pixels::Fetch = serde_json::from_value(cmd.clone())?;
                if let Some(path) = cmd["storage"]["path"].as_str() {
                    leases.extend(self.lease(path, fetch.storage.artifact())?);
                }
                Work::Fetch(fetch)
            }
            "copy" => {
                let request: files::Copy = serde_json::from_value(cmd.clone())?;
                for source in &request.sources {
                    leases.extend(self.lease(&source.path, source.artifact.as_deref())?);
                }
                Work::Copy(request)
            }
            "stored" => Work::Stored(
                files::resolve(cmd["path"].as_str().context("expected path")?)?,
                cmd.get("inspection").cloned(),
            ),
            "list" | "peek" | "download" => {
                let path = cmd["path"]
                    .as_str()
                    .or_else(|| (kind == "list" && cmd["path"].is_null()).then_some("~"))
                    .context("expected path")?;
                Work::File(
                    kind.into(),
                    files::resolve(path)?,
                    cmd["after"].as_str().map(str::to_owned),
                )
            }
            _ => return Ok(None),
        }))
    }
}

pub(super) enum Work {
    Job(
        tomojax_jobs::commands::Command,
        Arc<tomojax_jobs::service::Service>,
    ),
    Fetch(pixels::Fetch),
    Copy(files::Copy),
    Stored(PathBuf, Option<Value>),
    File(String, PathBuf, Option<String>),
}
pub enum Output {
    Result(Value),
    Stream(Payload),
}
impl Work {
    fn run(self, cancel: &AtomicBool, progress: impl FnMut(u64)) -> Result<Output> {
        Ok(match self {
            Work::Job(command, service) => Output::Result(command.execute(&service, |path| {
                files::resolve(
                    path.to_str()
                        .ok_or_else(|| std::io::Error::other("expected a UTF-8 path"))?,
                )
                .map_err(std::io::Error::other)
            })?),
            Work::Fetch(request) => {
                let (result, bytes) = pixels::fetch(&request)?;
                Output::Stream(Payload::Frame {
                    result,
                    bytes,
                    offset: 0,
                })
            }
            Work::Copy(request) => Output::Result(files::copy(&request, cancel, progress)?),
            Work::Stored(path, report) => Output::Result(match report {
                Some(report) => {
                    let arrays = crate::metadata::inspection_arrays(&report, &path)
                        .map_err(anyhow::Error::msg)?;
                    json!({"type":"stored","kind":crate::metadata::suggested_kind(&report),"arrays":arrays,"report":report})
                }
                None => files::stored(&path)?,
            }),
            Work::File(kind, path, after) => match kind.as_str() {
                "list" => Output::Result(files::list(&path, after.as_deref())?),
                "peek" => Output::Result(files::peek(&path)?),
                "download" => Output::Stream(Payload::File(transfer::Download::open(&path)?)),
                _ => bail!("unknown file operation"),
            },
        })
    }
}
