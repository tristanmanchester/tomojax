//! Blocking I/O runs in bounded pools. The owner retains request admission and file leases.
use super::Session;
use crate::{Event, files, pixels, transfer, wire::Payload};
use anyhow::{Context, Result};
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
                Work::Fetch(fetch, self.planes.clone())
            }
            "copy" => {
                let request: files::Copy = serde_json::from_value(cmd.clone())?;
                for source in &request.sources {
                    leases.extend(self.lease(&source.path, source.artifact.as_deref())?);
                }
                Work::Copy(request)
            }
            "project_results" => Work::Results(serde_json::from_value(cmd.clone())?),
            "stored" => Work::Stored(
                files::resolve(cmd["path"].as_str().context("expected path")?)?,
                cmd.get("inspection").cloned(),
            ),
            "list" => {
                let path: Option<String> = serde_json::from_value(cmd["path"].clone())?;
                Work::List(
                    files::resolve(path.as_deref().unwrap_or("~"))?,
                    cmd["after"].as_str().map(str::to_owned),
                    cmd.get("limit")
                        .map(|v| serde_json::from_value(v.clone()))
                        .transpose()?
                        .unwrap_or(4096),
                )
            }
            "peek" => Work::Peek(files::resolve(
                cmd["path"].as_str().context("expected path")?,
            )?),
            "download" => {
                let path = cmd["path"].as_str().context("expected path")?;
                leases.extend(self.lease(path, cmd["artifact"].as_str())?);
                Work::Download(files::resolve(path)?)
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
    Fetch(pixels::Fetch, Arc<pixels::PlaneCache>),
    Copy(files::Copy),
    Results(crate::results::Results),
    Stored(PathBuf, Option<Value>),
    List(PathBuf, Option<String>, usize),
    Peek(PathBuf),
    Download(PathBuf),
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
            Work::Fetch(request, cache) => {
                let (result, bytes) = pixels::fetch(&request, &cache)?;
                Output::Stream(Payload::Frame {
                    result,
                    bytes,
                    offset: 0,
                })
            }
            Work::Copy(request) => Output::Result(files::copy(&request, cancel, progress)?),
            Work::Results(request) => Output::Result(request.run()?),
            Work::Stored(path, report) => Output::Result(match report {
                Some(report) => {
                    let arrays = crate::metadata::inspection_arrays(&report, &path)
                        .map_err(anyhow::Error::msg)?;
                    json!({"type":"stored","kind":crate::metadata::suggested_kind(&report),"arrays":arrays,"report":report})
                }
                None => files::stored(&path)?,
            }),
            Work::List(path, after, limit) => {
                Output::Result(files::list(&path, after.as_deref(), limit)?)
            }
            Work::Peek(path) => Output::Result(files::peek(&path)?),
            Work::Download(path) => Output::Stream(Payload::File(transfer::Download::open(&path)?)),
        })
    }
}
