//! Bind durable jobs to this worker's executor generation and stored-array reader.
use crate::{Options, files, inline};
use anyhow::{Context, Result};
use serde_json::Value;
use std::{
    fs::File,
    io,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
use tomojax_jobs::{
    executor::Executor,
    locks::Machine,
    runner::{self, InlineSession},
    script::Launcher,
    service::Service,
};

impl Executor for inline::Stream {
    fn send(&mut self, line: &Value) -> io::Result<()> {
        self.send(line)
    }
    fn receive(&mut self, timeout: Duration) -> io::Result<Option<Value>> {
        self.receive(timeout)
    }
}

impl InlineSession for inline::Scripts {
    fn acquire(&self, stopping: &dyn Fn() -> bool) -> io::Result<Box<dyn Executor + '_>> {
        Ok(Box::new(self.acquire(stopping)?))
    }
}

fn stored(path: &Path) -> io::Result<Value> {
    let mut header = files::stored(path).map_err(io::Error::other)?;
    if header["kind"].is_null() {
        return Err(io::Error::other(
            "could not read the published dataset's metadata",
        ));
    }
    Ok(header["arrays"].take())
}

pub(crate) fn service(opts: &Options, scripts: inline::Scripts) -> Result<Arc<Service>> {
    Ok(Arc::new(Service::new(
        opts.jobs,
        (&opts.python).into(),
        std::env::current_exe()?,
        &opts.sbatch,
        Arc::new(scripts),
        Arc::new(stored),
    )?))
}

pub(crate) fn run(mut args: impl Iterator<Item = String>) -> Result<u8> {
    let path = args.next().context("job requires a directory")?;
    anyhow::ensure!(args.next().is_none(), "job takes one directory");
    Ok(runner::run_detached(&files::resolve(&path)?, &stored)? as u8)
}

/// The lease travels with a move to CPU, so its drop follows the synchronous reap.
pub enum Move {
    ToCpu(File),
    ToGpu,
}

pub struct Placement {
    machine: Machine,
    lease: Option<File>,
    next: Instant,
}

impl Placement {
    pub fn new(launcher: Launcher) -> Result<Option<Self>> {
        if launcher != Launcher::Process {
            return Ok(None);
        }
        let machine = Machine::local()?;
        let lease = machine.session_gpu()?;
        Ok(Some(Self {
            machine,
            lease,
            next: Instant::now(),
        }))
    }
    pub fn cpu_only(&self) -> bool {
        self.lease.is_none()
    }
    pub fn check(&mut self, idle: bool) -> io::Result<Option<Move>> {
        if Instant::now() < self.next {
            return Ok(None);
        }
        self.next = Instant::now() + Duration::from_secs(1);
        if self.lease.is_some() && self.machine.job_waiting_or_running()? {
            return Ok(self.lease.take().map(Move::ToCpu));
        }
        if self.lease.is_none()
            && idle
            && let Some(lease) = self.machine.session_gpu()?
        {
            self.lease = Some(lease);
            return Ok(Some(Move::ToGpu));
        }
        Ok(None)
    }
}
