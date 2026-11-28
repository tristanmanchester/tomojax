//! Host-local admission for process runners and their interactive peers.
use rustix::process::getuid;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, ErrorKind},
    os::unix::fs::{DirBuilderExt, MetadataExt},
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

pub fn try_lock(path: &Path) -> io::Result<Option<File>> {
    lock_file(path, false)
}

fn lock_file(path: &Path, shared: bool) -> io::Result<Option<File>> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    match if shared {
        file.try_lock_shared()
    } else {
        file.try_lock()
    } {
        Ok(()) => Ok(Some(file)),
        Err(std::fs::TryLockError::WouldBlock) => Ok(None),
        Err(std::fs::TryLockError::Error(e)) => Err(e),
    }
}

pub struct Machine(PathBuf);

impl Machine {
    pub fn local() -> io::Result<Self> {
        let uid = getuid().as_raw();
        let path = PathBuf::from(format!("/tmp/tomojax-{uid}"));
        match fs::DirBuilder::new().mode(0o700).create(&path) {
            Err(e) if e.kind() != ErrorKind::AlreadyExists => return Err(e),
            _ => {}
        }
        let meta = fs::symlink_metadata(&path)?;
        if !meta.is_dir() || meta.uid() != uid || meta.mode() & 0o077 != 0 {
            return Err(io::Error::other(
                "the machine lock directory must be private and owned by this user",
            ));
        }
        Ok(Self(path))
    }

    pub fn take_job(&self, stopped: &impl Fn() -> bool) -> io::Result<(File, File)> {
        let wait = |name, shared| loop {
            if stopped() {
                return Err(io::Error::new(
                    ErrorKind::Interrupted,
                    "job stopped waiting for machine",
                ));
            }
            if let Some(lock) = lock_file(&self.0.join(name), shared)? {
                return Ok(lock);
            }
            thread::sleep(Duration::from_millis(50));
        };
        let job = wait("jobs.lock", true)?;
        Ok((job, wait("gpu.lock", false)?))
    }

    /// Keep the returned lease until the GPU executor process has been reaped.
    pub fn session_gpu(&self) -> io::Result<Option<File>> {
        let Some(_admission) = try_lock(&self.0.join("jobs.lock"))? else {
            return Ok(None);
        };
        try_lock(&self.0.join("gpu.lock"))
    }

    pub fn job_waiting_or_running(&self) -> io::Result<bool> {
        Ok(try_lock(&self.0.join("jobs.lock"))?.is_none())
    }
}
