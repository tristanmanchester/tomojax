//! Host-local admission for process runners and their interactive peers.
use rustix::process::getuid;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, ErrorKind},
    os::{
        fd::AsRawFd,
        unix::fs::{DirBuilderExt, MetadataExt},
    },
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
    let kind = if shared { libc::F_RDLCK } else { libc::F_WRLCK };
    match ofd(&file, libc::F_OFD_SETLK, kind) {
        Ok(_) => Ok(Some(file)),
        Err(e) if matches!(e.raw_os_error(), Some(libc::EAGAIN | libc::EACCES)) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Query the conflicting open-file-description lock without ever taking it.
/// A reader must neither deny admission to a runner nor appear live to a peer.
pub fn held(path: &Path) -> io::Result<bool> {
    let Some(file) = crate::directory::optional(File::open(path))? else {
        return Ok(false);
    };
    Ok(ofd(&file, libc::F_OFD_GETLK, libc::F_WRLCK)? != libc::F_UNLCK as i16)
}

fn ofd(file: &File, command: libc::c_int, kind: libc::c_int) -> io::Result<libc::c_short> {
    let mut lock = libc::flock {
        l_type: kind as _,
        l_whence: libc::SEEK_SET as _,
        l_start: 0,
        l_len: 0,
        l_pid: 0,
    };
    // SAFETY: GETLK and SETLK both take a pointer to this initialized flock.
    // The descriptor and writable buffer remain live for the call.
    if unsafe { libc::fcntl(file.as_raw_fd(), command, &mut lock) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(lock.l_type)
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
        let lease = try_lock(&self.0.join("gpu.lock"))?;
        if lease.is_some() && self.job_waiting_or_running()? {
            return Ok(None);
        }
        Ok(lease)
    }

    pub fn job_waiting_or_running(&self) -> io::Result<bool> {
        held(&self.0.join("jobs.lock"))
    }
}
