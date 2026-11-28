// Ported from T3 Code (MIT), packages/ssh/src/tunnel.ts at d2c44031. See THIRD_PARTY_NOTICES.md.
use std::fs::{self, File};
use std::io;
use std::os::unix::fs::{DirBuilderExt, MetadataExt};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::Failure;
pub(super) const TICK: Duration = Duration::from_millis(20);
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

extern "C" fn interrupt(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::Relaxed);
}

/// Install in the dedicated worker or connection process. Its owner can cancel a login,
/// a probe or a live session with the same signal.
pub fn handle_signals() -> io::Result<()> {
    // SAFETY: the handler only sets a lock-free atomic flag.
    unsafe {
        let parent = libc::getppid();
        for sig in [libc::SIGTERM, libc::SIGHUP, libc::SIGINT] {
            if libc::signal(sig, interrupt as *const () as libc::sighandler_t) == libc::SIG_ERR {
                return Err(io::Error::last_os_error());
            }
        }
        if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM) < 0 {
            return Err(io::Error::last_os_error());
        }
        if libc::getppid() != parent {
            INTERRUPTED.store(true, Ordering::Relaxed);
        }
    }
    Ok(())
}

pub fn active() -> Result<(), Failure> {
    if INTERRUPTED.load(Ordering::Relaxed) {
        Err(Failure::interrupted())
    } else {
        Ok(())
    }
}

/// The unreaped leader pins the process-group ID until its owner waits.
pub fn own_group(command: &mut Command) -> &mut Command {
    owned_group(command, libc::SIGTERM)
}

/// A direct numerical child may hold a GPU or be stuck in Python shutdown.
pub fn own_executor(command: &mut Command) -> &mut Command {
    owned_group(command, libc::SIGKILL)
}

fn owned_group(command: &mut Command, death_signal: libc::c_int) -> &mut Command {
    // SAFETY: only async-signal-safe, allocation-free Linux operations run between fork and exec.
    unsafe {
        let parent = libc::getpid();
        command.process_group(0).pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, death_signal) < 0 {
                return Err(io::Error::last_os_error());
            }
            if libc::getppid() != parent {
                return Err(io::Error::from_raw_os_error(libc::ECANCELED));
            }
            Ok(())
        })
    }
}

pub(crate) fn kill_group(child: &mut Child) {
    // SAFETY: all callers own the unreaped process spawned with own_group.
    unsafe {
        libc::kill(-(child.id() as i32), libc::SIGKILL);
    }
}

pub(super) struct Owned(pub Child);
impl Owned {
    pub fn spawn(command: &mut Command) -> Result<Self, Failure> {
        active()?;
        own_group(command).spawn().map(Self).map_err(|e| {
            Failure::blocked(format!("Could not start {:?}: {e}", command.get_program()))
        })
    }
    pub fn wait(&mut self, deadline: Option<Instant>) -> Result<ExitStatus, Failure> {
        loop {
            active()?;
            if let Some(status) = self.0.try_wait()? {
                return Ok(status);
            }
            if deadline.is_some_and(|end| Instant::now() >= end) {
                return Err(Failure::silent("Timed out waiting for the command"));
            }
            std::thread::sleep(TICK);
        }
    }
}
impl Drop for Owned {
    fn drop(&mut self) {
        if matches!(self.0.try_wait(), Ok(None)) {
            kill_group(&mut self.0);
            let _ = self.0.wait();
        }
    }
}

pub(super) fn runtime_dir() -> Result<PathBuf, Failure> {
    // SAFETY: geteuid has no preconditions.
    let uid = unsafe { libc::geteuid() };
    let path = std::env::var_os("XDG_RUNTIME_DIR")
        .map(|p| PathBuf::from(p).join("tomojax"))
        .unwrap_or_else(|| PathBuf::from(format!("/tmp/tomojax-{uid}")));
    match fs::DirBuilder::new().mode(0o700).create(&path) {
        Err(e) if e.kind() != io::ErrorKind::AlreadyExists => return Err(e.into()),
        _ => {}
    }
    let meta = fs::symlink_metadata(&path)?;
    if !meta.is_dir() || meta.uid() != uid || meta.mode() & 0o077 != 0 {
        return Err(Failure::blocked(format!(
            "{} must be a private directory owned by this user",
            path.display()
        )));
    }
    Ok(path)
}

pub(super) fn login_lock(dir: &Path, key: &str, deadline: Instant) -> Result<File, Failure> {
    let file = File::options()
        .write(true)
        .create(true)
        .truncate(false)
        .open(dir.join(format!("{key}.lock")))?;
    loop {
        active()?;
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(std::fs::TryLockError::Error(e)) => return Err(e.into()),
            Err(std::fs::TryLockError::WouldBlock) if Instant::now() >= deadline => {
                return Err(Failure::silent(
                    "Timed out waiting for another login to finish",
                ));
            }
            Err(std::fs::TryLockError::WouldBlock) => std::thread::sleep(TICK),
        }
    }
}

pub(crate) fn terminate(child: &mut Child) -> io::Result<ExitStatus> {
    // SAFETY: caller owns the unreaped leader created by own_group.
    unsafe {
        libc::kill(-(child.id() as i32), libc::SIGTERM);
    }
    let deadline = Instant::now() + Duration::from_millis(500);
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        std::thread::sleep(TICK);
    }
    kill_group(child);
    child.wait()
}
