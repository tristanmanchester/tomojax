//! A job has a private stream of replies, including its executor's initial ready record.
//! Inline jobs receive that stream from the worker's dispatcher.
use crate::directory::optional;
use serde_json::Value;
use std::{
    fs::{self, File},
    io::{self, BufRead, BufReader, Write},
    os::unix::process::CommandExt,
    path::Path,
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{self, Receiver, RecvTimeoutError},
    thread,
    time::Duration,
};

pub trait Executor {
    fn send(&mut self, line: &Value) -> io::Result<()>;
    fn receive(&mut self, timeout: Duration) -> io::Result<Option<Value>>;
}

pub struct ChildExecutor {
    child: Child,
    input: ChildStdin,
    replies: Receiver<io::Result<Value>>,
}

impl ChildExecutor {
    pub fn start(python: &Path, dir: &Path) -> io::Result<Self> {
        let log = File::options()
            .append(true)
            .create(true)
            .open(dir.join("log.txt"))?;
        let mut command = Command::new(python);
        command
            .args(["-u", "-m", "tomojax.cli.main", "serve"])
            .env("TMPDIR", dir.join("work/session"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(log)
            .process_group(0);
        let parent = rustix::process::getpid();
        // SAFETY: prctl and getppid are syscalls and no process-shared resources are touched.
        unsafe {
            command.pre_exec(move || {
                rustix::process::set_parent_process_death_signal(Some(
                    rustix::process::Signal::KILL,
                ))?;
                if rustix::process::getppid() != Some(parent) {
                    return Err(io::Error::other("runner exited during executor startup"));
                }
                Ok(())
            });
        }
        let mut child = command.spawn()?;
        let stdout = child.stdout.take().expect("piped");
        let input = child.stdin.take().expect("piped");
        let (send, replies) = mpsc::sync_channel(32);
        thread::spawn(move || {
            for line in BufReader::new(stdout).split(b'\n') {
                let parsed = line.and_then(|s| Ok(serde_json::from_slice(&s)?));
                if send.send(parsed).is_err() {
                    break;
                }
            }
        });
        Ok(Self {
            child,
            input,
            replies,
        })
    }
}

impl Executor for ChildExecutor {
    fn send(&mut self, line: &Value) -> io::Result<()> {
        serde_json::to_writer(&mut self.input, line)?;
        self.input.write_all(b"\n")
    }
    fn receive(&mut self, timeout: Duration) -> io::Result<Option<Value>> {
        match self.replies.recv_timeout(timeout) {
            Ok(line) => line.map(Some),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(io::Error::other(format!(
                "executor stopped: {:?}",
                self.child.try_wait()?
            ))),
        }
    }
}

impl Drop for ChildExecutor {
    fn drop(&mut self) {
        // Each request has already drained its terminal; a process still starting has
        // no live request. A detached job owns no executor after its attempt closes.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub fn clear_work(dir: &Path) -> io::Result<()> {
    optional(fs::remove_dir_all(dir.join("work"))).map(drop)
}
