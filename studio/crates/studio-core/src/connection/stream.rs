//! Bound the first record even if ssh dribbles bytes. After ready the stream is
//! opaque, including any binary blocks following the newline in the same read.
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, RawFd};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use super::{
    Failure,
    process::{Owned, active},
};
use crate::protocol::{ExecutorState, Peer, Ready};
use serde_json::Value;

const LIMIT: usize = 1024 * 1024;
enum Stage {
    Starting,
    Forward,
    Executor(Value),
    Shutdown,
}
struct Handshake {
    line: Vec<u8>,
    stage: Stage,
    check: bool,
}
impl Handshake {
    fn output(&mut self, mut bytes: &[u8], worker: &mut Owned) -> Result<(), Failure> {
        while !bytes.is_empty() {
            match self.stage {
                Stage::Forward => {
                    let mut stdout = std::io::stdout().lock();
                    stdout.write_all(bytes)?;
                    stdout.flush()?;
                    break;
                }
                Stage::Shutdown => break,
                _ => {}
            }
            let end = bytes
                .iter()
                .position(|b| *b == b'\n')
                .map_or(bytes.len(), |i| i + 1);
            self.line.extend_from_slice(&bytes[..end]);
            bytes = &bytes[end..];
            if self.line.len() > LIMIT {
                return Err(Failure::blocked("Worker handshake exceeds 1 MiB"));
            }
            if !self.line.ends_with(b"\n") {
                break;
            }
            let record: Value = serde_json::from_slice(&self.line)
                .map_err(|e| Failure::blocked(format!("Invalid worker handshake: {e}")))?;
            self.record(record, worker)?;
            self.line.clear();
        }
        Ok(())
    }

    fn record(&mut self, record: Value, worker: &mut Owned) -> Result<(), Failure> {
        let (session, versions, devices) = match &self.stage {
            Stage::Starting => {
                if record["type"] != "ready" {
                    return Err(Failure::blocked("Expected worker ready"));
                }
                let ready: Ready = serde_json::from_value(record.clone())
                    .map_err(|e| Failure::blocked(format!("Invalid ready: {e}")))?;
                if !ready.supported() || !matches!(ready.peer, Peer::Worker { .. }) {
                    return Err(Failure::blocked("Unsupported worker protocol"));
                }
                if !self.check {
                    let mut stdout = std::io::stdout().lock();
                    stdout.write_all(&self.line)?;
                    stdout.flush()?;
                    self.stage = Stage::Forward;
                    return Ok(());
                }
                self.stage = Stage::Executor(record);
                return Ok(());
            }
            Stage::Executor(session) => {
                if record["type"] != "executor" {
                    return Ok(());
                }
                let state: ExecutorState = serde_json::from_value(record)
                    .map_err(|e| Failure::blocked(format!("Invalid executor event: {e}")))?;
                match state {
                    ExecutorState::Starting(_) => return Ok(()),
                    ExecutorState::Gone(reason) => {
                        return Err(Failure::silent(format!(
                            "Executor stopped before ready: {reason}"
                        )));
                    }
                    ExecutorState::Ready(Ready {
                        protocol: 1,
                        peer:
                            Peer::Executor {
                                progress: 2,
                                versions,
                                devices,
                            },
                        ..
                    }) => (session, versions, devices),
                    ExecutorState::Ready(_) => {
                        return Err(Failure::blocked("Unsupported executor handshake"));
                    }
                }
            }
            _ => return Ok(()),
        };
        println!(
            "{}",
            serde_json::json!({"host":session["host"],"jobs":session["jobs"],"tomojax":versions.tomojax,"devices":devices})
        );
        if let Some(mut stdin) = worker.0.stdin.take() {
            stdin.write_all(b"{\"id\":\"check\",\"command\":{\"type\":\"shutdown\"}}\n")?;
        }
        self.stage = Stage::Shutdown;
        Ok(())
    }
}

/// A ready fd can contain just one byte. Read it once and recheck the deadline.
fn readable(fds: [RawFd; 2]) -> std::io::Result<[bool; 2]> {
    let mut poll = fds.map(|fd| libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    });
    // SAFETY: poll receives a live array of exactly two pollfd values.
    let result = unsafe { libc::poll(poll.as_mut_ptr(), 2, 50) };
    if result < 0 {
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(error);
        }
        return Ok([false; 2]);
    }
    Ok(poll.map(|p| p.revents != 0))
}

pub(super) fn session(
    command: &mut Command,
    check: bool,
    timeout: Duration,
) -> Result<i32, Failure> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    if check {
        command.stdin(Stdio::piped());
    }
    let mut worker = Owned::spawn(command)?;
    let mut output = worker
        .0
        .stdout
        .take()
        .ok_or_else(|| Failure::blocked("Missing worker stdout"))?;
    let mut errors = worker
        .0
        .stderr
        .take()
        .ok_or_else(|| Failure::blocked("Missing worker stderr"))?;
    let mut fds = [output.as_raw_fd(), errors.as_raw_fd()];
    let mut handshake = Handshake {
        line: Vec::new(),
        stage: Stage::Starting,
        check,
    };
    let mut deadline = Some(Instant::now() + timeout);
    let mut tail = Vec::new();
    let mut next_ping = Instant::now() + Duration::from_secs(15);
    let mut sequence = 0u64;
    let mut buf = [0; 32768];
    while fds != [-1, -1] {
        active()?;
        if deadline.is_some_and(|d| Instant::now() >= d) {
            return Err(Failure::silent(
                if matches!(handshake.stage, Stage::Shutdown) {
                    "Worker did not shut down after check".to_string()
                } else {
                    format!("Worker did not send ready within {} s", timeout.as_secs())
                },
            ));
        }
        if matches!(handshake.stage, Stage::Executor(_)) && Instant::now() >= next_ping {
            sequence += 1;
            if let Some(stdin) = worker.0.stdin.as_mut() {
                writeln!(
                    stdin,
                    "{}",
                    serde_json::json!({"id":format!("check-{sequence}"),"command":{"type":"status"}})
                )?;
            }
            next_ping = Instant::now() + Duration::from_secs(15);
        }
        let [out, err] = readable(fds)?;
        if err {
            let n = errors.read(&mut buf)?;
            if n == 0 {
                fds[1] = -1;
            } else {
                std::io::stderr().lock().write_all(&buf[..n])?;
                tail.extend_from_slice(&buf[..n]);
                tail.drain(..tail.len().saturating_sub(65536));
            }
        }
        if out {
            let n = output.read(&mut buf)?;
            if n == 0 {
                fds[0] = -1;
            } else {
                let was_shutdown = matches!(handshake.stage, Stage::Shutdown);
                handshake.output(&buf[..n], &mut worker)?;
                match handshake.stage {
                    Stage::Shutdown if !was_shutdown => deadline = Some(Instant::now() + timeout),
                    Stage::Forward | Stage::Executor(_) => deadline = None,
                    _ => {}
                }
            }
        }
    }
    let status = worker.wait(deadline)?;
    let code = status.code().unwrap_or(1);
    if matches!(handshake.stage, Stage::Forward | Stage::Shutdown) {
        return Ok(code);
    }
    if matches!(handshake.stage, Stage::Executor(_)) {
        return Err(Failure::silent(
            "Worker exited before the executor was ready",
        ));
    }
    Err(startup_failure(status, &tail))
}

fn startup_failure(status: std::process::ExitStatus, stderr: &[u8]) -> Failure {
    let diagnostics = String::from_utf8_lossy(stderr);
    if status.code() == Some(255) || super::auth::refused(&diagnostics) {
        return Failure::ssh(diagnostics.trim());
    }
    let message = format!(
        "Worker exited before ready ({status}): {}",
        diagnostics.trim()
    );
    if status.code() == Some(4) {
        Failure::silent(message)
    } else {
        Failure::blocked(message)
    }
}
