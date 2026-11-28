//! Own one executor process group, including a pending scheduler launcher.
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, SyncSender};
use std::thread;

use anyhow::{Context, Result, ensure};
use rustix::process::{Pid, Signal, kill_process_group};
use tempfile::TempDir;

use crate::{Event, Options};

pub struct Executor {
    child: Child,
    input: Option<SyncSender<Vec<u8>>>,
    pub directory: TempDir,
}

impl Executor {
    pub fn start(
        options: &Options,
        generation: u64,
        events: &SyncSender<Event>,
        cpu_only: bool,
    ) -> Result<Self> {
        let base = options.results.join(".tomojax/sessions");
        fs::create_dir_all(&base)?;
        let directory = tempfile::Builder::new()
            .prefix("session-")
            .tempdir_in(base)?;
        let args = options
            .session
            .iter()
            .cloned()
            .chain([
                options.python.clone(),
                "-u".into(),
                "-m".into(),
                "tomojax.cli.main".into(),
                "serve".into(),
            ])
            .collect::<Vec<_>>();
        let mut command = Command::new(&args[0]);
        command
            .args(&args[1..])
            .env("TMPDIR", directory.path())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if cpu_only {
            command
                .env("JAX_PLATFORMS", "cpu")
                .env("CUDA_VISIBLE_DEVICES", "");
        }
        let command = if options.session.is_empty() {
            studio_core::connection::process::own_executor(&mut command)
        } else {
            studio_core::connection::process::own_group(&mut command)
        };
        let mut child = command
            .spawn()
            .with_context(|| format!("start {}", args[0]))?;
        let mut stdin = child.stdin.take().context("missing executor stdin")?;
        let stdout = child.stdout.take().context("missing executor stdout")?;
        let stderr = child.stderr.take().context("missing executor stderr")?;
        let (input, requests) = mpsc::sync_channel::<Vec<u8>>(128);
        thread::spawn(move || {
            for bytes in requests {
                if stdin.write_all(&bytes).is_err() {
                    break;
                }
            }
        });
        let send = events.clone();
        thread::spawn(move || {
            let mut stdout = BufReader::new(stdout);
            let reason = loop {
                let mut bytes = Vec::new();
                match stdout
                    .by_ref()
                    .take((16 << 20) + 1)
                    .read_until(b'\n', &mut bytes)
                {
                    Ok(0) => break "executor output ended".into(),
                    Ok(n) if n <= 16 << 20 && bytes.ends_with(b"\n") => {
                        if send.send(Event::Executor(generation, bytes)).is_err() {
                            return;
                        }
                    }
                    Ok(_) => break "incomplete or oversized executor reply".into(),
                    Err(e) => break e.to_string(),
                }
            };
            let _ = send.send(Event::Gone(generation, reason));
        });
        diagnostics(stderr, generation, events.clone());
        Ok(Self {
            child,
            input: Some(input),
            directory,
        })
    }

    pub fn send(&self, line: Vec<u8>) -> Result<()> {
        self.input
            .as_ref()
            .context("executor is closing")?
            .try_send(line)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
    pub fn eof(&mut self) {
        self.input.take();
    }
    pub fn validate_root(&self, root: &str) -> Result<PathBuf> {
        let root = PathBuf::from(root)
            .canonicalize()
            .context("executor session root is not visible")?;
        ensure!(
            root.is_dir() && root.parent() == Some(self.directory.path()),
            "executor session root is outside the shared session directory"
        );
        Ok(root)
    }
}

fn diagnostics(stderr: std::process::ChildStderr, generation: u64, send: SyncSender<Event>) {
    thread::spawn(move || {
        let mut reader = BufReader::new(stderr);
        loop {
            let mut line = Vec::new();
            if !matches!(reader.by_ref().take(65536).read_until(b'\n',&mut line),Ok(n) if n>0) {
                return;
            }
            if send
                .send(Event::Diagnostic(
                    generation,
                    String::from_utf8_lossy(&line).trim_end().into(),
                ))
                .is_err()
            {
                return;
            }
        }
    });
}

impl Drop for Executor {
    fn drop(&mut self) {
        self.eof();
        if let Some(group) = Pid::from_raw(self.child.id() as i32) {
            let _ = kill_process_group(group, Signal::TERM);
            // Slurm uses TERM on its launcher to withdraw a pending allocation.
            let until = std::time::Instant::now() + std::time::Duration::from_millis(500);
            while std::time::Instant::now() < until {
                if self.child.try_wait().ok().flatten().is_some() {
                    return;
                }
                thread::sleep(std::time::Duration::from_millis(10));
            }
            let _ = kill_process_group(group, Signal::KILL);
        }
        let _ = self.child.wait();
    }
}
