// Ported from T3 Code (MIT), packages/ssh/src/command.ts and tunnel.ts at d2c44031. See THIRD_PARTY_NOTICES.md.
use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use super::process::{Owned, login_lock, runtime_dir};
use super::{Failure, Profile, Status, auth};

pub(super) const STARTUP: Duration = Duration::from_secs(20);
const RECHECK: Duration = Duration::from_secs(10);
const LOGIN: Duration = Duration::from_secs(120);

pub(super) fn quote(arg: &str) -> String {
    format!("'{}'", arg.replace('\'', "'\\''"))
}
fn shell_command(args: &[String]) -> String {
    args.iter()
        .map(|arg| quote(arg))
        .collect::<Vec<_>>()
        .join(" ")
}

struct Output {
    success: bool,
    out: String,
    err: String,
}
fn probe(command: &mut Command, timeout: Duration) -> Result<Output, Failure> {
    run_probe(command, timeout, false)
}

fn run_probe(
    command: &mut Command,
    timeout: Duration,
    interactive: bool,
) -> Result<Output, Failure> {
    let mut out = tempfile::tempfile()?;
    let mut err = tempfile::tempfile()?;
    command
        .stdin(Stdio::null())
        .stdout(out.try_clone()?)
        .stderr(err.try_clone()?);
    let terminal;
    let mut child = Owned::spawn(command)?;
    terminal = if interactive {
        super::terminal::Terminal::lend(&child.0)?
    } else {
        None
    };
    let status = child.wait(Some(Instant::now() + timeout))?;
    drop(terminal);
    let read = |file: &mut std::fs::File| -> Result<String, Failure> {
        let offset = file.metadata()?.len().saturating_sub(65536);
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = Vec::new();
        file.take(65536).read_to_end(&mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes).trim().to_owned())
    };
    Ok(Output {
        success: status.success(),
        out: read(&mut out)?,
        err: read(&mut err)?,
    })
}

pub(super) struct Route<'a> {
    hops: &'a [String],
    options: Vec<String>,
    dir: PathBuf,
    key: String,
}
impl<'a> Route<'a> {
    pub fn resolve(profile: &'a Profile) -> Result<Option<Self>, Failure> {
        let hops = profile.hops();
        let Some(first) = hops.first() else {
            return Ok(None);
        };
        let output = probe(Command::new("ssh").args(["-G", "--", first]), RECHECK)?;
        if !output.success {
            return Err(Failure::blocked(output.err));
        }
        let mut values = BTreeMap::new();
        for (key, value) in output
            .out
            .lines()
            .filter_map(|l| l.trim().split_once(char::is_whitespace))
        {
            values.entry(key).or_insert(value.trim());
        }
        let dir = runtime_dir()?;
        let socket = values.get("controlpath").copied().filter(|p| *p != "none");
        let identity = ["hostname", "user", "port", "proxyjump", "proxycommand"]
            .map(|key| values.get(key).copied().unwrap_or(""));
        let fallback = format!("{first}\0{}", identity.join("\0"));
        let key = blake3::hash(socket.unwrap_or(&fallback).as_bytes()).to_hex()[..16].to_owned();
        let mut options = Vec::new();
        if socket.is_none() {
            options.extend(["-S".into(), dir.join(&key).to_string_lossy().into_owned()]);
        }
        if values
            .get("controlmaster")
            .is_none_or(|v| matches!(*v, "no" | "false"))
        {
            options.extend(["-o".into(), "ControlMaster=auto".into()]);
        }
        if values
            .get("controlpersist")
            .is_none_or(|v| matches!(*v, "no" | "0"))
        {
            options.extend(["-o".into(), "ControlPersist=8h".into()]);
        }
        Ok(Some(Self {
            hops,
            options,
            dir,
            key,
        }))
    }

    fn args(batch: bool) -> Vec<String> {
        [
            "-T",
            "-o",
            if batch {
                "BatchMode=yes"
            } else {
                "BatchMode=no"
            },
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
        ]
        .map(String::from)
        .into()
    }

    fn command(&self, depth: usize, mut remote: String, batch: bool) -> Command {
        for hop in self.hops[1..depth].iter().rev() {
            let mut args = vec!["ssh".into()];
            args.extend(Self::args(true));
            args.extend(["--".into(), hop.clone(), remote]);
            remote = format!("LC_ALL=C exec {}", shell_command(&args));
        }
        let mut cmd = Command::new("ssh");
        cmd.args(Self::args(batch))
            .args(&self.options)
            .args(["--", &self.hops[0], &remote])
            .env("LC_ALL", "C");
        if batch {
            cmd.env("SSH_ASKPASS_REQUIRE", "never");
        }
        cmd
    }

    pub fn worker(&self, profile: &Profile, program: &str) -> Command {
        let mut args = vec![program.to_string()];
        args.extend(profile.worker_args());
        let exec = format!("LC_ALL=C exec {}", shell_command(&args));
        self.command(self.hops.len(), exec, true)
    }

    pub fn classify(&self, error: Failure, depth: usize) -> Failure {
        if !matches!(error.status, Status::Transient | Status::LoginNeeded) {
            return error;
        }
        let recheck = probe(&mut self.command(1, "true".into(), true), RECHECK);
        if recheck
            .as_ref()
            .is_err_and(|e| e.status == Status::Interrupted)
        {
            return Failure::interrupted();
        }
        let answered = recheck.as_ref().is_ok_and(|o| o.success);
        let status = if error.status == Status::Transient {
            Status::Transient
        } else if answered && depth > 1 {
            Status::InnerRefusal
        } else if answered || recheck.as_ref().is_ok_and(|o| auth::refused(&o.err)) {
            Status::LoginNeeded
        } else {
            Status::Transient
        };
        Failure::new(
            status,
            format!(
                "{}. The first hop {} when checked again",
                error.message,
                if answered {
                    "answered"
                } else {
                    "did not answer"
                }
            ),
        )
    }

    pub fn login(&self, helper: Option<&Path>) -> Result<(), Failure> {
        let _lock = login_lock(&self.dir, &self.key, Instant::now() + LOGIN)?;
        let mut first = self.command(1, "true".into(), false);
        auth::askpass(&mut first, helper);
        for depth in 1..=self.hops.len() {
            let result = if depth == 1 {
                run_probe(
                    &mut first,
                    LOGIN,
                    helper.is_none()
                        && std::env::var("SSH_ASKPASS_REQUIRE").as_deref() != Ok("force"),
                )
            } else {
                probe(&mut self.command(depth, "true".into(), true), STARTUP)
            };
            result
                .and_then(|o| {
                    if o.success {
                        Ok(())
                    } else {
                        Err(Failure::ssh(&o.err))
                    }
                })
                .map_err(|e| self.classify(e, depth))?;
            if depth == 1 {
                let checked = probe(
                    Command::new("ssh").args(&self.options).args([
                        "-O",
                        "check",
                        "--",
                        &self.hops[0],
                    ]),
                    RECHECK,
                )?;
                if !checked.success {
                    return Err(Failure::blocked(format!(
                        "SSH login succeeded but no persistent control master is available: {}",
                        checked.err
                    )));
                }
            }
            eprintln!("{} answered", self.hops[depth - 1]);
        }
        Ok(())
    }
}
