//! Slurm commands are always scoped to the identity recorded before submission.
use crate::{
    script::Claim,
    state::{Account, Launch, matches_allocation},
};
use std::{
    env,
    io::{self, Write},
    path::Path,
    process::{Command, Stdio},
};

pub fn output(cmd: &mut Command) -> io::Result<String> {
    let out = cmd.output()?;
    if !out.status.success() {
        return Err(io::Error::other(format!(
            "{:?}: {}",
            cmd.get_program(),
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    String::from_utf8(out.stdout).map_err(io::Error::other)
}

pub fn cluster() -> io::Result<String> {
    output(Command::new("scontrol").args(["show", "config"]))?
        .lines()
        .filter_map(|line| line.split_once('='))
        .find(|(key, _)| key.trim() == "ClusterName")
        .map(|(_, name)| name.trim().to_string())
        .ok_or_else(|| io::Error::other("scontrol did not report ClusterName"))
}

pub fn resources(options: &[String]) -> io::Result<Vec<String>> {
    let mut args = options.iter().map(String::as_str);
    let mut result = Vec::new();
    while let Some(arg) = args.next() {
        let (key, value) = arg
            .split_once('=')
            .map_or((arg, None), |(k, v)| (k, Some(v)));
        let opt = match key {
            "-p" | "--partition" => "partition",
            "-q" | "--qos" => "qos",
            "-A" | "--account" => "account",
            "--reservation" => "reservation",
            "-t" | "--time" => "time",
            "-c" | "--cpus-per-task" => "cpus-per-task",
            "--mem" => "mem",
            "--mem-per-cpu" => "mem-per-cpu",
            "--mem-per-gpu" => "mem-per-gpu",
            "-G" | "--gpus" => "gpus",
            "--gpus-per-node" => "gpus-per-node",
            "--gpus-per-task" => "gpus-per-task",
            "--gres" => "gres",
            "-C" | "--constraint" => "constraint",
            "-w" | "--nodelist" => "nodelist",
            "-x" | "--exclude" => "exclude",
            "--exclusive" | "--nice" => {
                result.push(arg.into());
                continue;
            }
            _ => {
                return Err(io::Error::other(format!(
                    "unsupported scheduler option {arg}; use partition, QoS, account, reservation, time, CPU, memory, GPU, constraint, node list, exclusive or nice"
                )));
            }
        };
        let value = value
            .or_else(|| args.next())
            .filter(|v| !v.is_empty() && !v.starts_with('-'))
            .ok_or_else(|| io::Error::other(format!("missing value for {key}")))?;
        result.push(format!("--{opt}={value}"));
    }
    Ok(result)
}

pub fn submit(claim: &Claim, dir: &Path, options: &[String], checkpoint: bool) -> io::Result<u64> {
    let mut command = Command::new("sbatch");
    // Connection options and our identity are the whole submission. Submit-host
    // defaults can otherwise silently make an array or move the job to a federation.
    for (name, _) in env::vars_os() {
        if name.as_encoded_bytes().starts_with(b"SBATCH_") || name == "SLURM_CLUSTERS" {
            command.env_remove(name);
        }
    }
    command
        .args(options)
        .args(["--parsable", "--ntasks=1", "--open-mode=append"])
        .arg(format!("--job-name={}", claim.name))
        .arg(format!("--output={}", dir.join("log.txt").display()));
    if checkpoint {
        command.args(["--signal=TERM@300", "--requeue"]);
    }
    let quote = |p: &Path| {
        shlex::try_quote(
            p.to_str()
                .ok_or_else(|| io::Error::other("non-UTF8 job path"))?,
        )
        .map(|s| s.into_owned())
        .map_err(io::Error::other)
    };
    let mut script = format!(
        "#!/bin/sh\nsrun --ntasks=1 {} job {}\nstatus=$?\n",
        quote(&claim.worker)?,
        quote(dir)?
    );
    if checkpoint {
        script.push_str(&format!("if [ \"$status\" -eq 130 ] && [ ! -e {} ]; then\n  scontrol requeue \"$SLURM_JOB_ID\" || exit 1\n  exit 0\nfi\n",quote(&dir.join("cancel"))?));
    }
    script.push_str("exit \"$status\"\n");
    let mut child = command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let wrote = child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(script.as_bytes());
    let out = child.wait_with_output()?;
    wrote?;
    if !out.status.success() {
        let reason = String::from_utf8_lossy(&out.stderr);
        // A transport timeout after acceptance is ambiguous. Only record failures
        // that the controller explicitly identified as invalid requests.
        let rejected = reason.lines().any(|line| {
            line.contains("Batch job submission failed: Invalid ")
                || line.contains(
                    "Batch job submission failed: Requested node configuration is not available",
                )
        });
        return Err(io::Error::new(
            if rejected {
                io::ErrorKind::InvalidInput
            } else {
                io::ErrorKind::Other
            },
            reason.trim().to_owned(),
        ));
    }
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .split(';')
        .next()
        .unwrap_or("")
        .parse()
        .map_err(io::Error::other)
}

#[derive(Debug)]
pub struct Report {
    pub account: Account,
    pub id: Option<u64>,
    pub detail: String,
}

fn scheduler(cmd: &str, claim: &Claim) -> Command {
    let mut command = Command::new(cmd);
    command
        .arg(format!(
            "--clusters={}",
            claim.cluster.as_deref().expect("Slurm claim has cluster")
        ))
        .arg(format!("--user={}", claim.user))
        .arg(format!("--name={}", claim.name));
    command
}

pub fn lookup(claim: &Claim, launch: Launch) -> io::Result<Report> {
    let identify = |id: &str, name: &str| {
        id.parse::<u64>()
            .ok()
            .filter(|id| matches_allocation(&claim.name, launch, name, *id))
    };
    // Accounting can lag or contain the previous attempt of a requeued allocation.
    // Query it before the controller, so the controller's live phase always wins.
    let mut accounting = scheduler("sacct", claim);
    if let Launch::Accepted(id) = launch {
        // An allocation canceled while held has no eligible time. Some sites
        // omit it from time-range queries unless the ID is also supplied.
        accounting.arg(format!("--jobs={id}"));
    }
    let history = output(accounting.args([
        "-X",
        "-n",
        "-P",
        &format!("--starttime={}", claim.submitted),
        "-o",
        "JobIDRaw,JobName%200,State",
    ]));
    let live =
        output(scheduler("squeue", claim).args(["-h", "--states=all", "-o", "%i|%j|%T|%R|%S|%N"]))?;
    for line in live.lines() {
        let mut fields = line.trim().split('|');
        let Some(id) = identify(fields.next().unwrap_or(""), fields.next().unwrap_or("")) else {
            continue;
        };
        let phase = fields
            .next()
            .ok_or_else(|| io::Error::other("squeue omitted job state"))?;
        let account = match phase {
            "PENDING" | "SPECIAL_EXIT" | "REQUEUED" | "REQUEUE_FED" | "REQUEUE_HOLD"
            | "RESV_DEL_HOLD" => Account::Pending,
            // The batch shell exits with the runner's status: 0 only once it published.
            "COMPLETED" => Account::Finished,
            "CANCELLED" | "FAILED" | "TIMEOUT" | "OUT_OF_MEMORY" | "PREEMPTED" | "NODE_FAIL"
            | "BOOT_FAIL" | "DEADLINE" | "REVOKED" => Account::Absent { accounted: true },
            _ => Account::Allocated,
        };
        return Ok(Report {
            account,
            id: Some(id),
            detail: format!("{phase} {}", fields.collect::<Vec<_>>().join(" ")),
        });
    }
    let history = history?;
    let detail = history
        .lines()
        .filter_map(|line| {
            let mut f = line.trim().split('|');
            identify(f.next()?, f.next()?)?;
            let phase = f.next()?.split_whitespace().next()?;
            matches!(
                phase,
                "COMPLETED"
                    | "CANCELLED"
                    | "FAILED"
                    | "TIMEOUT"
                    | "OUT_OF_MEMORY"
                    | "PREEMPTED"
                    | "NODE_FAIL"
                    | "BOOT_FAIL"
                    | "DEADLINE"
                    | "REVOKED"
            )
            .then_some(phase)
        })
        .next_back();
    Ok(Report {
        account: match detail {
            Some("COMPLETED") => Account::Finished,
            _ => Account::Absent {
                accounted: detail.is_some(),
            },
        },
        id: None,
        detail: detail.unwrap_or("").into(),
    })
}

pub fn withdraw(claim: &Claim, id: u64) -> io::Result<()> {
    output(
        scheduler("scancel", claim)
            .arg("--ctld")
            .arg(id.to_string()),
    )
    .map(drop)
}
