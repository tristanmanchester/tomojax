//! The four worker job commands. Each poll can complete a standing cancel.
use crate::{
    directory::{Submission, create_record, optional, read_files},
    locks::held,
    log, publication,
    runner::{self, InlineSession},
    script::{
        Claim, Launcher, LauncherChoice, checkpoint_status, host, job_id, mark, prepare,
        publish_script, read_claim,
    },
    slurm::{self, Report},
    state::{Account, Launch, State, needs_withdrawal, observed},
};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    io,
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
};

pub type Stored = dyn Fn(&Path) -> io::Result<Value> + Send + Sync;

pub struct Service {
    launcher: Launcher,
    python: PathBuf,
    worker: PathBuf,
    resources: Vec<String>,
    inline: Arc<dyn InlineSession>,
    stored: Arc<Stored>,
    stop: Arc<AtomicBool>,
    tasks: Mutex<Vec<thread::JoinHandle<()>>>,
}

impl Service {
    pub fn new(
        launcher: Launcher,
        python: PathBuf,
        worker: PathBuf,
        options: &[String],
        inline: Arc<dyn InlineSession>,
        stored: Arc<Stored>,
    ) -> io::Result<Self> {
        if launcher == Launcher::Process && std::env::var_os("SLURM_JOB_ID").is_some() {
            return Err(io::Error::other(
                "use slurm jobs inside a Slurm allocation; detached processes end with the allocation",
            ));
        }
        Ok(Self {
            launcher,
            python,
            worker,
            resources: slurm::resources(options)?,
            inline,
            stored,
            stop: Arc::default(),
            tasks: Mutex::default(),
        })
    }

    pub fn submit(&self, dir: &Path, script: &[Value], options: &[String]) -> io::Result<Value> {
        let dir = std::path::absolute(dir)?;
        let mut resources = self.resources.clone();
        resources.extend(slurm::resources(options)?);
        let launcher = match publish_script(&dir, script)? {
            LauncherChoice::Configured => self.launcher,
            LauncherChoice::Inline => Launcher::Inline,
        };
        if dir.join("launch").try_exists()? {
            return status(&dir);
        }
        let cluster = (launcher == Launcher::Slurm)
            .then(slurm::cluster)
            .transpose()?;
        let claim = Claim {
            launcher,
            // Scripts are client-named. Two projects may legitimately submit "run-1".
            // Keep lookup unambiguous even before the scheduler returns an ID.
            name: format!(
                "tj-{:.64}-{:016x}",
                job_id(&dir)?,
                getrandom::u64().map_err(io::Error::other)?
            ),
            host: host(),
            user: slurm::output(Command::new("id").arg("-un"))?.trim().into(),
            cluster,
            python: self.python.clone(),
            worker: self.worker.clone(),
            submitted: slurm::output(Command::new("date").arg("+%Y-%m-%dT%H:%M:%S"))?
                .trim()
                .into(),
        };
        let checkpoint = prepare(&dir, launcher, &mut script.to_vec())?.is_some();
        if create_record(&dir.join("launch"), &claim)? {
            let response = if dir.join("cancel").try_exists()? {
                Some(Submission::Prevented)
            } else {
                match launcher {
                    Launcher::Inline => Some(match self.inline(&dir) {
                        Ok(()) => Submission::Accepted(std::process::id().into()),
                        Err(e) => Submission::Rejected(e.to_string()),
                    }),
                    Launcher::Process => Some(match detach(&claim.worker, &dir) {
                        Ok(id) => Submission::Accepted(id.into()),
                        Err(e) => Submission::Rejected(e.to_string()),
                    }),
                    Launcher::Slurm => match slurm::submit(&claim, &dir, &resources, checkpoint) {
                        Ok(id) => Some(Submission::Accepted(id)),
                        Err(e)
                            if e.kind() == io::ErrorKind::InvalidInput
                                || e.kind() == io::ErrorKind::NotFound =>
                        {
                            Some(Submission::Rejected(e.to_string()))
                        }
                        Err(e) => {
                            create_record(&dir.join("submission-error.json"), &e.to_string())?;
                            None
                        }
                    },
                }
            };
            if let Some(response) = response {
                create_record(&dir.join("submission.json"), &response)?;
            }
        }
        status(&dir)
    }

    fn inline(&self, dir: &Path) -> io::Result<()> {
        let session = self.inline.clone();
        let stored = self.stored.clone();
        let stop = self.stop.clone();
        let dir = dir.to_owned();
        let task = thread::Builder::new()
            .name("inline job".into())
            .spawn(move || {
                if let Err(e) = runner::run(&dir, &stop, Some(&*session), &*stored) {
                    eprintln!("inline job {}: {e}", dir.display());
                }
            })?;
        let mut tasks = self.tasks.lock().unwrap();
        tasks.retain(|task| !task.is_finished());
        tasks.push(task);
        Ok(())
    }

    pub fn stop_inline(&self) {
        self.stop.store(true, Ordering::Release);
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        self.stop_inline();
        for task in self.tasks.get_mut().unwrap().drain(..) {
            let _ = task.join();
        }
    }
}

fn detach(worker: &Path, dir: &Path) -> io::Result<u32> {
    let log = File::options()
        .append(true)
        .create(true)
        .open(dir.join("log.txt"))?;
    let mut cmd = Command::new(worker);
    cmd.arg("job")
        .arg(dir)
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    // SAFETY: setsid is a single syscall. The child leaves the worker's
    // process group and has no controlling terminal or inherited SSH pipes.
    unsafe {
        cmd.pre_exec(|| {
            rustix::process::setsid()?;
            Ok(())
        });
    }
    let mut child = cmd.spawn()?;
    let id = child.id();
    thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(id)
}

fn account(dir: &Path, claim: &Claim, launch: Launch) -> io::Result<Report> {
    if claim.launcher == Launcher::Slurm {
        return slurm::lookup(claim, launch);
    }
    if claim.host != host() {
        return Err(io::Error::other(format!(
            "job lock belongs to {}",
            claim.host
        )));
    }
    let held = held(&dir.join("lock"))?;
    Ok(Report {
        account: if held {
            Account::LocalLockHeld
        } else {
            Account::Absent { accounted: false }
        },
        id: None,
        detail: claim.host.clone(),
    })
}

pub fn status(dir: &Path) -> io::Result<Value> {
    let mut withdrew = false;
    let mut warning = None;
    loop {
        let before = read_files(dir)?;
        let claim = optional(read_claim(dir))?;
        let report = match &claim {
            Some(_)
                if (before.receipt && !before.cancel)
                    || matches!(before.launch, Launch::Prevented | Launch::Rejected) =>
            {
                Report {
                    account: Account::Absent { accounted: false },
                    id: None,
                    detail: String::new(),
                }
            }
            Some(c) => account(dir, c, before.launch).unwrap_or_else(|e| Report {
                account: Account::Offline,
                id: None,
                detail: e.to_string(),
            }),
            None => Report {
                account: Account::Absent { accounted: false },
                id: None,
                detail: String::new(),
            },
        };
        let after = read_files(dir)?;
        let Some(state) = observed(before, report.account, after) else {
            continue;
        };
        if after.launch == Launch::Claimed
            && let Some(id) = report.id
        {
            create_record(&dir.join("submission.json"), &Submission::Accepted(id))?;
            continue;
        }
        if !withdrew
            && needs_withdrawal(after, report.account)
            && let (Some(claim), Some(id)) = (&claim, report.id)
        {
            // The controller rechecks name, owner, cluster and ID at delivery.
            match slurm::withdraw(claim, id) {
                Ok(()) => mark(dir, "withdrawn")?,
                Err(e) => warning = Some(e.to_string()),
            }
            withdrew = true;
            continue;
        }
        let receipt = if state == State::Completed {
            Some(read_json(&dir.join("result/receipt.json"))?)
        } else {
            None
        };
        let checkpoint = claim
            .as_ref()
            .map(|c| checkpoint_status(dir, c.launcher))
            .transpose()?;
        let detail = if after.launch == Launch::Rejected {
            read_json(&dir.join("submission.json"))?["rejected"]
                .as_str()
                .unwrap_or("launch rejected")
                .to_string()
        } else if state == State::Unknown
            && let Some(err) = optional(fs::read(dir.join("submission-error.json")))?
        {
            serde_json::from_slice(&err)?
        } else {
            warning.unwrap_or(report.detail)
        };
        return Ok(
            json!({"directory":dir,"state":state,"cancel_requested":after.cancel,"receipt":receipt,"checkpoint":checkpoint,
            "launcher":claim.as_ref().map(|c|c.launcher),"job_name":claim.as_ref().map(|c|&c.name),"detail":detail}),
        );
    }
}

fn read_json(path: &Path) -> io::Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

pub fn jobs(root: &Path) -> io::Result<Vec<Value>> {
    let mut rows = Vec::new();
    if let Some(entries) = optional(fs::read_dir(root.join("jobs")))? {
        for entry in entries {
            let dir = entry?.path();
            if job_id(&dir).is_ok() && dir.join("script.jsonl").is_file() {
                rows.push(status(&dir)?);
            }
        }
    }
    rows.sort_by(|a, b| a["directory"].as_str().cmp(&b["directory"].as_str()));
    Ok(rows)
}

pub fn job(dir: &Path, offset: u64, limit: u64) -> io::Result<Value> {
    let mut row = status(dir)?;
    let (records, next) = log::page(dir, offset, limit)?;
    row["records"] = json!(records);
    row["offset"] = json!(next);
    row["preview"] = json!(publication::current_preview(dir)?);
    Ok(row)
}

pub fn control(dir: &Path, action: &str) -> io::Result<Value> {
    if !dir.join("script.jsonl").is_file() {
        return Err(io::Error::other("job does not exist"));
    }
    match action {
        "cancel" => mark(dir, "cancel")?,
        "abandon" => {
            if status(dir)?["state"] != "unknown" {
                return Err(io::Error::other("only an unknown job can be abandoned"));
            }
            mark(dir, "cancel")?;
            mark(dir, "abandoned")?;
        }
        "pause" | "resume" if read_claim(dir)?.launcher == Launcher::Inline => {
            if action == "pause" {
                mark(dir, "paused")?;
            } else {
                optional(fs::remove_file(dir.join("paused")))?;
            }
        }
        _ => {
            return Err(io::Error::other(
                "use cancel, abandon, or pause/resume for an inline job",
            ));
        }
    }
    status(dir)
}
