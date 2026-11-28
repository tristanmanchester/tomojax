use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    fs, io,
    path::{Path, PathBuf},
};

use crate::directory::{create_record, optional};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Launcher {
    Inline,
    Process,
    Slurm,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Claim {
    pub launcher: Launcher,
    pub name: String,
    pub host: String,
    pub user: String,
    pub cluster: Option<String>,
    pub submitted: String,
    pub python: PathBuf,
    pub worker: PathBuf,
}

pub fn read_claim(dir: &Path) -> io::Result<Claim> {
    let claim: Claim = serde_json::from_slice(&fs::read(dir.join("launch"))?)?;
    if claim.launcher == Launcher::Slurm && claim.cluster.as_deref().is_none_or(str::is_empty) {
        return Err(io::Error::other(
            "Slurm launch claim is missing its cluster",
        ));
    }
    Ok(claim)
}

pub fn job_id(dir: &Path) -> io::Result<&str> {
    dir.file_name()
        .and_then(|s| s.to_str())
        .filter(|s| {
            !s.is_empty()
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        })
        .ok_or_else(|| {
            io::Error::other(
                "job directory needs an alphanumeric ID (hyphens and underscores allowed)",
            )
        })
}

pub fn host() -> String {
    rustix::system::uname()
        .nodename()
        .to_string_lossy()
        .into_owned()
}

pub fn read_script(dir: &Path) -> io::Result<Vec<Value>> {
    fs::read_to_string(dir.join("script.jsonl"))?
        .lines()
        .map(|line| Ok(serde_json::from_str(line)?))
        .collect()
}

fn validate_inputs(value: &Value, previous: &HashSet<&str>) -> io::Result<()> {
    match value {
        Value::Object(_) if value["type"] == "member" => {
            if !value["handle"]
                .as_str()
                .is_some_and(|id| previous.contains(id))
            {
                return Err(io::Error::other(
                    "a script member must belong to an earlier line",
                ));
            }
            Ok(())
        }
        Value::Object(_) if value["type"] == "npy" && !value["artifact"].is_null() => {
            Err(io::Error::other("a job cannot borrow a session artifact"))
        }
        Value::Object(map) => map.values().try_for_each(|v| validate_inputs(v, previous)),
        Value::Array(items) => items.iter().try_for_each(|v| validate_inputs(v, previous)),
        _ => Ok(()),
    }
}

/// Persist the client's exact script before competing for the launch claim. Linux directory
/// rename refuses an existing, nonempty job directory even across submit hosts.
pub fn publish_script(dir: &Path, script: &[Value]) -> io::Result<LauncherChoice> {
    let id = job_id(dir)?;
    let parent = dir
        .parent()
        .ok_or_else(|| io::Error::other("job directory needs a parent"))?;
    fs::create_dir_all(parent)?;
    let mut ids = HashSet::new();
    let mut runs = false;
    let mut bytes = Vec::new();
    for line in script {
        let request = line["id"]
            .as_str()
            .filter(|s| {
                s.strip_prefix(id)
                    .is_some_and(|suffix| suffix.starts_with('.') && suffix.len() > 1)
            })
            .ok_or_else(|| {
                io::Error::other("script request ID must start with the job ID and a dot")
            })?;
        validate_inputs(&line["command"], &ids)?;
        if !ids.insert(request) {
            return Err(io::Error::other("duplicate script request ID"));
        }
        match line["command"]["type"].as_str() {
            Some("run") => runs = true,
            Some("open" | "inspect" | "plan") => {}
            _ => {
                return Err(io::Error::other(
                    "a script line must open, inspect, plan or run",
                ));
            }
        }
        serde_json::to_writer(&mut bytes, line)?;
        bytes.push(b'\n');
    }
    if script.is_empty() {
        return Err(io::Error::other("job script is empty"));
    }
    let stage = tempfile::Builder::new()
        .prefix(".submit-")
        .tempdir_in(parent)?;
    fs::write(stage.path().join("script.jsonl"), &bytes)?;
    if !dir.try_exists()? {
        match fs::rename(stage.path(), dir) {
            Ok(()) => {}
            Err(e)
                if matches!(
                    e.kind(),
                    io::ErrorKind::DirectoryNotEmpty | io::ErrorKind::AlreadyExists
                ) => {}
            Err(e) => return Err(e),
        }
    }
    if read_script(dir)? != script {
        return Err(io::Error::other(
            "job directory contains a different script",
        ));
    }
    Ok(if runs {
        LauncherChoice::Configured
    } else {
        LauncherChoice::Inline
    })
}

pub enum LauncherChoice {
    Inline,
    Configured,
}

pub fn prepare(dir: &Path, launcher: Launcher, lines: &mut [Value]) -> io::Result<Option<PathBuf>> {
    let mut checkpoint = None;
    for line in lines {
        if let Some(channel) = line["channel"].as_str() {
            line["channel"] = json!(format!("{}.{}", job_id(dir)?, channel));
        }
        let Some(operation) = line.pointer_mut("/command/operation") else {
            continue;
        };
        let slab = operation["type"] == "reconstruct_slabs";
        if slab {
            operation["path"] = json!(dir.join("work/result/result.nxs"));
        }
        if matches!(
            operation["type"].as_str(),
            Some("save" | "export" | "reconstruct_slabs")
        ) {
            operation["force"] = json!(false);
        }
        if matches!(
            operation["type"].as_str(),
            Some("reconstruct" | "align" | "reconstruct_slabs")
        ) {
            if operation["checkpoint"].is_null() && launcher != Launcher::Inline {
                operation["checkpoint"] = json!(dir.join("checkpoint"));
            }
            if let Some(path) = operation["checkpoint"].as_str() {
                checkpoint = Some(std::path::absolute(path)?);
            }
        }
    }
    Ok(checkpoint)
}

pub fn checkpoint_status(dir: &Path, launcher: Launcher) -> io::Result<Value> {
    let Some(path) = prepare(dir, launcher, &mut read_script(dir)?)? else {
        return Ok(Value::Null);
    };
    let usable = match optional(fs::read(path.join("checkpoint.json")))? {
        Some(bytes) => serde_json::from_slice::<Value>(&bytes)?["entries"]
            .as_object()
            .is_some_and(|entries| !entries.is_empty()),
        None => false,
    };
    Ok(json!({"path":path,"usable":usable}))
}

pub fn mark(dir: &Path, name: &str) -> io::Result<()> {
    create_record(&dir.join(name), &true).map(|_| ())
}
