//! Filesystem operations on the machine that owns the acquisition.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{ErrorKind, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result, bail, ensure};
use rustix::fs::{Access, CWD, RenameFlags, access, renameat_with};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::metadata;

pub fn resolve(path: &str) -> Result<PathBuf> {
    let path = Path::new(path);
    if let Ok(tail) = path.strip_prefix("~") {
        Ok(
            PathBuf::from(std::env::var_os("HOME").context("home directory unavailable")?)
                .join(tail),
        )
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn signature(path: &Path) -> bool {
    let mut bytes = [0; 8];
    File::open(path)
        .and_then(|mut file| file.read_exact(&mut bytes))
        .is_ok()
        && bytes == *b"\x89HDF\r\n\x1a\n"
}

pub fn list(path: &Path, after: Option<&str>) -> Result<Value> {
    let path = path.canonicalize()?;
    let mut names = BTreeSet::new();
    if path.is_dir() {
        for entry in fs::read_dir(&path)? {
            let name = entry?
                .file_name()
                .into_string()
                .map_err(|_| anyhow::anyhow!("filename is not UTF-8"))?;
            if after.is_none_or(|cursor| name.as_str() > cursor) {
                names.insert(name);
                if names.len() > 4097 {
                    names.pop_last();
                }
            }
        }
    } else if let Some(name) = path.file_name().and_then(|n| n.to_str())
        && after.is_none_or(|cursor| name > cursor)
    {
        names.insert(name.to_owned());
    }
    let next = if names.len() > 4096 {
        names.pop_last();
        names.last().cloned()
    } else {
        None
    };
    let mut entries = Vec::with_capacity(names.len());
    for name in names {
        let child = if path.is_dir() {
            path.join(&name)
        } else {
            path.clone()
        };
        let meta = child.symlink_metadata()?;
        let kind = if meta.is_dir() {
            "directory"
        } else if meta.is_file() {
            "file"
        } else if meta.is_symlink() {
            "symlink"
        } else {
            "other"
        };
        entries.push(json!({"name":name,"kind":kind,"size":meta.len(),"modified": meta.modified()?.duration_since(UNIX_EPOCH)?.as_secs_f64(), "hdf5":meta.is_file() && signature(&child)}));
    }
    Ok(
        json!({"type":"list","path":path,"writable":access(&path, Access::WRITE_OK).is_ok(),"entries":entries,"after":next}),
    )
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Source {
    pub path: String,
    pub name: String,
    pub artifact: Option<String>,
}

#[derive(Deserialize)]
pub struct Copy {
    pub sources: Vec<Source>,
    #[serde(default)]
    pub documents: BTreeMap<String, Value>,
    pub destination: String,
}

fn relative(name: &str) -> Result<&Path> {
    let path = Path::new(name);
    ensure!(
        !name.is_empty() && path.components().all(|p| matches!(p, Component::Normal(_))),
        "invalid relative destination: {name}"
    );
    Ok(path)
}

pub fn copy(request: &Copy, canceled: &AtomicBool, mut progress: impl FnMut(u64)) -> Result<Value> {
    let dest = resolve(&request.destination)?;
    let parent = dest
        .parent()
        .context("destination has no parent")?
        .canonicalize()?;
    let destination = parent.join(dest.file_name().context("missing destination name")?);
    ensure!(!destination.try_exists()?, "destination already exists");
    for source in &request.sources {
        ensure!(
            !destination.starts_with(resolve(&source.path)?.canonicalize()?),
            "choose a destination outside the source"
        );
    }
    let stage = tempfile::Builder::new()
        .prefix(".tomojax-copy-")
        .tempdir_in(&parent)?;
    let mut bytes = 0;
    let check = || {
        ensure!(!canceled.load(Ordering::Acquire), "canceled");
        Ok(())
    };
    for source in &request.sources {
        let mut pending = vec![(
            resolve(&source.path)?,
            stage.path().join(relative(&source.name)?),
        )];
        while let Some((from, to)) = pending.pop() {
            check()?;
            fs::create_dir_all(to.parent().context("missing parent")?)?;
            let meta = from.symlink_metadata()?;
            if meta.is_dir() {
                fs::create_dir(&to)?;
                for entry in fs::read_dir(from)? {
                    let entry = entry?;
                    pending.push((entry.path(), to.join(entry.file_name())));
                }
            } else if meta.is_file() {
                if signature(&from)
                    && metadata::reads_other_files(&from).map_err(anyhow::Error::msg)?
                {
                    bail!(
                        "{} reads data from other files, so it can't be copied. Import it without copying.",
                        from.display()
                    );
                }
                let mut input = File::open(from)?;
                let mut output = File::create_new(to)?;
                let mut buffer = vec![0; 1 << 20];
                loop {
                    check()?;
                    let n = input.read(&mut buffer)?;
                    if n == 0 {
                        break;
                    }
                    output.write_all(&buffer[..n])?;
                    bytes += n as u64;
                    progress(bytes);
                }
                output.sync_all()?;
            } else {
                bail!("only regular files and acquisition directories can be copied");
            }
        }
    }
    for (name, document) in &request.documents {
        check()?;
        let to = stage.path().join(relative(name)?);
        fs::create_dir_all(to.parent().context("missing parent")?)?;
        let file = File::create_new(to)?;
        serde_json::to_writer(&file, document)?;
        file.sync_all()?;
    }
    check()?;
    renameat_with(CWD, stage.path(), CWD, &destination, RenameFlags::NOREPLACE)?;
    Ok(json!({"type":"copy","path":destination,"bytes":bytes}))
}

pub fn stored(path: &Path) -> Result<Value> {
    let metadata = metadata::quick_metadata(path).map_err(anyhow::Error::msg)?;
    Ok(match metadata {
        Some((kind, arrays, report)) => {
            json!({"type":"stored","kind":kind,"arrays":arrays,"report":report})
        }
        None => json!({"type":"stored","kind":null,"arrays":{},"report":null}),
    })
}

pub fn peek(path: &Path) -> Result<Value> {
    let mut datasets = BTreeMap::new();
    if path.is_dir() {
        let mut frames = 0;
        let mut candidates = 0;
        let mut directories = 0;
        let mut subdirectories = Vec::new();
        let mut checkpoints = Vec::new();
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let child = entry.path();
            if child.is_dir() {
                directories += 1;
                if subdirectories.len() < 64 {
                    subdirectories.push(entry.file_name());
                }
                if child.join("checkpoint.json").is_file() {
                    checkpoints.push(child);
                }
                continue;
            }
            let kind = match child
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase()
                .as_str()
            {
                "tif" | "tiff" => {
                    frames += 1;
                    continue;
                }
                "xtekct" => "Nikon",
                "txrm" | "txm" => "Zeiss",
                "log" => "acquisition log",
                "xml" | "info" => "acquisition metadata",
                "npz" => "NumPy archive",
                "mat" => "PSI MATLAB acquisition",
                "nxs" | "nx" | "hdf" | "hdf5" | "h5" => "HDF5",
                _ => continue,
            };
            candidates += 1;
            if datasets.len() < 64 {
                datasets.insert(child.display().to_string(), json!(kind));
            }
        }
        subdirectories.sort();
        checkpoints.sort();
        let names: Vec<_> = subdirectories.iter().map(|s| s.to_string_lossy()).collect();
        return Ok(
            json!({"type":"peek","directory":path,"datasets":datasets,"candidate_count":candidates,"subdirectories":names,"subdirectory_count":directories,"tiff_files":frames,"checkpoints":checkpoints,"omitted":candidates-datasets.len()}),
        );
    }
    if signature(path) {
        let file = hdf5::File::open(path)?;
        let mut groups = vec![file.group("/")?];
        let mut visited = Vec::new();
        while let Some(group) = groups.pop() {
            let token = group.loc_info()?.token;
            if visited.contains(&token) {
                continue;
            }
            visited.push(token);
            for data in group.datasets()? {
                datasets.insert(data.name(), json!({"shape":data.shape(),"dtype":data.dtype()?.to_descriptor()?.to_string()}));
            }
            groups.extend(group.groups()?);
        }
    } else if path.extension().is_some_and(|s| s == "npz") {
        let mut archive = npyz::npz::NpzArchive::open(path)?;
        for name in archive.array_names().map(str::to_owned).collect::<Vec<_>>() {
            let array = archive.by_name(&name)?.context("missing NPZ member")?;
            datasets.insert(
                name,
                json!({"shape":array.shape(),"dtype":array.dtype().descr()}),
            );
        }
    } else {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "expected directory, HDF5 file or NPZ archive",
        )
        .into());
    }
    Ok(json!({"type":"peek","datasets":datasets}))
}
