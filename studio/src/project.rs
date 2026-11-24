//! Durable project state. Worker handles are deliberately absent from this file.
//!
//! A submission captures its input paths and parameters once. Session handles are
//! resolved by the scheduler; changing a form or relinking an asset cannot silently
//! change an already submitted job.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocol::{ArrayDescriptor, Provenance};

/// The project's own name, from its manifest, without opening the project.
pub fn manifest_name(root: &Path) -> Option<String> {
    let text = std::fs::read_to_string(root.join(MANIFEST)).ok()?;
    let value: serde_json::Value = serde_json::from_str(&text).ok()?;
    value
        .get("name")?
        .as_str()
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
}

pub const MANIFEST: &str = "project.tomojax.json";
const MAX_MANIFEST_BYTES: u64 = 32 * 1024 * 1024;

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub fn new_id(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    format!(
        "{prefix}-{:x}-{:x}",
        now_ms(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    )
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Asset {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub path: PathBuf,
    pub managed: bool,
    pub parent: Option<String>,
    pub created: u64,
    pub simulated: bool,
    pub method: Option<String>,
    #[serde(default)]
    pub arrays: BTreeMap<String, ArrayDescriptor>,
    #[serde(default)]
    pub inspection: Value,
    pub provenance: Option<Provenance>,
    #[serde(default)]
    pub issue: Option<String>,
    #[serde(default)]
    pub open_parameters: Value,
}

impl Asset {
    pub fn input(path: PathBuf, managed: bool) -> Self {
        Self {
            id: new_id("scan"),
            name: path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into(),
            kind: "scan".into(),
            path,
            managed,
            parent: None,
            created: now_ms(),
            simulated: false,
            method: None,
            arrays: BTreeMap::new(),
            inspection: Value::Null,
            provenance: None,
            issue: None,
            open_parameters: Value::Null,
        }
    }
    /// Records the kind and arrays a reader found in this asset's file (the
    /// header reader, or the worker's inspection). A worker result keeps the kind
    /// and arrays the worker registered, which know its units and domain (the
    /// file alone can't tell a crop of a reconstruction's scan, or a scan with
    /// poses, from its source); the read only fills in arrays it lacks. Paths
    /// inside the project are stored relative to it.
    pub fn register_read(
        &mut self,
        kind: &str,
        arrays: BTreeMap<String, ArrayDescriptor>,
        root: &Path,
    ) {
        if !self.is_result() {
            self.kind = kind.into();
            self.arrays.clear();
        }
        for (key, mut array) in arrays {
            array.relative_to(root);
            self.arrays.entry(key).or_insert(array);
        }
    }

    /// Points this asset at `path` (stored as the project stores paths). An
    /// import's registration is dropped, to be read again from the new file; a
    /// worker result keeps its registration (no reader can recover it from an
    /// array file), with storage that named the old file now naming the new one.
    pub fn relink(&mut self, path: PathBuf, root: &Path) {
        let resolve = |path: &Path| {
            if path.is_absolute() {
                path.to_owned()
            } else {
                root.join(path)
            }
        };
        let old = resolve(&self.path);
        let new = path.to_string_lossy().into_owned();
        self.path = path;
        self.managed = false;
        self.issue = None;
        if !self.is_result() {
            self.arrays.clear();
            return;
        }
        let repoint = |storage: &mut crate::protocol::Storage| {
            if let crate::protocol::Storage::Hdf5 { path, .. }
            | crate::protocol::Storage::Npy { path, .. } = storage
                && resolve(Path::new(path.as_str())) == old
            {
                *path = new.clone();
            }
        };
        for array in self.arrays.values_mut() {
            repoint(&mut array.storage);
            for axis in &mut array.axes {
                if let crate::protocol::Coordinates::Coordinates { values } = &mut axis.coordinates
                {
                    repoint(values);
                }
            }
        }
    }

    pub fn is_result(&self) -> bool {
        self.parent.is_some()
            || self
                .provenance
                .as_ref()
                .is_some_and(|record| record.operation != "open")
    }

    pub fn volume(&self) -> Option<&ArrayDescriptor> {
        ["volume", "phantom", "image"]
            .iter()
            .find_map(|key| self.arrays.get(*key))
    }
    pub fn projections(&self) -> Option<&ArrayDescriptor> {
        self.arrays
            .get("projections")
            .or_else(|| self.arrays.get("counts"))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Fingerprint {
    pub bytes: u64,
    pub modified_ns: Option<u128>,
    pub directory: bool,
    /// Metadata change detection, not a digest of potentially terabytes of pixels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tree_digest: Option<String>,
}
const LOCK: &str = ".tomojax.lock";

/// Whether a folder also holds a Studio project (its lock exists from the moment
/// the project is opened, before the first manifest is written).
fn holds_project(folder: &Path) -> bool {
    folder.join(MANIFEST).exists() || folder.join(LOCK).exists()
}

/// Whether `relative`, inside a folder that also holds a project, belongs to that
/// project (its manifest and the files it writes) rather than to the acquisition.
fn is_project_entry(relative: &Path) -> bool {
    let mut components = relative.components();
    let Some(first) = components.next() else {
        return false;
    };
    let first = first.as_os_str().to_string_lossy();
    let nested = components.next().is_some();
    (first.starts_with(MANIFEST) || first == LOCK) && !nested
        || matches!(first.as_ref(), "inputs" | "derived" | "cache")
}

impl Fingerprint {
    /// The other files in `file`'s folder and its subfolders, which a header
    /// file's loader reads (Nikon can keep projections in a subfolder). A project
    /// kept inside the folder is skipped, since it rewrites its own files.
    pub fn read_siblings(file: &Path) -> Result<Self, String> {
        let folder = file.parent().ok_or("The input has no containing folder")?;
        let mut hash = blake3::Hasher::new();
        hash.update(b"tomojax-sibling-metadata-v2\0");
        let mut bytes = 0u64;
        // A project kept in the acquisition's own folder writes its manifest and
        // its inputs, results and cache there; those are not acquisition files.
        let project_here = holds_project(folder);
        let mut pending = vec![PathBuf::new()];
        while let Some(relative) = pending.pop() {
            let current = folder.join(&relative);
            if (project_here && is_project_entry(&relative))
                || (!relative.as_os_str().is_empty() && holds_project(&current))
            {
                continue;
            }
            let mut entries = fs::read_dir(&current)
                .map_err(|e| format!("{}: {e}", current.display()))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?;
            entries.sort_by_key(|entry| entry.file_name());
            for entry in entries.into_iter().rev() {
                let meta = fs::symlink_metadata(entry.path())
                    .map_err(|e| format!("{}: {e}", entry.path().display()))?;
                let name = relative.join(entry.file_name());
                if meta.is_dir() {
                    pending.push(name);
                    continue;
                }
                if project_here && is_project_entry(&relative.join(entry.file_name())) {
                    continue;
                }
                let name = name.as_os_str().as_encoded_bytes();
                hash.update(&(name.len() as u64).to_le_bytes());
                hash.update(name);
                hash.update(&meta.len().to_le_bytes());
                let modified = meta
                    .modified()
                    .ok()
                    .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_nanos());
                hash.update(&modified.unwrap_or_default().to_le_bytes());
                bytes = bytes.saturating_add(meta.len());
            }
        }
        Ok(Self {
            bytes,
            modified_ns: None,
            directory: true,
            tree_digest: Some(hash.finalize().to_hex().to_string()),
        })
    }

    pub fn read(path: &Path) -> Result<Self, String> {
        let meta = fs::symlink_metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if meta.file_type().is_symlink() {
            return Err(
                "A captured input cannot contain symbolic links; choose its resolved dataset path"
                    .into(),
            );
        }
        if !meta.is_file() && !meta.is_dir() {
            return Err(format!(
                "{} is not a regular file or dataset directory",
                path.display()
            ));
        }
        let modified_ns = meta
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_nanos());
        if meta.is_file() {
            return Ok(Self {
                bytes: meta.len(),
                modified_ns,
                directory: false,
                tree_digest: None,
            });
        }
        let mut hash = blake3::Hasher::new();
        hash.update(b"tomojax-directory-metadata-v1\0");
        let project_here = holds_project(path);
        let mut pending = vec![PathBuf::new()];
        let mut bytes = 0u64;
        while let Some(relative) = pending.pop() {
            let current = path.join(&relative);
            // A project kept in the folder, or in a subfolder, is not acquisition data.
            if (project_here && is_project_entry(&relative))
                || (!relative.as_os_str().is_empty() && current.is_dir() && holds_project(&current))
            {
                continue;
            }
            let meta = fs::symlink_metadata(&current)
                .map_err(|e| format!("{}: {e}", current.display()))?;
            if meta.file_type().is_symlink() {
                return Err(format!(
                    "{} is a symbolic link; choose a dataset without linked descendants",
                    current.display()
                ));
            }
            if !meta.is_file() && !meta.is_dir() {
                return Err(format!(
                    "{} is not a regular dataset entry",
                    current.display()
                ));
            }
            let name = relative.as_os_str().as_encoded_bytes();
            hash.update(&(name.len() as u64).to_le_bytes());
            hash.update(name);
            hash.update(&[u8::from(meta.is_dir())]);
            let size = if meta.is_file() { meta.len() } else { 0 };
            hash.update(&size.to_le_bytes());
            // File times only: a folder's time changes whenever an entry is added,
            // and the entry names already record that.
            let modified = meta
                .modified()
                .ok()
                .filter(|_| meta.is_file())
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_nanos());
            hash.update(&[u8::from(modified.is_some())]);
            hash.update(&modified.unwrap_or_default().to_le_bytes());
            bytes = bytes
                .checked_add(size)
                .ok_or("The input's total byte count overflows")?;
            if meta.is_dir() {
                let mut entries = fs::read_dir(&current)
                    .map_err(|e| format!("{}: {e}", current.display()))?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| e.to_string())?;
                entries.sort_by_key(|entry| entry.file_name());
                pending.extend(
                    entries
                        .into_iter()
                        .rev()
                        .map(|entry| relative.join(entry.file_name())),
                );
            }
        }
        Ok(Self {
            bytes,
            modified_ns: None,
            directory: true,
            tree_digest: Some(hash.finalize().to_hex().to_string()),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CapturedInput {
    pub asset: String,
    pub name: String,
    pub path: PathBuf,
    pub kind: String,
    pub fingerprint: Fingerprint,
    #[serde(default)]
    pub open_parameters: Value,
    #[serde(default)]
    pub arrays: BTreeMap<String, ArrayDescriptor>,
    #[serde(default)]
    pub storage_fingerprints: BTreeMap<PathBuf, Fingerprint>,
    /// The files beside a parameter or header file, which its loader reads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub siblings: Option<Fingerprint>,
}

impl CapturedInput {
    /// Capture from a detached asset snapshot, so callers can perform all file
    /// metadata work on an I/O thread without borrowing the project or UI.
    pub fn from_asset(asset: &Asset, root: &Path) -> Result<Self, String> {
        let absolute = |path: &Path| {
            if path.is_absolute() {
                path.to_owned()
            } else {
                root.join(path)
            }
        };
        let path = absolute(&asset.path);
        let fingerprint = Fingerprint::read(&path)?;
        let mut arrays = asset.arrays.clone();
        let mut storage_fingerprints = BTreeMap::new();
        let mut resolve = |storage: &mut crate::protocol::Storage| -> Result<(), String> {
            if let crate::protocol::Storage::Hdf5 { path: stored, .. }
            | crate::protocol::Storage::Npy { path: stored, .. } = storage
            {
                let resolved = absolute(Path::new(stored));
                if resolved != path && !storage_fingerprints.contains_key(&resolved) {
                    storage_fingerprints.insert(resolved.clone(), Fingerprint::read(&resolved)?);
                }
                *stored = resolved.to_string_lossy().into();
            }
            Ok(())
        };
        for array in arrays.values_mut() {
            resolve(&mut array.storage)?;
            for axis in &mut array.axes {
                if let crate::protocol::Coordinates::Coordinates { values } = &mut axis.coordinates
                {
                    resolve(values)?;
                }
            }
        }
        // Parameter and header files (Nikon, SkyScan, ESRF) are read with the
        // frames beside them, so those frames are part of the input.
        let siblings = if path.is_file() && !holds_whole_acquisition(&path) {
            Some(Fingerprint::read_siblings(&path)?)
        } else {
            None
        };
        Self {
            asset: asset.id.clone(),
            name: asset.name.clone(),
            kind: asset.kind.clone(),
            fingerprint,
            path,
            open_parameters: asset.open_parameters.clone(),
            arrays,
            storage_fingerprints,
            siblings,
        }
        .with_files_named_in(&asset.open_parameters)
    }

    /// Fingerprints the files that loader or job settings name (flats, darks,
    /// angles): the worker reads them when the job runs, so they are inputs too.
    /// Only absolute paths are captured; the worker resolves relative ones itself.
    pub fn with_files_named_in(mut self, settings: &Value) -> Result<Self, String> {
        let mut references = Vec::new();
        referenced_files(settings, &mut references);
        for (file, typed) in references {
            let file = PathBuf::from(file);
            if !file.is_absolute()
                || file == self.path
                || self.storage_fingerprints.contains_key(&file)
                || (!typed && !file.exists())
            {
                continue;
            }
            let fingerprint = Fingerprint::read(&file)?;
            self.storage_fingerprints.insert(file, fingerprint);
        }
        Ok(self)
    }
}

/// Whether a file holds its whole acquisition. Parameter and header files (Nikon
/// `.xtekct`, SkyScan `.log`, ESRF `.edf`/`.par`) are read together with the frames
/// beside them.
pub fn holds_whole_acquisition(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| {
        matches!(
            e.to_ascii_lowercase().as_str(),
            "nx" | "nxs" | "h5" | "hdf" | "hdf5" | "npy" | "tif" | "tiff" | "txrm" | "txm" | "mat"
        )
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KeptPreviewContext {
    pub request: String,
    pub revision: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
    Interrupted,
}
impl JobStatus {
    pub fn active(self) -> bool {
        matches!(self, Self::Queued | Self::Running | Self::Paused)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub name: String,
    pub operation: String,
    pub parameters: Value,
    pub inputs: Vec<CapturedInput>,
    pub submitted: u64,
    pub started: Option<u64>,
    pub finished: Option<u64>,
    pub status: JobStatus,
    pub stage: String,
    pub error: Option<String>,
    pub result: Option<String>,
    pub destination: Option<PathBuf>,
    #[serde(default)]
    pub progress: Option<f32>,
    #[serde(default)]
    pub telemetry: crate::progress::JobProgress,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kept_preview: Option<KeptPreviewContext>,
}

impl Job {
    pub fn display_operation(&self) -> String {
        let old = self.operation.replace('_', " ");
        if let Some(suffix) = self.name.strip_prefix(&old) {
            format!("{}{suffix}", crate::operations::title(&self.operation))
        } else {
            self.name.clone()
        }
    }

    pub fn subject(&self) -> String {
        if self.operation == "simulate" {
            let phantom = self.parameters["phantom"]
                .as_str()
                .unwrap_or("shepp")
                .replace('_', " ");
            let mut chars = phantom.chars();
            return format!(
                "{}{} phantom",
                chars
                    .next()
                    .map(|c| c.to_uppercase().to_string())
                    .unwrap_or_default(),
                chars.as_str()
            );
        }
        self.inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    }

    pub fn result_name(&self, subject: &str, method: Option<&str>) -> String {
        if self.operation == "simulate" {
            return self.subject();
        }
        let product = method
            .map(crate::operations::method_name)
            .unwrap_or_else(|| crate::operations::title(&self.operation).into());
        if subject.is_empty() {
            product
        } else {
            format!("{subject} · {product}")
        }
    }
    pub fn new(operation: &str, parameters: Value, inputs: Vec<CapturedInput>) -> Self {
        Self {
            id: new_id("job"),
            name: crate::operations::title(operation).into(),
            operation: operation.into(),
            parameters,
            inputs,
            submitted: now_ms(),
            started: None,
            finished: None,
            status: JobStatus::Queued,
            stage: "Waiting".into(),
            error: None,
            result: None,
            destination: None,
            progress: None,
            telemetry: crate::progress::JobProgress::default(),
            kept_preview: None,
        }
    }
    pub fn check_inputs(&self) -> Result<(), String> {
        for input in &self.inputs {
            if input
                .path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("npz"))
            {
                return Err("NPZ inputs are not supported by the native workspace; use NXtomo/HDF5 or a recognised acquisition".into());
            }
            let current = Fingerprint::read(&input.path)?;
            if current != input.fingerprint
                || input.siblings.as_ref().is_some_and(|siblings| {
                    Fingerprint::read_siblings(&input.path).ok().as_ref() != Some(siblings)
                })
            {
                return Err(format!(
                    "{} changed after submission. Review the input and submit a new job.",
                    input.name
                ));
            }
            let mut checked = BTreeSet::new();
            for array in input.arrays.values() {
                let coordinates = array
                    .axes
                    .iter()
                    .filter_map(|axis| match &axis.coordinates {
                        crate::protocol::Coordinates::Coordinates { values } => Some(values),
                        crate::protocol::Coordinates::Uniform { .. } => None,
                    });
                for storage in std::iter::once(&array.storage).chain(coordinates) {
                    let path = match storage {
                        crate::protocol::Storage::Hdf5 { path, .. }
                        | crate::protocol::Storage::Npy { path, .. } => Path::new(path),
                        crate::protocol::Storage::Member { .. } => continue,
                    };
                    if path == input.path || !checked.insert(path) {
                        continue;
                    }
                    let expected = input.storage_fingerprints.get(path).ok_or_else(|| format!(
                        "The storage fingerprint for {} was not captured. Review {} and submit a new job.", path.display(), input.name
                    ))?;
                    if Fingerprint::read(path)? != *expected {
                        return Err(format!(
                            "{} changed after submission ({}). Review the input and submit a new job.",
                            input.name,
                            path.display()
                        ));
                    }
                }
            }
            // Calibration files from the loader settings.
            for (path, expected) in &input.storage_fingerprints {
                if !checked.contains(path.as_path()) && Fingerprint::read(path)? != *expected {
                    return Err(format!(
                        "{} changed after submission ({}). Review the input and submit a new job.",
                        input.name,
                        path.display()
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Files named in loader settings: typed `npy`/`hdf5` storage references (`true`),
/// and bare strings that may be paths (`false`; kept only if such a file exists).
fn referenced_files(value: &Value, out: &mut Vec<(String, bool)>) {
    match value {
        Value::Object(map) => {
            if let (Some("npy" | "hdf5"), Some(path)) = (
                map.get("type").and_then(Value::as_str),
                map.get("path").and_then(Value::as_str),
            ) {
                out.push((path.to_owned(), true));
            }
            for (key, item) in map {
                if !matches!(key.as_str(), "type" | "path") {
                    referenced_files(item, out);
                }
            }
        }
        Value::Array(items) => items.iter().for_each(|item| referenced_files(item, out)),
        Value::String(text) => out.push((text.clone(), false)),
        _ => {}
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Recipe {
    pub id: String,
    pub name: String,
    pub operation: String,
    pub parameters: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default, PartialEq)]
pub struct Workspace {
    pub active: Option<String>,
    pub selected: BTreeSet<String>,
    pub compare: [Option<String>; 2],
    pub view: String,
    pub workspace: String,
    pub indices: [usize; 3],
    pub projection: usize,
    pub sinogram: usize,
    pub window: Option<[f32; 2]>,
    pub colormap: String,
    pub invert: bool,
    pub camera: [f32; 2],
    pub clip: [f32; 3],
    pub density: f32,
    #[serde(default)]
    pub threshold: f32,
    pub lighting: f32,
    pub zoom: f32,
    pub pan: [f32; 2],
    pub operation: String,
    pub parameters: Value,
    #[serde(default)]
    pub draft_edits: BTreeMap<String, String>,
    #[serde(default)]
    pub draft_choices: BTreeMap<String, String>,
    #[serde(default)]
    pub measurements: Vec<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    pub version: u32,
    pub id: String,
    pub name: String,
    pub created: u64,
    pub modified: u64,
    pub assets: Vec<Asset>,
    pub jobs: Vec<Job>,
    pub recipes: Vec<Recipe>,
    pub workspace: Workspace,
}

impl Document {
    fn new(path: &Path) -> Self {
        Self {
            version: 2,
            id: new_id("project"),
            name: path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into(),
            created: now_ms(),
            modified: now_ms(),
            assets: Vec::new(),
            jobs: Vec::new(),
            recipes: Vec::new(),
            workspace: Workspace::default(),
        }
    }
    pub fn asset(&self, id: &str) -> Option<&Asset> {
        self.assets.iter().find(|a| a.id == id)
    }
    pub fn asset_mut(&mut self, id: &str) -> Option<&mut Asset> {
        self.assets.iter_mut().find(|a| a.id == id)
    }
    pub fn job_mut(&mut self, id: &str) -> Option<&mut Job> {
        self.jobs.iter_mut().find(|j| j.id == id)
    }
    pub fn recover(&mut self) -> usize {
        let mut count = 0;
        for job in &mut self.jobs {
            if job.status.active() {
                job.status = JobStatus::Interrupted;
                job.stage = "Review before resuming".into();
                job.error=Some("The previous session ended before this job was committed. Retry explicitly after reviewing its output destination.".into());
                count += 1;
            }
        }
        count
    }
}

struct Snapshot {
    generation: u64,
    bytes: Vec<u8>,
}
#[derive(Debug)]
pub struct SaveAck {
    pub generation: u64,
    pub error: Option<String>,
}

/// One coalescing background writer holds the OS lock for its complete lifetime.
pub struct Project {
    pub root: PathBuf,
    pub document: Document,
    pub generation: u64,
    pub saved_generation: u64,
    pub save_error: Option<String>,
    pending: Arc<Mutex<Option<Snapshot>>>,
    signal: mpsc::Sender<()>,
    completed: mpsc::Receiver<SaveAck>,
    writer: Option<JoinHandle<()>>,
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or("A project manifest needs a parent directory")?;
    // Named after the file it replaces, so folder fingerprints can skip it.
    let mut temp = tempfile::Builder::new()
        .prefix(path.file_name().unwrap_or_default())
        .tempfile_in(parent)
        .map_err(|e| e.to_string())?;
    temp.write_all(bytes)
        .and_then(|()| temp.as_file().sync_all())
        .map_err(|e| e.to_string())?;
    temp.persist(path).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    File::open(parent)
        .and_then(|file| file.sync_all())
        .map_err(|e| e.to_string())?;
    Ok(())
}

impl Project {
    pub fn open(path: &Path, create: bool) -> Result<Self, String> {
        let path = if path.file_name().is_some_and(|name| name == MANIFEST) {
            path.parent().unwrap_or(path)
        } else {
            path
        };
        if create {
            fs::create_dir_all(path).map_err(|e| e.to_string())?;
        }
        let root = fs::canonicalize(path)
            .map_err(|e| format!("Could not open project {}: {e}", path.display()))?;
        let manifest = root.join(MANIFEST);
        if create && manifest.exists() {
            return Err("This directory already contains a project. Open it instead.".into());
        }
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join(LOCK))
            .map_err(|e| e.to_string())?;
        lock.try_lock()
            .map_err(|_| "This project is already open in another TomoJAX window".to_string())?;
        let mut document = if create {
            Document::new(&root)
        } else {
            let size = fs::metadata(&manifest)
                .map_err(|e| format!("No project manifest in {}: {e}", root.display()))?
                .len();
            if size > MAX_MANIFEST_BYTES {
                return Err("Project manifest exceeds the 32 MiB limit".into());
            }
            serde_json::from_slice::<Document>(&fs::read(&manifest).map_err(|e| e.to_string())?)
                .map_err(|e| format!("Invalid project manifest: {e}"))?
        };
        if document.version == 1 {
            // Only replace names generated by Studio v1. Submitted inputs are immutable.
            let mut renamed = BTreeMap::new();
            for asset in &mut document.assets {
                let Some(job) = document
                    .jobs
                    .iter()
                    .find(|job| job.result.as_ref() == Some(&asset.id))
                else {
                    continue;
                };
                let input = job.inputs.first();
                let previous = input.map_or("Phantom", |i| &i.name);
                if asset.name
                    == format!(
                        "{previous} · {}",
                        asset.method.as_deref().unwrap_or(&job.operation)
                    )
                {
                    let subject = input
                        .and_then(|i| renamed.get(&i.asset))
                        .map_or(previous, String::as_str);
                    asset.name = job.result_name(subject, asset.method.as_deref());
                    renamed.insert(asset.id.clone(), asset.name.clone());
                }
            }
            // A v1 offline filter was a text field; a connected worker used a picker.
            // Keep the actual saved value as the initial choice in either case.
            let workspace = &mut document.workspace;
            if let Some(filter) = workspace.parameters["filter"].as_str() {
                workspace
                    .draft_choices
                    .insert("filter".into(), filter.into());
            }
            document.version = 2;
        }
        if document.version != 2 {
            return Err(format!(
                "Project version {} is not supported by this build",
                document.version
            ));
        }
        validate_document(&document)?;
        document.recover();
        for directory in ["derived", "inputs", "cache"] {
            fs::create_dir_all(root.join(directory)).map_err(|e| e.to_string())?;
        }
        if create {
            atomic_write(
                &manifest,
                &serde_json::to_vec_pretty(&document).map_err(|e| e.to_string())?,
            )?;
        }
        let pending = Arc::new(Mutex::new(None::<Snapshot>));
        let writer_pending = pending.clone();
        let (signal, work) = mpsc::channel();
        let (ack, completed) = mpsc::channel();
        let writer = thread::spawn(move || {
            let _lock = lock;
            while work.recv().is_ok() {
                while work.try_recv().is_ok() {}
                let snapshot = writer_pending.lock().unwrap().take();
                if let Some(snapshot) = snapshot {
                    let error = atomic_write(&manifest, &snapshot.bytes).err();
                    let _ = ack.send(SaveAck {
                        generation: snapshot.generation,
                        error,
                    });
                }
            }
            if let Some(snapshot) = writer_pending.lock().unwrap().take() {
                let error = atomic_write(&manifest, &snapshot.bytes).err();
                let _ = ack.send(SaveAck {
                    generation: snapshot.generation,
                    error,
                });
            }
        });
        let mut project = Self {
            root,
            document,
            generation: 0,
            saved_generation: 0,
            save_error: None,
            pending,
            signal,
            completed,
            writer: Some(writer),
        };
        project.changed()?;
        Ok(project)
    }
    pub fn resolve(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            path.into()
        } else {
            self.root.join(path)
        }
    }
    pub fn stored_path(&self, path: &Path) -> PathBuf {
        path.strip_prefix(&self.root).unwrap_or(path).into()
    }
    pub fn capture(&self, id: &str) -> Result<CapturedInput, String> {
        let asset = self
            .document
            .asset(id)
            .ok_or("The selected input no longer exists")?;
        CapturedInput::from_asset(asset, &self.root)
    }
    pub fn changed(&mut self) -> Result<(), String> {
        self.document.modified = now_ms();
        self.save_error = None;
        self.generation += 1;
        let bytes = match serde_json::to_vec_pretty(&self.document) {
            Ok(bytes) => bytes,
            Err(error) => {
                let error = error.to_string();
                self.save_error = Some(error.clone());
                return Err(error);
            }
        };
        if bytes.len() as u64 > MAX_MANIFEST_BYTES {
            let error = "Project metadata exceeds the 32 MiB manifest limit. Latest changes remain in memory; the previous saved manifest is intact.".to_owned();
            self.save_error = Some(error.clone());
            return Err(error);
        }
        *self.pending.lock().unwrap() = Some(Snapshot {
            generation: self.generation,
            bytes,
        });
        self.signal
            .send(())
            .map_err(|_| "The project writer stopped".to_string())
    }
    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        while let Ok(ack) = self.completed.try_recv() {
            changed = true;
            if ack.error.is_none() {
                self.saved_generation = self.saved_generation.max(ack.generation);
            }
            // An older valid write must not erase a newer serialization/size
            // error whose snapshot was deliberately withheld from the writer.
            if ack.generation == self.generation {
                self.save_error = ack.error;
            }
        }
        changed
    }
    pub fn dirty(&self) -> bool {
        self.generation > self.saved_generation
    }
    pub fn flush(&mut self) -> Result<(), String> {
        if let Some(error) = &self.save_error {
            return Err(error.clone());
        }
        while self.dirty() {
            let ack = self
                .completed
                .recv()
                .map_err(|_| "Project writer stopped before saving".to_string())?;
            if let Some(error) = ack.error {
                self.save_error = Some(error.clone());
                return Err(error);
            }
            self.saved_generation = self.saved_generation.max(ack.generation);
        }
        Ok(())
    }
}

impl Drop for Project {
    fn drop(&mut self) {
        let _ = self.flush();
        let (replacement, _) = mpsc::channel();
        drop(std::mem::replace(&mut self.signal, replacement));
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

fn validate_document(document: &Document) -> Result<(), String> {
    let mut ids = BTreeSet::new();
    for id in document
        .assets
        .iter()
        .map(|a| &a.id)
        .chain(document.jobs.iter().map(|j| &j.id))
    {
        if id.is_empty() || !ids.insert(id) {
            return Err("Project contains empty or duplicate identifiers".into());
        }
    }
    for asset in &document.assets {
        if asset.managed
            && (asset.path.is_absolute()
                || asset
                    .path
                    .components()
                    .any(|c| matches!(c, std::path::Component::ParentDir)))
        {
            return Err("Managed project data must stay inside its project directory".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_recent_project_is_listed_by_its_own_name() {
        let directory = tempfile::tempdir().unwrap();
        assert_eq!(super::manifest_name(directory.path()), None);
        std::fs::write(
            directory.path().join(super::MANIFEST),
            r#"{"version":1,"name":"Shepp review"}"#,
        )
        .unwrap();
        assert_eq!(
            super::manifest_name(directory.path()).as_deref(),
            Some("Shepp review")
        );
    }

    use super::*;
    use serde_json::json;

    #[test]
    fn old_generated_names_are_upgraded_once_and_captured_inputs_are_preserved() {
        let dir = tempfile::tempdir().unwrap();
        let mut p = Project::open(dir.path(), true).unwrap();
        let mut simulation = Asset::input("ground-truth.nxs".into(), true);
        simulation.name = "Phantom · simulate".into();
        let mut sim_job = Job::new("simulate", json!({"phantom":"random_shapes"}), vec![]);
        sim_job.result = Some(simulation.id.clone());
        std::fs::write(p.resolve(&simulation.path), []).unwrap();
        let input = CapturedInput::from_asset(&simulation, &p.root).unwrap();
        let mut reconstruction = Asset::input("result.nxs".into(), true);
        reconstruction.parent = Some(simulation.id.clone());
        reconstruction.method = Some("fbp".into());
        reconstruction.name = "Phantom · simulate · fbp".into();
        let mut job = Job::new("reconstruct", json!({"method":"fbp"}), vec![input]);
        job.result = Some(reconstruction.id.clone());
        p.document.assets = vec![simulation, reconstruction];
        p.document.jobs = vec![sim_job, job];
        p.document.version = 1;
        p.document.workspace.parameters = json!({"method":"fbp", "filter":"hann"});
        p.document
            .workspace
            .draft_choices
            .insert("filter".into(), String::new());
        p.changed().unwrap();
        p.flush().unwrap();
        drop(p);
        let mut p = Project::open(dir.path(), false).unwrap();
        assert_eq!(p.document.workspace.draft_choices["filter"], "hann");
        assert_eq!(p.document.assets[0].name, "Random shapes phantom");
        assert_eq!(p.document.assets[1].name, "Random shapes phantom · FBP");
        assert_eq!(p.document.jobs[1].inputs[0].name, "Phantom · simulate");
        p.document.assets[0].name = "Phantom · simulate".into();
        p.changed().unwrap();
        p.flush().unwrap();
        drop(p);
        assert_eq!(
            Project::open(dir.path(), false).unwrap().document.assets[0].name,
            "Phantom · simulate"
        );
    }

    #[test]
    fn oversized_history_keeps_memory_and_the_last_reopenable_manifest() {
        let directory = tempfile::tempdir().unwrap();
        let mut project = Project::open(directory.path(), true).unwrap();
        project.document.name = "Last valid checkpoint".into();
        project.changed().unwrap();
        let valid_generation = project.generation;
        let mut job = Job::new("align", json!({}), vec![]);
        job.telemetry.notes.push(crate::progress::JobNote {
            request: "alignment".into(),
            task: None,
            time: 1.,
            level: "warning".into(),
            message: "x".repeat(MAX_MANIFEST_BYTES as usize),
        });
        project.document.jobs.push(job);
        assert!(project.changed().unwrap_err().contains("32 MiB"));
        assert_eq!(
            project.document.jobs[0].telemetry.notes[0].message.len(),
            MAX_MANIFEST_BYTES as usize
        );
        let until = std::time::Instant::now() + std::time::Duration::from_secs(3);
        while project.saved_generation < valid_generation {
            project.poll();
            assert!(std::time::Instant::now() < until);
            std::thread::yield_now();
        }
        assert!(
            project.save_error.as_deref().unwrap().contains("32 MiB"),
            "the older successful write must not hide the unsaved-history error"
        );
        assert!(project.dirty());
        assert!(project.flush().is_err());
        drop(project);
        let reopened = Project::open(directory.path(), false).unwrap();
        assert_eq!(reopened.document.name, "Last valid checkpoint");
        assert!(reopened.document.jobs.is_empty());
    }

    #[test]
    #[cfg(unix)]
    fn serialization_failure_cannot_leave_flush_waiting_for_an_unqueued_snapshot() {
        use std::os::unix::ffi::OsStringExt;
        let directory = tempfile::tempdir().unwrap();
        let mut project = Project::open(directory.path(), true).unwrap();
        let invalid_path = PathBuf::from(std::ffi::OsString::from_vec(vec![0xff]));
        project
            .document
            .assets
            .push(Asset::input(invalid_path, false));
        assert!(project.changed().is_err());
        assert!(project.save_error.is_some());
        assert!(project.flush().is_err());
    }

    #[test]
    fn calibration_files_named_in_loader_settings_are_captured_inputs() {
        let directory = tempfile::tempdir().unwrap();
        let frames = directory.path().join("frames.h5");
        let flat = directory.path().join("flat.tif");
        let dark = directory.path().join("dark.npy");
        for (path, bytes) in [(&frames, "frames"), (&flat, "flat"), (&dark, "dark")] {
            fs::write(path, bytes).unwrap();
        }
        let mut asset = Asset::input(frames, false);
        asset.open_parameters = json!({
            "kind": "frames",
            "flats": flat,
            "darks": {"type": "npy", "path": dark},
        });
        let job = Job::new(
            "correct",
            json!({}),
            vec![CapturedInput::from_asset(&asset, directory.path()).unwrap()],
        );
        assert_eq!(job.inputs[0].storage_fingerprints.len(), 2);
        assert!(job.check_inputs().is_ok());
        fs::write(&flat, "replaced flat field").unwrap();
        assert!(job.check_inputs().unwrap_err().contains("flat.tif"));
    }

    #[test]
    fn frames_beside_a_header_file_and_job_calibration_files_are_inputs() {
        let directory = tempfile::tempdir().unwrap();
        let header = directory.path().join("scan.xtekct");
        let frame = directory.path().join("scan_0001.tif");
        let flat = directory.path().join("flat.tif");
        for (path, bytes) in [(&header, "header"), (&frame, "pixels"), (&flat, "flat")] {
            fs::write(path, bytes).unwrap();
        }
        let asset = Asset::input(header, false);
        let settings = json!({"flats": flat, "darks": "relative-dark.tif"});
        let input = CapturedInput::from_asset(&asset, directory.path())
            .unwrap()
            .with_files_named_in(&settings)
            .unwrap();
        // Absolute paths only: the worker resolves relative ones itself.
        assert_eq!(
            input.storage_fingerprints.keys().collect::<Vec<_>>(),
            [&flat]
        );
        let job = Job::new("open", settings, vec![input]);
        assert!(job.check_inputs().is_ok());
        fs::write(directory.path().join("project.tomojax.json"), "{}").unwrap();
        fs::create_dir(directory.path().join("derived")).unwrap();
        fs::write(directory.path().join("derived/corrected.nxs"), "result").unwrap();
        assert!(
            job.check_inputs().is_ok(),
            "a project kept beside the scan writes files that are not frames"
        );
        fs::write(&frame, "re-exported pixels").unwrap();
        assert!(job.check_inputs().is_err());

        // Nikon can keep its projections in a subfolder.
        let header = directory.path().join("nested/scan.xtekct");
        fs::create_dir_all(directory.path().join("nested/Projections")).unwrap();
        fs::write(&header, "header").unwrap();
        let frame = directory.path().join("nested/Projections/scan_0001.tif");
        fs::write(&frame, "pixels").unwrap();
        let job = Job::new(
            "correct",
            json!({}),
            vec![
                CapturedInput::from_asset(&Asset::input(header, false), directory.path()).unwrap(),
            ],
        );
        assert!(job.check_inputs().is_ok());
        fs::write(&frame, "re-exported pixels").unwrap();
        assert!(job.check_inputs().is_err());
    }

    #[test]
    fn a_project_kept_in_a_tiff_folder_is_not_part_of_the_acquisition() {
        let directory = tempfile::tempdir().unwrap();
        let folder = directory.path().join("tiffs");
        fs::create_dir(&folder).unwrap();
        fs::write(folder.join("view_0001.tif"), "pixels").unwrap();
        let mut project = Project::open(&folder, true).unwrap();
        let asset = Asset::input(folder.clone(), false);
        project.document.assets.push(asset.clone());
        let job = Job::new("open", json!({}), vec![project.capture(&asset.id).unwrap()]);
        project.document.jobs.push(job.clone());
        project.changed().unwrap();
        project.flush().unwrap();
        fs::write(folder.join("derived/result.nxs"), "result").unwrap();
        assert!(job.check_inputs().is_ok());
        fs::write(folder.join("view_0001.tif"), "re-exported pixels").unwrap();
        assert!(job.check_inputs().is_err());
    }

    #[test]
    fn a_project_in_a_subfolder_of_a_tiff_folder_is_not_part_of_the_acquisition() {
        let directory = tempfile::tempdir().unwrap();
        let folder = directory.path().join("tiffs");
        fs::create_dir(&folder).unwrap();
        fs::write(folder.join("view_0001.tif"), "pixels").unwrap();
        let mut project = Project::open(&folder.join("studio"), true).unwrap();
        let asset = Asset::input(folder.clone(), false);
        project.document.assets.push(asset.clone());
        let job = Job::new("open", json!({}), vec![project.capture(&asset.id).unwrap()]);
        project.document.jobs.push(job.clone());
        project.changed().unwrap();
        project.flush().unwrap();
        assert!(job.check_inputs().is_ok());
    }

    #[test]
    fn array_paths_inside_the_project_are_stored_relative() {
        let mut descriptor: ArrayDescriptor = serde_json::from_value(json!({
            "shape":[2], "dtype":"<f4", "unit":"1", "domain":"attenuation",
            "storage":{"type":"hdf5","path":"/data/project/inputs/a/scan.h5","dataset":"/data"},
            "axes":[{"name":"z","frame":"sample","unit":"mm","coordinates":{
                "type":"coordinates","values":{"type":"npy","path":"/elsewhere/z.npy"}}}],
            "fixed":[],"columns":[]
        }))
        .unwrap();
        descriptor.relative_to(Path::new("/data/project"));
        let value = serde_json::to_value(&descriptor).unwrap();
        assert_eq!(value["storage"]["path"], "inputs/a/scan.h5");
        assert_eq!(
            value["axes"][0]["coordinates"]["values"]["path"],
            "/elsewhere/z.npy"
        );
    }

    #[test]
    fn detached_input_capture_retains_storage_fingerprints_on_an_io_thread() {
        let directory = tempfile::tempdir().unwrap();
        let mut project = Project::open(&directory.path().join("project"), true).unwrap();
        let path = project.root.join("derived/input.npy");
        fs::write(&path, b"original captured bytes").unwrap();
        let asset = Asset::input("derived/input.npy".into(), true);
        project.document.assets.push(asset.clone());
        let expected = project.capture(&asset.id).unwrap();
        let root = project.root.clone();
        let captured = std::thread::spawn(move || CapturedInput::from_asset(&asset, &root))
            .join()
            .unwrap()
            .unwrap();
        assert_eq!(captured.path, path);
        assert_eq!(captured.fingerprint, expected.fingerprint);
        assert_eq!(captured.asset, expected.asset);
        assert!(
            Job::new("slice", json!({"index":3}), vec![captured])
                .check_inputs()
                .is_ok()
        );
    }

    #[test]
    fn saved_job_keeps_task_tree_metric_units_and_warning_notes() {
        let directory = tempfile::tempdir().unwrap();
        let mut project = Project::open(directory.path(), true).unwrap();
        let mut job = Job::new("align", json!({"method":"joint"}), vec![]);
        for value in [
            json!({"event":"started","schema":2,"task":1,"time":0.,"parent":null,"name":"align","label":"Alignment","total":2,"unit":"steps"}),
            json!({"event":"started","schema":2,"task":2,"time":0.1,"parent":1,"name":"joint","label":"Level 1","total":5,"unit":"iterations"}),
            json!({"event":"advanced","schema":2,"task":2,"time":1.,"done":1,"total":5,"metrics":{"objective":4.5,"pose_step_rms":0.25},"eta":4.}),
            json!({"event":"note","schema":2,"task":2,"time":1.1,"level":"warning","message":"Lengths are uncalibrated pixels"}),
        ] {
            job.telemetry
                .apply("alignment-request", &serde_json::from_value(value).unwrap());
        }
        job.kept_preview = Some(KeptPreviewContext {
            request: "preview-request".into(),
            revision: 3,
        });
        project.document.jobs.push(job);
        project.changed().unwrap();
        project.flush().unwrap();
        drop(project);
        let reopened = Project::open(directory.path(), false).unwrap();
        let job = &reopened.document.jobs[0];
        assert_eq!(
            job.telemetry
                .tree()
                .iter()
                .map(|(depth, _)| *depth)
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(
            job.telemetry.tasks[1].metrics["pose_step_rms"]
                .unit
                .as_deref(),
            Some("working pixels")
        );
        assert_eq!(
            job.telemetry.notes[0].message,
            "Lengths are uncalibrated pixels"
        );
        assert_eq!(job.kept_preview.as_ref().unwrap().revision, 3);
    }

    #[test]
    fn a_changed_existing_tiff_invalidates_a_captured_directory() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("acquisition");
        fs::create_dir_all(input.join("views")).unwrap();
        let frame = input.join("views/frame-001.tif");
        fs::write(&frame, b"first pixels").unwrap();
        let timestamp = UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
        File::options()
            .write(true)
            .open(&frame)
            .unwrap()
            .set_times(fs::FileTimes::new().set_modified(timestamp))
            .unwrap();
        let mut project = Project::open(&directory.path().join("project"), true).unwrap();
        let asset = Asset::input(input.clone(), false);
        let asset_id = asset.id.clone();
        project.document.assets.push(asset);
        let job = Job::new(
            "reconstruct",
            json!({"method":"fourier"}),
            vec![project.capture(&asset_id).unwrap()],
        );
        let old_directory_time = fs::metadata(&input).unwrap().modified().unwrap();
        assert!(job.check_inputs().is_ok());
        assert_eq!(
            job.inputs[0].fingerprint,
            Fingerprint::read(&input).unwrap()
        );
        fs::write(&frame, b"other pixels").unwrap();
        File::options()
            .write(true)
            .open(&frame)
            .unwrap()
            .set_times(
                fs::FileTimes::new().set_modified(timestamp + std::time::Duration::from_secs(2)),
            )
            .unwrap();
        assert_eq!(
            fs::metadata(&input).unwrap().modified().unwrap(),
            old_directory_time
        );
        assert!(
            job.check_inputs()
                .unwrap_err()
                .contains("changed after submission")
        );
    }

    #[test]
    #[cfg(unix)]
    fn directory_capture_does_not_follow_external_symbolic_links() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("acquisition");
        fs::create_dir(&input).unwrap();
        let external = directory.path().join("outside.tif");
        fs::write(&external, b"outside").unwrap();
        std::os::unix::fs::symlink(&external, input.join("linked.tif")).unwrap();
        assert!(
            Fingerprint::read(&input)
                .unwrap_err()
                .contains("symbolic link")
        );
    }

    #[test]
    fn captured_array_products_detect_changes_in_secondary_data_and_coordinates() {
        let directory = tempfile::tempdir().unwrap();
        let mut project = Project::open(&directory.path().join("project"), true).unwrap();
        for name in ["main.npy", "secondary.npy", "angles.npy"] {
            fs::write(project.root.join("derived").join(name), b"original data").unwrap();
        }
        let mut asset = Asset::input("derived/main.npy".into(), true);
        let primary: ArrayDescriptor = serde_json::from_value(json!({
            "storage":{"type":"npy","path":"derived/main.npy"},"shape":[2],"dtype":"float32",
            "axes":[{"name":"view","frame":"acquisition","unit":"radian",
                "coordinates":{"type":"coordinates","values":{"type":"npy","path":"derived/angles.npy"}}}],
            "unit":"1","domain":"line_integrals"
        })).unwrap();
        let mut secondary = primary.clone();
        secondary.storage = crate::protocol::Storage::Npy {
            path: "derived/secondary.npy".into(),
            artifact: None,
        };
        asset.arrays.insert("before".into(), primary);
        asset.arrays.insert("after".into(), secondary);
        let id = asset.id.clone();
        project.document.assets.push(asset);
        for changed in ["secondary.npy", "angles.npy"] {
            let captured = project.capture(&id).unwrap();
            assert_eq!(
                captured.storage_fingerprints.len(),
                2,
                "shared coordinate storage is captured once"
            );
            let job = Job::new("export_array", json!({}), vec![captured]);
            let encoded = serde_json::to_vec(&job).unwrap();
            let job: Job = serde_json::from_slice(&encoded).unwrap();
            assert!(job.check_inputs().is_ok());
            fs::write(
                project.root.join("derived").join(changed),
                b"changed scalar or coordinate data",
            )
            .unwrap();
            let error = job.check_inputs().unwrap_err();
            assert!(error.contains("changed after submission"));
            assert!(error.contains(changed));
        }
        let captured = project.capture(&id).unwrap();
        let mut old = serde_json::to_value(&captured).unwrap();
        old.as_object_mut().unwrap().remove("storage_fingerprints");
        let old: CapturedInput = serde_json::from_value(old).unwrap();
        assert!(
            Job::new("export_array", json!({}), vec![old])
                .check_inputs()
                .unwrap_err()
                .contains("was not captured")
        );
    }

    #[test]
    fn autosave_reopen_retains_context_and_marks_unfinished_jobs() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("Copper corrosion");
        {
            let mut project = Project::open(&root, true).unwrap();
            project.document.workspace.indices = [7, 19, 31];
            project.document.workspace.window = Some([-0.1, 2.0]);
            project
                .document
                .jobs
                .push(Job::new("simulate", json!({"size":32}), vec![]));
            project.changed().unwrap();
            project.flush().unwrap();
            assert!(!project.dirty());
            assert!(
                Project::open(&root, false).is_err(),
                "another window must not race this writer"
            );
        }
        let mut project = Project::open(&root, false).unwrap();
        assert_eq!(project.document.workspace.indices, [7, 19, 31]);
        assert_eq!(project.document.jobs[0].status, JobStatus::Interrupted);
        project.flush().unwrap();
    }

    #[test]
    fn submissions_reject_changed_inputs_and_survive_relinking() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("scan.nxs");
        fs::write(&source, b"original").unwrap();
        let mut project = Project::open(&dir.path().join("project"), true).unwrap();
        let asset = Asset::input(source.clone(), false);
        let id = asset.id.clone();
        project.document.assets.push(asset);
        let job = Job::new(
            "reconstruct",
            json!({"method":"fourier"}),
            vec![project.capture(&id).unwrap()],
        );
        project.document.asset_mut(&id).unwrap().path = dir.path().join("elsewhere.nxs");
        assert_eq!(job.inputs[0].path, source);
        assert!(job.check_inputs().is_ok());
        fs::write(&source, b"changed input content").unwrap();
        assert!(job.check_inputs().is_err());
    }

    #[test]
    fn incompatible_and_escaping_manifests_are_rejected_without_rewrite() {
        let dir = tempfile::tempdir().unwrap();
        let mut document = Document::new(dir.path());
        document.version = 99;
        let original = serde_json::to_vec(&document).unwrap();
        fs::write(dir.path().join(MANIFEST), &original).unwrap();
        assert!(Project::open(dir.path(), false).is_err());
        assert_eq!(fs::read(dir.path().join(MANIFEST)).unwrap(), original);
        document.version = 1;
        let mut asset = Asset::input("../outside.nxs".into(), true);
        asset.id = "a".into();
        document.assets.push(asset);
        assert!(validate_document(&document).is_err());
    }
}
