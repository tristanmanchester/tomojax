//! Paths interpreted on the worker. Only the local project folder is a client path.
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    path::{Path, PathBuf},
};

/// A path on the project's machine.
///
/// ```compile_fail
/// use studio_core::worker_path::WorkerPath;
/// let path = WorkerPath::from("/data/scan.nxs");
/// std::fs::read(path);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(transparent)]
pub struct WorkerPath(PathBuf);

impl WorkerPath {
    /// Lexical operations on a wire path, such as resolving array registrations.
    pub fn lexical(&self) -> &Path {
        &self.0
    }

    pub fn join(&self, name: &str) -> Self {
        Self(self.0.join(name))
    }
    pub fn parent(&self) -> Option<Self> {
        self.0.parent().map(|p| Self(p.into()))
    }
    pub fn name(&self) -> &str {
        self.0.file_name().and_then(|s| s.to_str()).unwrap_or("/")
    }
}
impl From<PathBuf> for WorkerPath {
    fn from(value: PathBuf) -> Self {
        Self(value)
    }
}
impl From<&Path> for WorkerPath {
    fn from(value: &Path) -> Self {
        Self(value.into())
    }
}
impl From<&str> for WorkerPath {
    fn from(value: &str) -> Self {
        Self(value.into())
    }
}

impl fmt::Display for WorkerPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.display().fmt(f)
    }
}
