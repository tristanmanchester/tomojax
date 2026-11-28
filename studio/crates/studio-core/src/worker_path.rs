//! Paths interpreted on the worker. Only the local project folder is a client path.
use serde::{Deserialize, Serialize};
use std::fmt;

/// A path on the project's machine.
///
/// ```compile_fail
/// use studio_core::worker_path::WorkerPath;
/// let path = WorkerPath::from("/data/scan.nxs");
/// std::fs::read(path);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(transparent)]
pub struct WorkerPath(String);

impl WorkerPath {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn join(&self, name: &str) -> Self {
        if name.starts_with('/') {
            Self::from(name)
        } else {
            Self(format!("{}/{}", self.0.trim_end_matches('/'), name))
        }
    }

    pub fn parent(&self) -> Option<Self> {
        let (parent, _) = self.0.trim_end_matches('/').rsplit_once('/')?;
        Some(Self::from(if parent.is_empty() { "/" } else { parent }))
    }

    pub fn name(&self) -> &str {
        self.0
            .trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or("")
    }
}

impl From<&str> for WorkerPath {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl fmt::Display for WorkerPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
