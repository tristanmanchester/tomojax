//! The marker ties a project on the client to one results tree on this machine.
use crate::files;
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::fs::{self, File};
use std::io::ErrorKind;
use std::path::{Component, Path};

#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Results {
    Claim {
        parent: String,
        name: String,
        project: String,
    },
    Verify {
        path: String,
        project: String,
    },
}

fn component(value: &str) -> Result<&str> {
    ensure!(
        Path::new(value).components().count() == 1
            && matches!(
                Path::new(value).components().next(),
                Some(Component::Normal(_))
            )
            && !value.contains('/'),
        "expected a file name"
    );
    Ok(value)
}

fn belongs(path: &Path, marker: &str) -> bool {
    path.symlink_metadata().is_ok_and(|m| m.is_dir())
        && path
            .join(marker)
            .symlink_metadata()
            .is_ok_and(|m| m.is_file() && m.len() == 0)
}

impl Results {
    pub fn run(self) -> Result<Value> {
        let project = match &self {
            Self::Claim { project, .. } | Self::Verify { project, .. } => component(project)?,
        };
        let marker = format!(".tomojax-project-{project}");
        let path = match self {
            Self::Verify { path, .. } => {
                let path = files::resolve(&path)?;
                ensure!(
                    belongs(&path, &marker),
                    "This connection reaches other data. Choose the connection that owns this project's results."
                );
                path
            }
            Self::Claim {
                parent,
                name,
                project,
            } => {
                component(&name)?;
                let parent = files::resolve(&parent)?.canonicalize()?;
                let short: String = project
                    .chars()
                    .rev()
                    .take(8)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect();
                let mut claimed = None;
                for name in [name.clone(), format!("{name}-{short}")] {
                    let path = parent.join(name);
                    match fs::create_dir(&path) {
                        Ok(()) => {
                            File::create_new(path.join(&marker))?.sync_all()?;
                        }
                        Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                            if !belongs(&path, &marker) {
                                continue;
                            }
                        }
                        Err(e) => {
                            return Err(e).context("Could not create the project's results folder");
                        }
                    }
                    for name in ["jobs", "kept", "inputs"] {
                        fs::create_dir_all(path.join(name))?;
                    }
                    claimed = Some(path);
                    break;
                }
                let Some(path) = claimed else {
                    bail!("Both results folders already exist. Choose another parent folder.")
                };
                path
            }
        };
        Ok(json!({"type":"project_results","path":path}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn same_name_gets_separate_claims_and_verification_is_read_only() {
        let root = tempfile::tempdir().unwrap();
        let claim = |id: &str| {
            Results::Claim {
                parent: root.path().display().to_string(),
                name: "Scan".into(),
                project: id.into(),
            }
            .run()
            .unwrap()["path"]
                .as_str()
                .unwrap()
                .to_owned()
        };
        let first = claim("first");
        let second = claim("second");
        assert_ne!(first, second);
        assert_eq!(claim("first"), first);
        assert!(Path::new(&second).join("kept").is_dir());
        assert!(
            Results::Verify {
                path: first,
                project: "second".into()
            }
            .run()
            .is_err()
        );
    }
}
