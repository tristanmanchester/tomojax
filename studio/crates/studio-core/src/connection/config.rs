use std::{collections::BTreeMap, env, fs, path::PathBuf};

use serde::{Deserialize, Serialize};
use std::io::Write;

use super::Failure;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Jobs {
    Inline,
    Process,
    Slurm,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Profile {
    ssh: Vec<String>,
    session: Option<String>,
    python: Option<String>,
    jobs: Option<Jobs>,
    sbatch: Option<String>,
    pub results: Option<String>,
}

impl Profile {
    pub fn load(name: &str) -> Result<Self, Failure> {
        if name == "local" {
            return Ok(Self {
                python: env::var("TOMOJAX_STUDIO_PYTHON").ok(),
                ..Self::default()
            });
        }
        let mut profiles = Self::catalog()?;
        let profile = profiles
            .remove(name)
            .ok_or_else(|| Failure::blocked(format!("Unknown connection {name:?}")))?;
        profile.validate()?;
        Ok(profile)
    }

    fn config_path() -> Result<PathBuf, Failure> {
        let root = env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
            .ok_or_else(|| Failure::blocked("No config directory for connections.toml"))?;
        Ok(root.join("tomojax/connections.toml"))
    }

    pub fn catalog() -> Result<BTreeMap<String, Self>, Failure> {
        let path = Self::config_path()?;
        let contents = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
            Err(e) => return Err(e.into()),
        };
        toml::from_str(&contents).map_err(|e| Failure::blocked(format!("{}: {e}", path.display())))
    }

    fn validate(&self) -> Result<(), Failure> {
        if self.ssh.len() > 2
            || self
                .ssh
                .iter()
                .any(|h| h.is_empty() || h.starts_with('-') || h.contains(['\0', '\n', '\r', ' ']))
        {
            return Err(Failure::blocked(
                "ssh needs zero, one or two SSH destinations",
            ));
        }
        Ok(())
    }

    /// Save one entry under a lock. An editor opened before another edit to
    /// this entry must reload it before replacing it.
    pub fn save(name: &str, prior: Option<&Profile>, entry: &Profile) -> Result<(), Failure> {
        if name.trim().is_empty() || name == "local" {
            return Err(Failure::blocked(
                "Choose a connection name other than local",
            ));
        }
        entry.validate()?;
        let path = Self::config_path()?;
        let parent = path
            .parent()
            .ok_or_else(|| Failure::blocked("Missing settings directory"))?;
        fs::create_dir_all(parent)?;
        let lock = fs::File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path.with_extension("lock"))?;
        lock.lock()?;
        let mut catalog = Self::catalog()?;
        if catalog.get(name) != prior {
            return Err(Failure::blocked(
                "This connection has changed. Reopen it before saving.",
            ));
        }
        catalog.insert(name.to_owned(), entry.clone());
        let text = toml::to_string_pretty(&catalog).map_err(|e| Failure::blocked(e.to_string()))?;
        let mut stage = tempfile::NamedTempFile::new_in(parent)?;
        stage.write_all(text.as_bytes())?;
        stage.as_file().sync_all()?;
        stage.persist(&path).map_err(|e| e.error)?;
        Ok(())
    }

    pub fn hops(&self) -> &[String] {
        &self.ssh
    }

    pub(super) fn worker_args(&self) -> Vec<String> {
        let jobs = self.jobs.unwrap_or(if self.ssh.is_empty() {
            Jobs::Inline
        } else {
            Jobs::Process
        });
        let jobs = match jobs {
            Jobs::Inline => "inline",
            Jobs::Process => "process",
            Jobs::Slurm => "slurm",
        };
        let mut args = vec![
            "serve".into(),
            "--jobs".into(),
            jobs.into(),
            "--idle-exit".into(),
            "120".into(),
        ];
        for (flag, value) in [
            ("--sbatch", &self.sbatch),
            ("--session", &self.session),
            ("--results", &self.results),
            ("--python", &self.python),
        ] {
            if let Some(value) = value {
                args.extend([flag.to_owned(), value.clone()]);
            }
        }
        args
    }
}
