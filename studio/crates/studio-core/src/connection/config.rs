use std::{collections::BTreeMap, env, fs, path::PathBuf};

use serde::Deserialize;

use super::Failure;

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Jobs {
    Inline,
    Process,
    Slurm,
}

#[derive(Clone, Debug, Default, Deserialize)]
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
        let root = env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
            .ok_or_else(|| Failure::blocked("No config directory for connections.toml"))?;
        let path = root.join("tomojax/connections.toml");
        let contents = fs::read_to_string(&path)
            .map_err(|e| Failure::blocked(format!("{}: {e}", path.display())))?;
        let mut profiles: BTreeMap<String, Profile> = toml::from_str(&contents)
            .map_err(|e| Failure::blocked(format!("{}: {e}", path.display())))?;
        let profile = profiles
            .remove(name)
            .ok_or_else(|| Failure::blocked(format!("Unknown connection {name:?}")))?;
        if profile.ssh.len() > 2
            || profile
                .ssh
                .iter()
                .any(|h| h.is_empty() || h.starts_with('-') || h.contains(['\0', '\n', '\r', ' ']))
        {
            return Err(Failure::blocked(
                "ssh needs zero, one or two SSH destinations",
            ));
        }
        Ok(profile)
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
