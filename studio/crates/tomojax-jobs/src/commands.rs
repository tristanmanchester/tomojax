//! Worker request payloads. The dispatcher supplies the ordinary protocol reply envelope.
use crate::service::{self, Service};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    io,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Command {
    Submit {
        directory: PathBuf,
        script: Vec<Value>,
        #[serde(default)]
        options: String,
    },
    Jobs {
        root: PathBuf,
    },
    Job {
        directory: PathBuf,
        #[serde(default)]
        offset: u64,
        #[serde(default = "page_size")]
        limit: u64,
    },
    JobControl {
        directory: PathBuf,
        action: String,
    },
}
fn page_size() -> u64 {
    65536
}

impl Command {
    pub fn execute(
        self,
        service: &Service,
        resolve: impl Fn(&Path) -> io::Result<PathBuf>,
    ) -> io::Result<Value> {
        let mut row = match self {
            Self::Submit {
                directory,
                script,
                options,
            } => {
                let args = shlex::split(&options)
                    .ok_or_else(|| io::Error::other("unclosed quote in scheduler options"))?;
                service.submit(&resolve(&directory)?, &script, &args)?
            }
            Self::Jobs { root } => {
                return Ok(json!({"type":"jobs","jobs":service::jobs(&resolve(&root)?)?}));
            }
            Self::Job {
                directory,
                offset,
                limit,
            } => service::job(&resolve(&directory)?, offset, limit)?,
            Self::JobControl { directory, action } => {
                service::control(&resolve(&directory)?, &action)?
            }
        };
        row["type"] = json!("job");
        Ok(row)
    }
}
