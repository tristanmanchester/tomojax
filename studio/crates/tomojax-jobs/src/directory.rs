//! Observations of durable job records. A killed writer may leave one incomplete log line.
use crate::state::{Files, Launch, Outcome};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File},
    io::{self, BufRead, BufReader, Write},
    path::Path,
};
use tempfile::NamedTempFile;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Submission {
    Accepted(u64),
    Rejected(String),
    Prevented,
}

/// Publish a complete record, including the initial launch claim, without replacing a winner.
pub fn create_record(path: &Path, value: &impl Serialize) -> io::Result<bool> {
    let mut file = NamedTempFile::new_in(path.parent().expect("record has a directory"))?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    match file.persist_noclobber(path) {
        Ok(_) => Ok(true),
        Err(error) if error.error.kind() == io::ErrorKind::AlreadyExists => Ok(false),
        Err(error) => Err(error.error),
    }
}

pub fn optional<T>(result: io::Result<T>) -> io::Result<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

pub fn read_files(dir: &Path) -> io::Result<Files> {
    let launch = match optional(fs::read(dir.join("submission.json")))? {
        Some(bytes) => match serde_json::from_slice(&bytes)? {
            Submission::Accepted(id) => Launch::Accepted(id),
            Submission::Rejected(_) => Launch::Rejected,
            Submission::Prevented => Launch::Prevented,
        },
        None if dir.join("launch").try_exists()? => Launch::Claimed,
        None => Launch::Unclaimed,
    };
    let (attempt, end) = last_attempt(dir)?;
    Ok(Files {
        launch,
        receipt: dir.join("result/receipt.json").try_exists()?,
        cancel: dir.join("cancel").try_exists()?,
        abandoned: dir.join("abandoned").try_exists()?,
        withdrawn: dir.join("withdrawn").try_exists()?,
        runner: dir.join("runner.json").try_exists()?,
        attempt,
        end,
    })
}

fn last_attempt(dir: &Path) -> io::Result<(u64, Option<Outcome>)> {
    #[derive(Deserialize)]
    #[serde(tag = "type", rename_all = "lowercase")]
    enum Record {
        Attempt {
            number: u64,
        },
        End {
            outcome: Outcome,
        },
        #[serde(other)]
        Other,
    }
    let mut attempt = (0, None);
    if let Some(file) = optional(File::open(dir.join("replies.jsonl")))? {
        let mut reader = BufReader::new(file);
        let mut line = Vec::new();
        while reader.read_until(b'\n', &mut line)? > 0 && line.ends_with(b"\n") {
            match serde_json::from_slice(&line)? {
                Record::Attempt { number } => attempt = (number, None),
                Record::End { outcome } => attempt.1 = Some(outcome),
                Record::Other => {}
            }
            line.clear();
        }
    }
    Ok(attempt)
}
