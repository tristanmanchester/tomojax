// Ported from T3 Code (MIT, © 2026 T3 Tools Inc.), apps/server/src/attachmentStore.ts at d2c44031; see THIRD_PARTY_NOTICES.md.
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use tempfile::{Builder, NamedTempFile};

pub const LIMIT: u64 = 100 << 20;
const PREFIX: &str = ".tomojax-part-";

pub fn check_size(size: u64) -> Result<()> {
    ensure!(
        size <= LIMIT,
        "file is {size} bytes, limit is {LIMIT} bytes"
    );
    Ok(())
}

fn sweep(folder: &Path) -> Result<()> {
    for entry in fs::read_dir(folder)? {
        let entry = entry?;
        if entry
            .file_name()
            .as_encoded_bytes()
            .starts_with(PREFIX.as_bytes())
            && entry.file_type()?.is_file()
            && entry
                .metadata()
                .and_then(|m| m.modified())
                .ok()
                .and_then(|time| time.elapsed().ok())
                .is_some_and(|age| age > Duration::from_secs(3600))
        {
            match fs::remove_file(entry.path()) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.into()),
                _ => {}
            }
        }
    }
    Ok(())
}

pub struct Upload {
    stage: NamedTempFile,
    destination: PathBuf,
    left: u64,
    expected: Option<String>,
    hash: Sha256,
}

impl Upload {
    pub fn start(destination: PathBuf, bytes: u64, expected: String) -> Result<Self> {
        check_size(bytes)?;
        ensure!(
            expected.len() == 64 && expected.bytes().all(|b| b.is_ascii_hexdigit()),
            "invalid SHA-256 digest"
        );
        Self::stage(destination, bytes, Some(expected))
    }

    pub fn download(destination: PathBuf, bytes: u64) -> Result<Self> {
        check_size(bytes)?;
        Self::stage(destination, bytes, None)
    }

    fn stage(destination: PathBuf, bytes: u64, expected: Option<String>) -> Result<Self> {
        ensure!(!destination.try_exists()?, "destination already exists");
        let parent = destination.parent().context("missing destination folder")?;
        sweep(parent)?;
        Ok(Self {
            stage: Builder::new().prefix(PREFIX).tempfile_in(parent)?,
            destination,
            left: bytes,
            expected,
            hash: Sha256::new(),
        })
    }

    pub fn write(&mut self, block: &[u8]) -> Result<()> {
        ensure!(
            block.len() as u64 <= self.left,
            "upload exceeds declared size"
        );
        self.stage.write_all(block)?;
        self.hash.update(block);
        self.left -= block.len() as u64;
        Ok(())
    }

    pub fn complete(&self) -> bool {
        self.left == 0
    }

    pub fn finish_download(mut self, bytes: u64, digest: String) -> Result<()> {
        ensure!(
            bytes == self.stage.as_file().metadata()?.len(),
            "download size differs"
        );
        self.expected = Some(digest);
        self.publish()
    }

    pub fn publish(self) -> Result<()> {
        ensure!(self.complete(), "upload ended before its declared size");
        ensure!(
            format!("{:x}", self.hash.finalize()).eq_ignore_ascii_case(
                self.expected
                    .as_deref()
                    .context("missing transfer digest")?
            ),
            "upload SHA-256 differs"
        );
        self.stage.as_file().sync_all()?;
        fs::hard_link(self.stage.path(), self.destination)?;
        Ok(())
    }
}

/// The open descriptor fixes the source identity. Read one extra byte to detect growth.
pub struct Download {
    file: File,
    pub size: u64,
    pub left: u64,
    hash: Sha256,
}

impl Download {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        ensure!(metadata.is_file(), "download requires a regular file");
        let size = metadata.len();
        check_size(size)?;
        Ok(Self {
            file,
            size,
            left: size,
            hash: Sha256::new(),
        })
    }

    pub fn read(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut bytes = vec![0; n.min(self.left as usize)];
        self.file.read_exact(&mut bytes)?;
        self.left -= bytes.len() as u64;
        self.hash.update(&bytes);
        if self.left == 0 {
            ensure!(
                self.file.read(&mut [0])? == 0,
                "download grew past its declared size"
            );
        }
        Ok(bytes)
    }

    pub fn digest(&self) -> String {
        format!("{:x}", self.hash.clone().finalize())
    }
}
