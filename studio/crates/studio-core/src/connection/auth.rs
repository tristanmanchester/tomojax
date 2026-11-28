// Ported from T3 Code (MIT), packages/ssh/src/auth.ts at d2c44031. See THIRD_PARTY_NOTICES.md.
use std::{path::Path, process::Command};

pub(super) fn askpass(command: &mut Command, helper: Option<&Path>) {
    if let Some(helper) = helper {
        command
            .env("SSH_ASKPASS", helper)
            .env("SSH_ASKPASS_REQUIRE", "force")
            .env("TOMOJAX_STUDIO_ASKPASS", "1");
        if std::env::var_os("DISPLAY").is_none() {
            command.env("DISPLAY", "tomojax");
        }
    }
}

pub(super) fn refused(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    [
        "permission denied (",
        "authentication failed",
        "too many authentication failures",
        "host key verification failed",
        "remote host identification has changed",
    ]
    .iter()
    .any(|pattern| message.contains(pattern))
}

use super::{
    Failure,
    process::{own_group, terminate},
};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    process::{Child, ExitStatus, Stdio},
};

pub(super) struct Login {
    child: Child,
    diagnostics: File,
}
impl Login {
    pub fn start(worker: &Path, name: &str, helper: &Path) -> Result<Self, Failure> {
        let mut command = Command::new(worker);
        let diagnostics = tempfile::tempfile()?;
        command
            .args(["connect", name, "--login"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(diagnostics.try_clone()?);
        askpass(&mut command, Some(helper));
        let child = own_group(&mut command).spawn()?;
        Ok(Self { child, diagnostics })
    }

    pub fn poll(&mut self) -> Result<Option<(ExitStatus, String)>, Failure> {
        let Some(status) = self.child.try_wait()? else {
            return Ok(None);
        };
        let offset = self.diagnostics.metadata()?.len().saturating_sub(65536);
        self.diagnostics.seek(SeekFrom::Start(offset))?;
        let mut bytes = Vec::new();
        self.diagnostics
            .by_ref()
            .take(65536)
            .read_to_end(&mut bytes)?;
        Ok(Some((
            status,
            String::from_utf8_lossy(&bytes).trim().into(),
        )))
    }
}
impl Drop for Login {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = terminate(&mut self.child);
        }
    }
}
