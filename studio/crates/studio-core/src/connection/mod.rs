// Ported from T3 Code (MIT), packages/ssh/src/errors.ts at d2c44031. See THIRD_PARTY_NOTICES.md.
//! Local and nested-SSH transports for a TomoJAX session worker.
mod auth;
mod command;
mod config;
pub(crate) mod process;
mod registry;
pub use registry::Registry;
mod stream;
mod supervisor;
mod terminal;

pub use config::Profile;
use std::{fmt, io, path::Path, process::Command};
pub use supervisor::{Owner, State};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Status {
    Usage = 2,
    LoginNeeded = 3,
    Transient = 4,
    Blocked = 5,
    InnerRefusal = 6,
    Interrupted = 130,
}

#[derive(Debug)]
pub struct Failure {
    pub status: Status,
    pub message: String,
}
impl Failure {
    fn new(status: Status, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
    fn silent(message: impl Into<String>) -> Self {
        Self::new(Status::Transient, message)
    }
    fn blocked(message: impl Into<String>) -> Self {
        Self::new(Status::Blocked, message)
    }
    fn interrupted() -> Self {
        Self::new(Status::Interrupted, "Connection cancelled")
    }
    fn ssh(message: &str) -> Self {
        let status = if auth::refused(message) {
            Status::LoginNeeded
        } else {
            Status::Transient
        };
        Self::new(
            status,
            if message.is_empty() {
                "SSH disconnected before answering"
            } else {
                message
            },
        )
    }
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}
impl std::error::Error for Failure {}
impl From<io::Error> for Failure {
    fn from(e: io::Error) -> Self {
        Self::blocked(e.to_string())
    }
}

/// Entry point of the dedicated `tomojax-worker connect` process. The worker
/// installer supplies the executable path on the selected machine. Login
/// requires no installation. Serve inherits stdin and relays binary stdout.
pub fn connect(
    args: &[String],
    executable: impl FnOnce(&Profile) -> Result<String, Failure>,
    askpass: Option<&Path>,
) -> Result<i32, Failure> {
    let (name, option) = match args {
        [name] => (name, ""),
        [name, option] if matches!(option.as_str(), "--check" | "--login") => {
            (name, option.as_str())
        }
        _ => {
            return Err(Failure::new(
                Status::Usage,
                "Usage: tomojax-worker connect NAME [--login | --check]",
            ));
        }
    };
    process::handle_signals()?;
    let profile = Profile::load(name)?;
    let route = command::Route::resolve(&profile)?;
    if option == "--login" {
        if let Some(route) = route {
            route.login(askpass)?;
        }
        return Ok(0);
    }
    let program = executable(&profile)?;
    let mut command = if let Some(route) = &route {
        route.worker(&profile, &program)
    } else {
        let mut cmd = Command::new(program);
        cmd.args(profile.worker_args());
        cmd
    };
    match stream::session(&mut command, option == "--check", command::STARTUP) {
        Err(e) if let Some(route) = route => Err(route.classify(e, profile.hops().len())),
        other => other,
    }
}
