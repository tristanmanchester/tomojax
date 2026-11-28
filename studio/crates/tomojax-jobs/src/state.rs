//! Deriving a job's state has no side effects. A read linearizes at the launcher lookup.
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum Launch {
    #[default]
    Unclaimed,
    Claimed,
    Accepted(u64),
    Rejected,
    Prevented,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Completed,
    Failed,
    Stopped,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct Files {
    pub launch: Launch,
    pub receipt: bool,
    pub cancel: bool,
    pub abandoned: bool,
    pub withdrawn: bool,
    pub runner: bool,
    /// Monotonic attempt number. A hard kill can leave the preceding attempt open.
    pub attempt: u64,
    pub end: Option<Outcome>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Account {
    Offline,
    Pending,
    Allocated,
    LocalLockHeld,
    Absent {
        accounted: bool,
    },
    /// The launcher reports a successful exit, which the runner gives only after publishing.
    Finished,
}

impl Account {
    pub fn live(self) -> bool {
        matches!(self, Self::Pending | Self::Allocated | Self::LocalLockHeld)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    NotLaunched,
    Queued,
    Running,
    Unknown,
    Completed,
    Failed,
    Canceled,
    Interrupted,
}

pub fn state(files: Files, account: Account) -> State {
    if files.receipt {
        return State::Completed;
    }
    match files.launch {
        Launch::Unclaimed if !files.cancel => return State::NotLaunched,
        Launch::Unclaimed | Launch::Prevented => return State::Canceled,
        Launch::Rejected => return State::Failed,
        Launch::Claimed | Launch::Accepted(_) => {}
    }

    let ended = files.runner || files.attempt > 0 || files.withdrawn;
    match account {
        Account::Pending => State::Queued,
        Account::LocalLockHeld if files.attempt == 0 => State::Queued,
        Account::LocalLockHeld | Account::Allocated => State::Running,
        // Finished, and its result not yet visible here: a network filesystem can show
        // the runner's files late.
        Account::Finished => State::Running,
        Account::Absent { accounted: true } => ended_state(files),
        _ if files.abandoned && !ended => State::Canceled,
        Account::Absent { .. } if ended => ended_state(files),
        _ => State::Unknown,
    }
}

fn ended_state(files: Files) -> State {
    match files.end {
        Some(Outcome::Failed) => State::Failed,
        // The runner records completion only after publishing its result.
        Some(Outcome::Completed) => State::Running,
        _ if files.cancel => State::Canceled,
        _ => State::Interrupted,
    }
}

/// Only equal observations bracket a state that existed at the launcher lookup.
/// Every field changes monotonically within one attempt.
pub fn observed(before: Files, account: Account, after: Files) -> Option<State> {
    (before == after).then(|| state(after, account))
}

/// The caller scopes both the queue and accounting queries to the recorded user and cluster.
pub fn matches_allocation(name: &str, launch: Launch, job_name: &str, id: u64) -> bool {
    name == job_name && !matches!(launch, Launch::Accepted(recorded) if recorded != id)
}

/// A standing cancel includes a job rediscovered after its submitter died.
pub fn needs_withdrawal(files: Files, account: Account) -> bool {
    files.cancel && account.live()
}
