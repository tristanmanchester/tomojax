//! File work has its own cancellation and lifetime, independent of the JAX worker.
use super::*;
use std::io::{Read, Write};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::JoinHandle;

pub(super) const CANCELLED: &str = "Cancelled during file work";
pub(super) type Cancellation = Arc<AtomicBool>;

pub(super) struct LocalTask {
    pub job: Option<String>,
    pub cancelled: Cancellation,
    pub thread: JoinHandle<()>,
}

pub(super) struct LocalMessage {
    pub task: String,
    pub event: LocalEvent,
}

impl AppState {
    pub(super) fn spawn_local(
        &mut self,
        job: Option<String>,
        work: impl FnOnce(Cancellation) -> LocalEvent + Send + 'static,
    ) {
        let task = project::new_id("io");
        let cancelled = Arc::new(AtomicBool::new(false));
        let token = cancelled.clone();
        let send = self.local_send.clone();
        let wake = self.wake.clone();
        let id = task.clone();
        let thread = std::thread::spawn(move || {
            let event = work(token);
            let _ = send.send(LocalMessage { task: id, event });
            wake();
        });
        self.local_tasks.insert(
            task,
            LocalTask {
                job,
                cancelled,
                thread,
            },
        );
    }

    pub(super) fn finish_local(&mut self, id: &str) {
        if let Some(task) = self.local_tasks.remove(id) {
            let _ = task.thread.join();
        }
    }

    pub(super) fn cancel_local(&mut self, job: &str) -> bool {
        let mut found = false;
        for task in self
            .local_tasks
            .values()
            .filter(|task| task.job.as_deref() == Some(job))
        {
            task.cancelled.store(true, Ordering::Release);
            found = true;
        }
        found
    }

    pub(super) fn local_job(&self, job: &str) -> bool {
        self.local_tasks
            .values()
            .any(|task| task.job.as_deref() == Some(job))
    }

    pub(super) fn stop_local_tasks(&mut self) {
        for task in self.local_tasks.values() {
            task.cancelled.store(true, Ordering::Release);
        }
        for (_, task) in self.local_tasks.drain() {
            let _ = task.thread.join();
        }
    }
}

pub(super) fn check_cancel(cancelled: &AtomicBool) -> Result<(), String> {
    if cancelled.load(Ordering::Acquire) {
        Err(CANCELLED.into())
    } else {
        Ok(())
    }
}

/// Exclusive creation and bounded copying make cancellation leave only owned staging data.
pub(super) fn copy_file(
    source: &Path,
    destination: &Path,
    cancelled: &AtomicBool,
) -> Result<(), String> {
    check_cancel(cancelled)?;
    let mut input = std::fs::File::open(source).map_err(|e| e.to_string())?;
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .map_err(|e| e.to_string())?;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        check_cancel(cancelled)?;
        let count = input.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        output
            .write_all(&buffer[..count])
            .map_err(|e| e.to_string())?;
    }
    check_cancel(cancelled)?;
    output.sync_all().map_err(|e| e.to_string())
}
