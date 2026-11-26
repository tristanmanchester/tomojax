//! File work has its own cancellation and lifetime, independent of the JAX worker.
use super::*;
use std::io::{Read, Write};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::JoinHandle;

pub(super) const CANCELED: &str = "Canceled during file work";
pub(super) type Cancellation = Arc<AtomicBool>;

pub(super) struct LocalTask {
    pub job: Option<String>,
    pub canceled: Cancellation,
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
        let canceled = Arc::new(AtomicBool::new(false));
        let token = canceled.clone();
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
                canceled,
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
            task.canceled.store(true, Ordering::Release);
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
            task.canceled.store(true, Ordering::Release);
        }
        for (_, task) in self.local_tasks.drain() {
            let _ = task.thread.join();
        }
    }
}

pub(super) fn check_cancel(canceled: &AtomicBool) -> Result<(), String> {
    if canceled.load(Ordering::Acquire) {
        Err(CANCELED.into())
    } else {
        Ok(())
    }
}

/// Exclusive creation and bounded copying make cancellation leave only owned staging data.
pub(super) fn copy_file(
    source: &Path,
    destination: &Path,
    canceled: &AtomicBool,
) -> Result<(), String> {
    check_cancel(canceled)?;
    let mut input = std::fs::File::open(source).map_err(|e| e.to_string())?;
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)
        .map_err(|e| e.to_string())?;
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        check_cancel(canceled)?;
        let count = input.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        output
            .write_all(&buffer[..count])
            .map_err(|e| e.to_string())?;
    }
    check_cancel(canceled)?;
    output.sync_all().map_err(|e| e.to_string())
}
