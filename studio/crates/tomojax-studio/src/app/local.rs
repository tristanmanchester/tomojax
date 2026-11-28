//! File work has its own cancellation and lifetime, independent of the JAX worker.
use super::*;
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
    pub(super) fn finish_audit(
        &mut self,
        asset: String,
        job: Option<String>,
        report: Value,
        result: Result<Option<HeaderMetadata>, String>,
    ) {
        let read = result
            .and_then(|header| header.ok_or("No stored metadata".into()))
            .map(|(kind, arrays, _)| (kind, arrays));
        if let Some(job) = job {
            if let Some(project) = &mut self.project {
                let root = project.data_root().to_owned();
                if let Some(asset) = project.document.asset_mut(&asset) {
                    if !asset.is_result()
                        && let Ok((_, arrays)) = &read
                    {
                        if let Some(mut projections) = arrays.get("projections").cloned() {
                            projections.relative_to(&root);
                            asset.arrays.insert("projections".into(), projections);
                        } else {
                            asset.arrays.remove("projections");
                        }
                    }
                    asset.issue = report["issues"]
                        .as_array()
                        .filter(|v| !v.is_empty())
                        .map(|v| {
                            v.iter()
                                .map(|s| s.as_str().unwrap_or("Inspection issue"))
                                .collect::<Vec<_>>()
                                .join("; ")
                        });
                    asset.inspection = report;
                }
            }
            self.complete_job(&job, None);
            self.image_key.clear();
            self.refresh_images();
        } else {
            self.record_read(&asset, read, report);
        }
    }

    pub(super) fn file_reads(&mut self) -> Option<studio_core::engine::FileReads> {
        if self.engine.is_none() {
            self.start_engine();
        }
        self.engine.as_ref().map(Engine::files)
    }

    pub(super) fn audit_stored(
        &mut self,
        asset: String,
        path: PathBuf,
        report: Value,
        job: Option<String>,
    ) {
        let files = self.file_reads();
        self.spawn_local(job.clone(), move |_| {
            let result = stored(files, &path, Some(&report));
            LocalEvent::Audited {
                asset,
                job,
                report,
                result,
            }
        });
    }

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

pub(super) fn copy(
    files: Option<studio_core::engine::FileReads>,
    destination: &Path,
    sources: Vec<Value>,
    documents: Value,
    canceled: &AtomicBool,
) -> Result<(), String> {
    check_cancel(canceled)?;
    let result = files.ok_or("The file worker is disconnected")?.copy(
        json!({"type":"copy","destination":destination,"sources":sources,"documents":documents}),
        canceled,
    );
    result.map_err(|error| {
        if canceled.load(Ordering::Acquire) {
            CANCELED.into()
        } else {
            error
        }
    })
}

pub(super) type HeaderMetadata = (
    String,
    std::collections::BTreeMap<String, crate::protocol::ArrayDescriptor>,
    Value,
);

pub(super) fn stored(
    files: Option<studio_core::engine::FileReads>,
    path: &Path,
    inspection: Option<&Value>,
) -> Result<Option<HeaderMetadata>, String> {
    let files = files.ok_or("The file worker is disconnected")?;
    let mut command = json!({"type":"stored","path":path});
    if let Some(report) = inspection {
        command["inspection"] = report.clone();
    }
    let ResultValue::Stored {
        kind,
        arrays,
        report,
    } = files.request(command)?
    else {
        return Err("Expected stored metadata".into());
    };
    Ok(kind.map(|k| (k, arrays, report)))
}

/// Copies a chosen input into `inputs` and returns the copy to open. A header
/// file brings its folder, whose frames it reads. An HDF5 file that reads data
/// from other files can't be copied: its links would still reach the originals.
pub(super) fn copy_acquisition(
    files: Option<studio_core::engine::FileReads>,
    path: &Path,
    inputs: &Path,
    canceled: &std::sync::atomic::AtomicBool,
) -> Result<PathBuf, String> {
    let name = path.file_name().ok_or("Missing acquisition name")?;
    let source = if path.is_file() && !project::holds_whole_acquisition(path) {
        path.parent().ok_or("Missing acquisition folder")?
    } else {
        path
    };
    let source_name = source
        .file_name()
        .ok_or("Missing acquisition folder name")?;
    let destination = inputs;
    local::copy(
        files,
        destination,
        vec![json!({"path":source,"name":source_name.to_string_lossy()})],
        json!({}),
        canceled,
    )?;
    let copy = destination.join(source_name);
    Ok(if source == path {
        copy
    } else {
        copy.join(name)
    })
}
