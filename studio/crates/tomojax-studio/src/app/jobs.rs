use super::*;
use crate::project::{Job, JobStatus, new_id, now_ms};
use crate::protocol::{Progress, Storage};

#[cfg(test)]
#[path = "jobs_tests.rs"]
mod tests;

impl AppState {
    /// The first durable action for an interactive preview. Ownership moves from
    /// its channel to the copy job; the worker manifest remains unchanged.
    pub(super) fn keep_preview(
        &mut self,
        preview: crate::previews::KeptPreview,
    ) -> Result<String, String> {
        let Some(project) = self.project.as_mut().filter(|_| !self.shutting_down) else {
            self.retire_output(&preview.output);
            return Err("Open a project before keeping the preview".into());
        };
        let mut job = Job::new(&preview.operation, preview.parameters, vec![preview.input]);
        job.telemetry = preview.telemetry;
        job.status = JobStatus::Running;
        job.started = Some(now_ms());
        job.stage = "Keeping interactive preview".into();
        job.kept_preview = Some(crate::project::KeptPreviewContext {
            request: preview.request,
            revision: preview.revision,
        });
        let id = job.id.clone();
        project.document.jobs.push(job);
        self.mark_dirty();
        self.copy_kept_preview(&id, preview.output);
        self.refresh_model();
        Ok(id)
    }

    fn retire_output(&mut self, output: &Output) {
        let artifacts = output
            .value
            .arrays()
            .into_iter()
            .chain(output.info.arrays.values())
            .flat_map(|array| array.artifacts().map(str::to_owned))
            .collect();
        self.previews
            .retire_resources(artifacts, output.value.handle().map(str::to_owned));
    }

    pub(super) fn queue_operation(&mut self, operation: &str) -> Result<(), String> {
        if matches!(
            operation,
            "slice" | "center_sweep" | "correction_preview" | "region_preview"
        ) {
            return self.request_interactive(operation);
        }
        if !self.model.backend.ready && !self.detached_jobs() {
            return Err("Wait for TomoJAX to connect".into());
        }
        self.poll_form_context();
        if let Some(error) = self.current_validation_error(operation) {
            return Err(error);
        }
        let parameters = forms::parameters(&self.ui, &self.catalog)?;
        let selected = self.selected_inputs();
        if operation != "simulate" && selected.is_empty() {
            return Err("Select a scan or result first".into());
        }
        let inputs = if operation == "simulate" {
            vec![]
        } else {
            self.current_form_inputs().to_vec()
        };
        // Results are saved as datasets, which hold scans, not raw frames.
        if matches!(operation, "crop" | "select")
            && inputs.iter().any(|input| input.kind == "frames")
        {
            return Err(
                "Raw frames can't be saved after cropping or selecting views. Run Corrections first, then crop or select views of the corrected scan.".into(),
            );
        }
        let jobs = if matches!(
            operation,
            "simulate" | "combine" | "project" | "backproject"
        ) {
            vec![Job::new(operation, parameters, inputs)]
        } else {
            inputs
                .into_iter()
                .map(|input| Job::new(operation, parameters.clone(), vec![input]))
                .collect()
        };
        let queued = self.capture_jobs(jobs)?;
        self.ui.activity.follow(queued);
        self.model.status = "Submitted settings are captured with each job".into();
        self.refresh_model();
        Ok(())
    }

    pub(super) fn result_asset(
        &self,
        job_id: &str,
        id: &str,
        path: &Path,
        output: &Output,
    ) -> Result<Asset, String> {
        let project = self
            .project
            .as_ref()
            .ok_or("The result's project is closed")?;
        let job = project
            .document
            .jobs
            .iter()
            .find(|j| j.id == job_id)
            .ok_or("The result's job no longer exists")?;
        let mut asset = Asset::at(project.stored_path(path), true, false);
        asset.id = id.into();
        asset.kind = output.value.kind().into();
        asset.parent = job.inputs.first().map(|input| input.asset.clone());
        asset.simulated = job.operation == "simulate"
            || job.inputs.iter().any(|i| {
                project
                    .document
                    .asset(&i.asset)
                    .is_some_and(|a| a.simulated)
            });
        asset.method = if let Product::Reconstruction { method, .. } = &output.value {
            Some(method.clone())
        } else {
            None
        };
        asset.name = job.result_name(&job.subject(), asset.method.as_deref());
        asset.provenance = Some(output.manifest.clone());
        Ok(asset)
    }

    fn copy_kept_preview(&mut self, job: &str, output: Box<Output>) {
        let id = new_id("result");
        let Some(project) = &self.project else {
            self.retire_output(&output);
            return;
        };
        let derived = project.data_root().join("kept");
        let destination = derived.join(&id);
        let path = destination.join("array.npy");
        let mut asset = match self.result_asset(job, &id, &path, &output) {
            Ok(asset) => asset,
            Err(error) => {
                self.retire_output(&output);
                self.fail_job(job, &error, false);
                return;
            }
        };
        let mut copies = Vec::new();
        let mut artifacts: Vec<String> = output
            .info
            .arrays
            .values()
            .flat_map(|array| array.artifacts().map(str::to_owned))
            .collect();
        let arrays = match &output.value {
            Product::CorrectionPreview {
                before,
                after,
                views,
            } => vec![("before", before), ("after", after), ("views", views)],
            Product::Sweep { planes } => vec![("sweep", planes)],
            Product::Plane { plane } => vec![("plane", plane)],
            Product::Projections { projections } => vec![("projections", projections)],
            Product::Volume { volume } => vec![("volume", volume)],
            Product::Region {
                arrays,
                grid,
                bounds,
                ..
            } => {
                asset.inspection = json!({"grid":grid,"bounds":bounds});
                arrays
                    .iter()
                    .map(|(key, value)| (key.as_str(), value))
                    .collect()
            }
            _ => {
                self.retire_output(&output);
                self.fail_job(job, "This output has no persistable array", false);
                return;
            }
        };
        for (key, array) in arrays {
            let mut durable = array.clone();
            if let Storage::Npy {
                path: source,
                artifact,
            } = &array.storage
            {
                let filename = format!("{key}.npy");
                let path = destination.join(&filename);
                copies.push(json!({"path":source,"name":filename,"artifact":artifact}));
                durable.storage = Storage::Npy {
                    path: path.to_string_lossy().into(),
                    artifact: None,
                };
                if let Some(artifact) = artifact {
                    artifacts.push(artifact.clone());
                }
                if copies.len() == 1 {
                    asset.path = project.stored_path(&path);
                }
            } else {
                self.retire_output(&output);
                self.fail_job(
                    job,
                    "This array's storage can't be copied into a project",
                    false,
                );
                return;
            }
            // Coordinate-vector artifacts must outlive the session too.
            for (axis_index, axis) in durable.axes.iter_mut().enumerate() {
                if let crate::protocol::Coordinates::Coordinates {
                    values:
                        Storage::Npy {
                            path: source,
                            artifact,
                        },
                } = &axis.coordinates
                {
                    let filename = format!("{key}-axis{axis_index}.npy");
                    let path = destination.join(&filename);
                    copies.push(json!({"path":source,"name":filename,"artifact":artifact}));
                    if let Some(artifact) = artifact {
                        artifacts.push(artifact.clone());
                    }
                    axis.coordinates = crate::protocol::Coordinates::Coordinates {
                        values: Storage::Npy {
                            path: path.to_string_lossy().into(),
                            artifact: None,
                        },
                    };
                }
            }
            asset.arrays.insert(key.into(), durable);
        }
        artifacts.sort();
        artifacts.dedup();
        let handle = output.value.handle().map(str::to_owned);
        if let Some(job) = self.project.as_mut().and_then(|p| p.document.job_mut(job)) {
            job.destination = Some(destination.clone().into());
            job.stage = "Copying preview arrays into project".into();
            job.progress = None;
        }
        self.mark_dirty();
        let job = job.to_owned();
        let files = self.file_reads();
        self.spawn_local(Some(job.clone()), move |canceled| {
            let result = local::copy(files, &destination, copies, json!({}), &canceled);
            LocalEvent::Persisted {
                job,
                asset,
                artifacts,
                handle,
                result,
            }
        });
    }

    pub(super) fn complete_asset(&mut self, job: &str, mut asset: Asset) {
        // A background result must not retarget an interactive form whose source
        // and latest intent are already captured. An explicit Keep is the user's
        // request to activate that newly durable result.
        // Nor may it replace a selection the user has moved on to: the result
        // becomes active only while its own inputs are still the selection.
        let entry = self
            .project
            .as_ref()
            .and_then(|project| project.document.jobs.iter().find(|entry| entry.id == job));
        let kept = entry.is_some_and(|entry| entry.kept_preview.is_some());
        let still_selected = entry.is_some_and(|entry| {
            let inputs: Vec<_> = entry
                .inputs
                .iter()
                .map(|input| input.asset.clone())
                .collect();
            self.selected_inputs() == inputs
        });
        let activate = kept || (self.preview_context.active.is_none() && still_selected);
        if let Some(project) = &self.project {
            for array in asset.arrays.values_mut() {
                let store = |storage: &mut Storage| {
                    if let Storage::Hdf5 { path, .. } | Storage::Npy { path, .. } = storage {
                        *path = project.stored_path(Path::new(path)).to_string();
                    }
                };
                store(&mut array.storage);
                for axis in &mut array.axes {
                    if let crate::protocol::Coordinates::Coordinates { values } =
                        &mut axis.coordinates
                    {
                        store(values);
                    }
                }
            }
        }
        let id = asset.id.clone();
        if let Some(project) = &mut self.project {
            project.document.assets.push(asset);
        }
        if !activate {
            self.complete_job(job, Some(id));
            self.refresh_model();
            return;
        }
        self.model.selected_assets = [id.clone()].into();
        self.activate(id.clone());
        self.complete_job(job, Some(id));
        self.refresh_model();
    }

    pub(super) fn complete_job(&mut self, id: &str, result: Option<String>) {
        if let Some(project) = &mut self.project
            && let Some(job) = project.document.job_mut(id)
        {
            job.status = JobStatus::Completed;
            job.finished = Some(now_ms());
            job.result = result;
            job.progress = Some(1.);
            self.model.status = format!("{} complete", job.display_operation());
            if !matches!(job.operation.as_str(), "plan") {
                job.stage = "Complete".into();
            }
        }
        self.mark_dirty();
    }

    pub(super) fn fail_job(&mut self, id: &str, error: &str, canceled: bool) {
        if let Some(project) = &mut self.project
            && let Some(job) = project.document.job_mut(id)
        {
            job.status = if canceled {
                JobStatus::Canceled
            } else {
                JobStatus::Failed
            };
            job.error = Some(error.into());
            job.stage = if canceled { "Canceled" } else { "Failed" }.into();
            job.finished = Some(now_ms());
            self.model.status = format!(
                "{} {}",
                job.name,
                if canceled { "canceled" } else { "failed" }
            );
        }
        self.mark_dirty();
    }

    pub(super) fn release_response(&mut self, response: &Response) {
        if let Response::Progress {
            record: Progress::Preview { arrays, .. },
            ..
        } = response
        {
            self.previews.retire_resources(
                arrays
                    .values()
                    .flat_map(|a| a.artifacts().map(str::to_owned))
                    .collect(),
                None,
            );
        }
        if let Response::Result {
            result: ResultValue::Output(output),
            ..
        } = response
        {
            self.release_output(output);
        }
    }

    pub(super) fn release_output(&mut self, output: &Output) {
        if let Some(handle) = output.value.handle() {
            self.release_handle(handle);
        } else {
            self.retire_output(output);
        }
    }

    fn release_handle(&mut self, handle: &str) {
        self.previews
            .retire_resources(Default::default(), Some(handle.into()));
    }
}

impl JobView {
    pub(super) fn from_record(p: &Project, job: &Job, waiting: bool) -> Self {
        let script = job.script.as_ref();
        let state_label = if job.status == JobStatus::Queued && waiting {
            "Waiting to submit"
        } else if job.status == JobStatus::Queued && script.is_some_and(|s| s.seen.is_none()) {
            "Awaiting confirmation"
        } else {
            job.status.title()
        };
        Self {
            id: job.id.clone(),
            name: job
                .result
                .as_deref()
                .and_then(|id| p.document.asset(id))
                .map(|a| a.name.clone())
                .unwrap_or_else(|| job.subject()),
            inputs: job
                .inputs
                .iter()
                .map(|input| input.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            operation: job.display_operation(),
            status: job.status,
            state_label,
            stage: job.stage.clone(),
            progress: job.progress,
            error: job.error.clone(),
            settings: job
                .parameters
                .as_object()
                .into_iter()
                .flatten()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        v.as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| v.to_string()),
                    )
                })
                .collect(),
            scheduler_options: script.map(|s| s.options.clone()).unwrap_or_default(),
            devices: script.map(|s| s.devices.join(", ")).unwrap_or_default(),
            elapsed: job.started.map(|start| {
                let seconds =
                    job.finished.unwrap_or_else(now_ms).saturating_sub(start) as f64 / 1000.;
                let prefix = if job.kept_preview.is_some() {
                    "save "
                } else {
                    ""
                };
                format!("{prefix}{seconds:.1} s")
            }),
            result_asset: job.result.clone(),
            can_pause: matches!(job.status, JobStatus::Running | JobStatus::Paused)
                && script.and_then(|s| s.launcher.as_deref()) == Some("inline"),
            pause_requested: script.is_some_and(|s| s.pause_requested),
            can_resume: matches!(job.status, JobStatus::Interrupted | JobStatus::Canceled)
                && script
                    .and_then(|s| s.checkpoint.as_ref())
                    .is_some_and(|c| c.usable),
            waiting,
        }
    }
}
