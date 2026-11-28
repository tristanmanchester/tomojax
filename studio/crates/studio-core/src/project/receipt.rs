//! A completed job, its read cursor and its project edits share one manifest write.
use super::{Asset, Job, JobStatus, Project, now_ms};
use crate::protocol::{ArrayDescriptor, Progress, Provenance, Storage};
use crate::worker_path::WorkerPath;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

type Arrays = BTreeMap<String, ArrayDescriptor>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Reading {
    pub directory: WorkerPath,
    pub launcher: Option<String>,
    pub state: String,
    pub detail: String,
    pub cancel_requested: bool,
    #[serde(default)]
    pub pause_requested: bool,
    #[serde(default)]
    pub abandoned: bool,
    pub checkpoint: Option<super::script::Checkpoint>,
    pub receipt: Option<Receipt>,
    #[serde(default)]
    pub records: Vec<Value>,
    pub offset: Option<u64>,
    pub preview: Option<Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Receipt {
    pub product: String,
    pub manifest: Provenance,
    pub method: Option<String>,
    #[serde(flatten)]
    pub output: Published,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Published {
    Dataset {
        path: WorkerPath,
        arrays: BTreeMap<String, ArrayDescriptor>,
    },
    Arrays {
        arrays: BTreeMap<String, ArrayDescriptor>,
    },
    Report {
        report: Value,
    },
    Source {},
    Files {
        paths: Vec<WorkerPath>,
    },
}

#[derive(Default)]
pub struct Adoption {
    pub asset: Option<String>,
    pub changed: bool,
}

impl Project {
    pub fn job_directory(&self, id: &str) -> WorkerPath {
        self.data_root().join("jobs").join(id).into()
    }

    /// Applies one ordered page. The caller schedules a single snapshot after this
    /// returns. A failed read leaves the previous cursor available for retry.
    pub fn apply_job(
        &mut self,
        id: &str,
        offset: u64,
        row: &Reading,
        inspected: Option<(String, BTreeMap<String, ArrayDescriptor>)>,
    ) -> Result<Adoption, String> {
        if self.job_directory(id) != row.directory {
            return Err("The worker returned a different job directory".into());
        }
        let root = self.data_root().to_owned();
        let doc = &mut self.document;
        let job = doc
            .jobs
            .iter_mut()
            .find(|j| j.id == id)
            .ok_or("Job no longer in project")?;
        let script = job.script.as_ref().ok_or("Job has no script")?;
        if script.offset != offset
            || (row.state == "not_launched" && job.status == JobStatus::Failed)
        {
            return Ok(Adoption::default());
        }
        let mut stage = row.detail.clone();
        let status = match row.state.as_str() {
            "queued" => JobStatus::Queued,
            "running" => JobStatus::Running,
            "unknown" => JobStatus::Unknown,
            "failed" => JobStatus::Failed,
            "completed" => JobStatus::Completed,
            "canceled" => JobStatus::Canceled,
            "not_launched" => {
                stage = "No launch recorded".into();
                JobStatus::Interrupted
            }
            "interrupted" => JobStatus::Interrupted,
            _ => return Err(format!("Unknown worker job state {}", row.state)),
        };
        let added = if status == JobStatus::Completed {
            if job.status == JobStatus::Completed {
                stage = job.stage.clone();
                None
            } else {
                let receipt = row.receipt.as_ref().ok_or("Completed job has no receipt")?;
                receipt.adopt(job, &mut doc.assets, &root, inspected, &mut stage)?
            }
        } else {
            None
        };
        let changed = row.observe(job, status, stage)?;
        Ok(Adoption {
            asset: added,
            changed,
        })
    }
}

impl Receipt {
    fn adopt(
        &self,
        job: &mut Job,
        assets: &mut Vec<Asset>,
        root: &Path,
        inspected: Option<(String, Arrays)>,
        stage: &mut String,
    ) -> Result<Option<String>, String> {
        match &self.output {
            Published::Dataset { path, arrays } => {
                Ok(self.register(job, assets, root, path.lexical(), arrays))
            }
            Published::Arrays { arrays } => {
                let path = arrays
                    .values()
                    .find_map(|a| match &a.storage {
                        Storage::Npy { path, .. } | Storage::Hdf5 { path, .. } => {
                            Some(Path::new(path))
                        }
                        _ => None,
                    })
                    .ok_or("Array receipt has no file storage")?;
                Ok(self.register(job, assets, root, path, arrays))
            }
            Published::Files { paths } => {
                job.outputs.clone_from(paths);
                Ok(None)
            }
            Published::Report { report } if job.operation == "plan" => {
                *stage = serde_json::to_string_pretty(report).map_err(|e| e.to_string())?;
                Ok(None)
            }
            Published::Report { report } => {
                if let Some(asset) = current_source(job, assets)? {
                    if let Some((kind, arrays)) = inspected {
                        asset.register_read(&kind, arrays, root);
                    }
                    asset.inspection = report["report"].clone();
                    asset.issue = asset.inspection["issues"]
                        .as_array()
                        .filter(|v| !v.is_empty())
                        .map(|v| {
                            v.iter()
                                .filter_map(Value::as_str)
                                .collect::<Vec<_>>()
                                .join("; ")
                        });
                }
                Ok(None)
            }
            Published::Source {} => {
                if let Some(asset) = current_source(job, assets)? {
                    asset.kind.clone_from(&self.product);
                    asset.open_parameters.clone_from(&job.parameters);
                    asset.provenance = Some(self.manifest.clone());
                    asset.issue = None;
                    job.result = Some(asset.id.clone());
                }
                Ok(None)
            }
        }
    }

    fn register(
        &self,
        job: &mut Job,
        assets: &mut Vec<Asset>,
        root: &Path,
        path: &Path,
        arrays: &Arrays,
    ) -> Option<String> {
        let id = format!("result-{}", job.id);
        job.result = Some(id.clone());
        if assets.iter().any(|a| a.id == id) {
            return None;
        }
        let mut asset = Asset::at(path.strip_prefix(root).unwrap_or(path).into(), true, false);
        asset.id.clone_from(&id);
        asset.kind.clone_from(&self.product);
        asset.method.clone_from(&self.method);
        asset.name = job.result_name(&job.subject(), self.method.as_deref());
        asset.parent = job
            .script
            .as_ref()
            .and_then(|s| s.roles.as_ref())
            .map_or(Some(0), |r| r.parent())
            .and_then(|i| job.inputs.get(i))
            .map(|i| i.asset.clone());
        asset.simulated = job.operation == "simulate"
            || job
                .inputs
                .iter()
                .any(|i| assets.iter().any(|a| a.id == i.asset && a.simulated));
        asset.provenance = Some(self.manifest.clone());
        asset.arrays.clone_from(arrays);
        for a in asset.arrays.values_mut() {
            a.relative_to(root);
        }
        assets.push(asset);
        Some(id)
    }
}

fn current_source<'a>(job: &Job, assets: &'a mut [Asset]) -> Result<Option<&'a mut Asset>, String> {
    let input = job.inputs.first().ok_or("Receipt needs a source asset")?;
    let intent = job.script.as_ref().and_then(|s| s.intent);
    Ok(assets
        .iter_mut()
        .find(|a| a.id == input.asset && Some(a.intent) == intent))
}

impl Reading {
    fn observe(
        &self,
        job: &mut Job,
        mut status: JobStatus,
        mut stage: String,
    ) -> Result<bool, String> {
        let script = job.script.as_mut().ok_or("Job has no script")?;
        let mut paused = job.status == JobStatus::Paused;
        for record in &self.records {
            match record["type"].as_str() {
                Some("attempt") => {
                    job.telemetry = Default::default();
                    paused = false;
                    script.devices.clear();
                }
                Some("ready") => {
                    script.devices = record["devices"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|d| d["name"].as_str().map(str::to_owned))
                        .collect();
                }
                Some("progress") => {
                    if let (Some(request), Ok(progress)) = (
                        record["request"].as_str(),
                        serde_json::from_value::<Progress>(record["record"].clone()),
                    ) {
                        match progress {
                            Progress::Paused { .. } => paused = true,
                            Progress::Resumed { .. } => paused = false,
                            _ => {}
                        }
                        job.telemetry.apply(request, &progress);
                    }
                }
                _ => {}
            }
        }
        if status == JobStatus::Running && paused {
            status = JobStatus::Paused;
        }
        if (self.cancel_requested || script.cancel_pending) && status.active() {
            stage = "Cancelling".into();
        } else if self.pause_requested && status == JobStatus::Running {
            stage = "Pause requested".into();
        } else if status == JobStatus::Running
            && let Some(task) = job.telemetry.current()
        {
            stage = if self.detail.is_empty() {
                task.title().into()
            } else {
                format!("{} · {}", task.title(), self.detail)
            };
        }
        if status == JobStatus::Running || status == JobStatus::Paused {
            job.started.get_or_insert_with(now_ms);
        }
        job.error = (matches!(
            status,
            JobStatus::Failed | JobStatus::Unknown | JobStatus::Interrupted
        ) && !stage.is_empty())
        .then(|| stage.clone());
        if !status.active() {
            job.finished.get_or_insert_with(now_ms);
        } else {
            job.finished = None;
        }
        let changed = !self.records.is_empty()
            || job.status != status
            || job.stage != stage
            || script.offset != self.offset.unwrap_or(script.offset)
            || script.checkpoint != self.checkpoint
            || script.launcher != self.launcher
            || script.pause_requested != self.pause_requested
            || script.seen.is_none();
        job.status = status;
        job.stage = stage;
        job.progress = if status == JobStatus::Completed {
            Some(1.)
        } else {
            job.telemetry.current().and_then(|t| t.fraction())
        };
        if self.cancel_requested || !status.active() && self.state != "not_launched" {
            script.cancel_pending = false;
        }
        script.pause_requested = self.pause_requested;
        script.offset = self.offset.unwrap_or(script.offset);
        script.seen = Some(now_ms());
        script.checkpoint = self.checkpoint.clone();
        script.launcher.clone_from(&self.launcher);
        Ok(changed)
    }
}
