//! Submission permission belongs to the action that captured the script. Opening
//! a manifest or reconnecting creates readers only.
use super::*;
use crate::project::{
    Job, JobStatus,
    receipt::{Published, Reading},
    script::Script,
};
use std::collections::{BTreeMap, HashSet};

#[derive(Default)]
pub(super) struct JobSession {
    pub waiting: HashMap<String, u64>,
    observed: HashSet<String>,
    reads: BTreeMap<String, Instant>,
    list_after: Option<Instant>,
    listed: bool,
}

pub(super) enum Pending {
    List,
    Submit(String),
    Read(String, u64),
    Inspect(String, u64, Box<Reading>),
    Control(String),
    Resume(String),
}

impl AppState {
    pub(crate) fn job_seen(&self, id: &str) -> bool {
        self.jobs.observed.contains(id)
    }

    pub(crate) fn launcher(&self) -> Option<&str> {
        match &self.engine.as_ref()?.ready()?.peer {
            crate::protocol::Peer::Worker { jobs, .. } => jobs.as_str(),
            _ => None,
        }
    }
    pub(crate) fn detached_jobs(&self) -> bool {
        matches!(self.launcher(), Some("process" | "slurm"))
    }

    pub(super) fn capture_jobs(&mut self, mut jobs: Vec<Job>) -> Result<Vec<String>, String> {
        let options = if self.launcher() == Some("slurm") {
            self.ui.scheduler_options.text.clone()
        } else {
            String::new()
        };
        let project = self.project.as_mut().ok_or("Open a project first")?;
        for job in &mut jobs {
            if job.operation != "export_array" {
                job.script = Some(Script::build(
                    job,
                    self.live.enabled(),
                    if job.session_operation() {
                        String::new()
                    } else {
                        options.clone()
                    },
                    None,
                )?);
            }
        }
        let mut ids = Vec::new();
        for mut job in jobs {
            if matches!(job.operation.as_str(), "open" | "inspect")
                && let Some(input) = job.inputs.first()
                && let Some(asset) = project.document.asset_mut(&input.asset)
                && let Some(script) = &mut job.script
            {
                asset.intent += 1;
                script.intent = Some(asset.intent);
            }
            self.jobs
                .waiting
                .insert(job.id.clone(), project.generation + 1);
            ids.push(job.id.clone());
            project.document.jobs.push(job);
        }
        self.mark_dirty();
        Ok(ids)
    }

    pub(super) fn schedule(&mut self) {
        let Some(launcher) = self.launcher().map(str::to_owned) else {
            return;
        };
        let Some(project) = &self.project else { return };
        if !self.jobs.listed {
            if self.jobs.list_after.is_some_and(|t| t > Instant::now())
                || self.pending.values().any(|p| matches!(p, Pending::List))
            {
                return;
            }
            let root = project.data_root().to_owned();
            if let Err(e) = self.send(json!({"type":"jobs","root":root}), Pending::List) {
                self.job_request_failed(Pending::List, e);
            }
            return;
        }
        let now = Instant::now();
        let due: Vec<_> = self
            .jobs
            .reads
            .iter()
            .filter(|(_, t)| **t <= now)
            .map(|(id, _)| id.clone())
            .collect();
        for id in due {
            self.jobs.reads.remove(&id);
            let Some(p) = &self.project else { return };
            let Some(script) = p
                .document
                .jobs
                .iter()
                .find(|j| j.id == id)
                .and_then(|j| j.script.as_ref())
            else {
                continue;
            };
            let offset = script.offset;
            let command = json!({"type":"job","directory":p.job_directory(&id),"offset":offset});
            if let Err(error) = self.send(command, Pending::Read(id.clone(), offset)) {
                self.job_request_failed(Pending::Read(id, offset), error);
            }
        }
        let Some(p) = &self.project else { return };
        if p.save_error.is_some() {
            return;
        }
        let busy = p.document.jobs.iter().any(|j| {
            j.script.is_some()
                && j.status.active()
                && !self.jobs.waiting.contains_key(&j.id)
                && (launcher == "inline" || j.session_operation())
        });
        let jobs: Vec<_> = p
            .document
            .jobs
            .iter()
            .filter(|j| {
                self.jobs
                    .waiting
                    .get(&j.id)
                    .is_some_and(|g| *g <= p.saved_generation)
            })
            .cloned()
            .collect();
        let mut inline_busy =
            busy || self.previews.has_priority_work() || !self.model.backend.ready;
        for job in jobs {
            let inline = launcher == "inline" || job.session_operation();
            if inline && inline_busy && job.operation != "export_array" {
                continue;
            }
            self.jobs.waiting.remove(&job.id);
            if job.operation == "export_array" {
                if let Some(j) = self
                    .project
                    .as_mut()
                    .and_then(|p| p.document.job_mut(&job.id))
                {
                    j.status = JobStatus::Running;
                    j.started = Some(project::now_ms());
                }
                self.export_array(&job);
                continue;
            }
            let Some(script) = &job.script else { continue };
            let directory = self.project.as_ref().map(|p| p.job_directory(&job.id));
            let command = json!({"type":"submit","directory":directory,"script":script.lines,"options":script.options});
            if let Err(error) = self.send(command, Pending::Submit(job.id.clone())) {
                self.fail_job(&job.id, &error, false);
            } else if inline {
                inline_busy = true;
            }
        }
    }

    pub(super) fn job_reply(&mut self, pending: Pending, result: ResultValue) {
        match (pending, result) {
            (Pending::List, ResultValue::Jobs { jobs }) => {
                self.jobs.listed = true;
                let Some(p) = &self.project else { return };
                let listed: std::collections::BTreeSet<_> =
                    jobs.iter().map(|r| &r.directory).collect();
                for j in &p.document.jobs {
                    if j.script.is_some()
                        && !self.jobs.waiting.contains_key(&j.id)
                        && (listed.contains(&p.job_directory(&j.id))
                            || j.status != JobStatus::Completed)
                    {
                        self.jobs.reads.insert(j.id.clone(), Instant::now());
                    }
                }
            }
            (Pending::Submit(id), ResultValue::Job(_)) => {
                self.jobs.reads.insert(id, Instant::now());
            }
            (Pending::Resume(id), ResultValue::Job(row)) => {
                self.jobs.reads.insert(id.clone(), Instant::now());
                if let Err(error) = self.resume_checkpoint(&id, &row) {
                    self.model.error = Some(error);
                }
            }
            (Pending::Control(id), ResultValue::Job(_)) => {
                if let Some(t) = self.jobs.reads.get_mut(&id) {
                    *t = Instant::now();
                }
            }
            (Pending::Read(id, offset), ResultValue::Job(row)) => {
                let input = self.project.as_ref().and_then(|p| {
                    let j = p
                        .document
                        .jobs
                        .iter()
                        .find(|j| j.id == id && j.status != JobStatus::Completed)?;
                    let input = j.inputs.first()?;
                    let asset = p.document.asset(&input.asset)?;
                    (j.script.as_ref()?.intent == Some(asset.intent)).then_some(input.path.clone())
                });
                if let (Some(path), Some(Published::Report { report })) =
                    (input, row.receipt.as_ref().map(|r| &r.output))
                    && let Some(report) = report.get("report")
                {
                    let command = json!({"type":"stored","path":path,"inspection":report});
                    if let Err(error) =
                        self.send(command, Pending::Inspect(id.clone(), offset, row))
                    {
                        self.read_error(id, error);
                    }
                } else {
                    self.adopt_page(id, offset, row, None);
                }
            }
            (Pending::Inspect(id, offset, row), ResultValue::Stored { kind, arrays, .. }) => {
                self.adopt_page(id, offset, row, kind.map(|k| (k, arrays)))
            }
            (pending, _) => self.job_request_failed(pending, "Unexpected worker job reply".into()),
        }
    }

    fn adopt_page(
        &mut self,
        id: String,
        offset: u64,
        row: Box<Reading>,
        inspected: Option<(String, BTreeMap<String, crate::protocol::ArrayDescriptor>)>,
    ) {
        let selected = self.selected_inputs();
        let Some(p) = &mut self.project else { return };
        let activate = p
            .document
            .jobs
            .iter()
            .find(|j| j.id == id)
            .is_some_and(|j| j.inputs.iter().map(|i| &i.asset).eq(selected.iter()))
            && self.preview_context.active.is_none();
        let focus = self.ui.selected_job.as_ref().or_else(|| {
            p.document
                .jobs
                .iter()
                .rev()
                .find(|j| j.status.active())
                .or_else(|| p.document.jobs.last())
                .map(|j| &j.id)
        });
        let focused = focus == Some(&id);
        let registration = p
            .document
            .jobs
            .iter()
            .find(|j| j.id == id && j.status != JobStatus::Completed)
            .and_then(|j| {
                if !matches!(
                    row.receipt.as_ref()?.output,
                    Published::Source {} | Published::Report { .. }
                ) {
                    return None;
                }
                let a = p.document.asset(&j.inputs.first()?.asset)?;
                (j.script.as_ref()?.intent == Some(a.intent))
                    .then(|| (a.id.clone(), a.kind.clone(), a.arrays.is_empty()))
            });
        let adoption = match p.apply_job(&id, offset, &row, inspected) {
            Ok(added) => added,
            Err(e) => {
                self.read_error(id, e);
                return;
            }
        };
        self.jobs.observed.insert(id.clone());
        if let Some(interval) = next_read(&row, focused) {
            self.jobs
                .reads
                .insert(id.clone(), Instant::now() + interval);
        }
        if focused
            && let Some(job) = p.document.jobs.iter().find(|j| j.id == id)
            && let Some(preview) = &row.preview
        {
            self.live.accept_published(job, preview);
        }
        if let Some(job) = p.document.job_mut(&id)
            && job.status.active()
            && job.script.as_ref().is_some_and(|s| s.cancel_pending)
            && !self
                .pending
                .values()
                .any(|r| matches!(r,Pending::Control(target) if target == &id))
        {
            let command =
                json!({"type":"job_control","action":"cancel","directory":p.job_directory(&id)});
            if let Err(error) = self.send(command, Pending::Control(id.clone())) {
                self.model.error = Some(error);
            }
        }
        if let Some((asset, previous, first)) = registration {
            self.refresh_registration(&asset, &previous, first);
        }
        if let Some(asset) = adoption.asset {
            if activate {
                self.model.selected_assets = [asset.clone()].into();
                self.activate(asset.clone());
            }
            if matches!(
                row.receipt.as_ref().map(|r| &r.output),
                Some(Published::Dataset { .. })
            ) {
                self.start_metadata(&asset, false);
            }
        }
        if adoption.changed {
            self.mark_dirty();
        }
    }

    fn read_error(&mut self, id: String, error: String) {
        self.jobs
            .reads
            .insert(id, Instant::now() + Duration::from_secs(2));
        self.model.error = Some(error);
    }

    pub(super) fn job_request_failed(&mut self, pending: Pending, error: String) {
        match pending {
            Pending::List => {
                self.jobs.list_after = Some(Instant::now() + Duration::from_secs(2));
                self.model.error = Some(error);
            }
            Pending::Submit(id) => {
                self.fail_job(&id, &error, false);
                self.jobs.reads.insert(id, Instant::now());
            }
            Pending::Inspect(id, ..) | Pending::Read(id, ..) => self.read_error(id, error),
            Pending::Control(_) | Pending::Resume(_) => self.model.error = Some(error),
        }
    }

    pub(super) fn control_job(&mut self, id: &str, action: &str) -> Result<(), String> {
        let p = self.project.as_ref().ok_or("Open a project first")?;
        let j = p
            .document
            .jobs
            .iter()
            .find(|j| j.id == id)
            .ok_or("Job not found")?;
        if action == "resume" && matches!(j.status, JobStatus::Canceled | JobStatus::Interrupted) {
            let script = j.script.as_ref().ok_or("This job has no checkpoint")?;
            if !self
                .pending
                .values()
                .any(|r| matches!(r,Pending::Resume(target) if target==id))
            {
                self.send(
                    json!({"type":"job","directory":p.job_directory(id),"offset":script.offset}),
                    Pending::Resume(id.into()),
                )?;
            }
            return Ok(());
        }
        if action == "cancel" && self.jobs.waiting.remove(id).is_some() {
            if let Some(job) = self.project.as_mut().and_then(|p| p.document.job_mut(id)) {
                job.script = None;
            }
            self.fail_job(id, "Canceled before submission", true);
            return Ok(());
        }
        if self.local_job(id) {
            if action != "cancel" {
                return Err("This file operation can only be cancelled".into());
            }
            self.cancel_local(id);
            return Ok(());
        }
        if j.script.is_none() {
            return Err("This job has no worker directory".into());
        }
        let directory = p.job_directory(id);
        if action == "cancel" {
            if let Some(script) = self
                .project
                .as_mut()
                .and_then(|p| p.document.job_mut(id))
                .and_then(|j| j.script.as_mut())
            {
                script.cancel_pending = true;
            }
            self.mark_dirty();
            if self
                .pending
                .values()
                .any(|r| matches!(r, Pending::Submit(submitted) if submitted == id))
            {
                return Ok(());
            }
        }
        self.send(
            json!({"type":"job_control","directory":directory,"action":action}),
            Pending::Control(id.into()),
        )?;
        Ok(())
    }
    fn resume_checkpoint(&mut self, id: &str, row: &Reading) -> Result<(), String> {
        let p = self.project.as_mut().ok_or("Open a project first")?;
        if row.directory != p.job_directory(id)
            || !matches!(row.state.as_str(), "canceled" | "interrupted")
        {
            return Err("The job is no longer interrupted or canceled".into());
        }
        let checkpoint = row
            .checkpoint
            .as_ref()
            .filter(|c| c.usable)
            .ok_or("No usable checkpoint")?;
        if p.document.jobs.iter().any(|j| {
            j.status.active() && j.parameters["checkpoint"].as_str() == Some(&checkpoint.path)
        }) {
            return Err("A job is already using this checkpoint".into());
        }
        let j = p
            .document
            .jobs
            .iter()
            .find(|j| j.id == id)
            .ok_or("Job no longer in project")?;
        let mut job = Job::new(&j.operation, j.parameters.clone(), j.inputs.clone());
        let mut script = Script::build(
            &job,
            self.live.enabled(),
            j.script
                .as_ref()
                .map(|s| s.options.clone())
                .unwrap_or_default(),
            None,
        )?;
        let operation = script
            .lines
            .get_mut(checkpoint.line)
            .and_then(|l| l.pointer_mut("/command/operation"))
            .ok_or("Checkpoint line has no operation")?;
        operation["checkpoint"] = json!(checkpoint.path);
        job.parameters["checkpoint"] = json!(checkpoint.path);
        job.script = Some(script);
        self.jobs.waiting.insert(job.id.clone(), p.generation + 1);
        self.ui.activity.follow(vec![job.id.clone()]);
        p.document.jobs.push(job);
        self.mark_dirty();
        Ok(())
    }
}

// A scheduler can finish before a network filesystem exposes the receipt. Recheck
// interrupted jobs and outstanding cancellations even after their reply log drains.
fn next_read(row: &Reading, focused: bool) -> Option<Duration> {
    if row.records.is_empty() {
        match row.state.as_str() {
            "completed" | "failed" => return None,
            "canceled" | "interrupted" if row.checkpoint.as_ref().is_some_and(|c| c.usable) => {
                return Some(Duration::from_secs(5));
            }
            "canceled" if !row.abandoned => return None,
            "interrupted" if row.launcher.as_deref() == Some("inline") => return None,
            "interrupted" | "not_launched" | "canceled" => return Some(Duration::from_secs(5)),
            _ => {}
        }
    }
    Some(if focused {
        Duration::from_millis(250)
    } else {
        Duration::from_secs(2)
    })
}

#[cfg(test)]
#[path = "durable_tests.rs"]
mod tests;
