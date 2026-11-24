use super::*;
use crate::project::JobStatus;

#[derive(Clone)]
pub(super) struct ControlWatch {
    job: String,
    target: String,
    action: String,
    next_check: Option<Instant>,
}

impl AppState {
    pub(super) fn control_job(&mut self, id: &str, action: &str) -> Result<(), String> {
        if self.local_job(id) {
            if action != "cancel" {
                return Err(
                    "File work can be cancelled; pause becomes available at a numerical checkpoint"
                        .into(),
                );
            }
            self.cancel_local(id);
            if let Some(job) = self.project.as_mut().and_then(|p| p.document.job_mut(id)) {
                job.stage = "Cancelling file work".into();
            }
            return Ok(());
        }
        let target = self
            .pending
            .iter()
            .find_map(|(request, pending)| match pending {
                Pending::Execute(job)
                | Pending::OpenInput { job, .. }
                | Pending::Save { job, .. }
                    if job == id =>
                {
                    Some(request.clone())
                }
                _ => None,
            })
            .ok_or("The job is not executing in the worker")?;
        let engine = self.engine.as_mut().ok_or("Worker not connected")?;
        let request = engine.control(&target, action)?;
        self.pending.insert(
            request,
            Pending::Control {
                job: id.into(),
                target: target.clone(),
                action: action.into(),
            },
        );
        self.control_watch = if action == "cancel" {
            None
        } else {
            Some(ControlWatch {
                job: id.into(),
                target,
                action: action.into(),
                next_check: None,
            })
        };
        if let Some(job) = self.project.as_mut().and_then(|p| p.document.job_mut(id)) {
            job.stage = format!("{action} requested; waiting for a safe point");
        }
        Ok(())
    }

    pub(super) fn poll_control(&mut self) {
        let Some(watch) = self.control_watch.clone() else {
            return;
        };
        if self.shutting_down || !self.pending.contains_key(&watch.target) {
            self.control_watch = None;
            return;
        }
        if watch.next_check.is_none_or(|due|due>Instant::now()) || self.pending.values().any(|pending|matches!(pending,Pending::ControlStatus { target,.. } if target==&watch.target)) {
            return;
        }
        if let Err(error) = self.send(
            json!({"type":"status"}),
            Pending::ControlStatus {
                job: watch.job.clone(),
                target: watch.target.clone(),
                action: watch.action.clone(),
            },
        ) {
            self.control_failed(&watch.job, &watch.target, &watch.action, error);
        } else if let Some(watch) = &mut self.control_watch {
            watch.next_check = Some(Instant::now() + Duration::from_millis(250));
        }
    }

    pub(super) fn control_result(&mut self, pending: Pending, result: ResultValue) {
        let (job, target, action, checking) = match pending {
            Pending::Control {
                job,
                target,
                action,
            } => (job, target, action, false),
            Pending::ControlStatus {
                job,
                target,
                action,
            } => (job, target, action, true),
            _ => return,
        };
        if !self.pending.contains_key(&target)
            || !self.control_watch.as_ref().is_some_and(|watch| {
                watch.job == job && watch.target == target && watch.action == action
            })
        {
            return;
        }
        if !checking {
            if matches!(result, ResultValue::Ack) {
                // Ack confirms the control request, not arrival at a safe point.
                self.control_watch.as_mut().unwrap().next_check = Some(Instant::now());
            } else {
                self.control_failed(
                    &job,
                    &target,
                    &action,
                    "The worker returned an invalid control acknowledgement".into(),
                );
            }
            return;
        }
        let ResultValue::Status { requests, .. } = result else {
            self.control_failed(
                &job,
                &target,
                &action,
                "The worker returned an invalid status report".into(),
            );
            return;
        };
        let Some(state) = requests.get(&target) else {
            // The target may have committed and have its terminal in the pipe.
            self.control_watch = None;
            return;
        };
        let settled =
            (action == "pause" && state == "paused") || (action == "resume" && state == "running");
        let update = match state.as_str() {
            "paused" => Some((JobStatus::Paused, "Paused at a safe point")),
            "running" if action == "resume" => Some((JobStatus::Running, "Resumed")),
            "pause-requested" => Some((
                JobStatus::Running,
                "Pause requested; waiting for a safe point",
            )),
            "queued" | "running" => None,
            _ => {
                self.control_failed(
                    &job,
                    &target,
                    &action,
                    format!("Unknown worker request state: {state}"),
                );
                return;
            }
        };
        let mut changed = false;
        if let Some((status, stage)) = update
            && let Some(job) = self.project.as_mut().and_then(|p| p.document.job_mut(&job))
            && job.status.active()
            && (job.status != status || job.stage != stage)
        {
            job.status = status;
            job.stage = stage.into();
            changed = true;
        }
        if settled {
            self.control_watch = None;
        }
        if changed {
            self.mark_dirty();
        }
    }

    pub(super) fn control_failed(&mut self, job: &str, target: &str, action: &str, error: String) {
        if self.control_watch.as_ref().is_some_and(|watch| {
            watch.job == job && watch.target == target && watch.action == action
        }) {
            self.control_watch = None;
        }
        if self.pending.contains_key(target) {
            self.model.error = Some(error);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn control_acknowledgement_and_stale_status_do_not_invent_or_restore_a_pause() {
        let directory = tempfile::tempdir().unwrap();
        let mut app = AppState::init();
        let mut project = Project::open(&directory.path().join("project"), true).unwrap();
        let mut job = project::Job::new("reconstruct", json!({}), vec![]);
        job.status = JobStatus::Running;
        let id = job.id.clone();
        project.document.jobs.push(job);
        app.project = Some(project);
        app.pending
            .insert("target".into(), Pending::Execute(id.clone()));
        app.control_watch = Some(ControlWatch {
            job: id.clone(),
            target: "target".into(),
            action: "pause".into(),
            next_check: None,
        });
        app.control_result(
            Pending::Control {
                job: id.clone(),
                target: "target".into(),
                action: "pause".into(),
            },
            ResultValue::Ack,
        );
        assert_eq!(
            app.project
                .as_mut()
                .unwrap()
                .document
                .job_mut(&id)
                .unwrap()
                .status,
            JobStatus::Running
        );
        assert!(app.control_watch.as_ref().unwrap().next_check.is_some());
        let paused = ResultValue::Status {
            requests: BTreeMap::from([("target".into(), "paused".into())]),
            handles: BTreeMap::new(),
            allocators: Vec::new(),
        };
        app.control_result(
            Pending::ControlStatus {
                job: id.clone(),
                target: "target".into(),
                action: "pause".into(),
            },
            paused.clone(),
        );
        assert_eq!(
            app.project
                .as_mut()
                .unwrap()
                .document
                .job_mut(&id)
                .unwrap()
                .status,
            JobStatus::Paused
        );
        assert!(app.control_watch.is_none());
        app.control_watch = Some(ControlWatch {
            job: id.clone(),
            target: "target".into(),
            action: "resume".into(),
            next_check: None,
        });
        app.control_result(
            Pending::ControlStatus {
                job: id.clone(),
                target: "target".into(),
                action: "pause".into(),
            },
            paused.clone(),
        );
        assert_eq!(app.control_watch.as_ref().unwrap().action, "resume");
        let running = ResultValue::Status {
            requests: BTreeMap::from([("target".into(), "running".into())]),
            handles: BTreeMap::new(),
            allocators: Vec::new(),
        };
        app.control_result(
            Pending::ControlStatus {
                job: id.clone(),
                target: "target".into(),
                action: "resume".into(),
            },
            running,
        );
        assert_eq!(
            app.project
                .as_mut()
                .unwrap()
                .document
                .job_mut(&id)
                .unwrap()
                .status,
            JobStatus::Running
        );
        app.pending.remove("target");
        app.project
            .as_mut()
            .unwrap()
            .document
            .job_mut(&id)
            .unwrap()
            .status = JobStatus::Completed;
        app.control_result(
            Pending::ControlStatus {
                job: id.clone(),
                target: "target".into(),
                action: "pause".into(),
            },
            paused,
        );
        assert_eq!(
            app.project
                .as_mut()
                .unwrap()
                .document
                .job_mut(&id)
                .unwrap()
                .status,
            JobStatus::Completed
        );
    }
}
