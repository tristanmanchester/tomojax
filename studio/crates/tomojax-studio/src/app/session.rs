//! Studio owns one connection. An engine becomes available to project reads only
//! after the worker confirms the marker in the manifest's results folder.
use super::*;
use studio_core::connection::{Owner, Profile, State};
use studio_core::engine::WorkerCommand;

#[derive(Default)]
pub(super) enum Session {
    #[default]
    Closed,
    Local(Box<Engine>),
    Named {
        owner: Box<Owner>,
        verified: bool,
    },
}

impl Session {
    pub fn local(engine: Engine) -> Self {
        Self::Local(Box::new(engine))
    }
    pub fn named(name: &str, wake: rune::Waker) -> Result<Self, String> {
        Ok(Self::Named {
            owner: Box::new(Owner::new(
                WorkerCommand::binary()?,
                name.into(),
                &[
                    "fetch",
                    "project_results",
                    "list",
                    "copy",
                    "stored",
                    "submit",
                    "jobs",
                    "job",
                    "job_control",
                ],
                wake,
            )),
            verified: false,
        })
    }
    pub fn as_ref(&self) -> Option<&Engine> {
        match self {
            Self::Local(engine) => Some(engine),
            Self::Named {
                owner,
                verified: true,
            } => owner.engine(),
            _ => None,
        }
    }
    pub fn as_mut(&mut self) -> Option<&mut Engine> {
        match self {
            Self::Local(engine) => Some(engine),
            Self::Named {
                owner,
                verified: true,
            } => owner.engine_mut(),
            _ => None,
        }
    }
    pub fn is_none(&self) -> bool {
        matches!(self, Self::Closed)
    }
    pub fn poll(&mut self) -> Vec<EngineEvent> {
        match self {
            Self::Local(engine) => engine.poll(),
            Self::Named { owner, .. } => owner.poll(),
            Self::Closed => Vec::new(),
        }
    }
    pub fn owner(&self) -> Option<&Owner> {
        match self {
            Self::Named { owner, .. } => Some(owner),
            _ => None,
        }
    }
    pub fn owner_mut(&mut self) -> Option<&mut Owner> {
        match self {
            Self::Named { owner, .. } => Some(owner),
            _ => None,
        }
    }
}

impl AppState {
    pub(crate) fn connection_status(&self) -> String {
        self.engine.owner().map_or_else(
            || "This computer".into(),
            |owner| owner.status_text(Instant::now()),
        )
    }
    /// Why the project's connection cannot be made, as the session reported it.
    pub(crate) fn connection_problem(&self) -> Option<&str> {
        self.engine
            .owner()
            .filter(|owner| owner.state() == State::Blocked)
            .map(Owner::diagnostic)
            .filter(|message| !message.is_empty())
    }
    pub(crate) fn connection_state(&self) -> Option<State> {
        self.engine.owner().map(Owner::state)
    }
    pub(crate) fn data_offline(&self) -> bool {
        self.remote_project().is_some() && self.engine.as_ref().is_none()
    }
    pub(super) fn interrupt_file_jobs(&mut self, reason: &str) {
        let ids: Vec<_> = self
            .local_tasks
            .values()
            .filter_map(|task| task.job.clone())
            .collect();
        for id in ids {
            if self.project.as_ref().is_some_and(|p| {
                p.document
                    .jobs
                    .iter()
                    .any(|j| j.id == id && j.status.active())
            }) {
                self.fail_job(&id, reason, false);
            }
        }
    }

    pub(super) fn session_ready(&mut self) {
        let Some(owner) = self.engine.owner() else {
            return;
        };
        if !matches!(owner.state(), State::Connected(_)) {
            return;
        }
        let Some(files) = owner.engine().map(Engine::files) else {
            return;
        };
        let Some(project) = &self.project else { return };
        if let Some(path) = project.document.results.clone() {
            let id = project.document.id.clone();
            self.spawn_local(None, move |_| LocalEvent::Identity(files.request(json!({"type":"project_results","action":"verify","path":path,"project":id})).map(|_| ())));
        } else if let Err(error) = self.choose_results() {
            self.model.error = Some(error);
        }
    }

    pub(super) fn verified(&mut self, result: Result<(), String>) {
        let Session::Named { verified, owner } = &mut self.engine else {
            return;
        };
        if let Err(error) = result {
            self.imaging.invalidate();
            self.refresh_plane_model();
            self.files.names = Profile::catalog()
                .map(|catalog| catalog.into_keys().collect())
                .unwrap_or_default();
            self.files.creating = false;
            self.files.panel = files::Panel::Connections;
            self.model.error = Some(error);
            return;
        }
        *verified = true;
        let Some(engine) = owner.engine() else { return };
        self.imaging.invalidate();
        self.imaging.bind(engine);
        self.live.bind(engine.files());
        self.image_key.clear();
        self.refresh_images();
        let unread: Vec<_> = self
            .project
            .iter()
            .flat_map(|p| &p.document.assets)
            .filter(|a| a.arrays.is_empty())
            .map(|a| a.id.clone())
            .collect();
        for id in unread {
            self.start_metadata(&id, false);
        }
    }
}
