//! Application ownership: persistent project, one official worker, independent viewer.
mod actions;
mod controls;
mod editing;
mod jobs;
mod local;
mod previewing;
#[cfg(test)]
mod tests;
pub(crate) mod validation;
mod viewing;

use crate::{
    engine::{Engine, EngineEvent},
    forms,
    imaging::Imaging,
    metadata,
    preferences::Preferences,
    project::{self, Asset, Project},
    protocol::{Output, Product, Response, ResultValue, SchemaCatalog},
    ui::model::*,
};
use controls::ControlWatch;
use local::{CANCELLED, LocalMessage, LocalTask, check_cancel};
use rune::TextEdit;
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

#[derive(Clone)]
enum Pending {
    Inspect(String),
    OpenInput {
        job: String,
        index: usize,
    },
    Execute(String),
    Save {
        job: String,
        asset: String,
        path: PathBuf,
        output: Box<Output>,
    },
    Control {
        job: String,
        target: String,
        action: String,
    },
    ControlStatus {
        job: String,
        target: String,
        action: String,
    },
}

struct Running {
    job: String,
    handles: Vec<String>,
    products: Vec<Product>,
    next_input: usize,
}
enum LocalEvent {
    FormCaptured {
        generation: u64,
        result: Result<Vec<project::CapturedInput>, String>,
    },
    Copied {
        asset: Asset,
        result: Result<PathBuf, String>,
    },
    Persisted {
        job: String,
        asset: Asset,
        artifacts: Vec<String>,
        handle: Option<String>,
        result: Result<(), String>,
    },
    Metadata {
        asset: String,
        result: Result<Option<metadata::HeaderMetadata>, String>,
    },
    Exported {
        job: String,
        result: Result<(), String>,
    },
    StartChecked {
        job: String,
        result: Result<(), String>,
    },
    OutputChecked {
        pending: Pending,
        output: Box<Output>,
        result: Result<(), String>,
    },
}

/// Plane ids of the two compared images, the split fraction and both windows, as bits.
type SplitKey = ([u64; 2], u32, Vec<u32>);

pub struct AppState {
    pub ui: UiState,
    pub model: UiModel,
    pub project: Option<Project>,
    pub preferences: Preferences,
    pub imaging: Imaging,
    pub(crate) previews: crate::previews::Previews,
    pub(crate) live: crate::live::LiveCheckpoints,
    pub(crate) validation: validation::Controller,
    preview_context: previewing::PreviewContext,
    engine: Option<Engine>,
    pub(crate) catalog: SchemaCatalog,
    pending: HashMap<String, Pending>,
    control_watch: Option<ControlWatch>,
    imports: VecDeque<String>,
    running: Option<Running>,
    wake: rune::Waker,
    local_send: mpsc::Sender<LocalMessage>,
    local_receive: mpsc::Receiver<LocalMessage>,
    local_tasks: HashMap<String, LocalTask>,
    shutting_down: bool,
    copying: usize,
    metadata_pending: usize,
    started: bool,
    auto_demo: bool,
    startup_input: Option<PathBuf>,
    last_context: Value,
    difference: Option<([u64; 2], std::sync::Arc<crate::imaging::PlaneData>)>,
    /// The Split composite and the (left, right, curtain) it was built from.
    split: Option<(SplitKey, std::sync::Arc<crate::imaging::PlaneData>)>,
    /// Project names read once from each recent project's manifest.
    recent_names: std::collections::HashMap<PathBuf, String>,
    applied_window: Option<[f32; 2]>,
    window_prefill: Option<[String; 2]>,
    centre_sinogram: bool,
    form_context: editing::FormContext,
    last_autosave: Instant,
    view_history: Vec<project::Workspace>,
    view_future: Vec<project::Workspace>,
    image_key: String,
    last_interaction: Instant,
    last_diagnostic: String,
    interacting: bool,
}

impl AppState {
    pub fn init() -> Self {
        let catalog = SchemaCatalog::new(
            serde_json::from_str(include_str!("../../../src/tomojax/cli/parameters.json"))
                .expect("bundled protocol schema"),
        )
        .expect("schema definitions");
        let (local_send, local_receive) = mpsc::channel();
        let preference_result = Preferences::load();
        let preference_error = preference_result.as_ref().err().cloned();
        let mut state = Self {
            ui: UiState::default(),
            model: UiModel::default(),
            project: None,
            preferences: preference_result.unwrap_or_default(),
            imaging: Imaging::with_budget(512 * 1024 * 1024),
            previews: crate::previews::Previews::default(),
            live: crate::live::LiveCheckpoints::default(),
            validation: validation::Controller::default(),
            preview_context: previewing::PreviewContext::default(),
            engine: None,
            catalog,
            pending: HashMap::new(),
            control_watch: None,
            imports: VecDeque::new(),
            running: None,
            wake: std::sync::Arc::new(|| {}),
            local_send,
            local_receive,
            local_tasks: HashMap::new(),
            shutting_down: false,
            copying: 0,
            metadata_pending: 0,
            started: false,
            auto_demo: std::env::var_os("TOMOJAX_STUDIO_DEMO").is_some(),
            startup_input: std::env::var_os("TOMOJAX_STUDIO_INPUT").map(PathBuf::from),
            last_context: Value::Null,
            difference: None,
            split: None,
            recent_names: Default::default(),
            applied_window: None,
            window_prefill: Some([String::new(), String::new()]),
            centre_sinogram: true,
            form_context: editing::FormContext::default(),
            last_autosave: Instant::now(),
            view_history: Vec::new(),
            view_future: Vec::new(),
            image_key: String::new(),
            last_interaction: Instant::now() - Duration::from_secs(1),
            last_diagnostic: String::new(),
            interacting: false,
        };
        state.set_operation("reconstruct", None);
        state.model.error = preference_error;
        if let Some(path) = state.startup_input.clone()
            && (path.join(project::MANIFEST).is_file()
                || path
                    .file_name()
                    .is_some_and(|name| name == project::MANIFEST))
        {
            if let Err(error) = state.open_project(&path, false) {
                state.model.error = Some(error);
            } else {
                state.startup_input = None;
            }
        }
        state.refresh_model();
        // Headless captures still read the actual project arrays through the native
        // loader, with a bounded wait before Rune starts its virtual frame clock.
        if rune::tooling::is_headless() && state.project.is_some() {
            state.refresh_images();
            let until = Instant::now() + Duration::from_secs(15);
            while Instant::now() < until {
                state.imaging.poll();
                if [1, 2, 3, 4, 5, 8].iter().all(|slot| {
                    !matches!(
                        state.imaging.status(*slot),
                        crate::imaging::ImagingStatus::Loading
                    )
                }) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            state.refresh_plane_model();
        }
        state
    }

    pub fn setup(&mut self, wake: rune::Waker) {
        self.wake = wake.clone();
        self.imaging.set_waker(wake.clone());
        self.live.set_waker(wake.clone());
        if std::env::var_os("TOMOJAX_STUDIO_OFFLINE").is_some() || rune::tooling::is_headless() {
            self.model.backend.status = "Viewing saved project".into();
            return;
        }
        let python = std::env::var_os("TOMOJAX_STUDIO_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(if cfg!(windows) { "python" } else { "python3" }));
        match Engine::spawn(python, wake) {
            Ok(engine) => self.engine = Some(engine),
            Err(error) => self.model.error = Some(error),
        }
        self.model.backend.status = "Starting TomoJAX".into();
    }

    pub fn has_pending_work(&self) -> bool {
        self.running.is_some()
            || !self.pending.is_empty()
            || !self.local_tasks.is_empty()
            || self.previews.has_work()
            || self.live.loading()
            || self.validation_pending()
            || self.engine.as_ref().is_some_and(|e| e.ready().is_none())
    }

    pub fn advanced_parameter(&self, parameter: &ParameterView) -> bool {
        parameter.advanced
            || (self
                .ui
                .parameter_choices
                .get("method")
                .is_some_and(|method| method == "auto")
                && forms::method_setting(&self.ui, &parameter.name))
    }

    pub fn set_workspace(&mut self, workspace: Workspace) {
        self.ui.workspace = workspace;
        self.ui.operation_menu = false;
        self.ui.choice_menu = None;
        if matches!(workspace, Workspace::Prepare | Workspace::Reconstruct)
            && Workspace::for_operation(&self.ui.operation) != workspace
        {
            self.set_operation(
                if workspace == Workspace::Prepare {
                    "open"
                } else {
                    "reconstruct"
                },
                None,
            );
        }
    }

    pub fn set_operation(&mut self, operation: &str, values: Option<&Value>) {
        self.model.parameters = forms::initialise(&mut self.ui, &self.catalog, operation, values);
        if operation == "open"
            && values.is_none()
            && let Some(asset) = self
                .model
                .active_asset
                .as_ref()
                .and_then(|id| self.project.as_ref()?.document.asset(id))
        {
            self.ui
                .parameter_choices
                .insert("kind".into(), asset.kind.clone());
        }
    }

    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        let interacting = self.last_interaction.elapsed() < Duration::from_millis(160);
        if self.interacting != interacting {
            self.interacting = interacting;
            changed = true;
        }
        let events = self.engine.as_mut().map(Engine::poll).unwrap_or_default();
        for event in events {
            changed = true;
            self.engine_event(event);
        }
        self.poll_control();
        while let Ok(message) = self.local_receive.try_recv() {
            changed = true;
            self.finish_local(&message.task);
            self.local_event(message.event);
        }
        if let Some(project) = &mut self.project {
            changed |= project.poll();
        }
        if self.imaging.poll() {
            changed = true;
            self.refresh_plane_model();
        }
        changed |= self.live.poll();
        changed |= self.poll_form_context();
        changed |= self.poll_interactive();
        changed |= self.poll_validation();
        self.previews
            .retire_resources(self.live.drain_releases().into_iter().collect(), None);
        for handle in self.live.drain_handles() {
            self.previews
                .retire_resources(Default::default(), Some(handle));
        }
        if let Some(engine) = &mut self.engine {
            self.previews
                .collect_retired(engine, |tokens| self.imaging.artifacts_in_use(tokens));
        }
        if self.model.backend.ready && !self.shutting_down {
            if !self.started {
                self.started = true;
                if let Some(path) = self.startup_input.take() {
                    self.model.error = Some(format!(
                        "Create or open a project, then import {}",
                        path.display()
                    ));
                    self.startup_input = Some(path);
                }
            }
            if self.auto_demo && self.project.is_some() {
                self.auto_demo = false;
                self.set_operation("simulate", None);
                let _ = self
                    .queue_operation("simulate")
                    .map_err(|e| self.model.error = Some(e));
            }
            self.submit_imports();
            self.schedule();
        }
        if self.project.is_some() && self.last_autosave.elapsed() > Duration::from_millis(400) {
            self.last_autosave = Instant::now();
            let context = self.capture_workspace();
            let value = serde_json::to_value(&context).unwrap_or(Value::Null);
            if value != self.last_context {
                self.last_context = value;
                if let Some(project) = &mut self.project {
                    project.document.workspace = context;
                }
                self.mark_dirty();
                changed = true;
            }
        }
        if changed {
            self.refresh_model();
        }
        changed
    }

    fn local_event(&mut self, event: LocalEvent) {
        match event {
            LocalEvent::FormCaptured { generation, result } => {
                self.form_context.complete(generation, result);
            }
            LocalEvent::Copied { mut asset, result } => {
                self.copying = self.copying.saturating_sub(1);
                match result {
                    Ok(path) => {
                        if let Some(project) = &mut self.project {
                            asset.path = project.stored_path(&path);
                            asset.managed = true;
                            let id = asset.id.clone();
                            project.document.assets.push(asset);
                            self.model.active_asset = Some(id.clone());
                            self.model.selected_assets.clear();
                            self.model.selected_assets.insert(id.clone());
                            self.centre_sinogram = true;
                            self.mark_dirty();
                            self.start_metadata(&id);
                        }
                    }
                    Err(error) => self.model.error = Some(error),
                }
            }
            LocalEvent::Persisted {
                job,
                asset,
                artifacts,
                handle,
                result,
            } => {
                match result {
                    Ok(()) => self.complete_asset(&job, asset),
                    Err(error) => {
                        if !self.shutting_down {
                            self.fail_job(&job, &error, error == CANCELLED)
                        }
                    }
                }
                if !handle
                    .as_ref()
                    .is_some_and(|handle| self.live.retire_handle(handle))
                {
                    self.previews
                        .retire_resources(artifacts.into_iter().collect(), handle);
                }
            }
            LocalEvent::Exported { job, result } => match result {
                Ok(()) => self.complete_job(&job, None),
                Err(error) => {
                    if !self.shutting_down {
                        self.fail_job(&job, &error, error == CANCELLED)
                    }
                }
            },
            LocalEvent::StartChecked { job, result } => {
                if !self.shutting_down {
                    match result {
                        Ok(()) => self.advance_job(),
                        Err(error) => self.fail_job(&job, &error, error == CANCELLED),
                    }
                }
            }
            LocalEvent::OutputChecked {
                pending,
                output,
                result,
            } => match result {
                Ok(()) => self.result_checked(pending, output),
                Err(error) => {
                    self.release_output(&output);
                    if !self.shutting_down {
                        self.request_failed(pending, error.clone(), error == CANCELLED);
                    } else if let Pending::Save { output, .. } = pending {
                        self.release_output(&output);
                    }
                }
            },
            LocalEvent::Metadata { asset: id, result } => {
                self.metadata_pending = self.metadata_pending.saturating_sub(1);
                match result {
                    Ok(Some((kind, arrays, report))) => {
                        if let Some(project) = &mut self.project {
                            let root = project.root.clone();
                            if let Some(asset) = project.document.asset_mut(&id) {
                                if self.model.active_asset.as_ref() == Some(&id)
                                    && self.ui.operation == "open"
                                    && self.ui.parameter_choices.get("kind") == Some(&asset.kind)
                                {
                                    self.ui
                                        .parameter_choices
                                        .insert("kind".into(), kind.clone());
                                }
                                asset.kind = kind;
                                asset.inspection = report;
                                asset.issue = None;
                                asset.arrays.clear();
                                self.model
                                    .measurements
                                    .retain(|measurement| measurement.asset != id);
                                for (key, mut array) in arrays {
                                    let store = |storage: &mut crate::protocol::Storage| {
                                        if let crate::protocol::Storage::Hdf5 { path, .. }
                                        | crate::protocol::Storage::Npy { path, .. } = storage
                                            && let Ok(relative) =
                                                Path::new(path).strip_prefix(&root)
                                        {
                                            *path = relative.to_string_lossy().into();
                                        }
                                    };
                                    store(&mut array.storage);
                                    for axis in &mut array.axes {
                                        if let crate::protocol::Coordinates::Coordinates {
                                            values,
                                        } = &mut axis.coordinates
                                        {
                                            store(values);
                                        }
                                    }
                                    asset.arrays.insert(key, array);
                                }
                            }
                        }
                        self.imaging.invalidate();
                        self.mark_dirty();
                        self.refresh_images();
                    }
                    Ok(None) => self.imports.push_back(id),
                    Err(error) => {
                        if let Some(asset) = self
                            .project
                            .as_mut()
                            .and_then(|p| p.document.asset_mut(&id))
                        {
                            asset.issue = Some(error.clone());
                        }
                        self.model.error = Some(error);
                        self.mark_dirty();
                    }
                }
            }
        }
    }

    fn engine_event(&mut self, event: EngineEvent) {
        if self.validation_event(&event) {
            return;
        }
        match event {
            EngineEvent::Ready(ready) => {
                self.model.backend = BackendView {
                    ready: true,
                    device: ready
                        .devices
                        .iter()
                        .map(|d| d.name.clone())
                        .collect::<Vec<_>>()
                        .join(" · "),
                    status: format!("TomoJAX {}", ready.versions.tomojax),
                    capabilities: ready.capabilities,
                };
            }
            EngineEvent::Reply { response, stale } => {
                if let Response::Result {
                    request,
                    result: ResultValue::Output(output),
                    ..
                } = &response
                    && let Some(handle) = output.value.handle()
                {
                    self.live.bind_result(request, handle);
                }
                let Some(response) = self.previews.consume(response, stale) else {
                    return;
                };
                if stale {
                    self.release_response(&response);
                    return;
                }
                match response {
                    Response::Progress {
                        request, record, ..
                    } => self.progress(&request, record),
                    Response::Result {
                        request, result, ..
                    } => {
                        if let ResultValue::Description { parameters } = &result
                            && let Ok(catalog) = SchemaCatalog::new(parameters.clone())
                        {
                            self.catalog = catalog;
                            self.model.parameters =
                                forms::refresh_catalog(&mut self.ui, &self.catalog);
                        }
                        if let Some(pending) = self.pending.remove(&request) {
                            match pending {
                                Pending::Control { .. } | Pending::ControlStatus { .. } => {
                                    self.control_result(pending, result)
                                }
                                _ => self.result(pending, result),
                            }
                        }
                    }
                    Response::Error { request, error, .. } => {
                        if let Some(pending) = self.pending.remove(&request) {
                            self.request_failed(
                                pending,
                                error.message,
                                matches!(
                                    error.kind,
                                    crate::protocol::ErrorKind::Cancelled
                                        | crate::protocol::ErrorKind::Superseded
                                ),
                            );
                        }
                    }
                    Response::ProtocolError {
                        request, message, ..
                    } => {
                        if let Some(pending) = request.and_then(|id| self.pending.remove(&id)) {
                            self.request_failed(pending, message, false);
                        } else {
                            self.model.error = Some(message);
                        }
                    }
                    Response::Ready(_) => {}
                }
            }
            EngineEvent::Diagnostic(message) => {
                self.last_diagnostic = message;
            }
            EngineEvent::Disconnected { reason, .. } => {
                self.disconnect_interactive();
                self.live.disconnected();
                self.model.backend.ready = false;
                self.model.backend.status = "Worker disconnected".into();
                if let Some(running) = self.running.take() {
                    self.fail_job(&running.job, &reason, false);
                }
                self.pending.clear();
                self.model.error = Some(reason);
            }
        }
    }

    fn send(&mut self, command: Value, pending: Pending) -> Result<String, String> {
        let engine = self
            .engine
            .as_mut()
            .ok_or("The TomoJAX worker is not connected")?;
        let id = engine.request(command, None)?;
        self.pending.insert(id.clone(), pending);
        Ok(id)
    }

    fn mark_dirty(&mut self) {
        if let Some(project) = &mut self.project
            && let Err(error) = project.changed()
        {
            self.model.error = Some(error);
        }
    }

    fn open_project(&mut self, path: &Path, create: bool) -> Result<(), String> {
        if self.running.is_some()
            || self.copying > 0
            || self.metadata_pending > 0
            || !self.pending.is_empty()
        {
            return Err("Finish or cancel active work before switching projects".into());
        }
        if let Some(current) = &self.project {
            let candidate = if path.file_name().is_some_and(|n| n == project::MANIFEST) {
                path.parent().unwrap_or(path)
            } else {
                path
            };
            if std::fs::canonicalize(candidate).ok().as_ref() == Some(&current.root) {
                return Ok(());
            }
        }
        // Save the current project, including its latest workspace, before leaving
        // it; a project that cannot be saved stays open.
        let workspace = self.capture_workspace();
        if let Some(current) = &mut self.project {
            current.document.workspace = workspace;
            current
                .changed()
                .and_then(|()| current.flush())
                .map_err(|error| {
                    format!(
                        "{} could not be saved, so it stays open: {error}",
                        current.document.name
                    )
                })?;
        }
        let project = Project::open(path, create)?;
        let context = project.document.workspace.clone();
        self.preferences.visit(&project.root)?;
        self.project = Some(project);
        self.previews.reset();
        self.preview_context = previewing::PreviewContext::default();
        self.live.clear();
        self.validation.reset();
        self.imports.clear();
        self.ui = UiState::default();
        self.imaging.invalidate();
        self.restore_workspace(&context);
        self.set_workspace(self.ui.workspace);
        let document = &self.project.as_ref().unwrap().document;
        self.model
            .selected_assets
            .retain(|id| document.asset(id).is_some());
        if context
            .active
            .as_ref()
            .and_then(|id| document.asset(id))
            .is_none()
        {
            let newest = document
                .assets
                .iter()
                .max_by_key(|asset| (asset.is_result(), asset.created));
            self.model.active_asset = newest.map(|asset| asset.id.clone());
            if let Some(id) = &self.model.active_asset {
                self.model.selected_assets.insert(id.clone());
            }
            self.ui.display = DisplayState::default();
            self.restore_view(&project::Workspace::default());
            self.ui.main_view = self.default_view();
            self.ui.display.indices = self.volume_dimensions().unwrap_or_default().map(|n| n / 2);
            self.centre_sinogram = true;
        }
        self.image_key.clear();
        self.last_context = Value::Null;
        self.model.error = None;
        self.model.status = if create {
            "Project created"
        } else {
            "Project reopened"
        }
        .into();
        self.refresh_model();
        self.refresh_images();
        let unread: Vec<_> = self
            .project
            .as_ref()
            .unwrap()
            .document
            .assets
            .iter()
            .filter(|asset| asset.arrays.is_empty())
            .map(|asset| asset.id.clone())
            .collect();
        for id in unread {
            self.start_metadata(&id);
        }
        if let Some(input) = self.startup_input.take()
            && input != path
        {
            self.import_paths(vec![input]);
        }
        Ok(())
    }

    pub fn import_paths(&mut self, paths: Vec<PathBuf>) {
        if self.project.is_none() {
            self.model.error = Some("Create or open a project before importing scans".into());
            return;
        }
        let mut metadata = Vec::new();
        for path in paths {
            if path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("npz"))
            {
                self.model.error = Some(
                    "NPZ files can't be imported. Use NXtomo/HDF5 or an acquisition folder.".into(),
                );
                continue;
            }
            let path = match std::fs::canonicalize(&path) {
                Ok(path) => path,
                Err(error) => {
                    self.model.error = Some(format!("{}: {error}", path.display()));
                    continue;
                }
            };
            let project = self.project.as_mut().unwrap();
            if project
                .document
                .assets
                .iter()
                .any(|a| project.resolve(&a.path) == path)
            {
                continue;
            }
            let mut asset = Asset::input(path.clone(), false);
            if self.ui.import_copy {
                let inputs = project.root.join("inputs").join(&asset.id);
                self.copying += 1;
                self.spawn_local(None, move |cancelled| {
                    let result = copy_acquisition(&path, &inputs, &cancelled);
                    LocalEvent::Copied { asset, result }
                });
            } else {
                asset.path = project.stored_path(&path);
                self.model.active_asset = Some(asset.id.clone());
                self.model.selected_assets.clear();
                self.model.selected_assets.insert(asset.id.clone());
                self.centre_sinogram = true;
                metadata.push(asset.id.clone());
                project.document.assets.push(asset);
            }
        }
        self.ui.workspace = Workspace::Prepare;
        self.set_operation("open", None);
        self.mark_dirty();
        self.refresh_model();
        for id in metadata {
            self.start_metadata(&id);
        }
    }

    fn start_metadata(&mut self, id: &str) {
        if self.shutting_down {
            return;
        }
        let Some(path) = self
            .project
            .as_ref()
            .and_then(|p| p.document.asset(id).map(|a| p.resolve(&a.path)))
        else {
            return;
        };
        let asset = id.to_owned();
        self.metadata_pending += 1;
        self.spawn_local(None, move |cancelled| {
            let result = check_cancel(&cancelled).and_then(|()| metadata::quick_metadata(&path));
            LocalEvent::Metadata { asset, result }
        });
    }

    fn submit_imports(&mut self) {
        while let Some(id) = self.imports.pop_front() {
            if self
                .engine
                .as_ref()
                .is_none_or(|e| e.execution_capacity() < 58)
            {
                self.imports.push_front(id);
                break;
            }
            let path = self
                .project
                .as_ref()
                .and_then(|p| p.document.asset(&id).map(|a| p.resolve(&a.path)));
            if let Some(path) = path
                && let Err(error) = self.send(
                    json!({"type":"inspect","path":path}),
                    Pending::Inspect(id.clone()),
                )
            {
                self.imports.push_front(id);
                self.model.error = Some(error);
                break;
            }
        }
    }

    pub fn shutdown(&mut self) {
        if self.shutting_down {
            return;
        }
        self.shutting_down = true;
        self.stop_local_tasks();
        while let Ok(message) = self.local_receive.try_recv() {
            self.local_event(message.event);
        }
        self.previews.reset();
        self.imaging.shutdown();
        let live_artifacts = self.live.shutdown();
        self.previews
            .retire_resources(live_artifacts.into_iter().collect(), None);
        for handle in self.live.drain_handles() {
            self.previews
                .retire_resources(Default::default(), Some(handle));
        }
        if let Some(engine) = &mut self.engine {
            self.previews.collect_retired(engine, |_| false);
        }
        if let Some(engine) = &mut self.engine {
            let events = engine.drain_shutdown(Duration::from_secs(3));
            for event in events {
                self.engine_event(event);
            }
        }
        self.stop_local_tasks();
        while let Ok(message) = self.local_receive.try_recv() {
            self.local_event(message.event);
        }
        if let Some(project) = &mut self.project {
            project.document.recover();
        }
        let workspace = self.capture_workspace();
        if let Some(project) = &mut self.project {
            project.document.workspace = workspace;
            if let Err(error) = project.changed().and_then(|()| project.flush()) {
                eprintln!("Could not save project: {error}");
                self.model.error = Some(error);
            }
        }
        self.imaging.shutdown();
    }
}

/// Copies everything a chosen input is read from into `inputs` and returns the
/// copy to open. A header file brings its folder; an HDF5 file brings the files
/// it links to, at the same relative paths.
fn copy_acquisition(
    path: &Path,
    inputs: &Path,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<PathBuf, String> {
    let name = path.file_name().unwrap_or_default();
    let folder = path.parent().filter(|_| path.is_file());
    let Some(folder) = folder else {
        let copy = inputs.join(name);
        return stage_copy_input(path, &copy, cancelled).map(|()| copy);
    };
    if !project::holds_whole_acquisition(path) {
        let copy = inputs.join(folder.file_name().unwrap_or_default());
        return stage_copy_input(folder, &copy, cancelled).map(|()| copy.join(name));
    }
    let mut linked = Vec::new();
    let links = if crate::metadata::is_hdf5(path) {
        crate::metadata::linked_files(path).unwrap_or_default()
    } else {
        Vec::new()
    };
    for file in links {
        // A copy could not keep such a link pointing at the same data.
        let inside = file.strip_prefix(folder).map_err(|_| {
            format!(
                "{} reads data from {}, outside its folder. Import it without copying.",
                name.to_string_lossy(),
                file.display()
            )
        })?;
        linked.push(inside.to_owned());
    }
    if linked.is_empty() {
        let copy = inputs.join(name);
        return stage_copy_input(path, &copy, cancelled).map(|()| copy);
    }
    let copy = inputs.join(folder.file_name().unwrap_or_default());
    let members: Vec<PathBuf> = std::iter::once(PathBuf::from(name)).chain(linked).collect();
    stage_copy_members(folder, &members, &copy, cancelled).map(|()| copy.join(name))
}

/// Copies `members` (relative to `folder`) into a new `destination` folder.
fn stage_copy_members(
    folder: &Path,
    members: &[PathBuf],
    destination: &Path,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<(), String> {
    check_cancel(cancelled)?;
    let before = members
        .iter()
        .map(|member| project::Fingerprint::read(&folder.join(member)))
        .collect::<Result<Vec<_>, _>>()?;
    let staging = destination.with_extension(format!("{}-partial", project::new_id("copy")));
    let result = members
        .iter()
        .try_for_each(|member| copy_input(&folder.join(member), &staging.join(member), cancelled))
        .and_then(|()| {
            for (member, before) in members.iter().zip(&before) {
                if project::Fingerprint::read(&folder.join(member))? != *before {
                    return Err("The acquisition changed while it was being copied. Import it again after acquisition has finished.".into());
                }
            }
            check_cancel(cancelled)?;
            if destination.exists() {
                return Err("The copy destination already exists".into());
            }
            std::fs::rename(&staging, destination).map_err(|e| e.to_string())
        });
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    result
}

fn stage_copy_input(
    source: &Path,
    destination: &Path,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<(), String> {
    check_cancel(cancelled)?;
    let before = project::Fingerprint::read(source)?;
    let staging = destination.with_extension(format!("{}-partial", project::new_id("copy")));
    let result=copy_input(source,&staging,cancelled).and_then(|()|{
        check_cancel(cancelled)?;
        if project::Fingerprint::read(source)?!=before{return Err("The acquisition changed while it was being copied. Import it again after acquisition has finished.".into());}
        check_cancel(cancelled)?;
        if destination.exists(){return Err("The copy destination already exists".into());}
        std::fs::rename(&staging,destination).map_err(|e|e.to_string())
    });
    if result.is_err() {
        if staging.is_dir() {
            let _ = std::fs::remove_dir_all(&staging);
        } else {
            let _ = std::fs::remove_file(&staging);
        }
    }
    result
}

fn copy_input(
    source: &Path,
    destination: &Path,
    cancelled: &std::sync::atomic::AtomicBool,
) -> Result<(), String> {
    check_cancel(cancelled)?;
    if destination.starts_with(source) {
        return Err("Choose a project outside the source folder before copying that folder".into());
    }
    let meta = std::fs::symlink_metadata(source).map_err(|e| e.to_string())?;
    if meta.file_type().is_symlink() {
        return Err(
            "Copying a source with symbolic links requires choosing its resolved dataset path"
                .into(),
        );
    }
    if destination.exists() {
        return Err(format!(
            "The destination already exists: {}",
            destination.display()
        ));
    }
    if meta.is_dir() {
        std::fs::create_dir_all(destination).map_err(|e| e.to_string())?;
        for entry in std::fs::read_dir(source).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            copy_input(
                &entry.path(),
                &destination.join(entry.file_name()),
                cancelled,
            )?;
        }
    } else if meta.is_file() {
        std::fs::create_dir_all(destination.parent().ok_or("Invalid copy destination")?)
            .map_err(|e| e.to_string())?;
        local::copy_file(source, destination, cancelled)?;
    } else {
        return Err("Only regular files and acquisition directories can be copied".into());
    }
    Ok(())
}
