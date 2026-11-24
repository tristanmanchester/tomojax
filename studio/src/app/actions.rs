use super::*;
use crate::project::{JobStatus, Recipe, new_id};
use rune::{FileDialogOptions, Tx};

impl AppState {
    pub fn dispatch(&mut self, action: UiAction, tx: &mut Tx) {
        let display_only = matches!(
            action,
            UiAction::ApplyWindow
                | UiAction::AutoWindow
                | UiAction::SetColormap(_)
                | UiAction::SetInvert(_)
                | UiAction::SetSlice { .. }
                | UiAction::StepSlice { .. }
                | UiAction::SetProjectionIndex(_)
                | UiAction::SetSinogramRow(_)
                | UiAction::SetCrosshair { .. }
                | UiAction::Pan { .. }
                | UiAction::Zoom(_)
                | UiAction::FitView
                | UiAction::CameraDrag { .. }
                | UiAction::SetClip { .. }
                | UiAction::SetDensity(_)
                | UiAction::SetLighting(_)
                | UiAction::SetThreshold(_)
                | UiAction::SetInspectTool(_)
                | UiAction::MeasureDrag { .. }
                | UiAction::CancelMeasurement
        );
        let image_request = matches!(
            action,
            UiAction::SetSlice { .. }
                | UiAction::StepSlice { .. }
                | UiAction::SetProjectionIndex(_)
                | UiAction::SetSinogramRow(_)
                | UiAction::SetCrosshair { .. }
        );
        // The split composite bakes in both display windows.
        let window_change = matches!(action, UiAction::ApplyWindow | UiAction::AutoWindow);
        if let Err(error) = self.action(action, tx) {
            self.model.error = Some(error);
        }
        if window_change && self.ui.compare_mode == CompareMode::Split {
            self.refresh_plane_model();
        }
        if !display_only {
            self.refresh_model();
            self.refresh_images();
        } else {
            self.model.can_undo = !self.view_history.is_empty();
            self.model.can_redo = !self.view_future.is_empty();
            if image_request {
                self.refresh_images();
            }
        }
        tx.wake_after(Duration::from_millis(450));
    }

    fn action(&mut self, action: UiAction, tx: &mut Tx) -> Result<(), String> {
        match action {
            UiAction::NewProject => {
                tx.open_folder(
                    FileDialogOptions {
                        title: "Choose a directory for the new project".into(),
                        ..Default::default()
                    },
                    |s: &mut AppState, result, tx| match result {
                        Ok(Some(path)) => {
                            if let Err(error) = s.open_project(&path, true) {
                                s.model.error = Some(error);
                            }
                            s.refresh_model();
                            tx.wake_after(Duration::from_millis(50));
                        }
                        Ok(None) => {}
                        Err(error) => s.model.error = Some(error.to_string()),
                    },
                );
            }
            UiAction::OpenProject => {
                tx.open_folder(
                    FileDialogOptions {
                        title: "Open a TomoJAX project directory".into(),
                        ..Default::default()
                    },
                    |s: &mut AppState, result, tx| match result {
                        Ok(Some(path)) => {
                            if let Err(error) = s.open_project(&path, false) {
                                s.model.error = Some(error);
                            }
                            s.refresh_model();
                            tx.wake_after(Duration::from_millis(50));
                        }
                        Ok(None) => {}
                        Err(error) => s.model.error = Some(error.to_string()),
                    },
                );
            }
            UiAction::OpenRecent(path) => self.open_project(Path::new(&path), false)?,
            UiAction::ImportFiles => {
                if self.project.is_none() {
                    return Err("Create or open a project before importing scans".into());
                }
                tx.open_file(FileDialogOptions{title:"Import tomography data".into(),..Default::default()}
                    .filter("Tomography data",&["nxs","nx","h5","hdf5","tif","tiff","xtekct","txrm","vol","log","xml","info"])
                    .filter("All files",&["*"]),|s:&mut AppState,result,_|match result{Ok(Some(path))=>s.import_paths(vec![path]),Ok(None)=>{},Err(error)=>s.model.error=Some(format!("Could not open the file chooser: {error}. Drop files into the window instead."))});
            }
            UiAction::ImportFolder => {
                if self.project.is_none() {
                    return Err("Create or open a project before importing scans".into());
                }
                tx.open_folder(FileDialogOptions{title:"Import an acquisition folder or a folder of NXtomo scans".into(),..Default::default()},|s:&mut AppState,result,_|match result{
                    Ok(Some(path))=>{
                        if path.extension().is_some_and(|e|e.eq_ignore_ascii_case("npz")){s.model.error=Some("NPZ inputs are not supported. Choose NXtomo/HDF5 or a recognised acquisition.".into());return;}
                        let mut scans:Vec<_>=std::fs::read_dir(&path).into_iter().flatten().filter_map(Result::ok).map(|e|e.path()).filter(|p|p.is_file()&&crate::metadata::is_hdf5(p)).collect();
                        scans.sort();
                        // A folder of NXtomo/HDF5 scans imports each scan; a folder of
                        // TIFF frames is one acquisition.
                        let frames=std::fs::read_dir(&path).into_iter().flatten().filter_map(Result::ok).any(|e|e.path().extension().is_some_and(|ext|matches!(ext.to_string_lossy().to_ascii_lowercase().as_str(),"tif"|"tiff")));
                        s.import_paths(if !scans.is_empty()&&!frames{scans}else{vec![path]});
                    },Ok(None)=>{},Err(error)=>s.model.error=Some(error.to_string())});
            }
            UiAction::SelectAsset { id, extend } => {
                if self
                    .project
                    .as_ref()
                    .and_then(|p| p.document.asset(&id))
                    .is_none()
                {
                    return Err("This project no longer contains that asset".into());
                }
                if !extend {
                    self.model.selected_assets.clear();
                }
                // The active asset is always one of the selected ones: removing it
                // from the selection makes the next selected asset active.
                if extend && self.model.selected_assets.remove(&id) {
                    self.settle_active();
                } else {
                    self.model.selected_assets.insert(id.clone());
                    self.activate(id);
                }
            }
            UiAction::SelectAllAssets => {
                self.model.selected_assets = self
                    .project
                    .as_ref()
                    .map(|p| p.document.assets.iter().map(|a| a.id.clone()).collect())
                    .unwrap_or_default();
            }
            UiAction::RenameResult => {
                let (id, draft) = self
                    .ui
                    .rename
                    .as_ref()
                    .ok_or("Select a result to rename.")?;
                let name = draft.text.trim();
                if name.is_empty() {
                    return Err("Enter a result name.".into());
                }
                let asset = self
                    .project
                    .as_mut()
                    .and_then(|p| p.document.asset_mut(id))
                    .filter(|a| a.is_result())
                    .ok_or("Select a result to rename.")?;
                asset.name = name.into();
                self.ui.rename = None;
                self.mark_dirty();
            }
            UiAction::RemoveAsset(id) => {
                let project = self
                    .project
                    .as_mut()
                    .ok_or("Create or open a project first")?;
                if project.document.jobs.iter().any(|job| {
                    job.status.active() && job.inputs.iter().any(|input| input.asset == id)
                }) {
                    return Err(
                        "A queued or running job uses this asset. Cancel the job first.".into(),
                    );
                }
                // Only the project's reference goes; files on disk stay where they are.
                project.document.assets.retain(|asset| asset.id != id);
                for asset in &mut project.document.assets {
                    if asset.parent.as_deref() == Some(id.as_str()) {
                        asset.parent = None;
                    }
                }
                self.model
                    .measurements
                    .retain(|measurement| measurement.asset != id);
                for side in [&mut self.model.compare_left, &mut self.model.compare_right] {
                    if side.as_deref() == Some(id.as_str()) {
                        *side = None;
                    }
                }
                self.model.selected_assets.remove(&id);
                self.settle_active();
                self.ui.confirm_remove = None;
                self.imaging.invalidate();
                self.image_key.clear();
                self.mark_dirty();
            }
            UiAction::RemoveRecipe(id) => {
                let project = self
                    .project
                    .as_mut()
                    .ok_or("Create or open a project first")?;
                project.document.recipes.retain(|recipe| recipe.id != id);
                self.mark_dirty();
            }
            UiAction::RelinkAsset(id) => {
                // An acquisition imported as a folder (a TIFF stack, frames beside a
                // header) is reconnected by choosing its folder.
                let folder = self
                    .project
                    .as_ref()
                    .and_then(|project| project.document.asset(&id))
                    .is_some_and(|asset| asset.folder);
                let reconnect = move |s: &mut AppState,
                                      result: rune::FileDialogResult,
                                      _: &mut Tx| match result {
                    Ok(Some(path)) => s.relink(&id, &path),
                    Ok(None) => {}
                    Err(error) => s.model.error = Some(error.to_string()),
                };
                let title = if folder {
                    "Reconnect the acquisition folder"
                } else {
                    "Reconnect the original data file"
                };
                let options = FileDialogOptions {
                    title: title.into(),
                    ..Default::default()
                };
                if folder {
                    tx.open_folder(options, reconnect);
                } else {
                    tx.open_file(options, reconnect);
                }
            }
            UiAction::SetWorkspace(workspace) => self.set_workspace(workspace),
            UiAction::SetMainView(plane) => {
                self.remember_view();
                self.ui.main_view = plane;
            }
            UiAction::SetPreviewArray(key) => {
                if !self
                    .model
                    .preview_choices
                    .iter()
                    .any(|(value, _)| value == &key)
                {
                    return Err("Choose an available preview array".into());
                }
                self.ui.preview_array = key;
                self.ui.main_view = PlaneKind::Projection;
                self.ui.display.projection_index = 0;
                self.ui.display.sinogram_row = 0;
                self.centre_sinogram = true;
                self.ui.display.window_low.replace_text("");
                self.ui.display.window_high.replace_text("");
                self.applied_window = None;
                self.window_prefill = Some([String::new(), String::new()]);
            }
            UiAction::Simulate => {
                self.ui.workspace = Workspace::Prepare;
                self.set_operation("simulate", None);
            }
            UiAction::SetOperation(operation) => self.set_operation(&operation, None),
            UiAction::QueueOperation(operation) => self.queue_operation(&operation)?,
            UiAction::RefreshPreview => {
                let operation = self.ui.operation.clone();
                if let Some(kind) = super::previewing::kind(&operation) {
                    self.previews.retry(kind);
                }
                self.preview_context.signature.clear();
                self.request_interactive(&operation)?;
            }
            UiAction::KeepPreview => self.keep_interactive()?,
            UiAction::CancelPreview => self.cancel_interactive(),
            UiAction::SetPreviewIndex(index) => {
                self.ui
                    .parameter_edits
                    .get_mut("index")
                    .ok_or("This preview has no slice index")?
                    .replace_text(index.to_string());
                self.poll_interactive();
            }
            UiAction::SetPreviewCentre(centre) => {
                if !centre.is_finite() {
                    return Err("Enter a finite centre offset".into());
                }
                self.ui
                    .parameter_edits
                    .get_mut("centres")
                    .ok_or("This preview has no centre sweep")?
                    .replace_text(json!([centre]).to_string());
                self.poll_interactive();
            }
            UiAction::SaveRecipe => {
                let parameters = forms::parameters(&self.ui, &self.catalog)?;
                let name = self.ui.recipe_name.text.trim();
                if name.is_empty() {
                    return Err("Give this recipe a name".into());
                }
                let project = self
                    .project
                    .as_mut()
                    .ok_or("Create or open a project first")?;
                project.document.recipes.push(Recipe {
                    id: new_id("recipe"),
                    name: name.into(),
                    operation: self.ui.operation.clone(),
                    parameters,
                });
                self.mark_dirty();
            }
            UiAction::ApplyRecipe(id) => {
                let recipe = self
                    .project
                    .as_ref()
                    .and_then(|p| p.document.recipes.iter().find(|r| r.id == id))
                    .cloned()
                    .ok_or("Recipe not found")?;
                self.set_operation(&recipe.operation, Some(&recipe.parameters));
                self.set_workspace(Workspace::for_operation(&recipe.operation));
            }
            UiAction::PauseJob(id) => self.control_job(&id, "pause")?,
            UiAction::ResumeJob(id) => self.control_job(&id, "resume")?,
            UiAction::CancelJob(id) => {
                let status = self
                    .project
                    .as_ref()
                    .and_then(|p| p.document.jobs.iter().find(|j| j.id == id))
                    .map(|j| j.status);
                if status == Some(JobStatus::Queued) {
                    self.fail_job(&id, "Cancelled before execution", true);
                } else {
                    self.control_job(&id, "cancel")?;
                }
            }
            UiAction::RetryJob(id) => {
                let project = self.project.as_mut().ok_or("No project")?;
                let old = project
                    .document
                    .jobs
                    .iter()
                    .find(|j| j.id == id)
                    .ok_or("Job not found")?;
                if old.status.active() {
                    return Err("This job is still active".into());
                }
                old.check_inputs()?;
                if old.destination.as_ref().is_some_and(|p| p.exists()) {
                    return Err(format!(
                        "An output already exists at {}. Move or delete it, then retry.",
                        old.destination.as_ref().unwrap().display()
                    ));
                }
                let mut job =
                    project::Job::new(&old.operation, old.parameters.clone(), old.inputs.clone());
                job.name = format!("{} · retry", old.name);
                project.document.jobs.push(job);
                self.mark_dirty();
            }
            UiAction::MoveJob { id, delta } => {
                let project = self.project.as_mut().ok_or("No project")?;
                let index = project
                    .document
                    .jobs
                    .iter()
                    .position(|j| j.id == id)
                    .ok_or("Job not found")?;
                if project.document.jobs[index].status != JobStatus::Queued {
                    return Err("Only waiting jobs can be reordered".into());
                }
                let target = if delta < 0 {
                    (0..index)
                        .rev()
                        .find(|i| project.document.jobs[*i].status == JobStatus::Queued)
                } else {
                    (index + 1..project.document.jobs.len())
                        .find(|i| project.document.jobs[*i].status == JobStatus::Queued)
                };
                if let Some(target) = target {
                    project.document.jobs.swap(index, target);
                    self.mark_dirty();
                }
            }
            UiAction::Export(format) => self.export_dialog(&format, tx)?,
            UiAction::ShowPalette => {
                self.ui.palette_open = true;
                self.ui.command_index = 0;
                self.ui.command_query.replace_text("");
                tx.focus("command-search", true);
            }
            UiAction::RunCommand(id) => {
                self.ui.palette_open = false;
                self.run_command(&id, tx)?;
            }
            UiAction::ShowShortcuts => {
                self.ui.shortcuts_open = true;
                self.ui.palette_open = false;
            }
            UiAction::BeginShortcutEdit(id) => {
                let default = crate::preferences::COMMANDS
                    .iter()
                    .find(|c| c.0 == id)
                    .ok_or("Unknown command")?
                    .3;
                self.ui
                    .shortcut_binding
                    .replace_text(self.preferences.binding(&id, default));
                self.ui.editing_shortcut = Some(id);
                self.model.shortcut_error = None;
            }
            UiAction::SaveShortcut { command, binding } => {
                match self.preferences.set_binding(&command, &binding) {
                    Ok(()) => {
                        self.ui.editing_shortcut = None;
                        self.model.shortcut_error = None;
                    }
                    Err(error) => self.model.shortcut_error = Some(error),
                }
            }
            UiAction::ResetShortcut(id) => {
                let default = crate::preferences::COMMANDS
                    .iter()
                    .find(|c| c.0 == id)
                    .ok_or("Unknown command")?
                    .3;
                match self.preferences.set_binding(&id, default) {
                    Ok(()) => self.model.shortcut_error = None,
                    Err(error) => self.model.shortcut_error = Some(error),
                }
            }
            UiAction::ApplyWindow => {
                let window = self.valid_window()?;
                self.remember_view();
                self.applied_window = window;
                self.window_prefill = Some([
                    self.ui.display.window_low.text.clone(),
                    self.ui.display.window_high.text.clone(),
                ]);
            }
            UiAction::AutoWindow => {
                self.remember_view();
                self.applied_window = None;
                self.ui.display.window_low.replace_text("");
                self.ui.display.window_high.replace_text("");
                self.window_prefill = Some([String::new(), String::new()]);
                self.prefill_window();
            }
            UiAction::SetColormap(map) => {
                self.remember_view();
                self.ui.display.colormap = map;
            }
            UiAction::SetInvert(invert) => {
                self.remember_view();
                self.ui.display.invert = invert;
            }
            UiAction::SetSlice { axis, index } => {
                if axis < 3 {
                    let count = if self.ui.workspace == Workspace::Compare
                        && Some(axis) == self.ui.compare_plane.axis()
                    {
                        self.model.comparison_count
                    } else {
                        self.slice_count(axis)
                    };
                    self.ui.display.indices[axis] = index.min(count.saturating_sub(1));
                    self.last_interaction = Instant::now();
                }
            }
            UiAction::StepSlice { axis, delta } => {
                if axis < 3 {
                    let index = self.ui.display.indices[axis].saturating_add_signed(delta as isize);
                    self.action(UiAction::SetSlice { axis, index }, tx)?;
                }
            }
            UiAction::SetProjectionIndex(index) => {
                let count = self
                    .active_projections()
                    .and_then(|a| a.shape.first().copied())
                    .unwrap_or(1);
                self.ui.display.projection_index = index.min(count.saturating_sub(1));
            }
            UiAction::SetSinogramRow(index) => {
                let count = self
                    .active_projections()
                    .and_then(|a| a.shape.get(1).copied())
                    .unwrap_or(1);
                self.ui.display.sinogram_row = index.min(count.saturating_sub(1));
            }
            UiAction::SetCrosshair { plane, position } => self.crosshair(plane, position),
            UiAction::Pan { delta } => {
                for (axis, value) in delta.iter().enumerate() {
                    if value.is_finite() {
                        self.ui.display.pan[axis] += value;
                    }
                }
            }
            UiAction::Zoom(factor) => {
                if factor.is_finite() && factor > 0. {
                    self.ui.display.zoom = (self.ui.display.zoom * factor).clamp(0.1, 20.);
                    self.last_interaction = Instant::now();
                }
            }
            UiAction::FitView => {
                self.remember_view();
                self.ui.display.pan = [0.; 2];
                self.ui.display.zoom = 1.;
            }
            UiAction::CameraDrag { delta } => {
                self.ui.display.azimuth = (self.ui.display.azimuth + delta[0]) % 360.;
                self.ui.display.elevation = (self.ui.display.elevation + delta[1]).clamp(-89., 89.);
                self.last_interaction = Instant::now();
                tx.wake_after(Duration::from_millis(180));
            }
            UiAction::SetClip { axis, value } => {
                if axis < 3 && value.is_finite() {
                    self.ui.display.clip[axis] = value.clamp(0.01, 1.);
                    self.last_interaction = Instant::now();
                }
            }
            UiAction::SetDensity(value) => self.ui.display.density = value.clamp(0.05, 8.),
            UiAction::SetThreshold(value) => self.ui.display.threshold = value.clamp(0., 0.99),
            UiAction::SetLighting(value) => self.ui.display.lighting = value.clamp(0., 1.),
            UiAction::SwapComparison => {
                std::mem::swap(&mut self.model.compare_left, &mut self.model.compare_right)
            }
            UiAction::CompareMode(mode) => self.ui.compare_mode = mode,
            UiAction::SetCompareSplit(fraction) => {
                if fraction.is_finite() {
                    self.ui.compare_split = fraction.clamp(0., 1.);
                }
            }
            UiAction::ComparePlane(plane) => self.ui.compare_plane = plane,
            UiAction::SetExportFormat(format) => self.ui.export_format = format,
            UiAction::RemoveMeasurement(id) => self
                .model
                .measurements
                .retain(|m| m.id != id || Some(&m.asset) != self.model.active_asset.as_ref()),
            UiAction::SetCompareLeft(id) => {
                self.model.compare_left = Some(id);
                self.ui.compare_menu = None;
            }
            UiAction::SetCompareRight(id) => {
                self.model.compare_right = Some(id);
                self.ui.compare_menu = None;
            }
            UiAction::CompareSelection => {
                let ids: Vec<_> = self.model.selected_assets.iter().cloned().collect();
                if ids.len() != 2 {
                    return Err("Select two results to compare".into());
                }
                self.model.compare_left = Some(ids[0].clone());
                self.model.compare_right = Some(ids[1].clone());
                self.ui.workspace = Workspace::Compare;
            }
            UiAction::SetCompareLinked(linked) => self.ui.compare_linked = linked,
            UiAction::SetInspectTool(tool) => {
                if self.model.preview.is_some() && tool != InspectTool::Navigate {
                    return Err("Keep the preview as a project result before measuring it".into());
                }
                self.ui.inspect_tool = tool;
            }
            UiAction::MeasureDrag {
                plane,
                start,
                end,
                finished,
            } => {
                if self.model.preview.is_some() {
                    return Err("Keep the preview as a project result before measuring it".into());
                }
                self.measure(plane, start, end, finished)?;
            }
            UiAction::CancelMeasurement => {
                self.model.measurements.retain(|m| m.id != "draft");
            }
            UiAction::CancelCopies => {
                self.cancel_local(IMPORT_COPY);
            }
            UiAction::ClearMeasurements => {
                self.model
                    .measurements
                    .retain(|m| Some(&m.asset) != self.model.active_asset.as_ref());
                self.mark_dirty();
            }
            UiAction::Undo => {
                if let Some(prior) = self.view_history.pop() {
                    self.view_future.push(self.capture_workspace());
                    self.restore_view(&prior);
                }
            }
            UiAction::Redo => {
                if let Some(next) = self.view_future.pop() {
                    self.view_history.push(self.capture_workspace());
                    self.restore_view(&next);
                }
            }
            UiAction::DismissError => self.model.error = None,
        }
        Ok(())
    }

    fn run_command(&mut self, id: &str, tx: &mut Tx) -> Result<(), String> {
        let action = match id {
            "new" => UiAction::NewProject,
            "open" => UiAction::OpenProject,
            "import" => UiAction::ImportFiles,
            "folder" => UiAction::ImportFolder,
            "palette" => UiAction::ShowPalette,
            "shortcuts" => UiAction::ShowShortcuts,
            "documentation" => {
                return tx
                    .open_url("https://tomojax.dev/")
                    .map_err(|_| "Could not open the browser. Visit tomojax.dev.".into());
            }
            "prepare" => UiAction::SetWorkspace(Workspace::Prepare),
            "reconstruct" => UiAction::SetWorkspace(Workspace::Reconstruct),
            "explore" => UiAction::SetWorkspace(Workspace::Explore),
            "compare" => UiAction::SetWorkspace(Workspace::Compare),
            "jobs" => UiAction::SetWorkspace(Workspace::Jobs),
            "queue" => UiAction::QueueOperation(self.ui.operation.clone()),
            "simulate" => UiAction::Simulate,
            "export" => UiAction::Export(self.model.export_format(&self.ui.export_format).into()),
            "fit" => UiAction::FitView,
            "window" => UiAction::AutoWindow,
            "axial" => UiAction::SetMainView(PlaneKind::Axial),
            "coronal" => UiAction::SetMainView(PlaneKind::Coronal),
            "sagittal" => UiAction::SetMainView(PlaneKind::Sagittal),
            "volume" => UiAction::SetMainView(PlaneKind::Volume),
            "previous" | "next" => UiAction::StepSlice {
                axis: self.ui.main_view.axis().unwrap_or(0),
                delta: if id == "previous" { -1 } else { 1 },
            },
            "undo" => UiAction::Undo,
            "redo" => UiAction::Redo,
            _ => return Err("Unknown command".into()),
        };
        self.action(action, tx)
    }

    fn export_dialog(&mut self, format: &str, tx: &mut Tx) -> Result<(), String> {
        if format.is_empty() {
            return Err("Raw frames can be exported once corrected".into());
        }
        let id = self
            .model
            .active_asset
            .clone()
            .ok_or("Select a result to export")?;
        let input = self.project.as_ref().ok_or("No project")?.capture(&id)?;
        if !matches!(
            input.kind.as_str(),
            "scan" | "frames" | "reconstruction" | "simulation" | "alignment"
        ) {
            tx.open_folder(
                FileDialogOptions {
                    title: "Choose a parent directory for the registered NPY array bundle".into(),
                    ..Default::default()
                },
                move |s: &mut AppState, result, _| match result {
                    Ok(Some(parent)) => {
                        let path = parent.join(format!(
                            "{}-array",
                            input.name.replace(['/', '\\', ':'], "_")
                        ));
                        s.queue_export(input, path, "array");
                    }
                    Ok(None) => {}
                    Err(error) => s.model.error = Some(error.to_string()),
                },
            );
            return Ok(());
        }
        let format = format.to_owned();
        if matches!(format.as_str(), "tiff" | "float32" | "uint16") {
            tx.open_folder(
                FileDialogOptions {
                    title: "Choose a parent directory for TIFF slices".into(),
                    ..Default::default()
                },
                move |s: &mut AppState, result, _| match result {
                    Ok(Some(path)) => {
                        let destination = path.join(format!(
                            "{}-slices",
                            input.name.replace(['/', '\\', ':'], "_")
                        ));
                        s.queue_export(input, destination, &format);
                    }
                    Ok(None) => {}
                    Err(error) => s.model.error = Some(error.to_string()),
                },
            );
        } else {
            tx.save_file(
                FileDialogOptions {
                    title: "Export result".into(),
                    file_name: Some(format!(
                        "{}.{}",
                        input.name,
                        if format == "raw" { "raw" } else { "nxs" }
                    )),
                    ..Default::default()
                },
                move |s: &mut AppState, result, _| match result {
                    Ok(Some(path)) => s.queue_export(input, path, &format),
                    Ok(None) => {}
                    Err(error) => s.model.error = Some(error.to_string()),
                },
            );
        }
        Ok(())
    }

    pub(super) fn queue_export(
        &mut self,
        input: project::CapturedInput,
        path: PathBuf,
        format: &str,
    ) {
        if path.exists() {
            self.model.error = Some(
                "That output already exists. Choose a new path to preserve the original.".into(),
            );
            return;
        }
        let op = if format == "array" {
            "export_array"
        } else if format == "nxs" {
            "save"
        } else {
            "export"
        };
        let params = if op == "save" {
            // A reconstruction saved without its scan has none to include.
            let scan = input.kind != "reconstruction"
                || self
                    .project
                    .as_ref()
                    .and_then(|p| p.document.asset(&input.asset))
                    .is_none_or(|asset| {
                        asset.inspection.pointer("/projection/found") != Some(&Value::Bool(false))
                    });
            json!({"path":path,"include_scan":scan})
        } else {
            json!({"path":path,"dtype":if format=="uint16"{"uint16"}else{"float32"}})
        };
        let mut job = project::Job::new(op, params, vec![input]);
        job.name = format!("Export {format}");
        job.destination = Some(path);
        if let Some(project) = &mut self.project {
            project.document.jobs.push(job);
            self.mark_dirty();
            self.ui.jobs_open = true;
            self.refresh_model();
        }
    }

    fn remember_view(&mut self) {
        self.view_history.push(self.capture_workspace());
        if self.view_history.len() > 48 {
            self.view_history.remove(0);
        }
        self.view_future.clear();
    }
}
