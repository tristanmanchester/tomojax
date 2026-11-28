use super::*;

impl AppState {
    pub fn import_paths(&mut self, paths: Vec<PathBuf>) {
        if self.remote_project().is_some() {
            self.model.error = Some("Use Import to select a file on this connection, or Copy from this computer to upload one.".into());
            return;
        }
        let mut selected = Vec::new();
        for path in paths {
            match std::fs::canonicalize(&path) {
                Ok(path) => {
                    let folder = path.is_dir();
                    selected.push((path, folder));
                }
                Err(e) => self.model.error = Some(format!("{}: {e}", path.display())),
            }
        }
        self.import_selected(selected);
    }

    pub(super) fn import_selected(&mut self, paths: Vec<(PathBuf, bool)>) {
        let selection = self.selected_inputs();
        let Some(project) = &mut self.project else {
            self.model.error = Some("Create or open a project before importing scans".into());
            return;
        };
        let mut metadata = Vec::new();
        let mut copies = Vec::new();
        for (path, folder) in paths {
            if path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("npz"))
            {
                self.model.error = Some(
                    "NPZ files can't be imported. Use NXtomo/HDF5 or an acquisition folder.".into(),
                );
                continue;
            }
            if project
                .document
                .assets
                .iter()
                .any(|a| project.resolve(&a.path).lexical() == path)
            {
                continue;
            }
            let mut asset = Asset::at(path.clone().into(), false, folder);
            if self.ui.import_copy {
                let inputs = project.data_root().join("inputs").join(&asset.id);
                copies.push((asset, path, inputs));
            } else {
                asset.path = project.stored_path(&path);
                metadata.push(asset.id.clone());
                project.document.assets.push(asset);
            }
        }
        for (asset, path, inputs) in copies {
            self.copying += 1;
            let selection = selection.clone();
            let files = self.file_reads();
            self.spawn_local(Some(IMPORT_COPY.into()), move |canceled| {
                let result =
                    local::copy_acquisition(files, &path, asset.folder, &inputs, &canceled);
                LocalEvent::Copied {
                    asset,
                    result,
                    selection,
                }
            });
        }
        self.ui.workspace = Workspace::Prepare;
        if let Some(id) = metadata.last() {
            self.model.selected_assets = [id.clone()].into();
            self.activate(id.clone());
        }
        self.set_operation("open", None);
        self.mark_dirty();
        self.refresh_model();
        for id in metadata {
            self.start_metadata(&id, true);
        }
    }

    pub(super) fn start_metadata(&mut self, id: &str, inspect_if_unknown: bool) {
        if self.shutting_down || !self.native_data_available() {
            return;
        }
        let Some((path, intent)) = self
            .project
            .as_ref()
            .and_then(|p| p.document.asset(id).map(|a| (p.resolve(&a.path), a.intent)))
        else {
            return;
        };
        let asset = id.to_owned();
        let files = self.file_reads();
        self.metadata_pending += 1;
        self.spawn_local(None, move |canceled| {
            let result =
                check_cancel(&canceled).and_then(|()| local::stored(files, path.lexical(), None));
            LocalEvent::Metadata {
                inspect_if_unknown,
                asset,
                intent,
                result,
            }
        });
    }

    pub(super) fn submit_imports(&mut self) {
        while let Some((id, intent)) = self.imports.pop_front() {
            if self
                .engine
                .as_ref()
                .is_none_or(|e| e.execution_capacity() < 58)
            {
                self.imports.push_front((id, intent));
                break;
            }
            if let Some(project) = &self.project
                && project
                    .document
                    .asset(&id)
                    .is_some_and(|a| a.intent == intent)
                && let Ok(input) = project.capture(&id)
                && let Err(error) =
                    self.capture_jobs(vec![project::Job::new("inspect", json!({}), vec![input])])
            {
                self.model.error = Some(error);
            }
        }
    }

    /// Points asset `id` at `path`, chosen by the user after its data moved.
    pub(super) fn relink(&mut self, id: &str, path: &Path) {
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("npz"))
        {
            self.model.error = Some(
                "NPZ inputs are not supported. Choose NXtomo/HDF5 or a recognized acquisition."
                    .into(),
            );
            return;
        }
        let mut reread = true;
        if let Some(project) = &mut self.project {
            let path = project.stored_path(path).lexical().to_owned();
            let root = project.data_root().to_owned();
            if let Some(asset) = project.document.asset_mut(id) {
                asset.relink(path, &root);
                reread = !asset.is_result();
            }
        }
        self.model
            .measurements
            .retain(|measurement| measurement.asset != id);
        self.imaging.invalidate();
        if reread {
            self.start_metadata(id, true);
        }
        self.image_key.clear();
        self.mark_dirty();
        self.refresh_model();
        self.refresh_images();
    }
}
