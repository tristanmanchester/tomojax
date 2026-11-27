//! Explore's registered export draft, worker preflight and durable queue entry.
use super::*;
use crate::protocol::ExportFormat;
use crate::ui::model::Export;
use rune::{FileDialogOptions, Tx};

impl AppState {
    pub(crate) fn active_export_format(&self) -> &str {
        let Some(asset) = self.model.active() else {
            return "";
        };
        match asset.export {
            Export::Array => "array",
            Export::Unavailable => "",
            Export::Volume | Export::Dataset => self
                .catalog
                .export_formats
                .iter()
                .find(|f| {
                    f.name == self.ui.export_format
                        && (!f.spatial_only || asset.export == Export::Volume)
                })
                .map_or("nxs", |f| f.name.as_str()),
        }
    }

    pub(crate) fn interchange_format(&self) -> Option<&ExportFormat> {
        let selected = self.active_export_format();
        self.catalog
            .export_formats
            .iter()
            .find(|f| f.name == selected)
    }

    pub(super) fn export_parameters(&self) -> Result<Value, String> {
        let format = self.interchange_format().ok_or("Choose an export format")?;
        let path = self
            .ui
            .export_path
            .as_deref()
            .ok_or("Choose a destination")?;
        let asset = self.model.active().ok_or("Select a result to export")?;
        let mut parameters = format.parameters(&self.catalog, asset.mask, path, |key| {
            self.ui.export_edits.get(key).map(|e| e.text.as_str())
        })?;
        if asset.export == Export::Dataset {
            parameters["member"] = json!("projections");
        }
        Ok(parameters)
    }

    pub(super) fn export_dialog(&mut self, format: &str, tx: &mut Tx) -> Result<(), String> {
        let id = self
            .model
            .active_asset
            .as_deref()
            .ok_or("Select a result to export")?;
        let input = self.project.as_ref().ok_or("No project")?.capture(id)?;
        if format.is_empty() {
            return Err("Correct raw frames before exporting them".into());
        }
        let entry = self
            .catalog
            .export_formats
            .iter()
            .find(|f| f.name == format);
        let folder = format == "array" || entry.is_some_and(|f| f.layout == "directory");
        let suffix = entry
            .map(|f| f.suffixes.first().map_or("-slices", String::as_str))
            .unwrap_or(if format == "array" { "-array" } else { ".nxs" });
        let name = format!("{}{suffix}", input.name.replace(['/', '\\', ':'], "_"));
        let format = format.to_string();
        let choose = move |state: &mut AppState, path: PathBuf| {
            if state.active_export_format() != format
                || state.model.active_asset.as_ref() != Some(&input.asset)
            {
                return;
            }
            if state.interchange_format().is_some() {
                state.ui.export_path = Some(path);
            } else {
                state.queue_export(input, path, &format);
            }
        };
        if folder {
            tx.open_folder(
                FileDialogOptions {
                    title: format!("Choose the parent directory for {name}"),
                    ..Default::default()
                },
                move |s: &mut AppState, result, _| match result {
                    Ok(Some(parent)) => choose(s, parent.join(&name)),
                    Err(e) => s.model.error = Some(e.to_string()),
                    Ok(None) => {}
                },
            );
        } else {
            tx.save_file(
                FileDialogOptions {
                    title: "Choose destination".into(),
                    file_name: Some(name),
                    ..Default::default()
                },
                move |s: &mut AppState, result, _| match result {
                    Ok(Some(path)) => choose(s, path),
                    Err(e) => s.model.error = Some(e.to_string()),
                    Ok(None) => {}
                },
            );
        }
        Ok(())
    }

    pub(super) fn queue_current_export(&mut self) -> Result<(), String> {
        if !self.export_plan_current() {
            return Err("Wait for the current export plan".into());
        }
        let id = self
            .model
            .active_asset
            .as_deref()
            .ok_or("Select an input")?;
        let input = self.project.as_ref().ok_or("No project")?.capture(id)?;
        let format = self.active_export_format().to_string();
        let path = self.ui.export_path.clone().ok_or("Choose a destination")?;
        self.queue_export(input, path, &format);
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
        } else if op == "export" {
            match self.export_parameters() {
                Ok(parameters) => parameters,
                Err(message) => {
                    self.model.error = Some(message);
                    return;
                }
            }
        } else {
            json!({"path":path})
        };
        let mut job = project::Job::new(op, params, vec![input]);
        job.name = if op == "save" {
            "Save TomoJAX result".into()
        } else {
            format!("Export {format}")
        };
        job.destination = Some(path);
        if let Some(project) = &mut self.project {
            let id = job.id.clone();
            project.document.jobs.push(job);
            self.mark_dirty();
            self.ui.activity.follow(vec![id]);
            self.refresh_model();
        }
    }
}
