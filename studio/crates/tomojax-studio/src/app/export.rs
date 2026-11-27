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
            } else if let Err(error) = state.queue_export(input, path, &format) {
                state.model.error = Some(error);
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
        self.queue_export(input, path, &format)?;
        self.ui.export_path = None;
        Ok(())
    }
    pub(super) fn queue_export(
        &mut self,
        input: project::CapturedInput,
        path: PathBuf,
        format: &str,
    ) -> Result<(), String> {
        if path.exists() {
            return Err("That output already exists. Choose a new destination.".into());
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
            self.export_parameters()?
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
        let project = self.project.as_mut().ok_or("No project")?;
        let id = job.id.clone();
        project.document.jobs.push(job);
        self.mark_dirty();
        self.ui.activity.follow(vec![id]);
        self.refresh_model();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    #[test]
    #[ignore = "runs a real worker, the CLI and optional export readers"]
    fn queued_exports_match_the_cli_and_preview_the_returned_plan() {
        let python =
            PathBuf::from(std::env::var_os("TOMOJAX_TEST_PYTHON").expect("installed Python"));
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let script = |program: &str| {
            let result = Command::new(&python)
                .args(["-c", program])
                .current_dir(root)
                .output()
                .unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
        };
        script(
            r#"
import numpy as np
import tomojax as tj
grid = tj.Grid(258, 40, 32, 1, 1, 1)
geometry = tj.ParallelGeometry(grid, tj.Detector(258, 32, 1, 1), [0.,90.])
z,y,x = np.indices((32,40,258))
volume = (x + y + z).astype('f4')
tj.save('ramp.nxs', tj.Reconstruction(volume,geometry,'fbp',{},length_unit='mm'))
"#,
        );
        let mut state = AppState::init();
        state.open_project(&root.join("project"), true).unwrap();
        state.engine = Some(Engine::spawn(python.clone(), std::sync::Arc::new(|| {})).unwrap());
        state.import_paths(vec![root.join("ramp.nxs")]);
        let drive = |state: &mut AppState, done: &dyn Fn(&AppState) -> bool| {
            let until = Instant::now() + Duration::from_secs(45);
            while !done(state) {
                assert!(
                    Instant::now() < until,
                    "worker timed out: {:?} {:?} {:?}",
                    state.model.error,
                    state.model.validation,
                    state.model.jobs
                );
                state.poll();
                std::thread::sleep(Duration::from_millis(5));
            }
        };
        drive(&mut state, &|s| {
            s.model.backend.ready && s.metadata_pending == 0
        });
        state.set_workspace(Workspace::Explore);
        let id = state.project.as_ref().unwrap().document.assets[0]
            .id
            .clone();
        state.model.active_asset = Some(id.clone());
        state.model.selected_assets = [id].into();
        state.ui.main_view = PlaneKind::Axial;
        state.refresh_model();
        for (name, format) in [
            ("volume.tif", "ImageJ TIFF"),
            ("slices", "TIFF slices"),
            ("volume.zarr", "OME-Zarr 0.5"),
        ] {
            state.ui.export_format = format.into();
            state.ui.export_path = Some(root.join(name));
            state
                .ui
                .export_edits
                .insert("crop".into(), rune::TextEdit::new("[[1, 31], null, null]"));
            drive(&mut state, &AppState::export_plan_current);
            state
                .ui
                .export_edits
                .get_mut("crop")
                .unwrap()
                .replace_text("[[2, 30], null, null]");
            assert!(
                state.queue_current_export().is_err(),
                "an old plan admitted a new crop"
            );
            drive(&mut state, &AppState::export_plan_current);
            let Some(Product::Validated { details }) = &state.model.validation.plan else {
                panic!("missing export plan")
            };
            assert_eq!(details["payload_bytes"], 28 * 40 * 258 * 4);
            assert_eq!(details["format"], format);
            state.queue_current_export().unwrap();
            drive(&mut state, &|s| {
                s.project
                    .as_ref()
                    .unwrap()
                    .document
                    .jobs
                    .iter()
                    .all(|job| !job.status.active())
            });
            let job = state
                .project
                .as_ref()
                .unwrap()
                .document
                .jobs
                .last()
                .unwrap();
            assert_eq!(job.status, project::JobStatus::Completed, "{:?}", job.error);
        }
        script(
            r#"
import numpy as np
import tifffile, zarr
from pathlib import Path
from tomojax.cli.main import main
source = Path('ramp.nxs')
for name in ['volume.tif', 'slices', 'volume.zarr']:
    dest = Path('cli') / name
    assert main(['export', str(source), '-o', str(dest), '--crop', '2:30,:,:']) == 0
    def arrays(path):
        if path.suffix == '.zarr':
            group = zarr.open_group(path,mode='r')
            return [group[key][:] for key in sorted(group.array_keys())]
        if path.is_dir():
            return [np.stack([tifffile.imread(p) for p in sorted(path.glob('*.tif'))])]
        return [tifffile.imread(path)]
    actual, expected = arrays(Path(name)), arrays(dest)
    assert len(actual) == len(expected) == (2 if name.endswith('.zarr') else 1)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
"#,
        );
        if let Some(settings) = std::env::var_os("TOMOJAX_EXPORT_UI_EVIDENCE") {
            let (path, size, scale): (PathBuf, String, String) =
                serde_json::from_str(settings.to_str().unwrap()).unwrap();
            // Ask about a new destination to display a current plan after the first three commits.
            state.ui.export_format = "ImageJ TIFF".into();
            state.ui.export_path = Some(root.join("detail.tif"));
            drive(&mut state, &AppState::export_plan_current);
            super::super::tests::capture_native_viewport(state, &path, &size, &scale);
        } else {
            state.shutdown();
        }
    }
}
