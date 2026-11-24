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
        if self.project.is_none() || self.shutting_down {
            self.retire_output(&preview.output);
            return Err("Open a project before keeping the preview".into());
        }
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
        self.project.as_mut().unwrap().document.jobs.push(job);
        self.mark_dirty();
        self.persist_arrays(&id, preview.output);
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
            "slice" | "centre_sweep" | "correction_preview" | "region_preview"
        ) {
            return self.request_interactive(operation);
        }
        if !self.model.backend.ready {
            return Err("Wait for the TomoJAX worker to become ready".into());
        }
        self.poll_form_context();
        if let Some(error) = self.current_validation_error(operation) {
            return Err(error);
        }
        let parameters = forms::parameters(&self.ui, &self.catalog)?;
        let selected = self.selected_inputs();
        let project = self
            .project
            .as_mut()
            .ok_or("Create or open a project first")?;
        if operation != "simulate" && selected.is_empty() {
            return Err("Select a scan or result first".into());
        }
        let inputs = if operation == "simulate" {
            vec![]
        } else {
            selected
                .iter()
                .map(|id| project.capture(id)?.with_files_named_in(&parameters))
                .collect::<Result<Vec<_>, _>>()?
        };
        // Results are saved as datasets, which hold scans, not raw frames.
        if matches!(operation, "crop" | "select")
            && inputs.iter().any(|input| input.kind == "frames")
        {
            return Err(
                "Raw frames can't be saved after cropping or selecting views. Run Corrections first, then crop or select views of the corrected scan.".into(),
            );
        }
        if operation == "simulate" {
            project
                .document
                .jobs
                .push(Job::new(operation, parameters, vec![]));
        } else if matches!(operation, "combine" | "project" | "backproject") {
            project
                .document
                .jobs
                .push(Job::new(operation, parameters, inputs));
        } else {
            for input in inputs {
                project
                    .document
                    .jobs
                    .push(Job::new(operation, parameters.clone(), vec![input]));
            }
        }
        self.mark_dirty();
        self.ui.jobs_open = true;
        self.model.status = "Submitted settings are captured with each job".into();
        self.refresh_model();
        Ok(())
    }

    pub(super) fn schedule(&mut self) {
        if self.running.is_some()
            || self.previews.has_priority_work()
            || self
                .engine
                .as_ref()
                .is_none_or(|e| e.execution_capacity() == 0)
            || self
                .project
                .as_ref()
                .is_none_or(|p| p.dirty() || p.save_error.is_some())
        {
            return;
        }
        let job = self
            .project
            .as_ref()
            .and_then(|p| {
                p.document
                    .jobs
                    .iter()
                    .find(|j| j.status == JobStatus::Queued)
            })
            .cloned();
        let Some(job) = job else { return };
        self.live.clear();
        if let Some(project) = &mut self.project
            && let Some(active) = project.document.job_mut(&job.id)
        {
            active.status = JobStatus::Running;
            active.started = Some(now_ms());
            active.stage = "Checking captured inputs".into();
        }
        self.running = Some(Running {
            job: job.id.clone(),
            handles: Vec::new(),
            products: Vec::new(),
            next_input: 0,
        });
        self.mark_dirty();
        self.spawn_local(Some(job.id.clone()), move |cancelled| {
            let result = check_cancel(&cancelled)
                .and_then(|()| job.check_inputs())
                .and_then(|()| check_cancel(&cancelled));
            LocalEvent::StartChecked {
                job: job.id,
                result,
            }
        });
    }

    fn active_job(&self) -> Option<Job> {
        self.running.as_ref().and_then(|r| {
            self.project
                .as_ref()?
                .document
                .jobs
                .iter()
                .find(|j| j.id == r.job)
                .cloned()
        })
    }

    fn export_array(&mut self, job: &Job) {
        let Some(input) = job.inputs.first().cloned() else {
            self.fail_job(&job.id, "Array export needs an input", false);
            return;
        };
        let Some(destination) = job
            .parameters
            .get("path")
            .and_then(Value::as_str)
            .map(PathBuf::from)
        else {
            self.fail_job(&job.id, "Array export needs a destination", false);
            return;
        };
        let id = job.id.clone();
        let captured = job.clone();
        self.spawn_local(Some(id.clone()),move|cancelled|{
            let result=(||->Result<(),String>{
                check_cancel(&cancelled)?;
                let parent=destination.parent().ok_or("Invalid export directory")?;
                if destination.exists(){return Err("The export directory already exists".into());}
                let stage=tempfile::Builder::new().prefix(".tomojax-export-").tempdir_in(parent).map_err(|e|e.to_string())?;
                let mut arrays=input.arrays.clone();
                for (index,(_name,array)) in arrays.iter_mut().enumerate(){
                    let copy=|storage:&mut Storage,suffix:&str|->Result<(),String>{
                        match storage {
                            Storage::Npy{path,..}=>{
                                let file=format!("array-{index}{suffix}.npy");
                                local::copy_file(Path::new(path),&stage.path().join(&file),&cancelled)?;
                                *storage=Storage::Npy{path:file,artifact:None};Ok(())
                            }
                            _=>Err("This bundle export supports durable NPY arrays only".into()),
                        }
                    };
                    copy(&mut array.storage,"")?;
                    for (axis_index,axis) in array.axes.iter_mut().enumerate(){if let crate::protocol::Coordinates::Coordinates{values}=&mut axis.coordinates{copy(values,&format!("-axis{axis_index}"))?;}}
                }
                let document=json!({"format":"tomojax-registered-arrays","version":1,"source":input.path,"arrays":arrays});
                std::fs::write(stage.path().join("arrays.json"),serde_json::to_vec_pretty(&document).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
                captured.check_inputs()?;
                check_cancel(&cancelled)?;
                if destination.exists(){return Err("The export directory was created by another process".into());}
                std::fs::rename(stage.path(),&destination).map_err(|e|e.to_string())?;
                Ok(())
            })();
            LocalEvent::Exported{job:id,result}
        });
    }

    pub(super) fn advance_job(&mut self) {
        if self.shutting_down {
            return;
        }
        let Some(job) = self.active_job() else { return };
        let index = self.running.as_ref().unwrap().next_input;
        if job.operation == "export_array" {
            self.export_array(&job);
            return;
        }
        if !matches!(job.operation.as_str(), "open" | "inspect") && index < job.inputs.len() {
            let input = &job.inputs[index];
            if let Some(product) = registered_array(input) {
                let running = self.running.as_mut().unwrap();
                running.handles.push(String::new());
                running.products.push(product);
                running.next_input += 1;
                self.advance_job();
                return;
            }
            let mut command = input
                .open_parameters
                .as_object()
                .cloned()
                .unwrap_or_default();
            command.insert("type".into(), json!("open"));
            command.insert("path".into(), json!(input.path));
            command.insert("kind".into(), json!(input.kind));
            if let Err(error) = self.send(
                Value::Object(command),
                Pending::OpenInput {
                    job: job.id.clone(),
                    index,
                },
            ) {
                self.fail_job(&job.id, &error, false);
            }
            return;
        }
        match self.job_command(&job) {
            Ok(command) => {
                if let Err(error) = self.send(command, Pending::Execute(job.id.clone())) {
                    self.fail_job(&job.id, &error, false);
                }
            }
            Err(error) => self.fail_job(&job.id, &error, false),
        }
    }

    fn job_command(&self, job: &Job) -> Result<Value, String> {
        let mut operation = job
            .parameters
            .as_object()
            .cloned()
            .ok_or("Job parameters must be an object")?;
        operation.insert(
            "type".into(),
            json!(if job.operation == "plan" {
                "reconstruct"
            } else {
                &job.operation
            }),
        );
        let running = self.running.as_ref().ok_or("No active job")?;
        let source = |index: usize, scan: bool| -> Result<Value, String> {
            job.inputs
                .get(index)
                .ok_or("This operation needs an input")?;
            let handle = running
                .handles
                .get(index)
                .ok_or("The input was not opened")?;
            if scan {
                let product = running
                    .products
                    .get(index)
                    .ok_or("No opened input metadata")?;
                if let Some(source) = product.scan_source() {
                    return serde_json::to_value(source).map_err(|e| e.to_string());
                }
                if !matches!(product, Product::Frames { .. }) {
                    return Err("This result does not retain a source scan. Select its original scan for processing.".into());
                }
            }
            Ok(json!({"type":"member","handle":handle,"member":"self"}))
        };
        match job.operation.as_str() {
            "simulate" => {}
            "open" => {
                let input = job.inputs.first().ok_or("Open needs an input")?;
                operation.insert("path".into(), json!(input.path));
                operation.entry("kind").or_insert_with(|| json!(input.kind));
                return Ok(Value::Object(operation));
            }
            "inspect" => {
                return Ok(
                    json!({"type":"inspect","path":job.inputs.first().ok_or("Inspect needs an input")?.path}),
                );
            }
            "combine" => {
                let sources = (0..job.inputs.len())
                    .map(|i| source(i, true))
                    .collect::<Result<Vec<_>, _>>()?;
                operation.insert("sources".into(), json!(sources));
            }
            "save" | "export" => {
                operation.insert("source".into(), source(0, false)?);
                operation.insert("force".into(), Value::Bool(false));
            }
            _ => {
                let index = if matches!(job.operation.as_str(), "project" | "backproject") {
                    let key = if job.operation == "project" {
                        "volume"
                    } else {
                        "projections"
                    };
                    let products: Vec<_> = running.products.iter().collect();
                    projection_geometry(&products, key)
                        .ok_or("Select a source scan alongside the array")?
                } else {
                    0
                };
                operation.insert("source".into(), source(index, true)?);
            }
        }
        if matches!(job.operation.as_str(), "project" | "backproject") {
            let key = if job.operation == "project" {
                "volume"
            } else {
                "projections"
            };
            if !operation.contains_key(key)
                && !fill_projection_array(&mut operation, running.products.iter(), key)?
            {
                return Err(format!(
                    "Select an input containing {key}, or provide its storage reference"
                ));
            }
        }
        if matches!(
            job.operation.as_str(),
            "slice" | "centre_sweep" | "correction_preview" | "region_preview"
        ) {
            return Ok(Value::Object(operation));
        }
        if job.operation == "plan" {
            Ok(json!({"type":"plan","operation":operation}))
        } else {
            let previews =
                self.live.enabled() && matches!(job.operation.as_str(), "reconstruct" | "align");
            Ok(json!({"type":"run","operation":operation,"previews":previews}))
        }
    }

    pub(super) fn result(&mut self, pending: Pending, result: ResultValue) {
        let ResultValue::Output(output) = result else {
            return;
        };
        let job = match &pending {
            Pending::OpenInput { job, .. } | Pending::Execute(job) | Pending::Save { job, .. } => {
                Some(job.clone())
            }
            _ => None,
        };
        if let Some(id) = job {
            if self.shutting_down && !matches!(pending, Pending::Save { .. }) {
                self.release_output(&output);
                return;
            }
            let captured = self
                .project
                .as_ref()
                .and_then(|p| p.document.jobs.iter().find(|j| j.id == id))
                .cloned();
            let Some(captured) = captured else {
                self.release_output(&output);
                return;
            };
            self.spawn_local(Some(id), move |cancelled| {
                let result = check_cancel(&cancelled)
                    .and_then(|()| captured.check_inputs())
                    .and_then(|()| check_cancel(&cancelled));
                LocalEvent::OutputChecked {
                    pending,
                    output,
                    result,
                }
            });
        } else {
            self.result_checked(pending, output);
        }
    }

    pub(super) fn result_checked(&mut self, pending: Pending, output: Box<Output>) {
        if self.shutting_down && matches!(pending, Pending::OpenInput { .. } | Pending::Execute(_))
        {
            self.release_output(&output);
            return;
        }
        match pending {
            Pending::Inspect(id) => {
                if let Product::Inspection { report } = output.value {
                    if let Some(project) = &mut self.project {
                        let root = project.root.clone();
                        if let Some(asset) = project.document.asset_mut(&id) {
                            let path = if asset.path.is_absolute() {
                                asset.path.clone()
                            } else {
                                root.join(&asset.path)
                            };
                            asset.kind = metadata::suggested_kind(&report).into();
                            asset.arrays.clear();
                            self.model
                                .measurements
                                .retain(|measurement| measurement.asset != id);
                            match metadata::inspection_arrays(&report, &path) {
                                Ok(arrays) => {
                                    for (key, mut array) in arrays {
                                        // Paths inside the project stay relative, so a
                                        // moved project still finds its managed data.
                                        array.relative_to(&root);
                                        asset.arrays.insert(key, array);
                                    }
                                }
                                Err(error) => asset.issue = Some(error),
                            }
                            asset.inspection = report;
                        }
                    }
                    self.imaging.invalidate();
                    self.mark_dirty();
                    self.image_key.clear();
                    self.refresh_images();
                }
            }
            Pending::OpenInput { job, index } => {
                if let Some(handle) = output.value.handle() {
                    if let Some(running) = &mut self.running
                        && running.job == job
                        && running.next_input == index
                    {
                        running.handles.push(handle.into());
                        running.products.push(output.value.clone());
                        running.next_input += 1;
                        self.advance_job();
                    }
                } else {
                    self.fail_job(&job, "The input did not return a reusable handle", false);
                }
            }
            Pending::Execute(job) => {
                if let Product::Inspection { report } = &output.value {
                    let mut refresh = false;
                    let input = self
                        .project
                        .as_ref()
                        .and_then(|p| p.document.jobs.iter().find(|j| j.id == job))
                        .and_then(|j| j.inputs.first())
                        .cloned();
                    let root = self.project.as_ref().map(|p| p.root.clone());
                    if let Some(input) = input
                        && let Some(root) = root
                        && let Some(asset) = self
                            .project
                            .as_mut()
                            .and_then(|p| p.document.asset_mut(&input.asset))
                    {
                        // The full inspection knows which stored frames are sample
                        // views, which the header reader may not. Worker results keep
                        // the worker's own registration.
                        if !asset.is_result()
                            && let Ok(arrays) = metadata::inspection_arrays(report, &input.path)
                        {
                            match arrays.get("projections").cloned() {
                                Some(mut projections) => {
                                    projections.relative_to(&root);
                                    asset.arrays.insert("projections".into(), projections);
                                }
                                None => {
                                    asset.arrays.remove("projections");
                                }
                            }
                            refresh = true;
                        }
                        asset.inspection = report.clone();
                        asset.issue = report
                            .get("issues")
                            .and_then(Value::as_array)
                            .filter(|a| !a.is_empty())
                            .map(|a| {
                                a.iter()
                                    .map(|v| v.as_str().unwrap_or("Inspection issue"))
                                    .collect::<Vec<_>>()
                                    .join("; ")
                            });
                    }
                    self.complete_job(&job, None);
                    if refresh {
                        self.image_key.clear();
                        self.refresh_images();
                    }
                } else if matches!(output.value, Product::Frames { .. })
                    && self.project.as_ref().is_some_and(|p| {
                        p.document
                            .jobs
                            .iter()
                            .any(|j| j.id == job && j.operation == "open")
                    })
                {
                    // Open & calibrate records its settings on the source asset;
                    // frames from crop or view selection are saved as new results.
                    let captured = self
                        .project
                        .as_ref()
                        .and_then(|p| p.document.jobs.iter().find(|j| j.id == job))
                        .cloned();
                    if let Some(captured) = captured
                        && let Some(input) = captured.inputs.first()
                    {
                        if let Some(asset) = self
                            .project
                            .as_mut()
                            .and_then(|p| p.document.asset_mut(&input.asset))
                        {
                            asset.kind = "frames".into();
                            asset.open_parameters = captured.parameters.clone();
                            asset.issue = None;
                            asset.provenance = Some(output.manifest.clone());
                        }
                        self.complete_job(&job, Some(input.asset.clone()));
                    }
                    self.release_output(&output);
                } else if matches!(output.value, Product::Region { .. }) {
                    self.persist_arrays(&job, output);
                } else if let Product::Files { paths } = &output.value {
                    self.complete_job(&job, None);
                    self.model.status = format!("Exported {} output files", paths.len());
                } else if matches!(
                    output.value,
                    Product::ReconstructionPlan { .. }
                        | Product::JointPlan { .. }
                        | Product::OppositeViewsPlan { .. }
                        | Product::Validated
                ) {
                    let summary = serde_json::to_string_pretty(&output.value).unwrap_or_default();
                    if let Some(project) = &mut self.project
                        && let Some(j) = project.document.job_mut(&job)
                    {
                        j.stage = summary;
                    }
                    self.complete_job(&job, None);
                } else if let Some(handle) = output.value.handle().map(str::to_owned) {
                    let id = new_id("result");
                    let path = self
                        .project
                        .as_ref()
                        .unwrap()
                        .root
                        .join("derived")
                        .join(format!("{id}.nxs"));
                    if let Some(project) = &mut self.project
                        && let Some(j) = project.document.job_mut(&job)
                    {
                        j.destination = Some(path.clone());
                        j.stage = "Saving result into project".into();
                        j.progress = None;
                    }
                    // A reconstruction opened without its scan has none to save.
                    let include_scan =
                        !matches!(output.value, Product::Reconstruction { scan: None, .. });
                    let command = json!({"type":"run","operation":{"type":"save","source":{"type":"member","handle":handle,"member":"self"},"path":path,"include_scan":include_scan}});
                    if let Err(error) = self.send(
                        command,
                        Pending::Save {
                            job: job.clone(),
                            asset: id,
                            path,
                            output,
                        },
                    ) {
                        self.release_handle(&handle);
                        self.fail_job(&job, &error, false);
                    }
                    self.mark_dirty();
                } else {
                    self.persist_arrays(&job, output);
                }
            }
            Pending::Save {
                job,
                asset,
                path,
                output: original,
            } => {
                if !matches!(output.value, Product::Files { .. }) {
                    self.release_output(&original);
                    self.fail_job(&job, "Saving did not return a committed file set", false);
                    return;
                }
                let mut result = self.result_asset(&job, &asset, &path, &original);
                match metadata::saved_arrays(&original.value, &path, None) {
                    Ok(arrays) => result.arrays = arrays,
                    Err(error) => {
                        self.release_output(&original);
                        self.fail_job(&job, &error, false);
                        return;
                    }
                }
                self.release_output(&original);
                self.complete_asset(&job, result);
                self.start_metadata(&asset);
            }
            Pending::Control { .. } | Pending::ControlStatus { .. } => {}
        }
    }

    pub(super) fn result_asset(
        &self,
        job_id: &str,
        id: &str,
        path: &Path,
        output: &Output,
    ) -> Asset {
        let job = self
            .project
            .as_ref()
            .unwrap()
            .document
            .jobs
            .iter()
            .find(|j| j.id == job_id)
            .unwrap();
        let mut asset = Asset::input(self.project.as_ref().unwrap().stored_path(path), true);
        asset.id = id.into();
        asset.kind = output.value.kind().into();
        // A back projection's parent is the input whose projections it used, which
        // Explore shows beside the volume: the same choice execution made.
        let source = (job.operation == "backproject")
            .then(|| {
                self.running
                    .as_ref()
                    .filter(|running| running.job == job.id)
            })
            .flatten()
            .and_then(|running| array_member(running.products.iter(), "projections"))
            .and_then(|array| job.inputs.get(array.index));
        // Projections given by reference belong to no input, so the result has no
        // parent whose projections Explore would show beside it.
        let referenced = job.operation == "backproject"
            && job
                .parameters
                .get("projections")
                .is_some_and(|v| !v.is_null());
        asset.parent = if referenced {
            None
        } else {
            source
                .or(job.inputs.first())
                .map(|input| input.asset.clone())
        };
        asset.simulated = job.operation == "simulate"
            || job.inputs.iter().any(|i| {
                self.project
                    .as_ref()
                    .unwrap()
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
        asset
    }

    fn persist_arrays(&mut self, job: &str, output: Box<Output>) {
        let id = new_id("result");
        let root = self.project.as_ref().unwrap().root.clone();
        let destination = root.join("derived").join(&id);
        let path = destination.join("array.npy");
        let mut asset = self.result_asset(job, &id, &path, &output);
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
                copies.push((PathBuf::from(source), filename));
                durable.storage = Storage::Npy {
                    path: path.to_string_lossy().into(),
                    artifact: None,
                };
                if let Some(artifact) = artifact {
                    artifacts.push(artifact.clone());
                }
                if copies.len() == 1 {
                    asset.path = self.project.as_ref().unwrap().stored_path(&path);
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
                    copies.push((PathBuf::from(source), filename));
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
        let captured = self
            .project
            .as_ref()
            .unwrap()
            .document
            .jobs
            .iter()
            .find(|j| j.id == job)
            .unwrap()
            .clone();
        if let Some(job) = self.project.as_mut().and_then(|p| p.document.job_mut(job)) {
            job.destination = Some(destination.clone());
            job.stage = "Copying preview arrays into project".into();
            job.progress = None;
        }
        self.mark_dirty();
        let job = job.to_owned();
        self.spawn_local(Some(job.clone()), move |cancelled| {
            let result = (|| -> Result<(), String> {
                check_cancel(&cancelled)?;
                captured.check_inputs()?;
                let stage = tempfile::Builder::new()
                    .prefix(".arrays-")
                    .tempdir_in(root.join("derived"))
                    .map_err(|e| e.to_string())?;
                for (source, filename) in copies {
                    local::copy_file(&source, &stage.path().join(filename), &cancelled)?;
                }
                captured.check_inputs()?;
                check_cancel(&cancelled)?;
                if destination.exists() {
                    return Err("The result destination already exists".into());
                }
                std::fs::rename(stage.path(), &destination).map_err(|e| e.to_string())?;
                Ok(())
            })();
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
                        *path = project
                            .stored_path(Path::new(path))
                            .to_string_lossy()
                            .into();
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
        self.model.active_asset = Some(id.clone());
        self.model.selected_assets.clear();
        self.model.selected_assets.insert(id.clone());
        self.centre_sinogram = true;
        self.applied_window = None;
        self.ui.display.window_low.replace_text("");
        self.ui.display.window_high.replace_text("");
        self.window_prefill = Some([String::new(), String::new()]);
        if let Some(shape) = self.volume_dimensions() {
            self.ui.display.indices = shape.map(|n| n / 2);
        }
        self.image_key.clear();
        self.complete_job(job, Some(id));
        self.refresh_model();
        self.refresh_images();
        self.ui.main_view = self.default_view();
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
        self.close_running(id);
        self.mark_dirty();
    }

    pub(super) fn fail_job(&mut self, id: &str, error: &str, cancelled: bool) {
        if let Some(project) = &mut self.project
            && let Some(job) = project.document.job_mut(id)
        {
            job.status = if cancelled {
                JobStatus::Cancelled
            } else {
                JobStatus::Failed
            };
            job.error = Some(error.into());
            job.stage = if cancelled { "Cancelled" } else { "Failed" }.into();
            job.finished = Some(now_ms());
            self.model.status = format!(
                "{} {}",
                job.name,
                if cancelled { "cancelled" } else { "failed" }
            );
        }
        self.close_running(id);
        self.mark_dirty();
    }

    fn close_running(&mut self, id: &str) {
        if self.running.as_ref().is_some_and(|r| r.job == id) {
            let running = self.running.take().unwrap();
            if let Some(engine) = &mut self.engine {
                for handle in running.handles {
                    if !handle.is_empty() {
                        let _ = engine.close_handle(&handle);
                    }
                }
            }
        }
    }

    pub(super) fn request_failed(&mut self, pending: Pending, error: String, cancelled: bool) {
        match pending {
            Pending::Inspect(id) => {
                if let Some(project) = &mut self.project
                    && let Some(asset) = project.document.asset_mut(&id)
                {
                    asset.issue = Some(error.clone());
                }
                self.model.error = Some(error);
                self.mark_dirty();
            }
            Pending::OpenInput { job, .. } | Pending::Execute(job) => {
                self.fail_job(&job, &error, cancelled)
            }
            Pending::Save { job, output, .. } => {
                self.release_output(&output);
                self.fail_job(&job, &error, cancelled);
            }
            Pending::Control {
                job,
                target,
                action,
            }
            | Pending::ControlStatus {
                job,
                target,
                action,
            } => {
                self.control_failed(&job, &target, &action, error);
            }
        }
    }

    pub(super) fn progress(&mut self, request: &str, record: Progress) {
        let id = match self.pending.get(request) {
            Some(
                Pending::OpenInput { job, .. } | Pending::Execute(job) | Pending::Save { job, .. },
            ) => job.clone(),
            _ => {
                if let Progress::Preview { arrays, .. } = record {
                    self.live.discard(arrays);
                }
                return;
            }
        };
        if let Progress::Preview {
            task, time, arrays, ..
        } = record
        {
            let job = self
                .project
                .as_ref()
                .and_then(|project| project.document.jobs.iter().find(|job| job.id == id));
            if let Some(job) = job
                && matches!(job.operation.as_str(), "reconstruct" | "align")
                && matches!(self.pending.get(request), Some(Pending::Execute(_)))
            {
                let step = job.telemetry.task(request, task).map(|task| task.done);
                self.live.accept(&id, request, task, time, step, arrays);
            } else {
                self.live.discard(arrays);
            }
            return;
        }
        let mut persist = !matches!(record, Progress::Advanced { .. });
        if let Some(job) = self.project.as_mut().and_then(|p| p.document.job_mut(&id)) {
            if let Progress::Advanced { task, time, .. } = &record {
                persist = job
                    .telemetry
                    .task(request, *task)
                    .is_none_or(|task| time.floor() > task.time.floor());
            }
            job.telemetry.apply(request, &record);
            match &record {
                Progress::Paused { .. } => {
                    job.status = JobStatus::Paused;
                }
                Progress::Resumed { .. } => {
                    job.status = JobStatus::Running;
                }
                _ => {}
            }
            if let Some(task) = job.telemetry.current() {
                job.stage = task.title().into();
                job.progress = task.fraction();
                if let Some(seconds) = task.eta {
                    self.model.status = format!(
                        "{} · about {:.0}s active work remaining",
                        task.title(),
                        seconds
                    );
                } else {
                    self.model.status = task.title().into();
                }
            } else {
                // A Finished task does not commit a job; the terminal and durable
                // result save still determine completion.
                job.progress = None;
                self.model.status = format!("{} · waiting for result", job.name);
            }
        }
        if persist {
            self.mark_dirty();
        }
    }

    pub(super) fn release_response(&mut self, response: &Response) {
        if let Response::Progress {
            record: Progress::Preview { arrays, .. },
            ..
        } = response
        {
            self.live.discard(arrays.clone());
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
        // A result handle owns its request's checkpoint artifacts too. The
        // native reader must surrender those leases before the final close.
        if !self.live.retire_handle(handle) {
            self.previews
                .retire_resources(std::collections::BTreeSet::new(), Some(handle.into()));
        }
    }
}

/// Where forward or back projection reads its volume or projections.
pub(super) struct ArraySource {
    /// The input that holds it.
    pub index: usize,
    /// Whether it is a selected array result rather than a member of a scan or
    /// reconstruction.
    pub registered: bool,
    pub storage: crate::protocol::Storage,
    pub descriptor: Option<crate::protocol::ArrayDescriptor>,
}

/// The volume or projections that forward or back projection reads, with its
/// registration when known. A selected array result (a saved volume or forward
/// projection) wins; otherwise the first input that holds one.
pub(super) fn array_member<'a>(
    products: impl Iterator<Item = &'a Product> + Clone,
    key: &str,
) -> Option<ArraySource> {
    products
        .clone()
        .enumerate()
        .find_map(|(index, product)| {
            let array = match product {
                Product::Volume { volume } if key == "volume" => volume,
                Product::Projections { projections } if key == "projections" => projections,
                _ => return None,
            };
            Some(ArraySource {
                index,
                registered: true,
                storage: array.storage.clone(),
                descriptor: Some(array.clone()),
            })
        })
        .or_else(|| {
            // A reconstructed or aligned volume before a simulation's ground-truth
            // phantom, then project order.
            products
                .enumerate()
                .filter_map(|(index, product)| {
                    let members = product.members();
                    let (rank, member, descriptor) = match product {
                        Product::Scan { projections, .. } if key == "projections" => {
                            (0, "projections", Some(projections))
                        }
                        _ if key == "projections" => (1, "projections", None),
                        Product::Reconstruction { volume, .. } => (0, "volume", Some(volume)),
                        Product::Alignment { image, .. } => (1, "image", image.as_ref()),
                        Product::Simulation { phantom, .. } => (2, "phantom", Some(phantom)),
                        _ => (3, "volume", None),
                    };
                    Some((
                        rank,
                        ArraySource {
                            index,
                            registered: false,
                            storage: members.get(member)?.clone(),
                            descriptor: descriptor.cloned(),
                        },
                    ))
                })
                .min_by_key(|(rank, source)| (*rank, source.index))
                .map(|(_, source)| source)
        })
}

/// The input whose scan supplies the geometry. Back projection uses the scan
/// whose projections it reads, unless a projection result is selected. Forward
/// projection prefers a selected scan other than the one holding the volume, so a
/// reconstruction can be projected into another scan's geometry.
pub(super) fn projection_geometry(products: &[&Product], key: &str) -> Option<usize> {
    let array = array_member(products.iter().copied(), key);
    let scans = || {
        products
            .iter()
            .enumerate()
            .filter(|(_, product)| product.scan_source().is_some())
            .map(|(index, _)| index)
    };
    match array {
        Some(array) if !array.registered && key == "projections" => Some(array.index),
        Some(array) if !array.registered => scans()
            .find(|&index| index != array.index)
            .or_else(|| scans().find(|&index| index == array.index)),
        _ => scans().next(),
    }
}

/// Puts the volume or projections into a projection command. Forward projection
/// also takes the volume's own grid, unless one is given, so its voxels keep
/// their registered positions. Returns whether an array was found.
pub(super) fn fill_projection_array<'a>(
    operation: &mut serde_json::Map<String, Value>,
    products: impl Iterator<Item = &'a Product> + Clone,
    key: &str,
) -> Result<bool, String> {
    let scan_unit = products.clone().find_map(|product| match product {
        Product::Scan { projections, .. } => projections
            .axes
            .get(1)
            .map(|axis| axis.unit.clone())
            .filter(|unit| !matches!(unit.as_str(), "pixel" | "index")),
        _ => None,
    });
    let Some(ArraySource {
        storage,
        descriptor,
        ..
    }) = array_member(products, key)
    else {
        return Ok(false);
    };
    // The worker reads the volume in the scan's length unit.
    if key == "volume"
        && let (Some(scan), Some(volume)) = (
            scan_unit,
            descriptor
                .as_ref()
                .and_then(|d| d.axes.first())
                .map(|axis| &axis.unit),
        )
        && !matches!(volume.as_str(), "index" | "voxel")
        && *volume != scan
    {
        return Err(format!(
            "The volume is registered in {volume} and the scan in {scan}; reconstruct the volume from this scan's geometry"
        ));
    }
    operation.insert(
        key.into(),
        serde_json::to_value(storage).map_err(|e| e.to_string())?,
    );
    if key == "volume"
        && operation.get("grid").is_none_or(Value::is_null)
        && let Some(grid) = descriptor.as_ref().and_then(registered_grid)
    {
        operation.insert("grid".into(), grid);
    }
    Ok(true)
}

/// The grid of a volume registered with uniform (z, y, x) voxel-centre axes.
fn registered_grid(volume: &crate::protocol::ArrayDescriptor) -> Option<Value> {
    let names: Vec<_> = volume.axes.iter().map(|axis| axis.name.as_str()).collect();
    if names != ["z", "y", "x"] || volume.shape.len() != 3 {
        return None;
    }
    let mut uniform = volume.axes.iter().map(|axis| match axis.coordinates {
        crate::protocol::Coordinates::Uniform { origin, spacing } => Some((origin, spacing)),
        crate::protocol::Coordinates::Coordinates { .. } => None,
    });
    let (z, y, x) = (uniform.next()??, uniform.next()??, uniform.next()??);
    Some(json!({
        "nx": volume.shape[2], "ny": volume.shape[1], "nz": volume.shape[0],
        "vx": x.1, "vy": y.1, "vz": z.1,
        "vol_origin": [x.0, y.0, z.0],
    }))
}

/// An array-only result (a saved volume or projections) is used as registered,
/// without opening it in the worker; `None` for inputs the worker opens.
pub(super) fn registered_array(input: &crate::project::CapturedInput) -> Option<Product> {
    if matches!(
        input.kind.as_str(),
        "scan" | "frames" | "reconstruction" | "simulation" | "alignment"
    ) {
        return None;
    }
    Some(if let Some(array) = input.arrays.get("volume") {
        Product::Volume {
            volume: array.clone(),
        }
    } else if let Some(array) = input.arrays.get("projections") {
        Product::Projections {
            projections: array.clone(),
        }
    } else {
        Product::Validated
    })
}
