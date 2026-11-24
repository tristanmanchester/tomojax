//! Connect edit intents to temporary worker previews without a durable submission.
use super::*;
use crate::previews::{PreviewKind, PreviewPhase};
use crate::protocol::{ArrayDescriptor, Grid};
use std::collections::BTreeMap;

#[derive(Default)]
pub(super) struct PreviewContext {
    pub active: Option<PreviewKind>,
    pub arrays: BTreeMap<String, ArrayDescriptor>,
    pub(super) signature: String,
    retained: String,
    displayed: Option<String>,
    suppressed: Option<String>,
    plan: Option<(Value, Value, Grid)>,
}

pub(crate) fn kind(operation: &str) -> Option<PreviewKind> {
    match operation {
        "slice" => Some(PreviewKind::Slice),
        "centre_sweep" => Some(PreviewKind::Centres),
        "correction_preview" => Some(PreviewKind::Corrections),
        "region_preview" => Some(PreviewKind::Region),
        _ => None,
    }
}

impl AppState {
    pub(super) fn disconnect_interactive(&mut self) {
        self.previews.disconnected();
        self.preview_context.arrays.clear();
        self.preview_context.displayed = None;
        self.preview_context.signature.clear();
        self.preview_context.retained.clear();
        self.preview_context.plan = None;
        self.model.preview = None;
        self.refresh_images();
    }

    pub(crate) fn preview_blocker(&self) -> Option<&JobView> {
        let running = self.running.as_ref()?;
        self.model.jobs.iter().find(|job| {
            job.id == running.job && matches!(job.status, JobStatus::Running | JobStatus::Paused)
        })
    }

    /// An explicit refresh uses the same channel as automatic edits.
    pub(super) fn request_interactive(&mut self, operation: &str) -> Result<(), String> {
        if kind(operation).is_none() {
            return Err("Choose an interactive preview operation".into());
        }
        if !self.model.backend.ready {
            return Err("Wait for the TomoJAX worker to become ready".into());
        }
        if !matches!(
            self.ui.workspace,
            Workspace::Prepare | Workspace::Reconstruct
        ) {
            self.ui.workspace = Workspace::Reconstruct;
        }
        self.preview_context.suppressed = None;
        if self.ui.operation != operation {
            self.set_operation(operation, None);
        }
        self.poll_form_context();
        self.poll_interactive();
        Ok(())
    }

    pub(super) fn poll_interactive(&mut self) -> bool {
        if self.shutting_down {
            return false;
        }
        let processing = matches!(
            self.ui.workspace,
            Workspace::Prepare | Workspace::Reconstruct
        );
        let active = processing.then(|| kind(&self.ui.operation)).flatten();
        let mut changed = false;
        if self.preview_context.active != active {
            self.preview_context.active = active;
            self.preview_context.arrays.clear();
            self.preview_context.displayed = None;
            self.preview_context.signature.clear();
            self.preview_context.suppressed = None;
            self.model.preview = None;
            if active.is_none() {
                self.ui.main_view = self.default_view();
            }
            self.refresh_images();
            changed = true;
        }

        // Only the inputs used by this form stay pinned. Selection alone does not
        // load every scan in a batch into device memory.
        let definition = forms::definition(&self.ui.operation);
        let properties = self
            .catalog
            .definition(&definition)
            .map(|schema| &schema["properties"]);
        let inputs = if !processing || self.form_inputs_pending() {
            Vec::new()
        } else if properties.is_some_and(|p| p.get("sources").is_some()) {
            self.current_form_inputs().to_vec()
        } else if properties.is_some_and(|p| p.get("source").is_some()) {
            self.current_form_input().into_iter().collect()
        } else {
            Vec::new()
        };
        let retained = serde_json::to_string(&inputs).unwrap_or_default();
        if retained != self.preview_context.retained {
            if let Err(error) = self.previews.retain_inputs(&inputs) {
                self.model.error = Some(error);
            }
            self.preview_context.retained = retained;
            changed = true;
        }

        if let Some(kind) = active {
            let input = self.current_form_input();
            let parameters = forms::parameters(&self.ui, &self.catalog);
            let mut message = None;
            if self.form_inputs_pending() {
                message = Some("Checking source metadata".to_owned());
            } else if let Some(error) = self.form_input_error() {
                message = Some(error.to_owned());
            } else if input.is_none() {
                message = Some("Select a source scan for this preview".to_owned());
            } else if let Err(error) = &parameters {
                message = Some(error.clone());
            } else if !self.model.backend.ready {
                message = Some(self.model.backend.status.clone());
            }
            if let (Some(input), Ok(parameters)) = (input, parameters) {
                let signature = json!([input, parameters, kind.operation()]).to_string();
                if signature != self.preview_context.signature {
                    self.preview_context.signature = signature.clone();
                    self.preview_context.arrays.clear();
                    self.preview_context.displayed = None;
                    // The pending image has no pixels from a previous parameter value.
                    self.ui.main_view = if kind == PreviewKind::Corrections {
                        PlaneKind::Projection
                    } else {
                        match parameters["axis"].as_str().unwrap_or("z") {
                            "x" => PlaneKind::Sagittal,
                            "y" => PlaneKind::Coronal,
                            _ => PlaneKind::Axial,
                        }
                    };
                    self.refresh_images();
                    changed = true;
                }
                if message.is_none() && self.preview_context.suppressed.as_ref() != Some(&signature)
                {
                    if let Err(error) =
                        self.previews
                            .request(kind, input.clone(), parameters.clone())
                    {
                        message = Some(error);
                    }
                }
                let plan_signature = forms::plan_signature(&self.ui, &self.catalog);
                let plan_input = serde_json::to_value(&input).expect("captured preview input");
                if let Some(Product::ReconstructionPlan { grid, .. }) = &self.model.validation.plan
                    && self.model.validation.plan_signature == plan_signature
                    && self.model.validation.plan_inputs.as_ref() == Some(&plan_input)
                    && let Some(signature) = plan_signature.clone()
                {
                    self.preview_context.plan = Some((signature, plan_input.clone(), grid.clone()));
                }
                if let Some(current) = self.previews.current(kind)
                    && self.previews.status(kind).phase == PreviewPhase::Ready
                    && current.parameters == parameters
                    && current.input.asset == input.asset
                    && self.preview_context.displayed.as_deref() != Some(&current.request)
                {
                    self.preview_context.arrays = match &current.output.value {
                        Product::Plane { plane } => {
                            BTreeMap::from([("plane".into(), plane.clone())])
                        }
                        Product::Sweep { planes } => {
                            BTreeMap::from([("sweep".into(), planes.clone())])
                        }
                        Product::CorrectionPreview {
                            before,
                            after,
                            views,
                        } => BTreeMap::from([
                            ("before".into(), before.clone()),
                            ("after".into(), after.clone()),
                            ("views".into(), views.clone()),
                        ]),
                        Product::Region { arrays, .. } => arrays.clone(),
                        _ => BTreeMap::new(),
                    };
                    self.preview_context.displayed = Some(current.request.clone());
                    self.ui.display.indices = [0; 3];
                    self.centre_sinogram = true;
                    self.refresh_images();
                    changed = true;
                }
                let status = self.previews.status(kind);
                let axis = parameters["axis"].as_str().unwrap_or("z");
                let count = self
                    .preview_context
                    .plan
                    .as_ref()
                    .filter(|(signature, source, _)| {
                        Some(signature) == plan_signature.as_ref() && *source == plan_input
                    })
                    .map(|(_, _, grid)| match axis {
                        "x" => grid.nx,
                        "y" => grid.ny,
                        _ => grid.nz,
                    });
                let model = PreviewView {
                    operation: kind.operation().into(),
                    message: message.unwrap_or_else(|| {
                        if self.preview_context.suppressed.as_ref() == Some(&signature) {
                            "Preview stopped. Change a setting or choose Update preview.".into()
                        } else if status.message.is_empty() {
                            "Preparing preview".into()
                        } else {
                            status.message.clone()
                        }
                    }),
                    phase: status.phase,
                    revision: status.revision,
                    can_keep: status.phase == PreviewPhase::Ready
                        && !self.preview_context.arrays.is_empty(),
                    has_image: !self.preview_context.arrays.is_empty(),
                    index: parameters["index"].as_u64().unwrap_or(0) as usize,
                    count,
                    centre: parameters["centres"].as_array().and_then(|values| {
                        (values.len() == 1).then(|| values[0].as_f64()).flatten()
                    }),
                    error_path: status.error_path,
                };
                changed |= self.model.preview.as_ref() != Some(&model);
                self.model.preview = Some(model);
            } else {
                if !self.preview_context.signature.is_empty()
                    || !self.preview_context.arrays.is_empty()
                    || self.preview_context.displayed.is_some()
                {
                    self.previews.clear(kind);
                    self.preview_context.signature.clear();
                    self.preview_context.arrays.clear();
                    self.preview_context.displayed = None;
                    self.refresh_images();
                    changed = true;
                }
                self.model.preview = Some(PreviewView {
                    operation: kind.operation().into(),
                    message: message.unwrap_or_default(),
                    ..Default::default()
                });
            }
        }
        if self.model.backend.ready
            && let Some(engine) = &mut self.engine
        {
            self.previews.poll(engine);
        }
        changed
    }

    pub(super) fn keep_interactive(&mut self) -> Result<(), String> {
        // Text edits and source selection can occur before the next frame poll.
        // Observe that intent before transferring any completed result.
        self.poll_form_context();
        self.poll_interactive();
        if !self
            .model
            .preview
            .as_ref()
            .is_some_and(|preview| preview.can_keep)
        {
            return Err("Wait for the current settings and source to produce a preview".into());
        }
        let kind = self
            .preview_context
            .active
            .ok_or("No interactive preview is selected")?;
        let preview = self.previews.take_for_keep(kind)?;
        self.preview_context.suppressed = Some(self.preview_context.signature.clone());
        self.preview_context.arrays.clear();
        self.preview_context.displayed = None;
        self.preview_context.active = None;
        self.model.preview = None;
        self.ui.workspace = Workspace::Explore;
        self.refresh_images();
        let id = self.keep_preview(preview)?;
        self.ui.selected_job = Some(id);
        Ok(())
    }

    pub(super) fn cancel_interactive(&mut self) {
        if let Some(kind) = self.preview_context.active {
            self.previews.clear(kind);
            self.preview_context.suppressed = Some(self.preview_context.signature.clone());
            self.preview_context.arrays.clear();
            self.preview_context.displayed = None;
            self.refresh_images();
        }
    }
}
