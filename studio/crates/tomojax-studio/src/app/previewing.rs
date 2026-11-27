//! Connect edit intents to temporary worker previews without a durable submission.
use super::*;
use crate::previews::{PreviewKind, PreviewPhase};
use crate::protocol::{ArrayDescriptor, Grid};
use std::collections::BTreeMap;

#[derive(Default)]
pub(super) struct PreviewContext {
    pub active: Option<PreviewKind>,
    pub(super) shown: Shown,
    pub(super) signature: String,
    retained: String,
    suppressed: Option<String>,
    plan: Option<(Value, Value, Grid)>,
}

/// What the viewer shows for the active preview's current settings.
#[derive(Default)]
pub(super) enum Shown {
    /// Nothing yet.
    #[default]
    Nothing,
    /// The settings changed: until their result arrives, the image last shown of
    /// `source` in `view` stays on screen.
    Holding {
        source: String,
        view: PlaneKind,
        data: std::sync::Arc<crate::imaging::PlaneData>,
    },
    /// The arrays of the result of worker request `request`.
    Result {
        request: String,
        arrays: BTreeMap<String, ArrayDescriptor>,
    },
}

impl Shown {
    pub(super) fn arrays(&self) -> Option<&BTreeMap<String, ArrayDescriptor>> {
        match self {
            Self::Nothing | Self::Holding { .. } => None,
            Self::Result { arrays, .. } => Some(arrays),
        }
    }

    fn has_image(&self) -> bool {
        self.arrays().is_some_and(|arrays| !arrays.is_empty())
    }

    /// New settings replace a shown result with `held`, the image on screen when the
    /// result was of the same source, until their own result arrives. A held image
    /// stays held through further edits.
    fn supersede(&mut self, held: Option<Self>) {
        *self = match std::mem::take(self) {
            Self::Result { .. } => held.unwrap_or_default(),
            Self::Nothing => Self::Nothing,
            holding @ Self::Holding { .. } => holding,
        };
    }
}

pub(crate) fn kind(operation: &str) -> Option<PreviewKind> {
    match operation {
        "slice" => Some(PreviewKind::Slice),
        "center_sweep" => Some(PreviewKind::Centers),
        "correction_preview" => Some(PreviewKind::Corrections),
        "region_preview" => Some(PreviewKind::Region),
        _ => None,
    }
}

impl AppState {
    pub(crate) fn pending_preview_plane(
        &self,
    ) -> Option<&std::sync::Arc<crate::imaging::PlaneData>> {
        let Shown::Holding { source, view, data } = &self.preview_context.shown else {
            return None;
        };
        let preview = self.model.preview.as_ref()?;
        (self.model.active_asset.as_ref() == Some(source)
            && *view == self.ui.main_view
            && !preview.has_image
            && matches!(
                preview.phase,
                PreviewPhase::Debouncing
                    | PreviewPhase::Opening
                    | PreviewPhase::Queued
                    | PreviewPhase::Running
            ))
        .then_some(data)
    }
    pub(super) fn disconnect_interactive(&mut self) {
        self.previews.disconnected();
        self.preview_context.shown = Shown::Nothing;
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
        self.ui.workspace = Workspace::for_operation(operation);
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
            // Leaving a preview's form stops its request, which would otherwise
            // hold the durable queue with no control left to stop it.
            if let Some(previous) = self.preview_context.active {
                self.previews.clear(previous);
            }
            self.preview_context.active = active;
            self.preview_context.shown = Shown::Nothing;
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
        let inputs = if self.validating_export() {
            self.current_form_input().into_iter().collect()
        } else if !processing {
            Vec::new()
        } else if properties.is_some_and(|p| {
            ["sources", "volume", "projections"]
                .iter()
                .any(|key| p.get(*key).is_some())
        }) {
            // Multi-input forms (combine, forward and back projection) validate
            // with every selected input open at once.
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
            if input.is_none() {
                message = Some("Select a source scan for this preview".to_owned());
            } else if let Err(error) = &parameters {
                message = Some(error.clone());
            } else if !self.model.backend.ready {
                message = Some(self.model.backend.status.clone());
            }
            if let (Some(input), Ok(parameters)) = (input, parameters) {
                let signature = json!([input, parameters, kind.operation()]).to_string();
                if signature != self.preview_context.signature {
                    let held = self
                        .previews
                        .current(kind)
                        .filter(|current| current.input.asset == input.asset)
                        .and_then(|_| self.imaging.plane(self.ui.main_view.mark()))
                        .map(|data| Shown::Holding {
                            source: input.asset.clone(),
                            view: self.ui.main_view,
                            data: data.clone(),
                        });
                    self.preview_context.shown.supersede(held);
                    self.preview_context.signature = signature.clone();
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
                if message.is_none()
                    && self.preview_context.suppressed.as_ref() != Some(&signature)
                    && let Err(error) =
                        self.previews
                            .request(kind, input.clone(), parameters.clone())
                {
                    message = Some(error);
                }
                let plan_signature = forms::plan_signature(&self.ui, &self.catalog);
                let plan_input = match serde_json::to_value(&input) {
                    Ok(input) => Some(input),
                    Err(error) => {
                        message = Some(format!("Could not encode the preview input: {error}"));
                        None
                    }
                };
                if let Some(Product::ReconstructionPlan { grid, .. }) = &self.model.validation.plan
                    && self.model.validation.plan_signature == plan_signature
                    && let (Some(signature), Some(input)) = (&plan_signature, &plan_input)
                    && self.model.validation.plan_inputs.as_ref() == Some(input)
                {
                    self.preview_context.plan =
                        Some((signature.clone(), input.clone(), grid.clone()));
                }
                if let Some(current) = self.previews.current(kind)
                    && self.previews.status(kind).phase == PreviewPhase::Ready
                    && current.parameters == parameters
                    && current.input.asset == input.asset
                    && !matches!(&self.preview_context.shown,
                        Shown::Result { request, .. } if *request == current.request)
                {
                    let arrays = match &current.output.value {
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
                    self.preview_context.shown = Shown::Result {
                        request: current.request.clone(),
                        arrays,
                    };
                    self.ui.display.indices = [0; 3];
                    self.center_sinogram = true;
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
                        Some(signature) == plan_signature.as_ref()
                            && Some(source) == plan_input.as_ref()
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
                        && self.preview_context.shown.has_image(),
                    has_image: self.preview_context.shown.has_image(),
                    index: parameters["index"].as_u64().unwrap_or(0) as usize,
                    count,
                    center: parameters["centers"].as_array().and_then(|values| {
                        (values.len() == 1).then(|| values[0].as_f64()).flatten()
                    }),
                    error_path: status.error_path,
                };
                changed |= self.model.preview.as_ref() != Some(&model);
                self.model.preview = Some(model);
            } else {
                if !self.preview_context.signature.is_empty()
                    || matches!(self.preview_context.shown, Shown::Result { .. })
                {
                    self.previews.clear(kind);
                    self.preview_context.signature.clear();
                    self.preview_context.shown = Shown::Nothing;
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
        self.preview_context.shown = Shown::Nothing;
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
            self.preview_context.shown = Shown::Nothing;
            self.refresh_images();
        }
    }
}
