//! Edit-time preflight uses the worker's exact operation and rejects obsolete drafts.
use super::AppState;
use crate::engine::EngineEvent;
use crate::forms::{self, DraftError, FieldIssue};
use crate::protocol::{ErrorKind, Product, Response, ResultValue, SchemaCatalog};
use crate::ui::model::{JobStatus, Workspace};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::time::{Duration, Instant};

const DEBOUNCE: Duration = Duration::from_millis(350);
const VALIDATE_CHANNEL: &str = "studio.form.validate";
const PLAN_CHANNEL: &str = "studio.form.plan";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Phase {
    #[default]
    Idle,
    Editing,
    Waiting,
    Validating,
    Planning,
    Valid,
    Invalid,
    Unavailable,
}

#[derive(Clone, Debug, Default)]
pub struct ValidationView {
    pub revision: u64,
    pub phase: Phase,
    pub message: String,
    pub issues: Vec<FieldIssue>,
    pub plan: Option<Product>,
    pub plan_signature: Option<Value>,
    pub plan_inputs: Option<Value>,
    pub scope: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Work {
    Validate,
    Plan,
}

#[derive(Clone, Copy, Debug)]
struct Request {
    revision: u64,
    work: Work,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Next {
    #[default]
    Validate,
    Plan,
    Complete,
}

#[derive(Default)]
pub struct Controller {
    signature: Option<Value>,
    changed_at: Option<Instant>,
    next: Next,
    in_flight: Option<String>,
    operation: Option<Value>,
    planned_operation: Option<Value>,
    input_context: Option<Value>,
    requests: HashMap<String, Request>,
    view: ValidationView,
}

impl Controller {
    /// Keep old request ownership while a project or form context is replaced.
    pub(super) fn reset(&mut self) {
        let revision = self.view.revision + 1;
        self.signature = None;
        self.changed_at = None;
        self.next = Next::Validate;
        self.in_flight = None;
        self.operation = None;
        self.planned_operation = None;
        self.input_context = None;
        self.view = ValidationView {
            revision,
            ..Default::default()
        };
    }

    fn observe(&mut self, signature: Value, now: Instant) -> bool {
        if self.signature.as_ref() == Some(&signature) {
            return false;
        }
        self.signature = Some(signature);
        self.changed_at = Some(now);
        self.next = Next::Validate;
        self.in_flight = None;
        self.operation = None;
        self.planned_operation = None;
        self.input_context = None;
        self.view = ValidationView {
            revision: self.view.revision + 1,
            phase: Phase::Editing,
            message: "Waiting for edits to settle…".into(),
            ..Default::default()
        };
        true
    }

    fn settled(&self, now: Instant) -> bool {
        self.changed_at
            .is_some_and(|changed| now.saturating_duration_since(changed) >= DEBOUNCE)
    }

    fn status(&mut self, phase: Phase, message: impl Into<String>) -> bool {
        let message = message.into();
        let changed = self.view.phase != phase || self.view.message != message;
        self.view.phase = phase;
        self.view.message = message;
        changed
    }

    fn submitted(&mut self, id: String, work: Work) {
        self.requests.insert(
            id.clone(),
            Request {
                revision: self.view.revision,
                work,
            },
        );
        self.in_flight = Some(id);
    }

    fn current(&self, id: &str, request: Request, stale: bool) -> bool {
        !stale && request.revision == self.view.revision && self.in_flight.as_deref() == Some(id)
    }
}

/// Read supported nested operation tags from the public schema's discriminated union.
fn accepts_operation(catalog: &SchemaCatalog, wrapper: &str, operation: &str) -> bool {
    fn visit(catalog: &SchemaCatalog, schema: &Value, operation: &str, depth: usize) -> bool {
        if depth > 16 {
            return false;
        }
        if let Some(reference) = schema.get("$ref").and_then(Value::as_str)
            && let Some(name) = reference.strip_prefix("#/$defs/")
        {
            return catalog
                .definition(name)
                .is_some_and(|schema| visit(catalog, schema, operation, depth + 1));
        }
        if schema["properties"]["type"]["enum"]
            .as_array()
            .is_some_and(|tags| tags.iter().any(|tag| tag == operation))
        {
            return true;
        }
        ["anyOf", "oneOf"].iter().any(|key| {
            schema[*key].as_array().is_some_and(|alternatives| {
                alternatives
                    .iter()
                    .any(|item| visit(catalog, item, operation, depth + 1))
            })
        })
    }
    catalog
        .definition(wrapper)
        .is_some_and(|schema| visit(catalog, &schema["properties"]["operation"], operation, 0))
}

impl AppState {
    fn validation_signature(&self) -> Value {
        let edits: HashMap<_, _> = self
            .ui
            .parameter_edits
            .iter()
            .map(|(key, edit)| (key, &edit.text))
            .collect();
        let corrections: HashMap<_, _> = self
            .ui
            .corrections
            .fields
            .iter()
            .map(|(key, edit)| (key, &edit.text))
            .collect();
        json!({
            "processing": matches!(self.ui.workspace, Workspace::Prepare | Workspace::Reconstruct),
            "operation": self.ui.operation, "edits": edits,
            "choices": self.ui.parameter_choices, "toggles": self.ui.parameter_toggles,
            "correction_fields": corrections, "correction_json": self.ui.corrections.json_mode,
            "active": self.model.active_asset, "selected": self.model.selected_assets,
            "active_input": self.current_form_input(),
            "inputs": self.current_form_inputs(),
        })
    }

    pub(super) fn validation_pending(&self) -> bool {
        !self.validation.requests.is_empty()
            || matches!(self.validation.view.phase, Phase::Editing | Phase::Waiting)
                && self.model.backend.ready
    }

    /// A known-invalid draft cannot be submitted, but an old error must not
    /// block corrected text or a newly selected source before the next poll.
    pub(crate) fn current_validation_error(&self, operation: &str) -> Option<String> {
        if operation != self.ui.operation
            || self.validation.view.phase != Phase::Invalid
            || self.validation.signature.as_ref() != Some(&self.validation_signature())
        {
            return None;
        }
        Some(
            self.validation
                .view
                .issues
                .first()
                .map(|issue| format!("{}: {}", issue.path, issue.message))
                .unwrap_or_else(|| self.validation.view.message.clone()),
        )
    }

    /// Called after the asynchronous form-input capture and preview controller poll.
    pub(super) fn poll_validation(&mut self) -> bool {
        let now = Instant::now();
        let processing = matches!(
            self.ui.workspace,
            Workspace::Prepare | Workspace::Reconstruct
        );
        let signature = self.validation_signature();
        let mut changed = self.validation.observe(signature, now);
        self.validation.view.plan_signature = forms::plan_signature(&self.ui, &self.catalog);
        if !processing || self.shutting_down {
            changed |= self.validation.status(Phase::Idle, "");
            self.model.validation = self.validation.view.clone();
            return changed;
        }
        let parameters = match forms::parameters_detailed(&self.ui, &self.catalog) {
            Ok(parameters) => parameters,
            Err(error) => {
                if changed || self.validation.view.phase != Phase::Invalid {
                    self.validation_error(error);
                    changed = true;
                }
                self.model.validation = self.validation.view.clone();
                return changed;
            }
        };
        if !self.model.backend.ready {
            changed |= self
                .validation
                .status(Phase::Waiting, "Waiting for TomoJAX");
            self.model.validation = self.validation.view.clone();
            return changed;
        }
        if !self.validation.settled(now) {
            self.model.validation = self.validation.view.clone();
            return changed;
        }
        changed |= self.advance_validation(parameters);
        self.model.validation = self.validation.view.clone();
        changed
    }

    fn advance_validation(&mut self, parameters: Value) -> bool {
        if self.validation.in_flight.is_some() || self.validation.next == Next::Complete {
            return false;
        }
        if !self.model.backend.ready {
            return self
                .validation
                .status(Phase::Waiting, "Waiting for TomoJAX");
        }
        let operation = if self.ui.operation == "plan" {
            "reconstruct"
        } else {
            &self.ui.operation
        };
        if !self
            .model
            .backend
            .capabilities
            .iter()
            .any(|name| name == "validate")
            || !accepts_operation(&self.catalog, "Validate", operation)
        {
            self.validation.next = Next::Complete;
            return self.validation.status(
                Phase::Unavailable,
                "Edit-time validation is not provided for this operation.",
            );
        }
        if self.form_inputs_pending() && operation != "simulate" {
            return self
                .validation
                .status(Phase::Waiting, "Checking input metadata…");
        }
        if let Some(message) = self.form_input_error().map(str::to_owned)
            && operation != "simulate"
        {
            self.validation.next = Next::Complete;
            return self.validation.status(Phase::Unavailable, message);
        }
        if self.validation.next == Next::Validate {
            match self.validation_operation(parameters) {
                Ok(Some(operation)) => {
                    self.validation.planned_operation =
                        forms::plan_operation(&operation, &self.catalog);
                    self.validation.operation = Some(operation);
                }
                Ok(None) => {
                    return self.validation.status(
                        Phase::Waiting,
                        self.validation_wait_message("Opening input for validation…"),
                    );
                }
                Err(error) => {
                    self.validation_error(error);
                    return true;
                }
            }
        }
        if self.previews.has_priority_work() {
            return self.validation.status(
                Phase::Waiting,
                self.validation_wait_message("Waiting for the interactive preview…"),
            );
        }
        let work = if self.validation.next == Next::Plan {
            Work::Plan
        } else {
            Work::Validate
        };
        let channel = if work == Work::Validate {
            VALIDATE_CHANNEL
        } else {
            PLAN_CHANNEL
        };
        let Some(engine) = self.engine.as_mut() else {
            return self
                .validation
                .status(Phase::Waiting, "Waiting for TomoJAX");
        };
        if engine.channel_capacity(channel) == 0 {
            return self
                .validation
                .status(Phase::Waiting, "Waiting for space in the worker queue…");
        }
        let tag = if work == Work::Validate {
            "validate"
        } else {
            "plan"
        };
        let command = json!({"type": tag, "operation": if work == Work::Validate {
            &self.validation.operation
        } else {
            &self.validation.planned_operation
        }});
        match engine.request(command, Some(channel)) {
            Ok(id) => {
                self.validation.submitted(id, work);
                self.validation.status(
                    Phase::Waiting,
                    self.validation_wait_message(if work == Work::Validate {
                        "Validation queued…"
                    } else {
                        "Plan queued…"
                    }),
                );
            }
            Err(message) => {
                self.validation.next = Next::Complete;
                self.validation.status(Phase::Unavailable, message);
            }
        }
        true
    }

    fn validation_wait_message(&self, otherwise: &str) -> String {
        if self
            .model
            .jobs
            .iter()
            .any(|job| job.status == JobStatus::Paused)
        {
            "Waiting for the paused job to release the worker. Resume or cancel it in Jobs.".into()
        } else if self
            .model
            .jobs
            .iter()
            .any(|job| job.status == JobStatus::Running)
        {
            "Queued behind the running job".into()
        } else {
            otherwise.into()
        }
    }

    fn validation_operation(&mut self, parameters: Value) -> Result<Option<Value>, DraftError> {
        let operation = if self.ui.operation == "plan" {
            "reconstruct"
        } else {
            self.ui.operation.as_str()
        };
        let mut value = parameters.as_object().cloned().unwrap_or_default();
        value.insert("type".into(), json!(operation));
        let definition = forms::definition(operation);
        let schema = self
            .catalog
            .definition(&definition)
            .cloned()
            .unwrap_or_default();
        let properties = &schema["properties"];
        if operation == "simulate" {
            self.validation.view.scope = "Current simulation settings".into();
            return Ok(Some(Value::Object(value)));
        }
        let input = self.current_form_input().ok_or_else(|| {
            DraftError::new("source", "Select an input to validate these settings.")
        })?;
        self.validation.input_context = Some(json!(input));
        if properties.get("path").is_some() {
            value.insert("path".into(), json!(input.path));
            self.validation.view.scope = input.name;
            return Ok(Some(Value::Object(value)));
        }
        let combined = properties.get("sources").is_some();
        let array_key = ["volume", "projections"]
            .into_iter()
            .find(|key| properties.get(*key).is_some());
        let inputs = if combined || array_key.is_some() {
            self.current_form_inputs().to_vec()
        } else {
            vec![input.clone()]
        };
        let mut sources = Vec::new();
        for input in &inputs {
            if array_key.is_some() && super::jobs::registered_array(input).is_some() {
                continue;
            }
            self.previews
                .ensure_input(input.clone())
                .map_err(|message| DraftError::new("source", message))?;
            let source = match self.previews.source_ref(input) {
                Ok(source) => source,
                // Forward and back projection take their geometry from whichever
                // input has a scan, as execution does; an array-only result
                // supplies the volume or projections.
                Err(_)
                    if array_key.is_some()
                        && matches!(self.previews.source_product(input), Ok(Some(_))) =>
                {
                    continue;
                }
                Err(message) => return Err(DraftError::new("source", message)),
            };
            match source {
                Some(source) => sources.push(source),
                None => return Ok(None),
            }
        }
        // The volume or projections as execution passes them: from the opened
        // inputs, in the worker's canonical axis order.
        if let Some(key) = array_key
            && !value.contains_key(key)
        {
            let products = inputs
                .iter()
                .filter_map(|input| {
                    super::jobs::registered_array(input)
                        .or_else(|| self.previews.source_product(input).ok().flatten().cloned())
                })
                .collect::<Vec<_>>();
            if let Some(member) = super::jobs::array_member(products.iter(), key) {
                value.insert(
                    key.into(),
                    serde_json::to_value(member)
                        .map_err(|error| DraftError::new(key, error.to_string()))?,
                );
            }
        }
        if combined {
            value.insert("sources".into(), json!(sources));
            self.validation.view.scope = format!("{} selected inputs", inputs.len());
        } else if properties.get("source").is_some() {
            let source = sources.first().ok_or_else(|| {
                DraftError::new("source", "Select an input with a retained source scan.")
            })?;
            value.insert("source".into(), json!(source));
            self.validation.view.scope = if self.current_form_inputs().len() > 1 {
                format!(
                    "{} · active input; each queued input is checked at execution",
                    input.name
                )
            } else {
                input.name
            };
        }
        Ok(Some(Value::Object(value)))
    }

    fn validation_error(&mut self, error: DraftError) {
        let issue = forms::field_issue(&self.ui, &self.model.parameters, error);
        if issue.control.as_deref() == Some("corrections-json") {
            crate::ui::corrections::reveal_json(&mut self.ui);
        }
        if self.model.parameters.iter().any(|parameter| {
            parameter.name == issue.parameter && self.advanced_parameter(parameter)
        }) {
            self.ui.advanced_open = true;
        }
        let highlighted = issue.control.is_some();
        self.validation.view.issues = vec![issue];
        self.validation.view.plan = None;
        self.validation.view.plan_inputs = None;
        self.validation.next = Next::Complete;
        self.validation.status(
            Phase::Invalid,
            if highlighted {
                "Review the highlighted setting."
            } else {
                "These settings could not be validated."
            },
        );
        // Revealing a JSON editor preserves the submitted values; it is not a new edit.
        self.validation.signature = Some(self.validation_signature());
    }

    /// Returns true only for requests owned by this controller.
    pub(super) fn validation_event(&mut self, event: &EngineEvent) -> bool {
        let EngineEvent::Reply { response, stale } = event else {
            if matches!(event, EngineEvent::Disconnected { .. }) {
                self.validation.requests.clear();
                self.validation.in_flight = None;
                self.validation.next = Next::Complete;
                self.validation.view.plan = None;
                self.validation.view.plan_inputs = None;
                self.validation.status(
                    Phase::Unavailable,
                    "Worker disconnected; settings are not validated.",
                );
                self.model.validation = self.validation.view.clone();
            }
            return false;
        };
        let Some(id) = response.request() else {
            return false;
        };
        let Some(request) = self.validation.requests.get(id).copied() else {
            return false;
        };
        // Input events can precede poll(): reject an old terminal already in the pipe.
        self.validation
            .observe(self.validation_signature(), Instant::now());
        self.validation.view.plan_signature = forms::plan_signature(&self.ui, &self.catalog);
        let current = self.validation.current(id, request, *stale);
        if response.terminal() {
            self.validation.requests.remove(id);
            if !current && self.validation.in_flight.as_deref() == Some(id) {
                self.validation.in_flight = None;
                self.validation.next = Next::Complete;
                self.validation.status(
                    Phase::Unavailable,
                    "Validation was superseded before completion.",
                );
            }
        }
        if !current {
            self.release_response(response);
            self.model.validation = self.validation.view.clone();
            return true;
        }
        match response {
            Response::Progress { .. } => {
                self.validation.status(
                    if request.work == Work::Validate {
                        Phase::Validating
                    } else {
                        Phase::Planning
                    },
                    if request.work == Work::Validate {
                        "Checking settings…"
                    } else {
                        "Resolving reconstruction plan…"
                    },
                );
            }
            Response::Result {
                result: ResultValue::Output(output),
                ..
            } => {
                self.validation.in_flight = None;
                match (&output.value, request.work) {
                    (Product::Validated, Work::Validate) => {
                        let operation = self
                            .validation
                            .planned_operation
                            .as_ref()
                            .and_then(|operation| operation["type"].as_str())
                            .unwrap_or("");
                        if accepts_operation(&self.catalog, "Plan", operation)
                            && self
                                .model
                                .backend
                                .capabilities
                                .iter()
                                .any(|name| name == "plan")
                        {
                            self.validation.next = Next::Plan;
                            self.validation.status(
                                Phase::Waiting,
                                "Settings valid; waiting to refresh the plan…",
                            );
                        } else {
                            self.validation.next = Next::Complete;
                            self.validation
                                .status(Phase::Valid, "Settings validated by TomoJAX.");
                        }
                    }
                    (
                        Product::ReconstructionPlan { .. }
                        | Product::JointPlan { .. }
                        | Product::OppositeViewsPlan { .. },
                        Work::Plan,
                    ) => {
                        self.validation.view.plan = Some(output.value.clone());
                        self.validation.view.plan_inputs = self.validation.input_context.clone();
                        self.validation.next = Next::Complete;
                        self.validation
                            .status(Phase::Valid, "Settings and plan are current.");
                    }
                    _ => {
                        self.validation.next = Next::Complete;
                        self.validation.status(
                            Phase::Unavailable,
                            "The worker returned an unexpected preflight result.",
                        );
                    }
                }
                self.release_response(response);
            }
            Response::Error { error, .. } => {
                self.validation.in_flight = None;
                if matches!(error.kind, ErrorKind::Cancelled | ErrorKind::Superseded) {
                    self.validation.next = Next::Complete;
                    self.validation
                        .status(Phase::Unavailable, "Validation did not complete.");
                } else {
                    self.validation_error(DraftError::new(
                        error.path.clone().unwrap_or_default(),
                        &error.message,
                    ));
                }
            }
            Response::ProtocolError { message, .. } => {
                self.validation.in_flight = None;
                self.validation.next = Next::Complete;
                self.validation.status(Phase::Invalid, message);
            }
            _ => {}
        }
        self.model.validation = self.validation.view.clone();
        true
    }
}

#[cfg(test)]
mod tests;
