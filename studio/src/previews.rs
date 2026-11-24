//! Temporary interactive operations on the official worker's superseding channels.
//!
//! No project writer or durable queue participates in this controller. An intent
//! changes before admission, so an already-piped old terminal cannot become the
//! image for a newer slider value. Retired files stay alive until native readers
//! have drained; Keep transfers their owner into the persistence pipeline.

mod responses;
mod sources;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::engine::Engine;
use crate::progress::JobProgress;
use crate::project::CapturedInput;
use crate::protocol::{Output, Product, Progress, Response, ResultValue, Storage};
use sources::{SourceState, Sources};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PreviewKind {
    Slice,
    Centres,
    Corrections,
    Region,
}

impl PreviewKind {
    pub fn operation(self) -> &'static str {
        match self {
            Self::Slice => "slice",
            Self::Centres => "centre_sweep",
            Self::Corrections => "correction_preview",
            Self::Region => "region_preview",
        }
    }
    pub fn channel(self) -> &'static str {
        match self {
            Self::Slice => "preview.slice",
            Self::Centres => "preview.centres",
            Self::Corrections => "preview.corrections",
            Self::Region => "preview.region",
        }
    }
    fn accepts(self, product: &Product) -> bool {
        matches!(
            (self, product),
            (Self::Slice, Product::Plane { .. })
                | (Self::Centres, Product::Sweep { .. })
                | (Self::Corrections, Product::CorrectionPreview { .. })
                | (Self::Region, Product::Region { .. })
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PreviewPhase {
    #[default]
    Idle,
    Debouncing,
    Opening,
    Queued,
    Running,
    Paused,
    Ready,
    Failed,
    Cancelled,
}

impl PreviewPhase {
    fn pending(self) -> bool {
        matches!(
            self,
            Self::Debouncing | Self::Opening | Self::Queued | Self::Running | Self::Paused
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct PreviewStatus {
    pub phase: PreviewPhase,
    pub message: String,
    pub progress: Option<f32>,
    pub error_path: Option<String>,
    pub previous: bool,
    pub blocked_by_paused: bool,
    pub request: Option<String>,
    pub revision: Option<u64>,
}

pub struct PreviewResult {
    pub input: CapturedInput,
    pub operation: String,
    pub parameters: Value,
    pub output: Box<Output>,
    pub request: String,
    pub revision: u64,
    pub generation: u64,
    pub telemetry: JobProgress,
}

pub struct KeptPreview {
    pub input: CapturedInput,
    pub operation: String,
    pub parameters: Value,
    pub output: Box<Output>,
    pub request: String,
    pub revision: u64,
    pub telemetry: JobProgress,
}

struct Intent {
    input: CapturedInput,
    source: String,
    parameters: Value,
    generation: u64,
    due: Instant,
    request: Option<String>,
}

#[derive(Default)]
struct Slot {
    generation: u64,
    revision: u64,
    intent: Option<Intent>,
    current: Option<PreviewResult>,
    status: PreviewStatus,
}

enum Pending {
    Source(String),
    Preview(Box<Submission>),
    Status,
    Cleanup,
}

struct Submission {
    kind: PreviewKind,
    generation: u64,
    input: CapturedInput,
    parameters: Value,
    revision: Option<u64>,
    telemetry: JobProgress,
}

struct Retired {
    artifacts: BTreeSet<String>,
    handle: Option<String>,
}

pub struct Previews {
    sources: Sources,
    slots: BTreeMap<PreviewKind, Slot>,
    pending: HashMap<String, Pending>,
    commands: VecDeque<Value>,
    retired: Vec<Retired>,
    requests: BTreeMap<String, String>,
    next_status: Instant,
    debounce: Duration,
}

impl Default for Previews {
    fn default() -> Self {
        Self {
            sources: Sources::default(),
            slots: BTreeMap::new(),
            pending: HashMap::new(),
            commands: VecDeque::new(),
            retired: Vec::new(),
            requests: BTreeMap::new(),
            next_status: Instant::now(),
            debounce: Duration::from_millis(35),
        }
    }
}

impl Previews {
    pub fn ensure_input(&mut self, input: CapturedInput) -> Result<String, String> {
        self.sources.ensure(input)
    }
    pub fn retain_inputs(&mut self, inputs: &[CapturedInput]) -> Result<(), String> {
        self.sources.retain(inputs)
    }
    pub fn source_ref(&self, input: &CapturedInput) -> Result<Option<Storage>, String> {
        self.sources.source_ref(input)
    }
    pub fn source_product(&self, input: &CapturedInput) -> Result<Option<&Product>, String> {
        self.sources.product(input)
    }

    pub fn request(
        &mut self,
        kind: PreviewKind,
        input: CapturedInput,
        parameters: Value,
    ) -> Result<u64, String> {
        if !parameters.is_object() {
            return Err("Preview parameters must be an object".into());
        }
        let source = self.ensure_input(input.clone())?;
        let slot = self.slots.entry(kind).or_default();
        if let Some(intent) = &slot.intent
            && intent.source == source
            && intent.parameters == parameters
        {
            return Ok(intent.generation);
        }
        slot.generation += 1;
        slot.intent = Some(Intent {
            input,
            source,
            parameters,
            generation: slot.generation,
            due: Instant::now() + self.debounce,
            request: None,
        });
        slot.status = PreviewStatus {
            phase: PreviewPhase::Debouncing,
            message: "Updating preview settings".into(),
            previous: slot.current.is_some(),
            ..Default::default()
        };
        Ok(slot.generation)
    }

    /// An explicit Update action may retry unchanged settings. Ordinary frame
    /// polling stays idempotent after failures and cancellation.
    pub fn retry(&mut self, kind: PreviewKind) {
        if let Some(slot) = self.slots.get_mut(&kind) {
            if let Some(intent) = &slot.intent
                && let Some(source) = self.sources.entries.get_mut(&intent.source)
                && matches!(source.state, SourceState::Failed(_))
            {
                source.state = SourceState::Waiting;
            }
            slot.generation += 1;
            slot.intent = None;
            slot.status = PreviewStatus {
                previous: slot.current.is_some(),
                ..Default::default()
            };
        }
    }

    pub fn current(&self, kind: PreviewKind) -> Option<&PreviewResult> {
        self.slots.get(&kind)?.current.as_ref()
    }
    pub fn status(&self, kind: PreviewKind) -> PreviewStatus {
        self.slots
            .get(&kind)
            .map(|slot| slot.status.clone())
            .unwrap_or_default()
    }
    pub fn has_priority_work(&self) -> bool {
        self.slots.values().any(|slot| slot.status.phase.pending())
    }
    pub fn has_work(&self) -> bool {
        !self.pending.is_empty()
            || !self.commands.is_empty()
            || !self.retired.is_empty()
            || self.has_priority_work()
            || self.sources.entries.values().any(|source| {
                matches!(source.state, SourceState::Waiting | SourceState::Opening(_))
            })
    }

    /// Dispatch without waiting for a manifest or numerical work. The single
    /// worker queue remains authoritative; this does not interrupt unrelated jobs.
    pub fn poll(&mut self, engine: &mut Engine) {
        while let Some(command) = self.commands.pop_front() {
            match engine.request(command.clone(), None) {
                Ok(id) => {
                    self.pending.insert(id, Pending::Cleanup);
                }
                Err(_) => {
                    self.commands.push_front(command);
                    return;
                }
            }
        }
        let keep = self
            .slots
            .values()
            .filter(|slot| slot.status.phase.pending())
            .filter_map(|slot| slot.intent.as_ref().map(|intent| intent.source.clone()))
            .collect();
        for source in self.sources.trim(&keep) {
            self.retire_source(source.state);
        }
        let waiting: Vec<_> = self
            .sources
            .entries
            .iter()
            .filter(|(_, source)| matches!(source.state, SourceState::Waiting))
            .map(|(key, _)| key.clone())
            .collect();
        for key in waiting {
            if engine.execution_capacity() == 0 {
                break;
            }
            let source = self.sources.entries.get_mut(&key).unwrap();
            let mut command = source
                .input
                .open_parameters
                .as_object()
                .cloned()
                .unwrap_or_default();
            command.insert("type".into(), json!("open"));
            command.insert("path".into(), json!(source.input.path));
            command.insert("kind".into(), json!(source.input.kind));
            match engine.request(
                Value::Object(command),
                Some(&format!("preview.source.{key}")),
            ) {
                Ok(id) => {
                    source.state = SourceState::Opening(id.clone());
                    self.pending.insert(id, Pending::Source(key));
                }
                Err(error) => source.state = SourceState::Failed(error),
            }
        }
        for (kind, slot) in &mut self.slots {
            let Some(intent) = &mut slot.intent else {
                continue;
            };
            if intent.request.is_some()
                || Instant::now() < intent.due
                || slot.status.phase == PreviewPhase::Failed
            {
                continue;
            }
            let source = match self.sources.source_ref(&intent.input) {
                Ok(Some(source)) => source,
                Ok(None) => {
                    slot.status.phase = PreviewPhase::Opening;
                    slot.status.message = "Opening source for interactive work".into();
                    continue;
                }
                Err(error) => {
                    slot.status.phase = PreviewPhase::Failed;
                    slot.status.message = error;
                    continue;
                }
            };
            let mut command = intent.parameters.as_object().unwrap().clone();
            command.insert("type".into(), json!(kind.operation()));
            command.insert("source".into(), serde_json::to_value(source).unwrap());
            if engine.channel_capacity(kind.channel()) == 0 {
                slot.status.phase = PreviewPhase::Queued;
                slot.status.message = "Waiting for worker admission capacity".into();
                continue;
            }
            match engine.request(Value::Object(command), Some(kind.channel())) {
                Ok(id) => {
                    intent.request = Some(id.clone());
                    slot.status.phase = PreviewPhase::Queued;
                    slot.status.message = "Queued for the numerical worker".into();
                    slot.status.request = Some(id.clone());
                    self.pending.insert(
                        id,
                        Pending::Preview(Box::new(Submission {
                            kind: *kind,
                            generation: intent.generation,
                            input: intent.input.clone(),
                            parameters: intent.parameters.clone(),
                            revision: None,
                            telemetry: JobProgress::default(),
                        })),
                    );
                }
                Err(error) => {
                    slot.status.phase = PreviewPhase::Failed;
                    slot.status.message = error;
                }
            }
        }
        let needs_status = self
            .pending
            .values()
            .any(|pending| matches!(pending, Pending::Source(_) | Pending::Preview(_)));
        if needs_status
            && Instant::now() >= self.next_status
            && !self
                .pending
                .values()
                .any(|pending| matches!(pending, Pending::Status))
            && let Ok(id) = engine.request(json!({"type":"status"}), None)
        {
            self.pending.insert(id, Pending::Status);
            self.next_status = Instant::now() + Duration::from_millis(250);
        }
        self.apply_status();
    }

    pub fn take_for_keep(&mut self, kind: PreviewKind) -> Result<KeptPreview, String> {
        let slot = self
            .slots
            .get_mut(&kind)
            .ok_or("No completed preview to keep")?;
        if slot.status.phase != PreviewPhase::Ready
            || !slot.current.as_ref().is_some_and(|current| {
                slot.intent
                    .as_ref()
                    .is_some_and(|intent| intent.generation == current.generation)
            })
        {
            return Err("Wait for the latest preview before keeping it".into());
        }
        let current = slot.current.take().unwrap();
        slot.intent = None;
        slot.status = PreviewStatus::default();
        Ok(KeptPreview {
            input: current.input,
            operation: current.operation,
            parameters: current.parameters,
            output: current.output,
            request: current.request,
            revision: current.revision,
            telemetry: current.telemetry,
        })
    }

    /// Only explicit UI actions use this control path. Acknowledgement is not a
    /// pause/cancel terminal; progress and cached status establish the outcome.
    pub fn control(
        &mut self,
        kind: PreviewKind,
        action: &str,
        engine: &mut Engine,
    ) -> Result<(), String> {
        let target = self
            .slots
            .get(&kind)
            .and_then(|slot| slot.intent.as_ref())
            .and_then(|intent| intent.request.clone());
        if let Some(target) = target {
            let id = engine.control(&target, action)?;
            self.pending.insert(id, Pending::Cleanup);
            self.slots.get_mut(&kind).unwrap().status.message =
                format!("{action} requested; waiting for a safe point");
            self.next_status = Instant::now();
        } else if action == "cancel" {
            if let Some(slot) = self.slots.get_mut(&kind) {
                slot.intent = None;
                slot.status.phase = PreviewPhase::Cancelled;
                slot.status.message = "Preview cancelled before admission".into();
            }
        } else {
            return Err("This preview has not entered the worker queue".into());
        }
        Ok(())
    }

    /// Root clears or replaces viewer bindings before calling collect_retired.
    pub fn clear(&mut self, kind: PreviewKind) {
        if let Some(slot) = self.slots.get_mut(&kind) {
            slot.generation += 1;
            slot.intent = None;
            slot.status = PreviewStatus::default();
            if let Some(current) = slot.current.take() {
                self.retire_output(current.output);
            }
        }
        for (id, pending) in &self.pending {
            if matches!(pending,Pending::Preview(submission) if submission.kind==kind) {
                self.commands
                    .push_back(json!({"type":"control","target":id,"action":"cancel"}));
            }
        }
    }

    pub fn reset(&mut self) {
        for kind in [
            PreviewKind::Slice,
            PreviewKind::Centres,
            PreviewKind::Corrections,
            PreviewKind::Region,
        ] {
            self.clear(kind);
        }
        for source in self.sources.clear() {
            self.retire_source(source.state);
        }
    }

    pub fn disconnected(&mut self) {
        // Handles, artifact tokens and channel revisions belong to one worker
        // session. A replacement worker starts its revisions at one again.
        *self = Self::default();
    }

    fn retire_source(&mut self, state: SourceState) {
        match state {
            SourceState::Ready(output) => self.retire_output(output),
            SourceState::Opening(id) => self
                .commands
                .push_back(json!({"type":"control","target":id,"action":"cancel"})),
            _ => {}
        }
    }

    fn retire_output(&mut self, output: Box<Output>) {
        let artifacts = output
            .value
            .arrays()
            .into_iter()
            .chain(output.info.arrays.values())
            .flat_map(|array| array.artifacts().map(str::to_owned))
            .collect();
        self.retire_resources(artifacts, output.value.handle().map(str::to_owned));
    }

    pub fn retire_resources(&mut self, artifacts: BTreeSet<String>, handle: Option<String>) {
        if !artifacts.is_empty() || handle.is_some() {
            self.retired.push(Retired { artifacts, handle });
        }
    }

    pub fn collect_retired(
        &mut self,
        engine: &mut Engine,
        in_use: impl Fn(&BTreeSet<String>) -> bool,
    ) {
        let mut remaining = Vec::new();
        for retired in self.retired.drain(..) {
            if in_use(&retired.artifacts) {
                remaining.push(retired);
                continue;
            }
            if let Some(handle) = retired.handle {
                self.commands
                    .push_back(json!({"type":"close","handle":handle}));
            } else {
                for artifact in retired.artifacts {
                    self.commands
                        .push_back(json!({"type":"release","artifact":artifact}));
                }
            }
        }
        self.retired = remaining;
        while let Some(command) = self.commands.pop_front() {
            match engine.request(command.clone(), None) {
                Ok(id) => {
                    self.pending.insert(id, Pending::Cleanup);
                }
                Err(_) => {
                    self.commands.push_front(command);
                    break;
                }
            }
        }
    }
}
