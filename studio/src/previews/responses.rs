use super::*;

impl Previews {
    /// Return unowned responses to the application. All owned terminals are
    /// consumed, even stale ones, so each published resource gets one owner.
    pub fn consume(&mut self, response: Response, stale: bool) -> Option<Response> {
        let Some(request) = response.request().map(str::to_owned) else {
            return Some(response);
        };
        let Some(mut pending) = self.pending.remove(&request) else {
            return Some(response);
        };
        let terminal = response.terminal();
        if let Response::Progress { record, .. } = &response {
            let state = match record {
                Progress::Started { .. } | Progress::Advanced { .. } | Progress::Resumed { .. } => {
                    Some("running")
                }
                Progress::Paused { .. } => Some("paused"),
                _ => None,
            };
            if let Some(state) = state {
                self.requests.insert(request.clone(), state.into());
            }
        }
        match &mut pending {
            Pending::Cleanup => {}
            Pending::Status => {
                if let Response::Result {
                    result: ResultValue::Status { requests, .. },
                    ..
                } = &response
                {
                    self.requests = requests.clone();
                    self.apply_status();
                }
            }
            Pending::Source(key) => self.source_response(key, &request, response, stale),
            Pending::Preview(submission) => {
                let kind = submission.kind;
                let current = self.slots.get_mut(&kind).filter(|slot| {
                    slot.intent.as_ref().is_some_and(|intent| {
                        intent.generation == submission.generation
                            && intent.request.as_deref() == Some(&request)
                    })
                });
                let admitted = response
                    .channel()
                    .filter(|channel| channel.name == kind.channel())
                    .map(|channel| channel.revision);
                let consistent =
                    admitted.is_none_or(|value| submission.revision.is_none_or(|old| old == value));
                if let Some(value) = admitted {
                    submission.revision = Some(value);
                }
                let valid = if let Some(slot) = current {
                    let valid = !stale
                        && consistent
                        && (admitted.is_some()
                            || matches!(response, Response::ProtocolError { .. }))
                        && admitted.is_none_or(|value| value >= slot.revision);
                    if valid && let Some(value) = admitted {
                        slot.revision = value;
                        slot.status.revision = Some(value);
                    }
                    if terminal && !valid {
                        slot.status.phase = PreviewPhase::Cancelled;
                        slot.status.message = "Preview superseded before publication".into();
                    }
                    valid
                } else {
                    false
                };
                if valid && let Response::Progress { record, .. } = &response {
                    submission.telemetry.apply(&request, record);
                }
                self.preview_response(submission, &request, response, valid, admitted);
            }
        }
        if !terminal {
            self.pending.insert(request, pending);
        } else {
            self.requests.remove(&request);
        }
        None
    }

    fn source_response(&mut self, key: &str, request: &str, response: Response, stale: bool) {
        let current =
            self.sources.entries.get_mut(key).filter(
                |source| matches!(&source.state, SourceState::Opening(id) if id == request),
            );
        match (response, current) {
            (
                Response::Result {
                    result: ResultValue::Output(output),
                    ..
                },
                Some(source),
            ) if !stale => {
                if output.value.handle().is_some() {
                    source.state = SourceState::Ready(output);
                } else {
                    source.state =
                        SourceState::Failed("The source did not return a reusable handle".into());
                    self.retire_output(output);
                }
            }
            (
                Response::Result {
                    result: ResultValue::Output(output),
                    ..
                },
                _,
            ) => self.retire_output(output),
            (Response::Error { error, .. }, Some(source)) => {
                source.state = SourceState::Failed(error.message);
            }
            (Response::ProtocolError { message, .. }, Some(source)) => {
                source.state = SourceState::Failed(message);
            }
            (
                Response::Progress {
                    record: Progress::Preview { arrays, .. },
                    ..
                },
                _,
            ) => self.retire_resources(
                arrays
                    .values()
                    .flat_map(|array| array.artifacts().map(str::to_owned))
                    .collect(),
                None,
            ),
            _ => {}
        }
    }

    fn preview_response(
        &mut self,
        submission: &mut Submission,
        request: &str,
        response: Response,
        valid: bool,
        revision: Option<u64>,
    ) {
        let kind = submission.kind;
        let slot = self.slots.get_mut(&kind).filter(|_| valid);
        match (response, slot) {
            (
                Response::Result {
                    result: ResultValue::Output(output),
                    ..
                },
                mut slot,
            ) => {
                if let (Some(slot), Some(revision)) = (slot.as_deref_mut(), revision)
                    && kind.accepts(&output.value)
                {
                    let old = slot.current.replace(PreviewResult {
                        input: submission.input.clone(),
                        operation: kind.operation().into(),
                        parameters: submission.parameters.clone(),
                        output,
                        request: request.into(),
                        revision,
                        generation: submission.generation,
                        telemetry: std::mem::take(&mut submission.telemetry),
                    });
                    slot.status = PreviewStatus {
                        phase: PreviewPhase::Ready,
                        message: "Temporary preview · Keep to add it to the project".into(),
                        request: Some(request.into()),
                        revision: Some(revision),
                        ..Default::default()
                    };
                    if let Some(old) = old {
                        self.retire_output(old.output);
                    }
                } else {
                    if let Some(slot) = slot {
                        slot.status.phase = PreviewPhase::Failed;
                        slot.status.message =
                            "The worker returned an unexpected preview product".into();
                    }
                    self.retire_output(output);
                }
            }
            (
                Response::Progress {
                    record: Progress::Preview { arrays, .. },
                    ..
                },
                _,
            ) => self.retire_resources(
                arrays
                    .values()
                    .flat_map(|array| array.artifacts().map(str::to_owned))
                    .collect(),
                None,
            ),
            (Response::Progress { record, .. }, Some(slot)) => match record {
                Progress::Started { label, name, .. } => {
                    slot.status.phase = PreviewPhase::Running;
                    slot.status.message = if label.is_empty() { name } else { label };
                }
                Progress::Advanced { done, total, .. } => {
                    slot.status.phase = PreviewPhase::Running;
                    slot.status.progress = total
                        .filter(|value| *value > 0)
                        .map(|value| (done as f32 / value as f32).clamp(0., 1.));
                }
                Progress::Paused { .. } => {
                    slot.status.phase = PreviewPhase::Paused;
                    slot.status.message = "Paused · the numerical queue is blocked".into();
                }
                Progress::Resumed { .. } => {
                    slot.status.phase = PreviewPhase::Running;
                    slot.status.message = "Preview resumed".into();
                }
                Progress::Note { message, .. } => slot.status.message = message,
                _ => {}
            },
            (Response::Error { error, .. }, Some(slot)) => {
                slot.status.phase = if matches!(
                    error.kind,
                    crate::protocol::ErrorKind::Cancelled | crate::protocol::ErrorKind::Superseded
                ) {
                    PreviewPhase::Cancelled
                } else {
                    PreviewPhase::Failed
                };
                slot.status.message = error.message;
                slot.status.error_path = error.path;
            }
            (Response::ProtocolError { message, .. }, Some(slot)) => {
                slot.status.phase = PreviewPhase::Failed;
                slot.status.message = message;
            }
            _ => {}
        }
    }

    pub(super) fn apply_status(&mut self) {
        let paused = self.requests.values().any(|state| state == "paused");
        for slot in self.slots.values_mut() {
            if !matches!(
                slot.status.phase,
                PreviewPhase::Opening
                    | PreviewPhase::Queued
                    | PreviewPhase::Running
                    | PreviewPhase::Paused
            ) {
                continue;
            }
            let target = slot
                .intent
                .as_ref()
                .and_then(|intent| intent.request.as_ref())
                .or_else(|| {
                    slot.intent
                        .as_ref()
                        .and_then(|intent| self.sources.entries.get(&intent.source))
                        .and_then(|source| {
                            if let SourceState::Opening(id) = &source.state {
                                Some(id)
                            } else {
                                None
                            }
                        })
                });
            match target
                .and_then(|target| self.requests.get(target))
                .map(String::as_str)
            {
                Some("queued") => {
                    slot.status.phase = PreviewPhase::Queued;
                    slot.status.blocked_by_paused = paused;
                    slot.status.message = if paused {
                        "Waiting behind paused work · resume or cancel it to continue"
                    } else {
                        "Waiting behind other numerical work"
                    }
                    .into();
                }
                Some("paused") => {
                    slot.status.phase = PreviewPhase::Paused;
                    slot.status.blocked_by_paused = true;
                    slot.status.message = "Paused · this request holds the numerical queue".into();
                }
                Some("pause-requested") => {
                    slot.status.phase = PreviewPhase::Running;
                    slot.status.message = "Pause requested; waiting for a safe point".into();
                }
                Some("running") => {
                    slot.status.phase = PreviewPhase::Running;
                    slot.status.blocked_by_paused = false;
                }
                _ => {}
            }
        }
    }
}
