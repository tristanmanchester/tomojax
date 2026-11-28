use super::*;

pub(super) fn footer(state: &AppState) -> El {
    if state.model.preview.is_some() {
        return crate::ui::preview::footer(state);
    }
    let mut content = col().px(22.).py(18.).gap(8.).fill(paper());
    match state.ui.workspace {
        Workspace::Prepare | Workspace::Reconstruct => {
            let name = match state.ui.operation.as_str() {
                "open" => "Open selected scans",
                "reconstruct" => "Queue reconstruction",
                "correct" => "Queue corrections",
                "align" => "Queue alignment",
                "simulate" => "Queue simulation",
                _ => "Queue operation",
            };
            let invalid = state
                .current_validation_error(&state.ui.operation)
                .is_some();
            let available = !state.data_offline()
                && (state.model.backend.ready || state.detached_jobs())
                && (state.model.selection_count() > 0 || state.ui.operation == "simulate")
                && state.model.active().is_none_or(|a| !a.missing)
                && !invalid;
            if state.launcher() == Some("slurm")
                && !matches!(state.ui.operation.as_str(), "plan" | "inspect" | "open")
            {
                content = content
                    .child(label("Scheduler options", 12., muted()))
                    .child(input_hint(
                        "scheduler-options",
                        &state.ui.scheduler_options,
                        "Scheduler options",
                        "",
                        "--time=00:30:00",
                        |s| &mut s.ui.scheduler_options,
                    ));
            }
            if !available {
                let reason = if state.data_offline() {
                    "Not connected"
                } else if !state.model.backend.ready {
                    if state.model.backend.status.is_empty() {
                        "Waiting for TomoJAX"
                    } else {
                        &state.model.backend.status
                    }
                } else if state.model.active().is_some_and(|asset| asset.missing) {
                    "Reconnect the source to queue processing."
                } else if invalid {
                    "Resolve the validation error to queue this operation."
                } else {
                    "Select inputs to queue this operation."
                };
                content = content.child(help(reason).key("queue-unavailable").role("status"));
            }
            content = content.child(
                action_button_icon(
                    "queue-operation",
                    name,
                    Icon::Play,
                    true,
                    available,
                    UiAction::QueueOperation(state.ui.operation.clone()),
                )
                .w_full()
                .h(40.),
            );
        }
        Workspace::Explore => content = content.child(crate::ui::export::submit(state)),
        Workspace::Compare | Workspace::Jobs => return el(),
    }
    col().w_full().child(separator()).child(content)
}
