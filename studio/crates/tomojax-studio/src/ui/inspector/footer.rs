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
            let available = state.operation_available_here(&state.ui.operation)
                && !state.data_offline()
                && state.model.backend.ready
                && (state.model.selection_count() > 0 || state.ui.operation == "simulate")
                && state.model.active().is_none_or(|a| !a.missing)
                && !invalid;
            if !available {
                let reason = if state.data_offline() {
                    "Not connected"
                } else if !state.operation_available_here(&state.ui.operation) {
                    "Processing jobs currently run on this computer. You can inspect scans and keep previews on this connection."
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
