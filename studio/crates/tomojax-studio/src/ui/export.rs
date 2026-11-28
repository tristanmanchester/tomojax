//! Explore's output choices use the worker's format and option descriptions.
use super::{controls::*, model::*, tokens::*};
use crate::{AppState, protocol::Product};
use rune::{El, TextEdit, col};
use serde_json::Value;

pub(super) fn form(state: &AppState) -> El {
    let selected = state.active_export_format();
    let name = match selected {
        "nxs" => "TomoJAX result (.nxs)",
        "array" => "NPY array bundle",
        name => name,
    };
    let mut body = format_picker(state, selected, name);
    let Some(format) = state.interchange_format() else {
        return section("Output", body);
    };
    let path = state
        .ui
        .export_path
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned());
    body = body.child(
        action_button(
            "export-path",
            path.as_deref().unwrap_or("Choose destination…"),
            false,
            true,
            UiAction::Export(selected.into()),
        )
        .w_full(),
    );
    if let Some(Product::Validated { details }) = &state.model.validation.plan
        && state.export_plan_current()
    {
        let files = &details["files"];
        let count = files["count"].as_u64().unwrap_or_default();
        let bytes = details["payload_bytes"].as_u64().unwrap_or_default();
        body = body.child(property(
            "Output",
            format!(
                "{count} file{} · {bytes} bytes",
                if count == 1 { "" } else { "s" }
            ),
        ));
        if let Some(first) = files["first"].as_str() {
            body = body.child(help(first).wrap());
        }
        if let Some(carries) = details["carries"].as_str() {
            body = body.child(help(carries).wrap());
        }
    } else if path.is_some() {
        let validation = &state.model.validation;
        let message = validation
            .issues
            .first()
            .map(|i| i.message.as_str())
            .unwrap_or(&validation.message);
        body = body.child(help(message).wrap());
    }
    let mask = state.model.active().is_some_and(|a| a.mask);
    for field in format.fields(&state.catalog, mask) {
        let key = field.name;
        let fallback = TextEdit::default();
        let edit = state.ui.export_edits.get(&key).unwrap_or(&fallback);
        let hint = match key.as_str() {
            "crop" => "[[0, 128], null, null]".into(),
            "range" => "[low, high]".into(),
            "compress" => "Automatic, true or false".into(),
            _ if !field.choices.is_empty() => field
                .choices
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", "),
            _ => field
                .default
                .as_ref()
                .filter(|v| !v.is_null())
                .map_or_else(|| "Automatic".into(), ToString::to_string),
        };
        let editor = input_hint(&format!("export-{key}"), edit, &key, "", &hint, {
            let key = key.clone();
            move |s| s.ui.export_edits.entry(key.clone()).or_default()
        })
        .help(&field.description);
        body = body.child(col().gap(4.).child(label(&key, 11., muted())).child(editor));
    }
    section("Output", body)
}

fn format_picker(state: &AppState, selected: &str, name: &str) -> El {
    let open = state.ui.choice_menu.as_deref() == Some("export-format");
    let mut body = col().gap(8.).child(
        button(
            "export-format",
            name,
            false,
            selected != "array",
            move |s, _| {
                s.ui.choice_menu = (!open).then(|| "export-format".into());
            },
        )
        .w_full()
        .justify_between()
        .child(icon(Icon::Down, 14., muted())),
    );
    if open {
        let spatial = state
            .model
            .active()
            .is_some_and(|a| a.export == Export::Volume);
        let choices = std::iter::once(("nxs", "TomoJAX result (.nxs)")).chain(
            state
                .catalog
                .export_formats
                .iter()
                .filter(|f| spatial || !f.spatial_only)
                .map(|f| (f.name.as_str(), f.name.as_str())),
        );
        for (key, label) in choices {
            let action = UiAction::SetExportFormat(key.into());
            body = body.child(menu_option(
                &format!("export-format-{key}"),
                label,
                key == selected,
                move |s, tx| s.dispatch(action.clone(), tx),
            ));
        }
    }
    body
}

pub(super) fn submit(state: &AppState) -> El {
    let format = state.active_export_format();
    let selected = state.interchange_format().is_some();
    let queue = selected && state.ui.export_path.is_some();
    let title = if queue {
        "Queue export"
    } else {
        match format {
            "array" => "Export NPY bundle",
            "nxs" => "Save TomoJAX result",
            _ => "Choose destination…",
        }
    };
    action_button_icon(
        "export-result",
        title,
        Icon::Export,
        true,
        !state.data_offline() && !format.is_empty() && (!queue || state.export_plan_current()),
        if queue {
            UiAction::QueueExport
        } else {
            UiAction::Export(format.into())
        },
    )
    .w_full()
    .h(40.)
}
