//! Guided correction drafts with an explicit JSON escape hatch.
//!
//! The bundled engine schema supplies defaults and required fields. Unrecognized
//! steps and properties survive a guided edit, so recipes retain their meaning.
use std::collections::{BTreeMap, BTreeSet};

use rune::{El, TextEdit, col};
use serde_json::{Map, Value};

use super::{controls::*, model::UiState, tokens::*};
use crate::AppState;
use crate::forms::DraftError;
use crate::protocol::SchemaCatalog;

#[derive(Default)]
pub struct CorrectionEditor {
    pub json_mode: bool,
    pub fields: BTreeMap<String, TextEdit>,
    definitions: Value,
}

struct Field {
    name: &'static str,
    title: &'static str,
    unit: &'static str,
}

struct Preset {
    name: &'static str,
    title: &'static str,
    help: &'static str,
    fields: &'static [Field],
}

const PRESETS: &[Preset] = &[
    Preset {
        name: "Zingers",
        title: "Remove zingers",
        help: "Replace isolated high-intensity pixels using a local median.",
        fields: &[
            Field {
                name: "threshold",
                title: "Threshold",
                unit: "open beam",
            },
            Field {
                name: "size",
                title: "Neighborhood",
                unit: "px",
            },
        ],
    },
    Preset {
        name: "Rings",
        title: "Suppress rings",
        help: "Fit detector response over angular bins, then correct every view. Enter null to fit every view.",
        fields: &[Field {
            name: "views",
            title: "Angular bins",
            unit: "bins",
        }],
    },
    Preset {
        name: "Paganin",
        title: "Retrieve phase",
        help: "Paganin retrieval needs the material ratio and sample-plane calibration. In cone beam, divide distance and detector pixel size by magnification.",
        fields: &[
            Field {
                name: "delta_beta",
                title: "Delta / beta",
                unit: "",
            },
            Field {
                name: "distance",
                title: "Effective propagation distance",
                unit: "m",
            },
            Field {
                name: "energy_kev",
                title: "Beam energy",
                unit: "keV",
            },
            Field {
                name: "pixel_size",
                title: "Pixel size at sample",
                unit: "m",
            },
        ],
    },
];

fn definition<'a>(ui: &'a UiState, name: &str) -> &'a Value {
    &ui.corrections.definitions[name]
}
fn field_key(name: &str, field: &str) -> String {
    format!("{name}.{field}")
}
fn text_value(value: &Value) -> String {
    match value {
        Value::Null => "null".into(),
        Value::String(s) => s.clone(),
        value => value.to_string(),
    }
}

fn draft(ui: &UiState) -> Result<Vec<Value>, String> {
    let text = ui
        .parameter_edits
        .get("corrections")
        .map_or("[]", |t| t.text.as_str())
        .trim();
    let value: Value = serde_json::from_str(if text.is_empty() { "[]" } else { text })
        .map_err(|e| format!("Corrections must be a JSON array: {e}"))?;
    value
        .as_array()
        .cloned()
        .ok_or_else(|| "Corrections must be a JSON array of correction objects".into())
}

/// Rebuild field drafts after selecting an operation or loading a recipe.
pub fn reset(ui: &mut UiState, catalog: &SchemaCatalog) {
    let definitions = PRESETS
        .iter()
        .filter_map(|preset| {
            catalog
                .definition(preset.name)
                .map(|schema| (preset.name.into(), schema.clone()))
        })
        .collect::<Map<String, Value>>();
    reset_definitions(ui, Value::Object(definitions));
}

fn reset_definitions(ui: &mut UiState, definitions: Value) {
    let parsed = draft(ui);
    let mut fields = BTreeMap::new();
    let mut seen = BTreeSet::new();
    let repeated = parsed.as_ref().is_ok_and(|steps| {
        steps.iter().any(|step| {
            step.get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| !seen.insert(kind.to_owned()))
        })
    });
    for preset in PRESETS {
        let existing = parsed
            .as_ref()
            .ok()
            .and_then(|steps| steps.iter().find(|step| step["type"] == preset.name));
        for field in preset.fields {
            let value = existing
                .and_then(|step| step.get(field.name))
                .or_else(|| definitions[preset.name]["properties"][field.name].get("default"));
            fields.insert(
                field_key(preset.name, field.name),
                TextEdit::new(value.map(text_value).unwrap_or_default()),
            );
        }
    }
    ui.corrections = CorrectionEditor {
        json_mode: parsed.is_err() || repeated,
        fields,
        definitions,
    };
}

/// Produce the exact array submitted to TomoJAX, preserving correction order.
pub fn parameters(ui: &UiState) -> Result<Value, String> {
    parameters_detailed(ui).map_err(|error| error.message)
}

pub fn parameters_detailed(ui: &UiState) -> Result<Value, DraftError> {
    let mut steps = draft(ui).map_err(|message| DraftError::new("corrections", message))?;
    if ui.corrections.json_mode {
        return Ok(Value::Array(steps));
    }
    for (index, step) in steps.iter_mut().enumerate() {
        let Some(kind) = step.get("type").and_then(Value::as_str).map(str::to_owned) else {
            continue;
        };
        let Some(preset) = PRESETS.iter().find(|preset| preset.name == kind) else {
            continue;
        };
        let object = step.as_object_mut().ok_or_else(|| {
            DraftError::new(
                format!("corrections[{index}]"),
                "Each correction must be an object",
            )
        })?;
        for field in preset.fields {
            let path = format!("corrections[{index}].{}", field.name);
            let key = field_key(&kind, field.name);
            let Some(edit) = ui.corrections.fields.get(&key) else {
                continue;
            };
            let input = edit.text.trim();
            if input.is_empty() {
                let required = definition(ui, &kind)["required"]
                    .as_array()
                    .is_some_and(|items| items.iter().any(|v| v == field.name));
                if required {
                    return Err(DraftError::new(
                        path,
                        format!("{}: {} is required", preset.title, field.title),
                    ));
                }
                object.remove(field.name);
                continue;
            }
            let value: Value = serde_json::from_str(input).map_err(|_| {
                DraftError::new(
                    &path,
                    format!("{}: enter a valid value for {}", preset.title, field.title),
                )
            })?;
            let property = &definition(ui, &kind)["properties"][field.name];
            let valid = match property.get("type").and_then(Value::as_str) {
                Some("number") => value.is_number(),
                Some("integer") => value.is_i64() || value.is_u64(),
                Some("boolean") => value.is_boolean(),
                _ => true,
            };
            if !valid {
                return Err(DraftError::new(
                    path,
                    format!(
                        "{}: {} must be {}",
                        preset.title,
                        field.title,
                        property["type"].as_str().unwrap_or("valid")
                    ),
                ));
            }
            object.insert(field.name.into(), value);
        }
    }
    Ok(Value::Array(steps))
}

/// Resolve an indexed worker error to the guided control, or its ordered JSON editor.
pub fn error_control(ui: &UiState, path: &str) -> String {
    let json = || "corrections-json".to_owned();
    if ui.corrections.json_mode {
        return json();
    }
    let Some(rest) = path.strip_prefix("corrections[") else {
        return json();
    };
    let Some((index, property)) = rest.split_once("].") else {
        return json();
    };
    let Some(index) = index.parse::<usize>().ok() else {
        return json();
    };
    let Ok(steps) = draft(ui) else {
        return json();
    };
    let Some(kind) = steps.get(index).and_then(|step| step["type"].as_str()) else {
        return json();
    };
    let Some(preset) = PRESETS.iter().find(|preset| preset.name == kind) else {
        return json();
    };
    if !preset.fields.iter().any(|field| property == field.name) {
        return json();
    }
    format!("correction-{}", field_key(kind, property))
}

/// Reveal a nested JSON error without losing valid guided edits.
pub fn reveal_json(ui: &mut UiState) {
    if ui.corrections.json_mode {
        return;
    }
    if let Ok(value) = parameters(ui) {
        ui.parameter_edits
            .insert("corrections".into(), TextEdit::new(value.to_string()));
    }
    ui.corrections.json_mode = true;
}

fn set_enabled(ui: &mut UiState, name: &str, enabled: bool) -> Result<(), String> {
    let mut steps = draft(ui)?;
    if enabled {
        if !steps.iter().any(|step| step["type"] == name) {
            let mut step = Map::new();
            step.insert("type".into(), Value::String(name.into()));
            if let Some(properties) = definition(ui, name)["properties"].as_object() {
                for (key, property) in properties {
                    if let Some(value) = property.get("default") {
                        step.insert(key.clone(), value.clone());
                    }
                }
            }
            steps.push(Value::Object(step));
        }
    } else {
        steps.retain(|step| step["type"] != name);
    }
    ui.parameter_edits.insert(
        "corrections".into(),
        TextEdit::new(Value::Array(steps).to_string()),
    );
    Ok(())
}

pub fn view(state: &AppState) -> El {
    let json_mode = state.ui.corrections.json_mode;
    let mut body = col().w_full().gap(12.);
    if json_mode {
        let fallback = TextEdit::new("[]");
        let value = state
            .ui
            .parameter_edits
            .get("corrections")
            .unwrap_or(&fallback);
        body = body.child(super::inspector::field_errors(
            state,
            "corrections-json",
            field(
                "corrections-json",
                value,
                "Ordered correction objects",
                "JSON",
                |s| {
                    s.ui.parameter_edits
                        .entry("corrections".into())
                        .or_default()
                },
            ),
        ));
    } else {
        let steps = draft(&state.ui).unwrap_or_default();
        for preset in PRESETS {
            if definition(&state.ui, preset.name).is_null() {
                continue;
            }
            let enabled = steps.iter().any(|step| step["type"] == preset.name);
            let name = preset.name;
            let mut item = col().w_full().gap(9.).child(
                toggle(
                    &format!("correction-{}", name.to_lowercase()),
                    preset.title,
                    enabled,
                    true,
                    move |s, _| {
                        if let Err(error) = set_enabled(&mut s.ui, name, !enabled) {
                            s.model.error = Some(error);
                        }
                    },
                )
                .help(preset.help),
            );
            if enabled {
                item = item.child(help(preset.help));
                for spec in preset.fields {
                    let key = field_key(name, spec.name);
                    let control = format!("correction-{key}");
                    let fallback = TextEdit::default();
                    let value = state.ui.corrections.fields.get(&key).unwrap_or(&fallback);
                    let editor = if definition(&state.ui, name)["properties"][spec.name]["type"]
                        == "boolean"
                    {
                        let checked = value.text == "true";
                        toggle(&control, spec.title, checked, true, move |s, _| {
                            s.ui.corrections.fields.insert(
                                key.clone(),
                                TextEdit::new(if checked { "false" } else { "true" }),
                            );
                        })
                    } else {
                        field(&control, value, spec.title, spec.unit, move |s| {
                            s.ui.corrections.fields.entry(key.clone()).or_default()
                        })
                    };
                    item = item.child(super::inspector::field_errors(state, &control, editor));
                }
            }
            body = body.child(item);
        }
        if !steps.is_empty() {
            let order = steps
                .iter()
                .filter_map(|step| step["type"].as_str())
                .collect::<Vec<_>>()
                .join(", ");
            body = body.child(label(format!("Order: {order}"), 11., muted()).wrap());
        }
    }
    body.child(button("corrections-mode",if json_mode{"Use guided controls"}else{"Edit corrections as JSON"},false,true,move|s,_|{
        if json_mode {
            match draft(&s.ui) {
                Ok(_)=>{let definitions=s.ui.corrections.definitions.clone();reset_definitions(&mut s.ui,definitions);if s.ui.corrections.json_mode{s.model.error=Some("Repeated correction steps require JSON mode to preserve their individual settings".into());}},
                Err(error)=>s.model.error=Some(error),
            }
        }else{
            match parameters(&s.ui){
                Ok(value)=>{s.ui.parameter_edits.insert("corrections".into(),TextEdit::new(value.to_string()));s.ui.corrections.json_mode=true;},
                Err(error)=>s.model.error=Some(error),
            }
        }
    }).w_full())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn catalog() -> SchemaCatalog {
        SchemaCatalog::new(
            serde_json::from_str(include_str!(
                "../../../../../docs/protocol/operations.schema.json"
            ))
            .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn guided_edits_preserve_order_and_unrepresented_corrections() {
        let mut ui = UiState::default();
        ui.parameter_edits.insert("corrections".into(),TextEdit::new(r#"[{"type":"BeamHardening","rates":[1,2],"weights":[0.5,0.5]},{"type":"Zingers","threshold":0.2,"size":5}]"#));
        reset(&mut ui, &catalog());
        ui.corrections
            .fields
            .insert("Zingers.size".into(), TextEdit::new("7"));
        assert_eq!(
            parameters(&ui).unwrap(),
            json!([
                {"type":"BeamHardening","rates":[1,2],"weights":[0.5,0.5]},
                {"type":"Zingers","threshold":0.2,"size":7}
            ])
        );
    }

    #[test]
    fn phase_retrieval_requires_explicit_calibration_and_can_be_removed() {
        let mut ui = UiState::default();
        reset(&mut ui, &catalog());
        set_enabled(&mut ui, "Paganin", true).unwrap();
        assert!(
            parameters(&ui)
                .unwrap_err()
                .contains("Delta / beta is required")
        );
        set_enabled(&mut ui, "Paganin", false).unwrap();
        assert_eq!(parameters(&ui).unwrap(), json!([]));
    }

    #[test]
    fn repeated_steps_keep_distinct_settings_in_json_mode() {
        let input = json!([{"type":"Zingers","threshold":0.1},{"type":"Zingers","threshold":0.3}]);
        let mut ui = UiState::default();
        ui.parameter_edits
            .insert("corrections".into(), TextEdit::new(input.to_string()));
        reset(&mut ui, &catalog());
        assert!(ui.corrections.json_mode);
        assert_eq!(parameters(&ui).unwrap(), input);
    }

    #[test]
    fn explicit_null_is_preserved_in_a_guided_correction() {
        let mut ui = UiState::default();
        ui.parameter_edits.insert(
            "corrections".into(),
            TextEdit::new(r#"[{"type":"Rings","views":null}]"#),
        );
        reset(&mut ui, &catalog());
        assert_eq!(ui.corrections.fields["Rings.views"].text, "null");
        let value = parameters(&ui).unwrap();
        assert_eq!(
            value[0].get("views"),
            Some(&Value::Null),
            "omitting views would restore the worker's 512-bin default"
        );
    }

    #[test]
    fn guided_defaults_come_from_the_current_worker_description() {
        let mut document = catalog().document().clone();
        document["$defs"]["Zingers"]["properties"]["threshold"]["default"] = json!(0.37);
        let catalog = SchemaCatalog::new(document).unwrap();
        let mut ui = UiState::default();
        reset(&mut ui, &catalog);
        set_enabled(&mut ui, "Zingers", true).unwrap();
        assert_eq!(parameters(&ui).unwrap()[0]["threshold"], 0.37);
    }
}
