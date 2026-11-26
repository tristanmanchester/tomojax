//! Translate the worker's public schema into editable, typed draft parameters.
use crate::protocol::SchemaCatalog;
use crate::ui::model::{Choice, ParameterKind, ParameterView, UiState};
use rune::TextEdit;
use serde_json::{Map, Value};

/// A draft can be incomplete without becoming a global application error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftError {
    pub path: String,
    pub message: String,
}

impl DraftError {
    pub fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

/// The exact worker path and the existing editor that owns that value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldIssue {
    pub path: String,
    pub parameter: String,
    pub control: Option<String>,
    pub message: String,
}

pub fn field_issue(ui: &UiState, parameters: &[ParameterView], error: DraftError) -> FieldIssue {
    let parameter = parameters
        .iter()
        .filter(|parameter| {
            error.path == parameter.name
                || error.path.starts_with(&format!("{}.", parameter.name))
                || error.path.starts_with(&format!("{}[", parameter.name))
        })
        .max_by_key(|parameter| parameter.name.len());
    let name = parameter.map_or_else(String::new, |parameter| parameter.name.clone());
    let control = parameter.map(|parameter| {
        if parameter.name == "corrections" {
            crate::ui::corrections::error_control(ui, &error.path)
        } else {
            format!("parameter-{}", parameter.name)
        }
    });
    FieldIssue {
        path: error.path,
        parameter: name,
        control,
        message: error.message,
    }
}

pub fn definition(operation: &str) -> String {
    if operation == "plan" {
        return "Reconstruct".into();
    }
    operation
        .split('_')
        .map(|word| {
            let mut c = word.chars();
            c.next()
                .map(|a| a.to_uppercase().to_string() + c.as_str())
                .unwrap_or_default()
        })
        .collect()
}

fn plan_definition(operation: &str) -> Option<(&'static str, &'static str)> {
    match operation {
        "reconstruct" | "plan" | "slice" | "center_sweep" => Some(("reconstruct", "Reconstruct")),
        "align" => Some(("align", "Align")),
        _ => None,
    }
}

/// Slice and center-sweep plans use the full reconstruction's declared fields.
pub fn plan_operation(operation: &Value, catalog: &SchemaCatalog) -> Option<Value> {
    let (tag, definition) = plan_definition(operation["type"].as_str()?)?;
    let properties = catalog.definition(definition)?["properties"].as_object()?;
    let mut values: Map<String, Value> = operation
        .as_object()?
        .iter()
        .filter(|(name, _)| properties.contains_key(*name))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    values.insert("type".into(), Value::String(tag.into()));
    Some(Value::Object(values))
}

/// A source-independent key for retaining a real plan while only an index moves.
pub fn plan_signature(ui: &UiState, catalog: &SchemaCatalog) -> Option<Value> {
    let (tag, definition) = plan_definition(&ui.operation)?;
    let properties = catalog.definition(definition)?["properties"].as_object()?;
    let mut values = Map::new();
    values.insert("type".into(), Value::String(tag.into()));
    for name in properties
        .keys()
        .filter(|name| name.as_str() != "type" && name.as_str() != "source")
    {
        values.insert(
            name.clone(),
            serde_json::json!({
                "text": ui.parameter_edits.get(name).map(|edit| &edit.text),
                "choice": ui.parameter_choices.get(name),
                "toggle": ui.parameter_toggles.get(name),
            }),
        );
    }
    Some(Value::Object(values))
}

fn scalar_schema(schema: &Value) -> &Value {
    if let Some(alternatives) = schema.get("anyOf").and_then(Value::as_array)
        && let Some(value) = alternatives
            .iter()
            .find(|v| v.get("type").and_then(Value::as_str) != Some("null"))
    {
        return scalar_schema(value);
    }
    schema
}

/// Whether a schema, or one of its alternatives, takes JSON of type `kind`
/// ("array", "object" or "string"; a `$ref` is a structured object).
fn accepts(schema: &Value, kind: &str) -> bool {
    schema.get("type").and_then(Value::as_str) == Some(kind)
        || (kind == "object" && schema.get("$ref").is_some())
        || schema
            .get("anyOf")
            .and_then(Value::as_array)
            .is_some_and(|alternatives| alternatives.iter().any(|a| accepts(a, kind)))
}

fn label(key: &str) -> String {
    match key {
        "tv_weight" => "TV regularization".into(),
        "n_views" => "Number of views".into(),
        "source_to_axis" => "Source to axis".into(),
        "source_to_detector" => "Source to detector".into(),
        "translation_sigma" => "Per-view shift (SD)".into(),
        "nonnegative" => "Nonnegative constraint".into(),
        "fit" => "Geometry parameters to fit".into(),
        "data_path" => "HDF5 data dataset".into(),
        "angles_path" => "HDF5 angles dataset".into(),
        "poses" => "Pose table (storage reference or null)".into(),
        _ => {
            let text = key.replace('_', " ");
            let mut chars = text.chars();
            chars
                .next()
                .map(|c| c.to_uppercase().to_string() + chars.as_str())
                .unwrap_or_default()
        }
    }
}

pub fn initialize(
    ui: &mut UiState,
    catalog: &SchemaCatalog,
    operation: &str,
    values: Option<&Value>,
) -> Vec<ParameterView> {
    ui.operation = operation.into();
    ui.parameter_edits.clear();
    ui.parameter_choices.clear();
    ui.parameter_toggles.clear();
    let mut views = Vec::new();
    for p in catalog.parameters(&definition(operation)) {
        if ["source", "sources", "path", "force"].contains(&p.name.as_str()) {
            continue;
        }
        let schema = scalar_schema(&p.schema);
        let data_type = schema.get("type").and_then(Value::as_str).unwrap_or("");
        let nullable = p
            .schema
            .get("anyOf")
            .and_then(Value::as_array)
            .is_some_and(|a| {
                a.iter()
                    .any(|v| v.get("type").and_then(Value::as_str) == Some("null"))
            })
            || p.default == Some(Value::Null);
        let mut choices = if !p.choices.is_empty() {
            p.choices.clone()
        } else {
            schema
                .get("enum")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        };
        if data_type == "boolean" {
            choices = vec![Value::Bool(true), Value::Bool(false)];
        }
        if nullable && !choices.is_empty() {
            choices.insert(0, Value::Null);
        }
        let default = values
            .and_then(|v| v.get(&p.name))
            .cloned()
            .or(p.default.clone())
            .unwrap_or(Value::Null);
        let text = match &default {
            Value::Null => String::new(),
            Value::String(s) => s.clone(),
            v => v.to_string(),
        };
        let kind = if !choices.is_empty() {
            ParameterKind::Choice
        } else {
            match data_type {
                "integer" => ParameterKind::Integer,
                "number" => ParameterKind::Number,
                _ => ParameterKind::Text,
            }
        };
        ui.parameter_edits
            .insert(p.name.clone(), TextEdit::new(&text));
        ui.parameter_choices.insert(p.name.clone(), text);
        let advanced = matches!(
            p.name.as_str(),
            "config"
                | "grid"
                | "detector"
                | "geometry"
                | "region"
                | "devices"
                | "seed"
                | "warm_start"
                | "views_per_batch"
                | "artifacts"
                | "image_key_path"
                | "data_path"
                | "angles_path"
                | "flat_exposure_times"
                | "exposure_times"
                | "volume"
                | "projections"
        );
        let structured =
            !matches!(data_type, "integer" | "number" | "string" | "boolean") && choices.is_empty();
        // The configuration value a null setting takes, when the schema states it.
        let effective_default = p.schema.get("x-default").cloned();
        views.push(ParameterView {
            name: p.name.clone(),
            label: label(&p.name),
            help: if structured {
                format!(
                    "{} Enter a JSON value; blank uses the TomoJAX default.",
                    p.description
                )
            } else {
                p.description
            },
            unit: crate::ui::unit_name(&p.unit.unwrap_or_default()).to_owned(),
            kind,
            advanced,
            required: p.required,
            placeholder: if p.required || !matches!(p.default, None | Some(Value::Null)) {
                String::new()
            } else {
                "Default".into()
            },
            inherited: effective_default,
            choices: choices
                .iter()
                .map(|v| Choice {
                    value: match v {
                        Value::String(s) => s.clone(),
                        Value::Null => String::new(),
                        v => v.to_string(),
                    },
                    label: match v {
                        Value::String(s) if p.name == "method" => crate::operations::method_name(s),
                        Value::String(s) if s == "auto" => "Automatic".into(),
                        Value::String(s) if p.name == "filter" => match s.as_str() {
                            "ramp" => "Ramp",
                            "hann" => "Hann",
                            "shepp-logan" => "Shepp–Logan",
                            other => other,
                        }
                        .into(),
                        Value::String(s) => sentence(&s.replace('_', " ")),
                        Value::Null => "Default".into(),
                        Value::Bool(true) => "Enabled".into(),
                        Value::Bool(false) => "Disabled".into(),
                        v => v.to_string(),
                    },
                })
                .collect(),
        });
    }
    let preferred = [
        "kind",
        "phantom",
        "size",
        "n_views",
        "method",
        "iterations",
        "filter",
        "tv_weight",
        "nonnegative",
        "fit",
        "motion",
        "corrections",
        "binning",
        "rows",
        "cols",
        "axis",
        "index",
        "centers",
    ];
    views.sort_by_key(|p| {
        (
            p.advanced,
            preferred
                .iter()
                .position(|key| *key == p.name)
                .unwrap_or(100),
            p.name.clone(),
        )
    });
    crate::ui::corrections::reset(ui, catalog);
    views
}

/// Refresh described controls without parsing or discarding unfinished drafts.
pub fn refresh_catalog(ui: &mut UiState, catalog: &SchemaCatalog) -> Vec<ParameterView> {
    let operation = ui.operation.clone();
    let edits = std::mem::take(&mut ui.parameter_edits);
    let choices = std::mem::take(&mut ui.parameter_choices);
    let toggles = std::mem::take(&mut ui.parameter_toggles);
    let corrections = std::mem::take(&mut ui.corrections);
    let views = initialize(ui, catalog, &operation, None);
    for (name, draft) in edits {
        if let Some(edit) = ui.parameter_edits.get_mut(&name) {
            *edit = draft;
        }
    }
    for (name, draft) in choices {
        if let Some(choice) = ui.parameter_choices.get_mut(&name) {
            *choice = draft;
        }
    }
    for (name, draft) in toggles {
        if ui.parameter_edits.contains_key(&name) {
            ui.parameter_toggles.insert(name, draft);
        }
    }
    if ui.parameter_edits.contains_key("corrections") {
        // Rebuild from the preserved ordered JSON before restoring guided edits.
        // An invalid or repeated-step array must remain editable in JSON mode.
        crate::ui::corrections::reset(ui, catalog);
        ui.corrections.json_mode |= corrections.json_mode;
        for (name, draft) in corrections.fields {
            let described = name.split_once('.').is_some_and(|(kind, property)| {
                catalog
                    .definition(kind)
                    .is_some_and(|schema| schema["properties"].get(property).is_some())
            });
            if described && let Some(edit) = ui.corrections.fields.get_mut(&name) {
                *edit = draft;
            }
        }
    }
    views
}

pub fn method_setting(ui: &UiState, name: &str) -> bool {
    matches!(
        ui.operation.as_str(),
        "reconstruct" | "plan" | "slice" | "center_sweep"
    ) && matches!(
        name,
        "filter" | "iterations" | "tv_weight" | "nonnegative" | "warm_start"
    )
}

/// Alignment by opposite views fits the center alone, from paired views: it takes no fit
/// choice, no image and no joint options.
fn opposite_views(ui: &UiState) -> bool {
    ui.operation == "align"
        && ui
            .parameter_choices
            .get("method")
            .is_some_and(|method| method == "opposite-views")
}

pub fn applies(ui: &UiState, catalog: &SchemaCatalog, name: &str) -> bool {
    if ui.operation == "simulate" {
        let geometry = ui
            .parameter_choices
            .get("geometry")
            .map_or("parallel", String::as_str);
        return match name {
            "source_to_axis" | "source_to_detector" => geometry == "cone",
            "tilt" => geometry != "parallel",
            _ => true,
        };
    }
    if ui.operation == "align" {
        return !opposite_views(ui) || name == "method";
    }
    if !method_setting(ui, name) {
        return true;
    }
    let method = ui
        .parameter_choices
        .get("method")
        .map_or("auto", String::as_str);
    if method == "auto" {
        return true;
    }
    if name == "warm_start" {
        return !matches!(method, "fbp" | "fourier");
    }
    let config = format!("{method}Config");
    catalog.document()["$defs"]
        .as_object()
        .and_then(|defs| {
            defs.iter()
                .find(|(key, _)| key.eq_ignore_ascii_case(&config))
                .map(|(_, schema)| schema)
        })
        .is_none_or(|schema| {
            schema["properties"]
                .get(if name == "nonnegative" {
                    "lower_bound"
                } else {
                    name
                })
                .is_some()
        })
}

pub fn parameters(ui: &UiState, catalog: &SchemaCatalog) -> Result<Value, String> {
    parameters_detailed(ui, catalog).map_err(|error| error.message)
}

pub fn parameters_detailed(ui: &UiState, catalog: &SchemaCatalog) -> Result<Value, DraftError> {
    let mut result = Map::new();
    for p in catalog.parameters(&definition(&ui.operation)) {
        if ["source", "sources", "path", "force"].contains(&p.name.as_str()) {
            continue;
        }
        if !applies(ui, catalog, &p.name) {
            continue;
        }
        if p.name == "corrections" {
            result.insert(p.name, crate::ui::corrections::parameters_detailed(ui)?);
            continue;
        }
        let schema = scalar_schema(&p.schema);
        let ty = schema.get("type").and_then(Value::as_str).unwrap_or("");
        let choice = !p.choices.is_empty() || schema.get("enum").is_some() || ty == "boolean";
        let text = if choice {
            ui.parameter_choices.get(&p.name).map(String::as_str)
        } else {
            ui.parameter_edits.get(&p.name).map(|t| t.text.as_str())
        }
        .unwrap_or("")
        .trim();
        if text.is_empty() {
            if p.required && !matches!(p.name.as_str(), "poses" | "volume" | "projections") {
                return Err(DraftError::new(
                    &p.name,
                    format!("{} is required", label(&p.name)),
                ));
            }
            if p.name == "poses" {
                result.insert(p.name, Value::Null);
            }
            continue;
        }
        // A string-or-list field (fit parameters) takes a JSON list too.
        let list = (ty == "string" && text.starts_with('[') && accepts(&p.schema, "array"))
            .then(|| serde_json::from_str::<Value>(text).ok())
            .flatten();
        let value = if let Some(list) = list {
            list
        } else if ty == "string"
            || schema
                .get("enum")
                .is_some_and(|v| v.as_array().is_some_and(|a| a.iter().all(Value::is_string)))
        {
            Value::String(text.into())
        } else if let Ok(value) = serde_json::from_str::<Value>(text) {
            value
        } else if accepts(&p.schema, "string") {
            // A field that also takes text (a flats or darks file path) takes it
            // as typed, with or without JSON quotes.
            Value::String(text.into())
        } else {
            serde_json::from_str::<Value>(text).map_err(|_| {
                DraftError::new(
                    &p.name,
                    format!(
                        "{} needs {}",
                        label(&p.name),
                        match ty {
                            "number" => "a finite number",
                            "integer" => "an integer",
                            _ => "valid JSON",
                        }
                    ),
                )
            })?
        };
        if ty == "integer" && !value.is_i64() && !value.is_u64() {
            return Err(DraftError::new(
                &p.name,
                format!("{} needs an integer", label(&p.name)),
            ));
        }
        // A field whose schema also takes an array or an object (a [u, v] pixel
        // size, a storage reference to per-view exposure times) accepts that JSON;
        // the worker validates it.
        let alternative = (value.is_array() && accepts(&p.schema, "array"))
            || (value.is_object() && accepts(&p.schema, "object"));
        if ty == "number" && !value.is_number() && !alternative {
            return Err(DraftError::new(
                &p.name,
                format!("{} needs a number", label(&p.name)),
            ));
        }
        result.insert(p.name, value);
    }
    if opposite_views(ui) {
        result.insert("fit".into(), "center".into());
    }
    Ok(Value::Object(result))
}

/// What a blank field shows: the value TomoJAX will use. A field that takes its value from
/// the configuration shows the form's own `config` entry when it sets one, else the schema's
/// stated default; an unreadable `config` leaves it unknown ("Default").
pub fn default_label(ui: &UiState, view: &ParameterView) -> String {
    let Some(inherited) = &view.inherited else {
        return view.placeholder.clone();
    };
    let config = ui
        .parameter_edits
        .get("config")
        .map_or("", |edit| edit.text.trim());
    let value = if config.is_empty() {
        inherited.clone()
    } else {
        match serde_json::from_str::<Value>(config) {
            Ok(Value::Object(fields)) => fields.get(&view.name).unwrap_or(inherited).clone(),
            _ => return "Default".into(),
        }
    };
    match value {
        Value::String(text) if view.kind == ParameterKind::Choice => {
            format!("Default ({})", sentence(&text.replace('_', " ")))
        }
        Value::String(text) => text,
        Value::Null => "Default".into(),
        other => other.to_string(),
    }
}

/// "reconstruction" → "Reconstruction": schema values shown as labels.
fn sentence(text: &str) -> String {
    let mut chars = text.chars();
    chars
        .next()
        .map(|first| first.to_uppercase().chain(chars).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_values_and_units_read_as_labels() {
        assert_eq!(sentence("reconstruction"), "Reconstruction");
        assert_eq!(sentence(""), "");
        assert_eq!(crate::ui::unit_name("geometry_length"), "scan unit");
        assert_eq!(crate::ui::unit_name("degree"), "degree");
        assert_eq!(crate::ui::unit_name("counts_per_line_integral"), "counts");
        assert_eq!(crate::ui::unit_name("working_pixel"), "working px");
        assert_eq!(crate::ui::unit_name("1"), "");
    }

    #[test]
    fn simulation_shows_only_the_settings_its_geometry_uses() {
        let catalog = SchemaCatalog::new(
            serde_json::from_str(
                // The description the worker sends: the wire schema with help, units and defaults.
                include_str!("../../../../src/tomojax/cli/parameters.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let mut ui = UiState::default();
        initialize(&mut ui, &catalog, "simulate", None);
        let shown = |ui: &UiState| {
            ["tilt", "source_to_axis", "source_to_detector"].map(|name| applies(ui, &catalog, name))
        };
        assert_eq!(shown(&ui), [false, false, false]);
        ui.parameter_choices
            .insert("geometry".into(), "lamino".into());
        assert_eq!(shown(&ui), [true, false, false]);
        ui.parameter_choices
            .insert("geometry".into(), "cone".into());
        assert_eq!(shown(&ui), [true, true, true]);
    }

    #[test]
    fn opposite_views_shows_and_sends_only_its_own_settings() {
        let catalog = SchemaCatalog::new(
            // The description the worker sends: the wire schema with help, units and defaults.
            serde_json::from_str(include_str!("../../../../src/tomojax/cli/parameters.json"))
                .unwrap(),
        )
        .unwrap();
        let mut ui = UiState::default();
        initialize(&mut ui, &catalog, "align", None);
        ui.parameter_edits
            .get_mut("iterations")
            .unwrap()
            .replace_text("8");
        assert!(applies(&ui, &catalog, "iterations"));
        ui.parameter_choices
            .insert("method".into(), "opposite-views".into());
        for name in ["iterations", "motion", "config", "grid", "region"] {
            assert!(!applies(&ui, &catalog, name), "{name} is a joint option");
        }
        assert!(!applies(&ui, &catalog, "fit"), "it fits the center alone");
        let sent = parameters(&ui, &catalog).unwrap();
        assert!(sent.get("iterations").is_none(), "{sent}");
        assert_eq!(sent["method"], "opposite-views");
        assert_eq!(sent["fit"], "center");
        // Joint alignment gets its own fit choice back.
        ui.parameter_choices.insert("method".into(), "joint".into());
        assert_ne!(parameters(&ui, &catalog).unwrap()["fit"], "center");
    }

    #[test]
    fn a_blank_setting_shows_the_value_tomojax_will_use() {
        let catalog = SchemaCatalog::new(
            // The description the worker sends: the wire schema with help, units and defaults.
            serde_json::from_str(include_str!("../../../../src/tomojax/cli/parameters.json"))
                .unwrap(),
        )
        .unwrap();
        let mut ui = UiState::default();
        let views = initialize(&mut ui, &catalog, "align", None);
        let view = |name: &str| views.iter().find(|view| view.name == name).unwrap();
        let shown = |ui: &UiState| {
            ["iterations", "motion", "interpolation"].map(|name| default_label(ui, view(name)))
        };
        assert_eq!(shown(&ui), ["8", "Default (Independent)", "Default"]);
        // The form's own configuration wins over the schema's stated default.
        ui.parameter_edits.insert(
            "config".into(),
            TextEdit::new(r#"{"iterations": 21, "motion": "smooth", "interpolation": "cubic"}"#),
        );
        assert_eq!(shown(&ui), ["21", "Default (Smooth)", "Default (Cubic)"]);
        ui.parameter_edits
            .insert("config".into(), TextEdit::new(r#"{"iterations":"#));
        assert_eq!(shown(&ui), ["Default", "Default", "Default"]);
    }

    #[test]
    fn catalog_refresh_preserves_unfinished_drafts_and_initializes_only_new_fields() {
        let mut described: Value = serde_json::from_str(
            // The description the worker sends: the wire schema with help, units and defaults.
            include_str!("../../../../src/tomojax/cli/parameters.json"),
        )
        .unwrap();
        described["$defs"]["Reconstruct"]["properties"]["seed"]["default"] = 17.into();
        described["$defs"]["Reconstruct"]["properties"]["config"]["description"] =
            "Current worker configuration metadata".into();
        let catalog = SchemaCatalog::new(described.clone()).unwrap();
        described["$defs"]["Reconstruct"]["properties"]
            .as_object_mut()
            .unwrap()
            .remove("seed");
        let previous = SchemaCatalog::new(described).unwrap();
        let mut ui = UiState::default();
        initialize(&mut ui, &previous, "reconstruct", None);
        ui.parameter_edits
            .get_mut("config")
            .unwrap()
            .replace_text("{\"interpolation\":");
        ui.parameter_edits
            .get_mut("iterations")
            .unwrap()
            .replace_text("-");
        ui.parameter_choices.insert("method".into(), "cgls".into());
        ui.parameter_choices
            .insert("nonnegative".into(), "false".into());
        ui.parameter_toggles.insert("nonnegative".into(), false);
        ui.parameter_edits
            .insert("obsolete".into(), TextEdit::new("old"));
        ui.parameter_choices.insert("obsolete".into(), "old".into());
        ui.parameter_toggles.insert("obsolete".into(), true);
        assert!(parameters(&ui, &previous).is_err());

        let views = refresh_catalog(&mut ui, &catalog);

        assert_eq!(ui.operation, "reconstruct");
        assert_eq!(ui.parameter_edits["config"].text, "{\"interpolation\":");
        assert_eq!(ui.parameter_edits["iterations"].text, "-");
        assert_eq!(ui.parameter_choices["method"], "cgls");
        assert_eq!(ui.parameter_choices["nonnegative"], "false");
        assert_eq!(ui.parameter_toggles.get("nonnegative"), Some(&false));
        assert_eq!(ui.parameter_edits["seed"].text, "17");
        assert!(!ui.parameter_edits.contains_key("obsolete"));
        assert!(!ui.parameter_choices.contains_key("obsolete"));
        assert!(!ui.parameter_toggles.contains_key("obsolete"));
        assert!(
            views
                .iter()
                .find(|view| view.name == "config")
                .unwrap()
                .help
                .contains("Current worker configuration metadata")
        );
        assert!(parameters(&ui, &catalog).is_err());
    }

    #[test]
    fn catalog_refresh_preserves_ordered_corrections_and_invalid_json() {
        let catalog = SchemaCatalog::new(
            serde_json::from_str(
                // The description the worker sends: the wire schema with help, units and defaults.
                include_str!("../../../../src/tomojax/cli/parameters.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let ordered = r#"[ {"type":"BeamHardening","rates":[1,2],"weights":[0.5,0.5]}, {"type":"Zingers","threshold":0.2}, {"type":"Rings","views":null} ]"#;
        let mut ui = UiState::default();
        initialize(&mut ui, &catalog, "correct", None);
        ui.parameter_edits
            .get_mut("corrections")
            .unwrap()
            .replace_text(ordered);
        crate::ui::corrections::reset(&mut ui, &catalog);
        ui.corrections
            .fields
            .get_mut("Zingers.threshold")
            .unwrap()
            .replace_text("0.");

        refresh_catalog(&mut ui, &catalog);

        assert_eq!(ui.parameter_edits["corrections"].text, ordered);
        assert_eq!(ui.corrections.fields["Zingers.threshold"].text, "0.");
        assert_eq!(ui.corrections.fields["Rings.views"].text, "null");
        assert!(!ui.corrections.json_mode);
        assert_eq!(
            parameters_detailed(&ui, &catalog).unwrap_err().path,
            "corrections[1].threshold"
        );

        let unfinished = "[ {\"type\": \"Zingers\", \"threshold\": ";
        ui.parameter_edits
            .get_mut("corrections")
            .unwrap()
            .replace_text(unfinished);
        ui.corrections.json_mode = true;
        refresh_catalog(&mut ui, &catalog);
        assert_eq!(ui.parameter_edits["corrections"].text, unfinished);
        assert!(ui.corrections.json_mode);
        assert_eq!(
            parameters_detailed(&ui, &catalog).unwrap_err().path,
            "corrections"
        );
    }

    #[test]
    fn a_recipe_path_survives_the_form() {
        let catalog = SchemaCatalog::new(
            serde_json::from_str(
                // The description the worker sends: the wire schema with help, units and defaults.
                include_str!("../../../../src/tomojax/cli/parameters.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let mut ui = UiState::default();
        let recipe = serde_json::json!({"kind": "frames", "flats": "/data/flat field.tif"});
        initialize(&mut ui, &catalog, "open", Some(&recipe));
        let parameters = parameters_detailed(&ui, &catalog).unwrap();
        assert_eq!(parameters["flats"], "/data/flat field.tif");
    }

    #[test]
    fn a_string_or_list_field_takes_a_list() {
        let catalog = SchemaCatalog::new(
            serde_json::from_str(include_str!(
                "../../../../docs/protocol/operations.schema.json"
            ))
            .unwrap(),
        )
        .unwrap();
        let mut ui = UiState::default();
        initialize(&mut ui, &catalog, "align", None);
        let Some(fit) = ui.parameter_edits.get_mut("fit") else {
            panic!("fit is a text field");
        };
        fit.replace_text(r#"["center", "translations"]"#);
        let parameters = parameters_detailed(&ui, &catalog).unwrap();
        assert_eq!(
            parameters["fit"],
            serde_json::json!(["center", "translations"])
        );
    }

    #[test]
    fn a_scalar_or_pair_field_takes_either() {
        let catalog = SchemaCatalog::new(
            serde_json::from_str(
                // The description the worker sends: the wire schema with help, units and defaults.
                include_str!("../../../../src/tomojax/cli/parameters.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let mut ui = UiState::default();
        initialize(&mut ui, &catalog, "open", None);
        ui.parameter_choices.insert("kind".into(), "frames".into());
        // A storage reference for per-view exposure times.
        ui.parameter_edits
            .get_mut("exposure_times")
            .unwrap()
            .replace_text(r#"{"type":"npy","path":"/data/exposures.npy"}"#);
        for (text, expected) in [
            ("0.5", Some(serde_json::json!(0.5))),
            ("[0.5, 0.75]", Some(serde_json::json!([0.5, 0.75]))),
            ("\"wide\"", None),
        ] {
            ui.parameter_edits
                .get_mut("pixel_size")
                .unwrap()
                .replace_text(text);
            let parameters = parameters_detailed(&ui, &catalog);
            match expected {
                Some(value) => {
                    let parameters = parameters.unwrap();
                    assert_eq!(parameters["pixel_size"], value, "{text}");
                    assert_eq!(parameters["exposure_times"]["type"], "npy");
                }
                None => assert_eq!(parameters.unwrap_err().path, "pixel_size", "{text}"),
            }
        }
    }

    #[test]
    fn nullable_booleans_preserve_automatic_and_required_json_is_validated() {
        let catalog = SchemaCatalog::new(
            serde_json::from_str(
                // The description the worker sends: the wire schema with help, units and defaults.
                include_str!("../../../../src/tomojax/cli/parameters.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let mut ui = UiState::default();
        initialize(&mut ui, &catalog, "reconstruct", None);
        let args = parameters(&ui, &catalog).unwrap();
        assert!(args.get("nonnegative").is_none());
        ui.parameter_choices
            .insert("nonnegative".into(), "false".into());
        assert_eq!(
            parameters(&ui, &catalog).unwrap()["nonnegative"],
            Value::Bool(false)
        );
        initialize(&mut ui, &catalog, "crop", None);
        assert!(parameters(&ui, &catalog).is_err());
        ui.parameter_edits
            .get_mut("rows")
            .unwrap()
            .replace_text("[0, 16]");
        ui.parameter_edits
            .get_mut("cols")
            .unwrap()
            .replace_text("[0, 32]");
        assert_eq!(parameters(&ui, &catalog).unwrap()["cols"][1], 32);
    }

    #[test]
    fn nested_worker_paths_identify_the_owning_control_without_losing_indices() {
        let catalog = SchemaCatalog::new(
            serde_json::from_str(
                // The description the worker sends: the wire schema with help, units and defaults.
                include_str!("../../../../src/tomojax/cli/parameters.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let mut ui = UiState::default();
        let fields = initialize(&mut ui, &catalog, "reconstruct", None);
        let error = field_issue(
            &ui,
            &fields,
            DraftError::new("config.interpolation", "unsupported for this geometry"),
        );
        assert_eq!(error.control.as_deref(), Some("parameter-config"));
        assert_eq!(error.path, "config.interpolation");
        let fields = initialize(
            &mut ui,
            &catalog,
            "correct",
            Some(&serde_json::json!({"corrections":[
                {"type":"Stripes"}, {"type":"Rings"}, {"type":"Zingers","threshold":-1.0}
            ]})),
        );
        let guided = field_issue(
            &ui,
            &fields,
            DraftError::new("corrections[2].threshold", "must be positive"),
        );
        assert_eq!(
            guided.control.as_deref(),
            Some("correction-Zingers.threshold")
        );
        let nested = field_issue(
            &ui,
            &fields,
            DraftError::new("corrections[2].alpha", "invalid value"),
        );
        assert_eq!(nested.control.as_deref(), Some("corrections-json"));
        assert_eq!(nested.path, "corrections[2].alpha");
        assert!(
            field_issue(
                &ui,
                &fields,
                DraftError::new("corrections_extra.alpha", "unknown field")
            )
            .control
            .is_none()
        );
    }

    #[test]
    fn full_grid_plan_uses_schema_intersection_and_does_not_change_on_slice_index() {
        let catalog = SchemaCatalog::new(
            serde_json::from_str(
                // The description the worker sends: the wire schema with help, units and defaults.
                include_str!("../../../../src/tomojax/cli/parameters.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let operation = serde_json::json!({"type":"center_sweep","source":{"type":"member","handle":"h1","member":"scan"},"method":"cgls","iterations":5,"axis":"x","index":2,"centers":[-1.0,0.0,1.0]});
        let plan = plan_operation(&operation, &catalog).unwrap();
        assert_eq!(plan["type"], "reconstruct");
        assert_eq!(plan["source"], operation["source"]);
        assert_eq!(plan["iterations"], 5);
        assert!(
            plan.get("axis").is_none()
                && plan.get("index").is_none()
                && plan.get("centers").is_none()
        );
        let mut ui = UiState::default();
        initialize(&mut ui, &catalog, "slice", Some(&operation));
        let before = plan_signature(&ui, &catalog);
        ui.parameter_edits
            .get_mut("index")
            .unwrap()
            .replace_text("9");
        assert_eq!(plan_signature(&ui, &catalog), before);
        ui.parameter_edits
            .get_mut("iterations")
            .unwrap()
            .replace_text("7");
        assert_ne!(plan_signature(&ui, &catalog), before);
    }
}
