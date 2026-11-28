//! Describe a malformed tagged argument using the same schema the executor advertises.
use super::describe::definitions;
use serde_json::{Value, json};
use std::fmt;
pub(super) struct ArgumentError {
    pub path: String,
    pub next: String,
    message: String,
}
impl fmt::Debug for ArgumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}
impl fmt::Display for ArgumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(f)
    }
}
impl std::error::Error for ArgumentError {}
fn resolve(node: &Value) -> &Value {
    node["$ref"]
        .as_str()
        .and_then(|s| s.strip_prefix("#/$defs/"))
        .and_then(|k| definitions().get(k))
        .unwrap_or(node)
}
fn outline(node: &Value, depth: usize) -> String {
    let n = resolve(node);
    if let Some(choices) = n["enum"].as_array() {
        return choices
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join(" | ");
    }
    if let Some(options) = n["anyOf"].as_array() {
        return options
            .iter()
            .filter(|v| v["type"] != "null")
            .map(|v| outline(v, depth))
            .collect::<Vec<_>>()
            .join(" | ");
    }
    if let Some(items) = n["prefixItems"].as_array() {
        return format!(
            "[{}]",
            items
                .iter()
                .map(|v| outline(v, depth))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    match n["type"].as_str().unwrap_or("any") {
        "array" => format!("[{}, ...]", outline(&n["items"], depth)),
        "object" if depth > 1 => "{...}".into(),
        "object" if n["properties"].is_object() => {
            let fields = n["properties"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(name, v)| {
                    let optional = if n["required"]
                        .as_array()
                        .is_some_and(|r| r.contains(&json!(name)))
                    {
                        ""
                    } else {
                        "?"
                    };
                    format!("{name}{optional}: {}", outline(v, depth + 1))
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{fields}}}")
        }
        kind => kind.to_owned(),
    }
}
pub(super) fn tagged(field: &str, args: &Value, node: &Value) -> Result<(), ArgumentError> {
    let input = &args[field];
    let tag = node["discriminator"]["propertyName"]
        .as_str()
        .expect("tagged argument");
    let variant = input[tag]
        .as_str()
        .and_then(|t| node["discriminator"]["mapping"][t].as_str())
        .and_then(|r| definitions().get(r.strip_prefix("#/$defs/")?));
    let error = |path: String, message: String| ArgumentError {
        path,
        next: format!("Give {field} as {}.", outline(variant.unwrap_or(node), 0)),
        message,
    };
    let Some(def) = variant else {
        if input.is_null() {
            return Err(error(
                field.into(),
                format!("Object missing required field `{field}`"),
            ));
        }
        return Err(error(
            field.into(),
            format!("Expected a valid {tag} in {field}"),
        ));
    };
    for key in def["required"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        if input.get(key).is_none() {
            return Err(error(
                field.into(),
                format!("Object missing required field `{key}` - at `$.{field}`"),
            ));
        }
    }
    if let Some(extra) = input
        .as_object()
        .into_iter()
        .flat_map(|m| m.keys())
        .find(|k| def["properties"].get(k).is_none())
    {
        return Err(error(
            field.into(),
            format!("Object contains unknown field `{extra}` - at `$.{field}`"),
        ));
    }
    Ok(())
}

pub fn output(name: &str) -> Value {
    let workflow = super::workflow::OPERATIONS.contains(&name) && name != "preview";
    let object = json!({"type":"object"});
    let array = json!({"type":"array"});
    let text = json!({"type":"string"});
    let integer = json!({"type":"integer"});
    let mut properties = match name {
        "status" => {
            json!({"versions":object,"devices":array,"allocators":array,"jobs":object,"handles":object,"python_namespace_bytes":integer,"connection":text,"host":text,"directory":text,"executor":text})
        }
        "inspect" => {
            json!({"valid":{"type":"boolean"},"issues":array,"projection":object,"volume":object,"geometry":object,"directory":text,"datasets":object,"loading":object,"checkpoints":array,"run":object,"request":{},"entries":object,"resume":text})
        }
        "describe" => json!({"operation":text,"parameters":object}),
        "control" => json!({"action":text}),
        "python" => json!({"stdout":text,"stderr":text,"value":{},"handles":object}),
        "view" => {
            json!({"count":integer,"mean":{"type":["number","null"]},"histogram":array,"profile":array,"columns":array,"table":array})
        }
        "preview" => json!({"handle":text,"bounds":array}),
        "submit" | "cancel_job" => json!({"job":text}),
        "jobs" => json!({"jobs":array,"state":text,"offset":integer}),
        "download" | "upload" => json!({"bytes":integer,"sha256":text}),
        _ => {
            json!({"handle":text,"kind":text,"members":object,"paths":{"type":"array","items":text},"type":text,"wall_seconds":{"type":"number"},"method":text,"manifest":object,"info":object})
        }
    };
    if matches!(name, "view" | "preview") {
        for key in ["panels", "images", "marks", "features"] {
            properties[key] = json!({"type":"array","items":object});
        }
        properties["image_limits"] = object;
        properties["intensity_limits"] = array.clone();
    }
    for key in ["continuation", "message", "next", "checkpoint"] {
        properties[key] = text.clone();
    }
    properties["notes"] = array;
    let mut spec = json!({"type":"object","properties":properties});
    if workflow {
        spec["anyOf"] = json!(
            ["handle", "paths", "type", "message", "continuation"].map(|k| json!({"required":[k]}))
        );
    }
    spec
}

pub fn annotations(name: &str) -> Value {
    json!({"readOnlyHint":matches!(name,"status"|"inspect"|"describe"|"view"|"preview"|"jobs"),
           "destructiveHint":matches!(name,"save"|"export"|"reconstruct_slabs"|"control"|"python"|"cancel_job"),
           "openWorldHint":false})
}
