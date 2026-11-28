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
