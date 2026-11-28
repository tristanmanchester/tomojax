//! Parameter help from the same document whose digest the executor announces.
use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::sync::LazyLock;

static DOCUMENT: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_slice(super::connection::PARAMETERS).expect("compiled parameter schema")
});

pub fn definitions() -> &'static Map<String, Value> {
    DOCUMENT["$defs"]
        .as_object()
        .expect("compiled parameter definitions")
}

pub fn operation(operation: &str, method: Option<&str>) -> Result<Value> {
    let defs = DOCUMENT["$defs"]
        .as_object()
        .context("missing parameter definitions")?;
    let operations: BTreeMap<_, _> = defs
        .iter()
        .filter(|(name, _)| !name.ends_with("Settings"))
        .filter_map(|(_, definition)| {
            Some((
                definition.pointer("/properties/type/enum/0")?.as_str()?,
                definition,
            ))
        })
        .collect();
    let mut entry = *operations.get(operation).with_context(|| {
        format!(
            "Unknown operation. Choose from {}",
            operations.keys().copied().collect::<Vec<_>>().join(", ")
        )
    })?;
    if let Some(method) = method {
        let choices = entry
            .pointer("/properties/method/x-choices")
            .and_then(Value::as_array);
        if !choices.is_some_and(|items| items.iter().any(|v| v == method)) {
            bail!(
                "method must be one of {}",
                choices
                    .into_iter()
                    .flatten()
                    .map(|v| v.as_str().unwrap_or_default())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        let config = if operation == "align" && method == "joint" {
            "AlignConfig".into()
        } else {
            format!("{method}Config")
        };
        if let Some((_, found)) = defs
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(&config))
        {
            entry = found;
        }
    }
    let mut parameters = Map::new();
    for (key, value) in entry["properties"]
        .as_object()
        .context("operation has no properties")?
    {
        if key != "type" {
            parameters.insert(key.clone(), expanded(value, defs, &mut Vec::new()));
        }
    }
    Ok(json!({"operation":operation,"method":method,"parameters":parameters}))
}
fn expanded(value: &Value, defs: &Map<String, Value>, seen: &mut Vec<String>) -> Value {
    match value {
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| expanded(item, defs, seen))
                .collect(),
        ),
        Value::Object(fields) => {
            let mut output = Map::new();
            if let Some(name) = fields
                .get("$ref")
                .and_then(Value::as_str)
                .and_then(|r| r.strip_prefix("#/$defs/"))
            {
                let Some(def) = defs.get(name).filter(|_| !seen.iter().any(|s| s == name)) else {
                    return json!({"title":name});
                };
                if name.ends_with("Config") {
                    return json!({"title":name,"description":"Expert settings: describe with method"});
                }
                seen.push(name.to_owned());
                if let Value::Object(parent) = expanded(def, defs, seen) {
                    output = parent;
                }
                seen.pop();
            }
            output.extend(
                fields
                    .iter()
                    .filter(|(k, _)| k.as_str() != "$ref")
                    .map(|(k, v)| (k.clone(), expanded(v, defs, seen))),
            );
            Value::Object(output)
        }
        _ => value.clone(),
    }
}
