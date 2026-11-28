//! An MCP submission owns one durable tomography operation on the selected machine.
use super::connection::Session;
use super::{connection::Inventory, tools::query};
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Submit {
    operation: Value,
    results: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Job {
    results: Option<String>,
    job: Option<String>,
    #[serde(default)]
    offset: u64,
}

pub const OPERATIONS: [&str; 5] = [
    "correct",
    "reconstruct",
    "reconstruct_slabs",
    "align",
    "simulate",
];

pub fn path(inventory: &Inventory, name: &str) -> Result<PathBuf> {
    let path = Path::new(name);
    Ok(if let Ok(tail) = path.strip_prefix("~") {
        Path::new(
            inventory.worker["home"]
                .as_str()
                .context("Worker has no home")?,
        )
        .join(tail)
    } else {
        Path::new(
            inventory.worker["directory"]
                .as_str()
                .context("Worker has no directory")?,
        )
        .join(path)
    })
}

fn durable_inputs(value: &Value) -> Result<()> {
    match value {
        Value::Object(map) => {
            ensure!(
                value["type"] != "member" && value["artifact"].is_null(),
                "A detached job cannot open a session handle or artifact. Save it and pass its path."
            );
            map.values().try_for_each(durable_inputs)
        }
        Value::Array(items) => items.iter().try_for_each(durable_inputs),
        _ => Ok(()),
    }
}

fn alternatives<'a>(schema: &'a Value, result: &mut Vec<&'a Value>) {
    if let Some(options) = schema["anyOf"].as_array() {
        for option in options {
            alternatives(option, result);
        }
    } else {
        result.push(schema);
    }
}

pub(super) fn prepare(
    value: &mut Value,
    schema: &Value,
    inventory: &Inventory,
    defs: &serde_json::Map<String, Value>,
    ledger: Option<&super::ledger::Ledger>,
) -> Result<()> {
    let mut options = Vec::new();
    alternatives(schema, &mut options);
    let names = options
        .iter()
        .filter_map(|v| v["$ref"].as_str()?.strip_prefix("#/$defs/"))
        .collect::<Vec<_>>();
    if let Some(s) = value.as_str() {
        if !names
            .iter()
            .any(|name| matches!(*name, "Npy" | "Member" | "PathInput"))
        {
            return Ok(());
        }
        if s.starts_with('@')
            && let Some(live) = ledger
        {
            *value = live.resolve(s, names.contains(&"PathInput") && !names.contains(&"Npy"))?;
            return Ok(());
        }
        ensure!(
            !s.starts_with('@'),
            "A detached job cannot open {s}. Save it and pass its path."
        );
        if options
            .iter()
            .any(|node| node["enum"].as_array().is_some_and(|v| v.contains(value)))
        {
            return Ok(());
        }
        let (file, dataset) = s.split_once("::/").map_or((s, None), |(f, d)| (f, Some(d)));
        let file = path(inventory, file)?;
        *value = if let Some(dataset) = dataset {
            json!({"type":"hdf5","path":file,"dataset":format!("/{dataset}")})
        } else if names.contains(&"Npy") && file.extension().is_some_and(|s| s == "npy") {
            json!({"type":"npy","path":file})
        } else if names.contains(&"PathInput") {
            json!({"type":"path","path":file})
        } else if options.iter().any(|s| s["type"] == "string") {
            json!(file)
        } else {
            bail!("Use a durable array path or an explicit HDF5 dataset for {s}");
        };
    } else if let Some(map) = value.as_object_mut() {
        let definition = names
            .into_iter()
            .filter_map(|name| defs.get(name))
            .find(|node| node.pointer("/properties/type/enum/0") == map.get("type"))
            .unwrap_or(schema);
        if let Some(config) = map
            .get("config")
            .filter(|v| v.is_object() && v.get("type").is_none())
        {
            let name = if map.get("type").and_then(Value::as_str) == Some("align") {
                "AlignConfig".into()
            } else {
                format!(
                    "{}Config",
                    map.get("method").and_then(Value::as_str).unwrap_or("auto")
                )
            };
            let key = defs
                .keys()
                .find(|k| k.eq_ignore_ascii_case(&name))
                .context("Choose a specific method or give config its type from describe.")?;
            let mut config = config.clone();
            config["type"] = json!(key);
            map.insert("config".into(), config);
        }
        for (key, value) in map {
            if matches!(key.as_str(), "path" | "checkpoint")
                && let Some(s) = value.as_str()
            {
                *value = json!(path(inventory, s)?);
            } else if let Some(field) = definition["properties"].get(key) {
                prepare(value, field, inventory, defs, ledger)?;
            }
        }
    } else if let Some(items) = value.as_array_mut()
        && let Some(item) = options.iter().find_map(|s| s.get("items"))
    {
        for value in items {
            prepare(value, item, inventory, defs, ledger)?;
        }
    }
    Ok(())
}

pub async fn run(
    session: &Session,
    inventory: &Inventory,
    default_root: Option<&str>,
    name: &str,
    args: Value,
) -> Result<Value> {
    let root =
        |results: Option<&str>| path(inventory, results.or(default_root).unwrap_or("~/tomojax"));
    if name == "submit" {
        let Submit {
            mut operation,
            results,
        } = serde_json::from_value(args)?;
        ensure!(
            operation["type"]
                .as_str()
                .is_some_and(|kind| OPERATIONS.contains(&kind)),
            "Choose one of {}",
            OPERATIONS.join(", ")
        );
        durable_inputs(&operation)?;
        let defs = super::describe::definitions();
        let tag = operation["type"]
            .as_str()
            .context("Operation needs a type")?;
        let planned = super::describe::workflows()
            .get(tag)
            .context("Unknown tomography operation")?
            .1;
        prepare(&mut operation, planned, inventory, defs, None)?;
        let id = uuid::Uuid::new_v4().simple().to_string();
        let directory = root(results.as_deref())?.join("jobs").join(&id);
        let line = json!({"id":format!("{id}.1"),"command":{"type":"run","operation":operation}});
        let mut reply = query(
            session,
            json!({"type":"submit","directory":directory,"script":[line]}),
        )
        .await?;
        reply["job"] = json!(id);
        reply["result_directory"] = json!(directory.join("result"));
        return Ok(reply);
    }
    let Job {
        job,
        results,
        offset,
    } = serde_json::from_value(args)?;
    let root = root(results.as_deref())?;
    let Some(id) = job else {
        ensure!(name == "jobs", "cancel_job needs a job ID");
        return query(session, json!({"type":"jobs","root":root})).await;
    };
    ensure!(
        tomojax_jobs::script::job_id(Path::new(&id))? == id,
        "Expected a job ID"
    );
    let directory = root.join("jobs").join(id);
    match name {
        "jobs" => {
            query(
                session,
                json!({"type":"job","directory":directory,"offset":offset}),
            )
            .await
        }
        "cancel_job" => {
            query(
                session,
                json!({"type":"job_control","directory":directory,"action":"cancel"}),
            )
            .await
        }
        _ => bail!("Unknown job action"),
    }
}

/// Include just the reachable parameter records. A detached input can also be
/// supplied as a file name at any field that accepts stored arrays or objects.
pub fn submission_schema() -> Value {
    let catalog = super::describe::workflows();
    let names = OPERATIONS.map(|tag| catalog[tag].0);
    let defs = parameter_defs(&names, false);
    let choices = names.map(|name| json!({"$ref":format!("#/$defs/{name}")}));
    json!({"type":"object","properties":{"operation":{"oneOf":choices},"results":{"type":"string"}},
        "required":["operation"],"additionalProperties":false,"$defs":defs})
}
pub(super) fn parameter_defs(names: &[&str], session: bool) -> serde_json::Map<String, Value> {
    use std::collections::{BTreeSet, VecDeque};
    let mut queue: VecDeque<_> = names.iter().map(|s| s.to_string()).collect();
    let mut defs = serde_json::Map::new();
    while let Some(name) = queue.pop_front() {
        if defs.contains_key(&name) {
            continue;
        }
        let mut node = super::describe::definitions()[&name].clone();
        if name == "Npy" {
            node["properties"]
                .as_object_mut()
                .unwrap()
                .remove("artifact");
        }
        if name.ends_with("Config")
            && let Some(required) = node["required"].as_array_mut()
        {
            required.retain(|k| k != "type");
        }
        let mut needed = BTreeSet::new();
        path_fields(&mut node, &mut needed, session);
        queue.extend(needed);
        defs.insert(name, node);
    }
    defs
}
fn path_fields(node: &mut Value, needed: &mut std::collections::BTreeSet<String>, session: bool) {
    match node {
        Value::Object(map) => {
            if let Some(name) = map
                .get("$ref")
                .and_then(Value::as_str)
                .and_then(|r| r.strip_prefix("#/$defs/"))
            {
                if name == "Member" {
                    *node = if session {
                        json!({"type":"string"})
                    } else {
                        Value::Bool(false)
                    };
                    return;
                }
                needed.insert(name.into());
                if matches!(name, "PathInput" | "Npy" | "Hdf5") {
                    *node = json!({"anyOf":[{"type":"string"},node.clone()]});
                }
            } else {
                // Paths can be passed as strings, without a wire type discriminator.
                map.remove("discriminator");
                for child in map.values_mut() {
                    path_fields(child, needed, session);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                path_fields(item, needed, session);
            }
        }
        _ => {}
    }
}
