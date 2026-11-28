//! File discovery and acquisition reports always use the worker's filesystem.
use super::{
    connection::{Inventory, Session},
    jobs::path,
    tools::query,
};
use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Form {
    path: String,
    #[serde(default)]
    preview: bool,
    detail: Option<Detail>,
}
#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Detail {
    Summary,
    Full,
}

pub async fn inspect(
    session: &Session,
    inventory: &Inventory,
    args: Value,
    local: bool,
) -> Result<(Value, Vec<Value>)> {
    let form: Form = serde_json::from_value(args)?;
    let path = path(inventory, &form.path)?;
    let peek = query(session, json!({"type":"peek","path":path})).await;
    if let Ok(mut listing) = peek.as_ref().cloned().map_err(|_| ())
        && listing["directory"].is_string()
    {
        // Checkpoint and exported slice directories are acquisitions for the executor.
        let special = listing.as_object_mut().unwrap().remove("record") == Some(json!(true));
        if !special {
            listing.as_object_mut().unwrap().remove("type");
            return Ok((listing, vec![]));
        }
    }
    session.inventory(true, local).await?;
    let report = query(session, json!({"type":"inspect","path":path})).await;
    let mut full = match report {
        Ok(reply) => reply
            .pointer("/value/report")
            .cloned()
            .context("Worker returned no inspection report")?,
        Err(reason) => {
            if let Ok(listing) = peek
                && listing["datasets"]
                    .as_object()
                    .is_some_and(|m| !m.is_empty())
            {
                return Ok((
                    json!({"path":path,"datasets":listing["datasets"],"next":"Select data_path and angles_path for open."}),
                    vec![],
                ));
            }
            return Err(reason);
        }
    };
    if let Some(checkpoint) = full.get("checkpoint").filter(|v| v.is_object()) {
        return Ok((checkpoint.clone(), vec![]));
    }
    let mut images = Vec::new();
    if form.preview {
        let stored = query(
            session,
            json!({"type":"stored","path":path,"inspection":full}),
        )
        .await?;
        let arrays = stored["arrays"]
            .as_object()
            .context("Missing inspection arrays")?;
        let image =
            if let Some(descriptor) = arrays.get("projections").or_else(|| arrays.get("volume")) {
                super::images::slice(
                    session,
                    &serde_json::from_value(descriptor.clone())?,
                    0,
                    1024,
                )
                .await?
                .1
            } else {
                executor_plane(session, &path, &full).await?
            };
        images.push(image);
    }
    if !matches!(form.detail, Some(Detail::Full)) {
        let keys = [
            "record",
            "contents",
            "input_path",
            "checkpoint",
            "valid",
            "issues",
            "loading",
            "projection",
            "volume",
            "geometry",
            "angles",
            "flats_darks",
        ];
        full.as_object_mut()
            .context("Expected an inspection report")?
            .retain(|k, _| keys.contains(&k.as_str()));
    }
    Ok((full, images))
}

async fn executor_plane(
    session: &Session,
    path: &std::path::Path,
    report: &Value,
) -> Result<Value> {
    let has_projections = report.pointer("/projection/found") == Some(&json!(true));
    let (handle, source, mode) = if has_projections {
        let opened = query(session, json!({"type":"open","kind":"scan","path":path})).await?;
        let id = opened
            .pointer("/value/handle")
            .and_then(Value::as_str)
            .context("missing scan handle")?
            .to_owned();
        (
            Some(id.clone()),
            json!({"type":"member","handle":id,"member":"projections"}),
            "projection",
        )
    } else {
        (None, json!({"type":"path","path":path}), "planes")
    };
    let read=query(session,json!({"type":"read","source":source,"mode":mode,"axis":if has_projections{"view"}else{"z"}})).await;
    if let Some(id) = handle {
        query(session, json!({"type":"close","handle":id})).await?;
    }
    let read = read?;
    let items = read
        .pointer("/value/planes")
        .and_then(Value::as_array)
        .context("missing display planes")?;
    let mut descriptors = Vec::new();
    for item in items {
        descriptors.push(serde_json::from_value::<
            studio_core::protocol::ArrayDescriptor,
        >(item["array"].clone())?);
    }
    let plane = descriptors.first().context("no inspection plane")?;
    let image = super::images::slice(session, plane, 0, 1024).await;
    for token in descriptors
        .iter()
        .flat_map(|d| d.artifacts())
        .collect::<std::collections::BTreeSet<_>>()
    {
        query(session, json!({"type":"release","artifact":token})).await?;
    }
    Ok(image?.1)
}

pub fn summary(path: &str, data: &Value) -> Option<String> {
    if data.get("run").is_some()
        && let Some(resume) = data.get("resume").and_then(Value::as_str)
    {
        return Some(format!(
            "Checkpoint of {}. {resume}",
            data["run"].as_str().unwrap_or("run")
        ));
    }
    data.get("valid")?;
    let name = std::path::Path::new(path).file_name()?.to_str()?;
    let shape = ["projection", "volume"]
        .into_iter()
        .filter_map(|k| data[k]["shape"].as_array())
        .next()
        .map_or_else(
            || "None".into(),
            |v| {
                format!(
                    "[{}]",
                    v.iter()
                        .map(Value::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            },
        );
    let issues = data["issues"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>()
        .join("; ");
    let mut text = format!(
        "{name}, shape {shape}. {}",
        if issues.is_empty() {
            "Inspection passed."
        } else {
            &issues
        }
    );
    if let Some(fields) = data["loading"].as_object()
        && !fields.is_empty()
    {
        let found = fields
            .iter()
            .filter(|(k, v)| k.as_str() != "message" && !v.is_null())
            .map(|(k, v)| {
                format!(
                    "{k} {}",
                    v.as_str().map_or_else(|| v.to_string(), String::from)
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let message = fields
            .get("message")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .unwrap_or(if found.is_empty() {
                "as recorded"
            } else {
                &found
            });
        text.push_str(&format!(" Loading: {message}."));
    }
    Some(text)
}
