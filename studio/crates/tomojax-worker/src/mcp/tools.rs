use super::{
    connection::{Connections, Session},
    describe,
};
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Describe {
    operation: String,
    method: Option<String>,
}

pub fn list() -> Vec<Value> {
    [
        ("inspect","Inspect an acquisition or list the datasets and checkpoints in a directory on the connected machine.",
            json!({"path":{"type":"string"},"preview":{"type":"boolean","default":false},"detail":{"enum":["summary","full"]}}),vec!["path"]),
        ("describe", "Look up parameters, defaults, units and choices before configuring an operation. Supply method to see its solver settings.",
         json!({"operation":{"type":"string","minLength":1},"method":{"type":["string","null"]}}), vec!["operation"]),
        ("status","List the connection, host, directory, launcher, executor devices, live handles and active requests.",
         json!({"detail":{"type":"string","enum":["summary","full"],"default":"summary"}}), vec![]),
        ("submit","Submit one tomography operation. The reply gives the job ID and its result directory. Use paths on the connection for all inputs.",
         json!({"operation":{"type":"object","properties":{"type":{"enum":super::jobs::OPERATIONS}},"required":["type"]},"results":{"type":"string"}}),vec!["operation"]),
        ("jobs","List durable jobs, or read a job by ID and byte offset. Use the returned offset to request new records.",
         json!({"job":{"type":"string"},"offset":{"type":"integer","minimum":0},"results":{"type":"string"}}),vec![]),
        ("cancel_job","Request cancellation of a durable job by ID.",json!({"job":{"type":"string"},"results":{"type":"string"}}),vec!["job"]),
    ].into_iter().map(|(name,description,properties,required)| json!({
        "name":name,"description":description,"inputSchema":{"type":"object","properties":properties,"required":required,"additionalProperties":false},
        "outputSchema":{"type":"object"},"annotations":{"readOnlyHint":!matches!(name,"submit"|"cancel_job"),"openWorldHint":false}
    })).collect()
}

pub async fn query(session: &Session, request: Value) -> Result<Value> {
    let reply = session.request(request).await?;
    if !reply.bytes.is_empty() {
        bail!("Unexpected binary worker response");
    }
    match reply.header["type"].as_str() {
        Some("result") => Ok(reply.header["result"].clone()),
        _ => bail!(
            "{}",
            reply.header["error"]["message"]
                .as_str()
                .or(reply.header["message"].as_str())
                .unwrap_or("Worker request failed")
        ),
    }
}

pub async fn call(connections: &Connections, name: &str, args: Value) -> Result<Value> {
    let input = args.get("path").and_then(Value::as_str).map(String::from);
    let (mut data, session, mut content) = if name == "inspect" {
        let (session, inventory) = connections.open(false).await?;
        let (data, images) = super::inspect::inspect(&session, &inventory, args).await?;
        (data, session, images)
    } else {
        let (data, session) = run(connections, name, args).await?;
        (data, session, Vec::new())
    };
    if name == "jobs" && data["type"] == "job" {
        match super::images::job_slices(&session, &data).await {
            Ok((panels, images)) if !panels.is_empty() => {
                data["panels"] = json!(panels);
                content = images;
            }
            Err(error) if data["receipt"].is_null() => {
                data["preview_message"] = json!(format!("Preview unavailable: {error}"));
            }
            Err(error) => return Err(error),
            _ => {}
        }
    }
    let text = input
        .as_deref()
        .filter(|_| name == "inspect")
        .and_then(|path| super::inspect::summary(path, &data))
        .unwrap_or_else(|| serde_json::to_string_pretty(&data).unwrap());
    content.insert(0, json!({"type":"text","text":text}));
    Ok(json!({"content":content,"structuredContent":data,"isError":false}))
}

pub async fn run(
    connections: &Connections,
    name: &str,
    args: Value,
) -> Result<(Value, std::sync::Arc<Session>)> {
    let (session, state) = connections
        .open(!matches!(name, "status" | "jobs" | "submit" | "cancel_job"))
        .await?;
    let value = match name {
        "jobs" | "submit" | "cancel_job" => {
            let profile = studio_core::connection::Profile::load(&connections.name)?;
            super::jobs::run(&session, &state, profile.results.as_deref(), name, args).await
        }
        "describe" => {
            let form: Describe = serde_json::from_value(args)?;
            describe::operation(&form.operation, form.method.as_deref())
        }
        "status" => {
            let mut report = json!({
                "connection":connections.name,
                "host":state.worker["host"],"directory":state.worker["directory"],"launcher":state.worker["jobs"],
                "executor":state.executor["state"],"handles":{},"worker_loss":null,
            });
            let executor = &state.executor["detail"];
            if state.executor["state"] == "ready" {
                let status = query(&session, json!({"type":"status"})).await?;
                report["versions"] = executor["versions"].clone();
                report["devices"] = executor["devices"].clone();
                report["jobs"] = status["requests"].clone();
                report["python_namespace_bytes"] = status
                    .get("python_namespace_bytes")
                    .cloned()
                    .unwrap_or(json!(0));
                let mut allocators = status["allocators"]
                    .as_array()
                    .context("missing allocator samples")?
                    .clone();
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs_f64();
                for sample in &mut allocators {
                    if let Some(sample) = sample.as_object_mut()
                        && let Some(time) = sample.remove("time").and_then(|v| v.as_f64())
                    {
                        sample.insert(
                            "seconds_ago".into(),
                            json!(((now - time).max(0.) * 1000.).round() / 1000.),
                        );
                    }
                }
                report["allocators"] = Value::Array(allocators);
            } else {
                report["detail"] = executor.clone();
            }
            Ok(report)
        }
        _ => bail!("Unknown tool {name}"),
    }?;
    Ok((value, session))
}
