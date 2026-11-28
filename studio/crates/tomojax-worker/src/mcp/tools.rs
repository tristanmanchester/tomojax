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

pub fn list(local: bool) -> Vec<Value> {
    let mut tools: Vec<_> = [
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
        ("control","Pause, resume or cancel a request, close a handle, or restart the session.",json!({"action":{"enum":["pause","resume","cancel","close","restart"]},"target":{"type":"string"}}),vec!["action"]),
        ("python","Run Python in the local TomoJAX executor. Use tj, np, handles, register, view and show with the session’s data.",json!({"code":{"type":"string"},"reset":{"type":"boolean"},"traceback":{"type":"boolean"}}),vec![]),
        ("view","View registered planes and projections, subtract two volumes, or read statistics, a line profile or a pose table.",json!({"source":{"anyOf":[{"type":"string"},{"type":"array","items":{"type":"string"}}]},"mode":{"enum":["planes","projection","sinogram","difference","statistics","profile","pose_table","pose_plot"]},"axis":{"type":"string"},"index":{"type":"integer","minimum":0},"roi":{"type":"array"},"stride":{"type":"array"},"limits":{"type":"array"},"percentiles":{"type":"array"},"bins":{"type":"integer"},"other":{"type":"string"},"marks":super::view::marks_schema(),"features":{"type":"array","items":{"type":"string"}},"grid":{"type":"boolean"},"ticks":{"type":"boolean"}}),vec!["source"]),
        ("download","Copy one worker file to a new file on the laptop. A relative destination uses the project directory.",
         json!({"path":{"type":"string"},"to":{"type":"string"}}),vec!["path","to"]),
        ("upload","Copy one laptop file to a new file on the connected machine. A relative source uses the project directory.",
         json!({"from":{"type":"string"},"path":{"type":"string"}}),vec!["from","path"]),
        ("cancel_job","Request cancellation of a durable job by ID.",json!({"job":{"type":"string"},"results":{"type":"string"}}),vec!["job"]),
     ].into_iter().filter(|(name,_,_,_)| local || *name != "python").map(|(name,description,properties,required)| {
        let schema=if name=="submit" {super::jobs::submission_schema()} else {json!({"type":"object","properties":properties,"required":required,"additionalProperties":false})};
        json!({"name":name,"description":description,"inputSchema":schema,
            "outputSchema":{"type":"object"},"annotations":{"readOnlyHint":!matches!(name,"submit"|"cancel_job"|"upload"|"download"|"python"|"control"),"openWorldHint":false}})
    }).collect();
    tools.extend(
        super::workflow::OPERATIONS
            .iter()
            .map(|name| super::workflow::tool(name)),
    );
    tools
}

pub async fn query(session: &Session, request: Value) -> Result<Value> {
    let reply = session.request(request).await?;
    if !reply.bytes.is_empty() {
        bail!("Unexpected binary worker response");
    }
    Ok(reply.result()?.clone())
}

pub async fn call(connections: &Connections, name: &str, args: Value) -> Result<Value> {
    let input = args.get("path").and_then(Value::as_str).map(String::from);
    let (mut data, session, mut content) = if name == "inspect" {
        let (session, inventory) = connections.open(false).await?;
        let (data, images) =
            super::inspect::inspect(&session, &inventory, args, connections.name == "local")
                .await?;
        (data, session, images)
    } else if name == "view" {
        let (session, inventory) = connections.open(true).await?;
        let (data, images) = super::view::run(&session, &inventory, args).await?;
        (data, session, images)
    } else if name == "python" && connections.name == "local"
        || super::workflow::OPERATIONS.contains(&name)
    {
        let (session, state) = connections.open(true).await?;
        let (data, images) = if name == "python" {
            super::workflow::python(&session, args).await?
        } else {
            super::workflow::run(&session, &state, name, args).await?
        };
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
    if connections.name == "local"
        && let Some(paths) = data["paths"].as_array()
    {
        const ESCAPE: &percent_encoding::AsciiSet = &percent_encoding::NON_ALPHANUMERIC
            .remove(b'/')
            .remove(b'-')
            .remove(b'_')
            .remove(b'.')
            .remove(b'~');
        for s in paths.iter().filter_map(Value::as_str) {
            let path = std::path::Path::new(s);
            if path.is_absolute() {
                let url = format!(
                    "file://{}",
                    percent_encoding::utf8_percent_encode(s, ESCAPE)
                );
                content.push(json!({"type":"resource_link","name":path.file_name().and_then(|s|s.to_str()),"uri":url}));
            }
        }
    }
    let text = input
        .as_deref()
        .filter(|_| name == "inspect")
        .and_then(|path| super::inspect::summary(path, &data))
        .or_else(|| super::workflow::summary(&data))
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
        .open(!matches!(
            name,
            "status" | "jobs" | "submit" | "cancel_job" | "download" | "upload"
        ))
        .await?;
    let value = match name {
        "control" => super::workflow::control(connections, &session, &args).await,
        "download" => super::transfers::download(&session, &state, args).await,
        "upload" => super::transfers::upload(&session, &state, args).await,
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
                let full = args["detail"] == "full";
                for (id, entry) in &session.ledger.lock().unwrap().records {
                    let mut summary = entry["summary"].clone();
                    if full {
                        summary["manifest"] = entry["manifest"].clone();
                    } else {
                        summary
                            .as_object_mut()
                            .unwrap()
                            .retain(|k, _| matches!(k.as_str(), "kind" | "shape" | "unit"));
                    }
                    if let Some(worker) = entry["product"]["handle"].as_str() {
                        summary["bytes"] = status["handles"][worker].clone();
                    }
                    report["handles"][id] = summary;
                }
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
