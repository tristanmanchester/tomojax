//! Run typed operations in the executor and publish their handles only on success.
use super::{
    connection::{Connections, Inventory, Session},
    describe, jobs, ledger,
    tools::query,
};
use anyhow::{Context, Result, bail, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use image::{DynamicImage, GrayImage, ImageFormat, RgbImage};
use serde_json::{Value, json};
use std::{collections::BTreeSet, io::Cursor, time::Instant};

pub const OPERATIONS: &[&str] = &[
    "open",
    "correct",
    "align",
    "reconstruct",
    "reconstruct_slabs",
    "evaluate",
    "simulate",
    "project",
    "backproject",
    "feature",
    "save",
    "export",
    "edit_scan",
];

pub async fn run(
    session: &Session,
    inventory: &Inventory,
    name: &str,
    mut cmd: Value,
) -> Result<(Value, Vec<Value>)> {
    let fields = cmd.as_object_mut().context("Expected tool arguments")?;
    let full = fields.shift_remove("detail") == Some(json!("full"));
    let dry = fields.shift_remove("dry_run") == Some(json!(true));
    if name == "edit_scan" {
        cmd = fields
            .shift_remove("operation")
            .context("Missing scan edit")?;
        ensure!(
            ["select", "crop", "bin", "poses", "combine"]
                .contains(&cmd["type"].as_str().unwrap_or("")),
            "Unknown scan edit"
        );
    } else {
        cmd["type"] = json!(name);
    }
    let tag = cmd["type"]
        .as_str()
        .context("Missing operation tag")?
        .to_owned();
    let catalog = describe::workflows();
    let (_, schema) = catalog.get(tag.as_str()).context("Unknown workflow")?;
    if name == "open" {
        let path = cmd["path"].as_str().context("open needs a file path")?;
        ensure!(
            !path.starts_with('@'),
            "open takes a file path. Use the existing reference directly."
        );
        if let Some((path, dataset)) = path.split_once("::/") {
            let dataset = format!("/{dataset}");
            ensure!(
                cmd["data_path"].is_null() || cmd["data_path"] == dataset,
                "data_path disagrees with the dataset in path"
            );
            cmd["path"] = json!(path);
            cmd["data_path"] = json!(dataset);
        }
    }
    jobs::prepare(
        &mut cmd,
        schema,
        inventory,
        describe::definitions(),
        Some(&session.ledger.lock().unwrap()),
    )?;
    let start = Instant::now();
    let request = if dry {
        json!({"type":"validate","operation":cmd})
    } else if name == "open" {
        cmd.clone()
    } else {
        json!({"type":"run","operation":cmd})
    };
    let mut output = query(session, request).await?;
    if dry && matches!(name, "align" | "reconstruct") {
        output = query(session, json!({"type":"plan","operation":cmd})).await?;
    }
    let mut data = output["value"].clone();
    if dry {
        if data["type"] == "validated" {
            data["operation"] = json!(tag);
            if let Some(Value::Object(details)) =
                data.as_object_mut().unwrap().shift_remove("details")
            {
                data.as_object_mut().unwrap().extend(details);
            }
            if data.as_object().unwrap().len() == 2 {
                data["parameters"] = cmd;
            }
        }
    } else {
        data = if data["type"] == "files" {
            let mut v = json!({"kind":"files"});
            if let Some(info) = output["info"]["summary"].as_object() {
                v.as_object_mut().unwrap().extend(info.clone());
            }
            v["paths"] = data["paths"].clone();
            v
        } else if name == "evaluate" {
            data
        } else {
            session
                .ledger
                .lock()
                .unwrap()
                .publish(&cmd, &output, &data, None)?
        };
        data["wall_seconds"] = json!((start.elapsed().as_secs_f64() * 1000.).round() / 1000.);
    }
    if full {
        data["info"] = output["info"].clone();
        data["manifest"] = output["manifest"].clone();
    }
    Ok((data, vec![]))
}

pub async fn control(connections: &Connections, session: &Session, args: &Value) -> Result<Value> {
    let action = args["action"].as_str().context("Missing control action")?;
    if action == "restart" {
        let lost: Vec<_> = session
            .ledger
            .lock()
            .unwrap()
            .records
            .keys()
            .cloned()
            .collect();
        connections.close().await;
        return Ok(json!({"action":action,"lost_handles":lost}));
    }
    let id = args["target"]
        .as_str()
        .context("target is required. Use status for handle and job IDs.")?;
    if action == "close" {
        let record = session.ledger.lock().unwrap().record(id)?.clone();
        let product = &record["product"];
        if let Some(worker) = product["handle"].as_str() {
            query(session, json!({"type":"close","handle":worker})).await?;
        } else {
            let mut all = BTreeSet::new();
            ledger::tokens(product, &mut all);
            for token in all {
                query(session, json!({"type":"release","artifact":token})).await?;
            }
        }
        session.ledger.lock().unwrap().records.shift_remove(id);
    } else {
        ensure!(
            matches!(action, "pause" | "resume" | "cancel"),
            "Unknown control action {action}"
        );
        query(
            session,
            json!({"type":"control","action":action,"target":id}),
        )
        .await?;
    }
    Ok(json!({"action":action,"target":id}))
}

// show() returns an already drawn uint8 NPY. Preserve its channels and intensity scale.
fn snapshot(bytes: &[u8]) -> Result<Value> {
    let array = npyz::NpyFile::new(bytes)?;
    let shape = array.shape().to_vec();
    let pixels = array.into_vec::<u8>()?;
    let img = match shape.as_slice() {
        [h, w] => DynamicImage::ImageLuma8(
            GrayImage::from_raw((*w).try_into()?, (*h).try_into()?, pixels)
                .context("Truncated gray image")?,
        ),
        [h, w, 3] => DynamicImage::ImageRgb8(
            RgbImage::from_raw((*w).try_into()?, (*h).try_into()?, pixels)
                .context("Truncated RGB image")?,
        ),
        _ => bail!("Unexpected Python image shape {shape:?}"),
    };
    let img = if img.width().max(img.height()) > 1024 {
        img.resize(1024, 1024, image::imageops::FilterType::Lanczos3)
    } else {
        img
    };
    let mut png = Cursor::new(Vec::new());
    img.write_to(&mut png, ImageFormat::Png)?;
    Ok(json!({"type":"image","mimeType":"image/png","data":STANDARD.encode(png.into_inner())}))
}

pub async fn python(session: &Session, mut args: Value) -> Result<(Value, Vec<Value>)> {
    let aliases = {
        let live = session.ledger.lock().unwrap();
        live.records
            .iter()
            .map(|(id, v)| {
                Ok((
                    id.clone(),
                    live.resolve(id, !v["product"]["handle"].is_null())?,
                ))
            })
            .collect::<Result<serde_json::Map<_, _>>>()?
    };
    args["type"] = json!("python");
    args["aliases"] = Value::Object(aliases);
    let output = query(session, args).await?;
    let mut value = output["value"].clone();
    let images = value
        .as_object_mut()
        .context("Expected Python reply")?
        .shift_remove("images")
        .context("Missing images")?;
    let mut summaries = serde_json::Map::new();
    {
        let mut ledger = session.ledger.lock().unwrap();
        for (id, product) in value["handles"]
            .as_object()
            .context("Missing Python handles")?
        {
            summaries.insert(
                id.clone(),
                ledger.publish(&json!({"type":"python"}), &output, product, Some(id))?,
            );
        }
    }
    value["handles"] = Value::Object(summaries);
    let mut blocks = vec![];
    let render = async {
        for image in images.as_array().context("Expected image list")? {
            let reply = session
                .request(
                    json!({"type":"download","path":image["path"],"artifact":image["artifact"]}),
                )
                .await?;
            reply.result()?;
            blocks.push(snapshot(&reply.bytes)?);
        }
        anyhow::Ok(())
    }
    .await;
    let mut tokens = BTreeSet::new();
    ledger::tokens(&images, &mut tokens);
    for token in tokens {
        query(session, json!({"type":"release","artifact":token})).await?;
    }
    render?;
    Ok((value, blocks))
}

pub fn tool(name: &str) -> Value {
    let catalog = describe::workflows();
    let choices: Vec<&str> = if name == "edit_scan" {
        vec!["select", "crop", "bin", "poses", "combine"]
    } else {
        vec![name]
    };
    let roots = choices.iter().map(|t| catalog[t].0).collect::<Vec<_>>();
    let mut defs = jobs::parameter_defs(&roots, true);
    let mut schema = if roots.len() == 1 {
        let mut item = defs.shift_remove(roots[0]).unwrap();
        item["properties"]
            .as_object_mut()
            .unwrap()
            .shift_remove("type");
        if let Some(required) = item["required"].as_array_mut() {
            required.retain(|f| f != "type");
        }
        item
    } else {
        json!({"type":"object","properties":{"operation":{"oneOf":roots.iter().map(|r|json!({"$ref":format!("#/$defs/{r}")})).collect::<Vec<_>>()}},"required":["operation"],"additionalProperties":false,"description":"Select views, crop or bin the detector, replace poses, or combine scans."})
    };
    schema["properties"]["dry_run"] = json!({"type":"boolean","default":false});
    schema["properties"]["detail"] = json!({"enum":["summary","full"],"default":"summary"});
    if !defs.is_empty() {
        schema["$defs"] = Value::Object(defs);
    }
    json!({"name":name,"description":schema["description"],"inputSchema":schema,"outputSchema":{"type":"object"},"annotations":{"readOnlyHint":false,"openWorldHint":false}})
}
