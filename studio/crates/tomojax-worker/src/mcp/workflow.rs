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
    "preview",
];

fn prepare(
    session: &Session,
    inventory: &Inventory,
    name: &str,
    mut cmd: Value,
) -> Result<(Value, bool, bool)> {
    let fields = cmd.as_object_mut().context("Expected tool arguments")?;
    let full = fields.shift_remove("detail") == Some(json!("full"));
    let dry = fields.shift_remove("dry_run") == Some(json!(true));
    if matches!(name, "edit_scan" | "preview") {
        cmd = fields
            .shift_remove("operation")
            .context("Missing scan edit")?;
        ensure!(
            choices(name).contains(&cmd["type"].as_str().unwrap_or("")),
            "Unknown {name} operation"
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
    if name == "feature" {
        super::schema::tagged("recipe", &cmd, &schema["properties"]["recipe"])?;
    }
    if name == "open" {
        cmd.as_object_mut()
            .unwrap()
            .entry("kind")
            .or_insert(json!("scan"));
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
        Some(&session.space.ledger.lock().unwrap()),
    )?;
    Ok((cmd, full, dry))
}
pub async fn run(
    session: &Session,
    inventory: &Inventory,
    name: &str,
    cmd: Value,
) -> Result<(Value, Vec<Value>)> {
    let (cmd, full, dry) = prepare(session, inventory, name, cmd)?;
    if name == "preview" {
        return super::view::preview::run(session, inventory, cmd).await;
    }
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
            data["operation"] = cmd["type"].clone();
            if let Some(Value::Object(details)) =
                data.as_object_mut().unwrap().shift_remove("details")
            {
                data.as_object_mut().unwrap().extend(details);
            }
            if data.as_object().unwrap().len() == 2 {
                data["parameters"] = cmd.clone();
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
                .space
                .ledger
                .lock()
                .unwrap()
                .publish(&cmd, &output, &data, None)?
        };
        if name != "evaluate" {
            let _ = super::activity::CURRENT.try_with(|a| a.report(&mut data));
        }
        data["wall_seconds"] = json!((start.elapsed().as_secs_f64() * 1000.).round() / 1000.);
    }
    if full {
        data["info"] = output["info"].clone();
        data["manifest"] = output["manifest"].clone();
    }
    let mut content = Vec::new();
    if !dry && matches!(name, "align" | "reconstruct") {
        for field in ["poses", "image", "volume"] {
            if field == "poses" {
                let poses = match cmd.get("fit") {
                    None | Some(Value::Null) => cmd["method"] != "opposite-views",
                    Some(Value::Array(v)) => {
                        v.iter().any(|k| k == "rotations" || k == "translations")
                    }
                    Some(v) => v == "rotations" || v == "translations",
                };
                if !poses {
                    continue;
                }
            }
            if let Some(reference) = data["members"].get(field) {
                match super::view::run(
                    session,
                    inventory,
                    json!({"source":reference,"mode":if field=="poses"{"pose_plot"}else{"planes"}}),
                )
                .await
                {
                    Ok((_, images)) => content.extend(images),
                    Err(e) => content
                        .push(json!({"type":"text","text":format!("Panel unavailable: {e}")})),
                }
            }
        }
    }
    Ok((data, content))
}

pub async fn control(connections: &Connections, session: &Session, args: &Value) -> Result<Value> {
    let action = args["action"].as_str().context("Missing control action")?;
    if action == "restart" {
        let lost: Vec<_> = session
            .space
            .ledger
            .lock()
            .unwrap()
            .records
            .keys()
            .cloned()
            .collect();
        connections.close("Worker restarted").await;
        connections.open(true).await?;
        return Ok(json!({"reason":"Worker restarted","lost":lost}));
    }
    let id = args["target"]
        .as_str()
        .context("target is required. Use status for handle and job IDs.")?;
    if action == "close" {
        let record = session.space.ledger.lock().unwrap().record(id)?.clone();
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
        session
            .space
            .ledger
            .lock()
            .unwrap()
            .records
            .shift_remove(id);
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
        let live = session.space.ledger.lock().unwrap();
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
        let mut ledger = session.space.ledger.lock().unwrap();
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
    if session.space.generation > 1 {
        value["namespace_generation"] = json!(session.space.generation);
        value["restart_notice"] = json!("Worker restart cleared the preceding namespace.");
    }
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

fn choices(name: &str) -> Vec<&str> {
    match name {
        "edit_scan" => vec!["select", "crop", "bin", "poses", "combine"],
        "preview" => vec![
            "slice",
            "center_sweep",
            "correction_preview",
            "region_preview",
        ],
        other => vec![other],
    }
}
pub fn tool(name: &str) -> Value {
    let catalog = describe::workflows();
    let choices = choices(name);
    let roots = choices.iter().map(|t| catalog[t].0).collect::<Vec<_>>();
    let mut defs = jobs::parameter_defs(&roots, true);
    if name == "preview" {
        for root in ["Slice", "CenterSweep"] {
            defs.get_mut(root).unwrap()["properties"]["roi"] = json!({"type":"array","description":"Three half-open z, y and x volume intervals."});
        }
    }
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
    if name == "open" {
        schema["properties"]["kind"]["default"] = json!("scan");
        if let Some(required) = schema["required"].as_array_mut() {
            required.retain(|k| k != "kind");
        }
    }
    if name == "preview" {
        schema["description"] = json!(
            "Preview a slice, sweep the rotation center, compare corrections or estimate a region."
        );
    }
    schema["properties"]["dry_run"] = json!({"type":"boolean","default":false});
    schema["properties"]["detail"] = json!({"enum":["summary","full"],"default":"summary"});
    if !defs.is_empty() {
        schema["$defs"] = Value::Object(defs);
    }
    json!({"name":name,"description":schema["description"],"inputSchema":schema,"outputSchema":{"type":"object"},"annotations":{"readOnlyHint":false,"openWorldHint":false}})
}

fn beyond_detector(data: &Value) -> Option<String> {
    let truncation = &data["truncation"];
    let mut details = vec![];
    if let Some(f) = truncation["outside_fraction"]
        .as_f64()
        .filter(|v| *v >= 0.005)
    {
        details.push(format!(
            "{:.0}% of attenuation beyond the detector",
            f * 100.
        ))
    }
    match truncation["background_treatment"].as_str() {
        Some("subtracted") => details.push("modeled exterior subtracted from fit data".into()),
        Some("additive_poisson") => {
            details.push("modeled exterior added to the Poisson mean".into())
        }
        _ => {}
    }
    (!details.is_empty()).then(|| details.join(", "))
}
pub fn summary(data: &Value) -> Option<String> {
    let time = data["wall_seconds"].as_f64()?;
    let kind = data["kind"].as_str()?;
    let mut parts = vec![
        data["handle"].as_str().unwrap_or("Saved").to_owned(),
        kind.into(),
    ];
    if let Some(shape) = data["shape"].as_array() {
        let sizes = shape
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join(" x ");
        parts.push(format!(
            "{sizes} {}",
            data["unit"].as_str().unwrap_or("None")
        ));
    }
    if let Some(method) = data["method"].as_str() {
        parts.push(format!("method {method}"));
    }
    if let (Some(n), Some(reason)) = (
        data["effective_iterations"].as_u64(),
        data["termination"].as_str(),
    ) {
        let stop = match reason {
            "projection_discrepancy" => "stopped at the estimated projection discrepancy",
            "noise_score" => "stopped at the estimated noise level",
            "objective_tolerance" => "stopped at the objective tolerance",
            "roundoff_limit" => "stopped at the float32 roundoff limit",
            "numerical_breakdown" => "stopped at a numerical breakdown",
            "iteration_limit" => "ran to the iteration limit",
            other => other,
        };
        let mut message = format!("{n} updates, {}", stop.replace('_', " "));
        if data["tv_weight"].is_null()
            && let Some(weight) = data["resolved_tv_weight"].as_f64()
        {
            message += &format!("; automatic TV weight {weight}");
        }
        parts.push(message);
    }
    for key in [
        "geometry",
        "calibration",
        "checkpoint",
        "slice_count",
        "first_file",
        "last_file",
    ] {
        if let Some(value) = data.get(key) {
            parts.push(format!(
                "{key} {}",
                value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string())
            ));
        }
    }
    if let Some(a) = data.pointer("/automatic/reasons").and_then(Value::as_array) {
        parts.extend(a.iter().filter_map(Value::as_str).map(str::to_owned));
    }
    parts.extend(beyond_detector(data));
    if let (Some(rms), Some(cols)) = (
        data.pointer("/poses/rms").and_then(Value::as_array),
        data.pointer("/poses/columns").and_then(Value::as_array),
    ) {
        let terms = cols
            .iter()
            .zip(rms)
            .map(|(col, v)| {
                format!(
                    "{} {} {}",
                    col["name"].as_str().unwrap_or(""),
                    v.as_f64().unwrap_or_default(),
                    col["unit"].as_str().unwrap_or("")
                )
            })
            .collect::<Vec<_>>();
        parts.push(format!("pose RMS {}", terms.join(", ")));
    }
    if let Some(m) = data
        .pointer("/disturbed_views/message")
        .and_then(Value::as_str)
    {
        parts.push(m.into());
    }
    for k in ["paths", "devices"] {
        if let Some(a) = data[k].as_array() {
            parts.extend(a.iter().filter_map(Value::as_str).map(str::to_owned));
        }
    }
    parts.push(format!("{time} s"));
    if let Some(a) = data["notes"].as_array() {
        parts.extend(a.iter().filter_map(Value::as_str).map(str::to_owned));
    }
    Some(parts.join(" · "))
}
