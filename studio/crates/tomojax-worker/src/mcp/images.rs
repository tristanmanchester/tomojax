//! Read central planes from registered worker storage without opening its files locally.
use super::connection::Session;
use anyhow::{Context, Result, anyhow, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use image::{ExtendedColorType, ImageEncoder, codecs::png::PngEncoder};
use serde_json::{Value, json};
use studio_core::{
    pixels::{Encoding, Pixels, PlaneSelection},
    protocol::ArrayDescriptor,
};

pub async fn slice(
    session: &Session,
    array: &ArrayDescriptor,
    axis: usize,
    edge: usize,
) -> Result<(Value, Value)> {
    let index = *array.shape.get(axis).context("No such slice axis")? / 2;
    let plane = PlaneSelection::new(&array.shape, axis, index, edge).map_err(anyhow::Error::msg)?;
    let samples = samples(session, array, &plane).await?;
    let [h, w] = plane.remaining.map(|ax| plane.count[ax]);
    let (png, limits) = gray(&samples, w, h)?;
    let header = json!({"axis":array.axes.get(axis).map(|a|&a.name),"index":index,
        "width":w,"height":h,"stride":plane.step,"intensity_limits":limits,"unit":array.unit});
    Ok((
        header,
        json!({"type":"image","mimeType":"image/png","data":STANDARD.encode(png)}),
    ))
}

async fn samples(
    session: &Session,
    array: &ArrayDescriptor,
    plane: &PlaneSelection,
) -> Result<Vec<f64>> {
    let encoding = if array.dtype == "float32" {
        Encoding::Float32
    } else {
        Encoding::Float64
    };
    let reply = session.request(plane.fetch(array, encoding, None)).await?;
    ensure!(
        reply.header["type"] == "result",
        "{}",
        reply.header["error"]["message"]
            .as_str()
            .unwrap_or("Slice unavailable")
    );
    let mut pixels: Pixels = serde_json::from_value(reply.header["result"].clone())?;
    pixels.attach(reply.bytes).map_err(anyhow::Error::msg)?;
    match encoding {
        Encoding::Float32 => pixels
            .floats(&plane.count)
            .map(|v| v.into_iter().map(f64::from).collect()),
        Encoding::Float64 => pixels.doubles(&plane.count),
        _ => unreachable!(),
    }
    .map_err(anyhow::Error::msg)
}

pub async fn inspection(session: &Session, array: &ArrayDescriptor) -> Result<Value> {
    // Compute display limits on the entire central plane before reducing its image.
    let edge = array
        .shape
        .iter()
        .copied()
        .max()
        .context("Missing inspection shape")?;
    let plane = PlaneSelection::new(&array.shape, 0, array.shape[0] / 2, edge)
        .map_err(anyhow::Error::msg)?;
    let values: Vec<f32> = samples(session, array, &plane)
        .await?
        .into_iter()
        .map(|x| x as f32)
        .collect();
    let mut finite: Vec<_> = values.iter().copied().filter(|v| v.is_finite()).collect();
    finite.sort_unstable_by(f32::total_cmp);
    let percentile = |q: f64| {
        let x = (finite.len() - 1) as f64 * q;
        let (a, b) = (finite[x.floor() as usize], finite[x.ceil() as usize]);
        let difference = f64::from(b - a);
        if x.fract() < 0.5 {
            (f64::from(a) + difference * x.fract()) as f32
        } else {
            (f64::from(b) - difference * (1. - x.fract())) as f32
        }
    };
    let (lo, hi) = if finite.is_empty() {
        (0., 1.)
    } else {
        (percentile(0.005), percentile(0.995))
    };
    let range = if hi > lo { hi - lo } else { 1. };
    let pixels = values
        .into_iter()
        .map(|v| (((v - lo) / range).clamp(0., 1.) * 255.).round_ties_even() as u8)
        .collect();
    let [h, w] = plane.remaining.map(|ax| plane.count[ax]);
    let input = image::GrayImage::from_raw(w.try_into()?, h.try_into()?, pixels)
        .context("Incomplete inspection plane")?;
    let image = super::thumbnail::fit(input, super::thumbnail::Filter::Bicubic);
    let mut png = Vec::new();
    PngEncoder::new(&mut png).write_image(
        &image,
        image.width(),
        image.height(),
        ExtendedColorType::L8,
    )?;
    Ok(json!({"type":"image","mimeType":"image/png","data":STANDARD.encode(png)}))
}

fn gray(samples: &[f64], w: usize, h: usize) -> Result<(Vec<u8>, [f64; 2])> {
    ensure!(w.checked_mul(h) == Some(samples.len()), "Incomplete plane");
    let mut finite: Vec<_> = samples.iter().copied().filter(|v| v.is_finite()).collect();
    finite.sort_unstable_by(f64::total_cmp);
    let percentile = |q: f64| {
        let x = (finite.len() - 1) as f64 * q;
        let bottom = finite[x.floor() as usize];
        bottom + (finite[x.ceil() as usize] - bottom) * x.fract()
    };
    let [lo, hi] = if finite.is_empty() {
        [0., 1.]
    } else {
        [percentile(0.005), percentile(0.995)]
    };
    let hi = if hi > lo { hi } else { lo + 1. };
    let values = samples
        .iter()
        .map(|v| {
            if v.is_finite() {
                ((v - lo) / (hi - lo) * 255.)
                    .clamp(0., 255.)
                    .round_ties_even() as u8
            } else {
                0
            }
        })
        .collect::<Vec<_>>();
    let mut png = Vec::new();
    PngEncoder::new(&mut png)
        .write_image(&values, w.try_into()?, h.try_into()?, ExtendedColorType::L8)
        .map_err(|e| anyhow!("Encode slice: {e}"))?;
    Ok((png, [lo, hi]))
}

pub async fn job_slices(session: &Session, job: &Value) -> Result<(Vec<Value>, Vec<Value>)> {
    let durable = job.pointer("/receipt/arrays").and_then(Value::as_object);
    let preview = job
        .pointer("/preview/record/arrays")
        .and_then(Value::as_object);
    let Some(arrays) = durable.or(preview) else {
        return Ok((vec![], vec![]));
    };
    let mut found: Vec<_> = arrays
        .iter()
        .filter(|(_, v)| {
            v["shape"]
                .as_array()
                .is_some_and(|s| (2..=3).contains(&s.len()))
        })
        .collect();
    if durable.is_some() {
        found.sort_by_key(|(k, _)| {
            ["volume", "phantom", "image", "projections", "counts"]
                .iter()
                .position(|n| k == n)
                .unwrap_or(5)
        });
        found.truncate(1);
    }
    let mut panels = vec![];
    let mut images = vec![];
    for (name, desc) in found {
        let array: ArrayDescriptor = serde_json::from_value(desc.clone())?;
        let axes = if array.shape.len() == 3 && array.axes.first().is_some_and(|a| a.name == "z") {
            3
        } else {
            1
        };
        for axis in 0..axes {
            let (mut panel, image) = slice(session, &array, axis, 512).await?;
            panel["array"] = json!(name);
            panels.push(panel);
            images.push(image);
        }
    }
    Ok((panels, images))
}
