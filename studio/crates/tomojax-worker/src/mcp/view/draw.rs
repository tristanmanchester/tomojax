//! Draw registered planes and their labels using the same cell dimensions as the executor.
use super::Panel;
use anyhow::{Result, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use image::{ExtendedColorType, GrayImage, ImageEncoder, Luma, codecs::png::PngEncoder};
use serde_json::{Value, json};
use std::sync::LazyLock;

// The last row of each 16 by 16 cell stores its advance. The baseline and left bearing
// are baked into the bitmap so labels have the same placement on every machine.
static FONT: LazyLock<GrayImage> = LazyLock::new(|| {
    image::load_from_memory(include_bytes!("aileron-10.png"))
        .unwrap()
        .into_luma8()
});
fn text(canvas: &mut GrayImage, mut x: i32, y: i32, line: &str) {
    for c in line.chars() {
        let index = u8::try_from(c as u32).unwrap_or(0) as u32;
        let (gx, gy) = (index % 16 * 16, index / 16 * 16);
        for dy in 0..15 {
            for dx in 0..16 {
                let alpha = u32::from(FONT.get_pixel(gx + dx, gy + dy).0[0]);
                if alpha > 0
                    && let Some(dst) = canvas
                        .get_pixel_mut_checked((x + dx as i32 - 1) as u32, (y + dy as i32) as u32)
                {
                    dst.0[0] =
                        ((u32::from(dst.0[0]) * (255 - alpha) + 248 * alpha + 127) / 255) as u8;
                }
            }
        }
        x += i32::from(FONT.get_pixel(gx, gy + 15).0[0]);
    }
}
pub(super) fn number(value: f64) -> String {
    if value.abs() < 1e-12 {
        return "0".into();
    }
    let exponent = value.abs().log10().floor() as i32;
    let decimal = |s: String| {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            s
        }
    };
    if !(-4..5).contains(&exponent) {
        let s = format!("{value:.4e}");
        let (sig, exp) = s.split_once('e').unwrap();
        format!(
            "{}e{:+03}",
            decimal(sig.into()),
            exp.parse::<i32>().unwrap()
        )
    } else {
        decimal(format!("{:.*}", (4 - exponent) as usize, value))
    }
}
fn lines(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > 49 {
            out.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
        while line.chars().count() > 49 {
            let k = line.char_indices().nth(49).unwrap().0;
            let tail = line.split_off(k);
            out.push(std::mem::replace(&mut line, tail));
        }
    }
    if !line.is_empty() {
        out.push(line);
    }
    if out.len() > 3 {
        out.truncate(3);
        let last = &mut out[2];
        while last.chars().count() > 46 {
            last.pop();
        }
        last.push_str("...");
    }
    out
}
pub(super) fn limits(panels: &[Panel], panel: &Panel) -> [f64; 2] {
    let mut values = panels
        .iter()
        .filter(|p| p.unit == panel.unit && p.domain == panel.domain)
        .flat_map(|p| p.pixels.iter().copied())
        .filter(|x| x.is_finite())
        .collect::<Vec<_>>();
    values.sort_unstable_by(f64::total_cmp);
    if values.is_empty() {
        return [0., 1.];
    }
    let percentile = |q: f64| {
        let x = q * (values.len() - 1) as f64;
        let a = values[x.floor() as usize];
        a + (values[x.ceil() as usize] - a) * x.fract()
    };
    let found = [percentile(0.01), percentile(0.99)];
    if found[0] == found[1] {
        [values[0], *values.last().unwrap()]
    } else {
        found
    }
}

pub(super) fn render(
    panels: &[Panel],
    explicit: Option<[f64; 2]>,
    first: u64,
) -> Result<(Value, Vec<Value>)> {
    if let Some([lo, hi]) = explicit {
        ensure!(lo < hi, "limits must be (low, high) with low < high");
    }
    let ranges: Vec<_> = panels
        .iter()
        .map(|p| explicit.unwrap_or_else(|| limits(panels, p)))
        .collect();
    let mut summaries = vec![];
    let mut images = vec![];
    let mut blocks = vec![];
    for (batch, group) in panels.chunks(3).enumerate() {
        let id = format!("render-{}", first + batch as u64);
        let mut labels = vec![];
        for (j, p) in group.iter().enumerate() {
            let [lo, hi] = ranges[batch * 3 + j];
            let scale = format!("limits [{}, {}] {}", number(lo), number(hi), p.unit);
            labels.push(
                std::iter::once(&p.title)
                    .chain(&p.labels)
                    .chain(std::iter::once(&scale))
                    .flat_map(|s| lines(s))
                    .collect::<Vec<_>>(),
            );
            summaries.push(json!({"array":p.title,"labels":p.labels,"unit":p.unit,"intensity_limits":[lo,hi],"image":id}));
        }
        let height = 250 + 18 * labels.iter().map(Vec::len).max().unwrap() as u32;
        let width = 328 * group.len() as u32;
        let mut canvas = GrayImage::from_pixel(width, height, Luma([28]));
        for (j, p) in group.iter().enumerate() {
            let [lo, hi] = ranges[batch * 3 + j];
            let scale = (316. / p.width as f64).min(240. / (p.height as f64 * p.aspect));
            let w = (p.width as f64 * scale).round_ties_even().max(1.) as usize;
            let h = (p.height as f64 * p.aspect * scale)
                .round_ties_even()
                .max(1.) as usize;
            let left = j * 328 + 6 + (316 - w) / 2;
            let top = (240 - h) / 2;
            // Pillow's nearest-neighbour transform advances a floating sample centre.
            let step_x = p.width as f64 / w as f64;
            let step_y = p.height as f64 / h as f64;
            let mut sy = 0.5 * step_y;
            for y in 0..h {
                let mut sx = 0.5 * step_x;
                for x in 0..w {
                    let v = p.pixels[(sy as usize).min(p.height - 1) * p.width
                        + (sx as usize).min(p.width - 1)];
                    let level = if hi > lo {
                        ((v - lo) / (hi - lo)).clamp(0., 1.) * 255.
                    } else {
                        0.
                    };
                    canvas.put_pixel((left + x) as u32, (top + y) as u32, Luma([level as u8]));
                    sx += step_x;
                }
                sy += step_y;
            }
            for (i, line) in labels[j].iter().enumerate() {
                text(
                    &mut canvas,
                    (j * 328 + 6) as i32,
                    (245 + i * 18) as i32,
                    line,
                );
            }
        }
        let mut png = vec![];
        PngEncoder::new(&mut png).write_image(
            canvas.as_raw(),
            width,
            height,
            ExtendedColorType::L8,
        )?;
        blocks.push(json!({"type":"image","data":STANDARD.encode(png),"mimeType":"image/png"}));
        images.push(json!({"image":id,"width":width,"height":height}));
    }
    let mut output =
        json!({"panels":summaries,"images":images,"image_limits":{"edge":1568,"pixels":1150000}});
    if let Some(first) = ranges.first()
        && ranges.iter().all(|v| v == first)
    {
        output["intensity_limits"] = json!(first);
    }
    Ok((output, blocks))
}
