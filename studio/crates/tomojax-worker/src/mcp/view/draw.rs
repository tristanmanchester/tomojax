//! Draw registered planes and their labels using the same cell dimensions as the executor.
use super::{Panel, View, marks};
use anyhow::{Result, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use image::{ExtendedColorType, GrayImage, ImageEncoder, Rgb, RgbImage, codecs::png::PngEncoder};
use serde_json::{Value, json};
use std::sync::LazyLock;

// The last row of each 16 by 16 cell stores its advance. The baseline and left bearing
// are baked into the bitmap so labels have the same placement on every machine.
static FONT: LazyLock<GrayImage> = LazyLock::new(|| {
    image::load_from_memory(include_bytes!("aileron-10.png"))
        .unwrap()
        .into_luma8()
});
pub(super) fn label(canvas: &mut RgbImage, mut x: i32, y: i32, line: &str, color: Rgb<u8>) {
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
                    for (v, c) in dst.0.iter_mut().zip(color.0) {
                        *v = ((u32::from(*v) * (255 - alpha) + u32::from(c) * alpha + 127) / 255)
                            as u8;
                    }
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
fn lines(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.chars().count() + 1 + word.chars().count() > width {
            out.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
        while line.chars().count() > width {
            let k = line.char_indices().nth(width).unwrap().0;
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
        while last.chars().count() > width - 3 {
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

#[derive(Clone, Copy)]
pub(super) struct Cell {
    pub left: usize,
    pub top: usize,
    pub width: usize,
    pub height: usize,
    pub sx: f64,
    pub sy: f64,
}
impl Cell {
    pub fn pixel(&self, r: f64, c: f64) -> [f64; 2] {
        [
            self.left as f64 + (c + 0.5) * self.sx,
            self.top as f64 + (r + 0.5) * self.sy,
        ]
    }
    pub fn contains(&self, x: i32, y: i32) -> bool {
        (self.left..self.left + self.width).contains(&(x as usize))
            && (self.top..self.top + self.height).contains(&(y as usize))
    }
}
fn advance(text: &str) -> i32 {
    text.chars()
        .map(|c| {
            let i = u8::try_from(c as u32).unwrap_or(0) as u32;
            i32::from(FONT.get_pixel(i % 16 * 16, i / 16 * 16 + 15).0[0])
        })
        .sum()
}
fn ticks(p: &Panel, c: Cell, img: &mut RgbImage) {
    let color = Rgb([150; 3]);
    for (dim, len) in p.displayed.iter().zip([p.height, p.width]) {
        let [start, _, stride] = p.window[*dim];
        let end = start + stride * (len - 1);
        let raw = ((end - start) as f64).max(1.) / 8.;
        let base = 10_f64.powf(raw.log10().floor());
        let step = [1., 2., 5., 10.]
            .map(|v| v * base)
            .into_iter()
            .min_by(|a, b| (a - raw).abs().total_cmp(&(b - raw).abs()))
            .unwrap()
            .round_ties_even()
            .max(1.) as usize;
        for index in (start.div_ceil(step) * step..=end).step_by(step) {
            let pos = (index - start) as f64 / stride as f64;
            let text = index.to_string();
            if *dim == p.displayed[1] {
                let x = c.pixel(0., pos)[0] as i32;
                let y = c.top as i32;
                for i in 1..5 {
                    img.put_pixel(x as u32, (y - i) as u32, color);
                }
                label(img, x - (advance(&text) + 1) / 2, y - 15, &text, color);
            } else {
                let x = c.left as i32;
                let y = c.pixel(pos, 0.)[1] as i32;
                for i in 1..5 {
                    img.put_pixel((x - i) as u32, y as u32, color);
                }
                label(img, x - 6 - advance(&text), y - 6, &text, color);
            }
        }
    }
    label(
        img,
        (c.left + c.width + 3) as i32,
        c.top as i32 - 15,
        &p.axes[1],
        color,
    );
    label(
        img,
        c.left as i32 - 6 - advance(&p.axes[0]),
        (c.top + c.height + 3) as i32 - 3,
        &p.axes[0],
        color,
    );
}
fn raster(
    canvas: &mut RgbImage,
    p: &Panel,
    [lo, hi]: [f64; 2],
    (cell_w, cell_h): (usize, usize),
    j: usize,
    decorated: bool,
) -> Cell {
    let (ml, mt, mr, mb) = if decorated {
        (40, 16, 14, 14)
    } else {
        (6, 0, 6, 0)
    };
    let (room_w, room_h) = (cell_w - ml - mr, cell_h - mt - mb);
    let scale = (room_w as f64 / p.width as f64).min(room_h as f64 / (p.height as f64 * p.aspect));
    let w = (p.width as f64 * scale).round_ties_even().max(1.) as usize;
    let h = (p.height as f64 * p.aspect * scale)
        .round_ties_even()
        .max(1.) as usize;
    let left = j * cell_w + ml + (room_w - w) / 2;
    let top = mt + (room_h - h) / 2;
    // Pillow's nearest-neighbour transform advances a floating sample centre.
    let step_x = p.width as f64 / w as f64;
    let step_y = p.height as f64 / h as f64;
    let mut sy = 0.5 * step_y;
    for y in 0..h {
        let mut sx = 0.5 * step_x;
        for x in 0..w {
            let v = p.pixels
                [(sy as usize).min(p.height - 1) * p.width + (sx as usize).min(p.width - 1)];
            let level = if hi > lo {
                ((v - lo) / (hi - lo)).clamp(0., 1.) * 255.
            } else {
                0.
            };
            canvas.put_pixel((left + x) as u32, (top + y) as u32, Rgb([level as u8; 3]));
            sx += step_x;
        }
        sy += step_y;
    }
    Cell {
        left,
        top,
        width: w,
        height: h,
        sx: w as f64 / p.width as f64,
        sy: h as f64 / p.height as f64,
    }
}
pub(super) fn encode(mut canvas: RgbImage, decorated: bool, grid: bool) -> Result<Value> {
    let (width, height) = canvas.dimensions();
    if grid {
        let color = Rgb([90, 170, 255]);
        for x in (100..width).step_by(100) {
            for y in (0..height).step_by(4) {
                canvas.put_pixel(x, y, color);
            }
            label(&mut canvas, x as i32 + 2, 2, &x.to_string(), color);
        }
        for y in (100..height).step_by(100) {
            for x in (0..width).step_by(4) {
                canvas.put_pixel(x, y, color);
            }
            label(&mut canvas, 2, y as i32 + 2, &y.to_string(), color);
        }
    }
    let mut png = vec![];
    if decorated {
        PngEncoder::new(&mut png).write_image(
            canvas.as_raw(),
            width,
            height,
            ExtendedColorType::Rgb8,
        )?;
    } else {
        let gray = canvas.pixels().map(|v| v.0[0]).collect::<Vec<_>>();
        PngEncoder::new(&mut png).write_image(&gray, width, height, ExtendedColorType::L8)?;
    }
    Ok(json!({"type":"image","data":STANDARD.encode(png),"mimeType":"image/png"}))
}
const COLORS: [(&str, [u8; 3]); 5] = [
    ("cyan", [0, 229, 255]),
    ("magenta", [255, 64, 245]),
    ("green", [110, 255, 60]),
    ("orange", [255, 140, 0]),
    ("red", [255, 48, 48]),
];
fn outlines(canvas: &mut RgbImage, cell: Cell, p: &Panel) {
    for (mask, (_, color)) in p.outlines.iter().zip(COLORS) {
        let (w, h) = (cell.width, cell.height);
        let mut selected = vec![false; w * h];
        let mut sy = 0.5 / cell.sy;
        for row in selected.chunks_mut(w) {
            let mut sx = 0.5 / cell.sx;
            for v in row {
                *v = mask
                    [(sy as usize).min(p.height - 1) * p.width + (sx as usize).min(p.width - 1)]
                    != 0.;
                sx += 1. / cell.sx;
            }
            sy += 1. / cell.sy;
        }
        for r in 0..h {
            for c in 0..w {
                let i = r * w + c;
                if selected[i]
                    && (r == 0
                        || c == 0
                        || r + 1 == h
                        || c + 1 == w
                        || !selected[i - w]
                        || !selected[i + w]
                        || !selected[i - 1]
                        || !selected[i + 1])
                {
                    for y in r..=(r + 1).min(h - 1) {
                        for x in c..=(c + 1).min(w - 1) {
                            canvas.put_pixel(
                                (cell.left + x) as u32,
                                (cell.top + y) as u32,
                                Rgb(color),
                            );
                        }
                    }
                }
            }
        }
    }
}
fn panel_labels(p:&Panel,[lo,hi]:[f64;2],decorated:bool)->Vec<String>{
    let scale=format!("limits [{}, {}] {}", number(lo), number(hi), p.unit);
    std::iter::once(&p.title).chain(&p.labels).chain(std::iter::once(&scale)).flat_map(|s|lines(s,if decorated{70}else{49})).collect()
}

pub(super) fn render(
    panels: &[Panel],
    form: &View,
    first: u64,
    placements: Option<&std::sync::Mutex<super::Placements>>,
) -> Result<(Value, Vec<Value>)> {
    let explicit = form.limits;
    let decorated = !form.features.is_empty()
        || !form.marks.is_empty()
        || form.ticks == Some(true)
        || form.grid;
    let mut reports = marks::reports(&form.marks);
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
            labels.push(panel_labels(p,[lo,hi],decorated));
            summaries.push(json!({"array":p.title,"labels":p.labels,"unit":p.unit,"intensity_limits":[lo,hi],"image":id}));
        }
        let (cell_w, cell_h) = if decorated {
            [(960, 720), (768, 576), (512, 400)][group.len() - 1]
        } else {
            (328usize, 240usize)
        };
        let height = (cell_h + 10 + 18 * labels.iter().map(Vec::len).max().unwrap()) as u32;
        let width = (cell_w * group.len()) as u32;
        let mut canvas = RgbImage::from_pixel(width, height, Rgb([28; 3]));
        let mut placed = Vec::new();
        for (j, p) in group.iter().enumerate() {
            let [lo, hi] = ranges[batch * 3 + j];
            let cell = raster(&mut canvas, p, [lo, hi], (cell_w, cell_h), j, decorated);
            if placements.is_some() {
                placed.push(super::placement::Placement::new(p, cell));
            }
            outlines(&mut canvas, cell, p);
            if decorated && form.ticks.unwrap_or(true) {
                ticks(p, cell, &mut canvas);
            }
            let overlay = marks::Overlay {
                panel: p,
                cell,
                index: batch * 3 + j,
                image: &id,
                limits: [lo, hi],
            };
            for (mark, report) in form.marks.iter().zip(&mut reports) {
                report["panels"]
                    .as_array_mut()
                    .unwrap()
                    .push(overlay.draw(&mut canvas, mark)?);
            }
            for (i, line) in labels[j].iter().enumerate() {
                label(
                    &mut canvas,
                    (j * cell_w + 6) as i32,
                    (cell_h + 5 + i * 18) as i32,
                    line,
                    Rgb([248; 3]),
                );
            }
        }
        blocks.push(encode(canvas, decorated, form.grid)?);
        if let Some(cache) = placements {
            cache.lock().unwrap().insert(id.clone(), placed);
        }
        images.push(json!({"image":id,"width":width,"height":height}));
    }
    let mut output =
        json!({"panels":summaries,"images":images,"image_limits":{"edge":1568,"pixels":1150000}});
    if let Some(first) = ranges.first()
        && ranges.iter().all(|v| v == first)
    {
        output["intensity_limits"] = json!(first);
    }
    if !reports.is_empty() {
        output["marks"] = json!(reports);
    }
    if !form.features.is_empty() {
        output["features"] = json!(
            form.features
                .iter()
                .zip(COLORS)
                .map(|(name, (color, _))| json!({"handle":name,"color":color}))
                .collect::<Vec<_>>()
        );
    }
    Ok((output, blocks))
}
