//! Plot the six fitted pose components in acquisition order.
use super::draw::{label, number};
use anyhow::{Context, Result, ensure};
use image::{Rgb, RgbImage};
use serde_json::Value;

fn range(values: impl Iterator<Item = f64>) -> [f64; 2] {
    values.fold([f64::INFINITY, f64::NEG_INFINITY], |[lo, hi], v| {
        [lo.min(v), hi.max(v)]
    })
}
fn normalize(v: f64, [lo, hi]: [f64; 2]) -> f64 {
    if hi > lo { (v - lo) / (hi - lo) } else { 0.5 }
}
fn line(image: &mut RgbImage, from: [i32; 2], to: [i32; 2], color: Rgb<u8>, wide: bool) {
    let [mut x, mut y] = from;
    let [x1, y1] = to;
    let (dx, dy) = ((x1 - x).abs(), -(y1 - y).abs());
    let (sx, sy) = ((x1 - x).signum(), (y1 - y).signum());
    let mut error = dx + dy;
    loop {
        image.put_pixel(x as u32, y as u32, color);
        if wide {
            let (ox, oy) = if dx >= -dy { (0, 1) } else { (1, 0) };
            image.put_pixel((x + ox) as u32, (y + oy) as u32, color);
        }
        if (x, y) == (x1, y1) {
            break;
        }
        let twice = 2 * error;
        if twice >= dy {
            error += dy;
            x += sx;
        }
        if twice <= dx {
            error += dx;
            y += sy;
        }
    }
}
pub(super) fn image(pixels: &[f64], columns: &Value, views: &[f64], unit: &str) -> Result<Value> {
    let columns = columns.as_array().context("Missing pose columns")?;
    ensure!(
        columns.len() == 6 && !views.is_empty() && pixels.len() == views.len() * 6,
        "incomplete pose table"
    );
    let mut canvas = RgbImage::from_pixel(840, 438, Rgb([255; 3]));
    let black = Rgb([0; 3]);
    label(&mut canvas, 12, 8, "Per-view pose corrections", black);
    let x_range = range(views.iter().copied());
    for (i, col) in columns.iter().enumerate() {
        let left = 12 + (i % 3) as i32 * 276;
        let top = 36 + (i / 3) as i32 * 196;
        let y_range = range(pixels.iter().skip(i).step_by(6).copied());
        let title = format!(
            "{}   [{}, {}] {}",
            col["name"].as_str().unwrap_or(""),
            number(y_range[0]),
            number(y_range[1]),
            col["unit"].as_str().unwrap_or("")
        );
        label(&mut canvas, left, top, &title, black);
        let corners = [
            [left + 2, top + 22],
            [left + 256, top + 22],
            [left + 256, top + 146],
            [left + 2, top + 146],
        ];
        for k in 0..4 {
            line(
                &mut canvas,
                corners[k],
                corners[(k + 1) % 4],
                Rgb([171, 180, 189]),
                false,
            );
        }
        let mut last = None;
        for (&view, &v) in views.iter().zip(pixels.iter().skip(i).step_by(6)) {
            let point = [
                (f64::from(left + 2) + 254. * normalize(view, x_range)) as i32,
                (f64::from(top + 146) - 124. * normalize(v, y_range)) as i32,
            ];
            if let Some(last) = last {
                line(&mut canvas, last, point, Rgb([11, 85, 125]), true);
            } else {
                canvas.put_pixel(point[0] as u32, point[1] as u32, Rgb([11, 85, 125]));
            }
            last = Some(point);
        }
        label(
            &mut canvas,
            left,
            top + 151,
            &format!(
                "view [{}, {}] {unit}",
                number(x_range[0]),
                number(x_range[1])
            ),
            black,
        );
    }
    super::draw::encode(canvas, true, false)
}
