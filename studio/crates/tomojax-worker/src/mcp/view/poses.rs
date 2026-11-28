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
        let mut mask =
            super::marks::raster::Mask::new(canvas.width() as usize, canvas.height() as usize);
        for k in 0..4 {
            mask.stroke(
                corners[k].map(i64::from),
                corners[(k + 1) % 4].map(i64::from),
                1,
                1,
            );
        }
        let mut last = None;
        for (&view, &v) in views.iter().zip(pixels.iter().skip(i).step_by(6)) {
            let point = [
                (f64::from(left + 2) + 254. * normalize(view, x_range)) as i64,
                (f64::from(top + 146) - 124. * normalize(v, y_range)) as i64,
            ];
            mask.stroke(last.unwrap_or(point), point, 2, 2);
            last = Some(point);
        }
        for (pixel, ink) in canvas.pixels_mut().zip(mask.pixels) {
            match ink {
                1 => *pixel = Rgb([171, 180, 189]),
                2 => *pixel = Rgb([11, 85, 125]),
                _ => (),
            }
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
