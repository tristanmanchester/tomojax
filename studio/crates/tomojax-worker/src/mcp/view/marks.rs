//! Positions in array order are mapped through the exact window returned by the executor.
use super::{
    Panel,
    draw::{self, Cell},
};
use anyhow::{Result, ensure};
use image::{Rgb, RgbImage};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Clone, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub(super) enum Mark {
    Point {
        at: Vec<f64>,
        label: Option<String>,
        color: Option<Color>,
        snap: Option<Snap>,
        within: Option<f64>,
    },
}
mod snapping;
use snapping::Snap;
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum Color {
    Yellow,
    Cyan,
    Magenta,
    Orange,
    Green,
    Red,
}
impl Color {
    fn pixel(self) -> Rgb<u8> {
        Rgb(match self {
            Self::Yellow => [255, 214, 0],
            Self::Cyan => [0, 229, 255],
            Self::Magenta => [255, 64, 245],
            Self::Orange => [255, 140, 0],
            Self::Green => [110, 255, 60],
            Self::Red => [255, 48, 48],
        })
    }
}

fn round(v: f64, digits: i32) -> f64 {
    let scale = 10_f64.powi(digits);
    (v * scale).round_ties_even() / scale
}
fn significant(v: f64) -> f64 {
    if v == 0. {
        0.
    } else {
        format!("{v:.5e}").parse().unwrap_or(v)
    }
}
pub(super) fn reports(marks: &[Mark]) -> Vec<Value> {
    marks.iter().map(|Mark::Point{at,label,..}|{
        let mut record=json!({"type":"point","positions":[at.iter().map(|v|round(*v,2)).collect::<Vec<_>>()],"panels":[]});
        if let Some(label)=label{record["label"]=json!(label);}
        record
    }).collect()
}
pub(super) struct Overlay<'a> {
    pub panel: &'a Panel,
    pub cell: Cell,
    pub index: usize,
    pub image: &'a str,
    pub limits: [f64; 2],
}

fn order(values: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut v = values.filter(|x| x.is_finite()).collect::<Vec<_>>();
    v.sort_unstable_by(f64::total_cmp);
    v
}
fn quantile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.;
    }
    let x = q * (v.len() - 1) as f64;
    let a = v[x.floor() as usize];
    a + x.fract() * (v[x.ceil() as usize] - a)
}
fn reflected(i: isize, n: usize) -> usize {
    let k = i.rem_euclid(2 * n as isize) as usize;
    if k < n { k } else { 2 * n - 1 - k }
}
fn smooth(pixels: &[f64], width: usize, height: usize, median: f64) -> Vec<f64> {
    let mut data: Vec<_> = pixels
        .iter()
        .map(|x| if x.is_finite() { *x } else { median })
        .collect();
    let weights = (-4..=4)
        .map(|i| (-0.5 * f64::from(i * i)).exp())
        .collect::<Vec<_>>();
    let sum = weights.iter().sum::<f64>();
    for vertical in [true, false] {
        let mut next = Vec::with_capacity(data.len());
        for (i, _) in data.iter().enumerate() {
            let (r, c) = (i / width, i % width);
            next.push(
                weights
                    .iter()
                    .enumerate()
                    .map(|(j, w)| {
                        let offset = j as isize - 4;
                        let ix = if vertical {
                            reflected(r as isize + offset, height) * width + c
                        } else {
                            r * width + reflected(c as isize + offset, width)
                        };
                        data[ix] * w / sum
                    })
                    .sum::<f64>(),
            );
        }
        data = next;
    }
    data
}
fn nearby(report: &mut Value, p: &Panel, at: [f64; 2], ordered: &[f64]) {
    let smooth = smooth(&p.pixels, p.width, p.height, quantile(ordered, 0.5));
    let sorted = order(smooth.iter().copied());
    let jumps = order(
        p.pixels
            .chunks_exact(p.width)
            .flat_map(|row| row.windows(2).map(|w| (w[1] - w[0]).abs())),
    );
    let mut spread = quantile(&sorted, 0.99) - quantile(&sorted, 0.01);
    if spread == 0. {
        spread = sorted.last().unwrap() - sorted[0];
    }
    let contrast =
        (spread / 4.).max(8. * quantile(&jumps, 0.5) / 0.954 / (2. * std::f64::consts::PI.sqrt()));
    let [r, c] = at.map(|n| n.round_ties_even() as usize);
    let level = smooth[r * p.width + c];
    let mut lo = level;
    let mut hi = level;
    for row in r.saturating_sub(2)..=(r + 2).min(p.height - 1) {
        for col in c.saturating_sub(2)..=(c + 2).min(p.width - 1) {
            if row.abs_diff(r).pow(2) + col.abs_diff(c).pow(2) <= 4 {
                let v = smooth[row * p.width + col];
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
    }
    if contrast > 0. && hi - lo >= contrast {
        return;
    }
    let flat = "Flat: nothing within 2 samples stands out from the value under this mark";
    let nearest = smooth
        .iter()
        .enumerate()
        .filter(|(_, v)| contrast > 0. && (**v - level).abs() >= contrast)
        .min_by_key(|(i, _)| (i / p.width).abs_diff(r).pow(2) + (i % p.width).abs_diff(c).pow(2));
    let Some((i, v)) = nearest else {
        report["warning"] = json!(format!("{flat}, and nothing on this plane does."));
        return;
    };
    let (row, col) = ((i / p.width) as f64, (i % p.width) as f64);
    let (dr, dc) = (row - r as f64, col - c as f64);
    let gap = dr.hypot(dc);
    let mut found = (row as usize, col as usize);
    let sign = (*v - level).signum();
    let mut best = f64::NEG_INFINITY;
    for j in 0..4 {
        let n = f64::from(j);
        let rr = (row + n * dr / gap)
            .round_ties_even()
            .clamp(0., (p.height - 1) as f64) as usize;
        let cc = (col + n * dc / gap)
            .round_ties_even()
            .clamp(0., (p.width - 1) as f64) as usize;
        let raw = p.pixels[rr * p.width + cc];
        let score = if raw.is_nan() {
            0.
        } else {
            sign * (raw - level)
        };
        if score > best {
            found = (rr, cc);
            best = score;
        }
    }
    let [rd, cd] = p.displayed;
    let dist = round(
        ((found.0 as f64 - at[0]) * p.window[rd][2] as f64)
            .hypot((found.1 as f64 - at[1]) * p.window[cd][2] as f64),
        2,
    );
    let mut coords: Vec<_> = p.window.iter().map(|w| w[0] as f64).collect();
    coords[rd] += found.0 as f64 * p.window[rd][2] as f64;
    coords[cd] += found.1 as f64 * p.window[cd][2] as f64;
    let value = significant(p.pixels[found.0 * p.width + found.1]);
    let rank = percentile(ordered, value);
    report["nearest"] = json!({"at":coords,"distance":dist,"value":value,"percentile":rank});
    report["warning"] = json!(format!(
        "{flat}, so it is not on a peak, trough or edge. The nearest structure is {dist} index units away at {coords:?} (value {value}, percentile {rank}); mark nearest.at if that is what you meant."
    ));
}
fn percentile(ordered: &[f64], value: f64) -> f64 {
    round(
        100. * ordered.partition_point(|x| *x < value) as f64 / ordered.len() as f64,
        1,
    )
}

impl Overlay<'_> {
    pub fn draw(&self, canvas: &mut RgbImage, mark: &Mark) -> Result<Value> {
        let Mark::Point {
            at,
            color,
            label,
            snap,
            within,
        } = mark;
        let p = self.panel;
        let c = self.cell;
        ensure!(
            at.len() == p.window.len(),
            "give one per source axis ({}) as array indices",
            p.source_axes.join(", ")
        );
        let mut reply = json!({"panel":self.index});
        if let Some(d) = p.normal {
            let distance = (at[d] - p.window[d][0] as f64).abs();
            if distance > 2. {
                return Ok(
                    json!({"panel":self.index,"drawn":false,"reason":"off_plane","distance":round(distance,3)}),
                );
            }
        }
        let [r, col] = p
            .displayed
            .map(|d| (at[d] - p.window[d][0] as f64) / p.window[d][2] as f64);
        if r < -0.5 || col < -0.5 || r >= p.height as f64 - 0.5 || col >= p.width as f64 - 0.5 {
            return Ok(json!({"panel":self.index,"drawn":false,"reason":"outside_roi"}));
        }
        let [r, col] = match snap {
            Some(kind) => snapping::apply(&mut reply, p, [r, col], *kind, within.unwrap_or(5.))?,
            None => [r, col],
        };
        let [x, y] = c.pixel(r, col);
        let tint = color.unwrap_or(Color::Yellow).pixel();
        let (cx, cy) = (x as i32, y as i32);
        // Two concentric nine-pixel point symbols, drawn with a dark halo.
        let halo = [56, 254, 254, 511, 495, 511, 254, 254, 56];
        let ring = [56, 254, 198, 387, 387, 387, 198, 254, 56];
        for (mask, rgb) in [(halo, Rgb([0; 3])), (ring, tint)] {
            for (dy, line) in mask.into_iter().enumerate() {
                for dx in 0..9 {
                    let (ix, iy) = (cx + dx - 4, cy + dy as i32 - 4);
                    if c.contains(ix, iy) && (line & (1 << dx) != 0 || dx == 4 && dy == 4) {
                        canvas.put_pixel(ix as u32, iy as u32, rgb);
                    }
                }
            }
        }
        if let Some(label) = label {
            draw::label(canvas, cx + 7, cy - 17, label, tint);
        }
        reply["drawn"] = json!(true);
        reply["image"] = json!(self.image);
        reply["pixel"] = json!([round(x, 1), round(y, 1)]);
        let raw = p.pixels[r.round_ties_even() as usize * p.width + col.round_ties_even() as usize];
        let value = significant(raw);
        reply["value"] = json!(value);
        let ordered = order(p.pixels.iter().copied());
        if value.is_finite() && !ordered.is_empty() {
            reply["percentile"] = json!(percentile(&ordered, value));
            let [lo, hi] = self.limits;
            reply["display"] = if hi > lo {
                json!(round((value - lo) / (hi - lo), 2))
            } else {
                Value::Null
            };
        }
        nearby(&mut reply, p, [r, col], &ordered);
        Ok(reply)
    }
}
