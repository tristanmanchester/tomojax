//! Plane intersections, sample footprints and clipped colored outlines for annotations.
use super::{Cell, Mark, Overlay, Panel, Shape, draw, nearby, round};
use anyhow::{Result, ensure};
use image::{Rgb, RgbImage};
use serde_json::{Value, json};
use tiny_skia::{FillRule, Paint, Path, PathBuilder, Pixmap, Rect, Stroke, Transform};

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Text,
    Box,
    Circle,
    Arrow,
    Polygon,
    Polyline,
}
struct Projection {
    kind: Kind,
    points: Vec<[f64; 2]>,
    radius: [f64; 2],
    away: f64,
    anchor: [f64; 2],
}
impl Projection {
    fn new(mark: &Mark, p: &Panel) -> Result<Self> {
        let positions = mark.shape.positions();
        let normal = |i: usize| {
            p.normal
                .map_or(0., |d| positions[i][d] - p.window[d][0] as f64)
        };
        let (kind, radius, away) = match &mark.shape {
            Shape::Text { .. } => {
                ensure!(
                    mark.label.as_ref().is_some_and(|l| !l.is_empty()),
                    "A text mark needs a label"
                );
                (Kind::Text, 0., normal(0).abs())
            }
            Shape::Circle { radius, .. } => {
                ensure!(*radius > 0., "radius must be positive");
                let distance = normal(0).abs();
                (
                    Kind::Circle,
                    (radius.powi(2) - distance.powi(2)).max(0.).sqrt(),
                    if distance > radius + 2. { distance } else { 0. },
                )
            }
            Shape::Box { .. } => {
                let (a, b) = (normal(0), normal(1));
                (Kind::Box, 0., 0_f64.max(a.min(b)).max(-a.max(b)))
            }
            Shape::Arrow { .. } => (Kind::Arrow, 0., normal(1).abs()),
            Shape::Polyline { .. } | Shape::Polygon { .. } => {
                let polygon = matches!(mark.shape, Shape::Polygon { .. });
                let min = if polygon { 3 } else { 2 };
                ensure!(
                    positions.len() >= min,
                    "{} needs at least {min} points",
                    mark.shape.kind()
                );
                (
                    if polygon {
                        Kind::Polygon
                    } else {
                        Kind::Polyline
                    },
                    0.,
                    (0..positions.len())
                        .map(|i| normal(i).abs())
                        .fold(0., f64::max),
                )
            }
            Shape::Point { .. } => unreachable!(),
        };
        let points: Vec<_> = positions
            .iter()
            .map(|q| {
                p.displayed
                    .map(|d| (q[d] - p.window[d][0] as f64) / p.window[d][2] as f64)
            })
            .collect();
        let anchor = if kind == Kind::Arrow {
            points[1]
        } else {
            [0, 1].map(|i| points.iter().map(|q| q[i]).sum::<f64>() / points.len() as f64)
        };
        Ok(Self {
            kind,
            points,
            anchor,
            radius: p.displayed.map(|d| radius / p.window[d][2] as f64),
            away,
        })
    }
    fn extended(&self) -> bool {
        matches!(
            self.kind,
            Kind::Box | Kind::Circle | Kind::Polygon | Kind::Polyline
        )
    }
    // Pixel paths are relative to the upper-left corner of the drawn plane.
    fn path(&self, scales: [f64; 2], samples: bool) -> Option<Path> {
        let [sx, sy] = scales;
        let coords = self
            .points
            .iter()
            .map(|[r, c]| {
                if samples {
                    [c.trunc() + 0.5, r.trunc() + 0.5]
                } else {
                    [(c + 0.5) * sx, (r + 0.5) * sy]
                }
            })
            .collect::<Vec<_>>();
        let [x, y] = coords[0];
        let mut path = PathBuilder::new();
        match self.kind {
            Kind::Text => return None,
            Kind::Box => {
                let [x1, y1] = coords[1];
                let rect = Rect::from_ltrb(
                    (x.min(x1) - sx / 2.) as f32,
                    (y.min(y1) - sy / 2.) as f32,
                    (x.max(x1) + sx / 2.) as f32,
                    (y.max(y1) + sy / 2.) as f32,
                )?;
                path.push_rect(rect);
            }
            Kind::Circle => {
                let [ry, rx] = self.radius;
                let (x0, y0, x1, y1) = if samples {
                    let [r, c] = self.points[0];
                    (
                        (c - rx).trunc(),
                        (r - ry).trunc(),
                        (c + rx).trunc() + 1.,
                        (r + ry).trunc() + 1.,
                    )
                } else {
                    let (rx, ry) = ((rx * sx).max(2.), (ry * sy).max(2.));
                    (x - rx, y - ry, x + rx, y + ry)
                };
                path.push_oval(Rect::from_ltrb(x0 as f32, y0 as f32, x1 as f32, y1 as f32)?);
            }
            _ => {
                path.move_to(x as f32, y as f32);
                for [u, v] in &coords[1..] {
                    path.line_to(*u as f32, *v as f32);
                }
                if self.kind == Kind::Polygon {
                    path.close()
                }
                if !samples && self.kind == Kind::Arrow {
                    let [u, v] = coords[1];
                    let length = (u - x).hypot(v - y).max(1e-9);
                    let (dx, dy) = ((u - x) / length, (v - y) / length);
                    for side in [-1., 1.] {
                        path.move_to(u as f32, v as f32);
                        path.line_to(
                            (u - 9. * dx + side * 5. * dy) as f32,
                            (v - 9. * dy - side * 5. * dx) as f32,
                        );
                    }
                }
            }
        }
        path.finish()
    }
    fn samples(&self, p: &Panel) -> Result<(bool, f64)> {
        let [r, c] = self.anchor;
        if self.extended() {
            let mut mask = canvas(p.width, p.height)?;
            if let Some(path) = self.path([1., 1.], true) {
                let paint = paint(Rgb([255; 3]));
                if self.kind != Kind::Polyline {
                    mask.fill_path(
                        &path,
                        &paint,
                        FillRule::EvenOdd,
                        Transform::identity(),
                        None,
                    );
                }
                if matches!(self.kind, Kind::Polyline | Kind::Polygon) {
                    stroke(&mut mask, &path, &paint, 1.)
                }
            }
            let mut sum = 0.;
            let mut n = 0;
            let mut covered = false;
            for (v, px) in p.pixels.iter().zip(mask.pixels()) {
                if px.alpha() != 0 {
                    covered = true;
                    if v.is_finite() {
                        sum += v;
                        n += 1;
                    }
                }
            }
            if covered {
                return Ok((true, sum / f64::from(n)));
            }
        }
        let inside =
            r >= -0.5 && r < p.height as f64 - 0.5 && c >= -0.5 && c < p.width as f64 - 0.5;
        Ok((
            inside,
            if inside {
                p.pixels[r.round_ties_even() as usize * p.width + c.round_ties_even() as usize]
            } else {
                f64::NAN
            },
        ))
    }
    fn outline(&self, target: &mut RgbImage, cell: Cell, color: Rgb<u8>) -> Result<()> {
        let Some(path) = self.path([cell.sx, cell.sy], false) else {
            return Ok(());
        };
        let mut layer = canvas(cell.width, cell.height)?;
        for (w, c) in [(4., Rgb([0; 3])), (2., color)] {
            stroke(&mut layer, &path, &paint(c), w)
        }
        for (i, px) in layer
            .pixels()
            .iter()
            .enumerate()
            .filter(|(_, p)| p.alpha() != 0)
        {
            target.put_pixel(
                (cell.left + i % cell.width) as u32,
                (cell.top + i / cell.width) as u32,
                Rgb([px.red(), px.green(), px.blue()]),
            );
        }
        Ok(())
    }
}
fn canvas(w: usize, h: usize) -> Result<Pixmap> {
    Pixmap::new(w.try_into()?, h.try_into()?)
        .ok_or_else(|| anyhow::anyhow!("Invalid drawing dimensions"))
}
fn paint(c: Rgb<u8>) -> Paint<'static> {
    let mut p = Paint {
        anti_alias: false,
        ..Paint::default()
    };
    p.set_color_rgba8(c[0], c[1], c[2], 255);
    p
}
fn stroke(img: &mut Pixmap, path: &Path, paint: &Paint, width: f32) {
    img.stroke_path(
        path,
        paint,
        &Stroke {
            width,
            ..Stroke::default()
        },
        Transform::identity(),
        None,
    );
}
pub(super) fn draw(
    o: &Overlay<'_>,
    mark: &Mark,
    target: &mut RgbImage,
    color: Rgb<u8>,
) -> Result<Value> {
    let s = Projection::new(mark, o.panel)?;
    if s.away > 2. {
        return Ok(
            json!({"panel":o.index,"drawn":false,"reason":"off_plane","distance":round(s.away,3)}),
        );
    }
    let (inside, value) = s.samples(o.panel)?;
    if !inside {
        return Ok(json!({"panel":o.index,"drawn":false,"reason":"outside_roi"}));
    }
    s.outline(target, o.cell, color)?;
    if let Some(label) = &mark.label {
        let [x, y] = o.cell.pixel(s.anchor[0], s.anchor[1]);
        let (dx, dy) = if s.kind == Kind::Text {
            (-draw::advance(label) / 2, -6)
        } else {
            (7, -17)
        };
        draw::label(target, x as i32 + dx, y as i32 + dy, label, color);
    }
    let mut report = json!({"panel":o.index});
    let sorted = o.measure(&mut report, s.anchor, value, s.extended());
    if s.kind == Kind::Arrow {
        nearby(&mut report, o.panel, s.anchor, &sorted);
    }
    Ok(report)
}
