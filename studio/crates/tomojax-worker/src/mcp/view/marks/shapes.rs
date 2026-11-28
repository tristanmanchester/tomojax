//! Plane intersections, sample footprints and clipped colored outlines for annotations.
use super::raster::Mask;
use super::{Cell, Mark, Overlay, Panel, Shape, draw, nearby, round};
use anyhow::{Result, ensure};
use image::{Rgb, RgbImage};
use serde_json::{Value, json};
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
    fn paint(&self, mask: &mut Mask, cell: Option<Cell>, width: i64, ink: u8) {
        // Work in absolute canvas coordinates before truncation. An off-plane
        // endpoint can sit to the left of the cell and still be on the canvas.
        let offset = cell.map_or([0, 0], |c| [c.left as i64, c.top as i64]);
        let point = |xy: [f64; 2]| [0, 1].map(|i| i64::from(xy[i] as i32) - offset[i]);
        let xy = self
            .points
            .iter()
            .map(|&[r, c]| cell.map_or([c, r], |cell| cell.pixel(r, c)))
            .collect::<Vec<_>>();
        let [x, y] = xy[0];
        match self.kind {
            Kind::Text => {}
            Kind::Box => {
                let [u, v] = xy[1];
                let [dx, dy] = cell.map_or([0., 0.], |c| [c.sx / 2., c.sy / 2.]);
                mask.rect(
                    point([x.min(u) - dx, y.min(v) - dy]),
                    point([x.max(u) + dx, y.max(v) + dy]),
                    width,
                    ink,
                );
            }
            Kind::Circle => {
                let [ry, rx] = self.radius;
                let [rx, ry] =
                    cell.map_or([rx, ry], |c| [(rx * c.sx).max(2.), (ry * c.sy).max(2.)]);
                mask.ellipse(point([x - rx, y - ry]), point([x + rx, y + ry]), width, ink);
            }
            Kind::Polygon if cell.is_none() => {
                mask.polygon(&xy.into_iter().map(point).collect::<Vec<_>>(), ink)
            }
            _ => {
                for pair in xy.windows(2) {
                    mask.stroke(point(pair[0]), point(pair[1]), width, ink)
                }
                if self.kind == Kind::Polygon {
                    mask.stroke(point(*xy.last().unwrap()), point(xy[0]), width, ink)
                } else if self.kind == Kind::Arrow {
                    let [u, v] = xy[1];
                    let len = (u - x).hypot(v - y).max(1e-9);
                    let (dx, dy) = ((u - x) / len, (v - y) / len);
                    for side in [-1., 1.] {
                        mask.stroke(
                            point([u, v]),
                            point([u - 9. * dx + side * 5. * dy, v - 9. * dy - side * 5. * dx]),
                            width,
                            ink,
                        )
                    }
                }
            }
        }
    }
    fn samples(&self, p: &Panel) -> Result<(bool, f64)> {
        let [r, c] = self.anchor;
        if self.extended() {
            let mut mask = Mask::new(p.width, p.height);
            self.paint(&mut mask, None, 0, 1);
            let mut sum = 0.;
            let mut n = 0;
            let mut covered = false;
            for (v, px) in p.pixels.iter().zip(mask.pixels) {
                if px != 0 {
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
        if self.kind != Kind::Text {
            let mut layer = Mask::new(cell.width, cell.height);
            self.paint(&mut layer, Some(cell), 4, 1);
            self.paint(&mut layer, Some(cell), 2, 2);
            for (i, ink) in layer
                .pixels
                .into_iter()
                .enumerate()
                .filter(|(_, ink)| *ink != 0)
            {
                target.put_pixel(
                    (cell.left + i % cell.width) as u32,
                    (cell.top + i / cell.width) as u32,
                    if ink == 1 { Rgb([0; 3]) } else { color },
                );
            }
        }
        Ok(())
    }
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
        draw::mark_label(target, [x, y], label, color, s.kind == Kind::Text);
    }
    let mut report = json!({"panel":o.index});
    let sorted = o.measure(&mut report, s.anchor, value, s.extended());
    if s.kind == Kind::Arrow {
        nearby(&mut report, o.panel, s.anchor, &sorted);
    }
    Ok(report)
}
