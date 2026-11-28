//! Snap on the sampled plane within a radius measured in original array indices.
use super::{Panel, order, quantile, round, smooth};
use anyhow::{Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Snap {
    Peak,
    Trough,
    Edge,
}

struct Patch {
    origin: [usize; 2],
    size: [usize; 2],
    pixels: Vec<f64>,
}
impl Patch {
    fn new(panel: &Panel, at: [f64; 2], reach: [f64; 2]) -> Self {
        let range = |d: usize, n: usize| {
            let center = at[d].round_ties_even();
            let extent = reach[d].ceil() + 2.;
            (center - extent).clamp(0., n as f64) as usize
                ..(center + extent + 1.).clamp(0., n as f64) as usize
        };
        let rows = range(0, panel.height);
        let cols = range(1, panel.width);
        let pixels = rows
            .clone()
            .flat_map(|r| {
                panel.pixels[r * panel.width + cols.start..r * panel.width + cols.end]
                    .iter()
                    .copied()
            })
            .collect();
        Self {
            origin: [rows.start, cols.start],
            size: [rows.len(), cols.len()],
            pixels,
        }
    }
    fn at(&self, i: usize) -> [f64; 2] {
        [
            (self.origin[0] + i / self.size[1]) as f64,
            (self.origin[1] + i % self.size[1]) as f64,
        ]
    }
    fn edge(&self, data: &[f64], i: usize) -> f64 {
        let [h, w] = self.size;
        let derivative = |index: usize, n: usize, step: usize| {
            let before = usize::from(index > 0);
            let after = usize::from(index + 1 < n);
            (data[i + after * step] - data[i - before * step]) / (before + after).max(1) as f64
        };
        derivative(i / w, h, w).hypot(derivative(i % w, w, 1))
    }
    fn best(&self, at: [f64; 2], reach: [f64; 2], kind: Snap) -> [f64; 2] {
        let sorted = order(self.pixels.iter().copied());
        let filtered = smooth(
            &self.pixels,
            self.size[1],
            self.size[0],
            quantile(&sorted, 0.5),
        );
        let near = |i: usize| {
            let [r, c] = self.at(i);
            ((r - at[0]) / reach[0]).powi(2) + ((c - at[1]) / reach[1]).powi(2) <= 1.
        };
        let candidates = (0..filtered.len()).filter(|i| near(*i));
        let score = |i: usize| match kind {
            Snap::Peak => filtered[i],
            Snap::Trough => -filtered[i],
            Snap::Edge => self.edge(&filtered, i),
        };
        let Some(best) = candidates
            .clone()
            .reduce(|a, b| if score(b) > score(a) { b } else { a })
        else {
            return at;
        };
        if matches!(kind, Snap::Edge) {
            return self.at(best);
        }
        let [row, col] = self.at(best);
        let sign = if matches!(kind, Snap::Peak) { 1. } else { -1. };
        let close = candidates.filter(|i| {
            let [r, c] = self.at(*i);
            (r - row).abs() <= 1. && (c - col).abs() <= 1. && self.pixels[*i].is_finite()
        });
        self.at(close
            .reduce(|a, b| {
                if self.pixels[b] * sign > self.pixels[a] * sign {
                    b
                } else {
                    a
                }
            })
            .unwrap_or(best))
    }
}
pub(super) fn apply(
    report: &mut Value,
    p: &Panel,
    at: [f64; 2],
    kind: Snap,
    within: f64,
) -> Result<[f64; 2]> {
    ensure!(within > 0., "within must be positive");
    let steps = p.displayed.map(|d| p.window[d][2] as f64);
    let reach = steps.map(|s| within / s);
    let found = Patch::new(p, at, reach).best(at, reach, kind);
    let mut indices = p.window.iter().map(|w| w[0] as f64).collect::<Vec<_>>();
    for j in 0..2 {
        indices[p.displayed[j]] = round(indices[p.displayed[j]] + found[j] * steps[j], 2);
    }
    report["snapped"] = json!(indices);
    report["moved"] = json!(round(
        ((found[0] - at[0]) * steps[0]).hypot((found[1] - at[1]) * steps[1]),
        2
    ));
    Ok(found)
}
