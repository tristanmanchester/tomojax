//! Convert a pixel in a recent view image to a position in the original array.
use super::{Panel, draw::Cell};
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::VecDeque;
pub(super) struct Placement {
    cell: Cell,
    window: Vec<[usize; 3]>,
    displayed: [usize; 2],
}
impl Placement {
    pub(super) fn new(panel: &Panel, cell: Cell) -> Self {
        Self {
            cell,
            window: panel.window.clone(),
            displayed: panel.displayed,
        }
    }
    fn indices(&self, [x, y]: [f64; 2]) -> Option<Vec<f64>> {
        let c = self.cell;
        let (dx, dy) = (x - c.left as f64, y - c.top as f64);
        if !(0.0..c.width as f64).contains(&dx) || !(0.0..c.height as f64).contains(&dy) {
            return None;
        }
        let mut at = self.window.iter().map(|w| w[0] as f64).collect::<Vec<_>>();
        for (axis, p) in self
            .displayed
            .into_iter()
            .zip([dy / c.sy - 0.5, dx / c.sx - 0.5])
        {
            at[axis] =
                ((at[axis] + p * self.window[axis][2] as f64) * 100.).round_ties_even() / 100.;
        }
        Some(at)
    }
}
#[derive(Default)]
pub(crate) struct Placements(VecDeque<(String, Vec<Placement>)>);
impl Placements {
    pub(super) fn insert(&mut self, id: String, planes: Vec<Placement>) {
        if self.0.len() == 256 {
            self.0.pop_front();
        }
        self.0.push_back((id, planes));
    }
    fn position(&self, v: &mut Value, index: usize) -> Result<()> {
        if !v.is_object() {
            return Ok(());
        }
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Position {
            pixel: [f64; 2],
            image: String,
        }
        let Position { pixel, image } = serde_json::from_value(v.clone())?;
        let (_, planes) = self
            .0
            .iter()
            .find(|(id, _)| *id == image)
            .with_context(|| format!("Unknown {image}. Use one of the last 256 view images."))?;
        let Some(at) = planes.iter().find_map(|p| p.indices(pixel)) else {
            bail!(
                "marks[{index}]: pixel ({}, {}) is not on a plane of {image}",
                pixel[0],
                pixel[1]
            )
        };
        *v = json!(at);
        Ok(())
    }
    pub(super) fn locate(&self, marks: &mut [Value]) -> Result<()> {
        for (i, mark) in marks.iter_mut().enumerate() {
            for key in ["at", "from", "to"] {
                if let Some(v) = mark.get_mut(key) {
                    self.position(v, i)?
                }
            }
            for key in ["corners", "points"] {
                if let Some(v) = mark.get_mut(key).and_then(Value::as_array_mut) {
                    for p in v {
                        self.position(p, i)?
                    }
                }
            }
        }
        Ok(())
    }
}
