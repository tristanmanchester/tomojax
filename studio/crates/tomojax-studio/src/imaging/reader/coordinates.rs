//! Bounded coordinate reads shared by related planes on one connection.
use super::{FileReads, Storage, pixels};
use serde_json::json;
use std::sync::Arc;

pub(in crate::imaging) type CoordinateVectors = [Option<CoordinateWindow>; 3];

pub(in crate::imaging) struct CoordinateWindow {
    start: usize,
    step: usize,
    positions: Arc<[f64]>,
}

impl CoordinateWindow {
    pub(in crate::imaging) fn new(start: usize, step: usize, positions: Arc<[f64]>) -> Self {
        Self {
            start,
            step,
            positions,
        }
    }

    fn sample(&self, [lo, hi]: [usize; 2], stride: usize) -> Option<Arc<[f64]>> {
        let offset = lo.checked_sub(self.start)?;
        if !offset.is_multiple_of(self.step) || !stride.is_multiple_of(self.step) {
            return None;
        }
        let first = offset / self.step;
        let step = stride / self.step;
        let count = (hi - lo).div_ceil(stride);
        let last = first + (count - 1) * step;
        if last >= self.positions.len() {
            return None;
        }
        if first == 0 && step == 1 && count == self.positions.len() {
            return Some(self.positions.clone());
        }
        Some(
            self.positions[first..=last]
                .iter()
                .step_by(step)
                .copied()
                .collect(),
        )
    }
}

pub(super) fn selected(
    files: &FileReads,
    storage: &Storage,
    count: usize,
    region: [usize; 2],
    stride: usize,
    cached: &mut Option<CoordinateWindow>,
) -> Result<Arc<[f64]>, String> {
    if let Some(positions) = cached.as_ref().and_then(|w| w.sample(region, stride)) {
        return Ok(positions);
    }
    let [lo, hi] = region;
    // Adjacent fixed indices share a small page. Short visible axes share
    // their native grid across preview and exact. Long spans stay decimated.
    let (start, end, step) = if hi - lo == 1 {
        let start = lo / 256 * 256;
        (start, count.min(start + 256), 1)
    } else {
        (lo, hi, if hi - lo <= 8192 { 1 } else { stride })
    };
    let values = pixels(
        files,
        json!({"type":"fetch","storage":storage,"shape":[count],
        "roi":[[start,end]],"stride":[step],"encoding":"float64"}),
    )?
    .doubles(&[(end - start).div_ceil(step)])?;
    let window = cached.insert(CoordinateWindow::new(start, step, values.into()));
    window
        .sample(region, stride)
        .ok_or_else(|| "The fetched coordinates do not cover the requested region.".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sampled_coordinates_keep_the_crop_origin_and_stride() {
        let window = CoordinateWindow {
            start: 12,
            step: 2,
            positions: Arc::from([8., 5., -1., -4.]),
        };
        assert_eq!(window.sample([14, 20], 4).as_deref(), Some(&[5., -4.][..]));
        assert!(window.sample([13, 18], 2).is_none());
        assert!(window.sample([12, 21], 4).is_none());
    }
}
