//! Registered scalar images. Display settings never change these quantitative samples.
use std::sync::Arc;

/// A registered table is retained at full precision; it is never a downsampled image.
#[derive(Clone, Debug)]
pub struct TableData {
    pub values: Arc<[f64]>,
    pub shape: [usize; 2],
    pub axes: [AxisScale; 2],
    pub columns: Vec<studio_core::protocol::Column>,
    pub domain: String,
}
impl TableData {
    pub fn row(&self, row: usize) -> Option<&[f64]> {
        if row >= self.shape[0] {
            return None;
        }
        let start = row * self.shape[1];
        Some(&self.values[start..start + self.shape[1]])
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ColorMap {
    #[default]
    Gray,
    Viridis,
    Magma,
    Bone,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DisplaySettings {
    pub window: Option<[f32; 2]>,
    pub colormap: ColorMap,
    pub invert: bool,
    pub zoom: f32,
    /// Translation as a fraction of the fitted image's width and height.
    pub pan: [f32; 2],
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            window: None,
            colormap: ColorMap::Gray,
            invert: false,
            zoom: 1.,
            pan: [0.; 2],
        }
    }
}

#[derive(Clone, Debug)]
pub struct AxisScale {
    pub name: String,
    pub frame: String,
    pub unit: String,
    /// Pixel-center registration, in stored index order. Signed spacing is preserved.
    pub origin: f64,
    pub spacing: f64,
    /// Nonuniform coordinates when the axis is an explicitly registered vector.
    pub coordinates: Option<Arc<[f64]>>,
}

impl AxisScale {
    pub fn coordinate(&self, index: f64) -> f64 {
        if let Some(values) = &self.coordinates {
            if values.is_empty() {
                return f64::NAN;
            }
            let position = index.clamp(0., values.len().saturating_sub(1) as f64);
            let low = position.floor() as usize;
            let high = (low + 1).min(values.len() - 1);
            values[low] + (values[high] - values[low]) * (position - low as f64)
        } else {
            self.origin + index * self.spacing
        }
    }
}

#[derive(Clone, Debug)]
pub struct Statistics {
    pub min: f32,
    pub max: f32,
    pub mean: f64,
    pub std: f64,
    pub finite_count: usize,
    pub missing_count: usize,
    pub histogram: Vec<u64>,
    pub window: [f32; 2],
}

#[derive(Clone, Debug)]
pub struct PlaneData {
    /// Only lossless frames may supply quantitative readouts.
    pub lossless: bool,
    /// Unique decoded-content identity; used to avoid re-uploading unchanged scalar pixels.
    pub id: u64,
    pub values: Arc<[f32]>,
    /// Stored row, column dimensions of the displayed samples.
    pub shape: [usize; 2],
    pub source_shape: [usize; 2],
    /// Half-open source row and column bounds covered by this texture.
    pub region: studio_core::pixels::PlaneRegion,
    /// A finer selection of this same plane, registered in source pixels.
    pub detail: Option<Arc<PlaneData>>,
    pub axes: [AxisScale; 2],
    pub fixed: Vec<studio_core::protocol::Fixed>,
    pub index: usize,
    pub count: usize,
    pub value_unit: String,
    pub domain: String,
    /// Source pixels per displayed pixel. One means full resolution.
    pub stride: usize,
    pub statistics: Statistics,
}

impl PlaneData {
    /// Subtract registered display samples. Callers verify the full volume grids first.
    /// Whether two planes sample the same grid, so they can be combined pixel by pixel.
    pub fn same_grid(&self, other: &Self) -> bool {
        self.shape == other.shape
            && self.region == other.region
            && self.stride == other.stride
            && self.source_shape == other.source_shape
            && self.fixed == other.fixed
            && self.value_unit == other.value_unit
            && self.domain == other.domain
    }

    /// Left minus right, on a shared grid.
    pub fn difference(&self, right: &Self) -> Option<Self> {
        self.combine_layers(right, |l, r| l.composite(r, |_, a, b| a - b))
    }

    /// Left samples up to `fraction` of the width, right samples after it, on a shared grid.
    ///
    /// The composite is drawn with the left window, so right samples are mapped from
    /// `windows[1]` onto `windows[0]` to keep the right side's own contrast.
    pub fn split(&self, right: &Self, fraction: f32, windows: [[f32; 2]; 2]) -> Option<Self> {
        let [[left_low, left_high], [right_low, right_high]] = windows;
        let scale = (left_high - left_low) / (right_high - right_low);
        self.combine_layers(right, |l, r| {
            let edge = l.split_column(fraction);
            l.composite(r, |i, a, b| {
                if i % l.shape[1] < edge {
                    a
                } else {
                    left_low + (b - right_low) * scale
                }
            })
        })
    }

    pub fn split_column(&self, fraction: f32) -> usize {
        let [start, end] = self.region[1];
        let cut = fraction.clamp(0., 1.) * self.source_shape[1] as f32;
        ((cut - start as f32) * self.shape[1] as f32 / (end - start) as f32).round() as usize
    }

    fn combine_layers(
        &self,
        right: &Self,
        combine: impl Fn(&Self, &Self) -> Option<Self>,
    ) -> Option<Self> {
        let mut plane = combine(self, right)?;
        plane.detail = self
            .detail
            .as_ref()
            .zip(right.detail.as_ref())
            .and_then(|(l, r)| combine(l, r))
            .map(Arc::new);
        Some(plane)
    }

    pub(crate) fn layers(&self, display: &DisplaySettings) -> impl Iterator<Item = (bool, &Self)> {
        std::iter::once((false, self)).chain(
            self.detail
                .as_deref()
                .filter(|_| display.zoom > 1.)
                .map(|d| (true, d)),
        )
    }

    pub fn content_id(&self) -> [u64; 2] {
        [self.id, self.detail.as_ref().map_or(0, |d| d.id)]
    }

    fn composite(&self, right: &Self, f: impl Fn(usize, f32, f32) -> f32) -> Option<Self> {
        if !self.same_grid(right) {
            return None;
        }
        let values: Arc<[f32]> = self
            .values
            .iter()
            .zip(right.values.iter())
            .enumerate()
            .map(|(i, (a, b))| f(i, *a, *b))
            .collect();
        let statistics = super::reader::statistics(&values).ok()?;
        Some(Self {
            id: super::reader::identity(),
            values,
            statistics,
            lossless: self.lossless && right.lossless,
            ..self.clone()
        })
    }
    pub fn aspect(&self) -> f32 {
        let ratio =
            if self.axes[0].unit == self.axes[1].unit && self.axes[0].frame == self.axes[1].frame {
                (self.axes[1].spacing / self.axes[0].spacing).abs()
            } else {
                1.
            };
        (self.shape[1] as f64 / self.shape[0] as f64 * ratio) as f32
    }

    pub fn sample(&self, row: usize, column: usize) -> Option<f32> {
        if !self.lossless || row >= self.shape[0] || column >= self.shape[1] {
            return None;
        }
        let value = self.values[row * self.shape[1] + column];
        value.is_finite().then_some(value)
    }

    /// Choose the texture painted at this position in the whole plane.
    pub fn layer_at(&self, [u, v]: [f32; 2]) -> &Self {
        self.detail
            .as_deref()
            .filter(|d| {
                [v, u].into_iter().enumerate().all(|(axis, x)| {
                    let [lo, hi] = d.region[axis];
                    x >= lo as f32 / self.source_shape[axis] as f32
                        && (x < hi as f32 / self.source_shape[axis] as f32
                            || (x == 1. && hi == self.source_shape[axis]))
                })
            })
            .unwrap_or(self)
    }

    pub fn texel(&self, [u, v]: [f32; 2]) -> [usize; 2] {
        std::array::from_fn(|axis| {
            let [lo, hi] = self.region[axis];
            let position =
                ([v, u][axis] * self.source_shape[axis] as f32 - lo as f32) / (hi - lo) as f32;
            ((position * self.shape[axis] as f32) as usize).min(self.shape[axis] - 1)
        })
    }

    pub fn displayed_index(&self, uv: [f32; 2]) -> [usize; 2] {
        let layer = self.layer_at(uv);
        let texel = layer.texel(uv);
        std::array::from_fn(|i| layer.region[i][0] + texel[i] * layer.stride)
    }

    pub fn probe(&self, uv: [f32; 2]) -> Option<([usize; 2], f32)> {
        let layer = self.layer_at(uv);
        let [row, col] = layer.texel(uv);
        layer
            .sample(row, col)
            .map(|value| (self.displayed_index(uv), value))
    }

    /// Source pixels visible inside the fitted pane after pan and zoom.
    pub fn visible_region(
        &self,
        display: &DisplaySettings,
        edge: usize,
    ) -> Option<studio_core::pixels::PlaneRegion> {
        if display.zoom <= 1. || self.stride == 1 {
            return None;
        }
        let region: [[usize; 2]; 2] = std::array::from_fn(|a| {
            let n = self.source_shape[a] as f32;
            let start = (0.5 + (-0.5 - display.pan[1 - a]) / display.zoom) * n;
            let end = (0.5 + (0.5 - display.pan[1 - a]) / display.zoom) * n;
            [
                start.floor().clamp(0., n) as usize,
                end.ceil().clamp(0., n) as usize,
            ]
        });
        let lengths = region.map(|[lo, hi]| hi - lo);
        (!lengths.contains(&0) && lengths.into_iter().max()?.div_ceil(edge) < self.stride)
            .then_some(region)
    }

    /// Distance between sample-center coordinates; rejects unlike coordinate frames/units.
    pub fn distance(&self, from: [f64; 2], to: [f64; 2]) -> Result<f64, String> {
        if self.axes[0].unit != self.axes[1].unit || self.axes[0].frame != self.axes[1].frame {
            return Err("A physical distance needs two axes in the same frame and unit.".into());
        }
        let delta: [f64; 2] = std::array::from_fn(|axis| {
            self.axes[axis].coordinate(to[axis]) - self.axes[axis].coordinate(from[axis])
        });
        Ok(delta[0].hypot(delta[1]))
    }

    /// Samples a registered line in displayed pixel coordinates, preserving missing pixels.
    pub fn profile(&self, from: [f64; 2], to: [f64; 2], count: usize) -> Vec<Option<f32>> {
        let count = count.clamp(2, 2048);
        (0..count)
            .map(|i| {
                let t = i as f64 / (count - 1) as f64;
                let row = (from[0] + t * (to[0] - from[0])).round();
                let col = (from[1] + t * (to[1] - from[1])).round();
                if row < 0. || col < 0. {
                    None
                } else {
                    self.sample(row as usize, col as usize)
                }
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct VolumeData {
    pub id: u64,
    /// Normalized scalar half floats, C order (z,y,x); original values use scalar_range.
    pub texels: Arc<[u16]>,
    pub shape: [usize; 3],
    pub source_shape: [usize; 3],
    /// Physical voxel-edge extent in x,y,z. Anisotropy is retained by the ray marcher.
    pub extent: [f32; 3],
    pub origin: [f64; 3],
    pub spacing: [f64; 3],
    pub scalar_range: [f32; 2],
    pub statistics: Statistics,
    pub value_unit: String,
    pub length_unit: String,
    pub stride: usize,
}

/// The exact fitted image transform shared by overlays, hit testing and the GPU shader.
pub fn image_rect(canvas: [f32; 4], aspect: f32, display: &DisplaySettings) -> [f32; 4] {
    let width = canvas[2].min(canvas[3] * aspect);
    let height = width / aspect;
    let zoom = display.zoom.clamp(0.05, 100.);
    [
        canvas[0] + (canvas[2] - width * zoom) * 0.5 + display.pan[0] * width,
        canvas[1] + (canvas[3] - height * zoom) * 0.5 + display.pan[1] * height,
        width * zoom,
        height * zoom,
    ]
}
