//! Registered plane selections and the worker pixel encodings.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::{bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions};

use crate::protocol::ArrayDescriptor;

mod stream;
pub use stream::{FrameQueue, PaneStream};
pub(crate) mod wire;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Encoding {
    Float32,
    Float64,
    Uint16,
    Jpeg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pixels {
    pub shape: Vec<usize>,
    pub encoding: Encoding,
    bytes: usize,
    range: Option<[f64; 2]>,
    #[serde(skip)]
    payload: Vec<u8>,
}

impl Pixels {
    fn shuffled<const N: usize>(
        &self,
        shape: &[usize],
        encoding: Encoding,
    ) -> Result<Vec<[u8; N]>, String> {
        if self.shape != shape || self.encoding != encoding {
            return Err("Worker returned a different selection or encoding".into());
        }
        let size = shape
            .iter()
            .try_fold(N, |n, dim| n.checked_mul(*dim))
            .filter(|n| *n <= 16 << 20)
            .ok_or("Selection exceeds the decoded limit")?;
        let raw = zstd::bulk::decompress(&self.payload, size).map_err(|e| e.to_string())?;
        if raw.len() != size {
            return Err("Incomplete shuffled scalar payload".into());
        }
        let count = size / N;
        Ok((0..count)
            .map(|i| std::array::from_fn(|b| raw[b * count + i]))
            .collect())
    }
    pub fn doubles(&self, shape: &[usize]) -> Result<Vec<f64>, String> {
        Ok(self
            .shuffled(shape, Encoding::Float64)?
            .into_iter()
            .map(f64::from_le_bytes)
            .collect())
    }
    pub fn floats(&self, shape: &[usize]) -> Result<Vec<f32>, String> {
        Ok(self
            .shuffled(shape, Encoding::Float32)?
            .into_iter()
            .map(f32::from_le_bytes)
            .collect())
    }
    pub fn volume(&self, shape: &[usize]) -> Result<Vec<f32>, String> {
        let [lo, hi] = self.range.ok_or("Volume has no scalar range")?;
        Ok(self
            .shuffled(shape, Encoding::Uint16)?
            .into_iter()
            .map(|b| (lo + (hi - lo) * f64::from(u16::from_le_bytes(b)) / 65535.) as f32)
            .collect())
    }
    pub fn decode(
        &self,
        selection: &PlaneSelection,
        encoding: Encoding,
    ) -> Result<Vec<f32>, String> {
        if self.shape != selection.count || self.encoding != encoding {
            return Err("Worker returned a different pixel selection or encoding".into());
        }
        match encoding {
            Encoding::Float32 => self.floats(&selection.count),
            Encoding::Float64 | Encoding::Uint16 => Err("Expected a plane encoding".into()),
            Encoding::Jpeg => {
                let [low, high] = self
                    .range
                    .filter(|[a, b]| a.is_finite() && b.is_finite() && a <= b)
                    .ok_or("Invalid JPEG scalar window")?;
                // The wire drops the first singleton axis for a three-dimensional selection.
                let mut image = self.shape.clone();
                if image.len() == 3 {
                    image.remove(
                        image
                            .iter()
                            .position(|&n| n == 1)
                            .ok_or("JPEG selection is not a plane")?,
                    );
                }
                let options = DecoderOptions::default()
                    .set_max_width(image[1])
                    .set_max_height(image[0])
                    .jpeg_set_out_colorspace(ColorSpace::Luma);
                let mut decoder =
                    JpegDecoder::new_with_options(ZCursor::new(&self.payload), options);
                decoder.decode_headers().map_err(|e| e.to_string())?;
                if decoder.dimensions() != Some((image[1], image[0]))
                    || decoder.input_colorspace() != Some(ColorSpace::Luma)
                {
                    return Err(
                        "JPEG dimensions or color space differ from the declared plane".into(),
                    );
                }
                let [low, high] = [low as f32, high as f32];
                let grey = decoder.decode().map_err(|e| e.to_string())?;
                Ok(grey
                    .into_iter()
                    .map(|g| low + (high - low) * (f32::from(g) / 255.))
                    .collect())
            }
        }
    }
}

/// Half-open source row and column bounds, in stored index order.
pub type PlaneRegion = [[usize; 2]; 2];

/// The same strided hyperslab feeds the native reader and a worker fetch.
#[derive(Debug, Clone)]
pub struct PlaneSelection {
    pub start: Vec<usize>,
    pub step: Vec<usize>,
    pub count: Vec<usize>,
    pub remaining: [usize; 2],
    pub region: PlaneRegion,
    pub stride: usize,
}

impl PlaneSelection {
    pub fn new(shape: &[usize], axis: usize, index: usize, edge: usize) -> Result<Self, String> {
        Self::for_region(shape, axis, index, edge, None)
    }

    pub fn for_region(
        shape: &[usize],
        axis: usize,
        index: usize,
        edge: usize,
        region: Option<PlaneRegion>,
    ) -> Result<Self, String> {
        if !(2..=3).contains(&shape.len())
            || shape.contains(&0)
            || edge == 0
            || axis >= shape.len()
            || (shape.len() == 3 && index >= shape[axis])
        {
            return Err("The requested slice lies outside the array.".into());
        }
        let remaining = if shape.len() == 2 {
            [0, 1]
        } else {
            [[1, 2], [0, 2], [0, 1]][axis]
        };
        let region = region.unwrap_or_else(|| remaining.map(|a| [0, shape[a]]));
        if region
            .iter()
            .zip(remaining)
            .any(|(&[lo, hi], a)| lo >= hi || hi > shape[a])
        {
            return Err("The requested region lies outside the plane.".into());
        }
        let [height, width] = region.map(|[lo, hi]| hi - lo);
        let stride = height.max(width).div_ceil(edge);
        let mut start = vec![0; shape.len()];
        let mut step = vec![stride; shape.len()];
        let mut count = vec![1; shape.len()];
        for (a, [lo, hi]) in remaining.into_iter().zip(region) {
            start[a] = lo;
            count[a] = (hi - lo).div_ceil(stride);
        }
        if shape.len() == 3 {
            start[axis] = index;
            step[axis] = 1;
            count[axis] = 1;
        }
        Ok(Self {
            start,
            step,
            count,
            remaining,
            stride,
            region,
        })
    }

    pub fn fetch(
        &self,
        descriptor: &ArrayDescriptor,
        encoding: Encoding,
        window: Option<[f32; 2]>,
    ) -> Value {
        let roi: Vec<_> = self
            .start
            .iter()
            .enumerate()
            .map(|(i, &start)| [start, start + (self.count[i] - 1) * self.step[i] + 1])
            .collect();
        json!({"type":"fetch", "storage":descriptor.storage, "shape":descriptor.shape, "roi":roi,
            "stride":self.step, "encoding":encoding, "window":window})
    }
}

#[cfg(test)]
mod tests;
