//! Registered plane selections and the worker's two pixel encodings.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use zune_jpeg::JpegDecoder;
use zune_jpeg::zune_core::{bytestream::ZCursor, colorspace::ColorSpace, options::DecoderOptions};

use crate::protocol::ArrayDescriptor;

mod stream;
pub use stream::{FrameQueue, PaneStream};
pub(crate) mod wire;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Encoding {
    Float32,
    Jpeg,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pixels {
    pub shape: Vec<usize>,
    pub encoding: Encoding,
    bytes: usize,
    range: Option<[f32; 2]>,
    #[serde(skip)]
    payload: Vec<u8>,
}

impl Pixels {
    pub fn decode(
        &self,
        selection: &PlaneSelection,
        encoding: Encoding,
    ) -> Result<Vec<f32>, String> {
        if self.shape != selection.count || self.encoding != encoding {
            return Err("Worker returned a different pixel selection or encoding".into());
        }
        let count = self
            .shape
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            .filter(|n| *n <= (16 << 20) / if encoding == Encoding::Float32 { 4 } else { 1 })
            .ok_or("Worker plane exceeds the 16 MiB decoded limit")?;
        match encoding {
            Encoding::Float32 => {
                let raw =
                    zstd::bulk::decompress(&self.payload, count * 4).map_err(|e| e.to_string())?;
                if raw.len() != count * 4 {
                    return Err("Incomplete shuffled float32 plane".into());
                }
                Ok((0..count)
                    .map(|i| f32::from_le_bytes(std::array::from_fn(|b| raw[b * count + i])))
                    .collect())
            }
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
                let grey = decoder.decode().map_err(|e| e.to_string())?;
                Ok(grey
                    .into_iter()
                    .map(|g| low + (high - low) * (f32::from(g) / 255.))
                    .collect())
            }
        }
    }
}

/// The same strided hyperslab feeds the native reader and a worker fetch.
#[derive(Debug, Clone)]
pub struct PlaneSelection {
    pub start: Vec<usize>,
    pub step: Vec<usize>,
    pub count: Vec<usize>,
    pub remaining: [usize; 2],
    pub stride: usize,
}

impl PlaneSelection {
    pub fn new(shape: &[usize], axis: usize, index: usize, edge: usize) -> Result<Self, String> {
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
        let stride = shape[remaining[0]].max(shape[remaining[1]]).div_ceil(edge);
        let mut start = vec![0; shape.len()];
        let mut step = vec![stride; shape.len()];
        let mut count: Vec<_> = shape.iter().map(|n| n.div_ceil(stride)).collect();
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
