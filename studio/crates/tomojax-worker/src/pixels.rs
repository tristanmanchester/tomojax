//! Encode a registered storage-order hyperslab. The descriptor stays with the client.
use anyhow::{Result, bail, ensure};
use hdf5::{Hyperslab, SliceOrIndex};
use image::{ExtendedColorType, codecs::jpeg::JpegEncoder};
use serde::Deserialize;
use serde_json::{Value, json};
use studio_core::protocol::Storage;

use crate::npy::Npy;

#[derive(Deserialize, Clone, Copy, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Encoding {
    Float32,
    Float64,
    Uint16,
    Jpeg,
}

#[derive(Deserialize)]
pub struct Fetch {
    pub storage: Storage,
    shape: Vec<usize>,
    roi: Option<Vec<[usize; 2]>>,
    stride: Option<Vec<usize>>,
    encoding: Encoding,
    window: Option<[f64; 2]>,
}

struct Selection {
    shape: Vec<usize>,
    start: Vec<usize>,
    count: Vec<usize>,
    stride: Vec<usize>,
}

impl Selection {
    fn new(cmd: &Fetch) -> Result<Self> {
        let shape = &cmd.shape;
        ensure!((1..=3).contains(&shape.len()), "expected one to three axes");
        let roi = cmd
            .roi
            .clone()
            .unwrap_or_else(|| shape.iter().map(|&n| [0, n]).collect());
        let stride = cmd.stride.clone().unwrap_or_else(|| vec![1; shape.len()]);
        ensure!(
            roi.len() == shape.len() && stride.len() == shape.len(),
            "supply one interval and stride per axis"
        );
        let mut count = Vec::new();
        let mut start = Vec::new();
        for (([lo, hi], n), step) in roi.into_iter().zip(shape).zip(&stride) {
            ensure!(lo < hi && hi <= *n && *step > 0, "invalid selection");
            count.push((hi - lo).div_ceil(*step));
            start.push(lo);
        }
        let width = match cmd.encoding {
            Encoding::Float32 => 4,
            Encoding::Float64 => 8,
            Encoding::Uint16 => 2,
            Encoding::Jpeg => 1,
        };
        ensure!(
            count
                .iter()
                .try_fold(width, |n: usize, &c| n.checked_mul(c))
                .is_some_and(|bytes| bytes <= 16 << 20),
            "decoded selection exceeds 16 MiB"
        );
        Ok(Self {
            shape: shape.clone(),
            start,
            count,
            stride,
        })
    }

    fn stored(&mut self, actual: &[usize]) -> Result<()> {
        if actual != self.shape {
            ensure!(
                self.shape.len() == 3
                    && self.shape[1] == 1
                    && actual == [self.shape[0], self.shape[2]],
                "stored shape differs from the descriptor"
            );
            self.start.remove(1);
            self.count.remove(1);
            self.stride.remove(1);
        }
        Ok(())
    }

    fn slab(&self) -> Hyperslab {
        Hyperslab::from(
            self.start
                .iter()
                .zip(&self.stride)
                .zip(&self.count)
                .map(|((&start, &step), &count)| SliceOrIndex::SliceCount {
                    start,
                    step,
                    count,
                    block: 1,
                })
                .collect::<Vec<_>>(),
        )
    }
}

fn compress<const N: usize>(values: impl ExactSizeIterator<Item = [u8; N]>) -> Result<Vec<u8>> {
    let n = values.len();
    let mut bytes = vec![0; n * N];
    for (i, value) in values.enumerate() {
        for (j, byte) in value.into_iter().enumerate() {
            bytes[j * n + i] = byte;
        }
    }
    Ok(zstd::bulk::compress(&bytes, 1)?)
}

fn limits(values: &[f64]) -> [f64; 2] {
    let mut finite = values.iter().copied().filter(|v| v.is_finite());
    let first = finite.next().unwrap_or(0.);
    finite.fold([first, first], |[lo, hi], v| [lo.min(v), hi.max(v)])
}

fn level(value: f64, [lo, hi]: [f64; 2], max: f64) -> u16 {
    if hi == lo || !value.is_finite() {
        return 0;
    }
    ((value.clamp(lo, hi) - lo) * (max / (hi - lo))).round_ties_even() as u16
}

pub fn fetch(cmd: &Fetch) -> Result<(Value, Vec<u8>)> {
    let mut selection = Selection::new(cmd)?;
    let shape = selection.count.clone();
    let mut range = None;
    let f32_only = cmd.encoding == Encoding::Float32;
    let (floats, doubles) = match &cmd.storage {
        Storage::Npy { path, .. } => {
            let file = Npy::open(path).map_err(anyhow::Error::msg)?;
            selection.stored(file.shape())?;
            if f32_only {
                (
                    file.read(&selection.start, &selection.stride, &selection.count),
                    vec![],
                )
            } else {
                (
                    vec![],
                    file.read_f64(&selection.start, &selection.stride, &selection.count),
                )
            }
        }
        Storage::Hdf5 { path, dataset } => {
            let file = hdf5::File::open(path)?;
            let data = file.dataset(dataset)?;
            selection.stored(&data.shape())?;
            if f32_only {
                (
                    data.read_slice::<f32, _, ndarray::IxDyn>(selection.slab())?
                        .into_raw_vec_and_offset()
                        .0,
                    vec![],
                )
            } else {
                (
                    vec![],
                    data.read_slice::<f64, _, ndarray::IxDyn>(selection.slab())?
                        .into_raw_vec_and_offset()
                        .0,
                )
            }
        }
        Storage::Member { .. } => bail!("fetch needs stored data"),
    };
    let (name, bytes) = match cmd.encoding {
        Encoding::Float32 => (
            "float32",
            compress(floats.into_iter().map(f32::to_le_bytes))?,
        ),
        Encoding::Float64 => (
            "float64",
            compress(doubles.into_iter().map(f64::to_le_bytes))?,
        ),
        Encoding::Uint16 => {
            let span = limits(&doubles);
            range = Some(span);
            (
                "uint16",
                compress(
                    doubles
                        .iter()
                        .map(|&v| level(v, span, 65535.).to_le_bytes()),
                )?,
            )
        }
        Encoding::Jpeg => {
            let span = cmd.window.unwrap_or_else(|| limits(&doubles));
            ensure!(
                span[0].is_finite() && span[1].is_finite() && span[0] <= span[1],
                "window needs two ordered finite limits"
            );
            range = Some(span);
            let mut plane = shape.clone();
            if plane.len() == 3 {
                plane.remove(
                    plane
                        .iter()
                        .position(|&n| n == 1)
                        .ok_or_else(|| anyhow::anyhow!("JPEG requires a plane"))?,
                );
            }
            let [h, w] = plane[..] else {
                bail!("JPEG requires a plane");
            };
            ensure!(
                h <= u16::MAX as usize && w <= u16::MAX as usize,
                "JPEG dimensions exceed 65535"
            );
            let gray = doubles
                .into_iter()
                .map(|v| level(v, span, 255.) as u8)
                .collect::<Vec<_>>();
            let mut bytes = Vec::new();
            JpegEncoder::new_with_quality(&mut bytes, 60).encode(
                &gray,
                w as u32,
                h as u32,
                ExtendedColorType::L8,
            )?;
            ("jpeg", bytes)
        }
    };
    Ok((
        json!({"type":"pixels","shape":shape,"encoding":name,"bytes":bytes.len(),"range":range}),
        bytes,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_before_lossy_encoding() {
        let values = [3., 4., 5., f64::NAN, f64::NEG_INFINITY, f64::INFINITY];
        let span = limits(&values);
        assert_eq!(span, [3., 5.]);
        assert_eq!(values.map(|v| level(v, span, 255.)), [0, 128, 255, 0, 0, 0]);
    }
}
