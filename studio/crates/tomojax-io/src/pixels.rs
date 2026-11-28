//! Encode a registered storage-order hyperslab. The descriptor stays with the client.
use anyhow::{Result, bail, ensure};
use hdf5::{Hyperslab, SliceOrIndex};
use jpeg_encoder::{ColorType, Encoder};
use serde::Deserialize;
use serde_json::{Value, json};
use studio_core::protocol::Storage;

use crate::npy::Npy;
#[cfg(feature = "static-hdf5")]
mod chunks;
mod neighbors;
#[cfg(feature = "static-hdf5")]
mod rows;
mod sec2;
use neighbors::Neighbors;

#[derive(Default)]
pub struct PlaneCache {
    floats: Neighbors<f32>,
    doubles: Neighbors<f64>,
}

use studio_core::pixels::Encoding;

#[derive(Deserialize)]
pub struct Fetch {
    pub storage: Storage,
    shape: Vec<usize>,
    roi: Option<Vec<[usize; 2]>>,
    stride: Option<Vec<usize>>,
    encoding: Encoding,
    window: Option<[f64; 2]>,
}

#[derive(Clone)]
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
            Encoding::Float32 | Encoding::Pco32 => 4,
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
            self.shape.remove(1);
            self.start.remove(1);
            self.count.remove(1);
            self.stride.remove(1);
        }
        Ok(())
    }

    fn slab(&self, strides: &[usize], counts: &[usize]) -> Hyperslab {
        Hyperslab::from(
            self.start
                .iter()
                .zip(strides)
                .zip(counts)
                .map(|((&start, &step), &count)| SliceOrIndex::SliceCount {
                    start,
                    step,
                    count,
                    block: 1,
                })
                .collect::<Vec<_>>(),
        )
    }
    /// A detector plane, or a contiguous stack of them, can be read in storage
    /// order and decimated without asking HDF5 to gather individual pixels.
    fn dense_counts<T>(&self, data: &hdf5::Dataset) -> Option<(usize, Vec<usize>)> {
        let first = self
            .count
            .iter()
            .rposition(|&n| n == 1)
            .map_or(0, |i| i + 1);
        if self.stride[first..].iter().all(|&s| s == 1) {
            return None;
        }
        let chunks = data.chunk();
        // When several selected rows share a chunk, HDF5 can gather them together.
        // Separate frame chunks instead benefit from one contiguous read per row.
        if self.count[..first].iter().any(|&n| n > 1)
            && chunks
                .as_ref()
                .is_some_and(|c| !(0..first).any(|a| self.count[a] > 1 && self.stride[a] >= c[a]))
        {
            return None;
        }
        let mut counts = self.count.clone();
        for (axis, n) in counts.iter_mut().enumerate().skip(first) {
            let step = self.stride[axis];
            if chunks.as_ref().is_some_and(|c| step > c[axis]) {
                return None; // A dense read would decompress chunks the ROI skips.
            }
            let span = (*n - 1) * step + 1;
            let available = self.shape[axis] - self.start[axis];
            // Include a short row tail so adjacent full rows form one run.
            *n = if available - span < step {
                available
            } else {
                span
            };
        }
        let expanded = counts
            .iter()
            .try_fold(1usize, |size, n| size.checked_mul(*n))?;
        let selected = self.count.iter().product::<usize>();
        (expanded <= (64 << 20) / size_of::<T>() && expanded <= selected * 32)
            .then_some((first, counts))
    }

    fn read<T>(&self, data: &hdf5::Dataset) -> Result<Vec<T>>
    where
        T: hdf5::H5Type + Send + Copy + From<u8> + From<u16> + From<f32>,
    {
        #[cfg(feature = "static-hdf5")]
        if let Some(row) = rows::read(data, self) {
            return Ok(row);
        }
        let Some((first, counts)) = self.dense_counts::<T>(data) else {
            return Ok(data
                .read_slice::<T, _, ndarray::IxDyn>(self.slab(&self.stride, &self.count))?
                .into_raw_vec_and_offset()
                .0);
        };
        let mut steps = self.stride.clone();
        steps[first..].fill(1);
        let slab = self.slab(&steps, &counts);
        let dtype = data.dtype()?;
        if dtype.is::<u8>() {
            self.decimate::<u8, T>(data, slab, first)
        } else if dtype.is::<u16>() {
            self.decimate::<u16, T>(data, slab, first)
        } else if dtype.is::<f32>() {
            self.decimate::<f32, T>(data, slab, first)
        } else {
            self.decimate::<T, T>(data, slab, first)
        }
    }

    fn decimate<S: hdf5::H5Type + Copy, T: From<S>>(
        &self,
        data: &hdf5::Dataset,
        slab: Hyperslab,
        first: usize,
    ) -> Result<Vec<T>> {
        let mut block = data.read_slice::<S, _, ndarray::IxDyn>(slab)?;
        for (axis, &step) in self.stride.iter().enumerate().skip(first) {
            if self.count[axis] > 1 && step > 1 {
                block.slice_axis_inplace(
                    ndarray::Axis(axis),
                    ndarray::Slice::new(0, None, step as isize),
                );
            }
        }
        Ok(block.iter().copied().map(T::from).collect())
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

pub fn fetch(cmd: &Fetch, cache: &PlaneCache) -> Result<(Value, Vec<u8>)> {
    let mut selection = Selection::new(cmd)?;
    let shape = selection.count.clone();
    let mut range = None;
    let f32_only = matches!(cmd.encoding, Encoding::Float32 | Encoding::Pco32);
    let (floats, doubles) = match &cmd.storage {
        Storage::Npy { path, .. } => {
            let file = Npy::open(path).map_err(anyhow::Error::msg)?;
            selection.stored(file.shape())?;
            if f32_only {
                (
                    file.read(&selection.start, &selection.stride, &selection.count)?,
                    vec![],
                )
            } else {
                (
                    vec![],
                    file.read_f64(&selection.start, &selection.stride, &selection.count)?,
                )
            }
        }
        Storage::Hdf5 { path, dataset } => {
            let file = hdf5::File::open(path)?;
            let data = file.dataset(dataset)?;
            selection.stored(&data.shape())?;
            if f32_only {
                (cache.floats.read(&data, &selection)?, vec![])
            } else if cmd.encoding == Encoding::Jpeg && !data.filters().is_empty() && {
                let dtype = data.dtype()?;
                dtype.is::<f32>()
                    || dtype.is::<u16>()
                    || dtype.is::<i16>()
                    || dtype.is::<u8>()
                    || dtype.is::<i8>()
            } {
                // All these stored values are exactly representable in f32. A matching
                // preview and quantitative request can share the decompressed strip.
                (
                    vec![],
                    cache
                        .floats
                        .read(&data, &selection)?
                        .into_iter()
                        .map(f64::from)
                        .collect(),
                )
            } else {
                (vec![], cache.doubles.read(&data, &selection)?)
            }
        }
        Storage::Member { .. } => bail!("fetch needs stored data"),
    };
    let (name, bytes) = match cmd.encoding {
        Encoding::Float32 => (
            "float32",
            compress(floats.into_iter().map(f32::to_le_bytes))?,
        ),
        Encoding::Pco32 => (
            "pco32",
            pco::standalone::simple_compress(
                &floats,
                &pco::ChunkConfig::default().with_compression_level(8),
            )?,
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
            let bytes = jpeg(&shape, doubles, span)?;
            ("jpeg", bytes)
        }
    };
    Ok((
        json!({"type":"pixels","shape":shape,"encoding":name,"bytes":bytes.len(),"range":range}),
        bytes,
    ))
}

fn jpeg(shape: &[usize], values: Vec<f64>, span: [f64; 2]) -> Result<Vec<u8>> {
    let mut plane = shape.to_vec();
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
    let gray = values
        .into_iter()
        .map(|v| level(v, span, 255.) as u8)
        .collect::<Vec<_>>();
    let mut bytes = Vec::new();
    let mut encoder = Encoder::new(&mut bytes, 60);
    encoder.set_optimized_huffman_tables(true);
    encoder.encode(&gray, w as u16, h as u16, ColorType::Luma)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_detector_roi_returns_only_the_requested_samples() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("detector.h5")).unwrap();
        let data = file
            .new_dataset::<u16>()
            .shape([3, 19, 31])
            .chunk([1, 8, 16])
            .create("detector")
            .unwrap();
        data.write_raw(&(0..3 * 19 * 31u16).collect::<Vec<_>>())
            .unwrap();
        let command: Fetch = serde_json::from_value(json!({
            "storage":{"type":"hdf5","path":"detector.h5","dataset":"detector"},
            "shape":[3,19,31], "roi":[[1,2],[3,18],[2,31]],"stride":[1,4,6],
            "encoding":"float32"
        }))
        .unwrap();
        let selection = Selection::new(&command).unwrap();
        let expected: Vec<_> = [3, 7, 11, 15]
            .into_iter()
            .flat_map(|y| [2, 8, 14, 20, 26].map(|x| ((19 + y) * 31 + x) as f32))
            .collect();
        assert_eq!(selection.read::<f32>(&data).unwrap(), expected);
    }

    #[test]
    fn sinogram_decimation_preserves_the_outer_view_stride() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("raw.h5")).unwrap();
        let source =
            ndarray::Array3::from_shape_fn((15, 3, 17), |(z, y, x)| (100 * z + 20 * y + x) as u16);
        let data = file
            .new_dataset_builder()
            .with_data(&source)
            .chunk((1, 3, 17))
            .create("counts")
            .unwrap();
        let selection = Selection {
            shape: vec![15, 3, 17],
            start: vec![2, 1, 3],
            count: vec![4, 1, 5],
            stride: vec![4, 1, 3],
        };
        assert!(selection.dense_counts::<f32>(&data).is_some());
        let want = source.slice(ndarray::s![2..15;4,1,3..16;3]).mapv(f32::from);
        assert_eq!(
            selection.read::<f32>(&data).unwrap(),
            want.iter().copied().collect::<Vec<_>>()
        );
    }

    #[test]
    fn levels_before_lossy_encoding() {
        let values = [3., 4., 5., f64::NAN, f64::NEG_INFINITY, f64::INFINITY];
        let span = limits(&values);
        assert_eq!(span, [3., 5.]);
        assert_eq!(values.map(|v| level(v, span, 255.)), [0, 128, 255, 0, 0, 0]);
    }
}
