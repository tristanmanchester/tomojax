//! Read independent gzip chunks under the HDF5 lock and inflate outside it.
//! Other filters, skipped filters and fill chunks use HDF5's normal reader.
use super::Selection;
use hdf5::filters::Filter;
use ndarray::Array3;
use std::thread;

#[derive(Clone, Copy)]
struct Tile {
    origin: u64,
    input: usize,
    output: usize,
    count: usize,
}

fn tiles(start: usize, step: usize, count: usize, chunk: usize) -> impl Iterator<Item = Tile> {
    let mut output = 0;
    std::iter::from_fn(move || {
        if output == count {
            return None;
        }
        let at = start + output * step;
        let input = at % chunk;
        let n = (chunk - input).div_ceil(step).min(count - output);
        let tile = Tile {
            origin: (at - input) as u64,
            input,
            output,
            count: n,
        };
        output += n;
        Some(tile)
    })
}

struct ChunkReader<'a> {
    data: &'a hdf5::Dataset,
    shape: [usize; 3],
    bytes: usize,
}

impl ChunkReader<'_> {
    fn inflate(&self, origin: [u64; 3], encoded: &mut [u8], decoded: &mut [u8]) -> Option<()> {
        use hdf5_sys::{h5d::H5Dread_chunk2, h5p::H5P_DEFAULT};
        let mut length = encoded.len();
        let mut mask = 0;
        // The packaged HDF5 2 reader takes the buffer capacity and returns the
        // actual size (without writing if it would exceed that capacity).
        let result = hdf5::sync::sync(|| unsafe {
            H5Dread_chunk2(
                self.data.id(),
                H5P_DEFAULT,
                origin.as_ptr(),
                &mut mask,
                encoded.as_mut_ptr().cast(),
                &mut length,
            )
        });
        if result < 0 || mask != 0 || length == 0 || length > encoded.len() {
            return None;
        }
        let mut size = decoded.len() as _;
        // Both slices are live for this call and zlib receives their exact bounds.
        let status = unsafe {
            libz_sys::uncompress(
                decoded.as_mut_ptr(),
                &mut size,
                encoded.as_ptr(),
                length as _,
            )
        };
        (status == libz_sys::Z_OK && size as usize == decoded.len()).then_some(())
    }

    fn rows<const N: usize, T: Copy>(
        &self,
        s: &Selection,
        z_from: usize,
        out: &mut [T],
        value: fn([u8; N]) -> T,
    ) -> Option<()> {
        let (height, width) = (s.count[1], s.count[2]);
        let mut encoded = vec![0; self.bytes + (self.bytes >> 8) + 64];
        let mut decoded = vec![0; self.bytes];
        let z_count = out.len() / (height * width);
        for z in tiles(
            s.start[0] + z_from * s.stride[0],
            s.stride[0],
            z_count,
            self.shape[0],
        ) {
            for y in tiles(s.start[1], s.stride[1], height, self.shape[1]) {
                for x in tiles(s.start[2], s.stride[2], width, self.shape[2]) {
                    self.inflate([z.origin, y.origin, x.origin], &mut encoded, &mut decoded)?;
                    for iz in 0..z.count {
                        let z_in = (z.input + iz * s.stride[0]) * self.shape[1];
                        let z_out = (z.output + iz) * height;
                        for iy in 0..y.count {
                            let input =
                                ((z_in + y.input + iy * s.stride[1]) * self.shape[2] + x.input) * N;
                            let output = (z_out + y.output + iy) * width + x.output;
                            for (ix, dest) in out[output..output + x.count].iter_mut().enumerate() {
                                let at = input + ix * s.stride[2] * N;
                                *dest = value(decoded[at..at + N].try_into().ok()?);
                            }
                        }
                    }
                }
            }
        }
        Some(())
    }

    fn read<const N: usize, T: Copy + Send + From<u8>>(
        &self,
        s: &Selection,
        value: fn([u8; N]) -> T,
    ) -> Option<Array3<T>> {
        let count: [usize; 3] = s.count.as_slice().try_into().ok()?;
        let mut out = Array3::from_elem(count, 0u8.into());
        let threads = thread::available_parallelism().ok()?.get().min(4);
        let per = count[0].div_ceil(threads);
        thread::scope(|scope| {
            let mut tasks = Vec::new();
            for (i, dest) in out
                .as_slice_mut()?
                .chunks_mut(per * count[1] * count[2])
                .enumerate()
            {
                tasks.push(
                    thread::Builder::new()
                        .spawn_scoped(scope, move || self.rows(s, i * per, dest, value))
                        .ok()?,
                );
            }
            for task in tasks {
                task.join().ok()??;
            }
            Some(())
        })?;
        Some(out)
    }
}

pub(super) fn read<T: Copy + Send + From<u8> + From<u16> + From<f32>>(
    data: &hdf5::Dataset,
    s: &Selection,
) -> Option<Array3<T>> {
    let shape: [usize; 3] = data.chunk()?.try_into().ok()?;
    let dtype = data.dtype().ok()?;
    let bytes = shape
        .into_iter()
        .try_fold(dtype.size(), usize::checked_mul)?;
    // Amortise raw-chunk setup on substantial transverse reads and bound the
    // transient encoded and decoded buffers of each inflater.
    if !(32768..=4 << 20).contains(&bytes)
        || (s.count[0] - 1) * s.stride[0] / shape[0] < 3
        || !matches!(data.filters().as_slice(), [Filter::Deflate(_)])
    {
        return None;
    }
    let reader = ChunkReader { data, shape, bytes };
    if dtype.is::<f32>() {
        reader.read(s, |b| f32::from_ne_bytes(b).into())
    } else if dtype.is::<u16>() {
        reader.read(s, |b| u16::from_ne_bytes(b).into())
    } else if dtype.is::<u8>() {
        reader.read(s, |[b]| b.into())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdf5::File;
    use ndarray::Array3;

    #[test]
    fn native_chunks_match_hdf5_across_partial_boundaries_and_fall_back_for_fill() {
        let dir = tempfile::tempdir().unwrap();
        let file = File::create(dir.path().join("chunks.h5")).unwrap();
        let values = Array3::from_shape_fn((39, 66, 129), |(z, y, x)| {
            if (z + y + x) % 71 == 0 {
                f32::from_bits(0x7fc0_1234)
            } else if (z + y + x) % 19 == 0 {
                -0.0
            } else {
                (z * 10000 + y * 130 + x) as f32
            }
        });
        let data = file
            .new_dataset_builder()
            .with_data(&values)
            .chunk((4, 64, 128))
            .deflate(1)
            .create("floats")
            .unwrap();
        let selection = Selection {
            shape: vec![39, 66, 129],
            start: vec![1, 5, 1],
            count: vec![19, 21, 64],
            stride: vec![2, 3, 2],
        };
        let expected = selection.read::<f32>(&data).unwrap();
        let actual = read::<f32>(&data, &selection).unwrap();
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        let counts = values.mapv(|v| v as u16);
        let data = file
            .new_dataset_builder()
            .with_data(&counts)
            .chunk((4, 64, 128))
            .deflate(4)
            .create("counts")
            .unwrap();
        assert!(
            read::<f64>(&data, &selection)
                .unwrap()
                .iter()
                .copied()
                .eq(selection.read::<f64>(&data).unwrap())
        );
        let data = file
            .new_dataset_builder()
            .with_data(&counts)
            .chunk((4, 64, 128))
            .shuffle()
            .deflate(1)
            .create("shuffled")
            .unwrap();
        assert!(read::<f32>(&data, &selection).is_none());
        let data = file
            .new_dataset::<u16>()
            .shape((39, 66, 129))
            .chunk((4, 64, 128))
            .fill_value(37u16)
            .deflate(1)
            .create("sparse")
            .unwrap();
        assert!(read::<f32>(&data, &selection).is_none());
        assert!(
            selection
                .read::<f32>(&data)
                .unwrap()
                .iter()
                .all(|&v| v == 37.0)
        );
    }
}
