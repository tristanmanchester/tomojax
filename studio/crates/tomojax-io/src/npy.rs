//! Read-only NumPy mapping. Arrays stay on disk; only the requested selection is copied.
use std::fs::File;
use std::io::{Cursor, Seek};

use memmap2::MmapOptions;
use npyz::{DType, Endianness, NpyHeader, TypeChar};

#[derive(Clone, Copy)]
struct ReadAxis {
    count: usize,
    file_step: usize,
    output_step: usize,
}

pub struct Npy {
    file: File,
    length: usize,
    offset: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
    kind: TypeChar,
    bytes: usize,
    big: bool,
}

impl Npy {
    pub fn open(path: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("Cannot open {path}: {e}"))?;
        // Published worker artifacts and project arrays are immutable while retained by the
        // project. Reference-in-place NPY inputs must likewise not be truncated during viewing.
        // The mapping is read-only, remains owned here, and never escapes as borrowed slices.
        let length = usize::try_from(file.metadata().map_err(|e| e.to_string())?.len())
            .map_err(|_| "Array is too large for this address space.")?;
        let map = unsafe {
            MmapOptions::new()
                .len(length.min(1024 * 1024 + 12))
                .map(&file)
        }
        .map_err(|e| e.to_string())?;
        if map.len() < 12 || &map[..6] != b"\x93NUMPY" {
            return Err("This file is not a complete NumPy array.".into());
        }
        let header_len = match map[6] {
            1 => u16::from_le_bytes([map[8], map[9]]) as usize,
            2 | 3 => u32::from_le_bytes([map[8], map[9], map[10], map[11]]) as usize,
            version => return Err(format!("Unsupported NumPy format version {version}.")),
        };
        if header_len > 1024 * 1024 || header_len > map.len() - 10 {
            return Err("The NumPy header is truncated or exceeds the metadata limit.".into());
        }
        let mut cursor = Cursor::new(&map[..]);
        let header = NpyHeader::from_reader(&mut cursor).map_err(|e| e.to_string())?;
        let offset = cursor.stream_position().map_err(|e| e.to_string())? as usize;
        let DType::Plain(dtype) = header.dtype() else {
            return Err("Structured and object NumPy arrays are not scalar images.".into());
        };
        let kind = dtype.type_char();
        let bytes = dtype
            .num_bytes()
            .ok_or("Object arrays cannot be viewed safely.")?;
        if !matches!(
            (kind, bytes),
            (TypeChar::Float, 2 | 4 | 8)
                | (TypeChar::Int | TypeChar::Uint, 1 | 2 | 4 | 8)
                | (TypeChar::Bool, 1)
        ) {
            return Err(format!("Unsupported scalar image datatype {dtype}."));
        }
        let required = (header.len() as usize)
            .checked_mul(bytes)
            .and_then(|n| n.checked_add(offset))
            .ok_or("Array dimensions overflow.")?;
        if required > length {
            return Err("The NumPy pixel data is truncated.".into());
        }
        Ok(Self {
            shape: header.shape().iter().map(|&n| n as usize).collect(),
            strides: header.strides().iter().map(|&n| n as usize).collect(),
            offset,
            kind,
            bytes,
            big: dtype.endianness() == Endianness::Big,
            file,
            length,
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn read(
        &self,
        start: &[usize],
        step: &[usize],
        count: &[usize],
    ) -> anyhow::Result<Vec<f32>> {
        if self.kind == TypeChar::Float && self.bytes == 4 {
            return self.gather(start, step, count, |bytes| {
                let raw = bytes.try_into().expect("validated four-byte float");
                if self.big {
                    f32::from_be_bytes(raw)
                } else {
                    f32::from_le_bytes(raw)
                }
            });
        }
        self.gather(start, step, count, |bytes| self.scalar(bytes) as f32)
    }

    pub fn read_f64(
        &self,
        start: &[usize],
        step: &[usize],
        count: &[usize],
    ) -> anyhow::Result<Vec<f64>> {
        self.gather(start, step, count, |bytes| self.scalar(bytes))
    }

    fn gather<T: Copy + Default>(
        &self,
        start: &[usize],
        step: &[usize],
        count: &[usize],
        decode: impl Fn(&[u8]) -> T,
    ) -> anyhow::Result<Vec<T>> {
        // Visit the file in storage order, including Fortran-order arrays. Each mapping
        // covers at most 32 MiB even when consecutive columns span a multi-GiB volume.
        let mut axes = [ReadAxis {
            count: 1,
            file_step: 0,
            output_step: 0,
        }; 3];
        let mut first = self.offset;
        let mut output_stride = 1;
        for axis in (0..count.len()).rev() {
            let byte_stride = self.strides[axis] * self.bytes;
            first += start[axis] * byte_stride;
            if count[axis] > 1 {
                axes[axis] = ReadAxis {
                    count: count[axis],
                    file_step: step[axis] * byte_stride,
                    output_step: output_stride,
                };
            }
            output_stride *= count[axis];
        }
        axes.sort_by_key(|a| (a.count > 1, std::cmp::Reverse(a.file_step)));
        let [outer, middle, inner] = axes;
        let window_at = |at: usize| -> std::io::Result<memmap2::Mmap> {
            // The caller holds the same immutable file for the lifetime of every mapping.
            let window = unsafe {
                MmapOptions::new()
                    .offset(at as u64)
                    .len((self.length - at).min(32 << 20))
                    .map(&self.file)?
            };
            #[cfg(unix)]
            if inner.file_step >= 4096 {
                let _ = window.advise(memmap2::Advice::Random);
            }
            Ok(window)
        };
        let mut origin = first;
        let mut window = window_at(origin)?;
        let mut values = vec![T::default(); output_stride];
        for i in 0..outer.count {
            for j in 0..middle.count {
                let row = first + i * outer.file_step + j * middle.file_step;
                let destination = i * outer.output_step + j * middle.output_step;
                for k in 0..inner.count {
                    let at = row + k * inner.file_step;
                    if at - origin + self.bytes > window.len() {
                        window = window_at(at)?;
                        origin = at;
                    }
                    let offset = at - origin;
                    values[destination + k * inner.output_step] =
                        decode(&window[offset..offset + self.bytes]);
                }
            }
        }
        Ok(values)
    }

    fn scalar(&self, data: &[u8]) -> f64 {
        macro_rules! number {
            ($t:ty) => {{
                let bytes = std::array::from_fn(|i| data[i]);
                if self.big {
                    <$t>::from_be_bytes(bytes)
                } else {
                    <$t>::from_le_bytes(bytes)
                }
            }};
        }
        match (self.kind, self.bytes) {
            (TypeChar::Float, 2) => half::f16::from_bits(number!(u16)).to_f64(),
            (TypeChar::Float, 4) => number!(f32) as f64,
            (TypeChar::Float, 8) => number!(f64),
            (TypeChar::Int, 1) => data[0] as i8 as f64,
            (TypeChar::Int, 2) => number!(i16) as f64,
            (TypeChar::Int, 4) => number!(i32) as f64,
            (TypeChar::Int, 8) => number!(i64) as f64,
            (TypeChar::Uint | TypeChar::Bool, 1) => data[0] as f64,
            (TypeChar::Uint, 2) => number!(u16) as f64,
            (TypeChar::Uint, 4) => number!(u32) as f64,
            (TypeChar::Uint, 8) => number!(u64) as f64,
            _ => unreachable!("Numeric datatype was validated on construction"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{SeekFrom, Write};

    #[test]
    fn strided_big_endian_planes_cross_mapping_boundaries_in_both_orders() {
        let shape = [3, 17, 1_000_003];
        for fortran in [false, true] {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            let order = if fortran { "True" } else { "False" };
            let header =
                format!("{{'descr': '>f4', 'fortran_order': {order}, 'shape': (3,17,1000003),}}\n");
            file.write_all(b"\x93NUMPY\x01\x00").unwrap();
            file.write_all(&(header.len() as u16).to_le_bytes())
                .unwrap();
            file.write_all(header.as_bytes()).unwrap();
            let offset = file.stream_position().unwrap();
            file.as_file()
                .set_len(offset + 4 * shape.iter().product::<u64>())
                .unwrap();
            let mut expected = Vec::new();
            for z in [0, 2] {
                for y in [3, 8, 13] {
                    for x in [1, 500_001, 1_000_001] {
                        let index = if fortran {
                            z + 3 * (y + 17 * x)
                        } else {
                            (z * 17 + y) * shape[2] + x
                        };
                        let value = (z * 100 + y * 7 + x) as f32;
                        file.seek(SeekFrom::Start(offset + 4 * index)).unwrap();
                        file.write_all(&value.to_be_bytes()).unwrap();
                        expected.push(value);
                    }
                }
            }
            let array = Npy::open(file.path().to_str().unwrap()).unwrap();
            assert_eq!(
                array
                    .read(&[0, 3, 1], &[2, 5, 500_000], &[2, 3, 3])
                    .unwrap(),
                expected
            );
        }
    }
}
