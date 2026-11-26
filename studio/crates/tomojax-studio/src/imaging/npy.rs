//! Read-only NumPy mapping. Arrays stay on disk; only the requested selection is copied.
use std::fs::File;
use std::io::{Cursor, Seek};

use memmap2::{Mmap, MmapOptions};
use npyz::{DType, Endianness, NpyHeader, TypeChar};

pub(super) struct Npy {
    map: Mmap,
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
        let map = unsafe { MmapOptions::new().map(&file) }.map_err(|e| e.to_string())?;
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
        if required > map.len() {
            return Err("The NumPy pixel data is truncated.".into());
        }
        Ok(Self {
            shape: header.shape().iter().map(|&n| n as usize).collect(),
            strides: header.strides().iter().map(|&n| n as usize).collect(),
            offset,
            kind,
            bytes,
            big: dtype.endianness() == Endianness::Big,
            map,
        })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn read(&self, start: &[usize], step: &[usize], count: &[usize]) -> Vec<f32> {
        let total: usize = count.iter().product();
        let native_f32 = if self.kind == TypeChar::Float
            && self.bytes == 4
            && self.big == cfg!(target_endian = "big")
        {
            bytemuck::try_cast_slice::<u8, f32>(&self.map[self.offset..]).ok()
        } else {
            None
        };
        let mut result = Vec::with_capacity(total);
        for linear in 0..total {
            let mut remainder = linear;
            let mut source = 0;
            for axis in (0..count.len()).rev() {
                let coordinate = remainder % count[axis];
                remainder /= count[axis];
                source += (start[axis] + coordinate * step[axis]) * self.strides[axis];
            }
            result.push(native_f32.map_or_else(|| self.scalar(source) as f32, |data| data[source]));
        }
        result
    }

    pub fn coordinates(&self) -> Vec<f64> {
        (0..self.shape[0])
            .map(|i| self.scalar(i * self.strides[0]))
            .collect()
    }

    pub fn table(&self) -> Vec<f64> {
        debug_assert_eq!(self.shape.len(), 2);
        (0..self.shape[0])
            .flat_map(|row| {
                (0..self.shape[1]).map(move |column| {
                    self.scalar(row * self.strides[0] + column * self.strides[1])
                })
            })
            .collect()
    }

    fn scalar(&self, index: usize) -> f64 {
        let at = self.offset + index * self.bytes;
        let data = &self.map[at..at + self.bytes];
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
