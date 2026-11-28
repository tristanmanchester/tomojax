//! A sinogram on a detector stored one unfiltered chunk per frame is a set of
//! short, independent file reads. HDF5 supplies the offsets of its allocated chunks.
use super::{Selection, sec2};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::thread;

pub(super) fn read<T: Copy + Send + From<u8> + From<u16> + From<f32>>(
    data: &hdf5::Dataset,
    s: &Selection,
) -> Option<Vec<T>> {
    let &[views, height, width] = s.shape.as_slice() else {
        return None;
    };
    let &[frames, 1, columns] = s.count.as_slice() else {
        return None;
    };
    if frames < 64 || columns < 2 || views > frames * 64 || views > 1_000_000 {
        return None;
    }
    let props = data.dcpl().ok()?;
    if props.chunk()?.as_slice() != [1, height, width] || !props.get_filters().ok()?.is_empty() {
        return None;
    }
    let file = data.file().ok()?;
    if file.create_plist().ok()?.get_userblock().ok()? != 0 {
        return None;
    }
    let dtype = data.dtype().ok()?;
    if !(dtype.is::<u16>() || dtype.is::<f32>() || dtype.is::<u8>()) {
        return None;
    }
    let fd = sec2::descriptor(&file)?;
    let element = dtype.size();
    let row_span = (columns - 1) * s.stride[2] + 1;
    if row_span.checked_mul(element)? > 256 << 10 || row_span > 32 * columns {
        return None;
    }
    let frame_size = height.checked_mul(width)?.checked_mul(element)? as u64;
    let within = ((s.start[1] * width + s.start[2]) * element) as u64;
    let mut offsets = vec![None; frames];
    data.chunks_visit(|c| {
        let z = c.offset[0] as usize;
        if let Some(delta) = z.checked_sub(s.start[0])
            && delta.is_multiple_of(s.stride[0])
            && let Some(offset) = offsets.get_mut(delta / s.stride[0])
            && c.filter_mask == 0
            && c.size == frame_size
        {
            *offset = c.addr.checked_add(within);
        }
        0
    })
    .ok()?;
    let offsets = offsets.into_iter().collect::<Option<Vec<_>>>()?;
    if dtype.is::<u16>() {
        gather(&fd, &offsets, columns, s.stride[2], |b| {
            u16::from_ne_bytes(b).into()
        })
    } else if dtype.is::<f32>() {
        gather(&fd, &offsets, columns, s.stride[2], |b| {
            f32::from_ne_bytes(b).into()
        })
    } else {
        gather(&fd, &offsets, columns, s.stride[2], |[b]| b.into())
    }
}

fn gather<T: Copy + Send + From<u8>, const N: usize>(
    file: &File,
    offsets: &[u64],
    columns: usize,
    step: usize,
    convert: fn([u8; N]) -> T,
) -> Option<Vec<T>> {
    let mut data = vec![0u8.into(); offsets.len() * columns];
    // Overlap NFS round trips while keeping enough reads per lane to amortize a thread.
    let per_thread = offsets.len().div_ceil(32).max(64);
    thread::scope(|scope| {
        let mut tasks = Vec::new();
        for (starts, output) in offsets
            .chunks(per_thread)
            .zip(data.chunks_mut(per_thread * columns))
        {
            tasks.push(
                thread::Builder::new()
                    .spawn_scoped(scope, move || {
                        let mut row = vec![0; ((columns - 1) * step + 1) * N];
                        for (&offset, samples) in
                            starts.iter().zip(output.chunks_exact_mut(columns))
                        {
                            file.read_exact_at(&mut row, offset).ok()?;
                            for (dest, bytes) in samples
                                .iter_mut()
                                .zip(row.as_chunks::<N>().0.iter().step_by(step))
                            {
                                *dest = convert(*bytes);
                            }
                        }
                        Some(())
                    })
                    .ok()?,
            );
        }
        for task in tasks {
            task.join().ok()??;
        }
        Some(())
    })?;
    Some(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn offset_rows_preserve_stride_external_links_and_unallocated_fill() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("det.h5")).unwrap();
        let values = ndarray::Array3::from_shape_fn((137, 23, 59), |(z, y, x)| {
            (z * 257 + y * 61 + x) as u16
        });
        file.new_dataset_builder()
            .with_data(&values)
            .chunk((1, 23, 59))
            .create("data")
            .unwrap();
        file.new_dataset::<u16>()
            .shape((137, 23, 59))
            .chunk((1, 23, 59))
            .fill_value(29u16)
            .create("fill")
            .unwrap();
        drop(file);
        let master = hdf5::File::create(dir.path().join("master.nxs")).unwrap();
        master.link_external("det.h5", "/data", "pictures").unwrap();
        master.link_external("det.h5", "/fill", "fill").unwrap();
        drop(master);
        let master = hdf5::File::open(dir.path().join("master.nxs")).unwrap();
        let selection = Selection {
            shape: vec![137, 23, 59],
            count: vec![68, 1, 17],
            stride: vec![2, 1, 3],
            start: vec![2, 13, 7],
        };
        let data = master.dataset("pictures").unwrap();
        let expected = data
            .read_slice::<f32, _, ndarray::Ix3>(selection.slab(&selection.stride, &selection.count))
            .unwrap();
        assert!(
            read::<f32>(&data, &selection)
                .unwrap()
                .into_iter()
                .eq(expected)
        );
        let fill = master.dataset("fill").unwrap();
        assert!(read::<f32>(&fill, &selection).is_none());
        assert!(
            selection
                .read::<f32>(&fill)
                .unwrap()
                .into_iter()
                .all(|v| v == 29.)
        );
    }
}
