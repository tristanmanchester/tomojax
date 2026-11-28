//! Adjacent planes within one filtered chunk require the same decompression.
//! Retain bounded strips of samples and release the HDF5 file after each request.
use super::Selection;
use anyhow::Result;
use ndarray::{Array3, Axis};
use std::collections::VecDeque;
use std::fs::File;
use std::os::fd::BorrowedFd;
use std::os::unix::fs::MetadataExt;
use std::sync::{Arc, Mutex};
use studio_core::sync::Locked;

#[derive(PartialEq, Eq, Clone)]
struct Version {
    device: u64,
    inode: u64,
    length: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}

impl Version {
    fn of(data: &hdf5::Dataset) -> Option<Self> {
        // Identify the open file, including the target of an external link. A
        // pathname can already refer to a replacement while this read is in flight.
        let file = data.file().ok()?;
        if !matches!(
            file.access_plist().ok()?.get_driver().ok()?,
            hdf5::plist::file_access::FileDriver::Sec2
        ) {
            return None;
        }
        let meta = hdf5::sync::sync(|| {
            let mut handle = std::ptr::null_mut();
            // The sec2 driver returns a pointer to its open POSIX file descriptor.
            // The HDF5 File above outlives the borrow and we own the duplicated fd.
            unsafe {
                if hdf5_sys::h5f::H5Fget_vfd_handle(
                    file.id(),
                    hdf5_sys::h5p::H5P_DEFAULT,
                    &mut handle,
                ) < 0
                    || handle.is_null()
                {
                    return None;
                }
                let fd = BorrowedFd::borrow_raw(*handle.cast::<std::os::fd::RawFd>());
                File::from(fd.try_clone_to_owned().ok()?).metadata().ok()
            }
        })?;
        Some(Self {
            device: meta.dev(),
            inode: meta.ino(),
            length: meta.len(),
            modified: (meta.mtime(), meta.mtime_nsec()),
            changed: (meta.ctime(), meta.ctime_nsec()),
        })
    }
}

#[derive(PartialEq, Eq)]
struct Key {
    file: Version,
    dataset: String,
    axis: usize,
    start: [usize; 3],
    count: [usize; 3],
    stride: [usize; 3],
}

type Entry<T> = (Key, Arc<Array3<T>>);

pub(super) struct Neighbors<T>(Mutex<VecDeque<Entry<T>>>);

impl<T> Default for Neighbors<T> {
    fn default() -> Self {
        Self(Mutex::default())
    }
}

impl<T: hdf5::H5Type + Send + Copy + From<u8> + From<u16> + From<f32>> Neighbors<T> {
    pub(super) fn read(&self, data: &hdf5::Dataset, selection: &Selection) -> Result<Vec<T>> {
        let Some(key) = Key::for_selection::<T>(data, selection) else {
            return selection.read(data);
        };
        let axis = key.axis;
        let index = selection.start[axis] - key.start[axis];
        let hit = self.0.locked().iter().rev().find_map(|(stored, block)| {
            if stored.file != key.file || stored.dataset != key.dataset || stored.axis != axis {
                return None;
            }
            stored.subgrid(selection).map(|grid| (block.clone(), grid))
        });
        if let Some((block, grid)) = hit {
            let mut view = block.view();
            for (dimension, slice) in grid.into_iter().enumerate() {
                view.slice_axis_inplace(Axis(dimension), slice);
            }
            return Ok(view.iter().copied().collect());
        }
        let block = {
            let mut region = selection.clone();
            region.start = key.start.to_vec();
            region.count = key.count.to_vec();
            region.stride[axis] = 1;
            #[cfg(feature = "static-hdf5")]
            let block = super::chunks::read(data, &region);
            #[cfg(not(feature = "static-hdf5"))]
            let block = None;
            let values = Arc::new(match block {
                Some(block) => block,
                None => data
                    .read_slice::<T, _, ndarray::Ix3>(region.slab(&region.stride, &region.count))?,
            });
            let mut entries = self.0.locked();
            // Three orthogonal panes, at most 16 MiB each, for each scalar precision.
            if entries.len() == 3 {
                entries.pop_front();
            }
            entries.push_back((key, values.clone()));
            values
        };
        Ok(block
            .index_axis(Axis(axis), index)
            .iter()
            .copied()
            .collect())
    }
}

impl Key {
    // A finer strip can supply a cropped or coarser pane when every requested
    // sample is present on the stored grid, including the fixed plane index.
    fn subgrid(&self, selection: &Selection) -> Option<[ndarray::Slice; 3]> {
        let mut slices = [ndarray::Slice::from(..); 3];
        for (a, slice) in slices.iter_mut().enumerate() {
            let delta = selection.start[a].checked_sub(self.start[a])?;
            let count = selection.count[a];
            let stride = if count == 1 {
                self.stride[a]
            } else {
                selection.stride[a]
            };
            if !delta.is_multiple_of(self.stride[a]) || !stride.is_multiple_of(self.stride[a]) {
                return None;
            }
            let first = delta / self.stride[a];
            let step = stride / self.stride[a];
            let last = first + (count - 1) * step;
            if last >= self.count[a] {
                return None;
            }
            *slice = ndarray::Slice::new(first as isize, Some((last + 1) as isize), step as isize);
        }
        Some(slices)
    }

    fn for_selection<T>(data: &hdf5::Dataset, selection: &Selection) -> Option<Self> {
        let mut count: [usize; 3] = selection.count.as_slice().try_into().ok()?;
        let axis = count.iter().position(|n| *n == 1)?;
        let chunk = data.chunk()?[axis];
        let width = chunk.min((16 << 20) / (count.iter().product::<usize>() * size_of::<T>()));
        if width < 2 || data.filters().is_empty() {
            return None;
        }
        let mut start: [usize; 3] = selection.start.as_slice().try_into().ok()?;
        let mut stride: [usize; 3] = selection.stride.as_slice().try_into().ok()?;
        let chunk_start = start[axis] / chunk * chunk;
        let chunk_end = (chunk_start + chunk).min(selection.shape[axis]);
        start[axis] = chunk_start + (start[axis] - chunk_start) / width * width;
        count[axis] = width.min(chunk_end - start[axis]);
        stride[axis] = 1;
        Some(Self {
            file: Version::of(data)?,
            dataset: data.name(),
            axis,
            start,
            count,
            stride,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearby_filtered_planes_match_the_requested_grid_after_a_replacement() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("volume.h5");
        let write = |path: &std::path::Path, bias: f32| {
            let file = hdf5::File::create(path).unwrap();
            let values = Array3::from_shape_fn((7, 11, 13), |(z, y, x)| {
                bias + (1000 * z + 20 * y + x) as f32
            });
            file.new_dataset_builder()
                .with_data(&values)
                .chunk((4, 5, 5))
                .deflate(1)
                .create("v")
                .unwrap();
        };
        write(&path, 0.);
        let cache = Neighbors::<f32>::default();
        let selection = |x| Selection {
            shape: vec![7, 11, 13],
            start: vec![1, 2, x],
            stride: vec![2, 3, 1],
            count: vec![3, 3, 1],
        };
        let a = hdf5::File::open(&path).unwrap().dataset("v").unwrap();
        for x in [10, 12, 11] {
            let s = selection(x);
            assert_eq!(cache.read(&a, &s).unwrap(), s.read::<f32>(&a).unwrap());
        }
        let mut crop = selection(12);
        crop.start[0] = 3;
        crop.count = vec![2, 2, 1];
        crop.stride = vec![2, 6, 1];
        assert_eq!(
            cache.read(&a, &crop).unwrap(),
            crop.read::<f32>(&a).unwrap()
        );
        assert_eq!(cache.0.locked().len(), 1);
        let replacement = tmp.path().join("replacement.h5");
        write(&replacement, 50000.);
        std::fs::rename(replacement, &path).unwrap();
        let b = hdf5::File::open(&path).unwrap().dataset("v").unwrap();
        let s = selection(11);
        assert_eq!(cache.read(&b, &s).unwrap(), s.read::<f32>(&b).unwrap());
        // An in-flight selection of the old inode still addresses its own samples.
        assert_ne!(cache.read(&a, &s).unwrap(), cache.read(&b, &s).unwrap());
    }
    #[test]
    fn external_links_keep_the_versions_of_their_open_target_files() {
        let tmp = tempfile::tempdir().unwrap();
        let master = hdf5::File::create(tmp.path().join("master.h5")).unwrap();
        for (name, bias) in [("left", 3u16), ("right", 29u16)] {
            let filename = format!("{name}.h5");
            let file = hdf5::File::create(tmp.path().join(&filename)).unwrap();
            file.new_dataset_builder()
                .with_data(&Array3::from_elem((4, 5, 6), bias))
                .chunk((2, 3, 3))
                .deflate(1)
                .create("data")
                .unwrap();
            master.link_external(&filename, "/data", name).unwrap();
        }
        drop(master);
        let master = hdf5::File::open(tmp.path().join("master.h5")).unwrap();
        let cache = Neighbors::<f32>::default();
        let selection = Selection {
            shape: vec![4, 5, 6],
            start: vec![0, 1, 0],
            stride: vec![1; 3],
            count: vec![4, 1, 6],
        };
        for (name, value) in [("left", 3.), ("right", 29.), ("left", 3.)] {
            let data = master.dataset(name).unwrap();
            assert_eq!(cache.read(&data, &selection).unwrap(), vec![value; 24]);
        }
        assert_eq!(cache.0.locked().len(), 2);
    }
}
