//! Bounded NPY/HDF5 selections and their physical registration. Called on the I/O thread.
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use super::{AxisScale, PlaneData, Statistics, VolumeData, npy::Npy};
use crate::protocol::{ArrayDescriptor, Axis, Coordinates, Fixed, Storage};
use hdf5::{Hyperslab, SliceOrIndex};

pub(super) fn identity() -> u64 {
    static ID: AtomicU64 = AtomicU64::new(1);
    ID.fetch_add(1, Ordering::Relaxed)
}

pub(super) fn validate(descriptor: &ArrayDescriptor) -> Result<(), String> {
    if !(2..=3).contains(&descriptor.shape.len())
        || descriptor.shape.contains(&0)
        || descriptor.shape.len() != descriptor.axes.len()
    {
        return Err("A registered image must have two or three nonempty axes.".into());
    }
    if matches!(descriptor.storage, Storage::Member { .. }) {
        return Err("This result still belongs to the numerical worker. Save it into the project before independent viewing.".into());
    }
    for axis in &descriptor.axes {
        if let Coordinates::Uniform { origin, spacing } = axis.coordinates
            && (!origin.is_finite() || !spacing.is_finite() || spacing == 0.)
        {
            return Err(format!(
                "The {} axis has invalid physical registration.",
                axis.name
            ));
        }
    }
    Ok(())
}

fn selection(
    storage: &Storage,
    shape: &[usize],
    start: &[usize],
    step: &[usize],
    count: &[usize],
) -> Result<Vec<f32>, String> {
    let elements = count
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v))
        .ok_or("Selection is too large.")?;
    if elements > 256 * 256 * 256 {
        return Err("This image selection exceeds the 64 MiB scalar budget.".into());
    }
    match storage {
        Storage::Npy { path, .. } => {
            let source = Npy::open(path)?;
            if source.shape() != shape {
                return Err("The NumPy shape differs from its registered descriptor.".into());
            }
            Ok(source.read(start, step, count))
        }
        Storage::Hdf5 { path, dataset } => {
            let file = hdf5::File::open(path).map_err(|e| format!("Cannot open {path}: {e}"))?;
            let source = file
                .dataset(dataset)
                .map_err(|e| format!("Cannot read dataset {dataset}: {e}"))?;
            // HDF5 fills a virtual dataset's missing sources with its fill value,
            // which is not a measurement; TomoJAX reads (and checks) those.
            if crate::metadata::is_virtual(&source) {
                return Err(format!(
                    "{dataset} is a virtual dataset; open the scan to view it through TomoJAX"
                ));
            }
            let stored_shape = source.shape();
            // TomoJAX registers a line detector as (view,1,u), while its HDF5 source can be (view,u).
            let singleton_v =
                shape.len() == 3 && shape[1] == 1 && stored_shape == [shape[0], shape[2]];
            if stored_shape != shape && !singleton_v {
                return Err("The HDF5 shape differs from its registered descriptor.".into());
            }
            let slices: Vec<_> = (0..shape.len())
                .filter(|&axis| !(singleton_v && axis == 1))
                .map(|axis| SliceOrIndex::SliceCount {
                    start: start[axis],
                    step: step[axis],
                    count: count[axis],
                    block: 1,
                })
                .collect();
            let pixels = source
                .read_slice::<f32, _, ndarray::IxDyn>(Hyperslab::from(slices))
                .map_err(|e| format!("Cannot read the requested scalar hyperslab: {e}"))?;
            Ok(pixels.into_raw_vec_and_offset().0)
        }
        Storage::Member { .. } => Err("The worker member has no independent storage yet.".into()),
    }
}

fn coordinates(storage: &Storage, count: usize) -> Result<Vec<f64>, String> {
    if count > 1_000_000 {
        return Err("The coordinate vector exceeds the registration budget.".into());
    }
    let values = match storage {
        Storage::Npy { path, .. } => {
            let source = Npy::open(path)?;
            if source.shape() != [count] {
                return Err("The coordinate vector has the wrong length.".into());
            }
            source.coordinates()
        }
        Storage::Hdf5 { path, dataset } => {
            let file = hdf5::File::open(path).map_err(|e| e.to_string())?;
            let source = file.dataset(dataset).map_err(|e| e.to_string())?;
            if source.shape() != [count] {
                return Err("The coordinate vector has the wrong length.".into());
            }
            source.read_raw::<f64>().map_err(|e| e.to_string())?
        }
        Storage::Member { .. } => {
            return Err("The coordinate vector has not been saved into the project.".into());
        }
    };
    if values.iter().any(|v| !v.is_finite()) {
        return Err("The coordinate vector contains nonfinite positions.".into());
    }
    Ok(values)
}

fn scale(axis: &Axis, count: usize, stride: usize) -> Result<AxisScale, String> {
    let (origin, spacing, positions) = match &axis.coordinates {
        Coordinates::Uniform { origin, spacing } => (*origin, *spacing * stride as f64, None),
        Coordinates::Coordinates { values } => {
            let positions: Vec<_> = coordinates(values, count)?
                .into_iter()
                .step_by(stride)
                .collect();
            let origin = positions[0];
            let spacing = if positions.len() > 1 {
                positions[1] - origin
            } else {
                1.
            };
            (origin, spacing, Some(Arc::from(positions)))
        }
    };
    Ok(AxisScale {
        name: axis.name.clone(),
        frame: axis.frame.clone(),
        unit: axis.unit.clone(),
        origin,
        spacing,
        coordinates: positions,
    })
}

pub(crate) fn plane(
    descriptor: &ArrayDescriptor,
    axis: usize,
    index: usize,
    max_edge: usize,
) -> Result<PlaneData, String> {
    validate(descriptor)?;
    let dimensions = descriptor.shape.len();
    if axis >= dimensions || (dimensions == 3 && index >= descriptor.shape[axis]) {
        return Err("The requested slice lies outside the array.".into());
    }
    let remaining: Vec<_> = (0..dimensions)
        .filter(|&a| dimensions == 2 || a != axis)
        .collect();
    let source_shape = [
        descriptor.shape[remaining[0]],
        descriptor.shape[remaining[1]],
    ];
    let stride = source_shape
        .into_iter()
        .max()
        .unwrap()
        .div_ceil(max_edge)
        .max(1);
    let mut start = vec![0; dimensions];
    let mut step = vec![stride; dimensions];
    let mut count: Vec<_> = descriptor
        .shape
        .iter()
        .map(|n| n.div_ceil(stride))
        .collect();
    let mut fixed = descriptor.fixed.clone();
    if dimensions == 3 {
        start[axis] = index;
        step[axis] = 1;
        count[axis] = 1;
        let fixed_axis = scale(&descriptor.axes[axis], descriptor.shape[axis], 1)?;
        fixed.push(Fixed {
            name: fixed_axis.name.clone(),
            frame: fixed_axis.frame.clone(),
            unit: fixed_axis.unit.clone(),
            index,
            coordinate: fixed_axis.coordinate(index as f64),
        });
    }
    let values = selection(
        &descriptor.storage,
        &descriptor.shape,
        &start,
        &step,
        &count,
    )?;
    let statistics = statistics(&values)?;
    Ok(PlaneData {
        id: identity(),
        values: values.into(),
        shape: [count[remaining[0]], count[remaining[1]]],
        source_shape,
        axes: [
            scale(&descriptor.axes[remaining[0]], source_shape[0], stride)?,
            scale(&descriptor.axes[remaining[1]], source_shape[1], stride)?,
        ],
        fixed,
        index: if dimensions == 3 { index } else { 0 },
        count: if dimensions == 3 {
            descriptor.shape[axis]
        } else {
            1
        },
        value_unit: descriptor.unit.clone(),
        domain: descriptor.domain.clone(),
        stride,
        statistics,
    })
}

pub(crate) fn table(descriptor: &ArrayDescriptor) -> Result<super::TableData, String> {
    validate(descriptor)?;
    if descriptor.shape.len() != 2 || descriptor.columns.len() != descriptor.shape[1] {
        return Err("A registered table needs one column definition per stored column.".into());
    }
    if descriptor.shape[0]
        .checked_mul(descriptor.shape[1])
        .is_none_or(|n| n > 1_000_000)
    {
        return Err("The live pose table exceeds the one-million-value display budget.".into());
    }
    let values = match &descriptor.storage {
        Storage::Npy { path, .. } => {
            let source = Npy::open(path)?;
            if source.shape() != descriptor.shape {
                return Err("The pose table shape differs from its registered descriptor.".into());
            }
            source.table()
        }
        Storage::Hdf5 { path, dataset } => {
            let file = hdf5::File::open(path).map_err(|e| e.to_string())?;
            let source = file.dataset(dataset).map_err(|e| e.to_string())?;
            if source.shape() != descriptor.shape {
                return Err("The pose table shape differs from its registered descriptor.".into());
            }
            source.read_raw::<f64>().map_err(|e| e.to_string())?
        }
        Storage::Member { .. } => {
            return Err("Live poses require independently readable storage.".into());
        }
    };
    Ok(super::TableData {
        values: values.into(),
        shape: [descriptor.shape[0], descriptor.shape[1]],
        axes: [
            scale(&descriptor.axes[0], descriptor.shape[0], 1)?,
            scale(&descriptor.axes[1], descriptor.shape[1], 1)?,
        ],
        columns: descriptor.columns.clone(),
        domain: descriptor.domain.clone(),
    })
}

pub(super) fn volume(
    descriptor: &ArrayDescriptor,
    max_edge: usize,
    cancelled: &dyn Fn() -> bool,
) -> Result<VolumeData, String> {
    validate(descriptor)?;
    if descriptor.shape.len() != 3 {
        return Err("Volume rendering needs a registered 3D array.".into());
    }
    let mut axis_order = [0; 3];
    for (i, name) in ["z", "y", "x"].into_iter().enumerate() {
        axis_order[i] = descriptor
            .axes
            .iter()
            .position(|a| a.name == name)
            .ok_or("Volume rendering needs explicit x, y and z spatial axes.")?;
    }
    let stride = descriptor
        .shape
        .iter()
        .max()
        .unwrap()
        .div_ceil(max_edge)
        .max(1);
    let count: Vec<_> = descriptor
        .shape
        .iter()
        .map(|n| n.div_ceil(stride))
        .collect();
    // Small slabs release HDF5's lock between reads so a second reader can serve navigation.
    let mut values = Vec::with_capacity(count.iter().product());
    for first in (0..count[0]).step_by(4) {
        if cancelled() {
            return Err("Volume loading was superseded.".into());
        }
        let mut slab = count.clone();
        slab[0] = (count[0] - first).min(4);
        values.extend(selection(
            &descriptor.storage,
            &descriptor.shape,
            &[first * stride, 0, 0],
            &[stride; 3],
            &slab,
        )?);
        std::thread::yield_now();
    }
    let statistics = statistics(&values)?;
    let range = [statistics.min, statistics.max];
    let span = (range[1] - range[0]).max(f32::EPSILON);
    let shape = axis_order.map(|a| count[a]);
    let source_shape = axis_order.map(|a| descriptor.shape[a]);
    let scales: Vec<_> = axis_order
        .iter()
        .map(|&a| scale(&descriptor.axes[a], descriptor.shape[a], 1))
        .collect::<Result<_, _>>()?;
    if scales
        .iter()
        .any(|a| a.coordinates.is_some() || a.frame != scales[0].frame || a.unit != scales[0].unit)
    {
        return Err(
            "Volume rendering needs a uniform spatial grid in one coordinate frame and unit."
                .into(),
        );
    }
    let mut texels = Vec::with_capacity(values.len());
    for z in 0..shape[0] {
        for y in 0..shape[1] {
            for x in 0..shape[2] {
                let mut source = [0; 3];
                for (i, position) in [z, y, x].into_iter().enumerate() {
                    source[axis_order[i]] = if scales[i].spacing < 0. {
                        shape[i] - 1 - position
                    } else {
                        position
                    };
                }
                let value = values[(source[0] * count[1] + source[1]) * count[2] + source[2]];
                texels.push(
                    half::f16::from_f32(if value.is_finite() {
                        (value - range[0]) / span
                    } else {
                        f32::NAN
                    })
                    .to_bits(),
                );
            }
        }
    }
    let origin_zyx: [f64; 3] = std::array::from_fn(|i| {
        scales[i]
            .origin
            .min(scales[i].coordinate((source_shape[i] - 1) as f64))
    });
    let spacing_zyx: [f64; 3] = std::array::from_fn(|i| scales[i].spacing.abs());
    let extent_zyx: [f32; 3] =
        std::array::from_fn(|i| (spacing_zyx[i] * source_shape[i] as f64) as f32);
    Ok(VolumeData {
        id: identity(),
        texels: texels.into(),
        shape,
        source_shape,
        extent: [extent_zyx[2], extent_zyx[1], extent_zyx[0]],
        origin: [origin_zyx[2], origin_zyx[1], origin_zyx[0]],
        spacing: [spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]],
        scalar_range: range,
        statistics,
        value_unit: descriptor.unit.clone(),
        length_unit: scales[0].unit.clone(),
        stride,
    })
}

pub(super) fn statistics(values: &[f32]) -> Result<Statistics, String> {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut mean = 0.;
    let mut m2 = 0.;
    let mut count = 0;
    for &value in values.iter().filter(|v| v.is_finite()) {
        min = min.min(value);
        max = max.max(value);
        count += 1;
        let delta = f64::from(value) - mean;
        mean += delta / count as f64;
        m2 += delta * (f64::from(value) - mean);
    }
    if count == 0 {
        return Err("The selected image contains no finite scalar samples.".into());
    }
    let mut histogram = vec![0u64; 256];
    let span = (max - min).max(f32::EPSILON);
    for &value in values.iter().filter(|v| v.is_finite()) {
        // Equal-width bins match the scalar interval shown by the UI's histogram axis.
        let bin = (((value - min) / span * 256.).floor() as usize).min(255);
        histogram[bin] += 1;
    }
    let percentile = |fraction: f64| {
        let mut seen = 0;
        let target = (count as f64 * fraction).ceil() as u64;
        for (bin, n) in histogram.iter().enumerate() {
            seen += n;
            if seen >= target {
                return (min + span * (bin as f32 + 0.5) / 256.).clamp(min, max);
            }
        }
        max
    };
    let low = percentile(0.005);
    let high = percentile(0.995);
    let window = [
        low,
        if high > low {
            high
        } else {
            low + low.abs().max(1.) * 0.01
        },
    ];
    Ok(Statistics {
        min,
        max,
        mean,
        std: (m2 / count as f64).sqrt(),
        finite_count: count,
        missing_count: values.len() - count,
        histogram,
        window,
    })
}

#[cfg(test)]
mod virtual_tests {
    use super::*;

    #[test]
    fn a_virtual_dataset_is_read_through_tomojax_not_directly() {
        let directory = tempfile::tempdir().unwrap();
        let pixels = directory.path().join("pixels.h5");
        hdf5::File::create(&pixels)
            .unwrap()
            .new_dataset_builder()
            .with_data(&[1.0f32, 2., 3., 4.])
            .create("data")
            .unwrap();
        let master = directory.path().join("scan.nxs");
        hdf5::File::create(&master)
            .unwrap()
            .new_dataset::<f32>()
            .shape((1, 1, 4))
            .with_dcpl(|plist| plist.virtual_map("pixels.h5", "/data", 4, .., (1, 1, 4), ..))
            .create("data")
            .unwrap();
        let storage = Storage::Hdf5 {
            path: master.to_string_lossy().into(),
            dataset: "/data".into(),
        };
        let error =
            selection(&storage, &[1, 1, 4], &[0, 0, 0], &[1, 1, 1], &[1, 1, 4]).unwrap_err();
        assert!(error.contains("virtual"), "{error}");
    }
}
