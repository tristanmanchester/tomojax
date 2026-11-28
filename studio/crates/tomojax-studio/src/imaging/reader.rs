//! Bounded NPY/HDF5 selections and their physical registration. Called on the I/O thread.
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use super::{AxisScale, PlaneData, Statistics, VolumeData};
use serde_json::{Value, json};
use studio_core::engine::FileReads;
use studio_core::pixels::PlaneSelection;
use studio_core::pixels::{Encoding, Pixels};
use studio_core::protocol::ResultValue;
use studio_core::protocol::{ArrayDescriptor, Coordinates, Fixed, Storage};

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

fn pixels(files: &FileReads, command: Value) -> Result<Pixels, String> {
    match files.request(command)? {
        ResultValue::Pixels(frame) => Ok(frame),
        _ => Err("Expected a pixel terminal".into()),
    }
}
fn doubles(files: &FileReads, storage: &Storage, shape: &[usize]) -> Result<Vec<f64>, String> {
    pixels(
        files,
        json!({"type":"fetch","storage":storage,"shape":shape,"encoding":"float64"}),
    )?
    .doubles(shape)
}
mod coordinates;
pub(super) use coordinates::{CoordinateVectors, CoordinateWindow};

fn scale(
    files: &FileReads,
    descriptor: &ArrayDescriptor,
    dimension: usize,
    region: Option<[usize; 2]>,
    stride: usize,
    cached: &mut Option<CoordinateWindow>,
) -> Result<AxisScale, String> {
    let axis = &descriptor.axes[dimension];
    let count = descriptor.shape[dimension];
    let [start, end] = region.unwrap_or([0, count]);
    let (origin, spacing, positions) = match &axis.coordinates {
        Coordinates::Uniform { origin, spacing } => (
            *origin + *spacing * start as f64,
            *spacing * stride as f64,
            None,
        ),
        Coordinates::Coordinates { values } => {
            if count > 1_000_000 {
                return Err("The coordinate vector exceeds the registration budget.".into());
            }
            let positions =
                coordinates::selected(files, values, count, [start, end], stride, cached)?;
            if positions.iter().any(|v| !v.is_finite()) {
                return Err("The coordinate vector contains nonfinite positions.".into());
            }
            let origin = positions[0];
            let spacing = if positions.len() > 1 {
                positions[1] - origin
            } else {
                1.
            };
            (origin, spacing, Some(positions))
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
    files: &FileReads,
    descriptor: &ArrayDescriptor,
    axis: usize,
    index: usize,
    max_edge: usize,
) -> Result<PlaneData, String> {
    validate(descriptor)?;
    let slab = PlaneSelection::new(&descriptor.shape, axis, index, max_edge)?;
    let values = pixels(files, slab.fetch(descriptor, Encoding::Float32, None))?
        .decode(&slab, Encoding::Float32)?;
    registered_plane(
        files,
        descriptor,
        (axis, index),
        &slab,
        values,
        true,
        &mut Default::default(),
    )
}

pub(super) fn registered_plane(
    files: &FileReads,
    descriptor: &ArrayDescriptor,
    (axis, index): (usize, usize),
    slab: &PlaneSelection,
    values: Vec<f32>,
    lossless: bool,
    coordinates: &mut CoordinateVectors,
) -> Result<PlaneData, String> {
    let dimensions = descriptor.shape.len();
    let remaining = slab.remaining;
    let stride = slab.stride;
    let count = &slab.count;
    let source_shape = remaining.map(|a| descriptor.shape[a]);
    let mut fixed = descriptor.fixed.clone();
    if dimensions == 3 {
        let fixed_axis = scale(
            files,
            descriptor,
            axis,
            Some([index, index + 1]),
            1,
            &mut coordinates[axis],
        )?;
        fixed.push(Fixed {
            name: fixed_axis.name.clone(),
            frame: fixed_axis.frame.clone(),
            unit: fixed_axis.unit.clone(),
            index,
            coordinate: fixed_axis.origin,
        });
    }
    let statistics = statistics(&values);
    Ok(PlaneData {
        lossless,
        id: identity(),
        values: values.into(),
        shape: [count[remaining[0]], count[remaining[1]]],
        source_shape,
        region: slab.region,
        detail: None,
        axes: [
            scale(
                files,
                descriptor,
                remaining[0],
                Some(slab.region[0]),
                stride,
                &mut coordinates[remaining[0]],
            )?,
            scale(
                files,
                descriptor,
                remaining[1],
                Some(slab.region[1]),
                stride,
                &mut coordinates[remaining[1]],
            )?,
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

pub(crate) fn table(
    files: &FileReads,
    descriptor: &ArrayDescriptor,
) -> Result<super::TableData, String> {
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
    let values = doubles(files, &descriptor.storage, &descriptor.shape)?;
    Ok(super::TableData {
        values: values.into(),
        shape: [descriptor.shape[0], descriptor.shape[1]],
        axes: [
            scale(files, descriptor, 0, None, 1, &mut None)?,
            scale(files, descriptor, 1, None, 1, &mut None)?,
        ],
        columns: descriptor.columns.clone(),
        domain: descriptor.domain.clone(),
    })
}

pub(super) fn volume(
    files: &FileReads,
    descriptor: &ArrayDescriptor,
    max_edge: usize,
    encoding: Encoding,
    canceled: &dyn Fn() -> bool,
) -> Result<VolumeData, String> {
    validate(descriptor)?;
    let &[depth, height, width] = descriptor.shape.as_slice() else {
        return Err("Volume rendering needs a registered 3D array.".into());
    };
    let mut axis_order = [0; 3];
    for (i, name) in ["z", "y", "x"].into_iter().enumerate() {
        axis_order[i] = descriptor
            .axes
            .iter()
            .position(|a| a.name == name)
            .ok_or("Volume rendering needs explicit x, y and z spatial axes.")?;
    }
    let stride = depth.max(height).max(width).div_ceil(max_edge).max(1);
    let count: Vec<_> = descriptor
        .shape
        .iter()
        .map(|n| n.div_ceil(stride))
        .collect();
    let values = volume_samples(files, descriptor, &count, stride, encoding, canceled)?;
    let statistics = statistics(&values);
    if statistics.finite_count == 0 {
        return Err("The selected volume contains no finite scalar samples.".into());
    }
    let range = [statistics.min, statistics.max];
    let span = (range[1] - range[0]).max(f32::EPSILON);
    let shape = axis_order.map(|a| count[a]);
    let source_shape = axis_order.map(|a| descriptor.shape[a]);
    let scales: Vec<_> = axis_order
        .iter()
        .map(|&a| scale(files, descriptor, a, None, 1, &mut None))
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

fn volume_samples(
    files: &FileReads,
    descriptor: &ArrayDescriptor,
    count: &[usize],
    stride: usize,
    encoding: Encoding,
    canceled: &dyn Fn() -> bool,
) -> Result<Vec<f32>, String> {
    let mut values = Vec::with_capacity(count.iter().product());
    // A 256-cubed volume exceeds the per-selection limit. Split it across z.
    let sample_bytes = if encoding == Encoding::Float32 { 4 } else { 2 };
    let slab_depth = ((16 << 20) / (sample_bytes * count[1] * count[2])).max(1);
    for first in (0..count[0]).step_by(slab_depth) {
        if canceled() {
            return Err("Volume loading was superseded.".into());
        }
        let end = (first + slab_depth).min(count[0]);
        let roi = [
            [first * stride, ((end - 1) * stride + 1)],
            [0, descriptor.shape[1]],
            [0, descriptor.shape[2]],
        ];
        let command = json!({"type":"fetch","storage":descriptor.storage,"shape":descriptor.shape,
            "roi":roi,"stride":[stride,stride,stride],"encoding":encoding});
        let frame = pixels(files, command)?;
        let shape = [end - first, count[1], count[2]];
        values.extend(if encoding == Encoding::Float32 {
            frame.floats(&shape)?
        } else {
            frame.volume(&shape)?
        });
    }
    Ok(values)
}

pub(super) fn statistics(values: &[f32]) -> Statistics {
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
        return Statistics {
            min: f32::NAN,
            max: f32::NAN,
            mean: f64::NAN,
            std: f64::NAN,
            finite_count: 0,
            missing_count: values.len(),
            histogram: Vec::new(),
            window: [0., 1.],
        };
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
    Statistics {
        min,
        max,
        mean,
        std: (m2 / count as f64).sqrt(),
        finite_count: count,
        missing_count: values.len() - count,
        histogram,
        window,
    }
}
