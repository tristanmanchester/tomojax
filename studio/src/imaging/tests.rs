use std::fs::File;
use std::io::BufWriter;
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use super::{Imaging, ImagingStatus, reader};
use crate::protocol::{ArrayDescriptor, Axis, Coordinates, Storage};
use npyz::WriterBuilder;

fn descriptor(storage: Storage, shape: [usize; 3]) -> ArrayDescriptor {
    ArrayDescriptor {
        storage,
        shape: shape.to_vec(),
        dtype: "<f4".into(),
        axes: ["z", "y", "x"]
            .into_iter()
            .zip([11., 4., -2.])
            .zip([3., 2., 0.5])
            .map(|((name, origin), spacing)| Axis {
                name: name.into(),
                frame: "world".into(),
                unit: "mm".into(),
                coordinates: Coordinates::Uniform { origin, spacing },
            })
            .collect(),
        unit: "mm^-1".into(),
        domain: "attenuation".into(),
        fixed: vec![],
        columns: vec![],
    }
}

fn ramp(shape: [usize; 3]) -> Vec<f32> {
    (0..shape[0])
        .flat_map(|z| {
            (0..shape[1])
                .flat_map(move |y| (0..shape[2]).map(move |x| (100 * z + 10 * y + x) as f32))
        })
        .collect()
}

fn npy(path: &std::path::Path, shape: [usize; 3], fortran: bool) {
    let order = if fortran {
        npyz::Order::Fortran
    } else {
        npyz::Order::C
    };
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&shape.map(|v| v as u64))
        .order(order)
        .writer(BufWriter::new(File::create(path).unwrap()))
        .begin_nd()
        .unwrap();
    if fortran {
        for x in 0..shape[2] {
            for y in 0..shape[1] {
                for z in 0..shape[0] {
                    writer.push(&((100 * z + 10 * y + x) as f32)).unwrap();
                }
            }
        }
    } else {
        for value in ramp(shape) {
            writer.push(&value).unwrap();
        }
    }
    writer.finish().unwrap();
}

#[test]
fn retired_artifacts_wait_for_active_reads_and_coordinate_prefetch_to_drain() {
    use std::collections::BTreeSet;
    use std::sync::Mutex;
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("volume.npy");
    let coordinates = directory.path().join("x.npy");
    npy(&path, [4, 5, 8], false);
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&[8])
        .writer(File::create(&coordinates).unwrap())
        .begin_nd()
        .unwrap();
    for i in 0..8 {
        writer.push(&(i as f64 * 0.5)).unwrap();
    }
    writer.finish().unwrap();
    let mut descriptor = descriptor(
        Storage::Npy {
            path: path.to_string_lossy().into(),
            artifact: Some("volume-token".into()),
        },
        [4, 5, 8],
    );
    descriptor.axes[2].coordinates = Coordinates::Coordinates {
        values: Storage::Npy {
            path: coordinates.to_string_lossy().into(),
            artifact: Some("coordinate-token".into()),
        },
    };
    let (entered, started) = mpsc::sync_channel(1);
    let (resume, blocked) = mpsc::sync_channel(1);
    let blocked = Mutex::new(Some(blocked));
    let mut imaging = Imaging::with_budget(8 * 1024 * 1024);
    imaging.set_waker(Arc::new(move || {
        if let Some(receiver) = blocked.lock().unwrap().take() {
            entered.send(()).unwrap();
            receiver.recv().unwrap();
        }
    }));
    imaging
        .request_plane(1, descriptor.clone(), 0, 1, 64)
        .unwrap();
    started.recv_timeout(Duration::from_secs(3)).unwrap();
    let scalar = BTreeSet::from(["volume-token".into()]);
    let coordinate = BTreeSet::from(["coordinate-token".into()]);
    let scalar_retained = imaging.artifacts_in_use(&scalar);
    let coordinate_retained = imaging.artifacts_in_use(&coordinate);
    imaging.clear(1);
    let clear_retained = imaging.artifacts_in_use(&coordinate);
    resume.send(()).unwrap();
    assert!(
        scalar_retained && coordinate_retained && clear_retained,
        "clearing a view must not release an active reader's scalar or coordinate files"
    );
    let until = Instant::now() + Duration::from_secs(3);
    while imaging.artifacts_in_use(&scalar) || imaging.artifacts_in_use(&coordinate) {
        imaging.poll();
        assert!(Instant::now() < until, "reader lease did not drain");
        std::thread::yield_now();
    }
    // A cached scalar view owns its bytes and no longer depends on the files.
    imaging.request_plane(1, descriptor, 0, 1, 64).unwrap();
    assert_eq!(imaging.status(1), &ImagingStatus::Ready);
    assert!(!imaging.artifacts_in_use(&scalar));
    assert!(!imaging.artifacts_in_use(&coordinate));
}

#[test]
fn anisotropic_planes_use_voxel_centers_and_keep_each_disk_axis() {
    let directory = tempfile::tempdir().unwrap();
    for fortran in [false, true] {
        let path = directory.path().join(format!("ramp-{fortran}.npy"));
        npy(&path, [3, 4, 5], fortran);
        let desc = descriptor(
            Storage::Npy {
                path: path.to_string_lossy().into_owned(),
                artifact: None,
            },
            [3, 4, 5],
        );
        let axial = reader::plane(&desc, 0, 1, 64).unwrap();
        assert_eq!(axial.shape, [4, 5]);
        assert_eq!(axial.sample(2, 3), Some(123.));
        assert_eq!(axial.fixed.last().unwrap().coordinate, 14.);
        assert!((axial.aspect() - 0.3125).abs() < 1e-6);
        assert!((axial.distance([0., 0.], [3., 4.]).unwrap() - 40f64.sqrt()).abs() < 1e-9);
        let sagittal = reader::plane(&desc, 2, 3, 64).unwrap();
        assert_eq!(sagittal.shape, [3, 4]);
        assert_eq!(sagittal.sample(2, 1), Some(213.));
        assert_eq!(sagittal.axes[0].name, "z");
        assert_eq!(sagittal.axes[1].name, "y");
        assert_eq!(sagittal.fixed.last().unwrap().coordinate, -0.5);
    }
}

#[test]
fn hdf5_hyperslabs_preserve_anisotropy_and_line_detector_expansion() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("scan.h5");
    {
        let file = hdf5::File::create(&path).unwrap();
        file.new_dataset::<f32>()
            .shape([3, 4, 5])
            .chunk([1, 4, 5])
            .lzf()
            .create("volume")
            .unwrap()
            .write_raw(&ramp([3, 4, 5]))
            .unwrap();
        file.new_dataset::<f32>()
            .shape([4, 5])
            .create("lines")
            .unwrap()
            .write_raw(&(0..20).map(|n| n as f32).collect::<Vec<_>>())
            .unwrap();
    }
    let desc = descriptor(
        Storage::Hdf5 {
            path: path.to_string_lossy().into_owned(),
            dataset: "volume".into(),
        },
        [3, 4, 5],
    );
    let coronal = reader::plane(&desc, 1, 2, 64).unwrap();
    assert_eq!(coronal.shape, [3, 5]);
    assert_eq!(coronal.sample(2, 4), Some(224.));
    let line = descriptor(
        Storage::Hdf5 {
            path: path.to_string_lossy().into_owned(),
            dataset: "lines".into(),
        },
        [4, 1, 5],
    );
    let projection = reader::plane(&line, 0, 2, 64).unwrap();
    assert_eq!(projection.shape, [1, 5]);
    assert_eq!(&*projection.values, &[10., 11., 12., 13., 14.]);
    let sinogram = reader::plane(&line, 1, 0, 64).unwrap();
    assert_eq!(sinogram.shape, [4, 5]);
    assert_eq!(sinogram.sample(3, 4), Some(19.));
}

#[test]
fn signed_spacings_orient_volumes_without_changing_quantitative_slice_order() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reversed.npy");
    npy(&path, [3, 4, 5], false);
    let mut desc = descriptor(
        Storage::Npy {
            path: path.to_string_lossy().into_owned(),
            artifact: None,
        },
        [3, 4, 5],
    );
    desc.axes[2].coordinates = Coordinates::Uniform {
        origin: -2.,
        spacing: -0.5,
    };
    let plane = reader::plane(&desc, 0, 0, 64).unwrap();
    assert_eq!(plane.sample(0, 0), Some(0.));
    assert_eq!(plane.axes[1].coordinate(4.), -4.);
    let volume = reader::volume(&desc, 64, &|| false).unwrap();
    assert_eq!(volume.extent, [2.5, 8., 9.]);
    assert_eq!(volume.origin, [-4., 4., 11.]);
    let first = half::f16::from_bits(volume.texels[0]).to_f32() * 234.;
    assert!(
        (first - 4.).abs() < 0.01,
        "A physical +x texture starts at the stored last x index"
    );
}

#[test]
fn irregular_view_coordinates_keep_acquisition_order_and_do_not_become_distances() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("scan.npy");
    npy(&path, [4, 3, 5], false);
    let positions = directory.path().join("angles.npy");
    npyz::to_file_1d(&positions, [0f64, 17., 4., 90.]).unwrap();
    let mut desc = descriptor(
        Storage::Npy {
            path: path.to_string_lossy().into_owned(),
            artifact: None,
        },
        [4, 3, 5],
    );
    desc.axes[0] = Axis {
        name: "view".into(),
        frame: "acquisition".into(),
        unit: "degree".into(),
        coordinates: Coordinates::Coordinates {
            values: Storage::Npy {
                path: positions.to_string_lossy().into_owned(),
                artifact: None,
            },
        },
    };
    let sinogram = reader::plane(&desc, 1, 1, 64).unwrap();
    assert_eq!(
        (0..4)
            .map(|i| sinogram.axes[0].coordinate(i as f64))
            .collect::<Vec<_>>(),
        [0., 17., 4., 90.]
    );
    assert!(sinogram.distance([0., 0.], [2., 4.]).is_err());
}

#[test]
fn finite_statistics_and_histogram_bins_match_the_reported_scalar_range() {
    let stats = reader::statistics(&[0., 0.25, 0.5, 0.75, 1., f32::NAN]).unwrap();
    assert_eq!(stats.finite_count, 5);
    assert_eq!(stats.missing_count, 1);
    assert!((stats.mean - 0.5).abs() < 1e-12);
    assert!((stats.std - 0.125f64.sqrt()).abs() < 1e-12);
    for bin in [0, 64, 128, 192, 255] {
        assert_eq!(stats.histogram[bin], 1);
    }
    assert_eq!(stats.histogram.iter().sum::<u64>(), 5);
}

#[test]
fn image_requests_coalesce_and_cache_hits_keep_the_scalar_data_identity() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("navigation.npy");
    npy(&path, [12, 128, 128], false);
    let desc = descriptor(
        Storage::Npy {
            path: path.to_string_lossy().into_owned(),
            artifact: None,
        },
        [12, 128, 128],
    );
    let (send, wake) = mpsc::channel();
    let mut images = Imaging::with_budget(8 * 1024 * 1024);
    images.set_waker(Arc::new(move || {
        let _ = send.send(());
    }));
    for index in 0..12 {
        images
            .request_plane(1, desc.clone(), 0, index, 256)
            .unwrap();
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        images.poll();
        if matches!(images.status(1), ImagingStatus::Ready) {
            break;
        }
        if let ImagingStatus::Error(error) = images.status(1) {
            panic!("{error}");
        }
        wake.recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .unwrap();
    }
    let loaded = images.plane(1).unwrap();
    assert_eq!(loaded.index, 11);
    assert_eq!(loaded.sample(2, 3), Some(1123.));
    let id = loaded.id;
    images.request_plane(2, desc, 0, 11, 256).unwrap();
    assert!(matches!(images.status(2), ImagingStatus::Ready));
    assert_eq!(images.plane(2).unwrap().id, id);
    assert!(images.timings().back().unwrap().cache_hit);
    assert!(images.cache_bytes() <= 8 * 1024 * 1024);
    images.shutdown();
}

#[test]
fn scalar_lru_evicts_over_budget_without_invalidating_a_visible_plane() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("large-navigation.npy");
    npy(&path, [12, 512, 512], false);
    let desc = descriptor(
        Storage::Npy {
            path: path.to_string_lossy().into_owned(),
            artifact: None,
        },
        [12, 512, 512],
    );
    let (send, wake) = mpsc::channel();
    let budget = 8 * 1024 * 1024;
    let mut images = Imaging::with_budget(budget);
    images.set_waker(Arc::new(move || {
        let _ = send.send(());
    }));
    let wait = |images: &mut Imaging, slot| {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            images.poll();
            match images.status(slot) {
                ImagingStatus::Ready => break,
                ImagingStatus::Error(error) => panic!("{error}"),
                _ => {}
            }
            wake.recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .unwrap();
        }
    };
    images.request_plane(1, desc.clone(), 0, 0, 512).unwrap();
    wait(&mut images, 1);
    let visible = images.plane(1).unwrap().clone();
    for index in 1..12 {
        images
            .request_plane(2, desc.clone(), 0, index, 512)
            .unwrap();
        wait(&mut images, 2);
        assert!(images.cache_bytes() <= budget);
    }
    images.request_plane(2, desc, 0, 0, 512).unwrap();
    assert!(matches!(images.status(2), ImagingStatus::Loading));
    wait(&mut images, 2);
    assert_ne!(images.plane(2).unwrap().id, visible.id);
    assert_eq!(visible.sample(2, 3), Some(23.));
    assert_eq!(images.plane(1).unwrap().id, visible.id);
    assert!(images.cache_bytes() <= budget);
    images.shutdown();
}

#[test]
fn explicit_source_refresh_reloads_a_replaced_file_at_the_same_path() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.npy");
    npy(&path, [3, 4, 5], false);
    let desc = descriptor(
        Storage::Npy {
            path: path.to_string_lossy().into_owned(),
            artifact: None,
        },
        [3, 4, 5],
    );
    let (send, wake) = mpsc::channel();
    let mut images = Imaging::default();
    images.set_waker(Arc::new(move || {
        let _ = send.send(());
    }));
    let wait = |images: &mut Imaging| {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            images.poll();
            match images.status(1) {
                ImagingStatus::Ready => break,
                ImagingStatus::Error(error) => panic!("{error}"),
                _ => {}
            }
            wake.recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .unwrap();
        }
    };
    images.request_plane(1, desc.clone(), 0, 1, 64).unwrap();
    wait(&mut images);
    assert_eq!(images.plane(1).unwrap().sample(2, 3), Some(123.));

    let replacement = directory.path().join("replacement.npy");
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&[3, 4, 5])
        .writer(BufWriter::new(File::create(&replacement).unwrap()))
        .begin_nd()
        .unwrap();
    for _ in 0..60 {
        writer.push(&999f32).unwrap();
    }
    writer.finish().unwrap();
    // Atomic replacement leaves any reader's existing immutable mmap valid.
    std::fs::rename(replacement, &path).unwrap();
    images.invalidate();
    assert_eq!(images.cache_bytes(), 0);
    assert!(images.plane(1).is_none());
    images.request_plane(1, desc, 0, 1, 64).unwrap();
    wait(&mut images);
    assert_eq!(images.plane(1).unwrap().sample(2, 3), Some(999.));
    images.shutdown();
}

#[test]
fn split_and_difference_combine_pixels_only_on_a_shared_grid() {
    let plane = |values: [f32; 4], unit: &str| super::PlaneData {
        id: 1,
        values: values.to_vec().into(),
        shape: [1, 4],
        source_shape: [1, 4],
        axes: ["y", "x"].map(|name| super::AxisScale {
            name: name.into(),
            frame: "world".into(),
            unit: "mm".into(),
            origin: 0.,
            spacing: 1.,
            coordinates: None,
        }),
        fixed: vec![],
        index: 0,
        count: 1,
        value_unit: unit.into(),
        domain: "attenuation".into(),
        stride: 1,
        statistics: reader::statistics(&values).unwrap(),
    };
    let left = plane([1., 2., 3., 4.], "mm^-1");
    let right = plane([10., 20., 30., 40.], "mm^-1");
    let same = [[0., 1.], [0., 1.]];
    assert_eq!(
        &*left.split(&right, 0.5, same).unwrap().values,
        &[1., 2., 30., 40.]
    );
    assert_eq!(
        &*left.split(&right, 0., same).unwrap().values,
        &[10., 20., 30., 40.]
    );
    assert_eq!(
        &*left.split(&right, 1., same).unwrap().values,
        &[1., 2., 3., 4.]
    );
    // Each side keeps its own window: the right's [0, 40] maps onto the left's [0, 4].
    assert_eq!(
        &*left
            .split(&right, 0.5, [[0., 4.], [0., 40.]])
            .unwrap()
            .values,
        &[1., 2., 3., 4.]
    );
    assert_eq!(
        &*left.difference(&right).unwrap().values,
        &[-9., -18., -27., -36.]
    );
    let other_unit = plane([10., 20., 30., 40.], "1/scan_length");
    assert!(left.split(&other_unit, 0.5, same).is_none());
    assert!(left.difference(&other_unit).is_none());
}
