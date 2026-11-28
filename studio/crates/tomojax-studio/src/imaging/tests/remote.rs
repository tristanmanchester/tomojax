use super::*;
use std::path::{Path, PathBuf};
use studio_core::engine::{Engine, EngineEvent};

fn worker(relay: &[String]) -> Engine {
    let python = std::env::var_os("TOMOJAX_TEST_PYTHON").unwrap();
    Engine::spawn_through(PathBuf::from(python), relay, Arc::new(|| {})).unwrap()
}

fn advance(imaging: &mut Imaging, engine: &mut Engine) -> bool {
    for event in engine.poll() {
        match event {
            EngineEvent::Reply { response, stale } => {
                imaging.worker_reply(response, stale);
            }
            EngineEvent::Disconnected { .. } => {
                imaging.disconnected();
                return false;
            }
            _ => {}
        }
    }
    imaging.dispatch(engine, &[Some([-2., 2.]); 8]);
    imaging.poll();
    true
}

fn fixtures(directory: &Path) -> (ArrayDescriptor, ArrayDescriptor, ArrayDescriptor) {
    let shape = [4, 51, 73];
    let path = directory.join("ramp.npy");
    npy(&path, shape, true);
    let mut npy = descriptor(
        Storage::Npy {
            path: path.to_string_lossy().into(),
            artifact: None,
        },
        shape,
    );
    let coordinate_path = directory.join("z.npy");
    let mut coords = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&[4])
        .writer(File::create(&coordinate_path).unwrap())
        .begin_nd()
        .unwrap();
    for value in [8.5f64, 6.75, 2., -0.125] {
        coords.push(&value).unwrap();
    }
    coords.finish().unwrap();
    npy.axes[0].coordinates = Coordinates::Coordinates {
        values: Storage::Npy {
            path: coordinate_path.to_string_lossy().into(),
            artifact: None,
        },
    };
    npy.axes[1].coordinates = Coordinates::Uniform {
        origin: 5.,
        spacing: -0.5,
    };
    let path = directory.join("ramp.h5");
    {
        let file = hdf5::File::create(&path).unwrap();
        let mut detector = vec![0f32; 51 * 73];
        detector[74] = 1.; // An impulse between the displayed sample centers.
        file.new_dataset::<f32>()
            .shape([51, 73])
            .create("lines")
            .unwrap()
            .write_raw(&detector)
            .unwrap();
        file.new_dataset::<f32>()
            .shape(shape)
            .chunk([1, 17, 10])
            .create("plane")
            .unwrap()
            .write_raw(&ramp(shape))
            .unwrap();
    }
    let hdf = descriptor(
        Storage::Hdf5 {
            path: path.to_string_lossy().into(),
            dataset: "plane".into(),
        },
        shape,
    );
    let lines = descriptor(
        Storage::Hdf5 {
            path: path.to_string_lossy().into(),
            dataset: "lines".into(),
        },
        [51, 1, 73],
    );
    (npy, hdf, lines)
}

#[test]
#[ignore = "requires TOMOJAX_TEST_PYTHON with the real worker"]
fn fetched_planes_match_native_npy_and_hdf5_bit_for_bit() {
    let directory = tempfile::tempdir().unwrap();
    let (npy, hdf, lines) = fixtures(directory.path());
    let mut engine = worker(&[]);
    let mut images = Imaging::default();
    images.select_source(true);
    images.request_volume(8, hdf.clone(), 32).unwrap();
    images.request_plane(1, npy.clone(), 0, 0, 32).unwrap();
    // Scroll before polling either completion. Both slots must retain their registration.
    for desc in [&npy, &hdf] {
        for axis in 0..3 {
            let expected = reader::plane(desc, axis, 2, 32).unwrap();
            images.request_plane(1, desc.clone(), axis, 2, 32).unwrap();
            let until = Instant::now() + Duration::from_secs(15);
            while !matches!(images.status(1), ImagingStatus::Ready) {
                assert!(advance(&mut images, &mut engine));
                assert!(
                    !matches!(images.status(1), ImagingStatus::Error(_)),
                    "{:?}",
                    images.status(1)
                );
                assert!(Instant::now() < until, "{:?}", images.status(1));
                std::thread::sleep(Duration::from_millis(1));
            }
            let actual = images.plane(1).unwrap();
            assert!(actual.lossless);
            assert_eq!(actual.shape, expected.shape);
            assert_eq!(actual.fixed, expected.fixed);
            if std::ptr::eq(desc, &npy) && axis == 2 {
                assert_eq!(
                    actual.axes[0].coordinates.as_deref(),
                    Some([8.5, 2.].as_slice())
                );
                assert_eq!(actual.axes[1].spacing, -1.);
                assert_eq!(actual.fixed.last().unwrap().coordinate, -1.);
            }
            assert!(
                actual
                    .values
                    .iter()
                    .map(|x| x.to_bits())
                    .eq(expected.values.iter().map(|x| x.to_bits()))
            );
        }
    }
    assert!(images.volume(8).is_some());
    images.request_plane(2, lines, 1, 0, 32).unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while images.status(2) != &ImagingStatus::Ready {
        assert!(
            advance(&mut images, &mut engine) && Instant::now() < deadline,
            "{:?}",
            images.status(2)
        );
    }
    let sinogram = images.plane(2).unwrap();
    assert_eq!((sinogram.shape, sinogram.stride), ([17, 25], 3));
    assert_eq!(sinogram.fixed.last().unwrap().coordinate, 4.);
    // The impulse at source (1, 1) falls between display texels after decimation.
    let cursor = [1.5 / 73., 1.5 / 51.];
    assert_eq!(sinogram.probe(cursor), Some(([0, 0], 0.)));
    assert_eq!(sinogram.probe([0.05, 0.08]), Some(([3, 3], 0.)));
    assert_eq!(sinogram.axes[1].coordinate(1.), -0.5);
    engine.drain_shutdown(Duration::from_secs(3));
}

#[test]
#[ignore = "requires TOMOJAX_TEST_PYTHON and tools/slowlink.py"]
fn killing_slowlink_mid_scroll_retires_old_frames() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("ramp.npy");
    let shape = [8, 128, 128];
    npy(&path, shape, false);
    let desc = descriptor(
        Storage::Npy {
            path: path.to_string_lossy().into(),
            artifact: None,
        },
        shape,
    );
    let link = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../tools/slowlink.py");
    let mut engine = worker(&[
        link.to_string_lossy().into(),
        "--kill-after".into(),
        "6".into(),
    ]);
    let mut images = Imaging::default();
    images.select_source(true);
    let until = Instant::now() + Duration::from_secs(15);
    let mut shown = 0;
    let mut index = 0;
    loop {
        images
            .request_plane(1, desc.clone(), 0, index, 128)
            .unwrap();
        if !advance(&mut images, &mut engine) {
            break;
        }
        if images.plane(1).is_some() {
            shown += 1;
        }
        index = (index + 1) % 8;
        assert!(Instant::now() < until);
        std::thread::sleep(Duration::from_millis(15));
    }
    assert!(shown > 0);
    assert!(images.plane(1).is_none());
    // Reuse the same slot, path and channel with a new worker whose request IDs start over.
    let mut replacement = worker(&[]);
    images.request_plane(1, desc, 0, 7, 128).unwrap();
    let until = Instant::now() + Duration::from_secs(15);
    loop {
        assert!(advance(&mut images, &mut replacement));
        if let Some(plane) = images.plane(1) {
            assert_eq!(plane.index, 7);
        }
        if matches!(images.status(1), ImagingStatus::Ready) {
            break;
        }
        assert!(Instant::now() < until);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(images.plane(1).unwrap().sample(0, 0), Some(700.));
    replacement.drain_shutdown(Duration::from_secs(3));
}
