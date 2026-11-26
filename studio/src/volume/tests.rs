//! Scientific GPU checks use the same Rune compositor as the application and snapshots.
use super::{RenderItem, ScientificCompositor, TransferPoint, VolumeSettings};
use crate::imaging::{AxisScale, ColorMap, DisplaySettings, PlaneData, Statistics, VolumeData};
use rune::compose::{Compositor, Frame};
use rune::render::{Gpu, OFFSCREEN_FORMAT, Offscreen, Renderer};
use rune::{Cx, El, Env, Runtime, el};
use std::sync::Arc;

fn view(_: &RenderItem, _: &mut Cx<RenderItem>) -> El {
    el().size(96., 96.).effect(1)
}
fn items(item: &RenderItem) -> Vec<RenderItem> {
    vec![item.clone()]
}

struct Rig {
    gpu: Gpu,
    ui: Renderer,
    target: Offscreen,
    runtime: Runtime<RenderItem>,
    compositor: ScientificCompositor<RenderItem>,
}

impl Rig {
    fn new(item: RenderItem) -> Self {
        let gpu = Gpu::headless();
        let ui = Renderer::new(&gpu.device, OFFSCREEN_FORMAT);
        let target = Offscreen::new(&gpu.device, 96, 96);
        let mut runtime = Runtime::new(
            item,
            view,
            (96., 96.),
            Env {
                system_dark: true,
                scale: 1.,
            },
        );
        runtime.transparent = true;
        Self {
            gpu,
            ui,
            target,
            runtime,
            compositor: ScientificCompositor::new(items),
        }
    }
    fn draw(&mut self, item: RenderItem) -> Vec<u8> {
        self.runtime.state = item;
        self.runtime.frame(0.);
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        self.compositor.render(&mut Frame {
            device: &self.gpu.device,
            queue: &self.gpu.queue,
            encoder: &mut encoder,
            target: &self.target.view,
            format: OFFSCREEN_FORMAT,
            size: (96, 96),
            dt: 0.,
            update_ms: 0.,
            rt: &mut self.runtime,
            ui: &mut self.ui,
        });
        self.target.copy(&mut encoder);
        self.ui.submit(&self.gpu.queue, encoder);
        self.target.read(&self.gpu.device)
    }
}

fn statistics() -> Statistics {
    Statistics {
        min: 0.,
        max: 1.,
        mean: 0.5,
        std: 0.5,
        finite_count: 4,
        missing_count: 0,
        histogram: vec![2, 2],
        window: [0., 1.],
    }
}

fn sample(bytes: &[u8], x: usize, y: usize) -> &[u8] {
    &bytes[(y * 96 + x) * 4..(y * 96 + x + 1) * 4]
}

#[test]
#[ignore = "requires a real wgpu adapter; exercised by the GPU verification command"]
fn scalar_window_and_inversion_are_gpu_updates_of_the_same_quantitative_texture() {
    let data = Arc::new(PlaneData {
        id: 1,
        values: vec![0., 10., 0., 10.].into(),
        shape: [2, 2],
        source_shape: [2, 2],
        axes: ["y", "x"].map(|name| AxisScale {
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
        value_unit: "mm^-1".into(),
        domain: "attenuation".into(),
        stride: 1,
        statistics: statistics(),
    });
    let mut display = DisplaySettings {
        window: Some([0., 10.]),
        ..Default::default()
    };
    let item = |display: &DisplaySettings| RenderItem::Plane {
        mark: 1,
        data: data.clone(),
        display: display.clone(),
    };
    let mut rig = Rig::new(item(&display));
    let original = rig.draw(item(&display));
    assert!(sample(&original, 24, 48)[0] < 8);
    assert!(sample(&original, 72, 48)[0] > 247);
    display.window = Some([0., 20.]);
    let windowed = rig.draw(item(&display));
    assert!((120..=135).contains(&sample(&windowed, 72, 48)[0]));
    display.invert = true;
    let inverted = rig.draw(item(&display));
    assert!(sample(&inverted, 24, 48)[0] > 247);
    assert_eq!(&*data.values, &[0., 10., 0., 10.]);
}

fn block() -> Arc<VolumeData> {
    Arc::new(VolumeData {
        id: 2,
        texels: vec![half::f16::ONE.to_bits(); 16 * 16 * 16].into(),
        shape: [16; 3],
        source_shape: [16; 3],
        extent: [2., 1., 1.],
        origin: [-0.9375, -0.46875, -0.46875],
        spacing: [0.125, 0.0625, 0.0625],
        scalar_range: [0., 1.],
        statistics: statistics(),
        value_unit: "mm^-1".into(),
        length_unit: "mm".into(),
        stride: 1,
    })
}

#[test]
#[ignore = "requires a real wgpu adapter; exercised by the GPU verification command"]
fn ray_integration_preserves_anisotropic_extent_and_step_independent_opacity() {
    let data = block();
    let mut settings = VolumeSettings {
        yaw: 0.,
        pitch: 0.,
        zoom: 0.8,
        opacity: 0.02,
        lighting: false,
        threshold: 0.,
        colormap: ColorMap::Gray,
        window: Some([0., 1.]),
        ..Default::default()
    };
    let item = |settings: &VolumeSettings| RenderItem::Volume {
        mark: 1,
        data: data.clone(),
        settings: settings.clone(),
    };
    let mut rig = Rig::new(item(&settings));
    let fine = rig.draw(item(&settings));
    // A central ray traverses exactly one physical unit after the longest edge is normalized to two.
    let expected = ((1. - (-80f64 * 0.02).exp()) * 255.).round() as u8;
    assert!(
        sample(&fine, 48, 48)[3].abs_diff(expected) <= 3,
        "ray opacity disagrees with Beer–Lambert integration"
    );
    let occupied: Vec<_> = fine
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .filter(|(_, p)| p[3] > 5)
        .map(|(i, _)| (i % 96, i / 96))
        .collect();
    assert!(
        !occupied.is_empty(),
        "The GPU volume slot must contain the scientific volume"
    );
    let width = occupied.iter().map(|p| p.0).max().unwrap()
        - occupied.iter().map(|p| p.0).min().unwrap()
        + 1;
    let height = occupied.iter().map(|p| p.1).max().unwrap()
        - occupied.iter().map(|p| p.1).min().unwrap()
        + 1;
    assert!(
        (width as f32 / height as f32 - 2.).abs() < 0.15,
        "Anisotropic extent was flattened: {width}×{height}"
    );
    settings.interacting = true;
    let interactive = rig.draw(item(&settings));
    assert!(
        sample(&interactive, 48, 48)[3].abs_diff(sample(&fine, 48, 48)[3]) <= 3,
        "Changing ray step size must not change material opacity"
    );
    settings.interacting = false;
    settings.yaw = 35.;
    settings.pitch = 25.;
    settings.zoom = 1.;
    settings.opacity = 1.;
    let fitted = rig.draw(item(&settings));
    assert!(sample(&fitted, 48, 48)[3] > 200);
    for edge in 0..96 {
        for (x, y) in [(0, edge), (95, edge), (edge, 0), (edge, 95)] {
            assert_eq!(
                sample(&fitted, x, y)[3],
                0,
                "100% zoom must fit the whole physical box without cropping"
            );
        }
    }
}

#[test]
#[ignore = "requires a real wgpu adapter; exercised by the GPU verification command"]
fn transfer_function_lighting_and_clipping_change_the_gpu_volume() {
    let data = block();
    let mut settings = VolumeSettings {
        yaw: 0.,
        pitch: 0.,
        zoom: 0.8,
        lighting: false,
        window: Some([0., 1.]),
        transfer: vec![
            TransferPoint {
                position: 0.,
                color: [1., 0., 0.],
                opacity: 0.,
            },
            TransferPoint {
                position: 1.,
                color: [1., 0., 0.],
                opacity: 1.,
            },
        ],
        ..Default::default()
    };
    let item = |settings: &VolumeSettings| RenderItem::Volume {
        mark: 1,
        data: data.clone(),
        settings: settings.clone(),
    };
    let mut rig = Rig::new(item(&settings));
    let red = rig.draw(item(&settings));
    assert!(sample(&red, 48, 48)[0] > 200 && sample(&red, 48, 48)[2] < 5);
    for point in &mut settings.transfer {
        point.color = [0., 0., 1.];
    }
    let blue = rig.draw(item(&settings));
    assert!(sample(&blue, 48, 48)[2] > 200 && sample(&blue, 48, 48)[0] < 5);
    settings.lighting = true;
    settings.ambient = 0.2;
    settings.diffuse = 0.;
    settings.specular = 0.;
    let shaded = rig.draw(item(&settings));
    assert!(sample(&shaded, 48, 48)[2] < sample(&blue, 48, 48)[2] - 40);
    settings.clip_min[0] = 0.5;
    let clipped = rig.draw(item(&settings));
    let left_alpha = |pixels: &[u8]| -> u64 {
        (0..96)
            .flat_map(|y| (0..47).map(move |x| u64::from(sample(pixels, x, y)[3])))
            .sum()
    };
    assert!(
        left_alpha(&clipped) < left_alpha(&shaded) / 20,
        "The x clipping plane must remove the physical left half"
    );
    assert!(
        sample(&clipped, 60, 48)[3] > 100,
        "Clipping must preserve the remaining material"
    );
    settings.opacity = 0.;
    let transparent = rig.draw(item(&settings));
    assert!(transparent.as_chunks::<4>().0.iter().all(|p| p[3] == 0));
}
