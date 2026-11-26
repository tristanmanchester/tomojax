//! Rune remains the window/input/layout owner; scientific pixels share its wgpu device.
use rune::compose::{Compositor, DeviceKey, Frame};
use rune::render::{Instance, KIND_EFFECT};
use std::collections::{HashMap, HashSet};

use super::{
    RenderItem,
    gpu::{ImageTexture, Pipelines},
    ray::{GpuFrame, VolumeGpu},
};

struct Resources {
    key: DeviceKey,
    pipes: Pipelines,
    planes: HashMap<u32, ImageTexture>,
    volumes: HashMap<u32, VolumeGpu>,
}

pub struct ScientificCompositor<S: 'static> {
    source: fn(&S) -> Vec<RenderItem>,
    resources: Option<Resources>,
    uploading: bool,
    frame_count: u64,
    cpu_ms: f64,
}

impl<S: 'static> ScientificCompositor<S> {
    pub fn new(source: fn(&S) -> Vec<RenderItem>) -> Self {
        Self {
            source,
            resources: None,
            uploading: false,
            frame_count: 0,
            cpu_ms: 0.,
        }
    }
}

impl<S: 'static> Compositor<S> for ScientificCompositor<S> {
    fn animating(&self) -> bool {
        self.uploading
    }

    fn render(&mut self, frame: &mut Frame<'_, S>) {
        let started = std::time::Instant::now();
        let items = (self.source)(&frame.rt.state);
        let sites: HashMap<u32, Instance> = frame
            .rt
            .instances
            .iter()
            .filter(|i| i.params[1] == KIND_EFFECT)
            .map(|instance| {
                // Rune's documented KIND_EFFECT encoding stores the tag in two exact
                // 16-bit floats. Runtime::marks instead tracks element inspection IDs.
                let mark = (instance.border[0] as u32 & 0xffff)
                    | ((instance.border[1] as u32 & 0xffff) << 16);
                (mark, *instance)
            })
            .collect();
        let device_key = frame.device_key();
        if self.resources.as_ref().is_none_or(|r| r.key != device_key) {
            self.resources = None;
        }
        let resources = self.resources.get_or_insert_with(|| Resources {
            key: device_key,
            pipes: Pipelines::new(frame.device, frame.format),
            planes: HashMap::new(),
            volumes: HashMap::new(),
        });
        resources.pipes.add_format(frame.device, frame.format);
        let mut planes = HashSet::new();
        let mut volumes = HashSet::new();
        self.uploading = false;
        for item in &items {
            match item {
                RenderItem::Plane { mark, data, .. } => {
                    if !sites.contains_key(mark) {
                        continue;
                    }
                    planes.insert(*mark);
                    if resources.planes.get(mark).is_none_or(|p| p.id != data.id) {
                        resources.planes.insert(
                            *mark,
                            ImageTexture::plane(frame.device, frame.queue, &resources.pipes, data),
                        );
                    }
                }
                RenderItem::Volume {
                    mark,
                    data,
                    settings,
                } => {
                    let Some(site) = sites.get(mark) else {
                        continue;
                    };
                    volumes.insert(*mark);
                    let volume = resources
                        .volumes
                        .entry(*mark)
                        .or_insert_with(|| VolumeGpu::new(frame.device));
                    self.uploading |= volume.prepare(
                        GpuFrame {
                            device: frame.device,
                            queue: frame.queue,
                            encoder: frame.encoder,
                        },
                        &resources.pipes,
                        data.clone(),
                        settings,
                        site.rect,
                    );
                }
            }
        }
        // Offscreen datasets do not keep VRAM. Scalar arrays remain in the bounded CPU cache.
        resources.planes.retain(|mark, _| planes.contains(mark));
        resources.volumes.retain(|mark, _| volumes.contains(mark));
        frame.ui.prepare(
            frame.device,
            frame.queue,
            &mut frame.rt.text.atlas,
            &frame.rt.instances,
            frame.size,
        );
        frame.ui.draw_with(
            frame.encoder,
            frame.target,
            frame.rt.clear_color(),
            &mut |pass, site| {
                for item in &items {
                    match item {
                        RenderItem::Plane {
                            mark,
                            data,
                            display,
                        } if *mark == site.mark => {
                            if let Some(image) = resources.planes.get(mark) {
                                image.draw(
                                    frame.queue,
                                    &resources.pipes,
                                    pass,
                                    site,
                                    Some((data, display)),
                                );
                            }
                        }
                        RenderItem::Volume { mark, .. } if *mark == site.mark => {
                            if let Some(image) =
                                resources.volumes.get(mark).and_then(|v| v.frame.as_ref())
                            {
                                image.draw(frame.queue, &resources.pipes, pass, site, None);
                            }
                        }
                        _ => {}
                    }
                }
            },
        );
        self.frame_count += 1;
        self.cpu_ms += started.elapsed().as_secs_f64() * 1000.;
    }

    fn summary(&mut self, _: &wgpu::Device) -> Option<String> {
        Some(format!(
            "Scientific compositor: {} frames; {:.3} ms mean CPU submission (GPU completion measured separately).",
            self.frame_count,
            self.cpu_ms / self.frame_count.max(1) as f64
        ))
    }
}
