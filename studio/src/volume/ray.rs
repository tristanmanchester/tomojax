//! Resident 3D scalar texture, transfer LUT, progressive upload, and cached ray-cast framebuffer.
use super::{
    VolumeSettings,
    gpu::{ImageTexture, Pipelines, color_at},
};
use crate::imaging::VolumeData;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RayUniform {
    eye: [f32; 4],
    forward: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    extent: [f32; 4],
    clip_min: [f32; 4],
    clip_max: [f32; 4],
    lighting: [f32; 4],
    dimensions: [f32; 4],
}

struct Resident {
    data: Arc<VolumeData>,
    texture: wgpu::Texture,
    bind: wgpu::BindGroup,
    uploaded_z: usize,
}

pub(super) struct GpuFrame<'a> {
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub encoder: &'a mut wgpu::CommandEncoder,
}

pub(super) struct VolumeGpu {
    active: Option<Resident>,
    pending: Option<Resident>,
    uniform: wgpu::Buffer,
    lut: wgpu::Texture,
    lut_view: wgpu::TextureView,
    pub frame: Option<ImageTexture>,
    last_settings: Option<VolumeSettings>,
    pub draws: u64,
}

impl VolumeGpu {
    pub fn new(device: &wgpu::Device) -> Self {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Volume camera, clipping, lighting"),
            size: std::mem::size_of::<RayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lut = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Opacity and color transfer function"),
            size: wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let lut_view = lut.create_view(&Default::default());
        Self {
            active: None,
            pending: None,
            uniform,
            lut,
            lut_view,
            frame: None,
            last_settings: None,
            draws: 0,
        }
    }

    fn resident(
        &self,
        device: &wgpu::Device,
        pipes: &Pipelines,
        data: Arc<VolumeData>,
    ) -> Resident {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Normalised scalar volume"),
            size: wgpu::Extent3d {
                width: data.shape[2] as u32,
                height: data.shape[1] as u32,
                depth_or_array_layers: data.shape[0] as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Resident volume ray resources"),
            layout: &pipes.ray_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&pipes.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.lut_view),
                },
            ],
        });
        Resident {
            data,
            texture,
            bind,
            uploaded_z: 0,
        }
    }

    /// Returns true while uploading. Native frames stage at most 4 MiB; saved snapshots finish immediately.
    pub fn prepare(
        &mut self,
        frame: GpuFrame<'_>,
        pipes: &Pipelines,
        data: Arc<VolumeData>,
        settings: &VolumeSettings,
        canvas: [f32; 4],
    ) -> bool {
        let GpuFrame {
            device,
            queue,
            encoder,
        } = frame;
        let mut dirty = false;
        if self.active.as_ref().is_none_or(|v| v.data.id != data.id)
            && self.pending.as_ref().is_none_or(|v| v.data.id != data.id)
        {
            self.pending = Some(self.resident(device, pipes, data));
        }
        if let Some(pending) = self.pending.as_mut() {
            let [nz, ny, nx] = pending.data.shape;
            let limit = if rune::tooling::is_headless() {
                nz
            } else {
                (4 * 1024 * 1024 / (nx * ny * 2)).max(1)
            };
            let planes = (nz - pending.uploaded_z).min(limit);
            let first = pending.uploaded_z * nx * ny;
            let last = first + planes * nx * ny;
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &pending.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: pending.uploaded_z as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&pending.data.texels[first..last]),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(nx as u32 * 2),
                    rows_per_image: Some(ny as u32),
                },
                wgpu::Extent3d {
                    width: nx as u32,
                    height: ny as u32,
                    depth_or_array_layers: planes as u32,
                },
            );
            pending.uploaded_z += planes;
            if pending.uploaded_z == nz {
                self.active = self.pending.take();
                dirty = true;
            }
        }
        let Some(active) = &self.active else {
            return self.pending.is_some();
        };
        let max_edge = if settings.interacting { 192. } else { 512. };
        let factor = (max_edge / canvas[2].max(canvas[3]).max(1.)).min(1.);
        let size = (
            (canvas[2] * factor).round().max(1.) as u32,
            (canvas[3] * factor).round().max(1.) as u32,
        );
        if self.frame.as_ref().is_none_or(|f| f.size != size) {
            self.frame = None;
            dirty = true;
        }
        let frame = self.frame.get_or_insert_with(|| {
            ImageTexture::new(
                device,
                pipes,
                active.data.id,
                size,
                wgpu::TextureFormat::Rgba16Float,
            )
        });
        if self.last_settings.as_ref() != Some(settings) {
            dirty = true;
        }
        if dirty {
            let table = transfer_table(settings);
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.lut,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&table),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256 * 8),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: 256,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            let uniform = camera(&active.data, settings, size);
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Tomography volume integration"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &frame.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipes.ray);
            pass.set_bind_group(0, &active.bind, &[]);
            pass.draw(0..3, 0..1);
            self.last_settings = Some(settings.clone());
            self.draws += 1;
        }
        self.pending.is_some()
    }
}

fn camera(data: &VolumeData, settings: &VolumeSettings, size: (u32, u32)) -> RayUniform {
    let yaw = settings.yaw.to_radians();
    let pitch = settings.pitch.clamp(-89., 89.).to_radians();
    let direction = Vec3::new(
        pitch.cos() * yaw.sin(),
        -pitch.cos() * yaw.cos(),
        pitch.sin(),
    );
    let forward = -direction;
    let right = forward.cross(Vec3::Z).normalize();
    let up = right.cross(forward).normalize();
    let largest = data.extent.into_iter().fold(0f32, f32::max);
    let extent = data.extent.map(|n| n / largest * 2.);
    let aspect = size.0 as f32 / size.1 as f32;
    let tangent = 17.5f32.to_radians().tan();
    // Fit the physical bounding sphere at 100% zoom, including narrow panels.
    // Distance stays independent of orbit angle and clipping, so neither interaction
    // causes apparent automatic zooming or moves the specimen's frame of reference.
    let radius = Vec3::from_array(extent).length() * 0.5;
    let limiting_angle = (tangent * aspect.min(1.)).atan();
    let distance = radius / limiting_angle.sin();
    let eye = direction * (distance * 1.08 / settings.zoom.clamp(super::ZOOM[0], super::ZOOM[1]));
    let range = data.scalar_range;
    let span = (range[1] - range[0]).max(f32::EPSILON);
    let window = settings
        .window
        .filter(|w| w[0].is_finite() && w[1].is_finite() && w[1] > w[0])
        .unwrap_or(data.statistics.window);
    RayUniform {
        eye: [eye.x, eye.y, eye.z, aspect],
        forward: [forward.x, forward.y, forward.z, tangent],
        right: [right.x, right.y, right.z, (window[0] - range[0]) / span],
        up: [up.x, up.y, up.z, (window[1] - range[0]) / span],
        extent: [
            extent[0],
            extent[1],
            extent[2],
            settings.opacity.clamp(0., 5.),
        ],
        clip_min: [
            settings.clip_min[0].clamp(0., 1.),
            settings.clip_min[1].clamp(0., 1.),
            settings.clip_min[2].clamp(0., 1.),
            0.,
        ],
        clip_max: [
            settings.clip_max[0].clamp(0., 1.),
            settings.clip_max[1].clamp(0., 1.),
            settings.clip_max[2].clamp(0., 1.),
            if settings.interacting { 96. } else { 384. },
        ],
        lighting: [
            settings.ambient,
            settings.diffuse,
            settings.specular,
            if settings.lighting { 1. } else { 0. },
        ],
        dimensions: [
            data.shape[2] as f32,
            data.shape[1] as f32,
            data.shape[0] as f32,
            if settings.interacting { 1.8 } else { 0.7 },
        ],
    }
}

fn transfer_table(settings: &VolumeSettings) -> Vec<u16> {
    let mut points = settings.transfer.clone();
    points.retain(|p| {
        p.position.is_finite() && p.opacity.is_finite() && p.color.iter().all(|v| v.is_finite())
    });
    points.sort_by(|a, b| a.position.total_cmp(&b.position));
    let mut table = Vec::with_capacity(256 * 4);
    for i in 0..256 {
        let t = i as f32 / 255.;
        let (color, opacity) = if points.is_empty() {
            let density = ((t - settings.threshold.clamp(0., 0.99))
                / (1. - settings.threshold.clamp(0., 0.99)))
            .clamp(0., 1.);
            let shade = if settings.invert { 1. - t } else { t };
            (color_at(settings.colormap, shade), density.powf(1.7))
        } else {
            let upper = points
                .partition_point(|p| p.position < t)
                .min(points.len() - 1);
            let lower = upper.saturating_sub(1);
            let (a, b) = (&points[lower], &points[upper]);
            let fraction = if b.position > a.position {
                ((t - a.position) / (b.position - a.position)).clamp(0., 1.)
            } else {
                0.
            };
            (
                std::array::from_fn(|c| a.color[c] + (b.color[c] - a.color[c]) * fraction),
                a.opacity + (b.opacity - a.opacity) * fraction,
            )
        };
        table.extend(
            [color[0], color[1], color[2], opacity]
                .map(|v| half::f16::from_f32(v.clamp(0., 1.)).to_bits()),
        );
    }
    table
}

#[cfg(test)]
mod invert_tests {
    use super::*;

    #[test]
    fn inverting_reverses_the_colours_but_keeps_dense_material_opaque() {
        let plain = transfer_table(&VolumeSettings::default());
        let inverted = transfer_table(&VolumeSettings {
            invert: true,
            ..Default::default()
        });
        let texel = |table: &[u16], i: usize| -> [f32; 4] {
            std::array::from_fn(|c| half::f16::from_bits(table[i * 4 + c]).to_f32())
        };
        // The densest value's colour is the plain table's lowest colour...
        assert_eq!(texel(&inverted, 255)[..3], texel(&plain, 0)[..3]);
        // ...and every value keeps its opacity.
        for i in [0, 128, 255] {
            assert_eq!(texel(&inverted, i)[3], texel(&plain, i)[3]);
        }
    }
}
