//! Device-owned pipelines and scalar textures. No scientific I/O or conversion runs here.
use bytemuck::{Pod, Zeroable};
use rune::render::EffectSite;
use std::collections::HashMap;

use crate::imaging::{ColorMap, DisplaySettings, PlaneData, image_rect};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ImageUniform {
    rect: [f32; 4],
    canvas: [f32; 4],
    target: [f32; 4],
    window: [f32; 4],
    info: [f32; 4],
    uv: [f32; 4],
    clip: [f32; 4],
    fade: [u32; 4],
}

pub(super) struct Pipelines {
    pub image_layout: wgpu::BindGroupLayout,
    pub ray_layout: wgpu::BindGroupLayout,
    pub ray: wgpu::RenderPipeline,
    pub sampler: wgpu::Sampler,
    image_shader: wgpu::ShaderModule,
    image_pipeline_layout: wgpu::PipelineLayout,
    images: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
}

fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture(
    binding: u32,
    dimension: wgpu::TextureViewDimension,
    filterable: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn pipeline(
    device: &wgpu::Device,
    label: &str,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
    blend: bool,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: blend.then_some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

impl Pipelines {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let image_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Scientific scalar plane"),
            entries: &[
                uniform(0),
                texture(1, wgpu::TextureViewDimension::D2, false),
            ],
        });
        let ray_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tomography ray caster"),
            entries: &[
                uniform(0),
                texture(1, wgpu::TextureViewDimension::D3, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                texture(3, wgpu::TextureViewDimension::D2, true),
            ],
        });
        let image_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Scalar window and color mapping"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}",
                    rune::render::CLIP_WGSL,
                    include_str!("image.wgsl")
                )
                .into(),
            ),
        });
        let ray_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Physical volume ray integration"),
            source: wgpu::ShaderSource::Wgsl(include_str!("raycast.wgsl").into()),
        });
        let image_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Scalar image layout"),
                bind_group_layouts: &[Some(&image_layout)],
                immediate_size: 0,
            });
        let ray_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ray caster layout"),
            bind_group_layouts: &[Some(&ray_layout)],
            immediate_size: 0,
        });
        let ray = pipeline(
            device,
            "Tomography ray caster",
            &ray_shader,
            &ray_pipeline_layout,
            wgpu::TextureFormat::Rgba16Float,
            false,
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Trilinear scalar interpolation"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let mut result = Self {
            image_layout,
            ray_layout,
            ray,
            sampler,
            image_shader,
            image_pipeline_layout,
            images: HashMap::new(),
        };
        result.add_format(device, format);
        result.add_format(device, wgpu::TextureFormat::Rgba16Float);
        result
    }

    pub fn add_format(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        self.images.entry(format).or_insert_with(|| {
            pipeline(
                device,
                "Scientific image composite",
                &self.image_shader,
                &self.image_pipeline_layout,
                format,
                true,
            )
        });
    }
}

pub(super) struct ImageTexture {
    pub id: u64,
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
    pub size: (u32, u32),
}

impl ImageTexture {
    pub fn new(
        device: &wgpu::Device,
        pipes: &Pipelines,
        id: u64,
        size: (u32, u32),
        format: wgpu::TextureFormat,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scientific image"),
            size: wgpu::Extent3d {
                width: size.0,
                height: size.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | if format == wgpu::TextureFormat::Rgba16Float {
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                } else {
                    wgpu::TextureUsages::empty()
                },
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Image transform and window"),
            size: std::mem::size_of::<ImageUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Scientific image resources"),
            layout: &pipes.image_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        Self {
            id,
            texture,
            view,
            uniform,
            bind,
            size,
        }
    }

    pub fn plane(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipes: &Pipelines,
        data: &PlaneData,
    ) -> Self {
        let result = Self::new(
            device,
            pipes,
            data.id,
            (data.shape[1] as u32, data.shape[0] as u32),
            wgpu::TextureFormat::R32Float,
        );
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &result.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data.values),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(result.size.0 * 4),
                rows_per_image: Some(result.size.1),
            },
            wgpu::Extent3d {
                width: result.size.0,
                height: result.size.1,
                depth_or_array_layers: 1,
            },
        );
        result
    }

    pub fn draw(
        &self,
        queue: &wgpu::Queue,
        pipes: &Pipelines,
        pass: &mut wgpu::RenderPass<'_>,
        site: &EffectSite<'_>,
        plane: Option<(&PlaneData, &DisplaySettings, &PlaneData)>,
    ) {
        let (rect, window, kind) = if let Some((data, display, layer)) = plane {
            let w = display
                .window
                .filter(|w| w[0].is_finite() && w[1].is_finite() && w[1] > w[0])
                .unwrap_or(data.statistics.window);
            let mut rect = image_rect(site.instance.rect, data.aspect(), display);
            for (i, a) in [1, 0].into_iter().enumerate() {
                let [lo, hi] = layer.region[a];
                let unit = rect[2 + i] / data.source_shape[a] as f32;
                rect[i] += lo as f32 * unit;
                rect[2 + i] = (hi - lo) as f32 * unit;
            }
            (
                rect,
                [
                    w[0],
                    w[1],
                    color_id(display.colormap),
                    u8::from(display.invert) as f32,
                ],
                0.,
            )
        } else {
            (site.instance.rect, [0., 1., 0., 0.], 1.)
        };
        let uniform = ImageUniform {
            rect,
            canvas: site.instance.rect,
            target: [
                site.size[0] as f32,
                site.size[1] as f32,
                site.origin[0],
                site.origin[1],
            ],
            window,
            info: [u8::from(site.srgb_blend) as f32, kind, 0., 0.],
            uv: plane.map_or([1., 1., 0., 0.], |(_, _, layer)| layer.texture_transform()),
            clip: site.instance.clip,
            fade: site.instance.fade,
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
        if let Some(pipeline) = pipes.images.get(&site.format) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.draw(0..6, 0..1);
        }
    }
}

pub(super) fn color_id(color: ColorMap) -> f32 {
    match color {
        ColorMap::Gray => 0.,
        ColorMap::Viridis => 1.,
        ColorMap::Magma => 2.,
        ColorMap::Bone => 3.,
    }
}

pub(super) fn color_at(map: ColorMap, value: f32) -> [f32; 3] {
    let t = value.clamp(0., 1.);
    if map == ColorMap::Gray {
        return [t; 3];
    }
    #[expect(
        clippy::approx_constant,
        reason = "0.318 is a sampled Magma red-channel coefficient, not the reciprocal of pi"
    )]
    let stops = match map {
        ColorMap::Viridis => [
            [0.267, 0.005, 0.329],
            [0.231, 0.322, 0.545],
            [0.129, 0.569, 0.549],
            [0.369, 0.788, 0.384],
            [0.993, 0.906, 0.144],
        ],
        ColorMap::Magma => [
            [0.001, 0., 0.016],
            [0.318, 0.071, 0.486],
            [0.718, 0.216, 0.475],
            [0.988, 0.537, 0.380],
            [0.988, 0.992, 0.749],
        ],
        _ => [
            [0.06, 0.10, 0.10],
            [0.22, 0.34, 0.30],
            [0.49, 0.58, 0.46],
            [0.78, 0.76, 0.60],
            [0.99, 0.97, 0.87],
        ],
    };
    let low = (t * 4.).floor().min(3.) as usize;
    let fraction = t * 4. - low as f32;
    std::array::from_fn(|c| stops[low][c] + (stops[low + 1][c] - stops[low][c]) * fraction)
}
