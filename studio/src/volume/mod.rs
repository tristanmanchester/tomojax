//! Scalar GPU image display and physically registered volume ray casting in Rune slots.
mod compositor;
mod gpu;
mod ray;

use crate::imaging::{ColorMap, DisplaySettings, PlaneData, VolumeData};
use std::sync::Arc;

pub use compositor::ScientificCompositor;

/// The 3D camera's zoom range; the view's controls keep to it.
pub const ZOOM: [f32; 2] = [0.25, 5.];

#[derive(Clone, Debug, PartialEq)]
pub struct TransferPoint {
    /// Position within the selected scalar window, from zero to one.
    pub position: f32,
    pub color: [f32; 3],
    pub opacity: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct VolumeSettings {
    pub yaw: f32,
    pub pitch: f32,
    pub zoom: f32,
    pub window: Option<[f32; 2]>,
    pub colormap: ColorMap,
    /// Reverses the colour ramp; opacity still follows the values, so dense
    /// material stays visible.
    pub invert: bool,
    pub threshold: f32,
    pub opacity: f32,
    /// An empty list selects the colormap/threshold ramp; sorted points define a custom LUT.
    pub transfer: Vec<TransferPoint>,
    pub lighting: bool,
    pub ambient: f32,
    pub diffuse: f32,
    pub specular: f32,
    /// Axis-aligned clipping fractions in physical x,y,z order.
    pub clip_min: [f32; 3],
    pub clip_max: [f32; 3],
    /// Reduces ray steps and framebuffer resolution while the pointer is manipulating a view.
    pub interacting: bool,
}

impl Default for VolumeSettings {
    fn default() -> Self {
        Self {
            yaw: 35.,
            pitch: 22.,
            zoom: 1.,
            window: None,
            colormap: ColorMap::Bone,
            invert: false,
            threshold: 0.08,
            opacity: 1.,
            transfer: vec![],
            lighting: true,
            ambient: 0.3,
            diffuse: 0.65,
            specular: 0.2,
            clip_min: [0.; 3],
            clip_max: [1.; 3],
            interacting: false,
        }
    }
}

#[derive(Clone)]
pub enum RenderItem {
    Plane {
        mark: u32,
        data: Arc<PlaneData>,
        display: DisplaySettings,
    },
    Volume {
        mark: u32,
        data: Arc<VolumeData>,
        settings: VolumeSettings,
    },
}

#[cfg(test)]
mod tests;
