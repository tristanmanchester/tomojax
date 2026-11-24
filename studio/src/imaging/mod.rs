//! Asynchronous scalar data access and bounded caching, independent of numerical jobs.
mod data;
mod npy;
mod reader;
mod service;

pub use data::{
    AxisScale, ColorMap, DisplaySettings, PlaneData, Statistics, TableData, VolumeData, image_rect,
};
pub(crate) use reader::{plane as read_checkpoint_plane, table as read_pose_table};
pub use service::{Imaging, ImagingStatus, LatencySample};

#[cfg(test)]
mod tests;
