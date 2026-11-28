//! Studio's projects, worker protocol and job state, independent of the window, GPU and HDF5.
#![deny(
    clippy::dbg_macro,
    clippy::expect_used,
    clippy::todo,
    clippy::unused_self,
    clippy::unwrap_used
)]
pub mod connection;
pub mod engine;
pub mod operations;
pub mod previews;
pub mod progress;
pub mod project;
pub mod protocol;
pub mod sync;
