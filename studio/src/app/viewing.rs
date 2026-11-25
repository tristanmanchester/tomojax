use super::*;
use crate::imaging::{ColorMap, DisplaySettings, ImagingStatus};
use crate::protocol::{ArrayDescriptor, Coordinates, Storage};
use crate::ui::model::Activity;
use crate::volume::{RenderItem, VolumeSettings};

impl AppState {
    pub(super) fn default_view(&self) -> PlaneKind {
        if self.active_volume().is_some() {
            return PlaneKind::Axial;
        }
        [PlaneKind::Axial, PlaneKind::Coronal, PlaneKind::Sagittal]
            .into_iter()
            .find(|plane| self.spatial_preview(*plane).is_some())
            .unwrap_or(PlaneKind::Projection)
    }
    fn spatial_preview(&self, plane: PlaneKind) -> Option<&ArrayDescriptor> {
        let arrays = if self.preview_context.active.is_some() {
            &self.preview_context.arrays
        } else {
            &self
                .model
                .active_asset
                .as_ref()
                .and_then(|id| self.project.as_ref()?.document.asset(id))?
                .arrays
        };
        arrays.values().find(|array| {
            let spatial = if array.shape.len() == 2 {
                true
            } else {
                array.shape.len() == 3
                    && array.axes.first().is_some_and(|axis| axis.name == "centre")
            };
            spatial && registered_plane_kind(array) == Some(plane)
        })
    }
    pub(super) fn slice_count(&self, axis: usize) -> usize {
        self.volume_dimensions()
            .map(|shape| shape[axis])
            .unwrap_or_else(|| {
                self.spatial_preview(
                    [PlaneKind::Axial, PlaneKind::Coronal, PlaneKind::Sagittal][axis],
                )
                .map_or(1, |array| {
                    if array.shape.len() == 3 {
                        array.shape[0]
                    } else {
                        1
                    }
                })
            })
    }
    pub(super) fn active_volume(&self) -> Option<&ArrayDescriptor> {
        if self.preview_context.active.is_some() {
            return None;
        }
        self.model
            .active_asset
            .as_ref()
            .and_then(|id| self.project.as_ref()?.document.asset(id)?.volume())
    }
    pub(super) fn volume_dimensions(&self) -> Option<[usize; 3]> {
        let volume = self.active_volume()?;
        let mut dimensions = [0; 3];
        for (physical, name) in ["z", "y", "x"].iter().enumerate() {
            let stored = volume.axes.iter().position(|axis| axis.name == *name)?;
            dimensions[physical] = *volume.shape.get(stored)?;
        }
        Some(dimensions)
    }
    pub(super) fn active_projections(&self) -> Option<&ArrayDescriptor> {
        if self.preview_context.active == Some(crate::previews::PreviewKind::Corrections) {
            return self.preview_context.arrays.get(&self.ui.preview_array);
        }
        let project = self.project.as_ref()?;
        let asset = project.document.asset(self.model.active_asset.as_ref()?)?;
        asset
            .arrays
            .get(&self.ui.preview_array)
            .filter(|array| array.axes.first().is_some_and(|axis| axis.name == "view"))
            .or_else(|| asset.projections())
            .or_else(|| {
                asset
                    .parent
                    .as_ref()
                    .and_then(|id| project.document.asset(id)?.projections())
            })
    }
    fn resolved(&self, descriptor: &ArrayDescriptor) -> ArrayDescriptor {
        let mut descriptor = descriptor.clone();
        let resolve = |storage: &mut Storage| match storage {
            Storage::Hdf5 { path, .. } | Storage::Npy { path, .. } => {
                if let Some(project) = &self.project {
                    *path = project.resolve(Path::new(path)).to_string_lossy().into();
                }
            }
            _ => {}
        };
        resolve(&mut descriptor.storage);
        for axis in &mut descriptor.axes {
            if let Coordinates::Coordinates { values } = &mut axis.coordinates {
                resolve(values);
            }
        }
        descriptor
    }

    /// Makes `id` the asset being viewed and prepared: its view starts fresh (the
    /// previous asset's pixels, window, slices and sinogram row don't carry over)
    /// and an Open & calibrate form takes its kind. Selection is the caller's.
    pub(super) fn activate(&mut self, id: String) {
        // Activating the active asset again keeps its view and drafts.
        if self.model.active_asset.as_ref() == Some(&id) {
            return;
        }
        self.ui.rename = None;
        self.model.active_asset = Some(id);
        self.image_key.clear();
        self.applied_window = None;
        self.ui.display.window_low.replace_text("");
        self.ui.display.window_high.replace_text("");
        self.window_prefill = Some([String::new(), String::new()]);
        self.centre_sinogram = true;
        self.ui.display.indices = self
            .volume_dimensions()
            .map_or([0; 3], |shape| shape.map(|n| n / 2));
        self.ui.main_view = self.default_view();
        if self.ui.operation == "open" {
            self.set_operation("open", None);
        }
        self.refresh_images();
    }

    /// Points asset `id` at `path`, chosen by the user after its data moved.
    pub(super) fn relink(&mut self, id: &str, path: &Path) {
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("npz"))
        {
            self.model.error = Some(
                "NPZ inputs are not supported. Choose NXtomo/HDF5 or a recognised acquisition."
                    .into(),
            );
            return;
        }
        let mut reread = true;
        if let Some(project) = &mut self.project {
            let path = project.stored_path(path);
            let root = project.root.clone();
            if let Some(asset) = project.document.asset_mut(id) {
                asset.relink(path, &root);
                reread = !asset.is_result();
            }
        }
        self.model
            .measurements
            .retain(|measurement| measurement.asset != id);
        self.imaging.invalidate();
        if reread {
            self.start_metadata(id);
        }
        self.image_key.clear();
        self.mark_dirty();
        self.refresh_model();
        self.refresh_images();
    }

    /// The zoom in effect on screen: the 3D camera keeps to its own range.
    pub(crate) fn shown_zoom(&self) -> f32 {
        let zoom = self.ui.display.zoom;
        if self.ui.main_view == PlaneKind::Volume && self.ui.workspace != Workspace::Compare {
            zoom.clamp(crate::volume::ZOOM[0], crate::volume::ZOOM[1])
        } else {
            zoom
        }
    }

    /// The two compared volumes, each with its stored axis for slice `axis`.
    fn compared_volumes(&self, axis: usize) -> [Option<(ArrayDescriptor, usize)>; 2] {
        [
            self.model.compare_left.clone(),
            self.model.compare_right.clone(),
        ]
        .map(|id| {
            let descriptor = self
                .project
                .as_ref()?
                .document
                .asset(&id?)?
                .volume()
                .cloned()?;
            let stored = descriptor
                .axes
                .iter()
                .position(|a| a.name == ["z", "y", "x"][axis])
                .unwrap_or(0);
            Some((descriptor, stored))
        })
    }

    /// How many slices along `axis` both compared volumes have, from their
    /// registration (known before any image is read).
    pub(super) fn compared_count(&self, axis: usize) -> Option<usize> {
        self.compared_volumes(axis)
            .iter()
            .flatten()
            .map(|(descriptor, stored)| descriptor.shape[*stored])
            .min()
    }

    /// The slice axis on screen: Compare's plane in Compare, else the main view's.
    pub(super) fn shown_axis(&self) -> Option<usize> {
        if self.ui.workspace == Workspace::Compare {
            self.ui.compare_plane.axis()
        } else {
            self.ui.main_view.axis()
        }
    }

    /// Restores the rule between selection and the active asset after the
    /// selection changed: the active asset is one of the selected ones (the
    /// first in project order when it left), or nothing is active when nothing
    /// is selected.
    pub(super) fn settle_active(&mut self) {
        if self
            .model
            .active_asset
            .as_ref()
            .is_some_and(|id| self.model.selected_assets.contains(id))
        {
            return;
        }
        let next = if self.model.selected_assets.is_empty() {
            None
        } else {
            self.selected_inputs().first().cloned()
        };
        match next {
            Some(id) => self.activate(id),
            None => {
                self.model.active_asset = None;
                self.image_key.clear();
            }
        }
    }

    /// Records what a read of asset `id`'s file found (the header reader or the
    /// worker's inspection), then brings the view up to date: an Open & calibrate
    /// form still on the old kind follows the new one, and an active asset seen
    /// for the first time opens on its default view at its central slices.
    pub(super) fn record_read(
        &mut self,
        id: &str,
        read: Result<(String, std::collections::BTreeMap<String, ArrayDescriptor>), String>,
        report: Value,
    ) {
        let Some(project) = &mut self.project else {
            return;
        };
        let root = project.root.clone();
        let Some(asset) = project.document.asset_mut(id) else {
            return;
        };
        let previous = asset.kind.clone();
        let first = asset.arrays.is_empty();
        asset.inspection = report;
        match read {
            Ok((kind, arrays)) => {
                asset.issue = None;
                asset.register_read(&kind, arrays, &root);
            }
            Err(error) => asset.issue = Some(error),
        }
        let kind = asset.kind.clone();
        self.model
            .measurements
            .retain(|measurement| measurement.asset != id);
        if self.model.active_asset.as_deref() == Some(id) {
            if self.ui.operation == "open"
                && self.ui.parameter_choices.get("kind") == Some(&previous)
            {
                self.ui.parameter_choices.insert("kind".into(), kind);
            }
            if first {
                self.ui.display.indices = self
                    .volume_dimensions()
                    .map_or([0; 3], |shape| shape.map(|n| n / 2));
                self.ui.main_view = self.default_view();
            }
        }
        self.imaging.invalidate();
        self.image_key.clear();
        self.mark_dirty();
        self.refresh_images();
    }

    pub(super) fn refresh_images(&mut self) {
        if let Some(volume) = self.active_volume().cloned() {
            let volume = self.resolved(&volume);
            for axis in 0..3 {
                let name = ["z", "y", "x"][axis];
                let Some(stored) = volume.axes.iter().position(|a| a.name == name) else {
                    self.model.error = Some(format!("Volume has no registered {name} axis"));
                    continue;
                };
                let count = volume.shape.get(stored).copied().unwrap_or(1);
                let index = self.ui.display.indices[axis].min(count.saturating_sub(1));
                if self.ui.workspace != Workspace::Compare {
                    self.ui.display.indices[axis] = index;
                }
                if let Err(error) = self.imaging.request_plane(
                    (axis + 1) as u32,
                    volume.clone(),
                    stored,
                    index,
                    1024,
                ) {
                    self.model.error = Some(error);
                }
            }
            if let Err(error) = self.imaging.request_volume(8, volume, 192) {
                self.model.error = Some(error);
            }
        } else {
            self.imaging.clear(8);
            for (axis, plane) in [PlaneKind::Axial, PlaneKind::Coronal, PlaneKind::Sagittal]
                .into_iter()
                .enumerate()
            {
                if let Some(array) = self.spatial_preview(plane).cloned() {
                    let count = if array.shape.len() == 3 {
                        array.shape[0]
                    } else {
                        1
                    };
                    let index = self.ui.display.indices[axis].min(count.saturating_sub(1));
                    if self.ui.workspace != Workspace::Compare {
                        self.ui.display.indices[axis] = index;
                    }
                    if let Err(error) = self.imaging.request_plane(
                        plane.mark(),
                        self.resolved(&array),
                        0,
                        index,
                        1024,
                    ) {
                        self.model.error = Some(error);
                    }
                } else {
                    self.imaging.clear(plane.mark());
                }
            }
            if self.ui.main_view == PlaneKind::Volume {
                self.ui.main_view = self.default_view();
            }
        }
        if let Some(projections) = self.active_projections().cloned() {
            let projections = self.resolved(&projections);
            if self.centre_sinogram {
                self.ui.display.sinogram_row = projections.shape[1] / 2;
                self.centre_sinogram = false;
            }
            self.ui.display.projection_index = self
                .ui
                .display
                .projection_index
                .min(projections.shape[0].saturating_sub(1));
            self.ui.display.sinogram_row = self
                .ui
                .display
                .sinogram_row
                .min(projections.shape[1].saturating_sub(1));
            if let Err(error) = self.imaging.request_plane(
                4,
                projections.clone(),
                0,
                self.ui.display.projection_index,
                1024,
            ) {
                self.model.error = Some(error);
            }
            if let Err(error) =
                self.imaging
                    .request_plane(5, projections, 1, self.ui.display.sinogram_row, 1024)
            {
                self.model.error = Some(error);
            }
        } else {
            self.imaging.clear(4);
            self.imaging.clear(5);
        }
        // Both sides show the same index, so it is clamped to the shorter volume.
        let plane_axis = self.ui.compare_plane.axis().unwrap();
        let compared = self.compared_volumes(plane_axis);
        let index = self
            .compared_count(plane_axis)
            .map_or(self.ui.display.indices[plane_axis], |count| {
                self.ui.display.indices[plane_axis].min(count.saturating_sub(1))
            });
        // Keep the stored index on the shown slice, so the next step starts from it.
        if self.ui.workspace == Workspace::Compare && compared.iter().any(Option::is_some) {
            self.ui.display.indices[plane_axis] = index;
        }
        for (slot, compared) in [6, 7].into_iter().zip(compared) {
            if let Some((descriptor, axis)) = compared {
                if let Err(error) =
                    self.imaging
                        .request_plane(slot, self.resolved(&descriptor), axis, index, 1024)
                {
                    self.model.error = Some(error);
                }
            } else {
                self.imaging.clear(slot);
            }
        }
        self.refresh_plane_model();
    }

    pub(super) fn refresh_plane_model(&mut self) {
        self.model.planes.clear();
        for (slot, kind) in [
            (1, PlaneKind::Axial),
            (2, PlaneKind::Coronal),
            (3, PlaneKind::Sagittal),
            (4, PlaneKind::Projection),
            (5, PlaneKind::Sinogram),
        ] {
            if let ImagingStatus::Error(error) = self.imaging.status(slot) {
                self.model.error = Some(error.clone());
            } else if let Some(plane) = self.imaging.plane(slot) {
                self.model.planes.insert(kind, plane_view(plane));
            }
        }
        // Both sides show the requested slice; a side still reading it is not
        // shown, so the pair never mixes two slices under one number.
        let requested = self.ui.display.indices[self.ui.compare_plane.axis().unwrap_or(0)];
        self.model.comparison_planes = [6, 7].map(|slot| {
            self.imaging.plane(slot).map(|plane| {
                let view = plane_view(plane);
                if view.index == requested {
                    view
                } else {
                    // Nothing of the previous slice (pixels, coordinate, statistics)
                    // is shown under the requested slice number.
                    PlaneView {
                        index: requested,
                        count: view.count,
                        ..PlaneView::default()
                    }
                }
            })
        });
        for slot in [6, 7] {
            if self.ui.workspace == Workspace::Compare
                && let ImagingStatus::Error(error) = self.imaging.status(slot)
            {
                self.model.error = Some(error.clone());
            }
        }
        // From the registrations, so the slider's range is right before either
        // side has loaded.
        self.model.comparison_count = self
            .compared_count(self.ui.compare_plane.axis().unwrap_or(0))
            .unwrap_or(0);
        self.model.comparison_index = requested;
        if let Some(volume) = self.imaging.volume(8) {
            let statistics = &volume.statistics;
            let bins = statistics.histogram.len().max(1);
            self.model.planes.insert(
                PlaneKind::Volume,
                PlaneView {
                    available: true,
                    pixel_shape: [volume.shape[1], volume.shape[2]],
                    spacing: [volume.spacing[1], volume.spacing[0]],
                    axes: ["y".into(), "x".into()],
                    units: volume.value_unit.clone(),
                    index: 0,
                    count: 1,
                    quality: if volume.stride == 1 {
                        "Full-resolution scalar texture".into()
                    } else {
                        format!("1:{} preview texture", volume.stride)
                    },
                    range: [statistics.min as f64, statistics.max as f64],
                    histogram: statistics.histogram.clone(),
                    histogram_edges: (0..=bins)
                        .map(|i| {
                            statistics.min as f64
                                + (statistics.max - statistics.min) as f64 * i as f64 / bins as f64
                        })
                        .collect(),
                    statistics: Some(Statistics {
                        min: statistics.min as f64,
                        max: statistics.max as f64,
                        mean: statistics.mean,
                        std: statistics.std,
                        sampled_pixels: statistics.finite_count,
                    }),
                },
            );
        }
        self.model.render_loading = [1, 2, 3, 4, 5, 6, 7, 8].iter().any(|slot| {
            matches!(
                self.imaging.status(*slot),
                ImagingStatus::Loading | ImagingStatus::Refining
            )
        });
        self.model.render_quality = if let Some(volume) = self.imaging.volume(8) {
            format!(
                "GPU volume · {} × {} × {}{}",
                volume.shape[2],
                volume.shape[1],
                volume.shape[0],
                if volume.stride > 1 {
                    format!(" · 1:{} preview", volume.stride)
                } else {
                    " · full resolution".into()
                }
            )
        } else {
            "Native scalar viewer".into()
        };
        if let ImagingStatus::Error(error) = self.imaging.status(8) {
            self.model.error = Some(error.clone());
        }
        let descriptors = [
            self.model.compare_left.as_ref(),
            self.model.compare_right.as_ref(),
        ]
        .map(|id| id.and_then(|id| self.project.as_ref()?.document.asset(id)?.volume()));
        self.model.comparison_note = match descriptors {
            [Some(a), Some(b)]
                if a.shape != b.shape
                    || a.axes != b.axes
                    || a.unit != b.unit
                    || a.domain != b.domain =>
            {
                Some("Grids differ: slices are matched by index, not position".into())
            }
            [Some(_), Some(_)] => None,
            _ => Some("Choose two reconstructed volumes".into()),
        };
        // Split and Difference combine pixels, so they need one shared grid.
        if self.model.comparison_note.is_some() && self.ui.compare_mode != CompareMode::Pair {
            self.ui.compare_mode = CompareMode::Pair;
        }
        if self.ui.workspace == Workspace::Compare
            && self.ui.compare_mode == CompareMode::Difference
            && let (Some(left), Some(right)) = (self.imaging.plane(6), self.imaging.plane(7))
        {
            let ids = [left.id, right.id];
            if self.difference.as_ref().is_none_or(|(old, _)| *old != ids) {
                self.difference = left
                    .difference(right)
                    .map(|plane| (ids, std::sync::Arc::new(plane)));
            }
        } else {
            self.difference = None;
        }
        if self.ui.workspace == Workspace::Compare
            && self.ui.compare_mode == CompareMode::Split
            && let (Some(left), Some(right)) = (self.imaging.plane(6), self.imaging.plane(7))
            && let (Some(left_window), Some(right_window)) =
                (self.compare_window(6), self.compare_window(7))
        {
            let windows = [left_window, right_window];
            let key = (
                [left.id, right.id],
                self.ui.compare_split.to_bits(),
                windows.as_flattened().iter().map(|v| v.to_bits()).collect(),
            );
            if self.split.as_ref().is_none_or(|(old, _)| *old != key) {
                self.split = left
                    .split(right, self.ui.compare_split, windows)
                    .map(|plane| (key, std::sync::Arc::new(plane)));
            }
        } else {
            self.split = None;
        }
        self.prefill_window();
    }

    pub(crate) fn split_ready(&self) -> bool {
        self.split.is_some()
    }

    pub(crate) fn difference_ready(&self) -> bool {
        self.difference.is_some()
    }
    pub(crate) fn compare_window(&self, slot: u32) -> Option<[f32; 2]> {
        let slot = if slot == 7 && self.ui.compare_linked {
            6
        } else {
            slot
        };
        let data = self.imaging.plane(slot)?;
        Some(if slot == 7 {
            data.statistics.window
        } else {
            self.applied_window.unwrap_or(data.statistics.window)
        })
    }

    pub(crate) fn current_window(&self) -> Option<[f32; 2]> {
        if self.ui.workspace == Workspace::Compare {
            return self.compare_window(6);
        }
        self.applied_window.or_else(|| {
            if self.ui.main_view == PlaneKind::Volume {
                self.imaging.volume(8).map(|data| data.statistics.window)
            } else {
                self.imaging
                    .plane(self.ui.main_view.mark())
                    .map(|data| data.statistics.window)
            }
        })
    }
    pub(super) fn prefill_window(&mut self) {
        let Some(previous) = &self.window_prefill else {
            return;
        };
        if self.ui.display.window_low.text != previous[0]
            || self.ui.display.window_high.text != previous[1]
        {
            // An unfinished edit belongs to the user, even when it is invalid.
            self.window_prefill = None;
            return;
        }
        if let Some([low, high]) = self.current_window() {
            let values = window_text(low, high);
            self.ui.display.window_low.replace_text(values[0].clone());
            self.ui.display.window_high.replace_text(values[1].clone());
            self.window_prefill = Some(values);
        }
    }

    pub(super) fn valid_window(&self) -> Result<Option<[f32; 2]>, String> {
        let low = self.ui.display.window_low.text.trim();
        let high = self.ui.display.window_high.text.trim();
        if low.is_empty() && high.is_empty() {
            return Ok(None);
        }
        let low = low
            .parse::<f32>()
            .map_err(|_| "Enter a finite lower window limit")?;
        let high = high
            .parse::<f32>()
            .map_err(|_| "Enter a finite upper window limit")?;
        if !low.is_finite() || !high.is_finite() || low >= high {
            return Err("Display window limits must be finite, with minimum below maximum".into());
        }
        Ok(Some([low, high]))
    }
    fn display_settings(&self) -> DisplaySettings {
        DisplaySettings {
            window: self.applied_window,
            colormap: match self.ui.display.colormap.as_str() {
                "viridis" => ColorMap::Viridis,
                "magma" => ColorMap::Magma,
                "bone" => ColorMap::Bone,
                _ => ColorMap::Gray,
            },
            invert: self.ui.display.invert,
            zoom: self.ui.display.zoom,
            pan: self.ui.display.pan,
        }
    }

    pub fn render_items(&self) -> Vec<RenderItem> {
        let display = self.display_settings();
        let mut items = Vec::new();
        if let Some(frame) = self.live.frame() {
            for (axis, mark) in [("z", 40), ("y", 41), ("x", 42)] {
                if let Some(data) = frame.planes.get(axis) {
                    items.push(RenderItem::Plane {
                        mark,
                        data: data.clone(),
                        display: DisplaySettings {
                            colormap: display.colormap,
                            ..Default::default()
                        },
                    });
                }
            }
        }
        for slot in 1..=7 {
            if let Some(data) = self.imaging.plane(slot) {
                let mut settings = display.clone();
                if slot <= 5 {
                    settings.zoom = 1.;
                    settings.pan = [0.; 2];
                }
                if matches!(slot, 6 | 7) {
                    settings.window = self.compare_window(slot);
                }
                items.push(RenderItem::Plane {
                    mark: slot,
                    data: data.clone(),
                    display: settings,
                });
            }
        }
        if let Some(data) = self.pending_preview_plane() {
            items.push(RenderItem::Plane {
                mark: 11,
                data: data.clone(),
                display: display.clone(),
            });
        }
        if let Some((_, data)) = &self.difference {
            // Symmetric about zero, in grey, so no change reads as mid-grey.
            let reach = data
                .statistics
                .min
                .abs()
                .max(data.statistics.max.abs())
                .max(f32::EPSILON);
            items.push(RenderItem::Plane {
                mark: 10,
                data: data.clone(),
                display: DisplaySettings {
                    window: Some([-reach, reach]),
                    colormap: crate::imaging::ColorMap::Gray,
                    invert: false,
                    ..display.clone()
                },
            });
        }
        if let Some((_, data)) = &self.split {
            items.push(RenderItem::Plane {
                mark: 12,
                data: data.clone(),
                display: DisplaySettings {
                    window: self.compare_window(6),
                    ..display.clone()
                },
            });
        }
        if let Some(data) = self.imaging.plane(self.ui.main_view.mark()) {
            items.push(RenderItem::Plane {
                mark: 9,
                data: data.clone(),
                display: display.clone(),
            });
        }
        if let Some(data) = self.imaging.volume(8) {
            let settings = VolumeSettings {
                yaw: self.ui.display.azimuth,
                pitch: self.ui.display.elevation,
                zoom: self.ui.display.zoom,
                window: display.window,
                colormap: display.colormap,
                invert: display.invert,
                opacity: self.ui.display.density,
                threshold: self.ui.display.threshold,
                lighting: self.ui.display.lighting > 0.,
                diffuse: self.ui.display.lighting,
                clip_max: self.ui.display.clip,
                interacting: self.last_interaction.elapsed() < Duration::from_millis(160),
                ..Default::default()
            };
            items.push(RenderItem::Volume {
                mark: 8,
                data: data.clone(),
                settings,
            });
        }
        items
    }

    pub(super) fn crosshair(&mut self, plane: PlaneKind, position: [f32; 2]) {
        let uv = self.data_uv(position);
        let Some(dimensions) = self.volume_dimensions() else {
            return;
        };
        let Some(data) = self.imaging.plane(plane.mark()) else {
            return;
        };
        for (i, stored) in [1, 0].iter().enumerate() {
            if let Some(axis) = ["z", "y", "x"]
                .iter()
                .position(|name| *name == data.axes[*stored].name)
            {
                let sample =
                    (uv[i] * data.shape[*stored] as f32 - 0.5).max(0.) * data.stride as f32;
                self.ui.display.indices[axis] =
                    (sample.round() as usize).min(dimensions[axis].saturating_sub(1));
            }
        }
    }
    fn data_uv(&self, position: [f32; 2]) -> [f32; 2] {
        std::array::from_fn(|i| {
            ((position[i] - 0.5 - self.ui.display.pan[i]) / self.ui.display.zoom + 0.5)
                .clamp(0., 1.)
        })
    }

    pub(super) fn measure(
        &mut self,
        plane: PlaneKind,
        start: [f32; 2],
        end: [f32; 2],
        finished: bool,
    ) -> Result<(), String> {
        if plane.axis().is_none() {
            return Err("Choose an orthogonal slice to measure".into());
        }
        let data = self
            .imaging
            .plane(plane.mark())
            .ok_or("Wait for the slice to load")?;
        let points = [self.data_uv(start), self.data_uv(end)];
        // Sample-centre coordinates: sample i is centred at i.
        let unclamped = points.map(|p| {
            [
                p[1] as f64 * data.shape[0] as f64 - 0.5,
                p[0] as f64 * data.shape[1] as f64 - 0.5,
            ]
        });
        let last = [data.shape[0], data.shape[1]].map(|n| n.saturating_sub(1) as f64);
        let pixels = unclamped.map(|p| [p[0].clamp(0., last[0]), p[1].clamp(0., last[1])]);
        let (name, value, detail) = match self.ui.inspect_tool {
            InspectTool::Distance => {
                let distance = data.distance(pixels[0], pixels[1])?;
                (
                    "Distance",
                    format!("{distance:.4} {}", crate::ui::unit_name(&data.axes[0].unit)),
                    "Pixel-centre coordinates".into(),
                )
            }
            InspectTool::Region => {
                if !finished {
                    self.model.measurements.retain(|m| m.id != "draft");
                    self.model.measurements.push(MeasurementView {
                        id: "draft".into(),
                        name: "Region".into(),
                        asset: self.model.active_asset.clone().unwrap_or_default(),
                        index: data.index,
                        plane,
                        tool: InspectTool::Region,
                        points,
                        value: "Release to measure".into(),
                        detail: "Statistics use finite scalar samples in this plane".into(),
                    });
                    return Ok(());
                }
                // The samples whose centres lie inside the drawn box.
                let low: [f64; 2] =
                    std::array::from_fn(|i| unclamped[0][i].min(unclamped[1][i]).ceil().max(0.));
                let high: [f64; 2] = std::array::from_fn(|i| {
                    unclamped[0][i].max(unclamped[1][i]).floor().min(last[i])
                });
                if low[0] > high[0] || low[1] > high[1] {
                    return Err("Drag a box that covers at least one sample".into());
                }
                let a = low.map(|v| v as usize);
                let b = high.map(|v| v as usize);
                let (mut n, mut mean, mut m2, mut min, mut max) =
                    (0usize, 0f64, 0f64, f32::INFINITY, f32::NEG_INFINITY);
                for row in a[0]..=b[0].min(data.shape[0] - 1) {
                    for col in a[1]..=b[1].min(data.shape[1] - 1) {
                        if let Some(value) = data.sample(row, col) {
                            n += 1;
                            let delta = value as f64 - mean;
                            mean += delta / n as f64;
                            m2 += delta * (value as f64 - mean);
                            min = min.min(value);
                            max = max.max(value);
                        }
                    }
                }
                if n == 0 {
                    return Err("The ROI contains no finite samples".into());
                }
                (
                    "Region",
                    format!("μ {mean:.5} {}", crate::ui::unit_name(&data.value_unit)),
                    format!(
                        "{} samples · σ {:.5} · range {:.5}–{:.5}{}",
                        n,
                        (m2 / n as f64).sqrt(),
                        min,
                        max,
                        if data.stride > 1 {
                            format!(" · sampled at 1:{}", data.stride)
                        } else {
                            String::new()
                        }
                    ),
                )
            }
            InspectTool::Navigate => return Ok(()),
        };
        self.model.measurements.retain(|m| m.id != "draft");
        self.model.measurements.push(MeasurementView {
            id: if finished {
                project::new_id("measurement")
            } else {
                "draft".into()
            },
            name: name.into(),
            asset: self.model.active_asset.clone().unwrap_or_default(),
            index: data.index,
            plane,
            tool: self.ui.inspect_tool,
            points,
            value,
            detail,
        });
        Ok(())
    }

    pub(super) fn capture_workspace(&self) -> project::Workspace {
        let mut draft_choices = self.ui.parameter_choices.clone();
        draft_choices.insert(
            "studio:prefill".into(),
            self.window_prefill
                .as_ref()
                .is_some_and(|v| {
                    v[0] == self.ui.display.window_low.text
                        && v[1] == self.ui.display.window_high.text
                })
                .to_string(),
        );
        draft_choices.insert("studio:preview_array".into(), self.ui.preview_array.clone());
        draft_choices.insert(
            "studio:compare_linked".into(),
            self.ui.compare_linked.to_string(),
        );
        draft_choices.insert(
            "studio:compare_plane".into(),
            format!("{:?}", self.ui.compare_plane),
        );
        draft_choices.insert(
            "studio:compare_mode".into(),
            format!("{:?}", self.ui.compare_mode),
        );
        draft_choices.insert(
            "studio:corrections_json".into(),
            self.ui.corrections.json_mode.to_string(),
        );
        let mut draft_edits: std::collections::BTreeMap<_, _> = self
            .ui
            .parameter_edits
            .iter()
            .map(|(key, edit)| (key.clone(), edit.text.clone()))
            .chain(
                self.ui
                    .corrections
                    .fields
                    .iter()
                    .map(|(key, edit)| (format!("correction:{key}"), edit.text.clone())),
            )
            .collect();
        draft_edits.insert(
            "studio:window_low".into(),
            self.ui.display.window_low.text.clone(),
        );
        draft_edits.insert(
            "studio:window_high".into(),
            self.ui.display.window_high.text.clone(),
        );
        project::Workspace {
            active: self.model.active_asset.clone(),
            selected: self.model.selected_assets.clone(),
            compare: [
                self.model.compare_left.clone(),
                self.model.compare_right.clone(),
            ],
            view: format!("{:?}", self.ui.main_view),
            workspace: format!("{:?}", self.ui.workspace),
            indices: self.ui.display.indices,
            projection: self.ui.display.projection_index,
            sinogram: self.ui.display.sinogram_row,
            window: self.applied_window,
            colormap: self.ui.display.colormap.clone(),
            invert: self.ui.display.invert,
            camera: [self.ui.display.azimuth, self.ui.display.elevation],
            clip: self.ui.display.clip,
            density: self.ui.display.density,
            threshold: self.ui.display.threshold,
            lighting: self.ui.display.lighting,
            zoom: self.ui.display.zoom,
            pan: self.ui.display.pan,
            operation: self.ui.operation.clone(),
            parameters: forms::parameters(&self.ui, &self.catalog).unwrap_or(Value::Null),
            draft_edits,
            draft_choices,
            measurements: self
                .model
                .measurements
                .iter()
                .filter(|measurement| measurement.id != "draft")
                .map(|measurement| {
                    json!({
                        "id": measurement.id,
                        "name": measurement.name,
                        "asset": measurement.asset,
                        "index": measurement.index,
                        "plane": format!("{:?}", measurement.plane),
                        "tool": format!("{:?}", measurement.tool),
                        "points": measurement.points,
                        "value": measurement.value,
                        "detail": measurement.detail,
                    })
                })
                .collect(),
        }
    }

    pub(super) fn restore_view(&mut self, context: &project::Workspace) {
        self.ui.main_view = parse_plane(&context.view);
        self.ui.display.indices = context.indices;
        self.ui.display.projection_index = context.projection;
        self.ui.display.sinogram_row = context.sinogram;
        self.centre_sinogram = false;
        self.applied_window = context.window;
        self.ui.display.window_low.replace_text("");
        self.ui.display.window_high.replace_text("");
        if let Some([low, high]) = context.window {
            self.ui.display.window_low = TextEdit::new(low.to_string());
            self.ui.display.window_high = TextEdit::new(high.to_string());
        }
        self.window_prefill = Some([
            self.ui.display.window_low.text.clone(),
            self.ui.display.window_high.text.clone(),
        ]);
        if !context.colormap.is_empty() {
            self.ui.display.colormap = context.colormap.clone();
        }
        self.ui.display.invert = context.invert;
        if context.zoom > 0. {
            self.ui.display.zoom = context.zoom;
            self.ui.display.pan = context.pan;
            self.ui.display.azimuth = context.camera[0];
            self.ui.display.elevation = context.camera[1];
            self.ui.display.clip = context.clip;
            self.ui.display.density = context.density;
            self.ui.display.lighting = context.lighting;
            self.ui.display.threshold = context.threshold;
        }
    }

    pub(super) fn restore_workspace(&mut self, context: &project::Workspace) {
        self.model.active_asset = context.active.clone();
        self.model.selected_assets = context.selected.clone();
        self.model.compare_left = context.compare[0].clone();
        self.model.compare_right = context.compare[1].clone();
        self.ui.workspace = match context.workspace.as_str() {
            "Prepare" => Workspace::Prepare,
            "Reconstruct" => Workspace::Reconstruct,
            "Compare" => Workspace::Compare,
            "Jobs" => Workspace::Jobs,
            _ => Workspace::Explore,
        };
        self.restore_view(context);
        self.set_operation(
            if context.operation.is_empty() {
                "reconstruct"
            } else {
                &context.operation
            },
            Some(&context.parameters),
        );
        for (name, value) in &context.draft_edits {
            if let Some(edit) = self.ui.parameter_edits.get_mut(name) {
                edit.replace_text(value.clone());
            }
        }
        crate::ui::corrections::reset(&mut self.ui, &self.catalog);
        for (name, value) in &context.draft_edits {
            if let Some(edit) = name
                .strip_prefix("correction:")
                .and_then(|key| self.ui.corrections.fields.get_mut(key))
            {
                edit.replace_text(value.clone());
            }
        }
        // Compare shows orthogonal slices only.
        self.ui.compare_plane = context
            .draft_choices
            .get("studio:compare_plane")
            .map(|plane| parse_plane(plane))
            .filter(|plane| plane.axis().is_some())
            .unwrap_or(PlaneKind::Axial);
        self.ui.compare_mode = match context
            .draft_choices
            .get("studio:compare_mode")
            .map(String::as_str)
        {
            Some("Split") => CompareMode::Split,
            Some("Difference") => CompareMode::Difference,
            _ => CompareMode::Pair,
        };
        // Corrections edited as JSON stay in JSON, so its text is what runs.
        if context
            .draft_choices
            .get("studio:corrections_json")
            .map(String::as_str)
            == Some("true")
        {
            self.ui.corrections.json_mode = true;
        }
        for (name, value) in &context.draft_choices {
            if let Some(choice) = self.ui.parameter_choices.get_mut(name) {
                *choice = value.clone();
            }
        }
        if let Some(value) = context.draft_edits.get("studio:window_low") {
            self.ui.display.window_low.replace_text(value.clone());
        }
        if let Some(value) = context.draft_edits.get("studio:window_high") {
            self.ui.display.window_high.replace_text(value.clone());
        }
        if context
            .draft_choices
            .get("studio:prefill")
            .map_or_else(|| self.valid_window().is_ok(), |v| v == "true")
        {
            self.window_prefill = Some([
                self.ui.display.window_low.text.clone(),
                self.ui.display.window_high.text.clone(),
            ]);
        }
        if let Some(value) = context.draft_choices.get("studio:preview_array")
            && matches!(value.as_str(), "before" | "after")
        {
            self.ui.preview_array = value.clone();
        }
        if let Some(value) = context.draft_choices.get("studio:compare_linked")
            && let Ok(linked) = value.parse::<bool>()
        {
            self.ui.compare_linked = linked;
        }
        self.model.measurements = context
            .measurements
            .iter()
            .filter_map(|m| {
                Some(MeasurementView {
                    id: m["id"].as_str()?.into(),
                    name: m["name"].as_str()?.into(),
                    asset: m["asset"].as_str()?.into(),
                    index: m["index"].as_u64()? as usize,
                    plane: parse_plane(m["plane"].as_str()?),
                    tool: if m["tool"] == "Region" {
                        InspectTool::Region
                    } else {
                        InspectTool::Distance
                    },
                    points: serde_json::from_value(m["points"].clone()).ok()?,
                    value: m["value"].as_str()?.into(),
                    detail: m["detail"].as_str()?.into(),
                })
            })
            .collect();
    }

    /// Open Activity as the user's choice, over the jobs it shows now.
    pub(crate) fn show_activity(&mut self) {
        let jobs = self
            .project
            .iter()
            .flat_map(|project| &project.document.jobs)
            .filter(|job| job.status.active())
            .map(|job| job.id.clone())
            .collect();
        self.ui.activity = Activity::Open {
            jobs,
            closes_itself: false,
        };
    }

    /// Show Activity for newly queued jobs; when it opens for them, it closes once they
    /// succeed.
    pub(super) fn open_activity(&mut self, queued: Vec<String>) {
        match &mut self.ui.activity {
            Activity::Open { jobs, .. } => jobs.extend(queued),
            Activity::Closed => {
                self.ui.activity = Activity::Open {
                    jobs: queued,
                    closes_itself: true,
                }
            }
        }
    }

    pub(super) fn refresh_model(&mut self) {
        self.model.preview_choices = if self.preview_context.active
            == Some(crate::previews::PreviewKind::Corrections)
            || self
                .model
                .active_asset
                .as_ref()
                .and_then(|id| self.project.as_ref()?.document.asset(id))
                .is_some_and(|asset| asset.kind == "correction_preview")
        {
            vec![
                ("before".into(), "Before correction".into()),
                ("after".into(), "After correction".into()),
            ]
        } else {
            Vec::new()
        };
        self.model.project = self.project.as_ref().map(|p| ProjectView {
            name: p.document.name.clone(),
            path: p.root.to_string_lossy().into(),
            dirty: p.dirty(),
            saving: p.dirty() && p.save_error.is_none(),
            save_error: p.save_error.clone(),
        });
        let names = &mut self.recent_names;
        self.model.recent_projects = self
            .preferences
            .recent
            .iter()
            .map(|p| RecentProject {
                name: names
                    .entry(p.clone())
                    .or_insert_with(|| {
                        project::manifest_name(p).unwrap_or_else(|| {
                            p.file_name().unwrap_or_default().to_string_lossy().into()
                        })
                    })
                    .clone(),
                path: p.to_string_lossy().into(),
                available: p.join(project::MANIFEST).is_file(),
            })
            .collect();
        self.model.assets = self
            .project
            .as_ref()
            .map(|p| {
                p.document
                    .assets
                    .iter()
                    .map(|a| {
                        let array = a
                            .volume()
                            .or_else(|| a.projections())
                            .or_else(|| a.arrays.values().find(|array| array.shape.len() >= 2));
                        let shape = array.map(|a| a.shape.clone()).unwrap_or_default();
                        let voxel_pitch = a.volume().and_then(|volume| {
                            let pitch = |name: &str| {
                                let axis = volume.axes.iter().find(|axis| axis.name == name)?;
                                if volume.axes.iter().any(|other| other.unit != axis.unit) {
                                    return None;
                                }
                                match axis.coordinates {
                                    Coordinates::Uniform { spacing, .. } => Some(spacing.abs()),
                                    _ => None,
                                }
                            };
                            Some([pitch("x")?, pitch("y")?, pitch("z")?])
                        });
                        AssetView {
                            export: match a.kind.as_str() {
                                "reconstruction" | "simulation" => Export::Volume,
                                "alignment" if a.volume().is_some() => Export::Volume,
                                "scan" | "alignment" => Export::Dataset,
                                "frames" => Export::Unavailable,
                                _ => Export::Array,
                            },
                            id: a.id.clone(),
                            name: a.name.clone(),
                            kind: if a.volume().is_some() {
                                AssetKind::Volume
                            } else if a.projections().is_some()
                                || matches!(a.kind.as_str(), "scan" | "frames")
                            {
                                AssetKind::Scan
                            } else {
                                AssetKind::Other
                            },
                            parent: a.parent.clone(),
                            path: p.resolve(&a.path).to_string_lossy().into(),
                            shape,
                            voxel_pitch,
                            units: array
                                .and_then(|a| a.axes.last())
                                .map(|a| a.unit.clone())
                                .unwrap_or_default(),
                            missing: !p.resolve(&a.path).exists(),
                            status: a.issue.clone().unwrap_or_else(|| {
                                if a.arrays.is_empty() {
                                    "Reading metadata"
                                } else {
                                    "Ready"
                                }
                                .into()
                            }),
                            description: format!(
                                "{} · {}",
                                a.kind,
                                if a.managed {
                                    "Stored in project"
                                } else {
                                    "Referenced in place"
                                }
                            ),
                            method: a.method.clone(),
                            simulated: a.simulated,
                            operation: if a.is_result() {
                                p.document
                                    .jobs
                                    .iter()
                                    .find(|job| job.result.as_deref() == Some(&a.id))
                                    .map(|job| job.operation.clone())
                                    .or_else(|| a.provenance.as_ref().map(|p| p.operation.clone()))
                            } else {
                                None
                            },
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        self.model.jobs =
            self.project
                .as_ref()
                .map(|p| {
                    p.document
                        .jobs
                        .iter()
                        .map(|j| JobView {
                            id: j.id.clone(),
                            name: j
                                .result
                                .as_ref()
                                .and_then(|id| p.document.asset(id))
                                .map(|asset| asset.name.clone())
                                .unwrap_or_else(|| j.subject()),
                            inputs: j
                                .inputs
                                .iter()
                                .map(|input| input.name.as_str())
                                .collect::<Vec<_>>()
                                .join(", "),
                            operation: j.display_operation(),
                            status: match j.status {
                                project::JobStatus::Queued => JobStatus::Queued,
                                project::JobStatus::Running => JobStatus::Running,
                                project::JobStatus::Paused => JobStatus::Paused,
                                project::JobStatus::Completed => JobStatus::Completed,
                                project::JobStatus::Failed => JobStatus::Failed,
                                project::JobStatus::Cancelled => JobStatus::Cancelled,
                                project::JobStatus::Interrupted => JobStatus::Interrupted,
                            },
                            stage: j.stage.clone(),
                            progress: j.progress,
                            error: j.error.clone(),
                            settings: j
                                .parameters
                                .as_object()
                                .map(|o| {
                                    o.iter()
                                        .map(|(k, v)| {
                                            (
                                                k.clone(),
                                                v.as_str()
                                                    .map(str::to_owned)
                                                    .unwrap_or_else(|| v.to_string()),
                                            )
                                        })
                                        .collect()
                                })
                                .unwrap_or_default(),
                            elapsed: j.started.map(|s| {
                                format!(
                                    "{}{:.1} s",
                                    if j.kept_preview.is_some() {
                                        "save "
                                    } else {
                                        ""
                                    },
                                    (j.finished.unwrap_or_else(project::now_ms).saturating_sub(s))
                                        as f64
                                        / 1000.
                                )
                            }),
                            result_asset: j.result.clone(),
                            can_pause: matches!(
                                j.status,
                                project::JobStatus::Running | project::JobStatus::Paused
                            ) && !self.local_job(&j.id)
                                && self.pending.values().any(
                                    |pending| matches!(pending,Pending::Execute(id) if id==&j.id),
                                ),
                        })
                        .collect()
                })
                .unwrap_or_default();
        self.model.recipes = self
            .project
            .as_ref()
            .map(|p| {
                p.document
                    .recipes
                    .iter()
                    .map(|r| RecipeView {
                        id: r.id.clone(),
                        name: r.name.clone(),
                        operation: r.operation.clone(),
                        summary: r
                            .parameters
                            .as_object()
                            .map(|o| {
                                o.iter()
                                    .take(3)
                                    .map(|(k, v)| format!("{k}: {v}"))
                                    .collect::<Vec<_>>()
                                    .join(" · ")
                            })
                            .unwrap_or_default(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        // Open Activity keeps every job it has shown, so a failure stays listed.
        if let Activity::Open { jobs, .. } = &mut self.ui.activity
            && let Some(project) = &self.project
        {
            for job in project
                .document
                .jobs
                .iter()
                .filter(|job| job.status.active())
            {
                if !jobs.contains(&job.id) {
                    jobs.push(job.id.clone());
                }
            }
        }
        if let Activity::Open {
            closes_itself: true,
            ..
        } = &self.ui.activity
            && let Some(project) = &self.project
        {
            let opened_for = || {
                project
                    .document
                    .jobs
                    .iter()
                    .filter(|job| self.ui.activity.tracks(&job.id))
            };
            // A failure stays listed, with its error, until the user closes Activity.
            if opened_for().all(|job| !job.status.active())
                && !opened_for().any(|job| job.status == project::JobStatus::Failed)
            {
                self.ui.activity = Activity::Closed;
            }
        }
        self.model.commands = crate::preferences::COMMANDS
            .iter()
            .map(|(id, title, category, default)| CommandView {
                id: (*id).into(),
                title: (*title).into(),
                category: (*category).into(),
                binding: rune::Chord::parse(self.preferences.binding(id, default))
                    .map(|c| c.label())
                    .unwrap_or_default(),
                enabled: match *id {
                    "queue" => self.project.is_some() && self.model.backend.ready,
                    "compare" => self.project.is_some(),
                    "export" => self.model.active_asset.is_some(),
                    "import" | "folder" => self.project.is_some(),
                    _ => true,
                },
            })
            .chain(crate::operations::GROUPS.iter().flat_map(|group| {
                group.operations.iter().map(|operation| CommandView {
                    id: format!("operation:{}", operation.id),
                    title: operation.title.into(),
                    category: group.title.into(),
                    binding: String::new(),
                    enabled: self.project.is_some(),
                })
            }))
            .collect();
        self.model.shortcuts = crate::preferences::COMMANDS
            .iter()
            .map(|(id, title, category, default)| ShortcutView {
                id: (*id).into(),
                title: (*title).into(),
                category: (*category).into(),
                binding: self.preferences.binding(id, default).into(),
                default_binding: (*default).into(),
            })
            .collect();
        self.model.can_undo = !self.view_history.is_empty();
        self.model.can_redo = !self.view_future.is_empty();
    }
}

fn parse_plane(value: &str) -> PlaneKind {
    match value {
        "Axial" => PlaneKind::Axial,
        "Coronal" => PlaneKind::Coronal,
        "Sagittal" => PlaneKind::Sagittal,
        "Projection" => PlaneKind::Projection,
        "Sinogram" => PlaneKind::Sinogram,
        _ => PlaneKind::Volume,
    }
}

fn registered_plane_kind(array: &ArrayDescriptor) -> Option<PlaneKind> {
    array
        .fixed
        .iter()
        .find_map(|fixed| match fixed.name.as_str() {
            "z" => Some(PlaneKind::Axial),
            "y" => Some(PlaneKind::Coronal),
            "x" => Some(PlaneKind::Sagittal),
            _ => None,
        })
}

fn plane_view(plane: &crate::imaging::PlaneData) -> PlaneView {
    let statistics = &plane.statistics;
    let bins = statistics.histogram.len().max(1);
    let same_units =
        plane.axes[0].unit == plane.axes[1].unit && plane.axes[0].frame == plane.axes[1].frame;
    PlaneView {
        available: true,
        pixel_shape: plane.shape,
        spacing: if same_units {
            [plane.axes[0].spacing.abs(), plane.axes[1].spacing.abs()]
        } else {
            [1., 1.]
        },
        axes: [plane.axes[0].name.clone(), plane.axes[1].name.clone()],
        units: plane.value_unit.clone(),
        index: plane.index,
        count: plane.count,
        // Position first: it is what changes as the slice moves.
        quality: plane
            .fixed
            .iter()
            .map(|fixed| {
                format!(
                    "{} = {:.4} {}",
                    fixed.name,
                    fixed.coordinate,
                    crate::ui::unit_name(&fixed.unit)
                )
            })
            .chain([
                plane.domain.replace('_', " "),
                if plane.stride == 1 {
                    "full resolution".into()
                } else {
                    format!("preview at 1:{} sampling", plane.stride)
                },
            ])
            .collect::<Vec<_>>()
            .join(" · "),
        range: [statistics.min as f64, statistics.max as f64],
        histogram: statistics.histogram.clone(),
        histogram_edges: (0..=bins)
            .map(|i| {
                statistics.min as f64
                    + (statistics.max - statistics.min) as f64 * i as f64 / bins as f64
            })
            .collect(),
        statistics: Some(Statistics {
            min: statistics.min as f64,
            max: statistics.max as f64,
            mean: statistics.mean,
            std: statistics.std,
            sampled_pixels: statistics.finite_count,
        }),
    }
}

/// A display window as field text: rounded for reading while the rounded bounds stay in
/// order, else the shortest text that reads back as each exact bound.
pub(crate) fn window_text(low: f32, high: f32) -> [String; 2] {
    let rounded = [low, high].map(|v| crate::ui::compact_number(f64::from(v)));
    let read = |text: &str| text.parse::<f32>().unwrap_or(f32::NAN);
    if low >= high || read(&rounded[0]) < read(&rounded[1]) {
        rounded
    } else {
        [low.to_string(), high.to_string()]
    }
}
