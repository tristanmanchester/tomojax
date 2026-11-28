use super::*;
use crate::protocol::{ArrayDescriptor, Coordinates, Storage};

impl AppState {
    pub(super) fn resolved(&self, descriptor: &ArrayDescriptor) -> ArrayDescriptor {
        let mut descriptor = descriptor.clone();
        let resolve = |storage: &mut Storage| match storage {
            Storage::Hdf5 { path, .. } | Storage::Npy { path, .. } => {
                if let Some(project) = &self.project {
                    *path = project
                        .resolve_storage(Path::new(path))
                        .to_string_lossy()
                        .into();
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
    pub(super) fn ensure_image_worker(&mut self) {
        if self.engine.is_none()
            && self
                .project
                .as_ref()
                .is_some_and(|p| !p.document.assets.is_empty())
        {
            self.start_engine();
        }
    }

    pub(super) fn configure_planes(&mut self) {
        if let Some(value) = std::env::var_os("TOMOJAX_STUDIO_SLOWLINK") {
            match serde_json::from_str::<Vec<String>>(&value.to_string_lossy()) {
                Ok(relay) => {
                    self.relay = Some(relay);
                }
                Err(error) => {
                    self.model.error = Some(format!(
                        "Invalid TOMOJAX_STUDIO_SLOWLINK argument list: {error}"
                    ))
                }
            }
        }
    }
    pub(crate) fn remote_project(&self) -> Option<&str> {
        self.project
            .as_ref()?
            .document
            .connection
            .as_deref()
            .filter(|name| *name != "local")
    }

    pub(super) fn native_data_available(&self) -> bool {
        self.remote_project().is_none() || self.engine.as_ref().is_some()
    }

    pub(super) fn select_project_source(&mut self) {
        self.jobs = JobSession::default();
        self.pending.clear();
        self.imports.clear();
        self.pane_edges = [1024; 8];
        self.imaging.invalidate();
        if self.remote_project().is_some() || self.engine.owner().is_some() {
            self.previews.disconnected();
            self.live.disconnected();
            self.imaging.detach();
            self.model.backend = Default::default();
        }
        if let Some(name) = self.remote_project() {
            self.engine = match session::Session::named(name, self.wake.clone()) {
                Ok(session) => session,
                Err(error) => {
                    self.model.error = Some(error);
                    session::Session::Closed
                }
            };
        } else {
            if self.engine.owner().is_some() {
                self.engine = session::Session::Closed;
            }
            if self.engine.is_none() {
                self.start_engine();
            }
        }
    }

    pub(super) fn preload_planes(&mut self) {
        if self.engine.is_none() {
            self.start_engine();
        }
        self.refresh_images();
        if self.remote_project().is_some() && self.engine.is_none() {
            return;
        }
        if self.remote_project().is_some() {
            let until = Instant::now() + Duration::from_secs(30);
            while Instant::now() < until {
                self.poll();
                if self.model.error.is_some()
                    || self.files.listing.is_some()
                    || (!self.data_offline()
                        && self.metadata_pending == 0
                        && (1..=8).all(|slot| {
                            !matches!(
                                self.imaging.status(slot),
                                crate::imaging::ImagingStatus::Loading
                                    | crate::imaging::ImagingStatus::Refining
                            )
                        }))
                {
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            return;
        }
        let until = Instant::now() + Duration::from_secs(30);
        while Instant::now() < until {
            if let Some(engine) = self.engine.as_mut() {
                for event in engine.poll() {
                    match event {
                        EngineEvent::Reply { response, stale } => {
                            self.imaging.worker_reply(response, stale);
                        }
                        EngineEvent::Disconnected { reason, .. } => {
                            self.model.error = Some(reason);
                            self.imaging.disconnected();
                            return;
                        }
                        _ => {}
                    }
                }
            }
            self.dispatch_planes();
            if (1..=8).all(|slot| {
                !matches!(
                    self.imaging.status(slot),
                    crate::imaging::ImagingStatus::Loading
                        | crate::imaging::ImagingStatus::Refining
                )
            }) {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        self.refresh_plane_model();
        self.last_autosave = Instant::now();
    }

    pub(crate) fn probe_text(&self, mark: u32, kind: PlaneKind) -> Option<String> {
        let (active, position) = self.ui.probe?;
        if active != mark {
            return None;
        }
        let uv = self.display_settings().source_uv(position);
        if !uv.iter().all(|x| (0. ..=1.).contains(x)) {
            return None;
        }
        let data = match mark {
            9 => self.imaging.plane(kind.mark())?,
            10 => &self.difference.as_ref()?.1,
            12 => {
                let composite = &self.split.as_ref()?.1;
                let layer = composite.layer_at(uv);
                if !layer.lossless {
                    return None;
                }
                let col = layer.texel(uv)[1];
                let edge = layer.split_column(self.ui.compare_split);
                let side = self.imaging.plane(if col < edge { 6 } else { 7 })?;
                let samples = if std::ptr::eq(layer, composite.as_ref()) {
                    side.as_ref()
                } else {
                    side.detail.as_deref()?
                };
                let [row, column] = samples.texel(uv);
                let value = samples.sample(row, column)?;
                let [r, c] = composite.displayed_index(uv);
                return Some(probe_label(samples, [r, c], value));
            }
            slot => self.imaging.plane(slot)?,
        };
        let ([row, col], value) = data.probe(uv)?;
        Some(probe_label(data, [row, col], value))
    }

    pub(super) fn start_engine(&mut self) {
        if self.remote_project().is_some() {
            return;
        }

        let python = std::env::var_os("TOMOJAX_STUDIO_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(if cfg!(windows) { "python" } else { "python3" }));
        match Engine::spawn_through(
            python,
            self.relay.as_deref().unwrap_or_default(),
            self.wake.clone(),
        ) {
            Ok(engine) => {
                self.imaging.bind(&engine);
                self.live.bind(engine.files());
                self.engine = session::Session::local(engine);
            }
            Err(error) => self.model.error = Some(error),
        }
        self.model.backend.status = "Starting TomoJAX".into();
    }

    pub fn render_panes(&mut self, sizes: &[(u32, f32)]) -> Vec<crate::volume::RenderItem> {
        let mut edges = [0; 8];
        let mut zoomed = [false; 8];
        for &(mark, length) in sizes {
            let edge = (length.ceil().clamp(256., 2048.) as usize).next_power_of_two();
            let mut pane = |slot: usize, interactive: bool| {
                if slot < edges.len() {
                    edges[slot] = edges[slot].max(edge);
                    zoomed[slot] |= interactive;
                }
            };
            match mark {
                1..=7 => pane(mark as usize, mark >= 6),
                9 => pane(self.ui.main_view.mark() as usize, true),
                10 | 12 => {
                    pane(6, true);
                    pane(7, true);
                }
                _ => {}
            }
        }
        let mut resized = false;
        for (old, new) in self.pane_edges.iter_mut().zip(edges) {
            if new != 0 && *old != new {
                *old = new;
                resized = true;
            }
        }
        if resized {
            self.refresh_images();
        }
        if !self.data_offline() {
            let display = self.display_settings();
            let mut changed = false;
            for (slot, active) in zoomed.into_iter().enumerate().skip(1) {
                changed |= self
                    .imaging
                    .view_region(slot as u32, active.then_some(&display));
            }
            if changed {
                self.refresh_plane_model();
            }
        }
        self.render_items()
    }

    pub(super) fn view_plane(
        &mut self,
        slot: u32,
        descriptor: ArrayDescriptor,
        axis: usize,
        index: usize,
        edge: usize,
    ) -> Result<(), String> {
        if self.data_offline() {
            self.imaging.retain_source(slot, &descriptor, Some(axis));
            Ok(())
        } else {
            self.imaging
                .request_plane(slot, descriptor, axis, index, edge)
        }
    }

    pub(super) fn request_native_volume(&mut self, volume: crate::protocol::ArrayDescriptor) {
        if self.native_data_available() {
            use studio_core::pixels::Encoding;
            let encoding = if self.relay.is_none() && self.remote_project().is_none() {
                Encoding::Float32
            } else {
                Encoding::Uint16
            };
            if let Err(error) = self.imaging.request_volume(8, volume, 192, encoding) {
                self.model.error = Some(error);
            }
        } else {
            self.imaging.retain_source(8, &volume, None);
        }
    }

    pub(super) fn dispatch_planes(&mut self) -> bool {
        let Some(engine) = self.engine.as_mut() else {
            return false;
        };
        let mut windows = [self.window.applied(); 8];
        if !self.ui.compare_linked {
            windows[7] = None;
        }
        self.imaging.dispatch(engine, &windows)
    }
}

pub(super) fn plane_view(plane: &crate::imaging::PlaneData) -> PlaneView {
    let visible = plane.detail.as_deref().unwrap_or(plane);
    let statistics = &visible.statistics;
    let bins = statistics.histogram.len().max(1);
    let same_units =
        plane.axes[0].unit == plane.axes[1].unit && plane.axes[0].frame == plane.axes[1].frame;
    let mut view = PlaneView {
        available: true,
        pixel_shape: plane.source_shape,
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
            .filter_map(|fixed| crate::ui::coordinate(&fixed.name, fixed.coordinate, &fixed.unit))
            .chain([
                plane.domain.replace('_', " "),
                if visible.stride == 1 {
                    "full resolution".into()
                } else {
                    format!("preview at 1:{} sampling", visible.stride)
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
    };
    if statistics.finite_count == 0 || !visible.lossless {
        view.histogram.clear();
        view.histogram_edges.clear();
        view.statistics = None;
        if !visible.lossless {
            view.quality.push_str(" · moving preview");
        }
    }
    view
}

fn probe_label(data: &crate::imaging::PlaneData, [row, col]: [usize; 2], value: f32) -> String {
    format!(
        "{}[{row}]  {}[{col}]   {value:.5} {}",
        data.axes[0].name,
        data.axes[1].name,
        crate::ui::unit_name(&data.value_unit)
    )
}
