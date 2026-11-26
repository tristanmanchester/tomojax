//! Image-led exploration. The compositor owns all scientific pixels and GPU work.
use rune::{El, Phase, PointerButton, Shape, canvas, col, el, row};

use super::{controls::*, model::*, tokens::*};
use crate::AppState;

pub fn workspace(state: &AppState) -> El {
    if state.model.assets.is_empty() {
        return empty_project(state);
    }
    if state.ui.workspace == Workspace::Compare {
        return compare(state);
    }
    let mut centre = col()
        .key("image-workspace")
        .w_full()
        .h_full()
        .min_w(0.)
        .min_h(0.)
        .padding(20.)
        .gap(14.);
    centre = centre.child(heading(state));
    if state.model.preview.is_some() {
        centre = centre.child(super::preview::status_bar(state));
    }
    let kind = state.ui.main_view;
    let mut main = col()
        .grow(1.)
        .min_w(0.)
        .min_h(180.)
        .radius(10.)
        .clip_rounded(10.)
        .fill(canvas_bg());
    main = main.child(image_toolbar(state, kind)).child(image(
        state,
        kind,
        if kind == PlaneKind::Volume { 8 } else { 9 },
        true,
    ));
    main = main.child(image_footer(state, kind));
    centre = centre.child(main);
    let candidates = if state
        .model
        .planes
        .get(&PlaneKind::Axial)
        .is_some_and(|p| p.available)
    {
        vec![
            PlaneKind::Axial,
            PlaneKind::Coronal,
            PlaneKind::Sagittal,
            PlaneKind::Projection,
            PlaneKind::Sinogram,
        ]
    } else {
        vec![PlaneKind::Projection, PlaneKind::Sinogram]
    };
    let mut strip = row()
        .key("linked-views")
        .gap(10.)
        .h(142.)
        .min_h(112.)
        .w_full();
    let mut count = 0;
    for thumb in candidates {
        if state.model.planes.get(&thumb).is_none_or(|p| !p.available) {
            continue;
        }
        count += 1;
        let selected = thumb == kind;
        strip = strip.child(
            col()
                .key(format!("thumbnail-{thumb:?}"))
                .role("button")
                .label(format!("Explore {} view", thumb.title()))
                .focusable(true)
                .focus_selected(selected)
                .grow(1.)
                .min_w(0.)
                .h_full()
                .radius(7.)
                .clip_rounded(7.)
                .fill(canvas_bg())
                .focus_ring(2., 2., forest())
                .border(if selected { 2. } else { 0. }, forest())
                .on_press_tx(move |s: &mut AppState, tx| {
                    s.dispatch(UiAction::SetMainView(thumb), tx)
                })
                .child(image(state, thumb, thumb.mark(), false))
                .child(
                    col()
                        .h(39.)
                        .w_full()
                        .px(10.)
                        .justify_center()
                        .fill(sage())
                        .child(
                            strong(thumb.title(), 11., forest())
                                .shrink()
                                .min_w(0.)
                                .truncate()
                                .help(axes_label(state, thumb)),
                        )
                        .child(number(
                            state.model.planes.get(&thumb).map_or(String::new(), |p| {
                                format!("{} / {}", p.index + 1, p.count)
                            }),
                            11.,
                            muted(),
                        )),
                ),
        );
    }
    if count > 1 {
        centre = centre.child(strip);
    }
    centre
}

fn heading(state: &AppState) -> El {
    let Some(asset) = state.model.active() else {
        return col()
            .min_h(60.)
            .child(strong("Select a scan or result", 23., ink()));
    };
    let mut details = Vec::new();
    if let Some(method) = asset.method.as_deref().map(crate::operations::method_name)
        && !asset.name.split(" · ").any(|part| part == method)
    {
        details.push(method);
    }
    let dimensions = [PlaneKind::Sagittal, PlaneKind::Coronal, PlaneKind::Axial]
        .map(|plane| state.model.planes.get(&plane).map_or(0, |p| p.count));
    if asset.kind == AssetKind::Volume && dimensions.iter().all(|&n| n > 0) {
        details.push(format!(
            "{} × {} × {} voxels",
            dimensions[0], dimensions[1], dimensions[2]
        ));
    } else if asset.kind == AssetKind::Scan
        && let [views, rows, columns] = asset.shape[..]
    {
        // Projection stacks are (view, v, u), as `tomojax inspect` reports them.
        details.push(format!("{views} views of {rows} × {columns} pixels"));
    } else if !asset.shape.is_empty() {
        details.push(
            asset
                .shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(" × "),
        );
    }
    if let Some([x, y, z]) = asset.voxel_pitch {
        let pitch = if x == y && y == z {
            compact_number(x)
        } else {
            format!(
                "{} × {} × {}",
                compact_number(x),
                compact_number(y),
                compact_number(z)
            )
        };
        // "128 × 128 × 128 voxels of 1 mm": the grid and its pitch read as one fact.
        let unit = unit_name(&asset.units);
        match details.last_mut() {
            Some(grid) if grid.ends_with(" voxels") => {
                grid.push_str(&format!(" of {pitch} {unit}"))
            }
            _ => details.push(format!("Voxels of {pitch} {unit}")),
        }
    }
    if asset.simulated {
        details.push("simulated".into());
    }
    let mut title = row().w_full().gap(14.).items_center().child(
        col()
            .gap(6.)
            .grow(1.)
            .min_w(0.)
            .child(strong(&asset.name, 24., ink()).truncate().help(&asset.name))
            .child(
                label(details.join("  ·  "), 11.5, muted())
                    .truncate()
                    .help(&asset.description),
            ),
    );
    if asset.missing {
        title = title.child(
            action_button(
                "relink-active",
                "Reconnect data",
                false,
                true,
                UiAction::RelinkAsset(asset.id.clone()),
            )
            .child(icon(Icon::Link, 14., muted())),
        );
    } else if !asset.status.is_empty() {
        title = title.child(
            row()
                .gap(7.)
                .items_center()
                .child(status_dot(forest()))
                .child(label(&asset.status, 11., forest())),
        );
    }
    title.min_h(58.)
}

fn image_toolbar(state: &AppState, kind: PlaneKind) -> El {
    let mut toolbar = row()
        .h(42.)
        .w_full()
        .px(14.)
        .gap(10.)
        .items_center()
        .fill(canvas_bg())
        .child(strong(kind.title(), 12., canvas_ink()))
        .child(label(axes_label(state, kind), 11., canvas_muted()))
        .child(spacer());
    let volume_available = state
        .model
        .planes
        .get(&PlaneKind::Volume)
        .is_some_and(|p| p.available);
    if volume_available {
        let volume = kind == PlaneKind::Volume;
        toolbar = toolbar.child(canvas_segments(vec![
            Segment::new(
                "show-volume",
                "3D",
                volume,
                UiAction::SetMainView(PlaneKind::Volume),
            ),
            Segment::new(
                "show-slice",
                "Slice",
                !volume,
                UiAction::SetMainView(PlaneKind::Axial),
            ),
        ]));
    }
    toolbar
        .child(tinted_icon_button(
            "undo-view",
            "Undo display change",
            Icon::Undo,
            state.model.can_undo,
            canvas_muted(),
            |s, tx| s.dispatch(UiAction::Undo, tx),
        ))
        .child(tinted_icon_button(
            "redo-view",
            "Redo display change",
            Icon::Redo,
            state.model.can_redo,
            canvas_muted(),
            |s, tx| s.dispatch(UiAction::Redo, tx),
        ))
        .child(zoom_controls(state))
}

/// One option of a segmented control drawn on the dark canvas.
struct Segment {
    key: String,
    title: String,
    selected: bool,
    enabled: bool,
    action: UiAction,
}

impl Segment {
    fn new(
        key: impl Into<String>,
        title: impl Into<String>,
        selected: bool,
        action: UiAction,
    ) -> Self {
        Self {
            key: key.into(),
            title: title.into(),
            selected,
            enabled: true,
            action,
        }
    }
    fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

fn canvas_segments(segments: Vec<Segment>) -> El {
    let mut group = row().gap(2.).role("radiogroup");
    for segment in segments {
        let Segment {
            key,
            title,
            selected,
            enabled,
            action,
        } = segment;
        group = group.child(
            el().key(key)
                .role("radio")
                .label(&title)
                .checked(selected)
                .focusable(enabled)
                .disabled(!enabled)
                .focus_ring(1.5, 0., canvas_muted())
                .h(28.)
                .px(8.)
                .center()
                .radius(4.)
                .fill(if selected { forest() } else { canvas_bg() })
                .hover(canvas_ink().alpha(0.06))
                .opacity(if enabled { 1. } else { 0.4 })
                .on_press_tx(move |s: &mut AppState, tx| {
                    if enabled {
                        s.dispatch(action.clone(), tx)
                    }
                })
                .child(strong(title, 11., canvas_ink())),
        );
    }
    group
}

fn zoom_controls(state: &AppState) -> El {
    row()
        .gap(10.)
        .items_center()
        .child(tinted_icon_button(
            "zoom-out",
            "Zoom out",
            Icon::Minus,
            true,
            canvas_muted(),
            |s, tx| s.dispatch(UiAction::Zoom(0.8), tx),
        ))
        .child(
            number(
                format!("{:.0}%", state.shown_zoom() * 100.),
                11.,
                canvas_muted(),
            )
            .min_w(34.),
        )
        .child(tinted_icon_button(
            "zoom-in",
            "Zoom in",
            Icon::Plus,
            true,
            canvas_muted(),
            |s, tx| s.dispatch(UiAction::Zoom(1.25), tx),
        ))
        .child(tinted_icon_button(
            "fit-image",
            "Fit image",
            Icon::Fit,
            true,
            canvas_muted(),
            |s, tx| s.dispatch(UiAction::FitView, tx),
        ))
}

pub fn image(state: &AppState, kind: PlaneKind, mark: u32, interactive: bool) -> El {
    let comparison = matches!(mark, 6 | 7 | 10 | 12);
    let info = if comparison {
        state.model.comparison_planes[usize::from(mark == 7)].as_ref()
    } else {
        state.model.planes.get(&kind)
    };
    let available = info.is_some_and(|p| p.available)
        && (mark != 10 || state.difference_ready())
        && (mark != 12 || state.split_ready());
    let mut area = el()
        .key(format!("canvas-area-{mark}"))
        .grow(1.)
        .min_w(0.)
        .min_h(0.)
        .w_full()
        .center()
        .fill(canvas_bg())
        .clip_rounded(0.);
    if !available
        && interactive
        && let Some(old) = state.pending_preview_plane()
    {
        let update = state
            .model
            .preview
            .as_ref()
            .map(|p| {
                p.centre.map_or_else(
                    || format!("Updating slice {}…", p.index + 1),
                    |c| format!("Updating centre {c:+.2} px…"),
                )
            })
            .unwrap_or_default();
        return area
            .child(
                el().aspect_fit(old.aspect())
                    .effect(11)
                    .fill(canvas_bg())
                    .child(el().w_full().h_full().fill(canvas_bg().alpha(0.45))),
            )
            .child(
                label(update, 11., canvas_ink())
                    .padding(7.)
                    .fill(canvas_bg())
                    .pin(Some(10.), Some(10.), None, None),
            );
    }
    if !available {
        let message = if state.model.preview.is_some() {
            "Waiting for preview…"
        } else if state.model.render_loading || (comparison && info.is_some()) {
            "Loading image…"
        } else if state.model.active().is_some_and(|a| a.missing) {
            "Source not found"
        } else if kind == PlaneKind::Volume {
            "No volume"
        } else {
            "Nothing selected"
        };
        return area.child(
            col()
                .center()
                .gap(12.)
                .padding(24.)
                .max_w(400.)
                .child(icon(
                    if kind == PlaneKind::Volume {
                        Icon::Cube
                    } else {
                        Icon::Layers
                    },
                    28.,
                    canvas_muted(),
                ))
                .child(
                    label(message, 13., canvas_muted())
                        .wrap()
                        .align(rune::Align::Center),
                ),
        );
    }
    let mut surface = el()
        .key(format!("scientific-canvas-{mark}"))
        .role("image")
        .label(format!("{} scientific image", kind.title()))
        .effect(mark)
        .fill(canvas_bg());
    if kind == PlaneKind::Volume {
        surface = surface.w_full().h_full();
    } else if let Some(p) = info {
        let ratio = if p.pixel_shape[0] > 0 && p.pixel_shape[1] > 0 {
            let pixel_aspect = if kind == PlaneKind::Sinogram || p.spacing[0] <= 0. {
                1.
            } else {
                p.spacing[1] / p.spacing[0]
            };
            p.pixel_shape[1] as f32 / p.pixel_shape[0] as f32 * pixel_aspect as f32
        } else {
            1.
        };
        surface = surface.aspect_fit(ratio);
    }
    if mark == 12 {
        surface = surface.child(curtain(
            state.ui.compare_split,
            state.ui.display.zoom,
            state.ui.display.pan,
        ));
    }
    if comparison {
        surface = surface.child(cursor_lines(
            state.ui.compare_cursor,
            state.ui.display.zoom,
            state.ui.display.pan,
            kind,
        ));
    }
    if !comparison
        && state.model.preview.is_none()
        && state.ui.display.crosshair
        && kind.axis().is_some()
        && state
            .model
            .active()
            .is_some_and(|asset| asset.kind == AssetKind::Volume)
    {
        surface = surface.child(crosshair(state, kind, interactive));
    }
    if !comparison && kind.axis().is_some() && state.model.preview.is_none() {
        surface = surface.child(measurements(state, kind, interactive));
    }
    if interactive {
        surface = surface
            .focusable(true)
            .focus_ring(1.5, -2., axis(kind.axis().unwrap_or(0)))
            .pointer_buttons([PointerButton::Middle])
            .on_drag(move |s: &mut AppState, drag, tx| {
                if drag.phase == Phase::Down {
                    s.ui.drag_last = [0.; 2];
                }
                let delta = [drag.dx - s.ui.drag_last[0], drag.dy - s.ui.drag_last[1]];
                s.ui.drag_last = [drag.dx, drag.dy];
                if !comparison
                    && kind.axis().is_some()
                    && s.ui.inspect_tool != InspectTool::Navigate
                    && tx.button() != PointerButton::Middle
                {
                    if drag.phase == Phase::Up && drag.cancelled() {
                        s.dispatch(UiAction::CancelMeasurement, tx);
                    } else {
                        s.dispatch(
                            UiAction::MeasureDrag {
                                plane: kind,
                                start: [
                                    (drag.start_x / drag.w.max(1.)).clamp(0., 1.),
                                    (drag.start_y / drag.h.max(1.)).clamp(0., 1.),
                                ],
                                end: [
                                    (drag.x / drag.w.max(1.)).clamp(0., 1.),
                                    (drag.y / drag.h.max(1.)).clamp(0., 1.),
                                ],
                                finished: drag.phase == Phase::Up,
                            },
                            tx,
                        );
                    }
                } else if matches!(drag.phase, Phase::Down | Phase::Move) {
                    if tx.button() == PointerButton::Middle {
                        s.dispatch(
                            UiAction::Pan {
                                delta: [delta[0] / drag.w.max(1.), delta[1] / drag.h.max(1.)],
                            },
                            tx,
                        );
                    } else if mark == 12 {
                        let u = drag.x / drag.w.max(1.);
                        let split = (u - 0.5 - s.ui.display.pan[0]) / s.ui.display.zoom + 0.5;
                        s.dispatch(UiAction::SetCompareSplit(split), tx);
                    } else if comparison {
                        s.ui.compare_cursor = std::array::from_fn(|i| {
                            let pos = [drag.x / drag.w.max(1.), drag.y / drag.h.max(1.)][i];
                            ((pos - 0.5 - s.ui.display.pan[i]) / s.ui.display.zoom + 0.5)
                                .clamp(0., 1.)
                        });
                    } else if kind == PlaneKind::Volume {
                        s.dispatch(
                            UiAction::CameraDrag {
                                delta: [delta[0] * 0.35, delta[1] * 0.35],
                            },
                            tx,
                        );
                    } else if kind.axis().is_some()
                        && s.model
                            .active()
                            .is_some_and(|asset| asset.kind == AssetKind::Volume)
                    {
                        s.dispatch(
                            UiAction::SetCrosshair {
                                plane: kind,
                                position: [
                                    (drag.x / drag.w.max(1.)).clamp(0., 1.),
                                    (drag.y / drag.h.max(1.)).clamp(0., 1.),
                                ],
                            },
                            tx,
                        );
                    }
                }
                tx.instant();
            })
            .on_wheel(move |s: &mut AppState, event, tx| {
                if event.mods.cmd || kind == PlaneKind::Volume {
                    s.dispatch(UiAction::Zoom((-event.dy * 0.005).exp().clamp(0.5, 2.)), tx);
                } else if event.dy.abs() > 0.1 {
                    let step = if event.dy > 0. { 1 } else { -1 };
                    if let Some(axis) = kind.axis() {
                        s.dispatch(UiAction::StepSlice { axis, delta: step }, tx);
                    } else if kind == PlaneKind::Projection {
                        let next =
                            s.ui.display
                                .projection_index
                                .saturating_add_signed(step as isize);
                        s.dispatch(UiAction::SetProjectionIndex(next), tx);
                    } else {
                        let next =
                            s.ui.display
                                .sinogram_row
                                .saturating_add_signed(step as isize);
                        s.dispatch(UiAction::SetSinogramRow(next), tx);
                    }
                }
                tx.instant();
                true
            })
            .on_magnify(|s: &mut AppState, event, tx| {
                s.dispatch(UiAction::Zoom((1. + event.delta).max(0.01)), tx);
                tx.instant();
                true
            })
            .on_double_press_tx(|s: &mut AppState, tx| s.dispatch(UiAction::FitView, tx));
    }
    area = area.child(surface);
    if interactive && state.model.render_loading {
        area = area.child(
            label("Refining preview…", 11., canvas_ink())
                .px(10.)
                .py(7.)
                .fill(canvas_bg().alpha(0.85))
                .radius(4.)
                .pin(Some(12.), Some(10.), None, None),
        );
    }
    area
}

fn axes_label(state: &AppState, kind: PlaneKind) -> String {
    if kind == PlaneKind::Volume {
        return String::new();
    }
    state
        .model
        .planes
        .get(&kind)
        .filter(|p| p.axes.iter().all(|name| !name.is_empty()))
        .map(|p| {
            let axis = |s: &str| match s {
                "u" => "detector column".into(),
                "v" => "detector row".into(),
                "view" => "view".into(),
                _ => s.to_uppercase(),
            };
            format!("{} × {}", axis(&p.axes[1]), axis(&p.axes[0]))
        })
        .unwrap_or_else(|| kind.axes().into())
}

fn crosshair(state: &AppState, kind: PlaneKind, interactive: bool) -> El {
    let dims = [PlaneKind::Axial, PlaneKind::Coronal, PlaneKind::Sagittal]
        .map(|plane| state.model.planes.get(&plane).map_or(0, |p| p.count));
    if dims.contains(&0) {
        return el().inert();
    }
    let fractions = std::array::from_fn::<_, 3, _>(|i| {
        (state.ui.display.indices[i] as f32 + 0.5) / dims[i] as f32
    });
    let Some(info) = state.model.planes.get(&kind) else {
        return el().inert();
    };
    let fraction = |name: &str| match name {
        "x" => Some(fractions[2]),
        "y" => Some(fractions[1]),
        "z" => Some(fractions[0]),
        _ => None,
    };
    let (Some(u), Some(v)) = (fraction(&info.axes[1]), fraction(&info.axes[0])) else {
        return el().inert();
    };
    let zoom = if interactive {
        state.ui.display.zoom
    } else {
        1.
    };
    let pan = if interactive {
        state.ui.display.pan
    } else {
        [0.; 2]
    };
    cursor_lines([u, v], zoom, pan, kind)
}
/// The Split curtain, in the same zoomed and panned frame as the image.
fn curtain(split: f32, zoom: f32, pan: [f32; 2]) -> El {
    let u = ((split - 0.5) * zoom + 0.5 + pan[0]) * 100.;
    let mut shape = Shape::new().view_box(100., 100.);
    if (0.0..=100.).contains(&u) {
        shape = shape.stroke(vec![[u, 0.], [u, 100.]], 0.35, canvas_ink().alpha(0.85));
    }
    canvas(shape).w_full().h_full().absolute(0., 0.).inert()
}

fn cursor_lines([u, v]: [f32; 2], zoom: f32, pan: [f32; 2], kind: PlaneKind) -> El {
    let u = ((u - 0.5) * zoom + 0.5 + pan[0]) * 100.;
    let v = ((v - 0.5) * zoom + 0.5 + pan[1]) * 100.;
    let tint = axis(kind.axis().unwrap_or(0)).alpha(0.6);
    let mut shape = Shape::new().view_box(100., 100.);
    if (0.0..=100.).contains(&u) {
        shape = shape.stroke(vec![[u, 0.], [u, 100.]], 0.18, tint);
    }
    if (0.0..=100.).contains(&v) {
        shape = shape.stroke(vec![[0., v], [100., v]], 0.18, tint);
    }
    canvas(shape).w_full().h_full().absolute(0., 0.).inert()
}

fn measurements(state: &AppState, kind: PlaneKind, interactive: bool) -> El {
    let mut shape = Shape::new().view_box(100., 100.);
    let zoom = if interactive {
        state.ui.display.zoom
    } else {
        1.
    };
    let pan = if interactive {
        state.ui.display.pan
    } else {
        [0.; 2]
    };
    for measurement in state.model.measurements.iter().filter(|m| {
        m.plane == kind
            && state.model.active_asset.as_ref() == Some(&m.asset)
            && state
                .model
                .planes
                .get(&kind)
                .is_some_and(|p| p.index == m.index)
    }) {
        let [a, b] = measurement.points.map(|point| {
            [
                ((point[0] - 0.5) * zoom + 0.5 + pan[0]) * 100.,
                ((point[1] - 0.5) * zoom + 0.5 + pan[1]) * 100.,
            ]
        });
        let path = if measurement.tool == InspectTool::Region {
            vec![a, [b[0], a[1]], b, [a[0], b[1]], a]
        } else {
            vec![a, b]
        };
        shape = shape.stroke(path, 0.28, color([0xE8, 0xD2, 0x96]));
    }
    canvas(shape).w_full().h_full().absolute(0., 0.).inert()
}

fn image_footer(state: &AppState, kind: PlaneKind) -> El {
    let info = state.model.planes.get(&kind);
    if kind == PlaneKind::Volume {
        return row()
            .h(34.)
            .w_full()
            .px(15.)
            .items_center()
            .gap(10.)
            .child(label("Drag to orbit · Scroll to zoom", 11., canvas_muted()))
            .child(spacer())
            .child(label(
                info.map_or("", |p| p.quality.as_str()),
                11.,
                canvas_muted(),
            ));
    }
    let (count, index) = if state.ui.workspace == Workspace::Compare {
        (state.model.comparison_count, state.model.comparison_index)
    } else {
        (info.map_or(0, |p| p.count), info.map_or(0, |p| p.index))
    };
    let value = if count > 1 {
        index as f32 / (count - 1) as f32
    } else {
        0.
    };
    col()
        .px(15.)
        .py(7.)
        .fill(canvas_bg())
        .child(
            label(
                if state.ui.workspace == Workspace::Compare {
                    state.model.comparison_planes[0]
                        .as_ref()
                        .map_or("", |p| p.quality.as_str())
                } else {
                    info.map_or("", |p| p.quality.as_str())
                },
                11.,
                canvas_muted(),
            )
            .truncate(),
        )
        .child(canvas_slider(
            "image-index",
            kind.title(),
            value,
            &format!("{} / {}", if count > 0 { index + 1 } else { 0 }, count),
            count > 1,
            move |f| {
                let index = (f * count.saturating_sub(1) as f32).round() as usize;
                if let Some(axis) = kind.axis() {
                    UiAction::SetSlice { axis, index }
                } else if kind == PlaneKind::Projection {
                    UiAction::SetProjectionIndex(index)
                } else {
                    UiAction::SetSinogramRow(index)
                }
            },
        ))
}

fn compare(state: &AppState) -> El {
    let left = state
        .model
        .assets
        .iter()
        .find(|a| Some(&a.id) == state.model.compare_left.as_ref());
    let right = state
        .model
        .assets
        .iter()
        .find(|a| Some(&a.id) == state.model.compare_right.as_ref());
    let main = col()
        .key("compare-workspace")
        .padding(20.)
        .gap(16.)
        .min_h(0.)
        .min_w(0.)
        .w_full()
        .h_full()
        .child(col().gap(6.).child(strong("Compare results", 24., ink())));
    let (Some(left), Some(right)) = (left, right) else {
        return main.child(
            el().grow(1.).min_h(0.).center().child(
                col()
                    .gap(16.)
                    .max_w(390.)
                    .center()
                    .child(icon(Icon::Compare, 36., forest()))
                    .child(strong("Choose two results", 20., ink()))
                    .child(help(
                        "Select two volumes in the navigator or the inspector.",
                    ))
                    .child(action_button(
                        "compare-selection",
                        "Compare selection",
                        true,
                        state.model.selection_count() >= 2,
                        UiAction::CompareSelection,
                    )),
            ),
        );
    };
    let plane = state.ui.compare_plane;
    let mode = state.ui.compare_mode;
    let shared_grid = state.model.comparison_note.is_none();
    let planes = [PlaneKind::Axial, PlaneKind::Coronal, PlaneKind::Sagittal]
        .into_iter()
        .map(|p| {
            Segment::new(
                format!("compare-plane-{}", p.title().to_lowercase()),
                p.title(),
                p == plane,
                UiAction::ComparePlane(p),
            )
        })
        .collect();
    let modes = [
        (CompareMode::Pair, "Side by side"),
        (CompareMode::Split, "Split"),
        (CompareMode::Difference, "Difference"),
    ]
    .into_iter()
    .map(|(m, title)| {
        Segment::new(
            format!("compare-mode-{}", title.to_lowercase().replace(' ', "-")),
            title,
            m == mode,
            UiAction::CompareMode(m),
        )
        .enabled(m == CompareMode::Pair || shared_grid)
    })
    .collect();
    let mut toolbar = row()
        .h(42.)
        .w_full()
        .px(14.)
        .gap(12.)
        .items_center()
        .fill(canvas_bg())
        .child(canvas_segments(planes))
        .child(spacer())
        .child(canvas_segments(modes))
        .child(zoom_controls(state));
    if let Some(note) = &state.model.comparison_note {
        toolbar = toolbar.child(
            label(note, 11., canvas_muted())
                .truncate()
                .max_w(260.)
                .help(note),
        );
    }
    let caption = |text: String| {
        row().h(36.).px(12.).gap(12.).items_center().child(
            strong(text, 12., canvas_ink())
                .truncate()
                .min_w(0.)
                .shrink(),
        )
    };
    let body = match mode {
        CompareMode::Pair => {
            let mut pair = row().gap(2.).grow(1.).min_h(0.).min_w(0.);
            for (asset, mark) in [(left, 6), (right, 7)] {
                pair = pair.child(
                    col()
                        .grow(1.)
                        .min_w(0.)
                        .min_h(0.)
                        .fill(canvas_bg())
                        .child(caption(asset.name.clone()).help(&asset.name))
                        .child(image(state, plane, mark, true)),
                );
            }
            pair
        }
        CompareMode::Split => col()
            .grow(1.)
            .min_h(0.)
            .min_w(0.)
            .fill(canvas_bg())
            .child(
                caption(left.name.clone()).child(spacer()).child(
                    strong(&right.name, 12., canvas_ink())
                        .truncate()
                        .min_w(0.)
                        .shrink(),
                ),
            )
            .child(image(state, plane, 12, true)),
        CompareMode::Difference => col()
            .grow(1.)
            .min_h(0.)
            .min_w(0.)
            .fill(canvas_bg())
            .child(caption(format!("{} − {}", left.name, right.name)))
            .child(image(state, plane, 10, true)),
    };
    main.child(
        col()
            .key("compare-canvas")
            .grow(1.)
            .min_h(0.)
            .min_w(0.)
            .radius(10.)
            .clip_rounded(10.)
            .fill(canvas_bg())
            .child(toolbar)
            .child(body)
            .child(image_footer(state, plane)),
    )
}

fn empty_project(state: &AppState) -> El {
    el().key("empty-project")
        .grow(1.)
        .min_h(0.)
        .min_w(0.)
        .center()
        .padding(36.)
        .child(
            col()
                .max_w(480.)
                .gap(18.)
                .child(icon(Icon::Folder, 36., forest()))
                .child(strong("No scans in this project", 27., ink()))
                .child(
                    row()
                        .gap(10.)
                        .child(action_button(
                            "empty-add-scans",
                            "Add scans",
                            true,
                            true,
                            UiAction::ImportFiles,
                        ))
                        .child(action_button(
                            "empty-add-folder",
                            "Add folder",
                            false,
                            true,
                            UiAction::ImportFolder,
                        )),
                )
                .child(separator().mt(10.))
                .child(
                    action_button(
                        "empty-simulate",
                        "Simulate a scan",
                        false,
                        state.model.backend.ready,
                        UiAction::Simulate,
                    )
                    .max_w(190.),
                ),
        )
}
