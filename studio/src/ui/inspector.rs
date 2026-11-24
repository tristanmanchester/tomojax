//! Contextual scientific controls driven by the engine's operation schema.
use super::{controls::*, model::*, tokens::*};
use crate::AppState;
use crate::app::validation::Phase;
use crate::protocol::{Grid, Placement, Product};
use rune::{El, ScrollSpec, Shape, TextEdit, canvas, col, el, row};

pub fn view(state: &AppState) -> El {
    let title = state.ui.workspace.title();
    let mut body = col().w_full().gap(18.);
    if let Some(asset) = state.model.active().filter(|a| a.missing) {
        body = body
            .child(help("Source not found."))
            .child(
                action_button(
                    "relink-inspector",
                    "Reconnect source",
                    true,
                    true,
                    UiAction::RelinkAsset(asset.id.clone()),
                )
                .w_full(),
            )
            .child(separator());
    }
    body = body.child(match state.ui.workspace {
        Workspace::Prepare | Workspace::Reconstruct => processing(state),
        Workspace::Explore => explore(state),
        Workspace::Compare => comparison(state),
        Workspace::Jobs => el(),
    });
    col()
        .key("inspector")
        .w(304.)
        .h_full()
        .min_h(0.)
        .fill(paper())
        .child(
            col()
                .px(22.)
                .pt(25.)
                .pb(21.)
                .gap(7.)
                .child(strong(title, 22., ink())),
        )
        .child(separator())
        .child(
            col()
                .key("inspector-scroll")
                .grow(1.)
                .min_h(0.)
                .w_full()
                .scroll(ScrollSpec {
                    stick_bottom: false,
                })
                .child(body.padding(22.)),
        )
        .child(footer(state))
}

fn processing(state: &AppState) -> El {
    let mut body = col().gap(18.).w_full();
    body = body.child(operation_selector(state));
    if state.ui.operation == "open" {
        body = body
            .child(
                row()
                    .gap(8.)
                    .child(
                        action_button(
                            "inspector-import-files",
                            "Add files",
                            false,
                            true,
                            UiAction::ImportFiles,
                        )
                        .grow(1.),
                    )
                    .child(
                        action_button(
                            "inspector-import-folder",
                            "Add folder",
                            false,
                            true,
                            UiAction::ImportFolder,
                        )
                        .grow(1.),
                    ),
            )
            .child(toggle(
                "copy-imports",
                "Copy sources into the project",
                state.ui.import_copy,
                true,
                |s, _| s.ui.import_copy = !s.ui.import_copy,
            ));
    }
    body = body.child(validation_summary(state));
    if state.model.preview.is_some() {
        body = body.child(super::preview::controls(state));
    }
    let mut params = col().gap(15.);
    let mut advanced = 0;
    for parameter in &state.model.parameters {
        if !crate::forms::applies(&state.ui, &state.catalog, &parameter.name) {
            continue;
        }
        if state.advanced_parameter(parameter) {
            advanced += 1;
            if !state.ui.advanced_open {
                continue;
            }
        }
        params = params.child(parameter_field(state, parameter));
    }
    if state.model.parameters.is_empty() {
        params = params.child(help(if state.model.backend.ready {
            "No settings for this operation."
        } else {
            "Connecting to TomoJAX…"
        }));
    }
    body = body.child(params);
    if advanced > 0 {
        body = body.child(
            button(
                "advanced-settings",
                &format!(
                    "{} advanced settings ({advanced})",
                    if state.ui.advanced_open {
                        "Hide"
                    } else {
                        "Show"
                    }
                ),
                false,
                true,
                |s, _| s.ui.advanced_open = !s.ui.advanced_open,
            )
            .w_full(),
        );
    }
    if let Some(plan) = &state.model.validation.plan {
        body = body.child(separator()).child(plan_details(state, plan));
    }
    let selection = state.model.selection_count();
    if selection > 0 {
        body = body
            .child(separator())
            .child(property("Selected inputs", selection.to_string()));
    }
    body.child(separator()).child(section(
        "Reuse these settings",
        col()
            .gap(10.)
            .child(field(
                "recipe-name",
                &state.ui.recipe_name,
                "Recipe name",
                "",
                |s| &mut s.ui.recipe_name,
            ))
            .child(
                action_button(
                    "save-recipe",
                    "Save recipe",
                    false,
                    state.model.backend.ready,
                    UiAction::SaveRecipe,
                )
                .w_full(),
            ),
    ))
}

fn validation_summary(state: &AppState) -> El {
    let validation = &state.model.validation;
    if validation.phase == Phase::Idle {
        return el();
    }
    let tone = match validation.phase {
        Phase::Invalid => error(),
        Phase::Valid => forest(),
        _ => muted(),
    };
    let mut content = col()
        .key("form-validation")
        .role("status")
        .w_full()
        .gap(6.)
        .child(label(&validation.message, 11., tone).wrap());
    if !validation.scope.is_empty() {
        content = content.child(label(&validation.scope, 11., muted()).wrap());
    }
    for issue in &validation.issues {
        if let Some(control) = &issue.control {
            let control = control.clone();
            let json = control == "corrections-json";
            content = content.child(
                button(
                    &format!("review-error-{}", issue.path),
                    &format!("Review {}", issue.path),
                    false,
                    true,
                    move |state, tx| {
                        state.ui.advanced_open = true;
                        if json {
                            super::corrections::reveal_json(&mut state.ui);
                        }
                        tx.focus(control.clone(), true);
                        tx.scroll_into_view(format!("validation-field-{control}"));
                    },
                )
                .w_full()
                .h(30.),
            );
        } else {
            content = content.child(label(&issue.message, 11., error()).wrap());
        }
    }
    content
}

fn plan_details(state: &AppState, plan: &Product) -> El {
    let mut content = col().key("resolved-plan").w_full().gap(9.);
    let (reasons, placement) = match plan {
        Product::ReconstructionPlan {
            method,
            reasons,
            iterations,
            grid,
            placement,
        } => {
            content = content.child(property("Method", crate::operations::method_name(method)));
            if let Some(iterations) = iterations {
                content = content.child(property("Iteration limit", iterations.to_string()));
            }
            content = content.child(plan_grid(state, grid));
            (reasons, placement)
        }
        Product::JointPlan {
            reasons,
            levels,
            placement,
        } => {
            for (index, level) in levels.iter().enumerate() {
                content = content
                    .child(strong(format!("Level {}", index + 1), 11., ink()))
                    .child(plan_grid(state, &level.grid))
                    .child(property(
                        "Outer / image iterations",
                        format!("{} / {}", level.iterations, level.image_iterations),
                    ));
            }
            (reasons, placement)
        }
        Product::OppositeViewsPlan {
            pairs,
            reasons,
            placement,
        } => {
            content = content.child(property("Angular pairs", pairs.to_string()));
            (reasons, placement)
        }
        _ => return el(),
    };
    for reason in reasons {
        content = content.child(help(reason));
    }
    content = content.child(plan_placement(placement));
    section("Resolved plan", content)
}

fn plan_grid(state: &AppState, grid: &Grid) -> El {
    let unit = state.model.active().map_or("source length unit", |asset| {
        if asset.units.is_empty() {
            "source length unit"
        } else {
            unit_name(&asset.units)
        }
    });
    col()
        .w_full()
        .gap(7.)
        .child(property(
            "Grid (x × y × z)",
            format!("{} × {} × {}", grid.nx, grid.ny, grid.nz),
        ))
        .child(property(
            "Voxel pitch",
            format!(
                "{} × {} × {} {unit}",
                compact_number(grid.vx),
                compact_number(grid.vy),
                compact_number(grid.vz)
            ),
        ))
}

fn plan_placement(placement: &Placement) -> El {
    let mut content = col()
        .gap(7.)
        .w_full()
        .child(property(
            "Placement",
            match placement.strategy.as_str() {
                "resident" => "Resident",
                "slabbed" => "Slabbed",
                other => other,
            },
        ))
        .child(property("Devices", placement.devices.join(", ")))
        .child(property(
            "Estimated host memory",
            memory_bytes(placement.host_bytes),
        ))
        .child(property(
            "Estimated device memory",
            memory_bytes(placement.device_bytes),
        ));
    for snapshot in &placement.snapshot {
        if let Some(available) = snapshot.available {
            content = content.child(
                property(
                    &format!("Available · {} / {}", snapshot.device, snapshot.scope),
                    memory_bytes(available),
                )
                .help(format!("Worker snapshot time: {} seconds", snapshot.time)),
            );
        }
    }
    content
}

fn memory_bytes(bytes: u64) -> String {
    for (scale, unit) in [(1_u64 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")] {
        if bytes >= scale {
            return format!("{:.1} {unit}", bytes as f64 / scale as f64);
        }
    }
    format!("{bytes} B")
}

pub(super) fn field_errors(state: &AppState, control: &str, field: El) -> El {
    let issues: Vec<_> = state
        .model
        .validation
        .issues
        .iter()
        .filter(|issue| issue.control.as_deref() == Some(control))
        .collect();
    if issues.is_empty() {
        return field;
    }
    let mut content = col()
        .key(format!("validation-field-{control}"))
        .w_full()
        .min_w(0.)
        .gap(6.)
        .padding(5.)
        .radius(6.)
        .border(1., error())
        .child(field);
    for issue in issues {
        content = content.child(
            label(format!("{}: {}", issue.path, issue.message), 11., error())
                .key(format!("validation-message-{}", issue.path))
                .role("alert")
                .wrap(),
        );
    }
    content
}

fn operation_selector(state: &AppState) -> El {
    let title = crate::operations::title(&state.ui.operation);
    let mut selector = col()
        .w_full()
        .gap(7.)
        .child(label("Operation", 11., muted()))
        .child(
            button("operation-selector", title, false, true, |s, _| {
                s.ui.operation_menu = !s.ui.operation_menu
            })
            .w_full()
            .justify_between()
            .child(icon(Icon::Down, 14., muted())),
        );
    if state.ui.operation_menu {
        let mut options = col().gap(3.).padding(5.);
        for group in crate::operations::GROUPS
            .iter()
            .filter(|group| group.reconstruction == (state.ui.workspace == Workspace::Reconstruct))
        {
            options = options.child(strong(group.title, 11., muted()).px(8.).pt(8.));
            for op in group.operations {
                let id = op.id;
                options = options.child(
                    menu_option(
                        &format!("operation-{}", op.id),
                        op.title,
                        state.ui.operation == op.id,
                        move |s, tx| {
                            s.ui.operation_menu = false;
                            s.dispatch(UiAction::SetOperation(id.into()), tx);
                        },
                    )
                    .help(op.help),
                );
            }
        }
        selector = selector.child(
            col()
                .key("operation-menu")
                .max_h(260.)
                .w_full()
                .scroll(ScrollSpec {
                    stick_bottom: false,
                })
                .fill(panel())
                .radius(6.)
                .child(options),
        );
    }
    selector
}

fn parameter_field(state: &AppState, parameter: &ParameterView) -> El {
    if parameter.name == "corrections" {
        return super::corrections::view(state);
    }
    let key = parameter.name.clone();
    let mut content = match parameter.kind {
        ParameterKind::Boolean => {
            let value = state
                .ui
                .parameter_toggles
                .get(&key)
                .copied()
                .unwrap_or(false);
            toggle(
                &format!("parameter-{key}"),
                &parameter.label,
                value,
                true,
                move |s, _| {
                    s.ui.parameter_toggles.insert(key.clone(), !value);
                },
            )
        }
        ParameterKind::Choice => parameter_choice(state, parameter),
        _ => {
            let fallback = TextEdit::default();
            let value = state.ui.parameter_edits.get(&key).unwrap_or(&fallback);
            field_hint(
                &format!("parameter-{key}"),
                value,
                &parameter.label,
                &parameter.unit,
                &parameter.placeholder,
                move |s| s.ui.parameter_edits.entry(key.clone()).or_default(),
            )
        }
    };
    if !parameter.help.is_empty() {
        content = content.help(&parameter.help);
    }
    if parameter.required && matches!(parameter.kind, ParameterKind::Text | ParameterKind::Path) {
        content = col()
            .gap(5.)
            .child(content)
            .child(label(&parameter.help, 11., muted()).wrap());
    }
    field_errors(state, &format!("parameter-{}", parameter.name), content)
}

fn parameter_choice(state: &AppState, parameter: &ParameterView) -> El {
    let value = state
        .ui
        .parameter_choices
        .get(&parameter.name)
        .map_or("", String::as_str);
    let title = parameter
        .choices
        .iter()
        .find(|choice| choice.value == value)
        .map_or(value, |choice| choice.label.as_str());
    let key = parameter.name.clone();
    let open = state.ui.choice_menu.as_deref() == Some(&key);
    let key_for_toggle = key.clone();
    let mut picker = col()
        .gap(7.)
        .w_full()
        .child(label(&parameter.label, 11., muted()))
        .child(
            button(
                &format!("parameter-{key}"),
                if title.is_empty() { "Choose…" } else { title },
                false,
                true,
                move |s, _| {
                    s.ui.choice_menu = if open {
                        None
                    } else {
                        Some(key_for_toggle.clone())
                    };
                },
            )
            .w_full()
            .justify_between()
            .child(icon(Icon::Down, 14., muted())),
        );
    if open {
        let mut choices = col().gap(3.);
        for option in &parameter.choices {
            let key = key.clone();
            let value = option.value.clone();
            let chosen = option.value
                == state
                    .ui
                    .parameter_choices
                    .get(&key)
                    .map_or("", String::as_str);
            choices = choices.child(menu_option(
                &format!("parameter-{key}-{}", option.value),
                &option.label,
                chosen,
                move |s, _| {
                    s.ui.parameter_choices.insert(key.clone(), value.clone());
                    s.ui.choice_menu = None;
                },
            ));
        }
        picker = picker.child(
            col()
                .max_h(220.)
                .w_full()
                .scroll(ScrollSpec {
                    stick_bottom: false,
                })
                .child(choices),
        );
    }
    picker
}

fn explore(state: &AppState) -> El {
    let kind = state.ui.main_view;
    let mut sections: Vec<El> = Vec::new();
    if !state.model.preview_choices.is_empty() {
        let mut choices = row().gap(6.).w_full();
        for (value, title) in &state.model.preview_choices {
            choices = choices.child(choice(
                &format!("preview-array-{value}"),
                title,
                state.ui.preview_array == *value,
                true,
                UiAction::SetPreviewArray(value.clone()),
            ));
        }
        sections.push(section("Correction preview", choices));
    }
    // The view's own tools lead so they stay visible at the minimum window size.
    if kind == PlaneKind::Volume {
        sections.push(section(
            "Volume appearance",
            col()
                .gap(10.)
                .child(slider(
                    "opacity-threshold",
                    "Opacity threshold",
                    state.ui.display.threshold,
                    &format!("{:.0}%", state.ui.display.threshold * 100.),
                    true,
                    UiAction::SetThreshold,
                ))
                .child(slider(
                    "density",
                    "Density",
                    state.ui.display.density / 4.,
                    &format!("{:.2}", state.ui.display.density),
                    true,
                    |v| UiAction::SetDensity(v * 4.),
                ))
                .child(slider(
                    "lighting",
                    "Lighting",
                    state.ui.display.lighting,
                    &format!("{:.0}%", state.ui.display.lighting * 100.),
                    true,
                    UiAction::SetLighting,
                )),
        ));
        let mut clipping = col().gap(10.);
        for (axis, name) in ["X extent", "Y extent", "Z extent"].into_iter().enumerate() {
            clipping = clipping.child(slider(
                &format!("clip-{axis}"),
                name,
                state.ui.display.clip[axis],
                &format!("{:.0}%", state.ui.display.clip[axis] * 100.),
                true,
                move |value| UiAction::SetClip { axis, value },
            ));
        }
        sections.push(section("Clip the volume", clipping));
    } else if kind.axis().is_some() {
        let enabled = state.model.preview.is_none()
            && state.model.planes.get(&kind).is_some_and(|p| p.available);
        let mut measure = col().gap(12.);
        measure = measure
            .child(row().gap(4.)
                .child(choice("tool-navigate","Navigate",state.ui.inspect_tool==InspectTool::Navigate,enabled,UiAction::SetInspectTool(InspectTool::Navigate)))
                .child(choice("tool-distance","Distance",state.ui.inspect_tool==InspectTool::Distance,enabled,UiAction::SetInspectTool(InspectTool::Distance)))
                .child(choice("tool-region","Region",state.ui.inspect_tool==InspectTool::Region,enabled,UiAction::SetInspectTool(InspectTool::Region))))
            .child(help(if state.model.preview.is_some() { "Keep the preview to measure it." } else { match state.ui.inspect_tool{
                InspectTool::Navigate if state.model.active().is_some_and(|a|a.kind==AssetKind::Volume)&&[PlaneKind::Axial,PlaneKind::Coronal,PlaneKind::Sagittal].iter().all(|p|state.model.planes.get(p).is_some_and(|p|p.available))=>"Drag to place the linked crosshair. Middle-drag pans; the wheel moves through slices.",
                InspectTool::Navigate=>"Middle-drag pans; the wheel moves through the available planes.",
                InspectTool::Distance=>"Drag between two points in the slice to measure their calibrated distance.",
                InspectTool::Region=>"Drag a rectangular region to inspect its sampled values."}}));
        for measurement in state
            .model
            .measurements
            .iter()
            .filter(|m| state.model.active_asset.as_ref() == Some(&m.asset))
        {
            measure = measure.child(
                col()
                    .gap(5.)
                    .child(
                        row()
                            .justify_between()
                            .gap(8.)
                            .child(strong(&measurement.name, 11., ink()))
                            .child(
                                number(&measurement.value, 11., forest())
                                    .shrink()
                                    .min_w(0.)
                                    .truncate(),
                            )
                            .child({
                                let id = measurement.id.clone();
                                icon_button(
                                    &format!("remove-measurement-{id}"),
                                    "Remove measurement",
                                    Icon::Close,
                                    true,
                                    move |s, tx| {
                                        s.dispatch(UiAction::RemoveMeasurement(id.clone()), tx)
                                    },
                                )
                            }),
                    )
                    .child(
                        label(
                            format!(
                                "{} · slice {}. {}",
                                measurement.plane.title(),
                                measurement.index + 1,
                                measurement.detail
                            ),
                            11.,
                            muted(),
                        )
                        .wrap(),
                    ),
            );
        }
        if state
            .model
            .measurements
            .iter()
            .any(|m| Some(&m.asset) == state.model.active_asset.as_ref())
        {
            measure = measure.child(
                action_button(
                    "clear-measurements",
                    "Clear measurements",
                    false,
                    true,
                    UiAction::ClearMeasurements,
                )
                .w_full()
                .help("Remove measurements from this asset."),
            );
        }
        sections.push(section("Measure", measure));
    }
    sections.push(display(state));
    if let Some(plane) = state.model.planes.get(&kind)
        && let Some(stats) = &plane.statistics
    {
        sections.push(section(
            "Sampled image values",
            col()
                .gap(10.)
                .child(property("Minimum", compact_number(stats.min)))
                .child(property("Maximum", compact_number(stats.max)))
                .child(property("Mean", compact_number(stats.mean)))
                .child(property("Standard deviation", compact_number(stats.std)))
                .child(help(format!("{} sampled pixels", stats.sampled_pixels))),
        ));
    }
    if !state
        .model
        .export_format(&state.ui.export_format)
        .is_empty()
    {
        sections.push(export_format(state));
    }
    let mut content = col().gap(18.);
    for (index, section) in sections.into_iter().enumerate() {
        if index > 0 {
            content = content.child(separator());
        }
        content = content.child(section);
    }
    content
}

fn display(state: &AppState) -> El {
    let comparison = state.ui.workspace == Workspace::Compare;
    let info = if comparison {
        state.model.comparison_planes[0].as_ref()
    } else {
        state.model.planes.get(&state.ui.main_view)
    };
    let ready = if comparison {
        info.is_some_and(|p| p.available)
    } else {
        state.model.active().is_some()
    };
    let mut body = col().gap(12.);
    if let Some(plane) = info.filter(|p| !p.histogram.is_empty()) {
        body = body.child(histogram(plane));
    }
    if let Some(plane) = info.filter(|p| !p.units.is_empty()) {
        body = body.child(label(format!("Unit: {}", unit_name(&plane.units)), 11., muted()).wrap());
    }
    body = body
        .child(
            row()
                .gap(8.)
                .child(submit_field(
                    "window-low",
                    &state.ui.display.window_low,
                    "Minimum",
                    UiAction::ApplyWindow,
                    |s| &mut s.ui.display.window_low,
                ))
                .child(submit_field(
                    "window-high",
                    &state.ui.display.window_high,
                    "Maximum",
                    UiAction::ApplyWindow,
                    |s| &mut s.ui.display.window_high,
                )),
        )
        .child(
            row()
                .gap(8.)
                .child(
                    action_button(
                        "apply-window",
                        "Apply window",
                        false,
                        ready,
                        UiAction::ApplyWindow,
                    )
                    .grow(1.),
                )
                .child(action_button(
                    "auto-window",
                    "Auto",
                    false,
                    ready,
                    UiAction::AutoWindow,
                )),
        )
        .child(
            row()
                .gap(4.)
                .child(choice(
                    "colormap-gray",
                    "Gray",
                    state.ui.display.colormap == "gray",
                    ready,
                    UiAction::SetColormap("gray".into()),
                ))
                .child(choice(
                    "colormap-bone",
                    "Bone",
                    state.ui.display.colormap == "bone",
                    ready,
                    UiAction::SetColormap("bone".into()),
                ))
                .child(choice(
                    "colormap-viridis",
                    "Viridis",
                    state.ui.display.colormap == "viridis",
                    ready,
                    UiAction::SetColormap("viridis".into()),
                ))
                .child(choice(
                    "colormap-magma",
                    "Magma",
                    state.ui.display.colormap == "magma",
                    ready,
                    UiAction::SetColormap("magma".into()),
                )),
        )
        .child(toggle(
            "invert-display",
            "Invert display",
            state.ui.display.invert,
            ready,
            |s, tx| s.dispatch(UiAction::SetInvert(!s.ui.display.invert), tx),
        ));
    section(
        if comparison && !state.ui.compare_linked {
            "Left display window"
        } else {
            "Display window"
        },
        body,
    )
}

fn histogram(plane: &PlaneView) -> El {
    let max = plane.histogram.iter().copied().max().unwrap_or(1).max(1) as f32;
    let count = plane.histogram.len().max(2);
    let points: Vec<_> = plane
        .histogram
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            [
                i as f32 / (count - 1) as f32 * 100.,
                97. - v as f32 / max * 87.,
            ]
        })
        .collect();
    let mut fill = vec![[0., 100.]];
    fill.extend(points.iter().copied());
    fill.push([100., 100.]);
    let shape = Shape::new()
        .view_box(100., 100.)
        .fill(fill, forest().alpha(0.12))
        .stroke(points, 0.8, color("#688E6E"));
    col()
        .gap(5.)
        .child(
            canvas(shape)
                .h(74.)
                .w_full()
                .role("image")
                .label("Sampled intensity distribution"),
        )
        .child(
            row()
                .justify_between()
                .child(number(
                    compact_number(
                        plane
                            .histogram_edges
                            .first()
                            .copied()
                            .unwrap_or(plane.range[0]),
                    ),
                    11.,
                    muted(),
                ))
                .child(number(
                    compact_number(
                        plane
                            .histogram_edges
                            .last()
                            .copied()
                            .unwrap_or(plane.range[1]),
                    ),
                    11.,
                    muted(),
                )),
        )
}

fn comparison(state: &AppState) -> El {
    let mut pickers = col().gap(10.);
    for (side, title, selected) in [
        (0, "Left", &state.model.compare_left),
        (1, "Right", &state.model.compare_right),
    ] {
        pickers = pickers.child(result_picker(state, side, title, selected.as_deref()));
    }
    pickers = pickers.child(
        action_button(
            "swap-comparison",
            "Swap left and right",
            false,
            true,
            UiAction::SwapComparison,
        )
        .w_full(),
    );
    let mut body = col().gap(18.).child(section("Results", pickers));
    // A difference is drawn on its own window, symmetric about zero; the left
    // result's histogram and window don't describe it.
    if state.ui.compare_mode != CompareMode::Difference {
        body = body
            .child(separator())
            .child(toggle(
                "link-comparison",
                "Share display window",
                state.ui.compare_linked,
                true,
                |s, tx| s.dispatch(UiAction::SetCompareLinked(!s.ui.compare_linked), tx),
            ))
            .child(display(state));
    }
    body
}

/// A compact picker for one side of the comparison; the list opens in place.
fn result_picker(state: &AppState, side: usize, title: &str, selected: Option<&str>) -> El {
    let key = if side == 0 { "left" } else { "right" };
    let current = state
        .model
        .assets
        .iter()
        .find(|a| Some(a.id.as_str()) == selected)
        .map_or("Choose a volume", |a| a.name.as_str());
    let open = state.ui.compare_menu == Some(side);
    let mut picker = col()
        .w_full()
        .gap(6.)
        .child(label(title, 11., muted()))
        .child(
            button(
                &format!("compare-{key}"),
                current,
                false,
                true,
                move |s, _| {
                    s.ui.compare_menu = if s.ui.compare_menu == Some(side) {
                        None
                    } else {
                        Some(side)
                    }
                },
            )
            .w_full()
            .justify_between()
            .child(icon(if open { Icon::Up } else { Icon::Down }, 14., muted())),
        );
    if open {
        let mut options = col()
            .gap(2.)
            .padding(4.)
            .radius(6.)
            .border(1., line())
            .fill(paper());
        for asset in state
            .model
            .assets
            .iter()
            .filter(|a| a.kind == AssetKind::Volume && !a.missing)
        {
            let chosen = Some(asset.id.as_str()) == selected;
            let action = if side == 0 {
                UiAction::SetCompareLeft(asset.id.clone())
            } else {
                UiAction::SetCompareRight(asset.id.clone())
            };
            options = options.child(menu_option(
                &format!("compare-{key}-{}", asset.id),
                &asset.name,
                chosen,
                move |s, tx| s.dispatch(action.clone(), tx),
            ));
        }
        picker = picker.child(options);
    }
    picker
}

fn export_format(state: &AppState) -> El {
    let mut choices = row().gap(4.);
    let format = state.model.export_format(&state.ui.export_format);
    let formats: &[(&str, &str)] = match format {
        "" => &[],
        "array" => &[("array", "NPY bundle")],
        _ if state.model.tiff_export() => &[
            ("nxs", "NXtomo"),
            ("float32", "TIFF 32"),
            ("uint16", "TIFF 16"),
        ],
        _ => &[("nxs", "NXtomo")],
    };
    for &(value, title) in formats {
        choices = choices.child(choice(
            &format!("export-format-{value}"),
            title,
            format == value,
            true,
            UiAction::SetExportFormat(value.into()),
        ));
    }
    let mut export = col().gap(9.).child(choices);
    if format == "uint16" {
        export = export.child(help(
            "16-bit TIFF rescales values to integers over a percentile range.",
        ));
    }
    section("Export", export)
}

fn footer(state: &AppState) -> El {
    if state.model.preview.is_some() {
        return super::preview::footer(state);
    }
    let mut content = col().px(22.).py(18.).gap(8.).fill(paper());
    match state.ui.workspace {
        Workspace::Prepare | Workspace::Reconstruct => {
            let name = match state.ui.operation.as_str() {
                "open" => "Open selected scans",
                "reconstruct" => "Queue reconstruction",
                "correct" => "Queue corrections",
                "align" => "Queue alignment",
                "simulate" => "Queue simulation",
                _ => "Queue operation",
            };
            let invalid = state
                .current_validation_error(&state.ui.operation)
                .is_some();
            let reading = state.ui.operation != "simulate" && state.form_inputs_pending();
            let available = state.model.backend.ready
                && (state.model.selection_count() > 0 || state.ui.operation == "simulate")
                && state.model.active().is_none_or(|a| !a.missing)
                && !invalid
                && !reading;
            if !available {
                let reason = if !state.model.backend.ready {
                    if state.model.backend.status.is_empty() {
                        "Waiting for TomoJAX"
                    } else {
                        &state.model.backend.status
                    }
                } else if state.model.active().is_some_and(|asset| asset.missing) {
                    "Reconnect the source to queue processing."
                } else if invalid {
                    "Resolve the validation error to queue this operation."
                } else if reading {
                    "Reading the selected inputs…"
                } else {
                    "Select inputs to queue this operation."
                };
                content = content.child(help(reason).key("queue-unavailable").role("status"));
            }
            content = content.child(
                action_button_icon(
                    "queue-operation",
                    name,
                    Icon::Play,
                    true,
                    available,
                    UiAction::QueueOperation(state.ui.operation.clone()),
                )
                .w_full()
                .h(40.),
            );
        }
        Workspace::Explore => {
            content = content.child(
                action_button_icon(
                    "export-result",
                    match state.model.export_format(&state.ui.export_format) {
                        "float32" => "Export TIFF (float32)",
                        "uint16" => "Export TIFF (uint16)",
                        "array" => "Export NPY bundle",
                        _ => "Export NXtomo",
                    },
                    Icon::Export,
                    true,
                    !state
                        .model
                        .export_format(&state.ui.export_format)
                        .is_empty(),
                    UiAction::Export(state.model.export_format(&state.ui.export_format).into()),
                )
                .w_full()
                .h(40.),
            );
        }
        Workspace::Compare | Workspace::Jobs => return el(),
    }
    col().w_full().child(separator()).child(content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rune::{Env, Runtime};

    #[test]
    fn temporary_preview_uses_keep_instead_of_queue_or_export_and_cannot_be_measured() {
        let mut state = AppState::init();
        state.model.project = Some(ProjectView {
            name: "Preview UI test".into(),
            ..Default::default()
        });
        state.model.backend.ready = true;
        state.ui.workspace = Workspace::Prepare;
        state.ui.main_view = PlaneKind::Axial;
        state.set_operation("slice", None);
        state.model.preview = Some(PreviewView {
            operation: "slice".into(),
            phase: crate::previews::PreviewPhase::Ready,
            can_keep: true,
            has_image: true,
            count: Some(16),
            ..Default::default()
        });
        state.model.planes.insert(
            PlaneKind::Axial,
            PlaneView {
                available: true,
                count: 16,
                ..Default::default()
            },
        );
        let mut runtime = Runtime::new(state, crate::ui::view, (1180., 780.), Env::new(false, 1.));
        runtime.frame(0.);
        runtime.frame(1. / 60.);
        assert!(
            runtime
                .elements()
                .iter()
                .any(|element| element.key.as_deref() == Some("keep-preview") && element.focusable)
        );
        assert!(
            !runtime
                .elements()
                .iter()
                .any(|element| element.key.as_deref() == Some("queue-operation"))
        );
        assert!(
            runtime
                .elements()
                .iter()
                .any(|element| element.key.as_deref() == Some("interactive-preview-controls"))
        );
        runtime.state.ui.workspace = Workspace::Explore;
        let mut runtime = Runtime::new(
            runtime.state,
            crate::ui::view,
            (1180., 780.),
            Env::new(false, 1.),
        );
        runtime.frame(0.);
        runtime.frame(1. / 60.);
        assert!(
            !runtime
                .elements()
                .iter()
                .any(|element| element.key.as_deref() == Some("export-result"))
        );
        for key in ["tool-distance", "tool-region"] {
            let control = runtime
                .elements()
                .into_iter()
                .find(|element| element.key.as_deref() == Some(key))
                .unwrap();
            assert!(
                !control.focusable,
                "temporary measurements have no durable asset context"
            );
        }
    }
}
