//! Registered checkpoints, task hierarchy and measured solver histories.
use super::{controls::*, model::*, tokens::*};
use crate::AppState;
use crate::progress::{MetricHistory, MetricSample, TaskProgress, TaskState};
use rune::{El, GrowthSpec, ScrollSpec, Shape, canvas, col, el, row};

pub fn live(state: &AppState, compact: bool) -> El {
    let Some(frame) = state.live.frame() else {
        return if state.live.loading() {
            help("Reading the current solver checkpoint…")
        } else if let Some(message) = state.live.error() {
            label(format!("Checkpoint unavailable: {message}"), 11., error()).wrap()
        } else {
            el()
        };
    };
    let job = state.model.jobs.iter().find(|job| job.id == frame.info.job);
    let running =
        job.is_some_and(|job| matches!(job.status, JobStatus::Running | JobStatus::Paused));
    let title = if running {
        "Live solver checkpoint"
    } else {
        "Last solver checkpoint"
    };
    let mut section = col()
        .key(if compact {
            "live-checkpoint-compact"
        } else {
            "live-checkpoint"
        })
        .gap(10.)
        .w_full()
        .child(
            row()
                .gap(8.)
                .items_center()
                .child(status_dot(if running { forest() } else { muted() }))
                .child(strong(title, 13., ink()))
                .child(spacer())
                .child(number(
                    frame
                        .info
                        .position
                        .as_ref()
                        .map(crate::live::CheckpointPosition::label)
                        .unwrap_or_default(),
                    11.,
                    muted(),
                )),
        );
    if !compact {
        section = section.child(help(format!(
            "{} · completed checkpoint at {:.2} s",
            job.map_or("Processing result", |job| job.name.as_str()),
            frame.info.time
        )));
    }
    let mut slices = row().gap(10.).w_full().h(if compact { 108. } else { 224. });
    for (axis, title, mark) in [
        ("z", "Axial", 40),
        ("y", "Coronal", 41),
        ("x", "Sagittal", 42),
    ] {
        if let Some(plane) = frame.planes.get(axis) {
            let fixed = plane.fixed.iter().find(|fixed| fixed.name == axis);
            let coordinate = fixed
                .map(|fixed| format!("{} = {:.4} {}", axis, fixed.coordinate, fixed.unit))
                .unwrap_or_else(|| axis.into());
            let mut caption = col()
                .px(9.)
                .py(6.)
                .gap(3.)
                .w_full()
                .fill(sage())
                .child(strong(title, 11., forest()))
                .child(
                    number(
                        if compact {
                            axis.into()
                        } else {
                            coordinate.clone()
                        },
                        11.,
                        muted(),
                    )
                    .truncate()
                    .help(&coordinate),
                );
            if !compact {
                let [low, high] = plane.statistics.window;
                caption = caption.child(
                    number(format!("Auto window {low:.3}–{high:.3}"), 11., muted())
                        .key(format!("live-window-{axis}"))
                        .truncate()
                        .help(format!(
                            "Automatic display window: {low} to {high} {}",
                            plane.value_unit
                        )),
                );
            }
            slices = slices.child(
                col()
                    .grow(1.)
                    .min_w(0.)
                    .h_full()
                    .radius(6.)
                    .clip_rounded(6.)
                    .fill(canvas_bg())
                    .child(
                        el().grow(1.).min_h(0.).w_full().center().child(
                            el().key(format!(
                                "live-slice-{axis}{}",
                                if compact { "-compact" } else { "" }
                            ))
                            .role("image")
                            .label(format!("{title} current iterate; {coordinate}"))
                            .effect(mark)
                            .aspect_fit(plane.aspect())
                            .fill(canvas_bg()),
                        ),
                    )
                    .child(caption),
            );
        }
    }
    let mut panel = row().key("live-checkpoint-panels").gap(18.).w_full();
    if !frame.planes.is_empty() {
        if compact {
            section = section.child(slices);
        } else {
            panel = panel.child(slices.grow(1.).min_w(0.));
        }
    }
    if let Some(poses) = &frame.poses {
        if compact {
            section = section.child(help(format!(
                "{} registered pose rows · open Jobs for the table",
                poses.shape[0]
            )));
        } else {
            let mut headings = row()
                .h(43.)
                .gap(6.)
                .px(8.)
                .items_center()
                .fill(sage())
                .child(strong("View", 11., forest()).w(44.));
            for column in &poses.columns {
                headings = headings.child(
                    col()
                        .gap(4.)
                        .grow(1.)
                        .min_w(0.)
                        .child(strong(&column.name, 11., forest()).truncate())
                        .child(label(unit_name(&column.unit), 11., muted()).truncate())
                        .help(format!("{} · {}", column.name, unit_name(&column.unit))),
                );
            }
            let poses_panel = col()
                .key("live-poses")
                .gap(8.)
                .grow(1.)
                .min_w(460.)
                .child(strong("Per-view pose increments", 12., ink()))
                .child(
                    col()
                        .key("pose-table")
                        .role("table")
                        .w_full()
                        .border(1., line())
                        .radius(5.)
                        .child(headings)
                        .child(
                            col()
                                .key("pose-table-scroll")
                                .h(140.)
                                .w_full()
                                .scroll(ScrollSpec {
                                    stick_bottom: false,
                                })
                                .child(
                                    col()
                                        .key("pose-rows")
                                        .w_full()
                                        .growth(GrowthSpec::default())
                                        .virtual_rows(
                                            poses.shape[0],
                                            28.,
                                            |_: &AppState, index| {
                                                format!("pose-row-{index}").into()
                                            },
                                            |state: &AppState, index| {
                                                let Some(poses) = state
                                                    .live
                                                    .frame()
                                                    .and_then(|frame| frame.poses.as_ref())
                                                else {
                                                    return el();
                                                };
                                                let mut cells = row()
                                                    .role("row")
                                                    .w_full()
                                                    .h(28.)
                                                    .gap(6.)
                                                    .px(8.)
                                                    .items_center()
                                                    .child(
                                                        number(
                                                            compact_number(
                                                                poses.axes[0]
                                                                    .coordinate(index as f64),
                                                            ),
                                                            11.,
                                                            muted(),
                                                        )
                                                        .w(44.),
                                                    );
                                                if let Some(values) = poses.row(index) {
                                                    for value in values {
                                                        cells = cells.child(
                                                            number(
                                                                format!("{value:.6}"),
                                                                11.,
                                                                ink(),
                                                            )
                                                            .grow(1.)
                                                            .min_w(0.)
                                                            .truncate(),
                                                        );
                                                    }
                                                }
                                                cells
                                            },
                                        )
                                        .virtual_rows_version((
                                            frame.info.request.clone(),
                                            frame.info.time.to_bits(),
                                        )),
                                ),
                        ),
                );
            let poses_panel = poses_panel.child(
                label(
                    format!(
                        "View coordinates: {} · {}",
                        poses.axes[0].frame,
                        unit_name(&poses.axes[0].unit)
                    ),
                    11.,
                    muted(),
                )
                .wrap(),
            );
            panel = panel.child(poses_panel);
        }
    }
    if !compact {
        section = section.child(panel);
    }
    section
}

pub fn telemetry(state: &AppState, job: &JobView, compact: bool, expanded: bool) -> El {
    let Some(progress) = state
        .project
        .as_ref()
        .and_then(|project| {
            project
                .document
                .jobs
                .iter()
                .find(|stored| stored.id == job.id)
        })
        .map(|job| &job.telemetry)
    else {
        return el();
    };
    let mut body = col().gap(9.).w_full();
    let active = matches!(job.status, JobStatus::Running | JobStatus::Paused);
    // A finished job's internal steps are a trace, shown on request.
    let trace_open = state.ui.worker_trace.as_deref() == Some(job.id.as_str());
    if !active && expanded && !compact {
        let id = job.id.clone();
        body = body.child(
            quiet_button(
                &format!("worker-trace-{}", job.id),
                if trace_open {
                    "Hide worker trace"
                } else {
                    "Show worker trace"
                },
                trace_open,
                move |s, _| {
                    s.ui.worker_trace = if s.ui.worker_trace.as_deref() == Some(id.as_str()) {
                        None
                    } else {
                        Some(id.clone())
                    }
                },
            )
            .h(30.),
        );
    }
    // Convergence plots are results and always show; the internal steps are the trace.
    let show_steps = active || trace_open;
    if active || expanded {
        let tree = progress.tree();
        let mut tasks = col().key("current-task-tree").gap(9.).w_full();
        for (depth, task) in &tree {
            if !expanded && !task.state.active() {
                continue;
            }
            let count = task_count(task);
            let mut line_view = col()
                .key(format!("task-{}-{}", task.request, task.id))
                .w_full()
                .gap(4.)
                .pl((*depth).min(8) as f32 * 14.)
                .child(
                    row()
                        .gap(10.)
                        .items_center()
                        .child(label(task.title(), 11., ink()).wrap().grow(1.).min_w(0.))
                        .child(number(count, 11., muted())),
                );
            if let Some(fraction) = task.fraction() {
                line_view = line_view.child(
                    el().h(3.)
                        .w_full()
                        .radius(1.5)
                        .fill(line())
                        .child(el().h_full().w_frac(fraction).fill(forest()).radius(1.5)),
                );
            }
            if task.state.active()
                && let Some(eta) = task.eta
            {
                line_view = line_view.child(label(
                    format!(
                        "ETA {:.1} s of active work{}",
                        eta,
                        if task.state == TaskState::Paused {
                            " · paused"
                        } else {
                            ""
                        }
                    ),
                    11.,
                    muted(),
                ));
            }
            if !compact && expanded {
                for metric in task.metrics.values() {
                    let value = metric
                        .samples
                        .last()
                        .map(|sample| format_value(&sample.value))
                        .unwrap_or_else(|| "—".into());
                    line_view = line_view.child(
                        row()
                            .gap(9.)
                            .items_center()
                            .child(
                                label(metric.name.replace('_', " "), 11., muted())
                                    .grow(1.)
                                    .min_w(0.)
                                    .wrap(),
                            )
                            .child(number(
                                format!(
                                    "{value} {}",
                                    metric.unit.as_deref().unwrap_or("unit not declared")
                                ),
                                11.,
                                ink(),
                            )),
                    );
                }
            }
            tasks = tasks.child(line_view);
        }
        let mut histories = col().gap(12.).w_full();
        let mut history_count = 0;
        if !compact {
            for (_, task) in tree {
                if !expanded && !task.state.active() {
                    continue;
                }
                let mut plots = row()
                    .key(format!("metric-plots-{}-{}", task.request, task.id))
                    .gap(18.)
                    .w_full();
                let mut count = 0;
                for (name, title) in [
                    ("objective", "Alignment objective"),
                    ("pose_step_rms", "Accepted pose step"),
                ] {
                    if let Some(metric) = task.metrics.get(name) {
                        plots = plots.child(metric_plot(metric, title).grow(1.).min_w(0.));
                        count += 1;
                    }
                }
                if count > 0 {
                    histories = histories.child(
                        col()
                            .gap(8.)
                            .w_full()
                            .child(strong(task.title(), 11., muted()))
                            .child(plots),
                    );
                    history_count += 1;
                }
            }
        }
        if !compact && !expanded && history_count > 0 {
            body = body.child(
                row()
                    .gap(24.)
                    .w_full()
                    .child(tasks.grow(1.).min_w(0.))
                    .child(histories.grow(2.).min_w(0.)),
            );
        } else {
            if show_steps {
                body = body.child(tasks);
            }
            if history_count > 0 {
                body = body.child(histories);
            }
        }
    }
    let notes: Vec<_> = progress
        .notes
        .iter()
        .filter(|note| matches!(note.level.as_str(), "warning" | "error" | "critical"))
        .collect();
    let show = if expanded { notes.len() } else { 3 };
    if !notes.is_empty() {
        body = body.child(strong(
            format!(
                "{} recorded warning{}",
                notes.len(),
                if notes.len() == 1 { "" } else { "s" }
            ),
            11.,
            warning(),
        ));
        for note in notes.iter().rev().take(show).rev() {
            body = body.child(
                label(
                    format!("{:.2} s · {}", note.time, note.message),
                    11.,
                    warning(),
                )
                .wrap(),
            );
        }
    }
    body
}

fn format_value(value: &serde_json::Value) -> String {
    value
        .as_f64()
        .map(|value| {
            if value.abs() > 1e4 || (value != 0. && value.abs() < 0.001) {
                format!("{value:.3e}")
            } else {
                format!("{value:.5}")
            }
        })
        .unwrap_or_else(|| {
            value
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| "—".into())
        })
}

fn task_count(task: &TaskProgress) -> String {
    let count = task
        .total
        .map(|total| format!("{} / {total} {}", task.done, task.unit))
        .unwrap_or_else(|| {
            if task.done > 0 {
                format!("{} {}", task.done, task.unit)
            } else {
                String::new()
            }
        });
    let status = match task.state {
        TaskState::Running => "",
        TaskState::Paused => "paused",
        TaskState::Done => "complete",
        TaskState::Failed => "failed",
        TaskState::Cancelled => "cancelled",
    };
    match (count.is_empty(), status.is_empty()) {
        (false, false) => format!("{count} · {status}"),
        (true, _) => status.into(),
        _ => count,
    }
}

/// Bound drawing geometry, while retaining the full worker history in the job.
/// Each bucket contributes its original extrema in sample order. Null/category
/// samples between retained points split the path; thinning never joins a gap.
fn plot_paths(samples: &[MetricSample], point_budget: usize) -> Vec<Vec<(u64, f64)>> {
    let width = samples.len().div_ceil((point_budget / 2).max(1)).max(1);
    let mut paths = Vec::new();
    let mut path = Vec::new();
    let mut previous = 0;
    for (bucket, chunk) in samples.chunks(width).enumerate() {
        let mut finite = chunk.iter().enumerate().filter_map(|(index, sample)| {
            sample
                .value
                .as_f64()
                .filter(|value| value.is_finite())
                .map(|value| (index, value))
        });
        let Some(initial) = finite.next() else {
            continue;
        };
        let mut low = initial;
        let mut high = initial;
        for point in finite {
            if point.1.total_cmp(&low.1).is_lt() {
                low = point;
            }
            if point.1.total_cmp(&high.1).is_ge() {
                high = point;
            }
        }
        let (first, second) = if low.0 <= high.0 {
            (low, high)
        } else {
            (high, low)
        };
        for (position, (offset, value)) in [first, second].into_iter().enumerate() {
            if position == 1 && first.0 == second.0 {
                continue;
            }
            let index = bucket * width + offset;
            let has_gap = samples[previous..index]
                .iter()
                .any(|sample| !sample.value.as_f64().is_some_and(|value| value.is_finite()));
            if has_gap && !path.is_empty() {
                paths.push(std::mem::take(&mut path));
            }
            path.push((samples[index].step, value));
            previous = index + 1;
        }
    }
    if !path.is_empty() {
        paths.push(path);
    }
    paths
}

fn metric_plot(metric: &MetricHistory, title: &str) -> El {
    let mut plot = col().gap(5.).child(strong(title, 11., ink())).child(label(
        metric.unit.as_deref().unwrap_or("unit not declared"),
        11.,
        muted(),
    ));
    let mut first = u64::MAX;
    let mut last = 0;
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;
    for sample in &metric.samples {
        if let Some(value) = sample.value.as_f64().filter(|value| value.is_finite()) {
            first = first.min(sample.step);
            last = last.max(sample.step);
            low = low.min(value);
            high = high.max(value);
        }
    }
    if !low.is_finite() {
        return plot.child(help("No measured samples yet"));
    }
    let range = (high - low).max(high.abs() * 0.01).max(1e-15);
    let mut shape =
        Shape::new()
            .view_box(300., 100.)
            .stroke(vec![[0., 97.], [300., 97.]], 0.7, line());
    for points in plot_paths(&metric.samples, 600) {
        let mut path: Vec<_> = points
            .into_iter()
            .map(|(step, value)| {
                [
                    ((step - first) as f64 / (last - first).max(1) as f64 * 300.) as f32,
                    (94. - (value - low) / range * 84.) as f32,
                ]
            })
            .collect();
        if path.len() == 1 {
            let point = path[0];
            path.push([point[0] + 0.5, point[1]]);
        }
        shape = shape.stroke(path, 1.3, forest());
    }
    plot = plot
        .child(label(format!("Range {low:.4}–{high:.4}"), 11., muted()))
        .child(
            canvas(shape)
                .key(format!("metric-plot-{}", metric.name))
                .h(96.)
                .w_full()
                .role("image")
                .label(format!(
                    "{title} against step number, {}",
                    metric.unit.as_deref().unwrap_or("unit not declared")
                )),
        )
        .child(
            row()
                .justify_between()
                .child(number(format!("step {first}"), 11., muted()))
                .child(number(last.to_string(), 11., muted())),
        );
    plot
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn plot_geometry_is_bounded_and_never_bridges_null_samples() {
        let samples: Vec<_> = (0..10_000)
            .map(|step| MetricSample {
                step,
                time: step as f64,
                value: if step == 4_997 || step == 5_003 {
                    json!(null)
                } else if step == 3_997 {
                    json!(1000.)
                } else {
                    json!(step as f64 / 10_000.)
                },
            })
            .collect();
        let paths = plot_paths(&samples, 600);
        assert!(paths.iter().map(Vec::len).sum::<usize>() <= 600);
        assert!(
            paths
                .iter()
                .flatten()
                .any(|&(step, value)| step == 3_997 && value == 1000.)
        );
        for gap in [4_997, 5_003] {
            assert!(
                paths
                    .iter()
                    .all(|path| !path.iter().any(|point| point.0 < gap)
                        || !path.iter().any(|point| point.0 > gap))
            );
        }
        assert_eq!(
            samples.len(),
            10_000,
            "display thinning must not discard the job history"
        );
    }

    #[test]
    fn repeated_step_samples_and_categories_remain_separate() {
        let samples = [
            json!(2.),
            json!(null),
            json!(4.),
            json!("stalled"),
            json!(3.),
        ]
        .into_iter()
        .map(|value| MetricSample {
            step: 7,
            time: 1.,
            value,
        })
        .collect::<Vec<_>>();
        assert_eq!(
            plot_paths(&samples, 600),
            vec![vec![(7, 2.)], vec![(7, 4.)], vec![(7, 3.)]]
        );
    }
}
