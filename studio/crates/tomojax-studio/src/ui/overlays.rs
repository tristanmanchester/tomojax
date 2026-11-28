//! Job visibility and keyboard fluency without taking the specimen out of context.
use super::{controls::*, model::*, tokens::*};
use crate::AppState;
use rune::{El, ScrollSpec, col, el, row, text_input};

pub fn jobs_workspace(state: &AppState) -> El {
    let mut content = col()
        .key("jobs-workspace")
        .padding(26.)
        .gap(20.)
        .h_full()
        .w_full()
        .min_h(0.)
        .min_w(0.)
        .child(
            row()
                .gap(20.)
                .items_center()
                .child(strong("Jobs", 26., ink()))
                .child(spacer())
                .child(action_button(
                    "jobs-prepare",
                    "Prepare more work",
                    false,
                    true,
                    UiAction::SetWorkspace(Workspace::Reconstruct),
                )),
        );
    if state.model.jobs.is_empty() {
        return content.child(
            el().grow(1.).center().child(
                col()
                    .max_w(390.)
                    .gap(16.)
                    .center()
                    .child(icon(Icon::Jobs, 36., forest()))
                    .child(strong("No jobs submitted", 21., ink()))
                    .child(action_button(
                        "jobs-start",
                        "Prepare reconstruction",
                        true,
                        true,
                        UiAction::SetWorkspace(Workspace::Reconstruct),
                    )),
            ),
        );
    }
    if state.live.frame().is_some() || state.live.loading() || state.live.error().is_some() {
        content = content.child(super::progress::live(state, false));
    }
    let mut list = col().gap(12.).w_full();
    for (title, category) in [("In progress", 0), ("Queued", 1), ("Processing record", 2)] {
        let jobs: Vec<_> = state
            .model
            .jobs
            .iter()
            .filter(|job| match category {
                0 => matches!(job.status, JobStatus::Running | JobStatus::Paused),
                1 => job.status == JobStatus::Queued,
                _ => !matches!(
                    job.status,
                    JobStatus::Running | JobStatus::Paused | JobStatus::Queued
                ),
            })
            .collect();
        if jobs.is_empty() {
            continue;
        }
        list = list.child(
            row()
                .gap(8.)
                .items_center()
                .pt(10.)
                .pb(3.)
                .child(strong(title, 13., ink()))
                .child(number(jobs.len().to_string(), 11., muted())),
        );
        for job in jobs {
            list = list.child(job_row(state, job, false));
        }
    }
    content = content.child(
        col()
            .key("jobs-scroll")
            .grow(1.)
            .min_h(0.)
            .w_full()
            .scroll(ScrollSpec {
                stick_bottom: false,
            })
            .child(list),
    );
    content
}

fn job_color(status: JobStatus) -> rune::Color {
    match status {
        JobStatus::Failed | JobStatus::Interrupted => error(),
        JobStatus::Paused => warning(),
        JobStatus::Running | JobStatus::Completed => forest(),
        _ => muted(),
    }
}

fn job_actions(job: &JobView, compact: bool) -> El {
    let mut actions = row().gap(7.).justify_end();
    let control = |key: &str, label: &str, action: fn(String) -> UiAction| {
        action_button(
            &format!("{key}-job-{}", job.id),
            label,
            false,
            true,
            action(job.id.clone()),
        )
        .h(30.)
    };
    match job.status {
        JobStatus::Running | JobStatus::Paused => {
            if job.status == JobStatus::Paused || job.pause_requested {
                actions = actions.child(control("resume", "Resume", UiAction::ResumeJob));
            } else if job.can_pause {
                actions = actions.child(control("pause", "Pause", UiAction::PauseJob));
            }
            actions = actions.child(control("cancel", "Cancel", UiAction::CancelJob));
        }
        JobStatus::Queued => {
            if !compact && job.waiting {
                for (key, label, icon, delta) in [
                    ("up", "Move earlier in queue", Icon::Up, -1),
                    ("down", "Move later in queue", Icon::Down, 1),
                ] {
                    let id = job.id.clone();
                    actions = actions.child(icon_button(
                        &format!("job-{key}-{id}"),
                        label,
                        icon,
                        true,
                        move |s, tx| {
                            s.dispatch(
                                UiAction::MoveJob {
                                    id: id.clone(),
                                    delta,
                                },
                                tx,
                            )
                        },
                    ));
                }
            }
            actions = actions.child(control(
                "cancel",
                if job.waiting { "Remove" } else { "Cancel" },
                UiAction::CancelJob,
            ));
        }
        JobStatus::Unknown => {
            actions = actions.child(control("abandon", "Abandon", UiAction::AbandonJob));
        }
        JobStatus::Failed | JobStatus::Interrupted | JobStatus::Canceled => {
            if job.can_resume {
                actions = actions.child(control("resume", "Resume", UiAction::ResumeJob));
            }
            actions = actions.child(control("retry", "Retry job", UiAction::RetryJob));
        }
        JobStatus::Completed => {
            if let Some(asset) = &job.result_asset {
                let id = asset.clone();
                actions = actions.child(
                    button(
                        &format!("open-job-result-{}", job.id),
                        "Explore result",
                        false,
                        true,
                        move |s, tx| {
                            s.dispatch(
                                UiAction::SelectAsset {
                                    id: id.clone(),
                                    extend: false,
                                },
                                tx,
                            );
                            s.dispatch(UiAction::SetWorkspace(Workspace::Explore), tx);
                            s.ui.activity.close();
                        },
                    )
                    .h(30.),
                );
            }
        }
    }
    actions
}

fn job_row(state: &AppState, job: &JobView, compact: bool) -> El {
    let expanded = !compact && state.ui.selected_job.as_deref() == Some(&job.id);
    let id = job.id.clone();
    let mut heading = row()
        .gap(12.)
        .items_center()
        .w_full()
        .child(icon(
            match job.status {
                JobStatus::Completed => Icon::Check,
                JobStatus::Failed | JobStatus::Interrupted => Icon::Alert,
                JobStatus::Paused => Icon::Pause,
                _ => Icon::Jobs,
            },
            18.,
            job_color(job.status),
        ))
        .child(
            col()
                .grow(1.)
                .min_w(0.)
                .gap(5.)
                .child(strong(&job.name, 12., ink()).truncate().help(&job.name))
                .child(label(&job.operation, 11., muted()).truncate()),
        )
        .child(label(job.state_label, 11., job_color(job.status)));
    if !compact && let Some(elapsed) = &job.elapsed {
        heading = heading.child(number(elapsed, 11., muted()));
    }
    if !compact {
        heading = heading.child(label(
            if expanded { "Hide details" } else { "Details" },
            11.,
            forest(),
        ));
    }
    let body = col()
        .key(format!("job-{}", job.id))
        .w_full()
        .min_w(0.)
        .padding(if compact { 13. } else { 17. })
        .gap(12.)
        .child(separator())
        .child(
            row()
                .gap(14.)
                .items_center()
                .w_full()
                .child(
                    el().role("button")
                        .key(format!("job-details-{}", job.id))
                        .label(format!(
                            "{}: {}. {} progress history, metrics and captured settings",
                            job.name,
                            job.state_label,
                            if expanded { "Hide" } else { "Show" }
                        ))
                        .focusable(!compact)
                        .focus_ring(1.5, 2., forest())
                        .grow(1.)
                        .min_w(0.)
                        .on_press(move |s: &mut AppState| {
                            s.ui.selected_job = if expanded { None } else { Some(id.clone()) }
                        })
                        .child(heading),
                )
                .child(job_actions(job, compact)),
        );
    job_body(state, job, body, compact, expanded)
}

fn job_body(state: &AppState, job: &JobView, mut body: El, compact: bool, expanded: bool) -> El {
    if !job.stage.is_empty()
        && job.stage != job.state_label
        && job.error.as_ref() != Some(&job.stage)
    {
        body = body.child(label(&job.stage, 11., muted()).wrap());
    }
    if let Some(progress) = job
        .progress
        .filter(|_| matches!(job.status, JobStatus::Running | JobStatus::Paused))
    {
        body = body.child(
            row()
                .gap(10.)
                .items_center()
                .child(
                    el().grow(1.).h(4.).radius(2.).fill(line()).child(
                        el().w_frac(progress.clamp(0., 1.))
                            .h_full()
                            .radius(2.)
                            .fill(forest()),
                    ),
                )
                .child(number(
                    format!("{:.0}%", progress.clamp(0., 1.) * 100.),
                    11.,
                    muted(),
                )),
        );
    }
    if let Some(error) = &job.error {
        body = body.child(label(error, 11.5, tokens::error()).wrap());
    }
    if compact || expanded {
        if !job.scheduler_options.is_empty() {
            body = body.child(property("Scheduler options", &job.scheduler_options));
        }
        if !job.devices.is_empty() {
            body = body.child(property("Job devices", &job.devices));
        }
    }
    body = body.child(super::progress::telemetry(state, job, compact, expanded));
    if expanded && (!job.settings.is_empty() || !job.inputs.is_empty()) {
        let mut details =
            col()
                .gap(8.)
                .child(separator())
                .child(strong("Captured settings", 11., muted()));
        if !job.inputs.is_empty() {
            details = details.child(property("Inputs", &job.inputs));
        }
        for (name, value) in &job.settings {
            details = details.child(property(name, value));
        }
        body = body.child(details);
    }
    job_files(state, job, body, expanded || compact)
}

fn job_files(state: &AppState, job: &JobView, mut body: El, show_files: bool) -> El {
    if state.remote_project().is_some()
        && let Some(record) = state
            .project
            .as_ref()
            .and_then(|p| p.document.jobs.iter().find(|j| j.id == job.id))
    {
        if state.data_offline() || (record.script.is_some() && !state.job_seen(&job.id)) {
            let seen = record
                .script
                .as_ref()
                .and_then(|s| s.seen)
                .or(record.finished)
                .or(record.started)
                .unwrap_or(record.submitted);
            let seconds = crate::project::now_ms().saturating_sub(seen) / 1000;
            body = body.child(label(
                format!("Last recorded state · {seconds} s ago"),
                11.,
                muted(),
            ));
        } else if job.status == JobStatus::Completed && show_files {
            let files = record.outputs.iter();
            let destination = record
                .destination
                .iter()
                .filter(|_| record.outputs.is_empty());
            for (i, path) in files.chain(destination).enumerate() {
                body = body.child(
                    action_button(
                        &format!("job-file-{}-{i}", job.id),
                        &format!("Show {}", path.name()),
                        false,
                        true,
                        UiAction::Files(crate::app::files::Action::Output(path.clone())),
                    )
                    .help(path.to_string()),
                );
            }
        }
    }
    body
}

pub fn jobs_drawer(state: &AppState) -> El {
    let jobs: Vec<_> = state
        .model
        .jobs
        .iter()
        .filter(|j| {
            matches!(
                j.status,
                JobStatus::Running | JobStatus::Paused | JobStatus::Queued | JobStatus::Unknown
            ) || state.ui.activity.tracks(&j.id)
        })
        .collect();
    let mut rows = col().gap(8.);
    if state.live.frame().is_some() || state.live.loading() {
        rows = rows.child(super::progress::live(state, true));
    }
    if jobs.is_empty() {
        rows = rows.child(help("No active jobs"));
    }
    for job in jobs {
        rows = rows.child(job_row(state, job, true));
    }
    // A panel under its top-bar trigger, sized to its content. It opens by itself when work is
    // queued, so it never blocks the rest of the window; Escape or its close button dismisses it.
    col()
        .key("jobs-drawer")
        .role("region")
        .label("Activity")
        .w(440.)
        .max_h(520.)
        .pin(None, Some(70.), Some(18.), None)
        .radius(10.)
        .fill(paper())
        .shadow(8., 30., ink().alpha(0.18))
        .child(
            row()
                .h(55.)
                .px(18.)
                .gap(10.)
                .items_center()
                .child(strong("Activity", 15., ink()))
                .child(spacer())
                .child(icon_button(
                    "close-jobs",
                    "Close activity",
                    Icon::Close,
                    true,
                    |s, _| s.ui.activity.close(),
                )),
        )
        .child(separator())
        .child(
            col()
                .key("jobs-drawer-scroll")
                .shrink()
                .min_h(0.)
                .w_full()
                .scroll(ScrollSpec {
                    stick_bottom: false,
                })
                .child(rows.padding(14.)),
        )
        .child(separator())
        .child(
            button("all-jobs", "View all jobs", false, true, |s, tx| {
                s.ui.activity.close();
                s.dispatch(UiAction::SetWorkspace(Workspace::Jobs), tx);
            })
            .w_full()
            .h(42.),
        )
}

fn matching_commands(state: &AppState) -> Vec<&CommandView> {
    let query = state.ui.command_query.text.to_lowercase();
    state
        .model
        .commands
        .iter()
        .filter(|command| {
            command.enabled
                && format!("{} {}", command.category, command.title)
                    .to_lowercase()
                    .contains(&query)
        })
        .collect()
}

pub fn command_palette(state: &AppState) -> El {
    let commands = matching_commands(state);
    let mut results = col().gap(3.).padding(8.);
    if commands.is_empty() {
        results = results.child(help("No matching commands").padding(18.));
    }
    for (index, command) in commands.into_iter().enumerate() {
        let selected = index == state.ui.command_index;
        let id = command.id.clone();
        results = results.child(
            row()
                .key(format!("command-{}", command.id))
                .role("option")
                .label(&command.title)
                .focus_selected(selected)
                .focusable(true)
                .w_full()
                .min_h(43.)
                .px(12.)
                .gap(12.)
                .items_center()
                .radius(5.)
                .fill(if selected { sage() } else { paper() })
                .hover(forest().alpha(0.05))
                .focus_ring(1.5, 0., forest())
                .on_press_tx(move |s: &mut AppState, tx| {
                    s.dispatch(UiAction::RunCommand(id.clone()), tx)
                })
                .child(strong(&command.title, 12., ink()))
                .child(label(&command.category, 11., muted()))
                .child(spacer())
                .child(number(&command.binding, 11., muted())),
        );
    }
    let search = text_input(&state.ui.command_query, type_style(15., 400.))
        .key("command-search")
        .label("Search commands")
        .placeholder("Search commands, workspaces and actions…", muted())
        .grow(1.)
        .min_w(0.)
        .single_line()
        .color(ink())
        .autofocus()
        .on_edit(|s: &mut AppState| {
            s.ui.command_index = 0;
            &mut s.ui.command_query
        })
        .on_submit(|s: &mut AppState, tx| {
            if let Some(command) = matching_commands(s).get(s.ui.command_index) {
                let id = command.id.clone();
                s.dispatch(UiAction::RunCommand(id), tx);
            }
        });
    let panel = col()
        .key("command-dialog")
        .role("dialog")
        .label("Command palette")
        .w(650.)
        .max_h(540.)
        .radius(12.)
        .fill(paper())
        .shadow(14., 44., ink().alpha(0.2))
        .modal(true, |s: &mut AppState| s.ui.palette_open = false)
        .focus_return("command-palette")
        .child(
            row()
                .h(64.)
                .px(20.)
                .gap(14.)
                .items_center()
                .child(icon(Icon::Search, 20., forest()))
                .child(search)
                .child(icon_button(
                    "close-palette",
                    "Close command palette",
                    Icon::Close,
                    true,
                    |s, _| s.ui.palette_open = false,
                )),
        )
        .child(separator())
        .child(
            col()
                .key("command-results")
                .role("listbox")
                .label("Matching commands")
                .max_h(400.)
                .w_full()
                .scroll(ScrollSpec {
                    stick_bottom: false,
                })
                .child(results),
        )
        .child(separator())
        .child(
            label(
                "Arrow keys to choose  ·  Enter to run  ·  Escape to close",
                11.,
                muted(),
            )
            .px(20.)
            .py(12.),
        );
    col()
        .w_full()
        .h_full()
        .absolute(0., 0.)
        .fill(ink().alpha(0.12))
        .items_center()
        .pt(106.)
        .child(panel)
}

pub fn shortcuts(state: &AppState) -> El {
    let query = state.ui.shortcut_query.text.to_lowercase();
    let mut rows = col().gap(5.).padding(18.);
    for shortcut in state.model.shortcuts.iter().filter(|s| {
        format!(
            "{} {} {} {}",
            s.title,
            s.category,
            s.binding,
            crate::preferences::binding_label(&s.binding)
        )
        .to_lowercase()
        .contains(&query)
    }) {
        let selected = state.ui.editing_shortcut.as_deref() == Some(&shortcut.id);
        let mut item = col().gap(10.).py(10.).w_full();
        item = item.child(
            row()
                .gap(12.)
                .items_center()
                .w_full()
                .child(
                    col()
                        .gap(4.)
                        .grow(1.)
                        .min_w(0.)
                        .child(strong(&shortcut.title, 12., ink()))
                        .child(label(&shortcut.category, 11., muted())),
                )
                .child(
                    action_button(
                        &format!("edit-shortcut-{}", shortcut.id),
                        &crate::preferences::binding_label(&shortcut.binding),
                        false,
                        true,
                        UiAction::BeginShortcutEdit(shortcut.id.clone()),
                    )
                    .min_w(104.),
                )
                .child(icon_button(
                    &format!("reset-shortcut-{}", shortcut.id),
                    "Restore default binding",
                    Icon::Refresh,
                    shortcut.binding != shortcut.default_binding,
                    {
                        let id = shortcut.id.clone();
                        move |s, tx| s.dispatch(UiAction::ResetShortcut(id.clone()), tx)
                    },
                )),
        );
        if selected {
            let id = shortcut.id.clone();
            item = item.child(
                row()
                    .gap(8.)
                    .items_center()
                    .w_full()
                    .child(
                        input(
                            "shortcut-binding",
                            &state.ui.shortcut_binding,
                            "Shortcut binding",
                            "",
                            |s| &mut s.ui.shortcut_binding,
                        )
                        .grow(1.),
                    )
                    .child(quiet_button(
                        "record-shortcut",
                        if state.ui.recording_shortcut {
                            "Press keys…"
                        } else {
                            "Record"
                        },
                        state.ui.recording_shortcut,
                        |s, _| s.ui.recording_shortcut = !s.ui.recording_shortcut,
                    ))
                    .child(button(
                        "save-shortcut",
                        "Apply",
                        true,
                        true,
                        move |s, tx| {
                            s.dispatch(
                                UiAction::SaveShortcut {
                                    command: id.clone(),
                                    binding: s.ui.shortcut_binding.text.clone(),
                                },
                                tx,
                            )
                        },
                    ))
                    .child(button("cancel-shortcut", "Cancel", false, true, |s, _| {
                        s.ui.editing_shortcut = None
                    })),
            );
        }
        rows = rows.child(item).child(separator());
    }
    let mut panel = col()
        .key("shortcuts-dialog")
        .role("dialog")
        .label("Keyboard shortcuts")
        .w(720.)
        .h(640.)
        .radius(12.)
        .fill(paper())
        .shadow(14., 44., ink().alpha(0.2))
        .modal(true, |s: &mut AppState| s.ui.shortcuts_open = false)
        .focus_return("shortcut-settings")
        .child(
            col()
                .padding(24.)
                .gap(16.)
                .child(
                    row()
                        .gap(12.)
                        .items_center()
                        .child(strong("Keyboard shortcuts", 23., ink()))
                        .child(spacer())
                        .child(action_button(
                            "documentation",
                            "Documentation",
                            false,
                            true,
                            UiAction::RunCommand("documentation".into()),
                        ))
                        .child(icon_button(
                            "close-shortcuts",
                            "Close keyboard shortcuts",
                            Icon::Close,
                            true,
                            |s, _| s.ui.shortcuts_open = false,
                        )),
                )
                .child(search_input(
                    "shortcut-search",
                    &state.ui.shortcut_query,
                    "Find a shortcut",
                    |s| &mut s.ui.shortcut_query,
                )),
        )
        .child(separator())
        .child(
            col()
                .key("shortcuts-scroll")
                .grow(1.)
                .min_h(0.)
                .w_full()
                .scroll(ScrollSpec {
                    stick_bottom: false,
                })
                .child(rows),
        );
    if let Some(error) = &state.model.shortcut_error {
        panel = panel.child(
            label(error, 12., tokens::error())
                .role("alert")
                .wrap()
                .px(24.)
                .py(12.),
        );
    }
    if state.ui.editing_shortcut.is_some() {
        panel = panel.child(separator()).child(
            label(
                "Format: mod+shift+r. mod is Ctrl, or Cmd on macOS.",
                11.,
                muted(),
            )
            .wrap()
            .px(24.)
            .py(14.),
        );
    }
    el().w_full()
        .h_full()
        .absolute(0., 0.)
        .fill(ink().alpha(0.12))
        .center()
        .child(panel)
}

use super::tokens;
