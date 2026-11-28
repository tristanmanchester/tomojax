//! The connection catalog and the file browser occupy the central workspace.
use super::{controls::*, model::UiAction, tokens::*};
use crate::{
    AppState,
    app::files::{Action, Panel, Purpose},
};
use rune::{El, ScrollSpec, col, el, row};
use studio_core::connection::State;

fn action(key: &str, title: &str, primary: bool, enabled: bool, act: Action) -> El {
    action_button(key, title, primary, enabled, UiAction::Files(act))
}

pub fn view(state: &AppState) -> El {
    let panel = &state.files;
    let (title, body) = match &panel.panel {
        Panel::Connections => (
            if panel.creating {
                "Where are the project’s scans?"
            } else {
                "Connections"
            },
            catalog(state),
        ),
        Panel::Editor => ("Connection settings", editor(state)),
        Panel::Browser(purpose) => {
            let title = match purpose {
                Purpose::Results => "Where should this project save results?",
                Purpose::FirstScan => "Choose the first scan",
                Purpose::Import => "Import a scan",
                Purpose::Folder => "Import an acquisition folder",
                Purpose::Relink(..) => "Reconnect the original data",
                Purpose::Browse => "Files",
                Purpose::Export { .. } => "Choose an output folder",
            };
            (title, browser(state, purpose))
        }
        Panel::Closed => return col(),
    };
    col()
        .padding(24.)
        .gap(18.)
        .grow(1.)
        .min_h(0.)
        .min_w(0.)
        .child(
            row()
                .w_full()
                .items_center()
                .gap(12.)
                .child(strong(title, 23., ink()).grow(1.))
                .child(action("close-files", "Close", false, true, Action::Close)),
        )
        .child(body.min_h(0.).grow(1.).scroll(ScrollSpec {
            stick_bottom: false,
        }))
}

fn editor(state: &AppState) -> El {
    let panel = &state.files;
    let launcher = launcher(panel);
    let mut fields = col().w_full().max_w(650.).gap(16.);
    for (i, (title, hint, explanation)) in [
        (
            "Connection name",
            "Workstation",
            "How Studio lists this machine.",
        ),
        (
            "SSH destinations",
            "gateway compute-node",
            "One or two hosts, separated by a space, in the order you reach them. Leave empty for this computer.",
        ),
        (
            "Python interpreter",
            "python3",
            "The Python environment with TomoJAX on the last host.",
        ),
        (
            "Session launcher",
            "Optional",
            "A command that TomoJAX runs under for interactive work, such as an srun line on a cluster.",
        ),
        ("Job launcher", "", ""),
        (
            "Scheduler options",
            "Optional",
            "Default Slurm options for each job, such as --time=02:00:00. You can change them per job.",
        ),
        (
            "Default results folder",
            "Optional",
            "Where a new project on this machine suggests saving its results.",
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let field = match i {
            0 if panel.original.is_some() => property(title, &panel.editor[0].text),
            4 => {
                fields = fields.child(launcher_choice(launcher));
                continue;
            }
            5 if launcher != "slurm" => continue,
            _ => field_hint(
                &format!("connection-field-{i}"),
                &panel.editor[i],
                title,
                "",
                hint,
                move |s| &mut s.files.editor[i],
            ),
        };
        fields = fields.child(col().w_full().gap(5.).child(field).child(help(explanation)));
    }
    fields.child(row().pt(4.).child(action(
        "save-profile",
        "Save connection",
        true,
        true,
        Action::Save,
    )))
}

/// The launcher the editor shows as chosen: the saved one, or the worker's default for
/// the SSH destinations entered (inline on this computer, a process elsewhere).
fn launcher(panel: &crate::app::files::FilePanel) -> &str {
    match panel.editor[4].text.trim() {
        "" if panel.editor[1].text.trim().is_empty() => "inline",
        "" => "process",
        chosen => chosen,
    }
}

fn launcher_choice(chosen: &str) -> El {
    let mut options = col()
        .key("job-launcher")
        .role("radiogroup")
        .label("Job launcher")
        .w_full()
        .gap(6.)
        .child(label("Job launcher", 11., muted()));
    for (value, title, meaning) in [
        (
            "inline",
            "In the session",
            "Runs in TomoJAX’s session while Studio is connected. Disconnecting interrupts it.",
        ),
        (
            "process",
            "Separate process",
            "Keeps running on that machine after you disconnect.",
        ),
        (
            "slurm",
            "Slurm batch job",
            "Queued with Slurm and run on the cluster, with or without Studio.",
        ),
    ] {
        let selected = chosen == value;
        options = options.child(
            row()
                .key(format!("job-launcher-{value}"))
                .role("radio")
                .label(title)
                .checked(selected)
                .focusable(true)
                .w_full()
                .px(12.)
                .py(9.)
                .gap(10.)
                .items_center()
                .radius(6.)
                .border(1., if selected { forest() } else { line() })
                .fill(if selected { sage() } else { paper() })
                .hover(forest().alpha(0.06))
                .focus_ring(1.5, 1., forest())
                .on_press(move |s: &mut AppState| {
                    s.files.editor[4] = rune::TextEdit::new(value);
                })
                .child(
                    el().size(14., 14.)
                        .radius(7.)
                        .border(
                            if selected { 4.5 } else { 1.5 },
                            if selected { forest() } else { muted() },
                        )
                        .fill(paper()),
                )
                .child(
                    col()
                        .grow(1.)
                        .min_w(0.)
                        .gap(3.)
                        .child(strong(title, 12., ink()))
                        .child(label(meaning, 11., muted()).wrap().w_full()),
                ),
        );
    }
    options
}

/// One machine in the catalog: its name, what Studio knows about it, and what can be done next.
fn machine(key: &str, name: &str, status: Option<(String, rune::Color)>, actions: El) -> El {
    let mut text = col()
        .grow(1.)
        .min_w(0.)
        .gap(5.)
        .child(strong(name, 13., ink()).truncate().help(name));
    if let Some((status, tone)) = status {
        text = text.child(
            row()
                .gap(7.)
                .items_center()
                .child(status_dot(tone))
                .child(label(status, 12., muted()).wrap()),
        );
    }
    row()
        .key(key)
        .w_full()
        .min_h(56.)
        .py(10.)
        .gap(12.)
        .items_center()
        .child(text)
        .child(actions)
}

fn catalog(state: &AppState) -> El {
    let panel = &state.files;
    let mut list = col().gap(0.).max_w(740.).w_full();
    let current = state.remote_project().filter(|_| !panel.creating);
    list = list.child(
        help(if panel.creating {
            "Choose the machine that holds the scans. Studio keeps the project itself on this computer."
                .to_owned()
        } else if let Some(name) = current {
            format!("This project’s scans are on {name}. Studio reaches it over SSH.")
        } else {
            "This project’s scans are on this computer.".to_owned()
        })
        .pb(12.),
    );
    if panel.creating {
        list = list.child(separator()).child(machine(
            "machine-local",
            "This computer",
            None,
            action(
                "choose-local",
                "Use this computer",
                false,
                true,
                Action::Choose(None),
            ),
        ));
    }
    for (i, name) in panel.names.iter().enumerate() {
        let pick = panel.creating || state.data_offline();
        let mut actions = row().gap(8.).items_center().child(action(
            &format!("profile-edit-{i}"),
            "Edit",
            false,
            true,
            Action::Edit(Some(name.clone())),
        ));
        let mut status = None;
        if current == Some(name.as_str()) {
            let connection = state.connection_state();
            let tone = match connection {
                Some(State::Connected(_)) => forest(),
                Some(State::Blocked | State::LoginNeeded) => error(),
                _ => muted(),
            };
            status = Some((state.connection_status(), tone));
            if let Some(next) = connection_action(state, connection) {
                actions = actions.child(next);
            }
        } else if pick {
            actions = actions.child(action(
                &format!("profile-use-{i}"),
                "Use connection",
                false,
                true,
                Action::Choose(Some(name.clone())),
            ));
        }
        list =
            list.child(separator())
                .child(machine(&format!("machine-{i}"), name, status, actions));
        if current == Some(name.as_str())
            && let Some(problem) = state.connection_problem()
        {
            list = list.child(
                col()
                    .key("connection-problem")
                    .role("status")
                    .w_full()
                    .gap(4.)
                    .pb(12.)
                    .child(label(problem, 12., error()).wrap().w_full())
                    .child(help(
                        "Check the SSH destinations and Python interpreter with Edit, then choose Retry now.",
                    )),
            );
        }
    }
    list.child(separator()).child(row().pt(14.).child(action(
        "profile-add",
        "Add connection…",
        false,
        true,
        Action::Edit(None),
    )))
}

/// The next step for the project's own connection, from its state.
fn connection_action(state: &AppState, connection: Option<State>) -> Option<El> {
    Some(match connection? {
        State::LoginNeeded => action("login", "Log in…", true, true, Action::Login),
        State::Blocked | State::Waiting(_) | State::Closed => {
            action("retry", "Retry now", true, true, Action::Retry)
        }
        State::Connected(_)
            if state
                .project
                .as_ref()
                .is_some_and(|p| p.document.results.is_none()) =>
        {
            action(
                "results",
                "Choose results folder…",
                true,
                true,
                Action::Browse(Purpose::Results),
            )
        }
        State::Connected(_) if !state.data_offline() => action(
            "browse",
            "Browse files…",
            true,
            true,
            Action::Browse(Purpose::Browse),
        ),
        _ => return None,
    })
}

fn browser(state: &AppState, purpose: &Purpose) -> El {
    let panel = &state.files;
    let mut content = col().gap(12.).w_full();
    let name = state.remote_project().unwrap_or("This computer");
    content = content.child(strong(name, 14., forest()));
    if *purpose == Purpose::Results {
        let project = state
            .project
            .as_ref()
            .map(|p| p.document.name.as_str())
            .unwrap_or("");
        content = content
            .child(help(format!(
                "Studio creates a folder named {project} in the folder you choose, for this project’s inputs and results. To start from a scan, choose it first and Studio suggests its folder."
            )))
            .child(row().child(action(
                "first-scan",
                "Choose first scan…",
                false,
                !panel.loading,
                Action::Browse(Purpose::FirstScan),
            )));
    }
    if matches!(
        purpose,
        Purpose::FirstScan | Purpose::Import | Purpose::Folder
    ) {
        content = content.child(toggle(
            "copy-selected-source",
            "Copy sources into the project",
            state.ui.import_copy,
            true,
            |s, _| s.ui.import_copy = !s.ui.import_copy,
        ));
    }
    content = content.child(
        row()
            .w_full()
            .gap(8.)
            .items_center()
            .child(
                submit_input(
                    "worker-path",
                    &panel.path,
                    "Path",
                    UiAction::Files(Action::EnterPath),
                    |s| &mut s.files.path,
                )
                .grow(1.),
            )
            .child(action("list-path", "Go", false, true, Action::EnterPath)),
    );
    content = folder_links(state, content);
    if let Some(message) = &panel.transfer {
        content = content.child(label(message, 12., forest()).role("status").wrap());
    }
    let Some(list) = &panel.listing else {
        return content.child(help(if panel.loading {
            "Reading folder…"
        } else {
            "Enter a path to browse."
        }));
    };
    let selected = panel.selected.as_ref();
    let enabled = !panel.loading;
    content = content
        .child(file_actions(state, purpose, list))
        .child(separator());
    content = file_rows(content, list, selected);
    if list.after.is_some() {
        content = content.child(action(
            "more-files",
            "More files",
            false,
            enabled,
            Action::More,
        ));
    }
    content
}

fn file_rows(
    mut content: El,
    list: &crate::protocol::Listing,
    selected: Option<&(studio_core::worker_path::WorkerPath, u64)>,
) -> El {
    for (i, item) in list.entries.iter().enumerate() {
        let path = if list.directory {
            list.path.join(&item.name)
        } else {
            list.path.clone()
        };
        let open = item.kind == "directory" || item.kind == "symlink";
        let chosen = selected.is_some_and(|(p, _)| p == &path);
        let act = if open {
            Action::List(Some(path))
        } else {
            Action::Select(path, item.size)
        };
        let kind = match item.kind.as_str() {
            _ if item.hdf5 => "HDF5",
            "directory" => "Folder",
            "symlink" => "Link",
            "file" => "File",
            other => other,
        };
        let detail = if open {
            kind.to_string()
        } else {
            format!("{kind} · {}", file_size(item.size))
        };
        content = content.child(
            row()
                .key(format!("file-{i}"))
                .role("option")
                .label(&item.name)
                .checked(chosen)
                .focusable(true)
                .w_full()
                .h(32.)
                .px(8.)
                .gap(10.)
                .items_center()
                .radius(4.)
                .fill(if chosen {
                    sage()
                } else {
                    rune::Color::TRANSPARENT
                })
                .hover(forest().alpha(0.06))
                .focus_ring(1.5, 0., forest())
                .on_press_tx(move |s: &mut AppState, tx| {
                    s.dispatch(UiAction::Files(act.clone()), tx)
                })
                .child(if open || item.hdf5 {
                    icon(
                        if open { Icon::Folder } else { Icon::Layers },
                        15.,
                        forest(),
                    )
                } else {
                    // Plain files carry no glyph, so folders and scans stand out.
                    el().size(15., 15.)
                })
                .child(strong(&item.name, 12., ink()).grow(1.).min_w(0.).truncate())
                .child(number(detail, 12., muted())),
        );
    }
    content
}

/// A file size in the units people read: `386 KB`, `1.2 GB`.
pub(crate) fn file_size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["KB", "MB", "GB", "TB"];
    if bytes < 1000 {
        return format!("{bytes} byte{}", if bytes == 1 { "" } else { "s" });
    }
    let mut value = bytes as f64;
    let mut unit = "";
    for name in UNITS {
        value /= 1000.;
        unit = name;
        if value < 1000. {
            break;
        }
    }
    if value < 10. {
        format!("{value:.1} {unit}")
    } else {
        format!("{value:.0} {unit}")
    }
}

fn folder_links(state: &AppState, content: El) -> El {
    let panel = &state.files;
    let mut links = row().gap(6.).items_center().child(action(
        "worker-home",
        "Home",
        false,
        true,
        Action::List(None),
    ));
    if let Some(results) = state
        .project
        .as_ref()
        .and_then(|p| p.document.results.as_ref())
        && state.remote_project().is_some()
    {
        links = links.child(action(
            "worker-results",
            "Results folder",
            false,
            true,
            Action::List(Some(results.clone())),
        ));
    }
    if let Some(parent) = panel.listing.as_ref().and_then(|l| l.path.parent()) {
        links = links.child(action(
            "worker-up",
            "Parent folder",
            false,
            true,
            Action::List(Some(parent)),
        ));
    }
    if let Some(recent) = state
        .preferences
        .recent_folders
        .get(state.remote_project().unwrap_or("local"))
        .filter(|recent| !recent.is_empty())
    {
        links = links
            .child(el().w(1.).h(20.).fill(line()))
            .child(label("Recent", 11., muted()));
        for (i, path) in recent.iter().take(3).enumerate() {
            links = links.child(
                action(
                    &format!("recent-folder-{i}"),
                    path.name(),
                    false,
                    true,
                    Action::List(Some(path.clone())),
                )
                .help(path.to_string()),
            );
        }
    }
    content.child(links)
}
fn file_actions(state: &AppState, purpose: &Purpose, list: &crate::protocol::Listing) -> El {
    let panel = &state.files;
    let selected = panel.selected.as_ref();
    let enabled = !panel.loading;
    let accept = match purpose {
        Purpose::Results | Purpose::Export { .. } => Some("Use this folder"),
        Purpose::FirstScan => Some("Use this scan"),
        Purpose::Relink(..) => Some("Reconnect"),
        Purpose::Import | Purpose::Folder => Some("Import"),
        // Browsing ends with Close; there is nothing to accept.
        Purpose::Browse => None,
    };
    let mut actions = row().gap(8.);
    if let Some(accept) = accept {
        actions = actions.child(action(
            "choose-worker-path",
            accept,
            true,
            enabled && (*purpose != Purpose::Results || (list.writable && selected.is_none())),
            Action::Accept,
        ));
    }
    if state.remote_project().is_some() && !state.data_offline() {
        actions = actions
            .child(action(
                "download",
                "Copy to this computer…",
                false,
                selected.is_some(),
                Action::Download,
            ))
            .child(action(
                "upload",
                "Copy from this computer…",
                false,
                list.directory && list.writable,
                Action::Upload,
            ));
    }
    actions
}
