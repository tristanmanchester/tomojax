//! The connection catalog and the file browser occupy the central workspace.
use super::{controls::*, model::UiAction, tokens::*};
use crate::{
    AppState,
    app::files::{Action, Panel, Purpose},
};
use rune::{El, ScrollSpec, col, row};
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
                Purpose::Bundle => "Export array bundle",
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
    let mut fields = col().w_full().max_w(650.).gap(10.);
    for (i, (title, hint)) in [
        ("Connection name", "Workstation"),
        ("SSH destinations", "gateway compute-node"),
        ("Python interpreter", "python3"),
        ("Session launcher", "Optional"),
        ("Job launcher", "inline, process or slurm"),
        ("Scheduler options", "Optional"),
        ("Default results folder", "Optional"),
    ]
    .into_iter()
    .enumerate()
    {
        if i == 0 && panel.original.is_some() {
            fields = fields.child(property(title, &panel.editor[0].text));
        } else {
            fields = fields.child(field_hint(
                &format!("connection-field-{i}"),
                &panel.editor[i],
                title,
                "",
                hint,
                move |s| &mut s.files.editor[i],
            ));
        }
    }
    fields.child(action(
        "save-profile",
        "Save connection",
        true,
        true,
        Action::Save,
    ))
}
fn catalog(state: &AppState) -> El {
    let panel = &state.files;
    let mut list = col().gap(10.).max_w(740.).w_full();
    if panel.creating {
        list=list.child(help("Choose the machine that holds the scans. Studio saves the project on this computer."))
                    .child(action("choose-local","This computer",false,true,Action::Choose(None)));
    }
    for (i, name) in panel.names.iter().enumerate() {
        let pick = panel.creating || state.data_offline();
        list = list.child(
            row()
                .items_center()
                .gap(12.)
                .child(strong(name, 14., ink()).grow(1.))
                .child(action(
                    &format!("profile-edit-{i}"),
                    "Edit",
                    false,
                    true,
                    Action::Edit(Some(name.clone())),
                ))
                .child(action(
                    &format!("profile-use-{i}"),
                    "Use connection",
                    false,
                    pick,
                    Action::Choose(Some(name.clone())),
                )),
        );
    }
    list = list.child(action(
        "profile-add",
        "Add connection…",
        false,
        true,
        Action::Edit(None),
    ));
    if !panel.creating
        && let Some(name) = state.remote_project()
    {
        list = list
            .child(separator())
            .child(strong(name, 18., ink()))
            .child(label(state.connection_status(), 13., muted()));
        list = match state.connection_state() {
            Some(State::LoginNeeded) => {
                list.child(action("login", "Log in…", true, true, Action::Login))
            }
            Some(State::Blocked | State::Waiting(_) | State::Closed) => {
                list.child(action("retry", "Retry now", true, true, Action::Retry))
            }
            Some(State::Connected(_))
                if state
                    .project
                    .as_ref()
                    .is_some_and(|p| p.document.results.is_none()) =>
            {
                list.child(action(
                    "results",
                    "Choose results folder…",
                    true,
                    true,
                    Action::Browse(Purpose::Results),
                ))
            }
            Some(State::Connected(_)) if !state.data_offline() => list.child(action(
                "browse",
                "Browse files…",
                true,
                true,
                Action::Browse(Purpose::Browse),
            )),
            _ => list,
        };
    }
    list
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
        content=content.child(help(format!("Studio will create a folder named {project} here for this project’s inputs and results.")))
            .child(action("first-scan","Choose first scan…",false,!panel.loading,Action::Browse(Purpose::FirstScan)));
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
        let kind = if item.hdf5 { "HDF5" } else { &item.kind };
        let detail = if open {
            kind.to_string()
        } else {
            format!("{kind} · {} bytes", item.size)
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
                .child(icon(
                    if open { Icon::Folder } else { Icon::Layers },
                    15.,
                    forest(),
                ))
                .child(strong(&item.name, 12., ink()).grow(1.).min_w(0.).truncate())
                .child(label(detail, 12., muted())),
        );
    }
    content
}

fn folder_links(state: &AppState, mut content: El) -> El {
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
            "Results",
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
    content = content.child(links);
    if let Some(recent) = state
        .preferences
        .recent_folders
        .get(state.remote_project().unwrap_or("local"))
    {
        let mut links = row()
            .gap(8.)
            .items_center()
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
        content = content.child(links);
    }
    content
}
fn file_actions(state: &AppState, purpose: &Purpose, list: &crate::protocol::Listing) -> El {
    let panel = &state.files;
    let selected = panel.selected.as_ref();
    let enabled = !panel.loading;
    let accept = match purpose {
        Purpose::Results => "Use this folder",
        Purpose::FirstScan => "Use this scan",
        Purpose::Relink(..) => "Reconnect",
        Purpose::Import | Purpose::Folder => "Import",
        Purpose::Browse => "Done",
        Purpose::Bundle => "Export bundle",
    };
    let mut actions = row().gap(8.).child(action(
        "choose-worker-path",
        accept,
        true,
        enabled && (*purpose != Purpose::Results || (list.writable && selected.is_none())),
        Action::Accept,
    ));
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
