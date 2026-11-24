//! Terrarium: a persistent project around a live scientific image workspace.
mod controls;
pub mod corrections;
mod inspector;
pub mod model;
mod overlays;
mod preview;
mod progress;
#[cfg(test)]
mod tests;
mod tokens;
mod viewer;

use crate::AppState;
use controls::*;
use model::*;
use rune::{Cx, El, ScrollSpec, col, el, row};
use tokens::*;

pub fn view(state: &AppState, cx: &mut Cx<AppState>) -> El {
    let narrow = cx
        .layout_with("studio-root", |r| r[2] < 1320.)
        .unwrap_or(false);
    let mut body = row().grow(1.).min_h(0.).min_w(0.).w_full();
    if state.model.project.is_some() && state.ui.tree_open {
        body = body
            .child(project_tree(state))
            .child(el().w(1.).h_full().fill(line()));
    }
    let main = if state.model.project.is_none() {
        welcome(state)
    } else if state.ui.workspace == Workspace::Jobs {
        overlays::jobs_workspace(state)
    } else {
        viewer::workspace(state)
    };
    body = body.child(main.grow(1.).min_w(0.).min_h(0.));
    if state.model.project.is_some()
        && state.ui.inspector_open
        && state.ui.workspace != Workspace::Jobs
    {
        body = body
            .child(el().w(1.).h_full().fill(line()))
            .child(inspector::view(state));
    }
    let mut result = col()
        .key("studio-root")
        .w_full()
        .h_full()
        .fill(background())
        .on_drop_files(|s: &mut AppState, files, _| s.import_paths(files.to_vec()))
        .child(topbar(state, narrow))
        .child(separator());
    if let Some(error) = &state.model.error {
        result = result.child(
            row()
                .key("error-banner")
                .role("alert")
                .px(20.)
                .py(10.)
                .gap(12.)
                .items_center()
                .fill(color("#FAEAE4"))
                .child(icon(Icon::Alert, 16., tokens::error()))
                .child(label(error, 12., tokens::error()).wrap().grow(1.).min_w(0.))
                .child(icon_button(
                    "dismiss-error",
                    "Dismiss message",
                    Icon::Close,
                    true,
                    |s, tx| s.dispatch(UiAction::DismissError, tx),
                )),
        );
    }
    result = result
        .child(body)
        .child(separator())
        .child(statusbar(state));
    if state.ui.jobs_open && state.ui.workspace != Workspace::Jobs {
        result = result.child(overlays::jobs_drawer(state));
    }
    if state.ui.palette_open {
        result = result.child(overlays::command_palette(state));
    }
    if state.ui.shortcuts_open {
        result = result.child(overlays::shortcuts(state));
    }
    result
}

fn topbar(state: &AppState, narrow: bool) -> El {
    let mut bar = row()
        .key("topbar")
        .h(64.)
        .w_full()
        .px(22.)
        .gap(12.)
        .items_center()
        .fill(paper())
        .child(icon(Icon::Layers, 27., forest()))
        .child(strong("TomoJAX", 19., ink()).min_w(94.));
    if state.model.project.is_some() {
        bar = bar.child(icon_button(
            "toggle-tree",
            "Show or hide project navigator",
            Icon::Tree,
            true,
            |s, _| s.ui.tree_open = !s.ui.tree_open,
        ));
    }
    let mut navigation = row()
        .gap(if narrow { 1. } else { 7. })
        .h_full()
        .items_center();
    for workspace in Workspace::ALL {
        let selected = workspace == state.ui.workspace;
        let available = state.model.project.is_some();
        let mut tab = el()
            .key(format!("workspace-{}", workspace.title().to_lowercase()))
            .role("tab")
            .label(workspace.title())
            .focus_selected(selected)
            .focusable(available)
            .disabled(!available)
            .h_full()
            .px(if narrow { 9. } else { 12. })
            .center()
            .hover(forest().alpha(0.04))
            .focus_ring(2., -3., forest())
            .opacity(if available { 1. } else { 0.45 })
            .on_press_tx(move |s: &mut AppState, tx| {
                if available {
                    s.dispatch(UiAction::SetWorkspace(workspace), tx)
                }
            })
            .child(strong(
                workspace.title(),
                12.,
                if selected { forest() } else { muted() },
            ));
        if selected && available {
            tab = tab.child(el().h(3.).radius(1.5).fill(color("#688E6E")).pin(
                Some(10.),
                None,
                Some(10.),
                Some(0.),
            ));
        }
        navigation = navigation.child(tab);
    }
    bar = bar.child(navigation).child(spacer());
    if let Some(project) = &state.model.project {
        bar = bar.child(
            label(&project.name, 12., ink())
                .min_w(0.)
                .max_w(if narrow { 140. } else { 220. })
                .shrink()
                .truncate_middle()
                .help(&project.path),
        );
    }
    if narrow {
        bar = bar.child(icon_button(
            "command-palette",
            "Search commands",
            Icon::Search,
            true,
            |s, tx| s.dispatch(UiAction::ShowPalette, tx),
        ));
    } else {
        bar = bar.child(
            button(
                "command-palette",
                "Search commands",
                false,
                true,
                |s, tx| s.dispatch(UiAction::ShowPalette, tx),
            )
            .child(icon(Icon::Search, 14., muted())),
        );
    }
    bar = bar.child(icon_button(
        "shortcut-settings",
        "Keyboard shortcuts",
        Icon::Keyboard,
        true,
        |s, tx| s.dispatch(UiAction::ShowShortcuts, tx),
    ));
    if state.model.project.is_some() {
        let count = state.model.active_jobs();
        let mut jobs = icon_button(
            "toggle-jobs",
            "Show background jobs",
            Icon::Jobs,
            true,
            |s, _| s.ui.jobs_open = !s.ui.jobs_open,
        );
        if count > 0 {
            jobs = jobs.child(
                el().size(15., 15.)
                    .radius(7.5)
                    .fill(forest())
                    .center()
                    .pin(None, Some(-2.), Some(-2.), None)
                    .child(number(count.to_string(), 9., paper())),
            );
        }
        bar = bar.child(jobs).child(icon_button(
            "toggle-inspector",
            "Show or hide inspector",
            Icon::Sliders,
            true,
            |s, _| s.ui.inspector_open = !s.ui.inspector_open,
        ));
    }
    bar
}

fn project_tree(state: &AppState) -> El {
    let project = state
        .model
        .project
        .as_ref()
        .expect("project tree requires an open project");
    let mut tree = col()
        .key("project-tree")
        .w(228.)
        .h_full()
        .min_h(0.)
        .fill(paper());
    tree = tree.child(
        col()
            .padding(18.)
            .gap(12.)
            .child(
                row()
                    .gap(8.)
                    .items_center()
                    .child(icon(Icon::Folder, 17., forest()))
                    .child(
                        strong(&project.name, 13., ink())
                            .grow(1.)
                            .min_w(0.)
                            .truncate()
                            .help(&project.path),
                    )
                    .child(icon_button(
                        "open-project",
                        "Open another project",
                        Icon::Arrow,
                        true,
                        |s, tx| s.dispatch(UiAction::OpenProject, tx),
                    )),
            )
            .child(search_input(
                "asset-search",
                &state.ui.asset_query,
                "Find scans and results",
                |s| &mut s.ui.asset_query,
            ))
            .child(
                row()
                    .gap(7.)
                    .child(
                        action_button(
                            "import-files",
                            "Add scans",
                            false,
                            true,
                            UiAction::ImportFiles,
                        )
                        .grow(1.)
                        .child(icon(Icon::Add, 13., forest())),
                    )
                    .child(icon_button(
                        "import-folder",
                        "Add a scan folder",
                        Icon::Folder,
                        true,
                        |s, tx| s.dispatch(UiAction::ImportFolder, tx),
                    )),
            ),
    );
    let query = state.ui.asset_query.text.trim().to_lowercase();
    let mut listing = col()
        .key("asset-list")
        .role("tree")
        .label("Project scans and results")
        .px(10.)
        .gap(3.)
        .w_full();
    if state.model.assets.is_empty() {
        listing = listing.child(
            help("Add scans to this project. Large datasets can stay in their original location.")
                .padding(8.),
        );
    } else if query.is_empty() {
        for asset in state.model.assets.iter().filter(|a| {
            a.parent
                .as_ref()
                .is_none_or(|p| !state.model.assets.iter().any(|b| &b.id == p))
        }) {
            listing = listing.child(asset_branch(state, asset, 0));
        }
    } else {
        let matches: Vec<_> = state
            .model
            .assets
            .iter()
            .filter(|a| a.name.to_lowercase().contains(&query))
            .collect();
        if matches.is_empty() {
            listing = listing.child(help("No matching scans or results.").padding(8.));
        }
        for asset in matches {
            listing = listing.child(asset_row(state, asset, 0, false));
        }
    }
    if !state.model.recipes.is_empty() {
        listing = listing
            .child(separator().mt(18.))
            .child(strong("Recipes", 11., muted()).px(8.).pt(15.).pb(8.));
        for recipe in &state.model.recipes {
            let id = recipe.id.clone();
            listing = listing.child(
                button(
                    &format!("recipe-{}", recipe.id),
                    &recipe.name,
                    false,
                    true,
                    move |s, tx| s.dispatch(UiAction::ApplyRecipe(id.clone()), tx),
                )
                .w_full()
                .justify_start()
                .help(&recipe.summary),
            );
        }
    }
    tree = tree.child(
        col()
            .key("project-scroll")
            .grow(1.)
            .min_h(0.)
            .w_full()
            .scroll(ScrollSpec {
                stick_bottom: false,
            })
            .child(listing),
    );
    if state.model.selection_count() > 1 {
        tree = tree.child(
            col()
                .padding(14.)
                .gap(8.)
                .fill(sage())
                .child(strong(
                    format!("{} assets selected", state.model.selection_count()),
                    12.,
                    forest(),
                ))
                .child(
                    action_button(
                        "batch-reconstruct",
                        "Prepare batch",
                        false,
                        true,
                        UiAction::SetWorkspace(Workspace::Reconstruct),
                    )
                    .w_full(),
                ),
        );
    }
    tree.child(
        col()
            .padding(18.)
            .gap(8.)
            .child(separator())
            .child(
                row()
                    .gap(7.)
                    .items_center()
                    .child(status_dot(if project.save_error.is_some() {
                        error()
                    } else {
                        forest()
                    }))
                    .child(label(
                        if project.save_error.is_some() {
                            "Autosave needs attention"
                        } else if project.saving {
                            "Saving project…"
                        } else if project.dirty {
                            "Changes pending"
                        } else {
                            "Project saved"
                        },
                        10.5,
                        muted(),
                    )),
            )
            .child(
                label(&project.path, 10., muted())
                    .truncate_middle()
                    .help(&project.path),
            ),
    )
}

fn asset_branch(state: &AppState, asset: &AssetView, depth: usize) -> El {
    let children: Vec<_> = state
        .model
        .assets
        .iter()
        .filter(|a| a.parent.as_deref() == Some(&asset.id))
        .collect();
    let has_children = !children.is_empty();
    let mut branch = col()
        .w_full()
        .gap(3.)
        .child(asset_row(state, asset, depth, has_children));
    if !state.ui.collapsed_assets.contains(&asset.id) && depth < 12 {
        for child in children {
            branch = branch.child(asset_branch(state, child, depth + 1));
        }
    }
    branch
}

fn asset_row(state: &AppState, asset: &AssetView, depth: usize, has_children: bool) -> El {
    let active = state.model.active_asset.as_deref() == Some(&asset.id);
    let selected = state.model.selected_assets.contains(&asset.id);
    let id = asset.id.clone();
    let mut name = row().gap(7.).items_center().w_full();
    if has_children {
        let expanded = !state.ui.collapsed_assets.contains(&asset.id);
        let toggle_id = asset.id.clone();
        name = name.child(
            icon_button(
                &format!("expand-{}", asset.id),
                if expanded {
                    "Collapse scan results"
                } else {
                    "Expand scan results"
                },
                if expanded { Icon::Down } else { Icon::Right },
                true,
                move |s, _| {
                    if !s.ui.collapsed_assets.remove(&toggle_id) {
                        s.ui.collapsed_assets.insert(toggle_id.clone());
                    }
                },
            )
            .size(18., 24.),
        );
    } else {
        name = name.child(el().size(18., 24.).inert());
    }
    name = name
        .child(icon(
            if asset.kind == AssetKind::Volume {
                Icon::Cube
            } else {
                Icon::Layers
            },
            15.,
            if active { forest() } else { muted() },
        ))
        .child(
            strong(&asset.name, 12., ink())
                .grow(1.)
                .min_w(0.)
                .truncate()
                .help(&asset.name),
        );
    if selected {
        name = name.child(icon(Icon::Check, 13., forest()));
    }
    let subtitle = if asset.missing {
        "Source missing".to_owned()
    } else if let Some(operation) = &asset.operation {
        crate::forms::OPERATIONS
            .iter()
            .find(|entry| entry.0 == operation)
            .map(|entry| entry.1.to_owned())
            .unwrap_or_else(|| operation.replace('_', " "))
    } else if asset.simulated {
        format!("Simulated {}", asset.kind.title().to_lowercase())
    } else if !asset.status.is_empty() {
        asset.status.clone()
    } else {
        asset.kind.title().to_owned()
    };
    col()
        .key(format!("asset-{}", asset.id))
        .role("treeitem")
        .label(&asset.name)
        .level((depth + 1) as u32)
        .focus_selected(active)
        .checked(selected)
        .focusable(true)
        .w_full()
        .min_w(0.)
        .pl(8. + depth.min(3) as f32 * 12.)
        .pr(8.)
        .py(9.)
        .gap(4.)
        .radius(6.)
        .fill(if active {
            sage()
        } else if selected {
            color("#EDF2E8")
        } else {
            paper()
        })
        .hover(forest().alpha(0.045))
        .focus_ring(1.5, 0., forest())
        .help(&asset.path)
        .on_press_tx(move |s: &mut AppState, tx| {
            s.dispatch(
                UiAction::SelectAsset {
                    id: id.clone(),
                    extend: tx.mods().cmd || tx.mods().shift,
                },
                tx,
            )
        })
        .child(name)
        .child(
            label(
                subtitle,
                10.5,
                if asset.missing { error() } else { muted() },
            )
            .pl(47.),
        )
}

fn welcome(state: &AppState) -> El {
    let mut content=col().w(520.).max_w(650.).gap(20.)
        .child(icon(Icon::Layers,42.,forest()))
        .child(strong("TomoJAX Studio",30.,ink()))
        .child(label("Create a project directory or open an existing project.\nProjects contain scans, processing settings and results.",14.,muted()).wrap())
        .child(row().gap(10.).pt(7.)
            .child(action_button("new-project","Create a project",true,true,UiAction::NewProject).h(42.))
            .child(action_button("open-project-welcome","Open project",false,true,UiAction::OpenProject).h(42.)));
    if !state.model.recent_projects.is_empty() {
        content = content
            .child(separator().mt(18.))
            .child(strong("Recent projects", 13., ink()));
        for recent in state.model.recent_projects.iter().take(5) {
            let path = recent.path.clone();
            content = content.child(
                row()
                    .key(format!("recent-{}", recent.path))
                    .role("button")
                    .label(&recent.name)
                    .focusable(true)
                    .w_full()
                    .py(9.)
                    .gap(12.)
                    .items_center()
                    .radius(5.)
                    .hover(forest().alpha(0.04))
                    .focus_ring(1.5, 1., forest())
                    .on_press_tx(move |s: &mut AppState, tx| {
                        s.dispatch(UiAction::OpenRecent(path.clone()), tx)
                    })
                    .child(icon(Icon::Folder, 18., muted()))
                    .child(
                        col()
                            .grow(1.)
                            .min_w(0.)
                            .gap(4.)
                            .child(strong(&recent.name, 12., ink()))
                            .child(label(&recent.path, 11., muted()).truncate_middle()),
                    )
                    .child(icon(
                        if recent.available {
                            Icon::Right
                        } else {
                            Icon::Link
                        },
                        15.,
                        muted(),
                    )),
            );
        }
    }
    el().key("welcome")
        .grow(1.)
        .min_w(0.)
        .min_h(0.)
        .center()
        .padding(40.)
        .child(content)
}

fn statusbar(state: &AppState) -> El {
    let status = if state.model.status.is_empty() {
        "Create or open a project"
    } else {
        &state.model.status
    };
    let mut bar = row()
        .key("statusbar")
        .role("status")
        .h(28.)
        .w_full()
        .px(16.)
        .gap(8.)
        .items_center()
        .fill(paper())
        .child(status_dot(if state.model.backend.ready {
            forest()
        } else {
            muted()
        }))
        .child(label(status, 10.5, muted()).shrink().min_w(0.).truncate())
        .child(spacer());
    if !state.model.render_quality.is_empty() {
        bar = bar.child(label(&state.model.render_quality, 10., muted()));
    }
    if !state.model.backend.device.is_empty() {
        bar = bar.child(label(&state.model.backend.device, 10., muted()).pl(12.));
    }
    bar
}
