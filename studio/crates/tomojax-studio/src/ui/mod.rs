//! Terrarium: a persistent project around a live scientific image workspace.
mod controls;
pub mod corrections;
mod export;
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
pub(crate) use controls::{compact_number, unit_name};
use model::*;
use rune::{Cx, El, ScrollSpec, col, el, row};
use tokens::*;

pub fn view(state: &AppState, cx: &mut Cx<AppState>) -> El {
    let narrow = cx
        .layout_with("studio-root", |r| r[2] < 1320.)
        .unwrap_or(false);
    let mut body = row().grow(1.).min_h(0.).min_w(0.).w_full();
    if let Some(project) = &state.model.project
        && state.ui.tree_open
    {
        body = body
            .child(project_tree(state, project))
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
                .fill(color([0xFA, 0xEA, 0xE4]))
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
    if state.ui.activity.is_open() && state.ui.workspace != Workspace::Jobs {
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
        .child(brand_mark(28.))
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
            tab = tab.child(el().h(3.).radius(1.5).fill(color([0x68, 0x8E, 0x6E])).pin(
                Some(10.),
                None,
                Some(10.),
                Some(0.),
            ));
        }
        navigation = navigation.child(tab);
    }
    bar = bar.child(navigation).child(spacer());
    // The navigator header names the project; the bar repeats it only while the navigator is hidden.
    if let Some(project) = state.model.project.as_ref().filter(|_| !state.ui.tree_open) {
        bar = bar.child(
            label(&project.name, 12., ink())
                .min_w(0.)
                .max_w(if narrow { 140. } else { 220. })
                .shrink()
                .truncate_middle()
                .help(&project.path),
        );
    }
    bar = bar.child(
        button(
            "command-palette",
            if narrow { "Search" } else { "Search commands" },
            false,
            true,
            |s, tx| s.dispatch(UiAction::ShowPalette, tx),
        )
        .help("Search commands"),
    );
    bar = bar.child(quiet_button(
        "shortcut-settings",
        "Shortcuts",
        state.ui.shortcuts_open,
        |s, tx| s.dispatch(UiAction::ShowShortcuts, tx),
    ));
    if state.model.project.is_some() && state.ui.workspace != Workspace::Jobs {
        let count = state.model.active_jobs();
        let title = if count == 0 {
            "Activity".into()
        } else {
            format!("Activity · {count}")
        };
        bar = bar
            .child(
                quiet_button(
                    "toggle-jobs",
                    &title,
                    state.ui.activity.is_open(),
                    |s, _| {
                        if s.ui.activity.is_open() {
                            s.ui.activity.close();
                        } else {
                            s.show_activity();
                        }
                    },
                )
                .label("Activity"),
            )
            .child(icon_button(
                "toggle-inspector",
                "Inspector",
                Icon::Inspector,
                true,
                |s, _| s.ui.inspector_open = !s.ui.inspector_open,
            ));
    }
    bar
}

fn project_tree(state: &AppState, project: &ProjectView) -> El {
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
                        action_button_icon(
                            "import-files",
                            "Add scans",
                            Icon::Add,
                            false,
                            true,
                            UiAction::ImportFiles,
                        )
                        .grow(1.),
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
        listing = listing.child(help("No scans").padding(8.));
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
            let (apply, remove) = (recipe.id.clone(), recipe.id.clone());
            listing = listing.child(
                row()
                    .gap(4.)
                    .w_full()
                    .items_center()
                    .child(
                        button(
                            &format!("recipe-{}", recipe.id),
                            &recipe.name,
                            false,
                            true,
                            move |s, tx| s.dispatch(UiAction::ApplyRecipe(apply.clone()), tx),
                        )
                        .grow(1.)
                        .min_w(0.)
                        .justify_start()
                        .help(&recipe.summary),
                    )
                    .child(icon_button(
                        &format!("remove-recipe-{}", recipe.id),
                        "Remove recipe",
                        Icon::Remove,
                        true,
                        move |s, tx| s.dispatch(UiAction::RemoveRecipe(remove.clone()), tx),
                    )),
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
        col().padding(18.).gap(8.).child(separator()).child(
            label(&project.path, 11., muted())
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
    }
    name = name.child(icon(
        if asset.kind == AssetKind::Volume {
            Icon::Cube
        } else {
            Icon::Layers
        },
        15.,
        if active { forest() } else { muted() },
    ));
    let renaming = state.ui.rename_asset.as_ref() == Some(&asset.id);
    if renaming {
        name = name.child(
            submit_input(
                "result-name",
                &state.ui.rename_draft,
                "Result name",
                UiAction::RenameResult,
                |s| &mut s.ui.rename_draft,
            )
            .grow(1.),
        );
    } else {
        name = name.child(
            strong(tree_label(state, asset), 12., ink())
                .grow(1.)
                .min_w(0.)
                .truncate()
                .help(&asset.name),
        );
    }
    let confirming = state.ui.confirm_remove.as_deref() == Some(asset.id.as_str());
    if active && !renaming && !confirming {
        if asset.operation.is_some() {
            let (id, current) = (asset.id.clone(), asset.name.clone());
            name = name.child(icon_button(
                "rename-result",
                "Rename result",
                Icon::Edit,
                true,
                move |s, tx| {
                    s.ui.rename_asset = Some(id.clone());
                    s.ui.rename_draft = rune::TextEdit::new(&current);
                    tx.focus("result-name", true);
                },
            ));
        }
        let id = asset.id.clone();
        name = name.child(icon_button(
            "remove-asset",
            "Remove from project",
            Icon::Remove,
            true,
            move |s, _| s.ui.confirm_remove = Some(id.clone()),
        ));
    } else if selected && !renaming && !confirming {
        name = name.child(icon(Icon::Check, 13., forest()));
    }
    let subtitle = if asset.missing {
        "Source missing".to_owned()
    } else if let Some(operation) = &asset.operation {
        crate::operations::title(operation).to_owned()
    } else if asset.simulated {
        format!("Simulated {}", asset.kind.title().to_lowercase())
    } else if !asset.status.is_empty() {
        asset.status.clone()
    } else {
        asset.kind.title().to_owned()
    };
    let item = col()
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
            color([0xED, 0xF2, 0xE8])
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
            .pl(if has_children { 47. } else { 22. }),
        );
    if confirming {
        return item.child(
            row()
                .gap(6.)
                .pl(22.)
                .child(
                    action_button(
                        "confirm-remove-asset",
                        "Remove from project",
                        true,
                        true,
                        UiAction::RemoveAsset(asset.id.clone()),
                    )
                    .h(30.),
                )
                .child(
                    quiet_button("cancel-remove-asset", "Cancel", false, |s, _| {
                        s.ui.confirm_remove = None
                    })
                    .h(30.),
                ),
        );
    }
    if renaming {
        let valid = !state.ui.rename_draft.text.trim().is_empty();
        return item.child(
            row()
                .gap(6.)
                .pl(22.)
                .child(
                    action_button(
                        "save-result-name",
                        "Save",
                        true,
                        valid,
                        UiAction::RenameResult,
                    )
                    .h(30.),
                )
                .child(
                    quiet_button("cancel-result-name", "Cancel", false, |s, _| {
                        s.ui.rename_asset = None
                    })
                    .h(30.),
                ),
        );
    }
    item
}

/// A child result is listed under its parent, so the parent's name prefix is dropped there.
/// The full name stays in the header and the tooltip; a custom name is shown as typed.
fn tree_label<'a>(state: &AppState, asset: &'a AssetView) -> &'a str {
    asset
        .parent
        .as_ref()
        .and_then(|parent| state.model.assets.iter().find(|a| &a.id == parent))
        .and_then(|parent| asset.name.strip_prefix(parent.name.as_str()))
        .and_then(|rest| rest.strip_prefix(" · "))
        .filter(|rest| !rest.is_empty())
        .unwrap_or(&asset.name)
}

fn welcome(state: &AppState) -> El {
    let mut content = col()
        .w(520.)
        .max_w(650.)
        .gap(20.)
        .child(brand_mark(46.))
        .child(strong("TomoJAX Studio", 30., ink()))
        .child(
            row()
                .gap(10.)
                .pt(7.)
                .child(
                    action_button(
                        "new-project",
                        "Create a project",
                        true,
                        true,
                        UiAction::NewProject,
                    )
                    .h(42.),
                )
                .child(
                    action_button(
                        "open-project-welcome",
                        "Open project",
                        false,
                        true,
                        UiAction::OpenProject,
                    )
                    .h(42.),
                )
                .child(
                    action_button(
                        "open-demo",
                        "Try a demo project",
                        false,
                        state.model.backend.ready,
                        UiAction::OpenDemo,
                    )
                    .h(42.),
                ),
        );
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
        .child(label(status, 11., muted()).shrink().min_w(0.).truncate())
        .child(spacer());
    if state.copying > 0 {
        bar = bar
            .child(label(
                format!(
                    "Copying {} source{} into the project",
                    state.copying,
                    if state.copying == 1 { "" } else { "s" }
                ),
                11.,
                muted(),
            ))
            .child(
                quiet_button("cancel-copy", "Cancel copy", false, |s, tx| {
                    s.dispatch(UiAction::CancelCopies, tx)
                })
                .h(24.),
            );
    }
    if let Some(project) = &state.model.project {
        let (text, tone) = if project.save_error.is_some() {
            ("Autosave failed", error())
        } else if project.saving {
            ("Saving project…", muted())
        } else if project.dirty {
            ("Changes pending", muted())
        } else {
            ("Project saved", muted())
        };
        bar = bar.child(
            label(text, 11., tone).help(project.save_error.as_deref().unwrap_or(&project.path)),
        );
    }
    if !state.model.render_quality.is_empty() {
        bar = bar.child(label(&state.model.render_quality, 11., muted()));
    }
    if !state.model.backend.device.is_empty() {
        bar = bar.child(label(&state.model.backend.device, 11., muted()).pl(12.));
    }
    bar
}
