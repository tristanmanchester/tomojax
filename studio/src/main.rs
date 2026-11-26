//! TomoJAX Studio: native project, processing and scientific viewing workspace.
mod app;
mod forms;
mod preferences;
mod ui;

pub use app::AppState;
use rune::{AppSpec, Key, KeyEvent, Tx};
use tomojax_studio::{
    engine, imaging, live, metadata, operations, previews, progress, project, protocol, volume,
};
use ui::model::{PlaneKind, UiAction, Workspace};

fn setup(state: &mut AppState, wake: rune::Waker) {
    state.setup(wake);
}
fn poll(state: &mut AppState, _: f64, tx: &mut Tx) -> bool {
    let changed = state.poll();
    if state.project.as_ref().is_some_and(|p| p.dirty()) || state.has_pending_work() {
        tx.wake_after(std::time::Duration::from_millis(16));
    } else if state.project.is_some() {
        tx.wake_after(std::time::Duration::from_millis(450));
    }
    changed
}
fn key(state: &mut AppState, event: &KeyEvent, tx: &mut Tx) -> bool {
    if state.ui.recording_shortcut {
        if event.key == Key::Escape {
            state.ui.recording_shortcut = false;
        } else if let Some(chord) = preferences::chord_text(event) {
            state.ui.shortcut_binding = rune::TextEdit::new(&chord);
            state.ui.recording_shortcut = false;
        }
        return true;
    }
    if event.key == Key::Escape {
        // The topmost layer closes first: dialogs, then open menus, then Activity.
        if state.ui.palette_open || state.ui.shortcuts_open {
            state.ui.palette_open = false;
            state.ui.shortcuts_open = false;
            return true;
        }
        if state.ui.operation_menu || state.ui.choice_menu.is_some() {
            state.ui.operation_menu = false;
            state.ui.choice_menu = None;
            return true;
        }
        if state.ui.rename_asset.take().is_some() {
            return true;
        }
        if state.ui.activity.is_open() {
            state.ui.activity = crate::ui::model::Activity::Closed;
            return true;
        }
        return false;
    }
    if state.ui.palette_open {
        let query = state.ui.command_query.text.to_lowercase();
        let commands: Vec<_> = state
            .model
            .commands
            .iter()
            .filter(|c| {
                c.enabled
                    && format!("{} {}", c.category, c.title)
                        .to_lowercase()
                        .contains(&query)
            })
            .collect();
        match event.key {
            Key::Down => {
                state.ui.command_index =
                    (state.ui.command_index + 1).min(commands.len().saturating_sub(1));
                if let Some(command) = commands.get(state.ui.command_index) {
                    tx.scroll_into_view(format!("command-{}", command.id));
                }
                return true;
            }
            Key::Up => {
                state.ui.command_index = state.ui.command_index.saturating_sub(1);
                if let Some(command) = commands.get(state.ui.command_index) {
                    tx.scroll_into_view(format!("command-{}", command.id));
                }
                return true;
            }
            Key::Enter => {
                if let Some(command) = commands.get(state.ui.command_index) {
                    let id = command.id.clone();
                    state.dispatch(UiAction::RunCommand(id), tx);
                }
                return true;
            }
            _ => {}
        }
    }
    for (id, _, _, default) in preferences::COMMANDS {
        if rune::Chord::parse(state.preferences.binding(id, default))
            .is_ok_and(|c| c.matches_tx(event, tx))
        {
            state.dispatch(UiAction::RunCommand((*id).into()), tx);
            return true;
        }
    }
    false
}
fn app_spec() -> AppSpec<AppState> {
    AppSpec {
        min_size: Some((1180., 780.)),
        setup: Some(setup),
        poll: Some(poll),
        on_key: Some(key),
        scenes: &[
            ("prepare", |s| s.set_workspace(Workspace::Prepare)),
            ("reconstruct", |s| s.set_workspace(Workspace::Reconstruct)),
            ("explore", |s| s.ui.workspace = Workspace::Explore),
            ("compare", |s| s.ui.workspace = Workspace::Compare),
            ("jobs", |s| s.ui.workspace = Workspace::Jobs),
            ("palette", |s| s.ui.palette_open = true),
            ("shortcuts", |s| s.ui.shortcuts_open = true),
            ("axial", |s| {
                s.ui.workspace = Workspace::Explore;
                s.ui.main_view = PlaneKind::Axial;
            }),
            ("projection", |s| {
                s.ui.workspace = Workspace::Explore;
                s.ui.main_view = PlaneKind::Projection;
            }),
        ],
        ..AppSpec::new("TomoJAX Studio", (1540., 980.), AppState::init, ui::view)
    }
}

fn main() {
    rune::tooling::set_app_id("org.tomojax.studio");
    let app = rune::App::new(app_spec())
        .compositor(volume::ScientificCompositor::new(AppState::render_items))
        .on_close(|s, _| s.shutdown())
        .on_quit(|s, _| s.shutdown());
    app.run();
}

#[test]
fn native_window_uses_the_product_name() {
    assert_eq!(app_spec().title, "TomoJAX Studio");
}
