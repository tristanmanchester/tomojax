//! Shared native controls. Each interactive surface carries its name and focus state.
use rune::{Color, El, Phase, Shape, TextEdit, Tx, canvas, col, el, row, text_input};

pub use super::format::*;
use super::{model::UiAction, tokens::*};
use crate::AppState;

#[derive(Clone, Copy)]
pub enum Icon {
    Folder,
    Add,
    Search,
    Cube,
    Layers,
    Compare,
    Jobs,
    Play,
    Pause,
    Check,
    Close,
    Arrow,
    Up,
    Down,
    Right,
    Export,
    Link,
    Fit,
    Plus,
    Minus,
    Undo,
    Redo,
    Refresh,
    Tree,
    Inspector,
    Edit,
    Remove,
    Alert,
}

pub fn icon(kind: Icon, size: f32, color: Color) -> El {
    let paths: Vec<Vec<[f32; 2]>> = match kind {
        Icon::Folder => vec![vec![
            [3., 19.],
            [3., 6.],
            [10., 6.],
            [12., 9.],
            [21., 9.],
            [21., 19.],
            [3., 19.],
        ]],
        Icon::Add | Icon::Plus => vec![vec![[5., 12.], [19., 12.]], vec![[12., 5.], [12., 19.]]],
        Icon::Minus => vec![vec![[5., 12.], [19., 12.]]],
        Icon::Search => vec![
            (0..=24)
                .map(|i| {
                    let a = i as f32 / 24. * std::f32::consts::TAU;
                    [10. + a.cos() * 6., 10. + a.sin() * 6.]
                })
                .collect(),
            vec![[15., 15.], [21., 21.]],
        ],
        Icon::Cube => vec![
            vec![
                [12., 3.],
                [21., 8.],
                [21., 17.],
                [12., 22.],
                [3., 17.],
                [3., 8.],
                [12., 3.],
            ],
            vec![[3., 8.], [12., 13.], [21., 8.]],
            vec![[12., 13.], [12., 22.]],
        ],
        Icon::Layers => vec![
            vec![[3., 7.], [12., 2.], [21., 7.], [12., 12.], [3., 7.]],
            vec![[3., 12.], [12., 17.], [21., 12.]],
            vec![[3., 17.], [12., 22.], [21., 17.]],
        ],
        Icon::Compare => vec![
            vec![[3., 4.], [21., 4.], [21., 20.], [3., 20.], [3., 4.]],
            vec![[12., 4.], [12., 20.]],
        ],
        Icon::Jobs => vec![
            vec![[8., 6.], [21., 6.]],
            vec![[8., 12.], [21., 12.]],
            vec![[8., 18.], [21., 18.]],
            vec![[3., 6.], [4., 6.]],
            vec![[3., 12.], [4., 12.]],
            vec![[3., 18.], [4., 18.]],
        ],
        Icon::Play => vec![vec![[8., 5.], [19., 12.], [8., 19.], [8., 5.]]],
        Icon::Pause => vec![vec![[8., 5.], [8., 19.]], vec![[16., 5.], [16., 19.]]],
        Icon::Check => vec![vec![[5., 12.], [10., 17.], [20., 6.]]],
        Icon::Close => vec![vec![[6., 6.], [18., 18.]], vec![[18., 6.], [6., 18.]]],
        Icon::Arrow | Icon::Down => vec![vec![[6., 9.], [12., 15.], [18., 9.]]],
        Icon::Up => vec![vec![[6., 15.], [12., 9.], [18., 15.]]],
        Icon::Right => vec![vec![[9., 6.], [15., 12.], [9., 18.]]],
        Icon::Export => vec![
            vec![[4., 14.], [4., 21.], [20., 21.], [20., 14.]],
            vec![[12., 3.], [12., 16.]],
            vec![[7., 8.], [12., 3.], [17., 8.]],
        ],
        Icon::Link => vec![
            vec![
                [10., 8.],
                [14., 4.],
                [18., 4.],
                [21., 7.],
                [21., 10.],
                [16., 15.],
            ],
            vec![
                [14., 16.],
                [10., 20.],
                [6., 20.],
                [3., 17.],
                [3., 14.],
                [8., 9.],
            ],
            vec![[8., 16.], [16., 8.]],
        ],
        Icon::Fit => vec![
            vec![[4., 9.], [4., 4.], [9., 4.]],
            vec![[15., 4.], [20., 4.], [20., 9.]],
            vec![[20., 15.], [20., 20.], [15., 20.]],
            vec![[9., 20.], [4., 20.], [4., 15.]],
        ],
        Icon::Undo => vec![
            vec![[8., 5.], [3., 10.], [8., 15.]],
            vec![[3., 10.], [15., 10.], [19., 13.], [19., 19.]],
        ],
        Icon::Redo => vec![
            vec![[16., 5.], [21., 10.], [16., 15.]],
            vec![[21., 10.], [9., 10.], [5., 13.], [5., 19.]],
        ],
        Icon::Refresh => vec![
            vec![
                [20., 7.],
                [16., 3.],
                [9., 3.],
                [4., 8.],
                [4., 16.],
                [9., 21.],
                [16., 21.],
                [20., 16.],
            ],
            vec![[20., 2.], [20., 8.], [14., 8.]],
        ],
        Icon::Tree => vec![
            vec![[3., 4.], [21., 4.], [21., 20.], [3., 20.], [3., 4.]],
            vec![[9., 4.], [9., 20.]],
        ],
        Icon::Edit => vec![
            vec![
                [4., 20.],
                [5., 16.],
                [16., 5.],
                [19., 8.],
                [8., 19.],
                [4., 20.],
            ],
            vec![[13.5, 7.5], [16.5, 10.5]],
        ],
        Icon::Remove => vec![
            vec![[4., 7.], [20., 7.]],
            vec![[9., 7.], [9., 4.], [15., 4.], [15., 7.]],
            vec![[6., 7.], [7., 20.], [17., 20.], [18., 7.]],
        ],
        Icon::Inspector => vec![
            vec![[3., 4.], [21., 4.], [21., 20.], [3., 20.], [3., 4.]],
            vec![[15., 4.], [15., 20.]],
        ],
        Icon::Alert => vec![
            vec![[12., 3.], [22., 21.], [2., 21.], [12., 3.]],
            vec![[12., 9.], [12., 14.]],
            vec![[12., 17.], [12., 18.]],
        ],
    };
    let mut shape = Shape::new().view_box(24., 24.);
    for path in paths {
        shape = shape.stroke(path, 1.5, color);
    }
    canvas(shape).size(size, size).inert()
}

/// Two-tone sliced-volume TomoJAX mark matching `assets/brand/tomojax-icon.svg`.
pub fn brand_mark(size: f32) -> El {
    let angle = -32.0_f32.to_radians();
    let (sin_a, cos_a) = angle.sin_cos();
    let rot = |x: f32, y: f32| -> [f32; 2] {
        let (dx, dy) = (x - 64., y - 64.);
        [
            64. + dx * cos_a - dy * sin_a - 12.,
            64. + dx * sin_a + dy * cos_a - 8.,
        ]
    };
    let arc = |cx: f32, cy: f32, start: f32, end: f32, steps: usize| -> Vec<[f32; 2]> {
        (0..=steps)
            .map(|i| {
                let a = start + (end - start) * (i as f32 / steps as f32);
                rot(cx + 45. * a.cos(), cy + 45. * a.sin())
            })
            .collect()
    };

    let cap_a = (18.5_f32 / 45.).asin();
    let mid_a = (11.5_f32 / 45.).asin();
    let pi = std::f32::consts::PI;

    let top_cap = arc(64., 64., -pi + cap_a, -cap_a, 24);
    let bottom_cap = arc(64., 64., cap_a, pi - cap_a, 24);
    let mut middle_slice = arc(74., 64., pi - mid_a, pi + mid_a, 10);
    middle_slice.extend(arc(74., 64., -mid_a, mid_a, 10));

    let outer = color([0x6C, 0x8A, 0x77]);
    let center = color([0x26, 0x5B, 0x49]);
    let shape = Shape::new()
        .view_box(112., 112.)
        .fill(top_cap, outer)
        .fill(middle_slice, center)
        .fill(bottom_cap, outer);
    canvas(shape).size(size, size).inert()
}

pub fn separator() -> El {
    el().w_full().h(1.).fill(line())
}
pub fn spacer() -> El {
    el().grow(1.).min_w(0.).min_h(0.)
}
pub fn help(value: impl Into<String>) -> El {
    label(value, 12., muted()).wrap().w_full()
}

pub fn button(
    key: &str,
    title: &str,
    primary: bool,
    enabled: bool,
    f: impl Fn(&mut AppState, &mut Tx) + 'static,
) -> El {
    button_with(key, title, None, primary, enabled, f)
}

/// An action button whose verb icon leads its label.
pub fn action_button_icon(
    key: &str,
    title: &str,
    glyph: Icon,
    primary: bool,
    enabled: bool,
    action: UiAction,
) -> El {
    button_with(key, title, Some(glyph), primary, enabled, move |s, tx| {
        s.dispatch(action.clone(), tx)
    })
}

fn button_with(
    key: &str,
    title: &str,
    glyph: Option<Icon>,
    primary: bool,
    enabled: bool,
    f: impl Fn(&mut AppState, &mut Tx) + 'static,
) -> El {
    let foreground = if primary { paper() } else { ink() };
    let mut body = row()
        .key(key)
        .role("button")
        .label(title)
        .focusable(enabled)
        .disabled(!enabled)
        .h(36.)
        .px(12.)
        .gap(7.)
        .center()
        .radius(6.)
        .fill(if primary { forest() } else { paper() })
        .border(if primary { 0. } else { 1. }, line())
        .hover(if primary {
            Color::WHITE.alpha(0.08)
        } else {
            forest().alpha(0.06)
        })
        .press(ink().alpha(0.09))
        .focus_ring(2., 2., forest())
        .opacity(if enabled { 1. } else { 0.42 })
        .on_press_tx(move |s: &mut AppState, tx| {
            if enabled {
                f(s, tx)
            }
        });
    if let Some(glyph) = glyph {
        body = body.child(icon(glyph, 14., if primary { paper() } else { forest() }));
    }
    body.child(strong(title, 12., foreground).min_w(0.).shrink().truncate())
}

/// One option in a drop-down list; the current choice is filled and ticked.
pub fn menu_option(
    key: &str,
    title: &str,
    selected: bool,
    f: impl Fn(&mut AppState, &mut Tx) + 'static,
) -> El {
    row()
        .key(key)
        .role("option")
        .label(title)
        .checked(selected)
        .focusable(true)
        .w_full()
        .h(32.)
        .px(8.)
        .gap(8.)
        .items_center()
        .radius(4.)
        .fill(if selected { sage() } else { Color::TRANSPARENT })
        .hover(forest().alpha(0.06))
        .focus_ring(1.5, 0., forest())
        .on_press_tx(f)
        .child(strong(title, 12., ink()).grow(1.).min_w(0.).truncate())
        .child(if selected {
            icon(Icon::Check, 13., forest())
        } else {
            el()
        })
}

/// A borderless text button for secondary chrome such as the top bar.
pub fn quiet_button(
    key: &str,
    title: &str,
    selected: bool,
    f: impl Fn(&mut AppState, &mut Tx) + 'static,
) -> El {
    row()
        .key(key)
        .role("button")
        .label(title)
        .focusable(true)
        .h(36.)
        .px(10.)
        .center()
        .radius(6.)
        .fill(if selected { sage() } else { Color::TRANSPARENT })
        .hover(forest().alpha(0.06))
        .press(ink().alpha(0.09))
        .focus_ring(2., 2., forest())
        .on_press_tx(f)
        .child(strong(title, 12., if selected { forest() } else { ink() }).truncate())
}

pub fn action_button(key: &str, title: &str, primary: bool, enabled: bool, action: UiAction) -> El {
    button(key, title, primary, enabled, move |s, tx| {
        s.dispatch(action.clone(), tx)
    })
}

pub fn icon_button(
    key: &str,
    title: &str,
    glyph: Icon,
    enabled: bool,
    f: impl Fn(&mut AppState, &mut Tx) + 'static,
) -> El {
    tinted_icon_button(key, title, glyph, enabled, muted(), f)
}
pub fn tinted_icon_button(
    key: &str,
    title: &str,
    glyph: Icon,
    enabled: bool,
    tint: Color,
    f: impl Fn(&mut AppState, &mut Tx) + 'static,
) -> El {
    el().key(key)
        .role("button")
        .label(title)
        .help(title)
        .focusable(enabled)
        .disabled(!enabled)
        .size(32., 32.)
        .center()
        .radius(5.)
        .hover(forest().alpha(0.07))
        .press(forest().alpha(0.12))
        .focus_ring(2., 1., forest())
        .opacity(if enabled { 1. } else { 0.35 })
        .on_press_tx(move |s: &mut AppState, tx| {
            if enabled {
                f(s, tx)
            }
        })
        .child(icon(glyph, 17., tint))
}

pub fn toggle(
    key: &str,
    title: &str,
    checked: bool,
    enabled: bool,
    f: impl Fn(&mut AppState, &mut Tx) + 'static,
) -> El {
    let mut tick = el()
        .size(16., 16.)
        .radius(3.)
        .center()
        .fill(if checked { forest() } else { paper() })
        .border(1., if checked { forest() } else { muted() });
    if checked {
        tick = tick.child(icon(Icon::Check, 13., paper()));
    }
    row()
        .key(key)
        .role("checkbox")
        .label(title)
        .checked(checked)
        .disabled(!enabled)
        .focusable(enabled)
        .focus_ring(2., 2., forest())
        .min_h(30.)
        .gap(9.)
        .items_center()
        .w_full()
        .opacity(if enabled { 1. } else { 0.45 })
        .on_press_tx(move |s: &mut AppState, tx| {
            if enabled {
                f(s, tx)
            }
        })
        .child(tick)
        .child(label(title, 12., ink()).wrap().grow(1.).min_w(0.))
}

pub fn input(
    key: &str,
    value: &TextEdit,
    title: &str,
    suffix: &str,
    edit: impl for<'a> Fn(&'a mut AppState) -> &'a mut TextEdit + 'static,
) -> El {
    input_hint(key, value, title, suffix, "", edit)
}

pub fn search_input(
    key: &str,
    value: &TextEdit,
    title: &str,
    edit: impl for<'a> Fn(&'a mut AppState) -> &'a mut TextEdit + 'static,
) -> El {
    input_hint(key, value, title, "", title, edit)
}

pub fn input_hint(
    key: &str,
    value: &TextEdit,
    title: &str,
    suffix: &str,
    placeholder: &str,
    edit: impl for<'a> Fn(&'a mut AppState) -> &'a mut TextEdit + 'static,
) -> El {
    text_field(key, value, title, suffix, placeholder, None, edit)
}

/// A single-line text field; `submit` runs when Enter is pressed in it.
fn text_field(
    key: &str,
    value: &TextEdit,
    title: &str,
    suffix: &str,
    placeholder: &str,
    submit: Option<UiAction>,
    edit: impl for<'a> Fn(&'a mut AppState) -> &'a mut TextEdit + 'static,
) -> El {
    let mut editor = text_input(value, type_style(12., 450.).features("'tnum' 1"))
        .key(key)
        .label(title)
        .color(ink())
        .single_line()
        .placeholder(placeholder, muted())
        .h_full()
        .py(9.)
        .grow(1.)
        .min_w(0.)
        .w_full()
        .on_edit(edit);
    if let Some(action) = submit {
        editor = editor.on_submit(move |s: &mut AppState, tx| s.dispatch(action.clone(), tx));
    }
    let mut field = row()
        .h(36.)
        .w_full()
        .min_w(0.)
        .px(10.)
        .gap(5.)
        .center_y()
        .radius(5.)
        .fill(paper())
        .border(1., line())
        .focus_within_ring(1.5, 0., forest())
        .child(editor);
    if !suffix.is_empty() {
        field = field.child(label(suffix, 11., muted()));
    }
    field
}

pub fn field(
    key: &str,
    value: &TextEdit,
    title: &str,
    suffix: &str,
    edit: impl for<'a> Fn(&'a mut AppState) -> &'a mut TextEdit + 'static,
) -> El {
    labeled(title, input(key, value, title, suffix, edit))
}

/// An unlabeled input that runs `submit` when Enter is pressed.
pub fn submit_input(
    key: &str,
    value: &TextEdit,
    title: &str,
    submit: UiAction,
    edit: impl for<'a> Fn(&'a mut AppState) -> &'a mut TextEdit + 'static,
) -> El {
    text_field(key, value, title, "", "", Some(submit), edit)
}

/// A labeled field with placeholder text shown while it is empty.
pub fn field_hint(
    key: &str,
    value: &TextEdit,
    title: &str,
    suffix: &str,
    placeholder: &str,
    edit: impl for<'a> Fn(&'a mut AppState) -> &'a mut TextEdit + 'static,
) -> El {
    labeled(
        title,
        input_hint(key, value, title, suffix, placeholder, edit),
    )
}

/// A labeled field that runs `submit` when Enter is pressed.
pub fn submit_field(
    key: &str,
    value: &TextEdit,
    title: &str,
    submit: UiAction,
    edit: impl for<'a> Fn(&'a mut AppState) -> &'a mut TextEdit + 'static,
) -> El {
    labeled(
        title,
        text_field(key, value, title, "", "", Some(submit), edit),
    )
}

fn labeled(title: &str, control: El) -> El {
    col()
        .w_full()
        .shrink()
        .min_w(0.)
        .min_h(57.)
        .gap(7.)
        .child(label(title, 11., muted()).wrap())
        .child(control)
}

pub fn section(title: &str, body: El) -> El {
    col()
        .w_full()
        .gap(13.)
        .pt(3.)
        .child(strong(title, 13., ink()))
        .child(body)
}

pub fn property(title: &str, value: impl Into<String>) -> El {
    let value = value.into();
    row()
        .w_full()
        .gap(10.)
        .justify_between()
        .items_center()
        .child(label(title, 11., muted()))
        .child(
            number(value.clone(), 11., ink())
                .min_w(0.)
                .shrink()
                .truncate()
                .help(value),
        )
}

pub fn choice(key: &str, title: &str, selected: bool, enabled: bool, action: UiAction) -> El {
    el().key(key)
        .role("radio")
        .label(title)
        .checked(selected)
        .focusable(enabled)
        .disabled(!enabled)
        .h(32.)
        .px(9.)
        .center()
        .radius(5.)
        .min_w(0.)
        .grow(1.)
        .fill(if selected { sage() } else { panel() })
        .hover(forest().alpha(0.06))
        .focus_ring(1.5, 1., forest())
        .opacity(if enabled { 1. } else { 0.4 })
        .on_press_tx(move |s: &mut AppState, tx| {
            if enabled {
                s.dispatch(action.clone(), tx)
            }
        })
        .child(strong(
            title,
            11.,
            if selected { forest() } else { muted() },
        ))
}

pub fn slider(
    key: &str,
    title: &str,
    value: f32,
    formatted: &str,
    enabled: bool,
    action: impl Fn(f32) -> UiAction + Clone + 'static,
) -> El {
    slider_on(
        false,
        key,
        (title, title),
        value,
        formatted,
        enabled,
        action,
    )
}

/// A slider drawn on the dark scientific canvas. `title` names it for assistive technology;
/// `caption` is the visible line beside the formatted value, so the canvas heading is not repeated.
pub fn canvas_slider(
    key: &str,
    title: &str,
    caption: &str,
    value: f32,
    formatted: &str,
    enabled: bool,
    action: impl Fn(f32) -> UiAction + Clone + 'static,
) -> El {
    slider_on(
        true,
        key,
        (title, caption),
        value,
        formatted,
        enabled,
        action,
    )
}

fn slider_on(
    dark: bool,
    key: &str,
    (title, caption): (&str, &str),
    value: f32,
    formatted: &str,
    enabled: bool,
    action: impl Fn(f32) -> UiAction + Clone + 'static,
) -> El {
    let value = value.clamp(0., 1.);
    let drag_action = action.clone();
    let key_action = action.clone();
    let track = el()
        .key(key)
        .role("slider")
        .label(title)
        .range_value(value as f64, 0., 1.)
        .a11y_value(formatted)
        .range_step(0.01)
        .disabled(!enabled)
        .focusable(enabled)
        .w_full()
        .h(26.)
        .center_y()
        .focus_ring(1.5, 1., forest())
        .on_drag(move |s: &mut AppState, drag, tx| {
            if enabled && matches!(drag.phase, Phase::Down | Phase::Move) {
                s.dispatch(
                    drag_action(((drag.x - 6.) / (drag.w - 12.).max(1.)).clamp(0., 1.)),
                    tx,
                );
            }
        })
        .on_step(move |s: &mut AppState, step, tx| {
            if enabled {
                s.dispatch(key_action((value + step as f32 * 0.01).clamp(0., 1.)), tx)
            }
        })
        .on_step_limit(move |s: &mut AppState, end, tx| {
            if enabled {
                s.dispatch(action(if end { 1. } else { 0. }), tx)
            }
        })
        .child(
            el().h(4.)
                .w_full()
                .px(6.)
                .fill(if dark {
                    canvas_muted().alpha(0.35)
                } else {
                    line()
                })
                .radius(2.)
                .child(
                    el().h_full()
                        .w_frac(value)
                        .fill(color([0x82, 0xA6, 0x8A]))
                        .radius(2.)
                        .child(
                            el().size(12., 12.)
                                .radius(6.)
                                .fill(paper())
                                .border(2., color([0x82, 0xA6, 0x8A]))
                                .pin(None, Some(-4.), Some(-6.), None),
                        ),
                ),
        );
    col()
        .gap(3.)
        .w_full()
        .opacity(if enabled { 1. } else { 0.45 })
        .child(
            row()
                .gap(12.)
                .justify_between()
                .child(
                    label(caption, 11., if dark { canvas_muted() } else { muted() })
                        .shrink()
                        .min_w(0.)
                        .truncate()
                        .help(caption),
                )
                .child(number(
                    formatted,
                    11.,
                    if dark { canvas_ink() } else { muted() },
                )),
        )
        .child(track)
}

pub fn status_dot(color: Color) -> El {
    el().size(6., 6.).radius(3.).fill(color)
}
