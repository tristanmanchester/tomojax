//! Terrarium's native UI palette and compact, readable type roles.
use rune::{Color, El, TextStyle, text};

pub fn color(value: &str) -> Color {
    rune::color::hex(value).expect("authored Terrarium palette uses valid hex")
}
pub fn background() -> Color {
    color("#F5F6F0")
}
pub fn paper() -> Color {
    color("#FCFDF9")
}
pub fn panel() -> Color {
    color("#F0F3EC")
}
pub fn sage() -> Color {
    color("#E2EADF")
}
pub fn forest() -> Color {
    color("#315D46")
}
pub fn ink() -> Color {
    color("#24382D")
}
pub fn muted() -> Color {
    color("#66776A")
}
pub fn line() -> Color {
    color("#DCE3D8")
}
pub fn canvas_bg() -> Color {
    color("#111A17")
}
pub fn canvas_ink() -> Color {
    color("#EDF2EA")
}
pub fn canvas_muted() -> Color {
    color("#B1BEB4")
}
pub fn warning() -> Color {
    color("#8B5E27")
}
pub fn error() -> Color {
    color("#A3493B")
}
pub fn axis(index: usize) -> Color {
    [color("#94BE9A"), color("#D6B991"), color("#A9B4D6")][index % 3]
}

pub fn label(value: impl Into<String>, size: f32, colour: Color) -> El {
    text(value).font(TextStyle::new(size, 400.0)).color(colour)
}
pub fn strong(value: impl Into<String>, size: f32, colour: Color) -> El {
    text(value).font(TextStyle::new(size, 600.0)).color(colour)
}
pub fn number(value: impl Into<String>, size: f32, colour: Color) -> El {
    text(value)
        .font(TextStyle::new(size, 450.0).features("'tnum' 1"))
        .color(colour)
}
