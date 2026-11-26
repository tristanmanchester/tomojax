//! Terrarium's native UI palette and compact, readable type roles.
use rune::{Color, El, TextStyle, text};

pub fn color(rgb: [u8; 3]) -> Color {
    let [r, g, b] = rgb.map(|channel| rune::color::srgb_decode(f32::from(channel) / 255.));
    Color { r, g, b, a: 1. }
}
pub fn background() -> Color {
    color([0xF5, 0xF6, 0xF0])
}
pub fn paper() -> Color {
    color([0xFC, 0xFD, 0xF9])
}
pub fn panel() -> Color {
    color([0xF0, 0xF3, 0xEC])
}
pub fn sage() -> Color {
    color([0xE2, 0xEA, 0xDF])
}
pub fn forest() -> Color {
    color([0x31, 0x5D, 0x46])
}
pub fn ink() -> Color {
    color([0x24, 0x38, 0x2D])
}
pub fn muted() -> Color {
    color([0x59, 0x6A, 0x5D])
}
pub fn line() -> Color {
    color([0xDC, 0xE3, 0xD8])
}
pub fn canvas_bg() -> Color {
    color([0x11, 0x1A, 0x17])
}
pub fn canvas_ink() -> Color {
    color([0xED, 0xF2, 0xEA])
}
pub fn canvas_muted() -> Color {
    color([0xB1, 0xBE, 0xB4])
}
pub fn warning() -> Color {
    color([0x8B, 0x5E, 0x27])
}
pub fn error() -> Color {
    color([0xA3, 0x49, 0x3B])
}
pub fn axis(index: usize) -> Color {
    [
        color([0x94, 0xBE, 0x9A]),
        color([0xD6, 0xB9, 0x91]),
        color([0xA9, 0xB4, 0xD6]),
    ][index % 3]
}

pub fn type_style(size: f32, weight: f32) -> TextStyle {
    let size = size.max(11.);
    TextStyle::new(size, weight).lh(size * 1.35)
}

pub fn label(value: impl Into<String>, size: f32, color: Color) -> El {
    text(value).font(type_style(size, 400.0)).color(color)
}
pub fn strong(value: impl Into<String>, size: f32, color: Color) -> El {
    text(value).font(type_style(size, 600.0)).color(color)
}
pub fn number(value: impl Into<String>, size: f32, color: Color) -> El {
    text(value)
        .font(type_style(size, 450.0).features("'tnum' 1"))
        .color(color)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn functional_type_and_muted_color_meet_the_design_floor() {
        assert_eq!(muted(), color([0x59, 0x6A, 0x5D]));
        for (requested, expected) in [(8.5, 11.), (11., 11.), (24., 24.)] {
            let style = type_style(requested, 450.);
            assert_eq!(style.size, expected);
            assert_eq!(style.line_px(), expected * 1.35);
        }
    }
}
