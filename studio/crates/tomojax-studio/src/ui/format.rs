//! Scientific values, units and positions as people read them.

/// A unit as people read it; the schema and arrays name units with identifiers.
pub fn unit_name(unit: &str) -> &str {
    match unit {
        "scan_length" | "geometry_length" => "scan unit",
        "1/scan_length" | "1/geometry_length" => "1/scan unit",
        // Raw frames carry no physical unit; say nothing rather than "unspecified".
        "1" | "unspecified" => "",
        "counts_per_line_integral" => "counts",
        "line_integral" => "line integral",
        "native_pixel" => "detector px",
        "working_pixel" => "working px",
        "pixel" => "px",
        "noise_sigma" => "noise SD",
        other => other,
    }
}

/// A slice position as people read it: `view = 22.5°`, `z = 0.5 scan unit`. An index position
/// returns `None` because the slice counter already shows it.
pub fn coordinate(name: &str, value: f64, unit: &str) -> Option<String> {
    let value = compact_number(value);
    match unit {
        "index" => None,
        "degree" => Some(format!("{name} = {value}°")),
        unit => Some(
            format!("{name} = {value} {}", unit_name(unit))
                .trim_end()
                .to_owned(),
        ),
    }
}

pub fn compact_number(value: f64) -> String {
    if value != 0. && (value.abs() < 0.001 || value.abs() >= 10000.) {
        format!("{value:.3e}")
    } else {
        format!("{value:.4}")
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_owned()
    }
}
