use super::{AppState, PlaneKind};

pub(super) fn plane_title(state: &AppState, kind: PlaneKind) -> &'static str {
    if kind == PlaneKind::Projection && state.shows_stored_frames() {
        "Stored detector frame"
    } else {
        kind.title()
    }
}

pub(super) fn axes_label(state: &AppState, kind: PlaneKind) -> String {
    if kind == PlaneKind::Volume {
        return String::new();
    }
    state
        .model
        .planes
        .get(&kind)
        .filter(|p| p.axes.iter().all(|name| !name.is_empty()))
        .map(|p| {
            let axis = |s: &str| match s {
                "u" => "detector column".into(),
                "v" => "detector row".into(),
                "view" => "view".into(),
                "frame" => "stored frame".into(),
                _ => s.to_uppercase(),
            };
            format!("{} × {}", axis(&p.axes[1]), axis(&p.axes[0]))
        })
        .unwrap_or_else(|| kind.axes().into())
}
