use super::*;

pub(super) fn curtain(split: f32, zoom: f32, pan: [f32; 2]) -> El {
    let u = ((split - 0.5) * zoom + 0.5 + pan[0]) * 100.;
    let mut shape = Shape::new().view_box(100., 100.);
    if (0.0..=100.).contains(&u) {
        shape = shape.stroke(vec![[u, 0.], [u, 100.]], 0.35, canvas_ink().alpha(0.85));
    }
    canvas(shape).w_full().h_full().absolute(0., 0.).inert()
}

pub(super) fn cursor_lines([u, v]: [f32; 2], zoom: f32, pan: [f32; 2], kind: PlaneKind) -> El {
    let u = ((u - 0.5) * zoom + 0.5 + pan[0]) * 100.;
    let v = ((v - 0.5) * zoom + 0.5 + pan[1]) * 100.;
    let tint = axis(kind.axis().unwrap_or(0)).alpha(0.6);
    let mut shape = Shape::new().view_box(100., 100.);
    if (0.0..=100.).contains(&u) {
        shape = shape.stroke(vec![[u, 0.], [u, 100.]], 0.18, tint);
    }
    if (0.0..=100.).contains(&v) {
        shape = shape.stroke(vec![[0., v], [100., v]], 0.18, tint);
    }
    canvas(shape).w_full().h_full().absolute(0., 0.).inert()
}

pub(super) fn measurements(state: &AppState, kind: PlaneKind, interactive: bool) -> El {
    let mut shape = Shape::new().view_box(100., 100.);
    let zoom = if interactive {
        state.ui.display.zoom
    } else {
        1.
    };
    let pan = if interactive {
        state.ui.display.pan
    } else {
        [0.; 2]
    };
    for measurement in state.model.measurements.iter().filter(|m| {
        m.plane == kind
            && state.model.active_asset.as_ref() == Some(&m.asset)
            && state
                .model
                .planes
                .get(&kind)
                .is_some_and(|p| p.index == m.index)
    }) {
        let [a, b] = measurement.points.map(|point| {
            [
                ((point[0] - 0.5) * zoom + 0.5 + pan[0]) * 100.,
                ((point[1] - 0.5) * zoom + 0.5 + pan[1]) * 100.,
            ]
        });
        let path = if measurement.tool == InspectTool::Region {
            vec![a, [b[0], a[1]], b, [a[0], b[1]], a]
        } else {
            vec![a, b]
        };
        shape = shape.stroke(path, 0.28, color([0xE8, 0xD2, 0x96]));
    }
    canvas(shape).w_full().h_full().absolute(0., 0.).inert()
}
