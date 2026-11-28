use super::*;
use rune::{Drag, Phase, Tx};

pub(super) fn handle(
    s: &mut AppState,
    kind: PlaneKind,
    mark: u32,
    comparison: bool,
    drag: &mut Drag,
    tx: &mut Tx,
) {
    let position = [
        (drag.x / drag.w.max(1.)).clamp(0., 1.),
        (drag.y / drag.h.max(1.)).clamp(0., 1.),
    ];
    if kind != PlaneKind::Volume && s.ui.inspect_tool == InspectTool::Navigate {
        s.ui.probe = (drag.phase != Phase::Up && tx.button() != PointerButton::Middle)
            .then_some((mark, position));
    }
    if drag.phase == Phase::Down {
        s.ui.drag_last = [0.; 2];
    }
    let delta = [drag.dx - s.ui.drag_last[0], drag.dy - s.ui.drag_last[1]];
    s.ui.drag_last = [drag.dx, drag.dy];
    if !comparison
        && kind.axis().is_some()
        && s.ui.inspect_tool != InspectTool::Navigate
        && tx.button() != PointerButton::Middle
    {
        if drag.phase == Phase::Up && drag.cancelled() {
            s.dispatch(UiAction::CancelMeasurement, tx);
        } else {
            s.dispatch(
                UiAction::MeasureDrag {
                    plane: kind,
                    start: [
                        (drag.start_x / drag.w.max(1.)).clamp(0., 1.),
                        (drag.start_y / drag.h.max(1.)).clamp(0., 1.),
                    ],
                    end: position,
                    finished: drag.phase == Phase::Up,
                },
                tx,
            );
        }
    } else if matches!(drag.phase, Phase::Down | Phase::Move) {
        if tx.button() == PointerButton::Middle {
            s.dispatch(
                UiAction::Pan {
                    delta: [delta[0] / drag.w.max(1.), delta[1] / drag.h.max(1.)],
                },
                tx,
            );
        } else if mark == 12 {
            let u = drag.x / drag.w.max(1.);
            let split = (u - 0.5 - s.ui.display.pan[0]) / s.ui.display.zoom + 0.5;
            s.dispatch(UiAction::SetCompareSplit(split), tx);
        } else if comparison {
            s.ui.compare_cursor = std::array::from_fn(|i| {
                let pos = position[i];
                ((pos - 0.5 - s.ui.display.pan[i]) / s.ui.display.zoom + 0.5).clamp(0., 1.)
            });
        } else if kind == PlaneKind::Volume {
            s.dispatch(
                UiAction::CameraDrag {
                    delta: [delta[0] * 0.35, delta[1] * 0.35],
                },
                tx,
            );
        } else if kind.axis().is_some()
            && s.model
                .active()
                .is_some_and(|asset| asset.kind == AssetKind::Volume)
        {
            s.dispatch(
                UiAction::SetCrosshair {
                    plane: kind,
                    position,
                },
                tx,
            );
        }
    }
    tx.instant();
}
