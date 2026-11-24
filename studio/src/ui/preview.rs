//! Temporary preview controls. Keeping a result is an explicit project action.
use super::{controls::*, model::*, tokens::*};
use crate::AppState;
use crate::previews::PreviewPhase;
use rune::{El, col, el, row};

pub fn controls(state: &AppState) -> El {
    let Some(preview) = &state.model.preview else {
        return el();
    };
    let mut body = col().key("interactive-preview-controls").gap(12.).w_full()
        .child(separator())
        .child(strong("Interactive preview", 13., ink()));
    if matches!(preview.operation.as_str(), "slice" | "centre_sweep") {
        let count = preview.count.unwrap_or(1);
        body = body.child(slider(
            "preview-index",
            "Slice index",
            preview.index as f32 / count.saturating_sub(1).max(1) as f32,
            &format!(
                "{}{}",
                preview.index,
                preview
                    .count
                    .map(|count| format!(" / {}", count.saturating_sub(1)))
                    .unwrap_or_default()
            ),
            state.model.backend.ready && preview.count.is_some_and(|count| count > 1),
            move |value| {
                UiAction::SetPreviewIndex((value * count.saturating_sub(1) as f32).round() as usize)
            },
        ));
    }
    if preview.operation == "centre_sweep" {
        let span = state
            .ui
            .preview_span
            .text
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite() && *v > 0.);
        let width = span.unwrap_or(8.);
        let centre = preview.centre.unwrap_or(0.);
        body = body
            .child(slider(
                "preview-centre",
                "Single centre offset",
                ((centre + width) / (2. * width)) as f32,
                &format!("{centre:+.3} input px"),
                state.model.backend.ready && span.is_some(),
                move |value| {
                    UiAction::SetPreviewCentre(
                        ((f64::from(value) * 2. - 1.) * width * 1000.).round() / 1000.,
                    )
                },
            ))
            .child(
                row()
                    .gap(8.)
                    .items_center()
                    .child(label("Slider range ±", 11., muted()).grow(1.))
                    .child(
                        input(
                            "preview-span",
                            &state.ui.preview_span,
                            "Centre slider half-range in input detector pixels",
                            "",
                            |state| &mut state.ui.preview_span,
                        )
                        .w(62.),
                    )
                    .child(label("input px", 10., muted())),
            );
    }
    if let Some(path) = &preview.error_path {
        body = body.child(label(format!("{path}: {}", preview.message), 11., error()).wrap());
    }
    body
}

pub fn footer(state: &AppState) -> El {
    let Some(preview) = &state.model.preview else {
        return el();
    };
    let pending = matches!(
        preview.phase,
        PreviewPhase::Opening
            | PreviewPhase::Debouncing
            | PreviewPhase::Queued
            | PreviewPhase::Running
            | PreviewPhase::Paused
    );
    let mut body = col()
        .key("preview-footer")
        .padding(20.)
        .gap(10.)
        .w_full()
        .fill(paper())
        .child(
            label(
                &preview.message,
                11.,
                if preview.phase == PreviewPhase::Failed {
                    error()
                } else {
                    muted()
                },
            )
            .wrap(),
        )
        .child(
            action_button(
                "keep-preview",
                "Keep result",
                true,
                preview.can_keep,
                UiAction::KeepPreview,
            )
            .w_full()
            .h(40.),
        )
        .child(
            row()
                .gap(8.)
                .child(
                    action_button(
                        "refresh-preview",
                        "Update preview",
                        false,
                        state.model.backend.ready,
                        UiAction::RefreshPreview,
                    )
                    .grow(1.),
                )
                .child(action_button(
                    "cancel-preview",
                    "Stop",
                    false,
                    pending,
                    UiAction::CancelPreview,
                )),
        );
    if let Some(job) = state.preview_blocker() {
        body = body
            .child(help(if job.status == JobStatus::Paused {
                "A paused job holds the worker. Resume or cancel it to run previews."
            } else {
                "Previews run after this job."
            }))
            .child(
                row()
                    .gap(8.)
                    .child(
                        action_button(
                            "preview-blocker-control",
                            if job.status == JobStatus::Paused {
                                "Resume job"
                            } else {
                                "Pause job"
                            },
                            false,
                            job.can_pause,
                            if job.status == JobStatus::Paused {
                                UiAction::ResumeJob(job.id.clone())
                            } else {
                                UiAction::PauseJob(job.id.clone())
                            },
                        )
                        .grow(1.),
                    )
                    .child(
                        action_button(
                            "preview-cancel-blocker",
                            "Cancel job",
                            false,
                            true,
                            UiAction::CancelJob(job.id.clone()),
                        )
                        .grow(1.),
                    ),
            );
    }
    col().w_full().child(separator()).child(body)
}

pub fn status_bar(state: &AppState) -> El {
    let Some(preview) = &state.model.preview else {
        return el();
    };
    row()
        .key("preview-status")
        .role("status")
        .w_full()
        .min_h(34.)
        .gap(9.)
        .px(12.)
        .py(8.)
        .fill(sage())
        .radius(5.)
        .child(status_dot(if preview.can_keep {
            forest()
        } else {
            muted()
        }))
        .child(strong(
            if preview.has_image {
                "Temporary preview"
            } else {
                "Preview pending"
            },
            11.,
            forest(),
        ))
        .child(
            label(&preview.message, 11., ink())
                .wrap()
                .grow(1.)
                .min_w(0.),
        )
        .child(number(
            preview
                .revision
                .map(|v| format!("revision {v}"))
                .unwrap_or_default(),
            10.,
            muted(),
        ))
}
