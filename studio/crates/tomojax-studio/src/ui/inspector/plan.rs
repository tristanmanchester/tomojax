use super::*;
use crate::protocol::{Grid, Placement};

pub(super) fn plan_details(state: &AppState, plan: &Product) -> El {
    let mut content = col().key("resolved-plan").w_full().gap(9.);
    let (reasons, placement) = match plan {
        Product::ReconstructionPlan {
            method,
            reasons,
            iterations,
            grid,
            placement,
        } => {
            content = content.child(property("Method", crate::operations::method_name(method)));
            if let Some(iterations) = iterations {
                content = content.child(property("Iteration limit", iterations.to_string()));
            }
            content = content.child(plan_grid(state, grid));
            (reasons, placement)
        }
        Product::JointPlan {
            reasons,
            levels,
            placement,
        } => {
            for (index, level) in levels.iter().enumerate() {
                content = content
                    .child(strong(format!("Level {}", index + 1), 11., ink()))
                    .child(plan_grid(state, &level.grid))
                    .child(property(
                        "Outer / image iterations",
                        format!("{} / {}", level.iterations, level.image_iterations),
                    ));
            }
            (reasons, placement)
        }
        Product::SinogramFocusPlan { registration, .. }
        | Product::OppositeViewsPlan(registration) => {
            content = content.child(property("Angular pairs", registration.pairs.to_string()));
            (&registration.reasons, &registration.placement)
        }
        _ => return el(),
    };
    for reason in reasons {
        content = content.child(help(reason));
    }
    if !state.detached_jobs() {
        content = content.child(plan_placement(placement));
    }
    section("Resolved plan", content)
}

fn plan_grid(state: &AppState, grid: &Grid) -> El {
    let unit = state.model.active().map_or("source length unit", |asset| {
        if asset.units.is_empty() {
            "source length unit"
        } else {
            unit_name(&asset.units)
        }
    });
    col()
        .w_full()
        .gap(7.)
        .child(property(
            "Grid (x × y × z)",
            format!("{} × {} × {}", grid.nx, grid.ny, grid.nz),
        ))
        .child(property(
            "Voxel pitch",
            format!(
                "{} × {} × {} {unit}",
                compact_number(grid.vx),
                compact_number(grid.vy),
                compact_number(grid.vz)
            ),
        ))
}

fn plan_placement(placement: &Placement) -> El {
    let mut content = col()
        .gap(7.)
        .w_full()
        .child(property(
            "Placement",
            match placement.strategy.as_str() {
                "resident" => "Resident",
                "slabbed" => "Slabbed",
                other => other,
            },
        ))
        .child(property("Devices", placement.devices.join(", ")))
        .child(property(
            "Estimated host memory",
            memory_bytes(placement.host_bytes),
        ))
        .child(property(
            "Estimated device memory",
            memory_bytes(placement.device_bytes),
        ));
    for snapshot in &placement.snapshot {
        if let Some(available) = snapshot.available {
            content = content.child(
                property(
                    &format!("Available · {} / {}", snapshot.device, snapshot.scope),
                    memory_bytes(available),
                )
                .help(format!("Worker snapshot time: {} seconds", snapshot.time)),
            );
        }
    }
    content
}

fn memory_bytes(bytes: u64) -> String {
    for (scale, unit) in [(1_u64 << 30, "GiB"), (1 << 20, "MiB"), (1 << 10, "KiB")] {
        if bytes >= scale {
            return format!("{:.1} {unit}", bytes as f64 / scale as f64);
        }
    }
    format!("{bytes} B")
}
