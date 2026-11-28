//! Compare sample centers in the declared units before overlaying or subtracting arrays.
use anyhow::{Context, Result, anyhow, ensure};
use serde_json::Value;
const TOLERANCE: f64 = 1e-6;
fn mismatch(reason: String) -> anyhow::Error {
    anyhow!(
        "grid mismatch ({reason}). Choose arrays on the same grid or resample onto the target grid before combining them."
    )
}
fn check_axes(a: &Value, b: &Value, n: usize) -> Result<()> {
    let name = a["name"].as_str().context("Missing axis name")?;
    for (field, property) in [("frame", "frame"), ("unit", "units")] {
        ensure!(
            a[field] == b[field],
            mismatch(format!("{name} {property} {} vs {}", a[field], b[field]))
        );
    }
    let (x, y) = (&a["coordinates"], &b["coordinates"]);
    let number = |v: &Value| v.as_f64().context("Missing coordinate");
    if x["type"] == "uniform" && y["type"] == "uniform" {
        let origin = number(&x["origin"])? - number(&y["origin"])?;
        ensure!(
            origin.abs() <= TOLERANCE,
            mismatch(format!("{name} origin {} vs {}", x["origin"], y["origin"]))
        );
        let pitch = number(&x["spacing"])? - number(&y["spacing"])?;
        ensure!(
            pitch.abs() <= TOLERANCE
                && (origin + n.saturating_sub(1) as f64 * pitch).abs() <= TOLERANCE,
            mismatch(format!("{name} pitch {} vs {}", x["spacing"], y["spacing"]))
        );
    } else {
        for i in 0..n {
            let at = |c: &Value| -> Result<f64> {
                if c["type"] == "uniform" {
                    Ok(number(&c["origin"])? + i as f64 * number(&c["spacing"])?)
                } else {
                    number(&c["values"][i])
                }
            };
            let (u, v) = (at(x)?, at(y)?);
            ensure!(
                (u - v).abs() <= TOLERANCE,
                mismatch(format!("{name} coordinate[{i}] {u} vs {v}"))
            );
        }
    }
    Ok(())
}

fn same_fixed(a: &Value, b: &Value) -> bool {
    fn axes(v: &Value) -> Option<&[Value]> {
        v.as_array().map(Vec::as_slice)
    }
    // The wire omits fixed axes on a full volume.
    let a = axes(a).unwrap_or_default();
    let b = axes(b).unwrap_or_default();
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            ["name", "frame", "unit"].iter().all(|k| x[k] == y[k])
                && x["coordinate"]
                    .as_f64()
                    .zip(y["coordinate"].as_f64())
                    .is_some_and(|(u, v)| (u - v).abs() <= TOLERANCE)
        })
}

pub(super) fn compare_grid(left: &Value, right: &Value, volume: bool) -> Result<Vec<usize>> {
    let axes = left["axes"].as_array().context("missing axes")?;
    let other = right["axes"]
        .as_array()
        .context("missing comparison axes")?;
    ensure!(
        axes.len() == other.len(),
        mismatch(format!("axis count {} vs {}", axes.len(), other.len()))
    );
    ensure!(
        same_fixed(&left["fixed"], &right["fixed"]),
        mismatch(format!(
            "fixed axes {} vs {}",
            left["fixed"], right["fixed"]
        ))
    );
    if volume {
        ensure!(
            axes.len() == 3
                && ["z", "y", "x"].iter().all(|n| axes
                    .iter()
                    .any(|a| a["name"] == *n && a["frame"] == "volume")),
            "difference requires volume axes x, y, z."
        );
    }
    let mut order = Vec::new();
    for (j, b) in other.iter().enumerate() {
        let i = axes
            .iter()
            .position(|a| a["name"] == b["name"])
            .ok_or_else(|| mismatch(format!("missing axis {}", b["name"])))?;
        ensure!(
            volume || i == j,
            mismatch("axes in different storage order".into())
        );
        ensure!(
            left["shape"][i] == right["shape"][j],
            mismatch(format!("shape {} vs {}", left["shape"], right["shape"]))
        );
        check_axes(
            &axes[i],
            b,
            left["shape"][i].as_u64().context("missing axis length")? as usize,
        )?;
        order.push(i);
    }
    ensure!(
        !volume || left["unit"] == right["unit"],
        mismatch(format!("value units {} vs {}", left["unit"], right["unit"]))
    );
    Ok(order)
}
