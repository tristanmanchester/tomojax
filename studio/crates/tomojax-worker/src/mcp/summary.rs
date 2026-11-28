//! Human-readable summaries share the Python CLI's numeric and collection formatting.
use serde_json::Value;
fn exponent(text: &str) -> String {
    if let Some((n, e)) = text.split_once('e')
        && let Ok(exp) = e.parse::<i32>()
    {
        format!("{n}e{exp:+03}")
    } else {
        text.into()
    }
}
fn repr(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(b) => {
            if *b {
                "True".into()
            } else {
                "False".into()
            }
        }
        Value::Number(n) if n.is_f64() => exponent(&format!("{:?}", n.as_f64().unwrap())),
        Value::Number(n) => n.to_string(),
        Value::String(s) => {
            let quote = if s.contains('\'') && !s.contains('"') {
                '"'
            } else {
                '\''
            };
            let escaped = s
                .replace('\\', "\\\\")
                .replace(quote, &format!("\\{quote}"))
                .replace('\n', "\\n")
                .replace('\r', "\\r")
                .replace('\t', "\\t");
            format!("{quote}{escaped}{quote}")
        }
        Value::Array(a) => format!("[{}]", a.iter().map(repr).collect::<Vec<_>>().join(", ")),
        Value::Object(m) => format!(
            "{{{}}}",
            m.iter()
                .map(|(k, v)| format!("{}: {}", repr(&Value::String(k.clone())), repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}
fn general(v: f64, precision: usize) -> String {
    let scientific = format!("{:.*e}", precision - 1, v);
    let Some((mantissa, exp)) = scientific.split_once('e') else {
        return scientific;
    };
    let power = exp.parse::<i32>().unwrap();
    if power < -4 || power >= precision as i32 {
        return exponent(&format!(
            "{}e{exp}",
            mantissa.trim_end_matches('0').trim_end_matches('.')
        ));
    }
    let plain = format!("{:.*}", (precision as i32 - 1 - power) as usize, v);
    if plain.contains('.') {
        plain.trim_end_matches('0').trim_end_matches('.').into()
    } else {
        plain
    }
}
fn beyond_detector(data: &Value) -> Option<String> {
    let truncation = &data["truncation"];
    let mut details = vec![];
    if let Some(f) = truncation["outside_fraction"]
        .as_f64()
        .filter(|v| *v >= 0.005)
    {
        details.push(format!(
            "{:.0}% of attenuation beyond the detector",
            f * 100.
        ))
    }
    match truncation["background_treatment"].as_str() {
        Some("subtracted") => details.push("modeled exterior subtracted from fit data".into()),
        Some("additive_poisson") => {
            details.push("modeled exterior added to the Poisson mean".into())
        }
        _ => {}
    }
    (!details.is_empty()).then(|| details.join(", "))
}
pub fn summary(data: &Value) -> Option<String> {
    let time = data["wall_seconds"].as_f64()?;
    let kind = data["kind"].as_str()?;
    let mut parts = vec![
        data["handle"].as_str().unwrap_or("Saved").to_owned(),
        kind.into(),
    ];
    if let Some(shape) = data["shape"].as_array() {
        let sizes = shape
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join(" x ");
        parts.push(format!(
            "{sizes} {}",
            data["unit"].as_str().unwrap_or("None")
        ));
    }
    if let Some(method) = data["method"].as_str() {
        parts.push(format!("method {method}"));
    }
    if let (Some(n), Some(reason)) = (
        data["effective_iterations"].as_u64(),
        data["termination"].as_str(),
    ) {
        let stop = match reason {
            "projection_discrepancy" => "stopped at the estimated projection discrepancy",
            "noise_score" => "stopped at the estimated noise level",
            "objective_tolerance" => "stopped at the objective tolerance",
            "roundoff_limit" => "stopped at the float32 roundoff limit",
            "numerical_breakdown" => "stopped at a numerical breakdown",
            "iteration_limit" => "ran to the iteration limit",
            other => other,
        };
        let mut message = format!("{n} updates, {}", stop.replace('_', " "));
        if data["tv_weight"].is_null()
            && let Some(weight) = data["resolved_tv_weight"].as_f64()
        {
            message += &format!("; automatic TV weight {}", general(weight, 3));
        }
        parts.push(message);
    }
    for key in [
        "geometry",
        "calibration",
        "checkpoint",
        "slice_count",
        "first_file",
        "last_file",
    ] {
        if let Some(value) = data.get(key) {
            parts.push(format!(
                "{key} {}",
                value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| repr(value))
            ));
        }
    }
    if let Some(a) = data.pointer("/automatic/reasons").and_then(Value::as_array) {
        parts.extend(a.iter().filter_map(Value::as_str).map(str::to_owned));
    }
    parts.extend(beyond_detector(data));
    if let (Some(rms), Some(cols)) = (
        data.pointer("/poses/rms").and_then(Value::as_array),
        data.pointer("/poses/columns").and_then(Value::as_array),
    ) {
        let terms = cols
            .iter()
            .zip(rms)
            .map(|(col, v)| {
                format!(
                    "{} {} {}",
                    col["name"].as_str().unwrap_or(""),
                    general(v.as_f64().unwrap_or_default(), 4),
                    col["unit"].as_str().unwrap_or("")
                )
            })
            .collect::<Vec<_>>();
        parts.push(format!("pose RMS {}", terms.join(", ")));
    }
    if let Some(m) = data
        .pointer("/disturbed_views/message")
        .and_then(Value::as_str)
    {
        parts.push(m.into());
    }
    for k in ["paths", "devices"] {
        if let Some(a) = data[k].as_array() {
            parts.extend(a.iter().filter_map(Value::as_str).map(str::to_owned));
        }
    }
    parts.push(format!("{} s", general(time, 6)));
    if let Some(a) = data["notes"].as_array() {
        parts.extend(a.iter().filter_map(Value::as_str).map(str::to_owned));
    }
    Some(parts.join(" · "))
}
