//! Names and storage owned by one executor generation.
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value, json};
use std::collections::BTreeSet;
use uuid::Uuid;

#[derive(Default)]
pub struct Ledger {
    pub records: Map<String, Value>,
}

pub fn primary(kind: &str) -> &str {
    match kind {
        "scan" => "projections",
        "frames" => "counts",
        "reconstruction" => "volume",
        "alignment" => "image",
        "simulation" => "phantom",
        "feature" => "mask",
        other => other,
    }
}

fn alias<'a>(product: &Value, kind: &'a str) -> &'a str {
    if product["handle"].is_null() {
        return "array";
    }
    match kind {
        "reconstruction" => "recon",
        "alignment" => "align",
        "simulation" => "sim",
        other => other,
    }
}

pub fn tokens(value: &Value, result: &mut BTreeSet<String>) {
    if value["type"] == "npy"
        && let Some(token) = value["artifact"].as_str()
    {
        result.insert(token.into());
    } else if let Some(fields) = value.as_object() {
        for child in fields.values() {
            tokens(child, result);
        }
    } else if let Some(items) = value.as_array() {
        for child in items {
            tokens(child, result);
        }
    }
}

impl Ledger {
    pub fn record(&self, id: &str) -> Result<&Value> {
        self.records.get(id).with_context(|| {
            format!(
                "Unknown reference {id}. Live handles: {}",
                if self.records.is_empty() {
                    "none".into()
                } else {
                    self.records.keys().cloned().collect::<Vec<_>>().join(", ")
                }
            )
        })
    }
    pub fn resolve(&self, name: &str, as_object: bool) -> Result<Value> {
        let (id, member) = name
            .split_once('.')
            .map_or((name, None), |(id, m)| (id, Some(m)));
        let record = self.record(id)?;
        let product = &record["product"];
        if product["handle"].is_null() {
            ensure!(
                !as_object && member.is_none(),
                "{id} names an array, not an object with members"
            );
            let a = product
                .get("array")
                .or(product.get(primary(product["type"].as_str().unwrap_or(""))))
                .context("Missing array product")?;
            return Ok(a["storage"].clone());
        }
        let kind = product["type"].as_str().context("Missing object kind")?;
        Ok(
            json!({"type":"member","handle":product["handle"],"member":member.unwrap_or(if as_object {"self"} else {primary(kind)})}),
        )
    }
    pub fn publish(
        &mut self,
        cmd: &Value,
        output: &Value,
        product: &Value,
        reserved: Option<&str>,
    ) -> Result<Value> {
        let kind = product["type"].as_str().context("Missing product kind")?;
        let alias = alias(product, kind);
        let id = reserved
            .map(str::to_owned)
            .unwrap_or_else(|| format!("@{alias}-{}", &Uuid::new_v4().simple().to_string()[..8]));
        if self.records.contains_key(&id) {
            bail!("Duplicate published handle {id}");
        }
        let array = product.get(primary(kind)).or(product.get("poses"));
        let mut summary = json!({"kind":kind,"shape":array.and_then(|a|a.get("shape")),"unit":array.and_then(|a|a.get("unit")),"operation":cmd["type"]});
        if cmd["type"] == "open" {
            summary["path"] = cmd["path"].clone();
        }
        let info = &output["info"]["summary"];
        for key in [
            "method",
            "calibration",
            "corrections",
            "diagnostics",
            "disturbed_views",
            "geometry",
            "automatic",
            "truncation",
            "placement",
            "name",
            "voxels",
            "bounds",
            "mean",
            "std",
            "recipe",
        ] {
            if let Some(item) = product.get(key).or(info.get(key)) {
                summary[key] = item.clone();
            }
        }
        if let Some(map) = info.as_object() {
            for (k, v) in map {
                if summary.get(k).is_none() && !v.is_array() && !v.is_object() {
                    summary[k] = v.clone();
                }
            }
        }
        if let Some(disturbance) = summary
            .get_mut("disturbed_views")
            .and_then(Value::as_object_mut)
        {
            for (key, value) in disturbance {
                if let Some(array) = output["info"]["arrays"].get(format!("/disturbed_views/{key}"))
                {
                    *value = array.clone();
                }
            }
        }
        self.records.insert(
            id.clone(),
            json!({"summary":summary,"product":product,"manifest":output["manifest"]}),
        );
        let mut result = json!({"handle":id});
        result
            .as_object_mut()
            .unwrap()
            .extend(summary.as_object().unwrap().clone());
        let members = product
            .as_object()
            .unwrap()
            .iter()
            .filter(|(_, v)| {
                alias != "array"
                    && v.is_object()
                    && (v.get("storage").is_some() || v["type"] == "member")
            })
            .map(|(k, _)| (k.clone(), json!(format!("{id}.{k}"))))
            .collect();
        result["members"] = Value::Object(members);
        if output["info"]["arrays"]
            .as_object()
            .is_some_and(|a| a.keys().any(|k| k.starts_with("/corrections/")))
        {
            result["info"] = output["info"].clone();
        }
        if kind == "alignment" {
            result["poses"] = json!({"shape":product["poses"]["shape"],"columns":product["poses"]["columns"],"rms":info["pose_rms"]});
        }
        Ok(result)
    }
}
