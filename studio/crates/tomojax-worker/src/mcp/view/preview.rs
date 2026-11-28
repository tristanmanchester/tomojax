//! Preview products describe the arrays on the executor, including their physical axes.
use super::{Panel, Reader, View, draw, samples};
use crate::mcp::{
    connection::{Inventory, Session},
    ledger,
    tools::query,
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::collections::BTreeSet;

impl Reader<'_> {
    async fn product_panel(
        &self,
        array: &Value,
        title: &str,
        layer: Option<usize>,
        roi: Option<&[[usize; 2]]>,
    ) -> Result<Panel> {
        let shape: Vec<usize> = serde_json::from_value(array["shape"].clone())?;
        let mut windows = Vec::new();
        for (i, n) in shape.iter().enumerate() {
            let interval = if i == 0
                && let Some(z) = layer
            {
                [z, z + 1]
            } else if let Some(crop) = roi {
                let axis = array["axes"][i]["name"]
                    .as_str()
                    .context("Missing preview axis")?;
                crop["zyx".find(axis).context("Expected a volume plane")?]
            } else {
                [0, *n]
            };
            windows.push(interval);
        }
        let item = json!({"array":array,"dimension":layer.map(|_|0),"index":layer.unwrap_or(0),
            "window":windows.iter().map(|[a,b]|[*a,*b,1]).collect::<Vec<_>>(),"display_roi":windows});
        self.panel(&item, title).await
    }

    async fn preview(&self, product: &Value, roi: Option<&[[usize; 2]]>) -> Result<Vec<Panel>> {
        let mut out = vec![];
        match product["type"].as_str() {
            Some("sweep") => {
                let array = &product["planes"];
                let layers = array["shape"][0].as_u64().context("Missing sweep length")?;
                for i in 0..layers as usize {
                    out.push(
                        self.product_panel(array, "Center sweep", Some(i), roi)
                            .await?,
                    );
                }
            }
            Some("region") => {
                for a in product["arrays"]
                    .as_object()
                    .context("Missing region arrays")?
                    .values()
                {
                    out.push(
                        self.product_panel(a, "Estimated region", None, None)
                            .await?,
                    );
                }
            }
            Some("correction_preview") => {
                let views = &product["views"];
                let n = views["shape"][0].as_u64().context("Missing view count")? as usize;
                let indices = samples(self.session, &views["storage"], &[n], "int64").await?;
                let before = *indices.get(n / 2).context("Empty preview")? as usize;
                out.push(
                    self.product_panel(
                        &product["before"],
                        "Before correction / native grid",
                        Some(before),
                        None,
                    )
                    .await?,
                );
                out.push(
                    self.product_panel(
                        &product["after"],
                        "After correction / corrected grid",
                        Some(n / 2),
                        None,
                    )
                    .await?,
                );
            }
            _ => out.push(
                self.product_panel(&product["plane"], "Reconstructed slice", None, roi)
                    .await?,
            ),
        }
        Ok(out)
    }
}

pub(crate) async fn run(
    session: &Session,
    inventory: &Inventory,
    mut command: Value,
) -> Result<(Value, Vec<Value>)> {
    let roi: Option<Vec<[usize; 2]>> = command
        .as_object_mut()
        .unwrap()
        .shift_remove("roi")
        .map(serde_json::from_value)
        .transpose()?;
    if let Some(roi) = &roi {
        ensure!(roi.len() == 3, "preview needs three ROI intervals");
        let dim = command["axis"]
            .as_str()
            .and_then(|a| "zyx".find(a))
            .context("expected a volume axis")?;
        let at = command["index"].as_u64().context("missing plane index")? as usize;
        ensure!(
            roi[dim][0] <= at && at < roi[dim][1],
            "slice index outside volume roi"
        );
    }
    let output = query(session, command.clone()).await?;
    let product = &output["value"];
    let reader = Reader {
        session,
        inventory,
        tokens: BTreeSet::new(),
    };
    let panels = reader.preview(product, roi.as_deref()).await;
    let region = product["type"] == "region";
    if !region {
        let mut release = BTreeSet::new();
        ledger::tokens(product, &mut release);
        for token in release {
            query(session, json!({"type":"release","artifact":token})).await?;
        }
    }
    let form: View = serde_json::from_value(json!({"source":""}))?;
    let (mut view, images) = draw::render(&panels?, &form, 0, None)?;
    let mut data = if region {
        let mut summary = session
            .space
            .ledger
            .lock()
            .unwrap()
            .publish(&command, &output, product, None)?;
        summary["grid"] = product["grid"].clone();
        summary["bounds"] = product["bounds"].clone();
        summary
    } else {
        json!({})
    };
    if let Some(limits) = view.get("intensity_limits") {
        data["intensity_limits"] = limits.clone();
    }
    for panel in view["panels"].as_array_mut().unwrap() {
        panel.as_object_mut().unwrap().shift_remove("image");
    }
    data["panels"] = view["panels"].take();
    Ok((data, images))
}
