//! Read worker-owned hyperslabs and compare the scientific samples before converting to gray.
mod draw;
use super::{
    connection::{Inventory, Session},
    jobs, ledger,
    tools::query,
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeSet, sync::atomic::Ordering};
use studio_core::pixels::{Encoding, Pixels};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct View {
    source: Sources,
    mode: Option<String>,
    axis: Option<String>,
    index: Option<usize>,
    other: Option<String>,
    limits: Option<[f64; 2]>,
    stride: Option<Vec<usize>>,
    roi: Option<Vec<[usize; 2]>>,
    percentiles: Option<Vec<f64>>,
    bins: Option<usize>,
}
#[derive(Deserialize)]
#[serde(untagged)]
enum Sources {
    One(String),
    Many(Vec<String>),
}
impl Sources {
    fn names(&self) -> &[String] {
        match self {
            Self::One(s) => std::slice::from_ref(s),
            Self::Many(v) => v,
        }
    }
}

struct Panel {
    title: String,
    unit: String,
    domain: String,
    width: usize,
    height: usize,
    aspect: f64,
    labels: Vec<String>,
    axes: [String; 2],
    pixels: Vec<f64>,
}

fn source(session: &Session, inventory: &Inventory, name: &str) -> Result<Value> {
    if name.starts_with('@') {
        return session.ledger.lock().unwrap().resolve(name, false);
    }
    if let Some((path, dataset)) = name.split_once("::/") {
        return Ok(
            json!({"type":"hdf5","path":jobs::path(inventory,path)?,"dataset":format!("/{dataset}")}),
        );
    }
    let path = jobs::path(inventory, name)?;
    Ok(
        json!({"type":if path.extension().is_some_and(|s|s.eq_ignore_ascii_case("npy")){"npy"}else{"path"},"path":path}),
    )
}
async fn samples(
    session: &Session,
    storage: &Value,
    shape: &[usize],
    dtype: &str,
) -> Result<Vec<f64>> {
    let f32 = dtype == "float32";
    let encoding = if f32 {
        Encoding::Float32
    } else {
        Encoding::Float64
    };
    let reply = session
        .request(json!({"type":"fetch","storage":storage,"shape":shape,"encoding":encoding}))
        .await?;
    let mut p: Pixels = serde_json::from_value(reply.result()?.clone())?;
    p.attach(reply.bytes).map_err(anyhow::Error::msg)?;
    if f32 {
        Ok(p.floats(shape)
            .map_err(anyhow::Error::msg)?
            .into_iter()
            .map(f64::from)
            .collect())
    } else {
        p.doubles(shape).map_err(anyhow::Error::msg)
    }
}
async fn coordinates(session: &Session, axis: &Value, n: usize) -> Result<Vec<f64>> {
    let coords = &axis["coordinates"];
    if coords["type"] == "coordinates" {
        return samples(session, &coords["values"], &[n], "float64").await;
    }
    let origin = coords["origin"].as_f64().context("missing axis origin")?;
    let step = coords["spacing"].as_f64().context("missing axis spacing")?;
    Ok((0..n).map(|i| origin + i as f64 * step).collect())
}

struct Reader<'a> {
    session: &'a Session,
    inventory: &'a Inventory,
    tokens: BTreeSet<String>,
}
impl Reader<'_> {
    async fn read(&mut self, cmd: Value) -> Result<Value> {
        let output = query(self.session, cmd).await?;
        let v = output.get("value").context("missing read result")?;
        ledger::tokens(v, &mut self.tokens);
        Ok(v.clone())
    }
    async fn layout(&mut self, src: &Value) -> Result<Value> {
        let v = self
            .read(json!({"type":"read","source":src,"metadata_only":true}))
            .await?;
        let mut grid = v["array"].clone();
        let shape: Vec<usize> = serde_json::from_value(grid["shape"].clone())?;
        for (ax, n) in grid["axes"]
            .as_array_mut()
            .context("missing array axes")?
            .iter_mut()
            .zip(shape)
        {
            if ax["coordinates"]["type"] == "coordinates" {
                ax["coordinates"] = json!({"values":coordinates(self.session,ax,n).await?});
            }
        }
        Ok(grid)
    }
    async fn panel(&self, item: &Value, title: &str) -> Result<Panel> {
        let a = &item["array"];
        let dims: Vec<usize> = serde_json::from_value(a["shape"].clone())?;
        let axes = a["axes"].as_array().context("missing axes")?;
        ensure!(axes.len() == dims.len(), "axis count differs from shape");
        let data = samples(
            self.session,
            &a["storage"],
            &dims,
            a["dtype"].as_str().unwrap_or(""),
        )
        .await?;
        let dim = item["dimension"].as_u64().map(|d| d as usize);
        let mut labels = vec![];
        if let Some(fixed) = a["fixed"].as_array() {
            for f in fixed {
                labels.push(format!(
                    "{}[{}] = {} {}",
                    f["name"].as_str().unwrap_or(""),
                    f["index"],
                    draw::number(f["coordinate"].as_f64().context("missing coordinate")?),
                    f["unit"].as_str().unwrap_or("")
                ));
            }
        }
        if let Some(i) = dim {
            let ax = axes.get(i).context("missing plane axis")?;
            ensure!(dims[i] == 1, "read returned more than one plane");
            labels.push(format!(
                "{}[{}] = {} {}",
                ax["name"].as_str().unwrap_or(""),
                item["index"],
                draw::number(coordinates(self.session, ax, 1).await?[0]),
                ax["unit"].as_str().unwrap_or("")
            ));
        }
        let kept: Vec<_> = (0..dims.len()).filter(|i| Some(*i) != dim).collect();
        let [r, c]: [usize; 2] = kept
            .try_into()
            .map_err(|_| anyhow::anyhow!("Expected two displayed axes"))?;
        for i in [r, c] {
            let coord = coordinates(self.session, &axes[i], dims[i]).await?;
            labels.push(format!(
                "{} [{}, {}] {}",
                axes[i]["name"].as_str().unwrap_or(""),
                draw::number(*coord.first().context("empty axis")?),
                draw::number(*coord.last().unwrap()),
                axes[i]["unit"].as_str().unwrap_or("")
            ));
        }
        if let Some(sampling) = item["sampling"].as_array() {
            labels.extend(sampling.iter().filter_map(Value::as_str).map(str::to_owned));
        }
        let aspect = if axes[r]["unit"] == axes[c]["unit"]
            && let (Some(y), Some(x)) = (
                axes[r]["coordinates"]["spacing"].as_f64(),
                axes[c]["coordinates"]["spacing"].as_f64(),
            ) {
            (y / x).abs()
        } else {
            1.
        };
        Ok(Panel {
            title: title.into(),
            unit: a["unit"].as_str().unwrap_or("").into(),
            domain: a["domain"].as_str().unwrap_or("").into(),
            width: dims[c],
            height: dims[r],
            aspect,
            labels,
            axes: [r, c].map(|i| axes[i]["name"].as_str().unwrap_or("").into()),
            pixels: data,
        })
    }
    async fn view(&mut self, form: &View) -> Result<(Value, Vec<Value>)> {
        let names = form.source.names();
        let first = names.first().context("give a source")?;
        let mode = form.mode.as_deref().unwrap_or("planes");
        let src = source(self.session, self.inventory, first)?;
        if mode == "statistics" {
            ensure!(names.len() == 1, "statistics needs one source");
            let mut cmd =
                json!({"type":"statistics","source":src,"roi":form.roi,"stride":form.stride});
            if let Some(p) = &form.percentiles {
                cmd["percentiles"] = json!(p);
            }
            if let Some(n) = form.bins {
                cmd["bins"] = json!(n);
            }
            return Ok((self.read(cmd).await?, vec![]));
        }
        if matches!(mode, "profile" | "pose_table") {
            ensure!(names.len() == 1, "{mode} needs one source");
            let v = self
                .read(json!({"type":"read","source":src,"roi":form.roi,"stride":form.stride}))
                .await?;
            let a = &v["array"];
            let dims: Vec<usize> = serde_json::from_value(a["shape"].clone())?;
            let pixels = samples(
                self.session,
                &a["storage"],
                &dims,
                a["dtype"].as_str().unwrap_or(""),
            )
            .await?;
            if mode == "profile" {
                let dims: Vec<_> = dims.iter().enumerate().filter(|(_, n)| **n > 1).collect();
                ensure!(
                    dims.len() == 1,
                    "profile needs a line ROI with one varying axis"
                );
                let ax = &a["axes"][dims[0].0];
                let coords = coordinates(self.session, ax, pixels.len()).await?;
                return Ok((
                    json!({"profile":pixels,"unit":a["unit"],"axis":ax["name"],"coordinates":coords,"axis_unit":ax["unit"]}),
                    vec![],
                ));
            }
            ensure!(
                a["domain"] == "poses" && dims.len() == 2 && dims[1] == 6,
                "choose a registered .poses member with all six columns"
            );
            let ax = &a["axes"][0];
            let coords = coordinates(self.session, ax, dims[0]).await?;
            return Ok((
                json!({"columns":a["columns"],"views":dims[0],"view_unit":ax["unit"],"coordinates":coords,"table":pixels.chunks(6).collect::<Vec<_>>()}),
                vec![],
            ));
        }
        ensure!(
            matches!(mode, "planes" | "projection" | "sinogram" | "difference"),
            "Unknown view mode {mode}"
        );
        let mut panels = vec![];
        let need_grid = names.len() > 1 || mode == "difference";
        let reference = if need_grid {
            Some(self.layout(&src).await?)
        } else {
            None
        };
        for name in names {
            let src = source(self.session, self.inventory, name)?;
            if name != first
                && let Some(reference) = &reference
            {
                let grid = self.layout(&src).await?;
                compare_grid(reference, &grid, false)?;
            }
            let kind = if mode == "difference" { "planes" } else { mode };
            let cmd = json!({"type":"read","source":src,"mode":kind,"axis":form.axis,"index":form.index,"roi":form.roi,"stride":form.stride});
            let v = self.read(cmd.clone()).await?;
            let items = v["planes"].as_array().context("missing planes")?;
            let mut other = vec![];
            if mode == "difference" {
                ensure!(names.len() == 1, "difference takes one source and other");
                let name = form.other.as_deref().context("difference requires other")?;
                let src = source(self.session, self.inventory, name)?;
                let grid = self.layout(&src).await?;
                let reference = reference.as_ref().unwrap();
                let order = compare_grid(reference, &grid, true)?;
                for item in items {
                    let d = item["dimension"]
                        .as_u64()
                        .context("difference requires a volume")?
                        as usize;
                    let mut read = cmd.clone();
                    read["source"] = src.clone();
                    read["axis"] = reference["axes"][d]["name"].clone();
                    read["index"] = item["index"].clone();
                    if let Some(roi) = &form.roi {
                        read["roi"] = json!(order.iter().map(|i| roi[*i]).collect::<Vec<_>>());
                    }
                    if let Some(stride) = &form.stride {
                        read["stride"] =
                            json!(order.iter().map(|i| stride[*i]).collect::<Vec<_>>());
                    }
                    let v = self.read(read).await?;
                    other.push(self.panel(&v["planes"][0], name).await?);
                }
            }
            for (j, item) in items.iter().enumerate() {
                let mut p = self.panel(item, name).await?;
                if name.ends_with(".image") {
                    p.title = format!("Alignment working image {name}");
                }
                if let Some(q) = other.get(j) {
                    let transpose = p.axes != q.axes;
                    ensure!(
                        (!transpose && p.width == q.width && p.height == q.height)
                            || (transpose
                                && p.axes[0] == q.axes[1]
                                && p.axes[1] == q.axes[0]
                                && p.width == q.height
                                && p.height == q.width),
                        "difference plane registration differs"
                    );
                    for (i, a) in p.pixels.iter_mut().enumerate() {
                        *a -= q.pixels[if transpose {
                            i % p.width * q.width + i / p.width
                        } else {
                            i
                        }];
                    }
                    p.title = format!("{name} minus {}", q.title);
                }
                panels.push(p);
            }
        }
        let first = self
            .session
            .renders
            .fetch_add(panels.len().div_ceil(3) as u64, Ordering::Relaxed);
        draw::render(&panels, form.limits, first)
    }
}

fn compare_grid(left: &Value, right: &Value, volume: bool) -> Result<Vec<usize>> {
    let axes = left["axes"].as_array().context("missing axes")?;
    let other = right["axes"]
        .as_array()
        .context("missing comparison axes")?;
    ensure!(
        axes.len() == other.len() && left["fixed"] == right["fixed"],
        "grid mismatch: axes or fixed planes differ"
    );
    if volume {
        ensure!(
            left["unit"] == right["unit"]
                && axes.len() == 3
                && ["z", "y", "x"].iter().all(|name| axes
                    .iter()
                    .any(|v| v["name"] == *name && v["frame"] == "volume")),
            "difference requires matching value units and volume axes x, y, z."
        );
    }
    let mut order = vec![];
    for (j, b) in other.iter().enumerate() {
        let i = axes
            .iter()
            .position(|a| a["name"] == b["name"])
            .context("grid mismatch: axis names differ")?;
        let a = &axes[i];
        ensure!(
            (volume || i == j)
                && a["frame"] == b["frame"]
                && a["unit"] == b["unit"]
                && left["shape"][i] == right["shape"][j],
            "grid mismatch: axis order, frame, units or dimensions differ"
        );
        let n = left["shape"][i].as_u64().context("missing axis length")? as usize;
        let (ac, bc) = (&a["coordinates"], &b["coordinates"]);
        let at = |c: &Value, k: usize| -> Result<f64> {
            if let Some(origin) = c["origin"].as_f64() {
                Ok(origin + c["spacing"].as_f64().context("missing pitch")? * k as f64)
            } else {
                c["values"][k].as_f64().context("missing coordinate")
            }
        };
        if ac["type"] == "uniform" && bc["type"] == "uniform" {
            for k in [0, 1, n.saturating_sub(1)] {
                ensure!(
                    (at(ac, k)? - at(bc, k)?).abs() <= 1e-6,
                    "grid mismatch: axis coordinates differ"
                );
            }
        } else {
            for k in 0..n {
                ensure!(
                    (at(ac, k)? - at(bc, k)?).abs() <= 1e-6,
                    "grid mismatch: axis coordinate {k} differs"
                );
            }
        }
        order.push(i);
    }
    Ok(order)
}
pub async fn run(
    session: &Session,
    inventory: &Inventory,
    args: Value,
) -> Result<(Value, Vec<Value>)> {
    let mut form: View = serde_json::from_value(args)?;
    if form.mode.as_deref() == Some("pose_table")
        && let Sources::One(name) = &mut form.source
        && name.starts_with('@')
        && !name.contains('.')
    {
        name.push_str(".poses");
    }
    let mut reader = Reader {
        session,
        inventory,
        tokens: BTreeSet::new(),
    };
    let result = reader.view(&form).await;
    for token in reader.tokens {
        query(session, json!({"type":"release","artifact":token})).await?;
    }
    result
}
