//! Immutable requests captured from the project when the user queues an operation.
use super::{CapturedInput, Job};
use crate::protocol::{ArrayDescriptor, Coordinates, Storage};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Script {
    pub lines: Vec<Value>,
    pub roles: Option<ProjectionRoles>,
    pub options: String,
    pub intent: Option<u64>,
    #[serde(default)]
    pub offset: u64,
    pub launcher: Option<String>,
    pub checkpoint: Option<Checkpoint>,
    pub seen: Option<u64>,
    #[serde(default)]
    pub pause_requested: bool,
    #[serde(default)]
    pub cancel_pending: bool,
    #[serde(default)]
    pub devices: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Checkpoint {
    pub line: usize,
    pub path: String,
    pub usable: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectionRoles {
    pub geometry: usize,
    pub array: Option<usize>,
    key: String,
    member: Option<String>,
    storage: Option<Storage>,
    grid: Option<Value>,
}

impl ProjectionRoles {
    pub fn choose(op: &str, inputs: &[CapturedInput], parameters: &Value) -> Result<Self, String> {
        let forward = op == "project";
        let key = if forward { "volume" } else { "projections" };
        let mut candidates = inputs.iter().enumerate().filter_map(|(index, input)| {
            let (rank, member, registered) = match (forward, input.kind.as_str()) {
                (true, "volume") => (0, "volume", true),
                (true, "reconstruction") => (1, "volume", false),
                (true, "alignment")
                    if input.arrays.contains_key("image")
                        || input.arrays.contains_key("volume") =>
                {
                    (2, "image", false)
                }
                (true, "simulation") => (3, "phantom", false),
                (false, "projections") => (0, "projections", true),
                (false, "scan") => (1, "projections", false),
                (false, _) if input.has_scan() => (2, "projections", false),
                _ => return None,
            };
            Some((rank, index, member, registered))
        });
        let selected = if parameters[key].is_null() {
            Some(candidates.by_ref().min().ok_or_else(|| {
                format!("Select an input containing {key}, or provide its storage reference")
            })?)
        } else {
            None
        };
        let array = selected.map(|(_, index, _, _)| index);
        let scans = || {
            inputs
                .iter()
                .enumerate()
                .filter(|(_, i)| i.has_scan())
                .map(|(i, _)| i)
        };
        let geometry = match selected {
            Some((_, i, _, false)) if !forward => Some(i),
            Some((_, i, _, false)) => scans().find(|&j| j != i).or_else(|| scans().next()),
            _ => scans().next(),
        }
        .ok_or("Select a scan to supply the geometry")?;
        let mut storage = None;
        let mut member = None;
        let mut grid = None;
        if let Some((_, index, name, registered)) = selected {
            let arrays = &inputs[index].arrays;
            let descriptor = arrays.get(name).or_else(|| arrays.get(key));
            if registered {
                storage = Some(
                    descriptor
                        .ok_or("The selected array has no storage")?
                        .storage
                        .clone(),
                );
            } else {
                member = Some(name.into());
            }
            if forward && parameters["grid"].is_null() {
                grid = descriptor.and_then(registered_grid);
            }
        }
        Ok(Self {
            geometry,
            array,
            key: key.into(),
            member,
            storage,
            grid,
        })
    }

    pub fn apply(&self, operation: &mut Value, handle: impl FnOnce(usize) -> String) {
        if let Some(storage) = &self.storage {
            operation[&self.key] = json!(storage);
        } else if let (Some(index), Some(member)) = (self.array, &self.member) {
            operation[&self.key] = json!({"type":"member","handle":handle(index),"member":member});
        }
        if let Some(grid) = &self.grid {
            operation["grid"] = grid.clone();
        }
    }

    pub fn parent(&self) -> Option<usize> {
        if self.key == "volume" {
            Some(self.geometry)
        } else {
            self.array
        }
    }

    pub fn describe(&self, inputs: &[CapturedInput]) -> String {
        let what = if self.key == "volume" {
            "Volume"
        } else {
            "Projections"
        };
        let array = self.array.map_or("given by reference", |i| &inputs[i].name);
        format!("{what}: {array} · geometry: {}", inputs[self.geometry].name)
    }
}

impl CapturedInput {
    pub fn dataset(&self) -> bool {
        matches!(
            self.kind.as_str(),
            "scan" | "frames" | "reconstruction" | "simulation" | "alignment" | "feature"
        )
    }
    fn has_scan(&self) -> bool {
        matches!(self.kind.as_str(), "scan" | "simulation" | "alignment")
            || (self.kind == "reconstruction" && self.arrays.contains_key("projections"))
    }
}

impl Script {
    pub fn build(
        job: &Job,
        previews: bool,
        options: String,
        intent: Option<u64>,
    ) -> Result<Self, String> {
        let mut lines = Vec::new();
        let mut roles = None;
        let id = |index: usize| format!("{}.open-{index}", job.id);
        let op = job.operation.as_str();
        let mut operation = job.parameters.clone();
        if !operation.is_object() {
            return Err("Job settings must be an object".into());
        }
        operation["type"] = json!(if op == "plan" { "reconstruct" } else { op });
        let first = || job.inputs.first().ok_or("Select an input");
        let source = |index: usize| -> Result<Value, String> {
            let input = job.inputs.get(index).ok_or("Select an input")?;
            let member = match input.kind.as_str() {
                "scan" | "frames" => "self",
                "reconstruction" | "simulation" | "alignment" => "scan",
                _ => return Err("Select a dataset to supply the scan".into()),
            };
            Ok(json!({"type":"member","member":member,"handle":id(index)}))
        };
        let command = match op {
            "open" => {
                let input = first()?;
                operation["path"] = json!(input.path);
                if operation["kind"].is_null() {
                    operation["kind"] = json!(input.kind);
                }
                operation
            }
            "inspect" => json!({"type":"inspect","path":first()?.path}),
            _ => {
                for (i, input) in job
                    .inputs
                    .iter()
                    .enumerate()
                    .filter(|(_, input)| input.dataset())
                {
                    let mut open = input.open_parameters.clone();
                    open["type"] = json!("open");
                    open["path"] = json!(input.path);
                    open["kind"] = json!(input.kind);
                    lines.push(json!({"id":id(i),"command":open}));
                }
                match op {
                    "simulate" => {}
                    "combine" => {
                        operation["sources"] = json!(
                            (0..job.inputs.len())
                                .map(source)
                                .collect::<Result<Vec<_>, _>>()?
                        )
                    }
                    "save" | "export" => {
                        first()?;
                        let member = operation
                            .as_object_mut()
                            .and_then(|map| map.remove("member"))
                            .unwrap_or(json!("self"));
                        operation["source"] =
                            json!({"type":"member","handle":id(0),"member":member});
                        operation["force"] = json!(false);
                    }
                    "project" | "backproject" => {
                        let chosen = ProjectionRoles::choose(op, &job.inputs, &operation)?;
                        operation["source"] = source(chosen.geometry)?;
                        chosen.apply(&mut operation, id);
                        roles = Some(chosen);
                    }
                    _ => operation["source"] = source(0)?,
                }
                if op == "plan" {
                    json!({"type":"plan","operation":operation})
                } else {
                    json!({"type":"run","operation":operation,"previews":previews && matches!(op,"align"|"reconstruct")})
                }
            }
        };
        lines.push(json!({"id":format!("{}.result",job.id),"command":command}));
        Ok(Self {
            lines,
            roles,
            options,
            intent,
            offset: 0,
            launcher: None,
            checkpoint: None,
            seen: None,
            pause_requested: false,
            cancel_pending: false,
            devices: vec![],
        })
    }
}
fn registered_grid(volume: &ArrayDescriptor) -> Option<Value> {
    let names: Vec<_> = volume.axes.iter().map(|axis| axis.name.as_str()).collect();
    if names != ["z", "y", "x"] || volume.shape.len() != 3 {
        return None;
    }
    let mut uniform = volume.axes.iter().map(|axis| match axis.coordinates {
        Coordinates::Uniform { origin, spacing } => Some((origin, spacing)),
        Coordinates::Coordinates { .. } => None,
    });
    let (z, y, x) = (uniform.next()??, uniform.next()??, uniform.next()??);
    Some(json!({
        "nx": volume.shape[2], "ny": volume.shape[1], "nz": volume.shape[0],
        "vx": x.1, "vy": y.1, "vz": z.1,
        "vol_origin": [x.0, y.0, z.0],
    }))
}
