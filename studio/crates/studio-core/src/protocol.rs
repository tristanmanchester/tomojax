//! Protocol 1 / progress 2: the official `tomojax serve` wire contract.
//!
//! Arrays remain registered storage references. These types deliberately preserve
//! their physical registration instead of turning results into anonymous pixels.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Storage {
    Npy {
        path: String,
        #[serde(default)]
        artifact: Option<String>,
    },
    Hdf5 {
        path: String,
        dataset: String,
    },
    Member {
        handle: String,
        member: String,
    },
}

impl Storage {
    pub fn artifact(&self) -> Option<&str> {
        match self {
            Self::Npy { artifact, .. } => artifact.as_deref(),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Coordinates {
    Uniform { origin: f64, spacing: f64 },
    Coordinates { values: Storage },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Axis {
    pub name: String,
    pub frame: String,
    pub unit: String,
    pub coordinates: Coordinates,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Fixed {
    pub name: String,
    pub frame: String,
    pub unit: String,
    pub index: usize,
    pub coordinate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Column {
    pub name: String,
    pub unit: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ArrayDescriptor {
    pub storage: Storage,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub axes: Vec<Axis>,
    pub unit: String,
    pub domain: String,
    #[serde(default)]
    pub fixed: Vec<Fixed>,
    #[serde(default)]
    pub columns: Vec<Column>,
}

impl ArrayDescriptor {
    /// Stores file paths inside `root` relative to it.
    pub fn relative_to(&mut self, root: &std::path::Path) {
        let relative = |storage: &mut Storage| {
            if let Storage::Hdf5 { path, .. } | Storage::Npy { path, .. } = storage
                && let Ok(inside) = std::path::Path::new(path.as_str()).strip_prefix(root)
            {
                *path = inside.to_string_lossy().into();
            }
        };
        relative(&mut self.storage);
        for axis in &mut self.axes {
            if let Coordinates::Coordinates { values } = &mut axis.coordinates {
                relative(values);
            }
        }
    }
    pub fn artifacts(&self) -> impl Iterator<Item = &str> {
        self.storage
            .artifact()
            .into_iter()
            .chain(self.axes.iter().filter_map(|axis| match &axis.coordinates {
                Coordinates::Coordinates { values } => values.artifact(),
                Coordinates::Uniform { .. } => None,
            }))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Device {
    pub id: String,
    pub platform: String,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Versions {
    pub tomojax: String,
    pub jax: String,
    pub cuda: Option<String>,
    pub cupy: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ready {
    pub protocol: u8,
    pub progress: u8,
    pub versions: Versions,
    pub capabilities: Vec<String>,
    pub devices: Vec<Device>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChannelRevision {
    pub name: String,
    pub revision: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Grid {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub vx: f64,
    pub vy: f64,
    pub vz: f64,
    #[serde(default)]
    pub vol_origin: Option<[f64; 3]>,
    #[serde(default)]
    pub vol_center: Option<[f64; 3]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub time: f64,
    pub device: String,
    pub scope: String,
    pub used: Option<u64>,
    pub peak: Option<u64>,
    pub available: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Placement {
    pub devices: Vec<String>,
    pub strategy: String,
    pub host_bytes: u64,
    pub device_bytes: u64,
    pub snapshot: Vec<Snapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level {
    pub grid: Grid,
    pub iterations: usize,
    pub image_iterations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CenterRegistrationPlan {
    pub pairs: usize,
    pub reasons: Vec<String>,
    pub placement: Placement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Info {
    pub summary: BTreeMap<String, Value>,
    pub arrays: BTreeMap<String, ArrayDescriptor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Provenance {
    pub operation: String,
    pub parameters: Value,
    pub inputs: Vec<Provenance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Product {
    Scan {
        handle: String,
        projections: ArrayDescriptor,
    },
    Frames {
        handle: String,
        counts: ArrayDescriptor,
    },
    Reconstruction {
        handle: String,
        volume: ArrayDescriptor,
        scan: Option<Storage>,
        method: String,
    },
    Alignment {
        handle: String,
        scan: Storage,
        poses: ArrayDescriptor,
        image: Option<ArrayDescriptor>,
    },
    Simulation {
        handle: String,
        scan: Storage,
        phantom: ArrayDescriptor,
    },
    Plane {
        plane: ArrayDescriptor,
    },
    Projections {
        projections: ArrayDescriptor,
    },
    Volume {
        volume: ArrayDescriptor,
    },
    Sweep {
        planes: ArrayDescriptor,
    },
    CorrectionPreview {
        before: ArrayDescriptor,
        after: ArrayDescriptor,
        views: ArrayDescriptor,
    },
    Region {
        handle: String,
        grid: Grid,
        bounds: [[f64; 2]; 3],
        arrays: BTreeMap<String, ArrayDescriptor>,
    },
    Files {
        paths: Vec<String>,
    },
    Inspection {
        report: Value,
    },
    ReconstructionPlan {
        method: String,
        reasons: Vec<String>,
        iterations: Option<usize>,
        grid: Grid,
        placement: Placement,
    },
    JointPlan {
        reasons: Vec<String>,
        levels: Vec<Level>,
        placement: Placement,
    },
    SinogramFocusPlan {
        views: usize,
        binning: usize,
        #[serde(flatten)]
        registration: CenterRegistrationPlan,
    },
    OppositeViewsPlan(CenterRegistrationPlan),
    Validated,
}

impl Product {
    pub fn is_plan(&self) -> bool {
        matches!(
            self,
            Self::ReconstructionPlan { .. }
                | Self::JointPlan { .. }
                | Self::OppositeViewsPlan(_)
                | Self::SinogramFocusPlan { .. }
        )
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Scan { .. } => "scan",
            Self::Frames { .. } => "frames",
            Self::Reconstruction { .. } => "reconstruction",
            Self::Alignment { .. } => "alignment",
            Self::Simulation { .. } => "simulation",
            Self::Plane { .. } => "plane",
            Self::Projections { .. } => "projections",
            Self::Volume { .. } => "volume",
            Self::Sweep { .. } => "sweep",
            Self::CorrectionPreview { .. } => "correction_preview",
            Self::Region { .. } => "region",
            Self::Files { .. } => "files",
            Self::Inspection { .. } => "inspection",
            Self::ReconstructionPlan { .. } => "reconstruction_plan",
            Self::JointPlan { .. } => "joint_plan",
            Self::SinogramFocusPlan { .. } => "sinogram_focus_plan",
            Self::OppositeViewsPlan(_) => "opposite_views_plan",
            Self::Validated => "validated",
        }
    }

    /// The named scan member is reusable without copying its owning result.
    pub fn scan_source(&self) -> Option<Storage> {
        match self {
            Self::Scan { handle, .. } => Some(Storage::Member {
                handle: handle.clone(),
                member: "self".into(),
            }),
            Self::Simulation { scan, .. } | Self::Alignment { scan, .. } => Some(scan.clone()),
            Self::Reconstruction { scan, .. } => scan.clone(),
            _ => None,
        }
    }

    /// Published members, matching the backend's direct member lookup contract.
    pub fn members(&self) -> BTreeMap<String, Storage> {
        let Some(handle) = self.handle() else {
            return BTreeMap::new();
        };
        let names: Vec<&str> = match self {
            Self::Scan { .. } => vec!["self", "scan", "projections", "angles"],
            Self::Frames { .. } => vec!["self", "counts", "angles"],
            Self::Simulation { .. } => vec!["self", "scan", "projections", "angles", "phantom"],
            Self::Reconstruction { scan, .. } if scan.is_some() => {
                vec!["self", "scan", "projections", "angles", "volume"]
            }
            Self::Reconstruction { .. } => vec!["self", "volume"],
            Self::Alignment { image, .. } if image.is_some() => {
                vec!["self", "scan", "projections", "angles", "poses", "image"]
            }
            Self::Alignment { .. } => vec!["self", "scan", "projections", "angles", "poses"],
            Self::Region { .. } => vec!["self"],
            _ => Vec::new(),
        };
        names
            .into_iter()
            .map(|name| {
                (
                    name.to_owned(),
                    Storage::Member {
                        handle: handle.to_owned(),
                        member: name.to_owned(),
                    },
                )
            })
            .collect()
    }

    pub fn handle(&self) -> Option<&str> {
        match self {
            Self::Scan { handle, .. }
            | Self::Frames { handle, .. }
            | Self::Reconstruction { handle, .. }
            | Self::Alignment { handle, .. }
            | Self::Simulation { handle, .. }
            | Self::Region { handle, .. } => Some(handle),
            _ => None,
        }
    }

    pub fn primary_array(&self) -> Option<&ArrayDescriptor> {
        match self {
            Self::Scan { projections, .. } | Self::Projections { projections } => Some(projections),
            Self::Frames { counts, .. } => Some(counts),
            Self::Reconstruction { volume, .. } | Self::Volume { volume } => Some(volume),
            Self::Alignment { image, .. } => image.as_ref(),
            Self::Simulation { phantom, .. } => Some(phantom),
            Self::Plane { plane } => Some(plane),
            Self::Sweep { planes } => Some(planes),
            Self::CorrectionPreview { after, .. } => Some(after),
            _ => None,
        }
    }

    pub fn arrays(&self) -> Vec<&ArrayDescriptor> {
        match self {
            Self::Alignment { poses, image, .. } => {
                std::iter::once(poses).chain(image.iter()).collect()
            }
            Self::CorrectionPreview {
                before,
                after,
                views,
            } => vec![before, after, views],
            Self::Region { arrays, .. } => arrays.values().collect(),
            _ => self.primary_array().into_iter().collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Output {
    pub value: Product,
    pub info: Info,
    pub manifest: Provenance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResultValue {
    Output(Box<Output>),
    Description {
        parameters: Value,
    },
    Devices {
        devices: Vec<Device>,
    },
    Status {
        requests: BTreeMap<String, String>,
        handles: BTreeMap<String, u64>,
        allocators: Vec<Snapshot>,
    },
    Ack,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Progress {
    Started {
        schema: u8,
        task: u64,
        time: f64,
        parent: Option<u64>,
        name: String,
        label: String,
        total: Option<u64>,
        unit: String,
    },
    Placed {
        schema: u8,
        task: u64,
        time: f64,
        devices: Vec<String>,
    },
    Advanced {
        schema: u8,
        task: u64,
        time: f64,
        done: u64,
        total: Option<u64>,
        metrics: BTreeMap<String, Value>,
        eta: Option<f64>,
    },
    Finished {
        schema: u8,
        task: u64,
        time: f64,
        status: String,
    },
    Note {
        schema: u8,
        task: Option<u64>,
        time: f64,
        level: String,
        message: String,
    },
    Paused {
        schema: u8,
        task: u64,
        time: f64,
    },
    Resumed {
        schema: u8,
        task: u64,
        time: f64,
    },
    Preview {
        schema: u8,
        task: u64,
        time: f64,
        arrays: BTreeMap<String, ArrayDescriptor>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    Input,
    InvalidParameter,
    OutOfMemory,
    Canceled,
    Superseded,
    Internal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerError {
    pub schema: u8,
    pub event: String,
    pub kind: ErrorKind,
    pub message: String,
    #[serde(default)]
    pub path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    Ready(Ready),
    Progress {
        request: String,
        channel: Option<ChannelRevision>,
        record: Progress,
    },
    Result {
        request: String,
        channel: Option<ChannelRevision>,
        result: ResultValue,
    },
    Error {
        request: String,
        channel: Option<ChannelRevision>,
        error: WorkerError,
    },
    ProtocolError {
        kind: String,
        message: String,
        request: Option<String>,
    },
}

impl Response {
    pub fn request(&self) -> Option<&str> {
        match self {
            Self::Progress { request, .. }
            | Self::Result { request, .. }
            | Self::Error { request, .. } => Some(request),
            Self::ProtocolError { request, .. } => request.as_deref(),
            Self::Ready(_) => None,
        }
    }

    pub fn channel(&self) -> Option<&ChannelRevision> {
        match self {
            Self::Progress { channel, .. }
            | Self::Result { channel, .. }
            | Self::Error { channel, .. } => channel.as_ref(),
            _ => None,
        }
    }

    pub fn terminal(&self) -> bool {
        matches!(
            self,
            Self::Result { .. } | Self::Error { .. } | Self::ProtocolError { .. }
        )
    }
}

/// Live parameter metadata, sourced from `describe`, never a parallel option list.
#[derive(Debug, Clone)]
pub struct SchemaCatalog {
    document: Value,
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub description: String,
    pub unit: Option<String>,
    pub default: Option<Value>,
    pub choices: Vec<Value>,
    pub required: bool,
    pub schema: Value,
}

impl SchemaCatalog {
    pub fn new(document: Value) -> Result<Self, String> {
        if !document.get("$defs").is_some_and(Value::is_object) {
            return Err("The worker description has no parameter definitions".into());
        }
        Ok(Self { document })
    }

    pub fn document(&self) -> &Value {
        &self.document
    }

    pub fn definition(&self, name: &str) -> Option<&Value> {
        self.document.get("$defs")?.get(name)
    }

    pub fn parameters(&self, definition: &str) -> Vec<Parameter> {
        let Some(object) = self.definition(definition) else {
            return Vec::new();
        };
        let Some(properties) = object.get("properties").and_then(Value::as_object) else {
            return Vec::new();
        };
        let required = object.get("required").and_then(Value::as_array);
        properties
            .iter()
            .filter(|(name, _)| name.as_str() != "type")
            .map(|(name, schema)| {
                let choices = schema
                    .get("x-choices")
                    .or_else(|| schema.get("enum"))
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                Parameter {
                    name: name.clone(),
                    description: schema
                        .get("description")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .into(),
                    unit: schema
                        .get("x-unit")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    default: schema.get("default").cloned(),
                    choices,
                    required: required
                        .is_some_and(|items| items.iter().any(|v| v.as_str() == Some(name))),
                    schema: schema.clone(),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_transcripts_preserve_registered_arrays() {
        for transcript in [
            include_str!("../../../../docs/protocol/transcripts/session.jsonl"),
            include_str!("../../../../docs/protocol/transcripts/simulation.jsonl"),
            include_str!("../../../../docs/protocol/transcripts/saved-volume.jsonl"),
            include_str!("../../../../docs/protocol/transcripts/alignment-scan.jsonl"),
            include_str!("../../../../docs/protocol/transcripts/raw-slabs.jsonl"),
            include_str!("../../../../docs/protocol/transcripts/edit-combine.jsonl"),
        ] {
            for line in transcript.lines() {
                let json: Value = serde_json::from_str(line).unwrap();
                if json.get("type").is_none() {
                    continue;
                }
                let response: Response = serde_json::from_str(line).unwrap();
                if let Response::Result {
                    result: ResultValue::Output(output),
                    ..
                } = response
                {
                    for array in output.value.arrays() {
                        assert_eq!(array.shape.len(), array.axes.len());
                        assert!(!array.unit.is_empty());
                        assert!(!array.domain.is_empty());
                    }
                }
            }
        }
    }

    #[test]
    fn scalar_form_metadata_comes_from_declared_schema() {
        let schema: Value = serde_json::from_str(include_str!(
            "../../../../docs/protocol/operations.schema.json"
        ))
        .unwrap();
        let catalog = SchemaCatalog::new(schema).unwrap();
        let parameters = catalog.parameters("Reconstruct");
        let method = parameters.iter().find(|p| p.name == "method").unwrap();
        assert_eq!(method.default, Some(Value::String("auto".into())));
        assert!(method.choices.iter().any(|v| v.as_str() == Some("fourier")));
        assert!(
            parameters
                .iter()
                .find(|p| p.name == "source")
                .unwrap()
                .required
        );
    }
}
