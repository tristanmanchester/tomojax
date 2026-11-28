//! The native view contract: drafts belong to UiState; facts arrive in UiModel.
//! UiAction is the only route from a view to project, engine or filesystem effects.
use std::collections::{BTreeMap, BTreeSet};

use rune::TextEdit;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Workspace {
    Prepare,
    Reconstruct,
    #[default]
    Explore,
    Compare,
    Jobs,
}
impl Workspace {
    pub fn for_operation(id: &str) -> Self {
        if crate::operations::lookup(id).is_some_and(|(group, _)| group.reconstruction) {
            Self::Reconstruct
        } else {
            Self::Prepare
        }
    }
    pub const ALL: [Self; 5] = [
        Self::Prepare,
        Self::Reconstruct,
        Self::Explore,
        Self::Compare,
        Self::Jobs,
    ];
    pub fn title(self) -> &'static str {
        match self {
            Self::Prepare => "Prepare",
            Self::Reconstruct => "Reconstruct",
            Self::Explore => "Explore",
            Self::Compare => "Compare",
            Self::Jobs => "Jobs",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InspectTool {
    #[default]
    Navigate,
    Distance,
    Region,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlaneKind {
    Axial,
    Coronal,
    Sagittal,
    Projection,
    Sinogram,
    #[default]
    Volume,
}
impl PlaneKind {
    pub fn title(self) -> &'static str {
        match self {
            Self::Axial => "Axial",
            Self::Coronal => "Coronal",
            Self::Sagittal => "Sagittal",
            Self::Projection => "Projection",
            Self::Sinogram => "Sinogram",
            Self::Volume => "3D volume",
        }
    }
    pub fn axes(self) -> &'static str {
        match self {
            Self::Axial => "XY",
            Self::Coronal => "XZ",
            Self::Sagittal => "YZ",
            Self::Projection => "u · v",
            Self::Sinogram => "u · view",
            Self::Volume => "x · y · z",
        }
    }
    pub fn axis(self) -> Option<usize> {
        match self {
            Self::Axial => Some(0),
            Self::Coronal => Some(1),
            Self::Sagittal => Some(2),
            _ => None,
        }
    }
    /// Compositor marks 1–5 are reserved for the linked strip.
    pub fn mark(self) -> u32 {
        match self {
            Self::Axial => 1,
            Self::Coronal => 2,
            Self::Sagittal => 3,
            Self::Projection => 4,
            Self::Sinogram => 5,
            Self::Volume => 8,
        }
    }
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum CompareMode {
    #[default]
    Pair,
    Split,
    Difference,
}

pub struct UiState {
    pub workspace: Workspace,
    pub main_view: PlaneKind,
    pub preview_array: String,
    pub preview_span: TextEdit,
    pub tree_open: bool,
    pub inspector_open: bool,
    pub activity: Activity,
    pub palette_open: bool,
    pub shortcuts_open: bool,
    pub advanced_open: bool,
    pub import_copy: bool,
    pub compare_linked: bool,
    pub compare_plane: PlaneKind,
    pub compare_mode: CompareMode,
    pub probe: Option<(u32, [f32; 2])>,
    pub compare_cursor: [f32; 2],
    /// Curtain position in Split mode, as a fraction of the image width from the left.
    pub compare_split: f32,
    /// The open result picker in Compare: 0 left, 1 right.
    pub compare_menu: Option<usize>,
    pub worker_trace: Option<String>,
    pub asset_query: TextEdit,
    pub rename_asset: Option<String>,
    pub rename_draft: TextEdit,
    /// The asset awaiting confirmation of its removal from the project.
    pub confirm_remove: Option<String>,
    pub command_query: TextEdit,
    pub shortcut_query: TextEdit,
    pub shortcut_binding: TextEdit,
    pub editing_shortcut: Option<String>,
    /// The shortcut editor is waiting for a key combination.
    pub recording_shortcut: bool,
    pub recipe_name: TextEdit,
    /// Keys are the schema's parameter names. Root initializes every visible edit.
    pub parameter_edits: BTreeMap<String, TextEdit>,
    pub parameter_choices: BTreeMap<String, String>,
    pub parameter_toggles: BTreeMap<String, bool>,
    pub corrections: super::corrections::CorrectionEditor,
    pub operation: String,
    /// The operation each processing workspace last showed, restored on return.
    pub prepare_operation: String,
    pub reconstruct_operation: String,
    pub operation_menu: bool,
    pub choice_menu: Option<String>,
    pub export_format: String,
    pub export_path: Option<std::path::PathBuf>,
    pub export_edits: BTreeMap<String, TextEdit>,
    pub selected_job: Option<String>,
    pub command_index: usize,
    pub collapsed_assets: BTreeSet<String>,
    pub drag_last: [f32; 2],
    pub inspect_tool: InspectTool,
    pub display: DisplayState,
}
impl Default for UiState {
    fn default() -> Self {
        Self {
            workspace: Workspace::Explore,
            main_view: PlaneKind::Volume,
            preview_array: "after".into(),
            preview_span: TextEdit::new("8"),
            tree_open: true,
            inspector_open: true,
            activity: Activity::Closed,
            palette_open: false,
            shortcuts_open: false,
            advanced_open: false,
            import_copy: false,
            compare_linked: true,
            compare_plane: PlaneKind::Axial,
            compare_mode: CompareMode::Pair,
            probe: None,
            compare_cursor: [0.5; 2],
            compare_split: 0.5,
            compare_menu: None,
            worker_trace: None,
            asset_query: TextEdit::default(),
            rename_asset: None,
            rename_draft: TextEdit::default(),
            confirm_remove: None,
            command_query: TextEdit::default(),
            shortcut_query: TextEdit::default(),
            shortcut_binding: TextEdit::default(),
            editing_shortcut: None,
            recording_shortcut: false,
            recipe_name: TextEdit::new("Reconstruction recipe"),
            parameter_edits: BTreeMap::new(),
            parameter_choices: BTreeMap::new(),
            parameter_toggles: BTreeMap::new(),
            corrections: super::corrections::CorrectionEditor::default(),
            operation: "reconstruct".into(),
            prepare_operation: String::new(),
            reconstruct_operation: String::new(),
            operation_menu: false,
            choice_menu: None,
            export_format: "nxs".into(),
            export_path: None,
            export_edits: BTreeMap::new(),
            selected_job: None,
            command_index: 0,
            collapsed_assets: BTreeSet::new(),
            drag_last: [0.; 2],
            inspect_tool: InspectTool::Navigate,
            display: DisplayState::default(),
        }
    }
}

pub struct DisplayState {
    pub indices: [usize; 3],
    pub projection_index: usize,
    pub sinogram_row: usize,
    pub colormap: String,
    pub invert: bool,
    pub zoom: f32,
    pub pan: [f32; 2],
    pub azimuth: f32,
    pub elevation: f32,
    /// Normalized clipping positions in physical x, y, z order.
    pub clip: [f32; 3],
    pub density: f32,
    /// Opacity cutoff within the display window, normalized to 0–1.
    pub threshold: f32,
    pub lighting: f32,
    pub crosshair: bool,
}
impl Default for DisplayState {
    fn default() -> Self {
        Self {
            indices: [0; 3],
            projection_index: 0,
            sinogram_row: 0,
            colormap: "bone".into(),
            invert: false,
            zoom: 1.0,
            pan: [0.0; 2],
            azimuth: 35.0,
            elevation: 25.0,
            clip: [1.0; 3],
            density: 1.0,
            threshold: 0.08,
            lighting: 0.7,
            crosshair: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct UiModel {
    pub project: Option<ProjectView>,
    pub recent_projects: Vec<RecentProject>,
    pub assets: Vec<AssetView>,
    pub active_asset: Option<String>,
    pub selected_assets: BTreeSet<String>,
    pub jobs: Vec<JobView>,
    pub recipes: Vec<RecipeView>,
    pub parameters: Vec<ParameterView>,
    pub validation: crate::app::validation::ValidationView,
    pub preview: Option<PreviewView>,
    pub backend: BackendView,
    pub planes: BTreeMap<PlaneKind, PlaneView>,
    pub preview_choices: Vec<(String, String)>,
    pub compare_left: Option<String>,
    pub compare_right: Option<String>,
    pub comparison_note: Option<String>,
    pub comparison_count: usize,
    pub comparison_index: usize,
    pub comparison_planes: [Option<PlaneView>; 2],
    pub commands: Vec<CommandView>,
    pub shortcuts: Vec<ShortcutView>,
    pub shortcut_error: Option<String>,
    pub status: String,
    pub error: Option<String>,
    pub can_undo: bool,
    pub can_redo: bool,
    pub render_quality: String,
    pub render_loading: bool,
    pub measurements: Vec<MeasurementView>,
}
impl UiModel {
    pub fn active(&self) -> Option<&AssetView> {
        self.assets
            .iter()
            .find(|asset| Some(&asset.id) == self.active_asset.as_ref())
    }
    pub fn selection_count(&self) -> usize {
        self.selected_assets
            .len()
            .max(usize::from(self.active_asset.is_some()))
    }
    pub fn active_jobs(&self) -> usize {
        self.jobs
            .iter()
            .filter(|job| {
                matches!(
                    job.status,
                    JobStatus::Queued | JobStatus::Running | JobStatus::Paused
                )
            })
            .count()
    }
}

#[derive(Clone, Debug, Default)]
pub struct ProjectView {
    pub name: String,
    pub path: String,
    pub dirty: bool,
    pub saving: bool,
    pub save_error: Option<String>,
}
#[derive(Clone, Debug, Default)]
pub struct RecentProject {
    pub name: String,
    pub path: String,
    pub available: bool,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AssetKind {
    #[default]
    Scan,
    Volume,
    Mask,
    Other,
}
impl AssetKind {
    pub fn title(self) -> &'static str {
        match self {
            Self::Scan => "Scan",
            Self::Volume => "Volume",
            Self::Mask => "Mask",
            Self::Other => "Result",
        }
    }
}
/// What Explore can export a result as.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Export {
    /// A scan or alignment with measured projections.
    #[default]
    Dataset,
    /// A result containing a spatial volume.
    Volume,
    /// An array-only result: a registered NPY bundle.
    Array,
    /// Raw frames, which are saved once corrected.
    Unavailable,
}
#[derive(Clone, Debug, Default)]
pub struct AssetView {
    pub export: Export,
    pub mask: bool,
    pub id: String,
    pub name: String,
    pub kind: AssetKind,
    pub parent: Option<String>,
    pub path: String,
    pub shape: Vec<usize>,
    pub voxel_pitch: Option<[f64; 3]>,
    pub units: String,
    pub missing: bool,
    pub status: String,
    pub description: String,
    pub method: Option<String>,
    pub simulated: bool,
    pub operation: Option<String>,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JobStatus {
    #[default]
    Queued,
    Running,
    Paused,
    Completed,
    Failed,
    Canceled,
    Interrupted,
}
impl JobStatus {
    pub fn title(self) -> &'static str {
        match self {
            Self::Queued => "Queued",
            Self::Running => "Running",
            Self::Paused => "Paused",
            Self::Completed => "Complete",
            Self::Failed => "Failed",
            Self::Canceled => "Canceled",
            Self::Interrupted => "Interrupted",
        }
    }
}
#[derive(Clone, Debug, Default)]
pub struct JobView {
    pub id: String,
    pub name: String,
    pub inputs: String,
    pub operation: String,
    pub status: JobStatus,
    pub stage: String,
    pub progress: Option<f32>,
    pub error: Option<String>,
    pub settings: Vec<(String, String)>,
    pub elapsed: Option<String>,
    pub result_asset: Option<String>,
    pub can_pause: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PreviewView {
    pub operation: String,
    pub message: String,
    pub phase: crate::previews::PreviewPhase,
    pub revision: Option<u64>,
    pub can_keep: bool,
    pub has_image: bool,
    pub index: usize,
    pub count: Option<usize>,
    pub center: Option<f64>,
    pub error_path: Option<String>,
}
#[derive(Clone, Debug, Default)]
pub struct RecipeView {
    pub id: String,
    pub name: String,
    pub operation: String,
    pub summary: String,
}
#[derive(Clone, Debug, Default)]
pub struct BackendView {
    pub ready: bool,
    pub device: String,
    pub status: String,
    pub capabilities: Vec<String>,
}

impl From<crate::protocol::Ready> for BackendView {
    fn from(ready: crate::protocol::Ready) -> Self {
        let (device, status) = match ready.peer {
            crate::protocol::Peer::Executor {
                devices, versions, ..
            } => (
                devices
                    .iter()
                    .map(|d| d.name.as_str())
                    .collect::<Vec<_>>()
                    .join(" · "),
                format!("TomoJAX {}", versions.tomojax),
            ),
            crate::protocol::Peer::Worker { host, .. } => (String::new(), host),
        };
        Self {
            ready: true,
            device,
            status,
            capabilities: ready.capabilities,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PlaneView {
    pub available: bool,
    pub pixel_shape: [usize; 2],
    pub spacing: [f64; 2],
    pub axes: [String; 2],
    pub units: String,
    pub index: usize,
    pub count: usize,
    pub quality: String,
    pub range: [f64; 2],
    pub histogram: Vec<u64>,
    pub histogram_edges: Vec<f64>,
    pub statistics: Option<Statistics>,
}
#[derive(Clone, Debug, Default)]
pub struct Statistics {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std: f64,
    pub sampled_pixels: usize,
}
#[derive(Clone, Debug, Default)]
pub struct MeasurementView {
    pub id: String,
    pub name: String,
    pub asset: String,
    pub index: usize,
    pub plane: PlaneKind,
    pub tool: InspectTool,
    /// Original-data UV coordinates, before the GPU display's zoom and pan.
    pub points: [[f32; 2]; 2],
    pub value: String,
    pub detail: String,
}
#[derive(Clone, Debug, Default)]
pub struct ParameterView {
    pub name: String,
    pub label: String,
    pub help: String,
    pub unit: String,
    pub kind: ParameterKind,
    pub advanced: bool,
    pub required: bool,
    pub choices: Vec<Choice>,
    /// Shown in an empty field: what the worker uses when the field is left blank.
    pub placeholder: String,
    /// The configuration value a blank field takes, when the schema states it (`x-default`;
    /// null when the configuration's own default is None).
    /// The form's own `config` overrides it; see [`crate::forms::default_label`].
    pub inherited: Option<serde_json::Value>,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ParameterKind {
    #[default]
    Number,
    Integer,
    Text,
    Boolean,
    Choice,
    Path,
}
#[derive(Clone, Debug, Default)]
pub struct Choice {
    pub value: String,
    pub label: String,
}
/// The Activity popover over the jobs panel. Each open state lists the jobs it has shown
/// or opened for; a failed one stays listed with its error until Activity closes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Activity {
    #[default]
    Closed,
    /// Opened by the user; it stays open until they close it.
    Shown { jobs: Vec<String> },
    /// Opened for queued work; it closes once that work has finished without a failure.
    Following { jobs: Vec<String> },
}

impl Activity {
    pub fn is_open(&self) -> bool {
        match self {
            Self::Closed => false,
            Self::Shown { .. } | Self::Following { .. } => true,
        }
    }

    /// Whether open Activity has shown, or opened for, the job `id`.
    pub fn tracks(&self, id: &str) -> bool {
        match self {
            Self::Closed => false,
            Self::Shown { jobs } | Self::Following { jobs } => jobs.iter().any(|job| job == id),
        }
    }

    /// Opens Activity as the user's choice, over the `active` jobs.
    pub fn show(&mut self, active: Vec<String>) {
        *self = Self::Shown { jobs: active };
    }

    /// Lists newly `queued` jobs, opening Activity for them when it is closed.
    pub fn follow(&mut self, queued: Vec<String>) {
        match self {
            Self::Closed => *self = Self::Following { jobs: queued },
            Self::Shown { jobs } | Self::Following { jobs } => jobs.extend(queued),
        }
    }

    pub fn close(&mut self) {
        *self = Self::Closed;
    }

    /// Lists every active job of `all`, and closes Activity opened for queued work once
    /// every job it lists has finished without a failure.
    pub fn observe(&mut self, all: &[crate::project::Job]) {
        use crate::project::JobStatus;
        let listed = match self {
            Self::Closed => return,
            Self::Shown { jobs } | Self::Following { jobs } => jobs,
        };
        for job in all.iter().filter(|job| job.status.active()) {
            if !listed.contains(&job.id) {
                listed.push(job.id.clone());
            }
        }
        let finished = all
            .iter()
            .filter(|job| listed.contains(&job.id))
            .all(|job| match job.status {
                JobStatus::Completed | JobStatus::Canceled | JobStatus::Interrupted => true,
                JobStatus::Queued | JobStatus::Running | JobStatus::Paused | JobStatus::Failed => {
                    false
                }
            });
        if finished && matches!(self, Self::Following { .. }) {
            self.close();
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CommandView {
    pub id: String,
    pub title: String,
    pub category: String,
    pub binding: String,
    pub enabled: bool,
}
#[derive(Clone, Debug, Default)]
pub struct ShortcutView {
    pub id: String,
    pub title: String,
    pub category: String,
    pub binding: String,
    pub default_binding: String,
}

#[derive(Clone, Debug)]
pub enum UiAction {
    Files(crate::app::files::Action),
    NewProject,
    OpenProject,
    OpenDemo,
    OpenRecent(String),
    ImportFiles,
    ImportFolder,
    SelectAsset {
        id: String,
        extend: bool,
    },
    SelectAllAssets,
    RelinkAsset(String),
    RenameResult,
    RemoveAsset(String),
    RemoveRecipe(String),
    SetWorkspace(Workspace),
    SetMainView(PlaneKind),
    SetPreviewArray(String),
    Simulate,
    SetOperation(String),
    QueueOperation(String),
    RefreshPreview,
    KeepPreview,
    CancelPreview,
    SetPreviewIndex(usize),
    SetPreviewCenter(f64),
    SaveRecipe,
    ApplyRecipe(String),
    PauseJob(String),
    ResumeJob(String),
    CancelJob(String),
    RetryJob(String),
    MoveJob {
        id: String,
        delta: i32,
    },
    Export(String),
    ShowPalette,
    RunCommand(String),
    ShowShortcuts,
    BeginShortcutEdit(String),
    SaveShortcut {
        command: String,
        binding: String,
    },
    ResetShortcut(String),
    ApplyWindow,
    AutoWindow,
    SetColormap(String),
    SetInvert(bool),
    SetSlice {
        axis: usize,
        index: usize,
    },
    StepSlice {
        axis: usize,
        delta: i32,
    },
    SetProjectionIndex(usize),
    SetSinogramRow(usize),
    SetCrosshair {
        plane: PlaneKind,
        position: [f32; 2],
    },
    Pan {
        delta: [f32; 2],
    },
    Zoom(f32),
    FitView,
    CameraDrag {
        delta: [f32; 2],
    },
    SetClip {
        axis: usize,
        value: f32,
    },
    SetDensity(f32),
    SetThreshold(f32),
    SetLighting(f32),
    SwapComparison,
    ComparePlane(PlaneKind),
    CompareMode(CompareMode),
    SetCompareSplit(f32),
    SetExportFormat(String),
    QueueExport,
    RemoveMeasurement(String),
    SetCompareLeft(String),
    SetCompareRight(String),
    CompareSelection,
    SetCompareLinked(bool),
    SetInspectTool(InspectTool),
    /// Pointer positions within the fitted unzoomed image extent; root inverts display zoom/pan.
    MeasureDrag {
        plane: PlaneKind,
        start: [f32; 2],
        end: [f32; 2],
        finished: bool,
    },
    CancelMeasurement,
    ClearMeasurements,
    CancelCopies,
    Undo,
    Redo,
    DismissError,
}
