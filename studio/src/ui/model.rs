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
pub enum CompareMode { #[default] Pair, Split, Difference }

pub struct UiState {
    pub workspace: Workspace,
    pub main_view: PlaneKind,
    pub preview_array: String,
    pub preview_span: TextEdit,
    pub tree_open: bool,
    pub inspector_open: bool,
    pub jobs_open: bool,
    pub palette_open: bool,
    pub shortcuts_open: bool,
    pub advanced_open: bool,
    pub import_copy: bool,
    pub compare_linked: bool,
    pub compare_plane: PlaneKind,
    pub compare_mode: CompareMode,
    pub compare_cursor: [f32;2],
    pub worker_trace: Option<String>,
    pub asset_query: TextEdit,
    pub rename: Option<(String, TextEdit)>,
    pub command_query: TextEdit,
    pub shortcut_query: TextEdit,
    pub shortcut_binding: TextEdit,
    pub editing_shortcut: Option<String>,
    pub recipe_name: TextEdit,
    /// Keys are the schema's parameter names. Root initialises every visible edit.
    pub parameter_edits: BTreeMap<String, TextEdit>,
    pub parameter_choices: BTreeMap<String, String>,
    pub parameter_toggles: BTreeMap<String, bool>,
    pub corrections: super::corrections::CorrectionEditor,
    pub operation: String,
    pub operation_menu: bool,
    pub choice_menu: Option<String>,
    pub export_format: String,
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
            jobs_open: false,
            palette_open: false,
            shortcuts_open: false,
            advanced_open: false,
            import_copy: false,
            compare_linked: true,
            compare_plane: PlaneKind::Axial,
            compare_mode: CompareMode::Pair,
            compare_cursor: [0.5;2],
            worker_trace: None,
            asset_query: TextEdit::default(),
            rename: None,
            command_query: TextEdit::default(),
            shortcut_query: TextEdit::default(),
            shortcut_binding: TextEdit::default(),
            editing_shortcut: None,
            recipe_name: TextEdit::new("Reconstruction recipe"),
            parameter_edits: BTreeMap::new(),
            parameter_choices: BTreeMap::new(),
            parameter_toggles: BTreeMap::new(),
            corrections: super::corrections::CorrectionEditor::default(),
            operation: "reconstruct".into(),
            operation_menu: false,
            choice_menu: None,
            export_format: "nxs".into(),
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
    pub window_low: TextEdit,
    pub window_high: TextEdit,
    pub colormap: String,
    pub invert: bool,
    pub zoom: f32,
    pub pan: [f32; 2],
    pub azimuth: f32,
    pub elevation: f32,
    /// Normalised clipping positions in physical x, y, z order.
    pub clip: [f32; 3],
    pub density: f32,
    /// Opacity cutoff within the display window, normalised to 0–1.
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
            window_low: TextEdit::default(),
            window_high: TextEdit::default(),
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
#[derive(Clone, Debug, Default)]
pub struct AssetView {
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
    Cancelled,
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
            Self::Cancelled => "Cancelled",
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
    pub centre: Option<f64>,
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
    NewProject,
    OpenProject,
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
    SetPreviewCentre(f64),
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
    SetExportFormat(String),
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
    Undo,
    Redo,
    DismissError,
}
