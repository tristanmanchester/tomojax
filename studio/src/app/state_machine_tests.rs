//! Random sequences of user and worker actions against the real application state,
//! checked after every step against a small reference model of what must hold.
use super::window::Reading;
use super::*;
use proptest::prelude::*;
use std::collections::BTreeSet;

/// The value scale of each fixture volume's two files: relinking switches between them.
const SCALES: [[f32; 2]; 3] = [[1., 2.5], [7., 0.5], [0.3, 4.]];
const DEPTH: usize = 3;
const SIDE: usize = 4;

/// Texts typed into a window field. None is a reading Studio writes (each starts with a
/// space or is not a number), so typing one always leaves an edited draft.
const DRAFTS: [&str; 7] = [" 0.5", " 2", " 40", " -3", "-", "abc", "nan"];

#[derive(Clone, Debug)]
enum Step {
    Select(usize),
    Workspace(Workspace),
    View(PlaneKind),
    Scroll(i32),
    EditWindow(usize, &'static str),
    ApplyWindow,
    AutoWindow,
    SubmitJob,
    JobFailed(usize),
    JobFinished(usize),
    ToggleActivity,
    CloseActivity,
    RefreshViewer,
    Relink(usize),
    Reopen,
}

fn step() -> impl Strategy<Value = Step> {
    let workspace = prop_oneof![
        Just(Workspace::Prepare),
        Just(Workspace::Reconstruct),
        Just(Workspace::Explore),
        Just(Workspace::Jobs),
    ];
    let view = prop_oneof![
        Just(PlaneKind::Axial),
        Just(PlaneKind::Coronal),
        Just(PlaneKind::Sagittal),
    ];
    prop_oneof![
        1 => (0..3usize).prop_map(Step::Select),
        1 => workspace.prop_map(Step::Workspace),
        1 => view.prop_map(Step::View),
        1 => prop_oneof![Just(-1), Just(1)].prop_map(Step::Scroll),
        3 => (0..2usize, proptest::sample::select(DRAFTS.to_vec()))
            .prop_map(|(side, text)| Step::EditWindow(side, text)),
        2 => Just(Step::ApplyWindow),
        1 => Just(Step::AutoWindow),
        2 => Just(Step::SubmitJob),
        2 => (0..4usize).prop_map(Step::JobFailed),
        1 => (0..4usize).prop_map(Step::JobFinished),
        1 => Just(Step::ToggleActivity),
        1 => Just(Step::CloseActivity),
        1 => Just(Step::RefreshViewer),
        1 => (0..3usize).prop_map(Step::Relink),
        1 => Just(Step::Reopen),
    ]
}

/// The value at voxel (z, y, x) of a volume with `scale`: every plane varies, so every
/// automatic window is a proper range.
fn voxel(scale: f32, z: usize, y: usize, x: usize) -> f32 {
    scale * (1 + x + SIDE * y + SIDE * SIDE * z) as f32
}

fn write_volume(path: &Path, scale: f32) {
    use std::io::Write;
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({DEPTH}, {SIDE}, {SIDE}), }}");
    while !(10 + header.len() + 1).is_multiple_of(64) {
        header.push(' ');
    }
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend((header.len() as u16).to_le_bytes());
    bytes.extend(header.as_bytes());
    for z in 0..DEPTH {
        for y in 0..SIDE {
            for x in 0..SIDE {
                bytes.extend(voxel(scale, z, y, x).to_le_bytes());
            }
        }
    }
    std::fs::File::create(path)
        .and_then(|mut file| file.write_all(&bytes))
        .unwrap();
}

/// The largest value of the plane `plane` shows at `index`, read from the formula.
fn plane_max(scale: f32, plane: PlaneKind, index: usize) -> f32 {
    let mut max = f32::MIN;
    for z in 0..DEPTH {
        for y in 0..SIDE {
            for x in 0..SIDE {
                let fixed = match plane {
                    PlaneKind::Axial => z,
                    PlaneKind::Coronal => y,
                    PlaneKind::Sagittal => x,
                    _ => unreachable!("the fixture views only orthogonal planes"),
                };
                if fixed == index {
                    max = max.max(voxel(scale, z, y, x));
                }
            }
        }
    }
    max
}

struct Fixture {
    _directory: tempfile::TempDir,
    root: PathBuf,
    ids: Vec<String>,
    /// Each asset's two files.
    files: Vec<[PathBuf; 2]>,
}

impl Fixture {
    fn new() -> (Self, AppState) {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("project");
        let mut state = open(&root, true);
        let mut ids = Vec::new();
        let mut files = Vec::new();
        for (index, scales) in SCALES.iter().enumerate() {
            let paths = [0, 1].map(|file| directory.path().join(format!("v{index}-{file}.npy")));
            for (path, scale) in paths.iter().zip(scales) {
                write_volume(path, *scale);
            }
            let axis = |name: &str| json!({"name":name,"frame":"sample","unit":"mm","coordinates":{"type":"uniform","origin":0.,"spacing":1.}});
            let descriptor = serde_json::from_value(json!({
                "shape":[DEPTH,SIDE,SIDE], "dtype":"<f4", "unit":"1/mm", "domain":"attenuation",
                "storage":{"type":"npy","path":paths[0]},
                "axes":[axis("z"), axis("y"), axis("x")],"fixed":[],"columns":[]
            }))
            .unwrap();
            let mut asset = Asset::input(paths[0].clone(), false);
            asset.kind = "reconstruction".into();
            asset.provenance = Some(crate::protocol::Provenance {
                operation: "reconstruct".into(),
                parameters: json!({}),
                inputs: vec![],
            });
            asset.arrays.insert("volume".into(), descriptor);
            ids.push(asset.id.clone());
            files.push(paths);
            state.project.as_mut().unwrap().document.assets.push(asset);
        }
        state.dispatch(
            UiAction::SelectAsset {
                id: ids[0].clone(),
                extend: false,
            },
            &mut rune::Tx::default(),
        );
        settle(&mut state);
        let fixture = Self {
            _directory: directory,
            root,
            ids,
            files,
        };
        (fixture, state)
    }

    /// Which of its two files asset `index` reads now.
    fn linked(&self, state: &AppState, index: usize) -> usize {
        let project = state.project.as_ref().unwrap();
        let path = project.resolve(&project.document.asset(&self.ids[index]).unwrap().path);
        usize::from(path == self.files[index][1])
    }
}

fn open(root: &Path, create: bool) -> AppState {
    let mut state = AppState::init();
    state.demo = Demo::Off;
    state.open_project(root, create).unwrap();
    // Jobs can be queued; with no worker attached they wait in the queue.
    state.model.backend.ready = true;
    state
}

/// Runs frames until the native reads and file reads Studio started have landed.
fn settle(state: &mut AppState) {
    let limit = Instant::now() + Duration::from_secs(20);
    loop {
        state.poll();
        let loading = (1..=8).any(|slot| {
            matches!(
                state.imaging.status(slot),
                crate::imaging::ImagingStatus::Loading | crate::imaging::ImagingStatus::Refining
            )
        });
        if !loading && state.metadata_pending == 0 {
            return;
        }
        assert!(Instant::now() < limit, "the viewer did not settle");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn dispatch(state: &mut AppState, action: UiAction) {
    state.dispatch(action, &mut rune::Tx::default());
}

/// The failed jobs open Activity lists, as the jobs drawer shows them.
fn shown_failures(state: &AppState) -> BTreeSet<String> {
    state
        .model
        .jobs
        .iter()
        .filter(|job| job.status == JobStatus::Failed && state.ui.activity.tracks(&job.id))
        .map(|job| job.id.clone())
        .collect()
}

/// The jobs open Activity lists, whatever their state.
fn listed(state: &AppState) -> Vec<String> {
    state
        .model
        .jobs
        .iter()
        .filter(|job| state.ui.activity.tracks(&job.id))
        .map(|job| job.id.clone())
        .collect()
}

fn active_jobs(state: &AppState) -> Vec<String> {
    state
        .project
        .iter()
        .flat_map(|project| &project.document.jobs)
        .filter(|job| job.status.active())
        .map(|job| job.id.clone())
        .collect()
}

/// What the user set and what reopening must bring back.
#[derive(Debug, PartialEq)]
struct Kept {
    active: Option<String>,
    view: PlaneKind,
    indices: [usize; 3],
    applied: Option<[f32; 2]>,
    texts: [String; 2],
    readings: [String; 2],
}

fn kept(state: &AppState) -> Kept {
    Kept {
        active: state.model.active_asset.clone(),
        view: state.ui.main_view,
        indices: state.ui.display.indices,
        applied: state.window.applied(),
        texts: [0, 1].map(|side| state.window.field(side).text.clone()),
        readings: [0, 1].map(|side| format!("{:?}", state.window.reading(side))),
    }
}

/// What the window fields should hold: the committed window and each field's draft.
#[derive(Clone, Debug, Default)]
struct WindowModel {
    applied: Option<[f32; 2]>,
    drafts: [Option<&'static str>; 2],
}

impl WindowModel {
    /// The window [`UiAction::ApplyWindow`] commits, given the window on screen.
    fn parse(&self, shown: [f32; 2]) -> Option<[f32; 2]> {
        let limit = |side: usize| match self.drafts[side] {
            None => Some(shown[side]),
            Some(text) => text.trim().parse::<f32>().ok(),
        };
        let window = [limit(0)?, limit(1)?];
        (window.iter().all(|value| value.is_finite()) && window[0] < window[1]).then_some(window)
    }
}

/// The automatic window of the plane on screen.
fn auto_window(state: &AppState) -> [f32; 2] {
    state
        .imaging
        .plane(state.ui.main_view.mark())
        .expect("the active volume's plane is on screen")
        .statistics
        .window
}

fn run(steps: Vec<Step>) {
    let (fixture, mut state) = Fixture::new();
    let mut window = WindowModel::default();
    for step in steps {
        let listed_before = listed(&state);
        let mut dismissed = false;
        match step.clone() {
            Step::Select(index) => {
                if state.model.active_asset.as_ref() != Some(&fixture.ids[index]) {
                    window = WindowModel::default();
                }
                dispatch(
                    &mut state,
                    UiAction::SelectAsset {
                        id: fixture.ids[index].clone(),
                        extend: false,
                    },
                );
            }
            Step::Workspace(workspace) => dispatch(&mut state, UiAction::SetWorkspace(workspace)),
            Step::View(plane) => dispatch(&mut state, UiAction::SetMainView(plane)),
            Step::Scroll(delta) => {
                let axis = state.ui.main_view.axis().unwrap_or(0);
                dispatch(&mut state, UiAction::StepSlice { axis, delta });
            }
            Step::EditWindow(side, text) => {
                // What the field's editor does with a keystroke.
                state.window.field_mut(side).replace_text(text);
                window.drafts[side] = Some(text);
            }
            Step::ApplyWindow => {
                let shown = window.applied.unwrap_or_else(|| auto_window(&state));
                state.model.error = None;
                dispatch(&mut state, UiAction::ApplyWindow);
                match window.parse(shown) {
                    Some(applied) => {
                        window = WindowModel {
                            applied: Some(applied),
                            drafts: [None; 2],
                        };
                    }
                    None => assert!(state.model.error.is_some(), "an invalid draft is reported"),
                }
            }
            Step::AutoWindow => {
                window = WindowModel::default();
                dispatch(&mut state, UiAction::AutoWindow);
            }
            Step::SubmitJob => {
                dispatch(&mut state, UiAction::QueueOperation("simulate".into()));
            }
            Step::JobFailed(pick) | Step::JobFinished(pick) => {
                let active = active_jobs(&state);
                if !active.is_empty() {
                    let id = &active[pick % active.len()];
                    if matches!(step, Step::JobFailed(_)) {
                        state.fail_job(id, "The device ran out of memory", false);
                    } else {
                        state.complete_job(id, None);
                    }
                    state.refresh_model();
                }
            }
            Step::ToggleActivity => {
                // The top bar's Activity button.
                dismissed = state.ui.activity.is_open();
                if dismissed {
                    state.ui.activity.close();
                } else {
                    state.show_activity();
                }
            }
            Step::CloseActivity => {
                dismissed = true;
                state.ui.activity.close();
            }
            Step::RefreshViewer => state.refresh_images(),
            Step::Relink(index) => {
                let other = 1 - fixture.linked(&state, index);
                let path = fixture.files[index][other].clone();
                state.relink(&fixture.ids[index], &path);
            }
            Step::Reopen => {
                let before = kept(&state);
                state.shutdown();
                drop(state);
                state = open(&fixture.root, false);
                settle(&mut state);
                assert_eq!(kept(&state), before, "reopening keeps the view and drafts");
                dismissed = true;
            }
        }
        settle(&mut state);

        // A job Activity lists stays listed with its failure, until the user dismisses
        // Activity: neither a failure nor other work finishing closes it.
        if !dismissed {
            let shown = shown_failures(&state);
            let failed: Vec<_> = state
                .model
                .jobs
                .iter()
                .filter(|job| job.status == JobStatus::Failed && listed_before.contains(&job.id))
                .map(|job| &job.id)
                .collect();
            assert!(
                failed.iter().all(|id| shown.contains(*id)),
                "{step:?} dropped failures {failed:?} from Activity, which shows {shown:?}"
            );
        }

        // The committed window changes only through Apply, Auto or a new image.
        assert_eq!(state.window.applied(), window.applied, "after {step:?}");
        let shown = window.applied.unwrap_or_else(|| auto_window(&state));
        for (side, (limit, draft)) in shown.into_iter().zip(window.drafts).enumerate() {
            let field = &state.window.field(side).text;
            match draft {
                // A draft stays as typed, whether or not it is valid.
                Some(draft) => {
                    assert_eq!(field, draft, "after {step:?}");
                    assert_eq!(state.window.reading(side), Reading::Draft(draft));
                }
                // An untouched field stands for the exact limit on screen and reads as it.
                None => {
                    assert_eq!(
                        state.window.reading(side),
                        Reading::Limit(limit),
                        "field {side} after {step:?}"
                    );
                    let read: f32 = field.parse().unwrap();
                    assert!(
                        (read - limit).abs() <= 1e-3 * limit.abs().max(1e-2),
                        "field {side} reads {field} for {limit} after {step:?}"
                    );
                }
            }
        }

        // The viewer and the form read each asset's current file.
        let active = fixture
            .ids
            .iter()
            .position(|id| state.model.active_asset.as_ref() == Some(id))
            .expect("an asset is active");
        let linked = fixture.linked(&state, active);
        let plane = state.imaging.plane(state.ui.main_view.mark()).unwrap();
        assert_eq!(
            plane.statistics.max,
            plane_max(SCALES[active][linked], state.ui.main_view, plane.index),
            "the viewer shows asset {active}'s file {linked} after {step:?}"
        );
        let input = state
            .current_form_input()
            .expect("the active asset is an input");
        assert_eq!(input.path, fixture.files[active][linked], "after {step:?}");
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn studio_state_holds_its_invariants_through_any_action_sequence(
        steps in proptest::collection::vec(step(), 1..24)
    ) {
        run(steps);
    }
}
