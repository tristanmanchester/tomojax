//! Reads every plane and pose table from one published preview generation before displaying it.
use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex, PoisonError, mpsc};
use std::thread::JoinHandle;

use crate::imaging::{PlaneData, read_pose_table, read_preview_plane};
use studio_core::engine::FileReads;
use studio_core::protocol::ArrayDescriptor;
use studio_core::sync::Locked;

pub use crate::imaging::TableData as PoseTable;

#[derive(Clone, Debug)]
pub struct SolverPreviewInfo {
    pub job: String,
    pub request: String,
    pub task: u64,
    pub time: f64,
    /// How far the task that emitted the preview had come, in its own unit.
    pub position: Option<SolverPreviewPosition>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SolverPreviewPosition {
    pub done: u64,
    pub total: Option<u64>,
    pub unit: String,
}

impl SolverPreviewPosition {
    /// "8 / 8 steps", or "3 steps" when the total is open.
    pub fn label(&self) -> String {
        match self.total {
            Some(total) => format!("{} / {total} {}", self.done, self.unit),
            None => format!("{} {}", self.done, self.unit),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LiveFrame {
    pub info: SolverPreviewInfo,
    /// Official preview keys: z (axial), y (coronal), x (sagittal).
    pub planes: BTreeMap<String, Arc<PlaneData>>,
    /// Full f64 values with the original column names, units and view registration.
    pub poses: Option<PoseTable>,
}

struct SolverPreview {
    files: Option<FileReads>,
    generation: u64,
    info: SolverPreviewInfo,
    arrays: BTreeMap<String, ArrayDescriptor>,
}

#[derive(Default)]
struct Inbox {
    pending: Option<Arc<SolverPreview>>,
    stopping: bool,
}
struct Shared {
    inbox: Mutex<Inbox>,
    changed: Condvar,
    wake: Mutex<rune::Waker>,
}
struct Completion {
    preview: Arc<SolverPreview>,
    result: Result<LiveFrame, String>,
}
type Reader = Arc<dyn Fn(&SolverPreview) -> Result<LiveFrame, String> + Send + Sync>;

pub struct LivePreviews {
    files: Option<FileReads>,
    enabled: bool,
    generation: u64,
    desired: Option<Arc<SolverPreview>>,
    displayed: Option<u64>,
    publication: Option<(String, String)>,
    frame: Option<LiveFrame>,
    error: Option<String>,
    shared: Arc<Shared>,
    events: Option<mpsc::Receiver<Completion>>,
    thread: Option<JoinHandle<()>>,
}

impl Default for LivePreviews {
    fn default() -> Self {
        Self::new()
    }
}
impl LivePreviews {
    pub fn new() -> Self {
        Self::with_reader(Arc::new(read))
    }

    fn with_reader(read: Reader) -> Self {
        let shared = Arc::new(Shared {
            inbox: Mutex::new(Inbox::default()),
            changed: Condvar::new(),
            wake: Mutex::new(Arc::new(|| {})),
        });
        let (send, events) = mpsc::sync_channel(2);
        let thread_shared = shared.clone();
        let thread = std::thread::spawn(move || {
            loop {
                let preview = {
                    let inbox = thread_shared.inbox.locked();
                    let mut inbox = thread_shared
                        .changed
                        .wait_while(inbox, |inbox| inbox.pending.is_none() && !inbox.stopping)
                        .unwrap_or_else(PoisonError::into_inner);
                    if inbox.stopping {
                        break;
                    }
                    let Some(preview) = inbox.pending.take() else {
                        continue;
                    };
                    preview
                };
                // The result never contains file-backed or borrowed scalar storage.
                let result = read(&preview);
                if send.send(Completion { preview, result }).is_err() {
                    break;
                }
                let wake = thread_shared.wake.locked().clone();
                wake();
            }
        });
        Self {
            files: None,
            enabled: true,
            generation: 0,
            desired: None,
            displayed: None,
            publication: None,
            frame: None,
            error: None,
            shared,
            events: Some(events),
            thread: Some(thread),
        }
    }

    pub fn bind(&mut self, files: FileReads) {
        self.files = Some(files);
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.clear();
        }
    }
    pub fn set_waker(&mut self, wake: rune::Waker) {
        *self.shared.wake.locked() = wake;
    }
    pub fn frame(&self) -> Option<&LiveFrame> {
        self.frame.as_ref()
    }
    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }
    pub fn loading(&self) -> bool {
        self.desired
            .as_ref()
            .is_some_and(|s| Some(s.generation) != self.displayed)
    }
    pub fn pending_info(&self) -> Option<&SolverPreviewInfo> {
        self.desired.as_ref().map(|s| &s.info)
    }

    pub fn accept_published(
        &mut self,
        job: &studio_core::project::Job,
        preview: &serde_json::Value,
    ) {
        let (Some(generation), Some(request)) =
            (preview["generation"].as_str(), preview["request"].as_str())
        else {
            return;
        };
        if !self.enabled
            || self
                .publication
                .as_ref()
                .is_some_and(|(id, g)| id == &job.id && g == generation)
        {
            return;
        }
        let Ok(studio_core::protocol::Progress::Preview {
            arrays, task, time, ..
        }) = serde_json::from_value(preview["record"].clone())
        else {
            return;
        };
        self.publication = Some((job.id.clone(), generation.into()));
        let position = job
            .telemetry
            .task(request, task)
            .map(|t| SolverPreviewPosition {
                done: t.done,
                total: t.total,
                unit: t.unit.clone(),
            });
        self.accept(&job.id, request, task, time, position, arrays);
    }

    pub fn accept(
        &mut self,
        job: &str,
        request: &str,
        task: u64,
        time: f64,
        position: Option<SolverPreviewPosition>,
        arrays: BTreeMap<String, ArrayDescriptor>,
    ) {
        self.generation += 1;
        let generation = self.generation;
        if !self.enabled || self.thread.is_none() {
            return;
        }
        let preview = Arc::new(SolverPreview {
            files: self.files.clone(),
            generation,
            info: SolverPreviewInfo {
                job: job.into(),
                request: request.into(),
                task,
                time,
                position,
            },
            arrays,
        });
        self.desired = Some(preview.clone());
        self.error = None;
        let mut inbox = self.shared.inbox.locked();
        inbox.pending = Some(preview);
        drop(inbox);
        self.shared.changed.notify_one();
    }

    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        while let Some(completion) = self
            .events
            .as_ref()
            .and_then(|events| events.try_recv().ok())
        {
            if Some(completion.preview.generation) != self.desired.as_ref().map(|p| p.generation) {
                continue;
            }
            match completion.result {
                Ok(frame) => {
                    self.displayed = Some(completion.preview.generation);
                    self.frame = Some(frame);
                    self.error = None;
                }
                Err(error) => {
                    self.error = Some(error);
                    self.desired = None;
                }
            }
            changed = true;
        }
        changed
    }

    pub fn clear(&mut self) {
        self.publication = None;
        self.desired = None;
        self.displayed = None;
        self.frame = None;
        self.error = None;
        let mut inbox = self.shared.inbox.locked();
        inbox.pending = None;
    }

    pub fn disconnected(&mut self) {
        self.clear();
        self.files = None;
    }
    pub fn shutdown(&mut self) {
        self.clear();
        self.shared.inbox.locked().stopping = true;
        self.shared.changed.notify_all();
        self.events.take(); // Unblock a sender before joining it.
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
impl Drop for LivePreviews {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn read(preview: &SolverPreview) -> Result<LiveFrame, String> {
    let files = preview
        .files
        .as_ref()
        .ok_or("The file worker is disconnected")?;
    let mut planes = BTreeMap::new();
    for key in ["z", "y", "x"] {
        if let Some(array) = preview.arrays.get(key) {
            if array.shape.len() != 2 {
                return Err(format!("The live {key} preview is not a registered plane."));
            }
            planes.insert(
                key.into(),
                Arc::new(read_preview_plane(files, array, 0, 0, 1024)?),
            );
        }
    }
    let poses = preview
        .arrays
        .get("poses")
        .map(|desc| read_pose_table(files, desc))
        .transpose()?;
    if planes.is_empty() && poses.is_none() {
        return Err(
            "The worker preview contains no supported registered slices or pose table.".into(),
        );
    }
    Ok(LiveFrame {
        info: preview.info.clone(),
        planes,
        poses,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(done: u64) -> Option<SolverPreviewPosition> {
        Some(SolverPreviewPosition {
            done,
            total: None,
            unit: "steps".into(),
        })
    }
    use npyz::WriterBuilder;
    use std::time::{Duration, Instant};
    use studio_core::protocol::{Axis, Coordinates, Storage};

    fn npy(path: &std::path::Path, shape: &[u64], values: &[f64]) {
        let mut writer = npyz::WriteOptions::new()
            .default_dtype()
            .shape(shape)
            .writer(std::fs::File::create(path).unwrap())
            .begin_nd()
            .unwrap();
        for value in values {
            writer.push(value).unwrap();
        }
        writer.finish().unwrap();
    }

    fn descriptor(token: &str, coordinate: &str) -> ArrayDescriptor {
        ArrayDescriptor {
            storage: Storage::Npy {
                path: format!("{token}.npy"),
                artifact: Some(token.into()),
            },
            shape: vec![2, 2],
            dtype: "<f4".into(),
            axes: vec![
                Axis {
                    name: "y".into(),
                    frame: "world".into(),
                    unit: "mm".into(),
                    coordinates: Coordinates::Coordinates {
                        values: Storage::Npy {
                            path: "coordinates.npy".into(),
                            artifact: Some(coordinate.into()),
                        },
                    },
                },
                Axis {
                    name: "x".into(),
                    frame: "world".into(),
                    unit: "mm".into(),
                    coordinates: Coordinates::Uniform {
                        origin: 0.,
                        spacing: 1.,
                    },
                },
            ],
            unit: "mm^-1".into(),
            domain: "attenuation".into(),
            fixed: vec![],
            columns: vec![],
        }
    }
    fn wait_for(live: &mut LivePreviews, step: u64) {
        let deadline = Instant::now() + Duration::from_secs(3);
        while live
            .frame()
            .is_none_or(|frame| frame.info.position.as_ref().map(|p| p.done) != Some(step))
        {
            live.poll();
            assert!(Instant::now() < deadline, "preview failed to finish");
            std::thread::yield_now();
        }
    }

    #[test]
    fn an_expired_generation_keeps_the_previous_complete_frame() {
        let mut live = LivePreviews::with_reader(Arc::new(|p| {
            if p.info.time == 2. {
                return Err("Preview generation expired".into());
            }
            Ok(LiveFrame {
                info: p.info.clone(),
                planes: Default::default(),
                poses: None,
            })
        }));
        live.accept("job", "run", 1, 1., step(1), Default::default());
        wait_for(&mut live, 1);
        live.accept("job", "run", 1, 2., step(2), Default::default());
        let until = Instant::now() + Duration::from_secs(2);
        while live.loading() {
            live.poll();
            assert!(Instant::now() < until);
        }
        assert_eq!(live.frame().unwrap().info.time, 1.);
        live.accept("job", "run", 1, 3., step(3), Default::default());
        wait_for(&mut live, 3);
    }

    #[test]
    fn registered_live_plane_and_pose_table_keep_axes_columns_and_f64_values() {
        let directory = tempfile::tempdir().unwrap();
        let image_path = directory.path().join("plane.npy");
        let pose_path = directory.path().join("poses.npy");
        let coordinates_path = directory.path().join("views.npy");
        npy(&image_path, &[2, 2], &[1., 2., 3., 4.]);
        let precise = 0.123456789012345;
        npy(
            &pose_path,
            &[2, 6],
            &[precise, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        );
        npy(&coordinates_path, &[2], &[23., 7.]);
        let mut image = descriptor("image", "unused");
        image.storage = Storage::Npy {
            path: image_path.to_string_lossy().into(),
            artifact: None,
        };
        image.dtype = "<f8".into();
        image.axes[0].coordinates = Coordinates::Uniform {
            origin: 4.,
            spacing: 2.,
        };
        image.axes[1].coordinates = Coordinates::Uniform {
            origin: -2.,
            spacing: 0.5,
        };
        image.fixed.push(studio_core::protocol::Fixed {
            name: "z".into(),
            frame: "world".into(),
            unit: "mm".into(),
            index: 17,
            coordinate: 51.,
        });
        let mut poses = descriptor("poses", "view-coordinates");
        poses.storage = Storage::Npy {
            path: pose_path.to_string_lossy().into(),
            artifact: None,
        };
        poses.dtype = "<f8".into();
        poses.shape = vec![2, 6];
        poses.unit = "mixed".into();
        poses.domain = "poses".into();
        poses.axes[0] = Axis {
            name: "view".into(),
            frame: "acquisition".into(),
            unit: "degree".into(),
            coordinates: Coordinates::Coordinates {
                values: Storage::Npy {
                    path: coordinates_path.to_string_lossy().into(),
                    artifact: None,
                },
            },
        };
        poses.axes[1] = Axis {
            name: "column".into(),
            frame: "pose".into(),
            unit: "index".into(),
            coordinates: Coordinates::Uniform {
                origin: 0.,
                spacing: 1.,
            },
        };
        poses.columns = ["alpha", "beta", "phi", "x", "z", "y"]
            .into_iter()
            .enumerate()
            .map(|(i, name)| studio_core::protocol::Column {
                name: name.into(),
                unit: if i < 3 { "degree" } else { "mm" }.into(),
            })
            .collect();
        let worker =
            studio_core::engine::Engine::spawn_through("/bin/false", &[], Arc::new(|| {})).unwrap();
        let mut live = LivePreviews::new();
        live.bind(worker.files());
        live.accept(
            "alignment",
            "run",
            2,
            4.,
            step(3),
            BTreeMap::from([("z".into(), image), ("poses".into(), poses)]),
        );
        wait_for(&mut live, 3);
        let frame = live.frame().unwrap().clone();
        assert_eq!(frame.info.task, 2);
        assert_eq!(frame.planes["z"].axes[0].coordinate(1.), 6.);
        assert_eq!(frame.planes["z"].axes[1].coordinate(1.), -1.5);
        assert_eq!(frame.planes["z"].fixed[0].index, 17);
        assert_eq!(frame.planes["z"].fixed[0].coordinate, 51.);
        let table = frame.poses.as_ref().unwrap();
        assert_eq!(table.row(0).unwrap()[0], precise);
        assert_eq!(table.row(1).unwrap(), [7., 8., 9., 10., 11., 12.]);
        assert_eq!(table.axes[0].coordinate(1.), 7.);
        assert_eq!(
            table
                .columns
                .iter()
                .map(|c| c.name.as_str())
                .collect::<Vec<_>>(),
            ["alpha", "beta", "phi", "x", "z", "y"]
        );
        assert_eq!(table.columns[0].unit, "degree");
        assert_eq!(table.columns[3].unit, "mm");
        live.clear();
        std::fs::remove_file(image_path).unwrap();
        std::fs::remove_file(pose_path).unwrap();
        std::fs::remove_file(coordinates_path).unwrap();
        // Display buffers remain independent from the retired worker files.
        assert_eq!(frame.planes["z"].values.as_ref(), [1., 2., 3., 4.]);
        assert_eq!(frame.poses.unwrap().row(0).unwrap()[0], precise);
    }
}
