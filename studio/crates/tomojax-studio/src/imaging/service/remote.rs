//! Per-pane fetch channels and a single off-UI decoder. Connection lifetimes own their requests.
use std::sync::Weak;
use studio_core::engine::{Engine, Lifecycle};
use studio_core::pixels::{Encoding, FrameQueue, PaneStream, Pixels, PlaneSelection};
use studio_core::protocol::{Response, ResultValue};

use super::*;

struct Flight {
    files: studio_core::engine::FileReads,
    request: Request,
    selection: PlaneSelection,
    encoding: Encoding,
}

impl Flight {
    fn decode(&self, frame: Pixels) -> Result<Asset, String> {
        let values = frame.decode(&self.selection, self.encoding)?;
        let (axis, index) = self
            .request
            .key
            .plane
            .ok_or("A plane selection is required")?;
        let data = reader::registered_plane(
            &self.files,
            &self.request.descriptor,
            axis,
            index,
            &self.selection,
            values,
            self.encoding == Encoding::Float32,
        )?;
        Ok(Asset::Plane(Arc::new(data)))
    }

    fn completion(&self, asset: Result<Asset, String>) -> Completion {
        Completion {
            request: self.request.clone(),
            asset,
            finished: self.encoding == Encoding::Float32,
            cache_hit: false,
        }
    }
}

pub(super) struct Remote {
    panes: HashMap<u32, PaneStream>,
    flights: HashMap<String, Arc<Flight>>,
    leases: Vec<Weak<Flight>>,
    queue: Arc<FrameQueue<(Arc<Flight>, Pixels)>>,
    decoder: Option<JoinHandle<()>>,
    completed: Option<mpsc::Receiver<Completion>>,
    send: mpsc::SyncSender<Completion>,
}

impl Remote {
    pub(super) fn new(shared: Arc<Shared>) -> Self {
        // One queued decode and one new terminal per slot, plus the decode in progress.
        let (send, completed) = mpsc::sync_channel(2 * MAX_SLOTS + 1);
        let queue = Arc::new(FrameQueue::<(Arc<Flight>, Pixels)>::new(MAX_SLOTS));
        let worker = queue.clone();
        let events = send.clone();
        let decoder = std::thread::spawn(move || {
            while let Some((flight, frame)) = worker.pop() {
                let result = flight.decode(frame);
                if events.send(flight.completion(result)).is_err() {
                    break;
                }
                drop(flight);
                let wake = shared.waker.locked().clone();
                wake();
            }
        });
        Self {
            panes: HashMap::new(),
            flights: HashMap::new(),
            leases: Vec::new(),
            queue,
            decoder: Some(decoder),
            completed: Some(completed),
            send,
        }
    }

    pub(super) fn dispatch(
        &mut self,
        engine: &mut Engine,
        slots: &mut HashMap<u32, Slot>,
        windows: &[Option<[f32; 2]>],
    ) -> bool {
        if engine.lifecycle() != &Lifecycle::Ready {
            return false;
        }
        self.leases.retain(|flight| flight.strong_count() > 0);
        let mut changed = false;
        for (&id, slot) in slots {
            let request = &slot.request;
            let Some((axis, index)) = request.key.plane else {
                continue;
            };
            if matches!(slot.status, ImagingStatus::Error(_) | ImagingStatus::Ready) {
                continue;
            }
            let pane = self.panes.entry(id).or_default();
            let Some(encoding) = pane.next(request.generation) else {
                continue;
            };
            let mut edge = request.crop.map_or(request.key.edge, |crop| crop.edge);
            if encoding == Encoding::Jpeg {
                edge = edge.min(512);
            }
            let flight = PlaneSelection::for_region(
                &request.descriptor.shape,
                axis,
                index,
                edge,
                request.crop.map(|crop| crop.bounds),
            )
            .and_then(|selection| {
                let window = windows.get(id as usize).copied().flatten();
                let command = selection.fetch(&request.descriptor, encoding, window);
                let rid = engine.request(command, Some(&format!("view.{id}")))?;
                Ok((
                    rid,
                    Arc::new(Flight {
                        files: engine.files(),
                        request: request.clone(),
                        selection,
                        encoding,
                    }),
                ))
            });
            match flight {
                Ok((rid, flight)) => {
                    *pane = PaneStream::Waiting(request.generation, encoding);
                    self.leases.push(Arc::downgrade(&flight));
                    self.flights.insert(rid, flight);
                }
                Err(error) => {
                    slot.status = ImagingStatus::Error(error);
                    if request.crop.is_none() {
                        slot.asset = None;
                    }
                    changed = true;
                }
            }
        }
        changed
    }

    pub(super) fn receive(&mut self, response: Response, stale: bool) -> Option<Response> {
        let Some(flight) = response.request().and_then(|id| self.flights.remove(id)) else {
            return Some(response);
        };
        let request = &flight.request;
        let pane = self.panes.get_mut(&request.slot)?;
        if !pane.receive(request.generation, flight.encoding) || stale {
            return None;
        }
        let error = match response {
            Response::Result {
                result: ResultValue::Pixels(frame),
                ..
            } => {
                self.queue.push(
                    (flight.encoding == Encoding::Jpeg).then_some(request.slot),
                    (flight, frame),
                );
                return None;
            }
            Response::Error { error, .. } => error.message,
            Response::ProtocolError { message, .. } => message,
            _ => "Expected a pixel result".into(),
        };
        let _ = self.send.send(flight.completion(Err(error)));
        None
    }

    pub(super) fn poll(&self) -> Option<Completion> {
        self.completed.as_ref()?.try_recv().ok()
    }

    pub(super) fn uses(&self, artifacts: &BTreeSet<String>) -> bool {
        self.leases.iter().filter_map(Weak::upgrade).any(|f| {
            f.request
                .descriptor
                .artifacts()
                .any(|a| artifacts.contains(a))
        })
    }

    pub(super) fn clear(&mut self, slot: u32) {
        self.panes.remove(&slot);
    }
    pub(super) fn invalidate(&mut self) {
        self.panes.clear();
    }
}

impl Drop for Remote {
    fn drop(&mut self) {
        self.completed.take();
        self.queue.close();
        if let Some(decoder) = self.decoder.take() {
            let _ = decoder.join();
        }
    }
}
