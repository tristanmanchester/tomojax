//! Registered view slots shared by the native readers and the worker pixel source.
mod remote;
use remote::Remote;
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex, PoisonError, mpsc};
use std::thread::JoinHandle;
use std::time::Instant;
use studio_core::{engine::Engine, protocol::Response};

use super::{PlaneData, VolumeData, reader};
use studio_core::protocol::ArrayDescriptor;
use studio_core::sync::Locked;

const MAX_SLOTS: usize = 32;

#[derive(Clone, Debug, Default, PartialEq)]
pub enum ImagingStatus {
    #[default]
    Idle,
    Loading,
    Refining,
    Ready,
    Error(String),
}

#[derive(Clone, Debug)]
pub struct LatencySample {
    pub slot: u32,
    pub milliseconds: f64,
    pub cache_hit: bool,
    pub refined: bool,
    pub bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Key {
    epoch: u64,
    source: Arc<str>,
    plane: Option<(usize, usize)>,
    edge: usize,
}

#[derive(Clone)]
enum Asset {
    Plane(Arc<PlaneData>),
    Volume(Arc<VolumeData>),
}

impl Asset {
    fn bytes(&self) -> usize {
        match self {
            Self::Plane(p) => p.values.len() * 4,
            Self::Volume(v) => v.texels.len() * 2,
        }
    }
}

struct Cache {
    epoch: u64,
    entries: HashMap<Key, Asset>,
    order: VecDeque<Key>,
    bytes: usize,
    budget: usize,
}

impl Cache {
    fn get(&mut self, key: &Key) -> Option<Asset> {
        let asset = self.entries.get(key)?.clone();
        self.order.retain(|k| k != key);
        self.order.push_back(key.clone());
        Some(asset)
    }

    fn insert(&mut self, key: Key, asset: Asset) {
        if key.epoch != self.epoch || asset.bytes() > self.budget {
            return;
        }
        if let Some(old) = self.entries.remove(&key) {
            self.bytes -= old.bytes();
        }
        self.order.retain(|k| k != &key);
        while self.bytes + asset.bytes() > self.budget || self.entries.len() >= 256 {
            let Some(old) = self.order.pop_front() else {
                break;
            };
            if let Some(asset) = self.entries.remove(&old) {
                self.bytes -= asset.bytes();
            }
        }
        self.bytes += asset.bytes();
        self.order.push_back(key.clone());
        self.entries.insert(key, asset);
    }
}

#[derive(Clone)]
struct Request {
    slot: u32,
    generation: u64,
    descriptor: Arc<ArrayDescriptor>,
    key: Key,
    submitted: Instant,
    lineage: u64,
}

#[derive(Default)]
struct Inbox {
    pending: HashMap<u32, Request>,
    /// Installed under the same lock that removes a pending request. A retired
    /// artifact must never appear idle between queueing and its first file open.
    active: HashMap<u64, Arc<ArrayDescriptor>>,
    desired: HashMap<u32, u64>,
    stopping: bool,
}

struct Shared {
    inbox: Mutex<Inbox>,
    changed: Condvar,
    cache: Mutex<Cache>,
    waker: Mutex<rune::Waker>,
}

struct Completion {
    request: Request,
    asset: Result<Asset, String>,
    finished: bool,
    cache_hit: bool,
}

struct Slot {
    request: Request,
    asset: Option<Asset>,
    status: ImagingStatus,
    displayed: (u64, bool),
}

/// Owns two background readers and a byte-bounded LRU. Hot cache hits need no worker roundtrip.
pub struct Imaging {
    remote: Option<Remote>,
    shared: Arc<Shared>,
    events: Option<mpsc::Receiver<Completion>>,
    threads: Vec<JoinHandle<()>>,
    slots: HashMap<u32, Slot>,
    next_generation: u64,
    epoch: u64,
    timings: VecDeque<LatencySample>,
}

impl Default for Imaging {
    fn default() -> Self {
        Self::with_budget(256 * 1024 * 1024)
    }
}

impl Imaging {
    pub fn with_budget(bytes: usize) -> Self {
        let shared = Arc::new(Shared {
            inbox: Mutex::new(Inbox::default()),
            changed: Condvar::new(),
            cache: Mutex::new(Cache {
                epoch: 0,
                entries: HashMap::new(),
                order: VecDeque::new(),
                bytes: 0,
                budget: bytes.clamp(8 * 1024 * 1024, 1024 * 1024 * 1024),
            }),
            waker: Mutex::new(Arc::new(|| {})),
        });
        let (send, events) = mpsc::sync_channel(16);
        let threads = (0..2)
            .map(|_| {
                let shared = shared.clone();
                let send = send.clone();
                std::thread::spawn(move || worker(shared, send))
            })
            .collect();
        Self {
            remote: None,
            shared,
            events: Some(events),
            threads,
            slots: HashMap::new(),
            next_generation: 0,
            epoch: 0,
            timings: VecDeque::new(),
        }
    }

    /// Project replacement changes the source of every pane together.
    pub fn select_source(&mut self, fetch: bool) {
        self.invalidate();
        if fetch != self.remote.is_some() {
            self.remote = fetch.then(|| Remote::new(self.shared.clone()));
        }
    }

    pub fn dispatch(&mut self, engine: &mut Engine, windows: &[Option<[f32; 2]>]) -> bool {
        // Consume completed reads before opening another round of up to MAX_SLOTS fetches.
        let changed = self.poll();
        self.remote
            .as_mut()
            .is_some_and(|r| r.dispatch(engine, &mut self.slots, windows))
            || changed
    }

    pub fn worker_reply(&mut self, response: Response, stale: bool) -> Option<Response> {
        match &mut self.remote {
            Some(remote) => remote.receive(response, stale),
            None => Some(response),
        }
    }

    pub fn disconnected(&mut self) {
        if self.remote.is_some() {
            self.remote = Some(Remote::new(self.shared.clone()));
            for slot in self
                .slots
                .values_mut()
                .filter(|s| s.request.key.plane.is_some())
            {
                self.next_generation += 1;
                slot.request.generation = self.next_generation;
                slot.request.lineage = self.next_generation;
                slot.asset = None;
                slot.status = ImagingStatus::Loading;
            }
        }
    }

    pub fn set_waker(&mut self, waker: rune::Waker) {
        *self.shared.waker.locked() = waker;
    }

    pub fn request_plane(
        &mut self,
        slot: u32,
        descriptor: ArrayDescriptor,
        axis: usize,
        index: usize,
        max_edge: usize,
    ) -> Result<(), String> {
        reader::validate(&descriptor)?;
        if axis >= descriptor.shape.len()
            || (descriptor.shape.len() == 3 && index >= descriptor.shape[axis])
        {
            return Err("The requested slice is outside the dataset.".into());
        }
        if !(32..=4096).contains(&max_edge) {
            return Err("Slice quality must be between 32 and 4096 pixels.".into());
        }
        let longest = descriptor
            .shape
            .iter()
            .enumerate()
            .filter(|(a, _)| descriptor.shape.len() == 2 || *a != axis)
            .map(|(_, &length)| length)
            .max()
            .unwrap_or(max_edge);
        self.request(slot, descriptor, Some((axis, index)), max_edge.min(longest))
    }

    pub fn request_volume(
        &mut self,
        slot: u32,
        descriptor: ArrayDescriptor,
        max_edge: usize,
    ) -> Result<(), String> {
        reader::validate(&descriptor)?;
        if descriptor.shape.len() != 3 {
            return Err("Volume rendering requires a three-dimensional array.".into());
        }
        if !(32..=256).contains(&max_edge) {
            return Err(
                "Volume preview quality must be between 32 and 256 voxels per edge.".into(),
            );
        }
        self.request(slot, descriptor, None, max_edge)
    }

    fn request(
        &mut self,
        slot: u32,
        descriptor: ArrayDescriptor,
        plane: Option<(usize, usize)>,
        edge: usize,
    ) -> Result<(), String> {
        if self.events.is_none() {
            return Err("The image reader has been shut down.".into());
        }
        let key = Key {
            epoch: self.epoch,
            source: serde_json::to_string(&descriptor)
                .map_err(|e| e.to_string())?
                .into(),
            plane,
            edge,
        };
        if self
            .slots
            .get(&slot)
            .is_some_and(|s| s.request.key == key && !matches!(s.status, ImagingStatus::Error(_)))
        {
            return Ok(());
        }
        self.next_generation += 1;
        let remote_plane = self.remote.is_some() && plane.is_some();
        let previous = self.slots.remove(&slot).filter(|s| {
            s.request.key.source == key.source
                && s.request.key.plane.map(|p| p.0) == plane.map(|p| p.0)
        });
        let lineage = previous
            .as_ref()
            .map_or(self.next_generation, |s| s.request.lineage);
        let displayed = previous.as_ref().map_or((0, false), |s| s.displayed);
        if previous.is_none()
            && let Some(remote) = &mut self.remote
        {
            remote.clear(slot);
        }
        let request = Request {
            slot,
            generation: self.next_generation,
            descriptor: Arc::new(descriptor),
            key: key.clone(),
            submitted: Instant::now(),
            lineage,
        };
        let cached = if remote_plane {
            None
        } else {
            self.shared.cache.locked().get(&key)
        };
        let mut inbox = self.shared.inbox.locked();
        if inbox.desired.len() >= MAX_SLOTS && !inbox.desired.contains_key(&slot) {
            return Err("The workbench already has 32 active image views.".into());
        }
        inbox.desired.insert(slot, request.generation);
        if cached.is_some() || remote_plane {
            inbox.pending.remove(&slot);
        } else {
            inbox.pending.insert(slot, request.clone());
        }
        drop(inbox);
        self.slots.insert(
            slot,
            Slot {
                request: request.clone(),
                asset: cached.clone().or_else(|| previous.and_then(|s| s.asset)),
                displayed,
                status: if cached.is_some() {
                    ImagingStatus::Ready
                } else {
                    ImagingStatus::Loading
                },
            },
        );
        if let Some(asset) = cached {
            self.record(&request, &asset, true, true);
        } else if remote_plane {
            let wake = self.shared.waker.locked().clone();
            wake();
        } else {
            self.shared.changed.notify_one();
        }
        Ok(())
    }

    pub fn poll(&mut self) -> bool {
        let mut changed = false;
        while let Some(event) = self
            .events
            .as_ref()
            .and_then(|r| r.try_recv().ok())
            .or_else(|| self.remote.as_ref().and_then(Remote::poll))
        {
            changed |= self.install(event);
        }
        changed
    }

    fn install(&mut self, event: Completion) -> bool {
        let Some(slot) = self.slots.get_mut(&event.request.slot) else {
            return false;
        };
        let current = slot.request.generation == event.request.generation;
        let arrival = (event.request.generation, event.finished);
        let interim = !event.finished
            && matches!(&event.asset, Ok(Asset::Plane(p)) if !p.lossless)
            && event.request.lineage == slot.request.lineage;
        if arrival < slot.displayed || (!current && !interim) {
            return false;
        }
        match event.asset {
            Ok(asset) => {
                slot.asset = Some(asset.clone());
                slot.displayed = arrival;
                slot.status = if !current {
                    ImagingStatus::Loading
                } else if event.finished {
                    ImagingStatus::Ready
                } else {
                    ImagingStatus::Refining
                };
                self.record(&event.request, &asset, event.cache_hit, event.finished);
            }
            // The previous pixels are no longer the requested plane.
            Err(error) => {
                slot.asset = None;
                slot.status = ImagingStatus::Error(error);
            }
        }
        true
    }

    fn record(&mut self, request: &Request, asset: &Asset, cache_hit: bool, refined: bool) {
        self.timings.push_back(LatencySample {
            slot: request.slot,
            milliseconds: request.submitted.elapsed().as_secs_f64() * 1000.,
            cache_hit,
            refined,
            bytes: asset.bytes(),
        });
        if self.timings.len() > 256 {
            self.timings.pop_front();
        }
    }

    pub fn plane(&self, slot: u32) -> Option<&Arc<PlaneData>> {
        match self.slots.get(&slot)?.asset.as_ref()? {
            Asset::Plane(p) => Some(p),
            _ => None,
        }
    }
    pub fn volume(&self, slot: u32) -> Option<&Arc<VolumeData>> {
        match self.slots.get(&slot)?.asset.as_ref()? {
            Asset::Volume(v) => Some(v),
            _ => None,
        }
    }
    pub fn status(&self, slot: u32) -> &ImagingStatus {
        self.slots
            .get(&slot)
            .map_or(&ImagingStatus::Idle, |s| &s.status)
    }
    pub fn cache_bytes(&self) -> usize {
        self.shared.cache.locked().bytes
    }
    pub fn timings(&self) -> &VecDeque<LatencySample> {
        &self.timings
    }

    pub fn clear(&mut self, slot: u32) {
        if let Some(remote) = &mut self.remote {
            remote.clear(slot);
        }
        self.slots.remove(&slot);
        let mut inbox = self.shared.inbox.locked();
        inbox.desired.remove(&slot);
        inbox.pending.remove(&slot);
    }

    /// After the caller clears/rebinds every obsolete view, this remains true
    /// until pending reads, refinements and neighbor prefetch have all stopped
    /// using the scalar *and coordinate* artifacts. Decoded cache entries own
    /// their bytes and do not retain a filesystem lease.
    pub fn artifacts_in_use(&self, artifacts: &BTreeSet<String>) -> bool {
        if artifacts.is_empty() {
            return false;
        }
        if self.remote.as_ref().is_some_and(|r| r.uses(artifacts)) {
            return true;
        }
        let inbox = self.shared.inbox.locked();
        inbox
            .pending
            .values()
            .map(|request| &request.descriptor)
            .chain(inbox.active.values())
            .any(|descriptor| {
                descriptor
                    .artifacts()
                    .any(|artifact| artifacts.contains(artifact))
            })
    }

    /// Forget all decoded data after project replacement, relinking or explicit source
    /// refresh. No filesystem checks run on hot navigation; the project owns this boundary.
    /// An epoch prevents a reader already working on old storage from repopulating the cache.
    pub fn invalidate(&mut self) {
        if let Some(remote) = &mut self.remote {
            remote.invalidate();
        }
        self.epoch += 1;
        self.slots.clear();
        {
            let mut inbox = self.shared.inbox.locked();
            inbox.pending.clear();
            inbox.desired.clear();
        }
        let mut cache = self.shared.cache.locked();
        cache.epoch = self.epoch;
        cache.entries.clear();
        cache.order.clear();
        cache.bytes = 0;
    }

    pub fn shutdown(&mut self) {
        self.remote.take();
        {
            let mut inbox = self.shared.inbox.locked();
            inbox.stopping = true;
            inbox.pending.clear();
        }
        self.shared.changed.notify_all();
        self.events.take();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
        self.slots.clear();
    }
}

impl Drop for Imaging {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn current(shared: &Shared, request: &Request) -> bool {
    let inbox = shared.inbox.locked();
    !inbox.stopping && inbox.desired.get(&request.slot) == Some(&request.generation)
}

fn load(shared: &Shared, request: &Request, key: &Key) -> Result<(Asset, bool), String> {
    if let Some(asset) = shared.cache.locked().get(key) {
        return Ok((asset, true));
    }
    let asset = if let Some((axis, index)) = key.plane {
        Asset::Plane(Arc::new(reader::plane(
            &request.descriptor,
            axis,
            index,
            key.edge,
        )?))
    } else {
        Asset::Volume(Arc::new(reader::volume(
            &request.descriptor,
            key.edge,
            &|| !current(shared, request),
        )?))
    };
    shared.cache.locked().insert(key.clone(), asset.clone());
    Ok((asset, false))
}

fn worker(shared: Arc<Shared>, events: mpsc::SyncSender<Completion>) {
    loop {
        let request = {
            let inbox = shared.inbox.locked();
            let mut inbox = shared
                .changed
                .wait_while(inbox, |inbox| inbox.pending.is_empty() && !inbox.stopping)
                .unwrap_or_else(PoisonError::into_inner);
            if inbox.stopping {
                return;
            }
            let slot = inbox
                .pending
                .values()
                .max_by_key(|request| request.generation)
                .map(|request| request.slot);
            let Some(request) = slot.and_then(|slot| inbox.pending.remove(&slot)) else {
                continue;
            };
            inbox
                .active
                .insert(request.generation, request.descriptor.clone());
            request
        };
        let _reading = ActiveRead {
            shared: &shared,
            generation: request.generation,
        };
        let coarse = if request.key.plane.is_some() { 384 } else { 80 };
        let edges =
            if request.key.edge > coarse && request.descriptor.shape.iter().any(|&n| n > coarse) {
                vec![coarse, request.key.edge]
            } else {
                vec![request.key.edge]
            };
        for (i, edge) in edges.iter().enumerate() {
            if !current(&shared, &request) {
                break;
            }
            let mut key = request.key.clone();
            key.edge = *edge;
            let result = load(&shared, &request, &key);
            if !current(&shared, &request) {
                break;
            }
            let (asset, cache_hit) = match result {
                Ok((a, hit)) => (Ok(a), hit),
                Err(e) => (Err(e), false),
            };
            let failed = asset.is_err();
            if events
                .send(Completion {
                    request: request.clone(),
                    asset,
                    finished: i + 1 == edges.len(),
                    cache_hit,
                })
                .is_err()
            {
                return;
            }
            let wake = shared.waker.locked().clone();
            wake();
            if failed {
                break;
            }
        }
        // Prefetch only neighbors of a completed current view, yielding to foreground work.
        if let Some((axis, index)) = request.key.plane {
            if request.descriptor.shape.len() != 3 {
                continue;
            }
            for offset in [1isize, -1, 2, -2] {
                if !current(&shared, &request) || !shared.inbox.locked().pending.is_empty() {
                    break;
                }
                let neighbor = index as isize + offset;
                if neighbor < 0 || neighbor as usize >= request.descriptor.shape[axis] {
                    continue;
                }
                let mut key = request.key.clone();
                key.plane = Some((axis, neighbor as usize));
                let _ = load(&shared, &request, &key);
            }
        }
    }
}

struct ActiveRead<'a> {
    shared: &'a Shared,
    generation: u64,
}
impl Drop for ActiveRead<'_> {
    fn drop(&mut self) {
        self.shared.inbox.locked().active.remove(&self.generation);
        let wake = self.shared.waker.locked().clone();
        wake();
    }
}

#[cfg(test)]
mod generation_tests;
