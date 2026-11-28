//! Exhaustive model checks of Studio's asynchronous protocol with `tomojax serve`.
//!
//! Each module abstracts one owner in `src/` to the facts its guarantees depend
//! on, interleaves it with the worker and the pipes between them, and lets
//! `stateright` visit every reachable state breadth-first. Transitions name the
//! function they mirror. Violations are recorded on the state as a `Breach` so a
//! counterexample shows the exact step that broke the guarantee, and `Seen`
//! flags back `sometimes` properties that prove the interesting interleavings
//! are reachable rather than vacuously safe.
//!
//! Every model also takes a `Mutation` that reintroduces a plausible bug. The
//! `*_is_caught` tests check that the matching property finds it, so a model
//! that silently stopped exercising a guarantee would fail. `STATECHART.md`
//! draws the verified machines.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;
use std::time::Instant;

use stateright::{Checker, Model};

/// Visit every reachable state within the model's bounds and require every
/// property to hold.
fn verify<M>(model: M)
where
    M: Model + Send + Sync + 'static,
    M::State: Debug + Hash + Send + Sync + Clone + PartialEq + 'static,
    M::Action: Debug + Clone + PartialEq,
{
    let name = std::any::type_name::<M>();
    let started = Instant::now();
    let checker = model.checker().spawn_bfs().join();
    checker.assert_properties();
    let states = checker.unique_state_count();
    eprintln!(
        "{name}: {states} unique states ({} generated), depth {}, {:.2?}",
        checker.state_count(),
        checker.max_depth(),
        started.elapsed()
    );
}

/// Require `property` to fail on the mutated model and return a shortest
/// failing path. The checker keeps the last violation it visits, so the depth
/// bound grows one step at a time until the first violation appears.
fn counterexample<M>(model: M, property: &'static str) -> Vec<M::Action>
where
    M: Model + Clone + Send + Sync + 'static,
    M::State: Debug + Hash + Send + Sync + Clone + PartialEq + 'static,
    M::Action: Debug + Clone + PartialEq,
{
    for depth in 2..=64 {
        let checker = model
            .clone()
            .checker()
            .target_max_depth(depth)
            .spawn_bfs()
            .join();
        if let Some(path) = checker.discovery(property) {
            eprintln!("{property}:\n{path}");
            return path.into_actions();
        }
    }
    panic!("the mutation should violate {property:?}");
}

/// Preview channels: `engine.rs` `Requests::observe`, `previews.rs` intents and
/// `collect_retired`, `previews/responses.rs` `consume`, and the worker's
/// `admit`/`_accept` channel revisions in `cli/_session.py`.
mod preview {
    use super::*;
    use stateright::Property;

    /// Slider values the user sets, and channel requests Studio may submit.
    const EDITS: u8 = 4;
    const REQUESTS: u8 = 4;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub enum Mutation {
        #[default]
        None,
        /// The bridge predicts a revision when it sends a request.
        SpeculativeRevision,
        /// Only engine revisions decide staleness; the slot has no intent generation.
        RevisionOnly,
        /// Retirement closes outputs without asking whether a native read holds them.
        CloseDuringRead,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Reply {
        Result {
            request: u8,
            revision: u8,
        },
        Superseded {
            request: u8,
            revision: u8,
        },
        /// `ProtocolError`: rejected before admission, so it carries no channel.
        Rejected {
            request: u8,
        },
    }

    impl Reply {
        fn request(self) -> u8 {
            match self {
                Self::Result { request, .. }
                | Self::Superseded { request, .. }
                | Self::Rejected { request } => request,
            }
        }
        fn revision(self) -> Option<u8> {
            match self {
                Self::Result { revision, .. } | Self::Superseded { revision, .. } => Some(revision),
                Self::Rejected { .. } => None,
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Action {
        Edit,
        Submit,
        WorkerReceive { admit: bool },
        WorkerFinish,
        Deliver,
        BeginRead,
        EndRead,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct Shown {
        request: u8,
        revision: u8,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum Breach {
        FalselyStale,
        PublishedStaleIntent,
        RevisionRegressed,
        ClosedDuringRead,
        ClosedTwice,
        UnknownRequest,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum Seen {
        RejectedAfterAdmission,
        BufferedTerminalDiscarded,
        CloseDeferredByRead,
    }

    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    pub struct State {
        // Worker: requests in its stdin pipe, the channel revision, admitted
        // unfinished requests, its stdout pipe and the outputs it still holds.
        to_worker: VecDeque<u8>,
        revision: u8,
        admitted: Vec<(u8, u8)>,
        to_studio: VecDeque<Reply>,
        finished: BTreeSet<u8>,
        open: BTreeSet<u8>,
        // Engine::Requests: pending request -> admitted revision, newest observed.
        pending: BTreeMap<u8, Option<u8>>,
        newest: u8,
        submitted: u8,
        // Previews slot: intent (generation, request), Failed phase, accepted
        // revision, published result, submissions -> generation, retired outputs.
        generation: u8,
        intent: Option<(u8, Option<u8>)>,
        failed: bool,
        slot_revision: u8,
        current: Option<Shown>,
        submissions: BTreeMap<u8, u8>,
        retired: BTreeSet<u8>,
        /// The output a native reader (`Imaging`) is reading.
        reading: Option<u8>,
        breaches: BTreeSet<Breach>,
        seen: BTreeSet<Seen>,
    }

    impl State {
        fn settled(&self) -> bool {
            self.to_worker.is_empty()
                && self.admitted.is_empty()
                && self.to_studio.is_empty()
                && self.reading.is_none()
        }
    }

    #[derive(Clone, Default)]
    pub struct Previews {
        pub mutation: Mutation,
    }

    impl Previews {
        /// `Previews::collect_retired` with `Imaging::artifacts_in_use`.
        fn collect(&self, s: &mut State) {
            for output in s.retired.clone() {
                if s.reading == Some(output) {
                    if self.mutation != Mutation::CloseDuringRead {
                        s.seen.insert(Seen::CloseDeferredByRead);
                        continue;
                    }
                    s.breaches.insert(Breach::ClosedDuringRead);
                }
                if !s.open.remove(&output) {
                    s.breaches.insert(Breach::ClosedTwice);
                }
                s.retired.remove(&output);
            }
        }

        fn deliver(&self, s: &mut State, reply: Reply) {
            let request = reply.request();
            // Requests::observe
            let Some(admitted) = s.pending.get_mut(&request) else {
                s.breaches.insert(Breach::UnknownRequest);
                return;
            };
            let mut stale = false;
            if let Some(revision) = reply.revision() {
                *admitted = Some(revision);
                stale = revision < s.newest;
                s.newest = s.newest.max(revision);
                if stale && s.revision <= revision {
                    s.breaches.insert(Breach::FalselyStale);
                }
            } else if s.newest > 0 {
                s.seen.insert(Seen::RejectedAfterAdmission);
            }
            s.pending.remove(&request);
            // Previews::consume
            let submitted = s
                .submissions
                .remove(&request)
                .expect("one submission per request");
            let current = self.mutation == Mutation::RevisionOnly
                || s.intent == Some((submitted, Some(request)));
            let admitted = reply.revision();
            let valid =
                current && !stale && admitted.is_none_or(|revision| revision >= s.slot_revision);
            if valid && let Some(revision) = admitted {
                s.slot_revision = revision;
            }
            match reply {
                Reply::Result { revision, .. } if valid => {
                    if submitted != s.generation {
                        s.breaches.insert(Breach::PublishedStaleIntent);
                    }
                    if s.current.is_some_and(|shown| shown.revision > revision) {
                        s.breaches.insert(Breach::RevisionRegressed);
                    }
                    let shown = Shown { request, revision };
                    if let Some(old) = s.current.replace(shown) {
                        s.retired.insert(old.request);
                    }
                }
                Reply::Result { .. } => {
                    if !stale && !current {
                        s.seen.insert(Seen::BufferedTerminalDiscarded);
                    }
                    s.retired.insert(request);
                }
                Reply::Rejected { .. } if valid => s.failed = true,
                _ => {}
            }
            self.collect(s);
        }
    }

    impl Model for Previews {
        type State = State;
        type Action = Action;

        fn init_states(&self) -> Vec<State> {
            vec![State::default()]
        }

        fn actions(&self, s: &State, actions: &mut Vec<Action>) {
            if s.generation < EDITS {
                actions.push(Action::Edit);
            }
            if matches!(s.intent, Some((_, None))) && !s.failed && s.submitted < REQUESTS {
                actions.push(Action::Submit);
            }
            if !s.to_worker.is_empty() {
                actions.push(Action::WorkerReceive { admit: true });
                actions.push(Action::WorkerReceive { admit: false });
            }
            if !s.admitted.is_empty() {
                actions.push(Action::WorkerFinish);
            }
            if !s.to_studio.is_empty() {
                actions.push(Action::Deliver);
            }
            if s.current.is_some() && s.reading.is_none() {
                actions.push(Action::BeginRead);
            }
            if s.reading.is_some() {
                actions.push(Action::EndRead);
            }
        }

        fn next_state(&self, last: &State, action: Action) -> Option<State> {
            let mut s = last.clone();
            match action {
                // Previews::request: a changed setting is a new intent generation.
                Action::Edit => {
                    s.generation += 1;
                    s.intent = Some((s.generation, None));
                    s.failed = false;
                }
                // Previews::poll -> Engine::request on the slot's channel.
                Action::Submit => {
                    s.submitted += 1;
                    let request = s.submitted;
                    s.pending.insert(request, None);
                    if self.mutation == Mutation::SpeculativeRevision {
                        s.newest = s.newest.max(request);
                    }
                    s.submissions.insert(request, s.generation);
                    s.intent = Some((s.generation, Some(request)));
                    s.to_worker.push_back(request);
                }
                // Session.admit/_accept: only an admitted request takes a
                // revision, and admission supersedes the channel's older work.
                Action::WorkerReceive { admit } => {
                    let request = s.to_worker.pop_front()?;
                    if admit {
                        s.revision += 1;
                        for (old, revision) in s.admitted.drain(..) {
                            s.to_studio.push_back(Reply::Superseded {
                                request: old,
                                revision,
                            });
                        }
                        s.admitted.push((request, s.revision));
                    } else {
                        s.to_studio.push_back(Reply::Rejected { request });
                    }
                }
                Action::WorkerFinish => {
                    let (request, revision) = s.admitted.remove(0);
                    s.to_studio.push_back(Reply::Result { request, revision });
                    s.finished.insert(request);
                    s.open.insert(request);
                }
                Action::Deliver => {
                    let reply = s.to_studio.pop_front()?;
                    self.deliver(&mut s, reply);
                }
                Action::BeginRead => s.reading = s.current.map(|shown| shown.request),
                Action::EndRead => {
                    s.reading = None;
                    self.collect(&mut s);
                }
            }
            Some(s)
        }

        fn properties(&self) -> Vec<Property<Self>> {
            vec![
                Property::always("stale only after a newer admission", |_, s: &State| {
                    !s.breaches.contains(&Breach::FalselyStale)
                }),
                Property::always(
                    "published preview matches the latest intent",
                    |_, s: &State| !s.breaches.contains(&Breach::PublishedStaleIntent),
                ),
                Property::always("published revisions never go backwards", |_, s: &State| {
                    !s.breaches.contains(&Breach::RevisionRegressed)
                }),
                Property::always("no output closed during a native read", |_, s: &State| {
                    !s.breaches.contains(&Breach::ClosedDuringRead)
                }),
                Property::always("each output closed once", |_, s: &State| {
                    !s.breaches.contains(&Breach::ClosedTwice)
                }),
                Property::always(
                    "every reply correlates with a pending request",
                    |_, s: &State| !s.breaches.contains(&Breach::UnknownRequest),
                ),
                Property::always(
                    "at rest every request and superseded output is resolved",
                    |_, s: &State| {
                        !s.settled()
                            || (s.pending.is_empty()
                                && s.submissions.is_empty()
                                && s.retired.is_empty()
                                && s.open
                                    .iter()
                                    .copied()
                                    .eq(s.current.map(|shown| shown.request)))
                    },
                ),
                Property::always(
                    "at rest the latest finished intent is published",
                    |_, s: &State| {
                        !s.settled()
                            || !matches!(s.intent, Some((_, Some(r))) if s.finished.contains(&r))
                            || s.current.map(|shown| shown.request) == s.intent.and_then(|i| i.1)
                    },
                ),
                Property::sometimes(
                    "a replacement is rejected after an admission",
                    |_, s: &State| s.seen.contains(&Seen::RejectedAfterAdmission),
                ),
                Property::sometimes("a buffered older terminal is discarded", |_, s: &State| {
                    s.seen.contains(&Seen::BufferedTerminalDiscarded)
                }),
                Property::sometimes(
                    "a superseded output waits for its reader",
                    |_, s: &State| s.seen.contains(&Seen::CloseDeferredByRead),
                ),
            ]
        }
    }
}

/// One captured script, one manifest writer, and a worker whose directory survives a drop.
mod job {
    use super::*;
    use stateright::Property;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct Snapshot {
        known: bool,
        offset: u8,
        assets: u8,
    }
    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    pub struct State {
        action_taken: bool,
        permission: bool,
        submits: u8,
        in_pipe: bool,
        launched: bool,
        receipt: bool,
        read_reply: bool,
        connected: bool,
        memory: Snapshot,
        writing: Option<Snapshot>,
        disk: Snapshot,
    }
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub enum Action {
        Capture,
        Snapshot,
        Commit,
        Submit,
        Launch,
        Finish,
        Poll,
        Adopt,
        Drop,
        Connect,
        Crash,
    }
    #[derive(Clone, Default)]
    pub struct Jobs {
        pub replay: bool,
        pub split_save: bool,
    }
    impl Model for Jobs {
        type State = State;
        type Action = Action;
        fn init_states(&self) -> Vec<State> {
            vec![State {
                connected: true,
                ..Default::default()
            }]
        }
        fn actions(&self, s: &State, a: &mut Vec<Action>) {
            if s.connected && !s.action_taken {
                a.push(Action::Capture);
            }
            if s.memory != s.disk && s.writing != Some(s.memory) {
                a.push(Action::Snapshot);
            }
            if s.writing.is_some() {
                a.push(Action::Commit);
            }
            if s.permission && s.disk.known && s.connected && s.submits < 2 {
                a.push(Action::Submit);
            }
            if s.in_pipe {
                a.push(Action::Launch);
            }
            if s.launched && !s.receipt {
                a.push(Action::Finish);
            }
            if s.connected && s.memory.known && s.receipt && !s.read_reply {
                a.push(Action::Poll);
            }
            if s.read_reply {
                a.push(Action::Adopt);
            }
            if s.connected {
                a.push(Action::Drop);
            } else {
                a.push(Action::Connect);
            }
            a.push(Action::Crash);
        }
        fn next_state(&self, last: &State, action: Action) -> Option<State> {
            let mut s = last.clone();
            match action {
                Action::Capture => {
                    s.action_taken = true;
                    s.permission = true;
                    s.memory.known = true;
                }
                Action::Snapshot => s.writing = Some(s.memory),
                Action::Commit => s.disk = s.writing.take()?,
                Action::Submit => {
                    s.permission = false;
                    s.in_pipe = true;
                    s.submits += 1;
                }
                Action::Launch => {
                    s.launched = true;
                    s.in_pipe = false;
                }
                Action::Finish => s.receipt = true,
                Action::Poll => s.read_reply = true,
                Action::Adopt => {
                    s.read_reply = false;
                    if s.memory.offset == 0 {
                        s.memory.offset = 1;
                        if self.split_save {
                            s.writing = Some(s.memory);
                        }
                        s.memory.assets += 1;
                    }
                }
                Action::Drop => {
                    s.connected = false;
                    s.permission = false;
                    s.read_reply = false;
                }
                Action::Connect => s.connected = true,
                Action::Crash => {
                    s.connected = false;
                    s.memory = s.disk;
                    s.writing = None;
                    s.permission = self.replay && s.disk.known;
                    s.read_reply = false;
                }
            }
            (s != *last).then_some(s)
        }
        fn properties(&self) -> Vec<Property<Self>> {
            vec![
                Property::always("one submit per user action", |_, s: &State| s.submits <= 1),
                Property::always("every launch has a saved project record", |_, s: &State| {
                    !s.launched || s.disk.known
                }),
                Property::always("the saved offset and result agree", |_, s: &State| {
                    s.disk.assets == s.disk.offset && s.disk.assets <= 1
                }),
                Property::always("a receipt is always recoverable", |_, s: &State| {
                    !s.receipt || (s.disk.known && (s.disk.assets == 1 || s.disk.offset == 0))
                }),
                Property::sometimes(
                    "a receipt survives a restart before adoption",
                    |_, s: &State| s.receipt && !s.connected && s.disk.assets == 0,
                ),
                Property::sometimes("a result is saved", |_, s: &State| s.disk.assets == 1),
            ]
        }
    }
}

mod save {
    use super::*;
    use stateright::Property;

    /// Project edits after the save on open.
    const CHANGES: u8 = 4;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub enum Mutation {
        #[default]
        None,
        /// Acknowledgments carry no generation; any success marks the latest saved.
        AckWithoutGeneration,
        /// Every acknowledgment replaces the save error.
        AnyAckSetsError,
        /// `flush` as it was before this model: any failed acknowledgment fails it.
        LegacyFlush,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Action {
        Change { oversized: bool },
        WriterTake,
        WriterFinish { ok: bool },
        Poll,
        Flush,
        FlushReceive,
    }

    /// The generation each error describes.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct SaveError(u8);

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum FlushBreach {
        OkWhileUnsaved,
        FailedOnOlderError,
        WaitsForever,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum Seen {
        OlderAckAfterNewerChange,
        OlderSuccessAfterNewerError,
        FlushSkipsOlderFailure,
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct State {
        // Project
        generation: u8,
        saved: u8,
        save_error: Option<SaveError>,
        flushing: bool,
        // Writer thread: the pending snapshot slot, the signal and the write.
        pending: Option<u8>,
        signal: bool,
        writing: Option<u8>,
        acks: VecDeque<(u8, bool)>,
        /// Generation of the manifest on disk.
        disk: u8,
        /// The latest generation is known to have failed: withheld, or its
        /// failed acknowledgment has been received.
        latest_failed: bool,
        breaches: BTreeSet<FlushBreach>,
        seen: BTreeSet<Seen>,
    }

    #[derive(Clone, Default)]
    pub struct Save {
        pub mutation: Mutation,
    }

    impl Save {
        /// Project::apply, the acknowledgment rule poll and flush share.
        fn apply(&self, s: &mut State, generation: u8, ok: bool) {
            if generation < s.generation {
                s.seen.insert(Seen::OlderAckAfterNewerChange);
                if ok && s.save_error.is_some() {
                    s.seen.insert(Seen::OlderSuccessAfterNewerError);
                }
            }
            if generation == s.generation && !ok {
                s.latest_failed = true;
            }
            if ok {
                s.saved = match self.mutation {
                    Mutation::AckWithoutGeneration => s.generation,
                    _ => s.saved.max(generation),
                };
            }
            let error = (!ok).then_some(SaveError(generation));
            if generation == s.generation || self.mutation == Mutation::AnyAckSetsError {
                s.save_error = error;
            }
        }

        fn finish_flush(&self, s: &mut State, result: Result<(), SaveError>) {
            s.flushing = false;
            match result {
                Ok(()) if s.disk != s.generation => {
                    s.breaches.insert(FlushBreach::OkWhileUnsaved);
                }
                Err(SaveError(generation)) if generation != s.generation => {
                    s.breaches.insert(FlushBreach::FailedOnOlderError);
                }
                _ => {}
            }
        }

        /// Project::flush, one received acknowledgment per step.
        fn flush_step(&self, s: &mut State) {
            if let Some(error) = s.save_error {
                self.finish_flush(s, Err(error));
                return;
            }
            if s.saved >= s.generation {
                self.finish_flush(s, Ok(()));
                return;
            }
            let Some((generation, ok)) = s.acks.pop_front() else {
                if s.writing.is_none() && !s.signal {
                    s.breaches.insert(FlushBreach::WaitsForever);
                    s.flushing = false;
                }
                return;
            };
            if self.mutation == Mutation::LegacyFlush {
                if !ok {
                    s.save_error = Some(SaveError(generation));
                    self.finish_flush(s, Err(SaveError(generation)));
                    return;
                }
                s.saved = s.saved.max(generation);
            } else {
                if !ok && generation < s.generation {
                    s.seen.insert(Seen::FlushSkipsOlderFailure);
                }
                self.apply(s, generation, ok);
            }
            self.flush_step(s);
        }
    }

    impl Model for Save {
        type State = State;
        type Action = Action;

        fn init_states(&self) -> Vec<State> {
            // Project::open ends with changed(): generation 1 is queued.
            vec![State {
                generation: 1,
                saved: 0,
                save_error: None,
                flushing: false,
                pending: Some(1),
                signal: true,
                writing: None,
                acks: VecDeque::new(),
                disk: 0,
                latest_failed: false,
                breaches: BTreeSet::new(),
                seen: BTreeSet::new(),
            }]
        }

        fn actions(&self, s: &State, actions: &mut Vec<Action>) {
            if s.flushing {
                // The UI thread is blocked in flush.
                actions.push(Action::FlushReceive);
            } else {
                if s.generation <= CHANGES {
                    actions.push(Action::Change { oversized: false });
                    actions.push(Action::Change { oversized: true });
                }
                if !s.acks.is_empty() {
                    actions.push(Action::Poll);
                }
                actions.push(Action::Flush);
            }
            if s.signal && s.writing.is_none() {
                actions.push(Action::WriterTake);
            }
            if s.writing.is_some() {
                actions.push(Action::WriterFinish { ok: true });
                actions.push(Action::WriterFinish { ok: false });
            }
        }

        fn next_state(&self, last: &State, action: Action) -> Option<State> {
            let mut s = last.clone();
            match action {
                // Project::changed: an oversized snapshot is withheld with an error.
                Action::Change { oversized } => {
                    s.generation += 1;
                    s.save_error = None;
                    s.latest_failed = oversized;
                    if oversized {
                        s.save_error = Some(SaveError(s.generation));
                    } else {
                        s.pending = Some(s.generation);
                        s.signal = true;
                    }
                }
                // The writer drains every signal, then takes the newest snapshot.
                Action::WriterTake => {
                    s.signal = false;
                    s.writing = s.pending.take();
                }
                Action::WriterFinish { ok } => {
                    let generation = s.writing.take()?;
                    if ok {
                        s.disk = generation;
                    }
                    s.acks.push_back((generation, ok));
                }
                Action::Poll => {
                    while let Some((generation, ok)) = s.acks.pop_front() {
                        self.apply(&mut s, generation, ok);
                    }
                }
                Action::Flush | Action::FlushReceive => {
                    s.flushing = true;
                    self.flush_step(&mut s);
                }
            }
            Some(s)
        }

        fn properties(&self) -> Vec<Property<Self>> {
            vec![
                Property::always(
                    "clean means the latest generation is on disk",
                    |_, s: &State| s.generation > s.saved || s.disk == s.generation,
                ),
                Property::always(
                    "a known failure of the latest generation stays visible",
                    |_, s: &State| !s.latest_failed || s.save_error.is_some(),
                ),
                Property::always(
                    "flush succeeds only when the latest generation is on disk",
                    |_, s: &State| !s.breaches.contains(&FlushBreach::OkWhileUnsaved),
                ),
                Property::always(
                    "flush fails only on the latest generation's error",
                    |_, s: &State| !s.breaches.contains(&FlushBreach::FailedOnOlderError),
                ),
                Property::always(
                    "flush always has an acknowledgment to wait for",
                    |_, s: &State| !s.breaches.contains(&FlushBreach::WaitsForever),
                ),
                Property::sometimes(
                    "an older acknowledgment arrives after a newer change",
                    |_, s: &State| s.seen.contains(&Seen::OlderAckAfterNewerChange),
                ),
                Property::sometimes(
                    "an older success arrives after a newer error",
                    |_, s: &State| s.seen.contains(&Seen::OlderSuccessAfterNewerError),
                ),
                Property::sometimes("flush waits past an older failure", |_, s: &State| {
                    s.seen.contains(&Seen::FlushSkipsOlderFailure)
                }),
            ]
        }
    }
}

#[test]
fn preview_channel_supersession_and_intent_generations() {
    verify(preview::Previews::default());
}

#[test]
fn speculative_channel_revisions_are_caught() {
    let model = || preview::Previews {
        mutation: preview::Mutation::SpeculativeRevision,
    };
    counterexample(model(), "stale only after a newer admission");
    // Without a rejection the predicted and admitted revisions agree; after
    // one, the newest intent's result is discarded as stale and never shown.
    let path = counterexample(model(), "at rest the latest finished intent is published");
    assert!(path.contains(&preview::Action::WorkerReceive { admit: false }));
}

#[test]
fn revision_only_staleness_is_caught() {
    counterexample(
        preview::Previews {
            mutation: preview::Mutation::RevisionOnly,
        },
        "published preview matches the latest intent",
    );
}

#[test]
fn closing_a_superseded_output_during_its_read_is_caught() {
    let path = counterexample(
        preview::Previews {
            mutation: preview::Mutation::CloseDuringRead,
        },
        "no output closed during a native read",
    );
    assert!(path.contains(&preview::Action::BeginRead));
}

#[test]
fn durable_jobs_survive_replies_drops_autosaves_and_restarts() {
    verify(job::Jobs::default());
}

#[test]
fn replay_and_split_adoption_are_caught() {
    counterexample(
        job::Jobs {
            replay: true,
            ..Default::default()
        },
        "one submit per user action",
    );
    counterexample(
        job::Jobs {
            split_save: true,
            ..Default::default()
        },
        "the saved offset and result agree",
    );
}

#[test]
fn manifest_save_generations() {
    verify(save::Save::default());
}

#[test]
fn acknowledgments_without_generations_are_caught() {
    counterexample(
        save::Save {
            mutation: save::Mutation::AckWithoutGeneration,
        },
        "clean means the latest generation is on disk",
    );
}

#[test]
fn an_older_success_clearing_a_newer_error_is_caught() {
    counterexample(
        save::Save {
            mutation: save::Mutation::AnyAckSetsError,
        },
        "a known failure of the latest generation stays visible",
    );
}

/// The flush Studio shipped before this model failed on any failed
/// acknowledgment, including an older generation's failure while the newer
/// snapshot was still queued or already on disk.
#[test]
fn legacy_flush_failing_on_an_older_generation_is_caught() {
    counterexample(
        save::Save {
            mutation: save::Mutation::LegacyFlush,
        },
        "flush fails only on the latest generation's error",
    );
}
