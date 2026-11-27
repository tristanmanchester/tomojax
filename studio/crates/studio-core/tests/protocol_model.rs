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

/// Durable job control: `app/controls.rs` (control watch), `app/jobs.rs`
/// (`schedule`, `progress`, terminals) and the worker's single device queue
/// with `progress.Control` pause, resume and cancel.
mod job {
    use super::*;
    use stateright::Property;

    /// Pause, resume and cancel clicks the user makes.
    const CONTROLS: u8 = 3;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub enum Mutation {
        #[default]
        None,
        /// A control acknowledgment is applied as the job's new state or outcome.
        AckIsOutcome,
        /// The scheduler treats a paused job as having freed the device.
        PauseFreesScheduler,
        /// The control watch keeps polling after the target's terminal.
        WatchOutlivesTerminal,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum Request {
        /// The running reconstruction.
        Job,
        /// The next queued durable job.
        Next,
        /// An interactive preview, admitted outside the durable scheduler.
        Preview,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Control {
        Pause,
        Resume,
        Cancel,
    }

    /// `progress.Control._state`, or "queued" before the device takes the job.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Reported {
        Queued,
        Running,
        PauseRequested,
        Paused,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Outcome {
        Completed,
        Canceled,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum ToWorker {
        Execute(Request),
        Control { id: u8, action: Control },
        Status,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum ToStudio {
        Paused,
        Resumed,
        /// `paused` records whether the worker's job was paused when it acknowledged.
        Ack {
            id: u8,
            paused: bool,
        },
        Unknown {
            id: u8,
        },
        Status(Option<Reported>),
        Terminal(Request, Outcome),
    }

    /// `ControlWatch::next_check`: unset until the acknowledgment, then due
    /// now, then 250 ms after each status check.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Check {
        AwaitingAck,
        Later,
        Due,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Action {
        Control(Control),
        /// 250 ms pass on the UI clock.
        Tick,
        SubmitPreview,
        Schedule,
        Deliver,
        WorkerReceive,
        WorkerStart,
        TaskStarted,
        Boundary,
        Wake,
        Finish,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum Breach {
        PausedWithoutEvidence,
        NextJobWhileDeviceHeld,
        StatusAfterTerminal,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum Seen {
        PausedBeforeTask,
        PauseSettledByStatus,
        PreviewBehindPausedJob,
        CanceledWhilePaused,
    }

    /// Studio's view of the job record.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Shown {
        Running,
        Paused,
        Done(Outcome),
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    pub struct State {
        // Worker: the job's control, the queue and the single device.
        state: Reported,
        canceled: bool,
        /// Blocked in `Control.boundary` waiting for resume or cancel.
        blocked: bool,
        /// A numerical task exists, so pause and resume emit progress events.
        task: bool,
        outcome: Option<Outcome>,
        queue: VecDeque<Request>,
        device: Option<Request>,
        to_worker: VecDeque<ToWorker>,
        to_studio: VecDeque<ToStudio>,
        // Studio: pending requests, the job record and the control watch.
        executing: BTreeSet<Request>,
        controls: BTreeMap<u8, Control>,
        status: Option<Control>,
        shown: Shown,
        /// `AppState::running` still holds the job.
        running: bool,
        watch: Option<(Control, Check)>,
        clicks: u8,
        next_sent: bool,
        preview_sent: bool,
        breaches: BTreeSet<Breach>,
        seen: BTreeSet<Seen>,
    }

    impl State {
        fn settled(&self) -> bool {
            self.to_worker.is_empty() && self.to_studio.is_empty() && self.status.is_none()
        }
        fn active(&self) -> bool {
            self.outcome.is_none()
                && (self.device == Some(Request::Job) || self.queue.contains(&Request::Job))
        }
        fn emit(&mut self, message: ToStudio) {
            self.to_studio.push_back(message);
        }
        fn finish_job(&mut self, outcome: Outcome) {
            if self.blocked && outcome == Outcome::Canceled {
                self.seen.insert(Seen::CanceledWhilePaused);
            }
            self.outcome = Some(outcome);
            self.blocked = false;
            self.emit(ToStudio::Terminal(Request::Job, outcome));
        }
        fn show_paused(&mut self, evidence: bool) {
            if !evidence {
                self.breaches.insert(Breach::PausedWithoutEvidence);
            }
            if !self.task {
                self.seen.insert(Seen::PausedBeforeTask);
            }
            self.shown = Shown::Paused;
        }
    }

    #[derive(Clone, Default)]
    pub struct Jobs {
        pub mutation: Mutation,
    }

    impl Jobs {
        /// AppState::poll_control, run after every Studio step.
        fn poll_control(&self, s: &mut State) {
            let Some((action, check)) = s.watch else {
                return;
            };
            let target_pending = s.executing.contains(&Request::Job);
            if !target_pending && self.mutation != Mutation::WatchOutlivesTerminal {
                s.watch = None;
                return;
            }
            if check != Check::Due || s.status.is_some() {
                return;
            }
            if !target_pending {
                s.breaches.insert(Breach::StatusAfterTerminal);
            }
            s.status = Some(action);
            s.to_worker.push_back(ToWorker::Status);
            s.watch = Some((action, Check::Later));
        }

        /// AppState::control_result for a status check.
        fn status_result(&self, s: &mut State, action: Control, report: Option<Reported>) {
            if !s.executing.contains(&Request::Job) || s.watch.map(|w| w.0) != Some(action) {
                return;
            }
            let Some(state) = report else {
                s.watch = None;
                return;
            };
            let settled = matches!(
                (action, state),
                (Control::Pause, Reported::Paused) | (Control::Resume, Reported::Running)
            );
            match state {
                Reported::Paused => s.show_paused(true),
                Reported::Running if action == Control::Resume => s.shown = Shown::Running,
                Reported::PauseRequested => s.shown = Shown::Running,
                Reported::Queued | Reported::Running => {}
            }
            if settled {
                s.seen.insert(Seen::PauseSettledByStatus);
                s.watch = None;
            }
        }

        fn deliver(&self, s: &mut State, message: ToStudio) {
            match message {
                // AppState::progress applies these only to a pending job.
                ToStudio::Paused if s.executing.contains(&Request::Job) => s.show_paused(true),
                ToStudio::Resumed if s.executing.contains(&Request::Job) => {
                    s.shown = Shown::Running
                }
                ToStudio::Paused | ToStudio::Resumed => {}
                ToStudio::Ack { id, paused } => {
                    let action = s.controls.remove(&id).expect("pending control");
                    if self.mutation == Mutation::AckIsOutcome {
                        match action {
                            Control::Pause => s.show_paused(paused),
                            Control::Resume => s.shown = Shown::Running,
                            Control::Cancel => {
                                s.executing.remove(&Request::Job);
                                s.shown = Shown::Done(Outcome::Canceled);
                                s.running = false;
                            }
                        }
                        s.watch = None;
                    } else if s.executing.contains(&Request::Job)
                        && s.watch.map(|w| w.0) == Some(action)
                    {
                        // Ack confirms the request, not arrival at a safe point.
                        s.watch = Some((action, Check::Due));
                    }
                }
                ToStudio::Unknown { id } => {
                    let action = s.controls.remove(&id).expect("pending control");
                    if s.watch.map(|w| w.0) == Some(action) {
                        s.watch = None;
                    }
                }
                ToStudio::Status(report) => {
                    let action = s.status.take().expect("pending status");
                    self.status_result(s, action, report);
                }
                ToStudio::Terminal(request, outcome) => {
                    if s.executing.remove(&request) && request == Request::Job {
                        s.shown = Shown::Done(outcome);
                        s.running = false;
                    }
                }
            }
        }
    }

    impl Model for Jobs {
        type State = State;
        type Action = Action;

        fn init_states(&self) -> Vec<State> {
            // AppState::schedule has sent the job.
            vec![State {
                state: Reported::Running,
                canceled: false,
                blocked: false,
                task: false,
                outcome: None,
                queue: VecDeque::new(),
                device: None,
                to_worker: VecDeque::from([ToWorker::Execute(Request::Job)]),
                to_studio: VecDeque::new(),
                executing: BTreeSet::from([Request::Job]),
                controls: BTreeMap::new(),
                status: None,
                shown: Shown::Running,
                running: true,
                watch: None,
                clicks: 0,
                next_sent: false,
                preview_sent: false,
                breaches: BTreeSet::new(),
                seen: BTreeSet::new(),
            }]
        }

        fn actions(&self, s: &State, actions: &mut Vec<Action>) {
            // AppState::control_job routes controls to a pending execution.
            if s.clicks < CONTROLS && s.executing.contains(&Request::Job) {
                actions.extend(
                    [Control::Pause, Control::Resume, Control::Cancel].map(Action::Control),
                );
            }
            if matches!(s.watch, Some((_, Check::Later))) {
                actions.push(Action::Tick);
            }
            if !s.preview_sent {
                actions.push(Action::SubmitPreview);
            }
            let free = !s.running
                || (self.mutation == Mutation::PauseFreesScheduler && s.shown == Shown::Paused);
            if !s.next_sent && free {
                actions.push(Action::Schedule);
            }
            if !s.to_studio.is_empty() {
                actions.push(Action::Deliver);
            }
            if !s.to_worker.is_empty() {
                actions.push(Action::WorkerReceive);
            }
            if s.device.is_none() && !s.queue.is_empty() {
                actions.push(Action::WorkerStart);
            }
            match s.device {
                Some(Request::Job) => {
                    if !s.task && !s.blocked {
                        actions.push(Action::TaskStarted);
                    }
                    if !s.blocked && (s.canceled || s.state == Reported::PauseRequested) {
                        actions.push(Action::Boundary);
                    }
                    if s.blocked && (s.canceled || s.state != Reported::Paused) {
                        actions.push(Action::Wake);
                    }
                    if !s.blocked {
                        actions.push(Action::Finish);
                    }
                }
                Some(_) => actions.push(Action::Finish),
                None => {}
            }
        }

        fn next_state(&self, last: &State, action: Action) -> Option<State> {
            let mut s = last.clone();
            let studio = matches!(
                action,
                Action::Control(_)
                    | Action::Tick
                    | Action::SubmitPreview
                    | Action::Schedule
                    | Action::Deliver
            );
            match action {
                Action::Control(control) => {
                    s.clicks += 1;
                    let id = s.clicks;
                    s.controls.insert(id, control);
                    s.to_worker.push_back(ToWorker::Control {
                        id,
                        action: control,
                    });
                    s.watch = (control != Control::Cancel).then_some((control, Check::AwaitingAck));
                }
                Action::Tick => {
                    if let Some((_, check)) = &mut s.watch {
                        *check = Check::Due;
                    }
                }
                Action::SubmitPreview => {
                    s.preview_sent = true;
                    s.executing.insert(Request::Preview);
                    s.to_worker.push_back(ToWorker::Execute(Request::Preview));
                }
                Action::Schedule => {
                    if s.device.is_some_and(|r| r == Request::Job)
                        || s.queue.contains(&Request::Job)
                    {
                        s.breaches.insert(Breach::NextJobWhileDeviceHeld);
                    }
                    s.next_sent = true;
                    s.executing.insert(Request::Next);
                    s.to_worker.push_back(ToWorker::Execute(Request::Next));
                }
                Action::Deliver => {
                    let message = s.to_studio.pop_front()?;
                    self.deliver(&mut s, message);
                }
                // Session.admit and Session._fast.
                Action::WorkerReceive => match s.to_worker.pop_front()? {
                    ToWorker::Execute(request) => s.queue.push_back(request),
                    ToWorker::Control { id, action } if s.active() => {
                        match action {
                            Control::Cancel => {
                                s.canceled = true;
                                if let Some(index) = s.queue.iter().position(|r| *r == Request::Job)
                                {
                                    s.queue.remove(index);
                                    s.finish_job(Outcome::Canceled);
                                }
                            }
                            Control::Pause if s.state == Reported::Running && !s.canceled => {
                                s.state = Reported::PauseRequested;
                            }
                            Control::Pause => {}
                            Control::Resume => s.state = Reported::Running,
                        }
                        let paused = s.state == Reported::Paused;
                        s.emit(ToStudio::Ack { id, paused });
                    }
                    ToWorker::Control { id, .. } => s.emit(ToStudio::Unknown { id }),
                    ToWorker::Status => {
                        let report = s.active().then(|| {
                            if s.queue.contains(&Request::Job) {
                                Reported::Queued
                            } else {
                                s.state
                            }
                        });
                        s.emit(ToStudio::Status(report));
                    }
                },
                Action::WorkerStart => {
                    let request = s.queue.pop_front()?;
                    s.device = Some(request);
                }
                Action::TaskStarted => s.task = true,
                // Control.boundary
                Action::Boundary => {
                    if s.canceled {
                        s.device = None;
                        s.finish_job(Outcome::Canceled);
                    } else {
                        s.state = Reported::Paused;
                        s.blocked = true;
                        if s.task {
                            s.emit(ToStudio::Paused);
                        }
                    }
                }
                Action::Wake => {
                    if s.canceled {
                        s.device = None;
                        s.finish_job(Outcome::Canceled);
                    } else {
                        s.blocked = false;
                        if s.task {
                            s.emit(ToStudio::Resumed);
                        }
                    }
                }
                // Session._execute: a canceled job reports cancellation even
                // after its operation returned.
                Action::Finish => match s.device.take()? {
                    Request::Job => {
                        let outcome = if s.canceled {
                            Outcome::Canceled
                        } else {
                            Outcome::Completed
                        };
                        s.finish_job(outcome);
                    }
                    other => s.emit(ToStudio::Terminal(other, Outcome::Completed)),
                },
            }
            if studio {
                self.poll_control(&mut s);
            }
            if s.blocked && s.queue.contains(&Request::Preview) {
                s.seen.insert(Seen::PreviewBehindPausedJob);
            }
            Some(s)
        }

        fn properties(&self) -> Vec<Property<Self>> {
            vec![
                Property::always("paused is shown only on worker evidence", |_, s: &State| {
                    !s.breaches.contains(&Breach::PausedWithoutEvidence)
                }),
                Property::always(
                    "job outcome comes from its terminal",
                    |_, s: &State| match s.shown {
                        Shown::Done(outcome) => s.outcome == Some(outcome),
                        _ => true,
                    },
                ),
                Property::always("an unfinished job stays controllable", |_, s: &State| {
                    s.outcome.is_some() || s.executing.contains(&Request::Job)
                }),
                Property::always(
                    "a paused job keeps the device until its terminal",
                    |_, s: &State| !s.breaches.contains(&Breach::NextJobWhileDeviceHeld),
                ),
                Property::always("no status poll after the terminal", |_, s: &State| {
                    !s.breaches.contains(&Breach::StatusAfterTerminal)
                }),
                Property::always(
                    "at rest a finished job has no control watch",
                    |_, s: &State| {
                        !s.settled() || s.executing.contains(&Request::Job) || s.watch.is_none()
                    },
                ),
                Property::sometimes(
                    "a pause before any task is seen through status",
                    |_, s: &State| s.seen.contains(&Seen::PausedBeforeTask),
                ),
                Property::sometimes(
                    "a status check settles a pause or resume",
                    |_, s: &State| s.seen.contains(&Seen::PauseSettledByStatus),
                ),
                Property::sometimes("a preview waits behind a paused job", |_, s: &State| {
                    s.seen.contains(&Seen::PreviewBehindPausedJob)
                }),
                Property::sometimes("a paused job is canceled", |_, s: &State| {
                    s.seen.contains(&Seen::CanceledWhilePaused)
                }),
            ]
        }
    }
}

/// Live previews: `live.rs` read leases (the `Arc` strong count), token
/// references, result-handle retirement and `disconnected`, drained each frame
/// in `AppState::poll` order (releases, then handle closes).
mod live {
    use super::*;
    use stateright::Property;

    /// Previews each job emits before its result, and worker restarts.
    const PREVIEWS: u8 = 3;
    const RESTARTS: u8 = 1;

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub enum Mutation {
        #[default]
        None,
        /// drain_releases ignores the `Arc` read lease.
        IgnoreReadLease,
        /// drain_handles closes a retiring handle while its previews are owned.
        CloseIgnoresOwners,
        /// disconnected keeps the old session's owners, references and handles.
        KeepLedgerOnDisconnect,
        /// disconnected restarts preview generations at zero.
        ResetGeneration,
    }

    /// Token names recur in every worker session.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum Token {
        Scalar(u8),
        Coordinates,
    }

    /// One `Arc<SolverPreview>`; equality is identity because generations do not
    /// repeat within a session.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct SolverPreview {
        session: u8,
        generation: u8,
        index: u8,
    }

    impl SolverPreview {
        fn tokens(self) -> [Token; 2] {
            [Token::Scalar(self.index), Token::Coordinates]
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Action {
        Accept,
        Result,
        RetireResult,
        Clear,
        Disconnect,
        Poll,
        ReaderTake,
        ReaderFinish { ok: bool },
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum Breach {
        ReleasedUnderReader,
        ClosedUnderReader,
        ReleasedTwice,
        ClosedTwice,
        CrossSessionRelease,
        FrameFromOldSession,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub enum Seen {
        ReleaseDeferredByRead,
        CloseDeferredByRead,
        OldReadFinishedAfterRestart,
    }

    #[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
    pub struct State {
        // LivePreviews
        generation: u8,
        desired: Option<u8>,
        displayed: Option<u8>,
        owners: BTreeMap<u8, SolverPreview>,
        references: BTreeMap<Token, u8>,
        /// The session job's result handle: Some(retiring) once bound.
        handle: Option<bool>,
        /// Session of the displayed frame.
        frame: Option<u8>,
        // Reader thread: the inbox slot, the active read and completions.
        inbox: Option<SolverPreview>,
        active: Option<SolverPreview>,
        completions: VecDeque<(SolverPreview, bool)>,
        // Worker session: issued tokens and the open result handle.
        session: u8,
        emitted: u8,
        result: bool,
        retired: bool,
        issued: BTreeSet<Token>,
        handle_open: bool,
        restarts: u8,
        breaches: BTreeSet<Breach>,
        seen: BTreeSet<Seen>,
    }

    impl State {
        /// Arc::strong_count of an owner: the ledger plus every reader-side copy.
        fn strong(&self, preview: SolverPreview) -> usize {
            1 + usize::from(self.inbox == Some(preview))
                + usize::from(self.active == Some(preview))
                + self
                    .completions
                    .iter()
                    .filter(|(c, _)| *c == preview)
                    .count()
        }
        /// A preview of the live session the reader may still open.
        fn leased(&self, token: Option<Token>) -> bool {
            [self.inbox, self.active]
                .into_iter()
                .flatten()
                .any(|c| c.session == self.session && token.is_none_or(|t| c.tokens().contains(&t)))
        }
        fn idle(&self) -> bool {
            self.inbox.is_none() && self.active.is_none() && self.completions.is_empty()
        }
    }

    #[derive(Clone, Default)]
    pub struct Live {
        pub mutation: Mutation,
    }

    impl Live {
        /// drain_releases then drain_handles, sent to the worker in that order.
        fn drain(&self, s: &mut State) {
            let retiring = s.handle == Some(true);
            let mut released = Vec::new();
            for (&generation, &owner) in &s.owners {
                let unshown = Some(generation) != s.desired && Some(generation) != s.displayed;
                if !(unshown || retiring) {
                    continue;
                }
                if s.strong(owner) > 1 {
                    s.seen.insert(Seen::ReleaseDeferredByRead);
                    if self.mutation != Mutation::IgnoreReadLease {
                        continue;
                    }
                }
                released.push(generation);
            }
            for generation in released {
                let owner = s.owners.remove(&generation).unwrap();
                for token in owner.tokens() {
                    let count = s.references.get_mut(&token).unwrap();
                    *count -= 1;
                    if *count == 0 {
                        s.references.remove(&token);
                        self.release(s, owner.session, token);
                    }
                }
            }
            if retiring {
                let owned = !s.owners.is_empty();
                if owned {
                    s.seen.insert(Seen::CloseDeferredByRead);
                }
                if !owned || self.mutation == Mutation::CloseIgnoresOwners {
                    s.handle = None;
                    self.close(s);
                }
            }
        }

        /// Session `release`: removes one issued token of the live session.
        fn release(&self, s: &mut State, owner_session: u8, token: Token) {
            if owner_session != s.session {
                s.breaches.insert(Breach::CrossSessionRelease);
            }
            if s.leased(Some(token)) {
                s.breaches.insert(Breach::ReleasedUnderReader);
            }
            if !s.issued.remove(&token) {
                s.breaches.insert(Breach::ReleasedTwice);
            }
        }

        /// Session `_close_handle`: drops the handle and every artifact its
        /// request published, tolerating tokens already released.
        fn close(&self, s: &mut State) {
            if s.leased(None) {
                s.breaches.insert(Breach::ClosedUnderReader);
            }
            if !s.handle_open {
                s.breaches.insert(Breach::ClosedTwice);
            }
            s.handle_open = false;
            s.issued.clear();
        }
    }

    impl Model for Live {
        type State = State;
        type Action = Action;

        fn init_states(&self) -> Vec<State> {
            vec![State::default()]
        }

        fn actions(&self, s: &State, actions: &mut Vec<Action>) {
            if !s.result && s.emitted < PREVIEWS {
                actions.push(Action::Accept);
            }
            if !s.result {
                actions.push(Action::Result);
            }
            if s.result && !s.retired {
                actions.push(Action::RetireResult);
            }
            actions.push(Action::Clear);
            if s.restarts < RESTARTS {
                actions.push(Action::Disconnect);
            }
            if !s.completions.is_empty() {
                actions.push(Action::Poll);
            }
            if s.inbox.is_some() && s.active.is_none() {
                actions.push(Action::ReaderTake);
            }
            if s.active.is_some() && s.completions.len() < 2 {
                actions.push(Action::ReaderFinish { ok: true });
                actions.push(Action::ReaderFinish { ok: false });
            }
        }

        fn next_state(&self, last: &State, action: Action) -> Option<State> {
            let mut s = last.clone();
            match action {
                // AppState::progress -> LivePreviews::accept
                Action::Accept => {
                    s.emitted += 1;
                    s.generation += 1;
                    let preview = SolverPreview {
                        session: s.session,
                        generation: s.generation,
                        index: s.emitted,
                    };
                    for token in preview.tokens() {
                        s.issued.insert(token);
                        *s.references.entry(token).or_default() += 1;
                    }
                    s.owners.insert(s.generation, preview);
                    s.desired = Some(s.generation);
                    s.inbox = Some(preview);
                }
                // The job's terminal publishes a handle that owns its
                // preview artifacts; bind_result tracks it if any is owned.
                Action::Result => {
                    s.result = true;
                    s.handle_open = true;
                    if !s.owners.is_empty() {
                        s.handle = Some(false);
                    }
                }
                // AppState::release_handle after the durable save.
                Action::RetireResult => {
                    s.retired = true;
                    match s.handle {
                        Some(_) => s.handle = Some(true),
                        None => self.close(&mut s),
                    }
                }
                Action::Clear => {
                    s.desired = None;
                    s.displayed = None;
                    s.frame = None;
                    s.inbox = None;
                }
                Action::Disconnect => {
                    s.restarts += 1;
                    s.desired = None;
                    s.displayed = None;
                    s.frame = None;
                    s.inbox = None;
                    if self.mutation != Mutation::KeepLedgerOnDisconnect {
                        s.owners.clear();
                        s.references.clear();
                        s.handle = None;
                    }
                    if self.mutation == Mutation::ResetGeneration {
                        s.generation = 0;
                    }
                    // The replacement process starts with no files.
                    s.session += 1;
                    s.issued.clear();
                    s.handle_open = false;
                    s.emitted = 0;
                    s.result = false;
                    s.retired = false;
                }
                // LivePreviews::poll drops each completion's Arc.
                Action::Poll => {
                    while let Some((preview, ok)) = s.completions.pop_front() {
                        if preview.session != s.session {
                            s.seen.insert(Seen::OldReadFinishedAfterRestart);
                        }
                        if Some(preview.generation) != s.desired {
                            continue;
                        }
                        if ok {
                            s.displayed = s.desired;
                            s.frame = Some(preview.session);
                            if preview.session != s.session {
                                s.breaches.insert(Breach::FrameFromOldSession);
                            }
                        } else {
                            s.desired = None;
                        }
                    }
                }
                Action::ReaderTake => s.active = s.inbox.take(),
                Action::ReaderFinish { ok } => {
                    let preview = s.active.take()?;
                    s.completions.push_back((preview, ok));
                }
            }
            if !matches!(action, Action::ReaderTake | Action::ReaderFinish { .. }) {
                self.drain(&mut s);
            }
            Some(s)
        }

        fn properties(&self) -> Vec<Property<Self>> {
            vec![
                Property::always("no token released under a reader", |_, s: &State| {
                    !s.breaches.contains(&Breach::ReleasedUnderReader)
                }),
                Property::always("no handle closed under a reader", |_, s: &State| {
                    !s.breaches.contains(&Breach::ClosedUnderReader)
                }),
                Property::always("each token released once", |_, s: &State| {
                    !s.breaches.contains(&Breach::ReleasedTwice)
                }),
                Property::always("each handle closed once", |_, s: &State| {
                    !s.breaches.contains(&Breach::ClosedTwice)
                }),
                Property::always(
                    "old-session tokens never reach the new worker",
                    |_, s: &State| !s.breaches.contains(&Breach::CrossSessionRelease),
                ),
                Property::always("the frame comes from the live session", |_, s: &State| {
                    !s.breaches.contains(&Breach::FrameFromOldSession)
                }),
                Property::always("at rest an undisplayed ledger is empty", |_, s: &State| {
                    !s.idle()
                        || s.desired.is_some()
                        || s.displayed.is_some()
                        || (s.owners.is_empty() && s.references.is_empty() && s.issued.is_empty())
                }),
                Property::sometimes("a superseded read defers its release", |_, s: &State| {
                    s.seen.contains(&Seen::ReleaseDeferredByRead)
                }),
                Property::sometimes("a read defers its result handle's close", |_, s: &State| {
                    s.seen.contains(&Seen::CloseDeferredByRead)
                }),
                Property::sometimes(
                    "an old-session read finishes after a restart",
                    |_, s: &State| s.seen.contains(&Seen::OldReadFinishedAfterRestart),
                ),
            ]
        }
    }
}

/// Project autosave: `project.rs` `changed`, the coalescing writer thread,
/// `poll` and `flush`, with generation-tagged acknowledgments.
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
fn job_control_watch_and_device_queue() {
    verify(job::Jobs::default());
}

#[test]
fn control_acknowledgment_as_outcome_is_caught() {
    let model = || job::Jobs {
        mutation: job::Mutation::AckIsOutcome,
    };
    counterexample(model(), "paused is shown only on worker evidence");
    counterexample(model(), "an unfinished job stays controllable");
}

#[test]
fn scheduling_behind_a_paused_job_is_caught() {
    counterexample(
        job::Jobs {
            mutation: job::Mutation::PauseFreesScheduler,
        },
        "a paused job keeps the device until its terminal",
    );
}

#[test]
fn control_watch_outliving_its_terminal_is_caught() {
    counterexample(
        job::Jobs {
            mutation: job::Mutation::WatchOutlivesTerminal,
        },
        "no status poll after the terminal",
    );
}

#[test]
fn preview_read_leases_and_deferred_handle_close() {
    verify(live::Live::default());
}

#[test]
fn ignoring_read_leases_is_caught() {
    counterexample(
        live::Live {
            mutation: live::Mutation::IgnoreReadLease,
        },
        "no token released under a reader",
    );
}

#[test]
fn closing_a_handle_under_its_reader_is_caught() {
    counterexample(
        live::Live {
            mutation: live::Mutation::CloseIgnoresOwners,
        },
        "no handle closed under a reader",
    );
}

#[test]
fn keeping_the_token_ledger_across_restarts_is_caught() {
    counterexample(
        live::Live {
            mutation: live::Mutation::KeepLedgerOnDisconnect,
        },
        "old-session tokens never reach the new worker",
    );
}

#[test]
fn restarting_preview_generations_is_caught() {
    counterexample(
        live::Live {
            mutation: live::Mutation::ResetGeneration,
        },
        "the frame comes from the live session",
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
