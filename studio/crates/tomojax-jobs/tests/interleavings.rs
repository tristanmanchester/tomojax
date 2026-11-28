//! Bounded explicit-state search. File observations are interned by their on-disk contents.
//! An independent reader can take each of its three steps at any node on a writer path.
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::Path;
use tomojax_jobs::directory::{Submission, create_record, read_files};
use tomojax_jobs::state::{
    Account, Files, Launch, Outcome, State, matches_allocation, needs_withdrawal, observed, state,
};

const OURS: &str = "tj-one";
const ID: u64 = 17;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
enum Queue {
    #[default]
    Empty,
    Pending,
    Allocated,
    Gone,
    Reused,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
enum Submit {
    #[default]
    New,
    Claimed,
    Checked,
    Reply,
    Done,
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
struct World {
    files: Files,
    submit: Submit,
    queue: Queue,
    runner_exited: bool,
    // The batch shell exited 0: the runner returned after recording `completed`.
    finished: bool,
    requeued: bool,
    accounting: bool,
    // 0 connected, 1 offline, 2 reconnected
    network: u8,
    // A poll can pause after identifying a job but before withdrawing it.
    withdrawing: bool,
}

impl World {
    fn account(self) -> Account {
        if self.network == 1 {
            return Account::Offline;
        }
        let name = if self.queue == Queue::Reused {
            "unrelated"
        } else {
            OURS
        };
        if matches_allocation(OURS, self.files.launch, name, ID) {
            match self.queue {
                Queue::Pending => return Account::Pending,
                Queue::Allocated => return Account::Allocated,
                Queue::Reused => return Account::Allocated,
                _ => {}
            }
        }
        if self.accounting && self.finished {
            return Account::Finished;
        }
        Account::Absent {
            accounted: self.accounting,
        }
    }

    fn next(self) -> Vec<Self> {
        let mut next = Vec::new();
        self.submit_steps(&mut next);
        self.scheduler_steps(&mut next);
        if self.network < 2 {
            next.push(Self {
                network: self.network + 1,
                ..self
            });
        }
        if !self.files.cancel {
            next.push(self.files(|f| f.cancel = true));
        }
        if self.files.cancel
            && !self.files.abandoned
            && state(self.files, self.account()) == State::Unknown
        {
            next.push(self.files(|f| f.abandoned = true));
        }
        if needs_withdrawal(self.files, self.account()) && !self.withdrawing {
            next.push(Self {
                withdrawing: true,
                ..self
            });
        }
        if self.withdrawing && self.network != 1 {
            // The selector travels to the controller, which checks it at signal delivery.
            let mut withdrawn = self;
            if matches!(
                self.queue,
                Queue::Pending | Queue::Allocated | Queue::Reused
            ) {
                let name = if self.queue == Queue::Reused {
                    "unrelated"
                } else {
                    OURS
                };
                if matches_allocation(OURS, self.files.launch, name, ID) {
                    assert_ne!(
                        self.queue,
                        Queue::Reused,
                        "withdrew an unrelated allocation"
                    );
                    withdrawn.queue = Queue::Gone;
                    withdrawn.files.withdrawn = true;
                    withdrawn.runner_exited = true;
                }
            }
            withdrawn.withdrawing = false;
            next.push(withdrawn);
        }
        next
    }

    fn files(self, change: impl FnOnce(&mut Files)) -> Self {
        let mut world = self;
        change(&mut world.files);
        world
    }

    fn submit_steps(self, next: &mut Vec<Self>) {
        let mut w = self;
        match self.submit {
            Submit::New => {
                w.files.launch = Launch::Claimed;
                w.submit = Submit::Claimed;
            }
            Submit::Claimed if self.files.cancel => {
                w.files.launch = Launch::Prevented;
                w.submit = Submit::Done;
            }
            Submit::Claimed => w.submit = Submit::Checked,
            Submit::Checked => {
                assert_eq!(self.queue, Queue::Empty, "launched twice");
                w.queue = Queue::Pending;
                w.submit = Submit::Reply;
                let mut rejected = self;
                rejected.files.launch = Launch::Rejected;
                rejected.submit = Submit::Done;
                next.push(rejected);
            }
            Submit::Reply => {
                w.files.launch = Launch::Accepted(ID);
                w.submit = Submit::Done;
            }
            Submit::Done => return,
        }
        next.push(w);
        // Death before sbatch, or loss of the reply after scheduler acceptance.
        if matches!(
            self.submit,
            Submit::Claimed | Submit::Checked | Submit::Reply
        ) {
            next.push(Self {
                submit: Submit::Done,
                ..self
            });
        }
    }

    fn scheduler_steps(self, next: &mut Vec<Self>) {
        let f = self.files;
        match self.queue {
            Queue::Pending => {
                next.push(Self {
                    queue: Queue::Allocated,
                    runner_exited: false,
                    ..self
                });
            }
            Queue::Allocated if !self.runner_exited => {
                if !f.runner {
                    next.push(self.files(|f| f.runner = true));
                } else if f.attempt == u64::from(self.requeued) || f.attempt == 0 {
                    next.push(self.files(|f| {
                        f.attempt += 1;
                        f.end = None;
                    }));
                } else if f.end.is_none() {
                    if f.receipt {
                        next.push(self.files(|f| f.end = Some(Outcome::Completed)));
                    } else {
                        next.push(self.files(|f| f.end = Some(Outcome::Stopped)));
                        if !f.cancel {
                            next.push(self.files(|f| f.end = Some(Outcome::Failed)));
                            next.push(self.files(|f| f.receipt = true));
                        }
                    }
                }
                // SIGKILL, a normal exit after end, or the runner dying before entry.
                next.push(Self {
                    runner_exited: true,
                    ..self
                });
                if f.end == Some(Outcome::Completed) {
                    next.push(Self {
                        runner_exited: true,
                        finished: true,
                        ..self
                    });
                }
            }
            _ => {}
        }
        if self.queue == Queue::Pending || (self.queue == Queue::Allocated && self.runner_exited) {
            next.push(Self {
                queue: Queue::Gone,
                runner_exited: true,
                ..self
            });
        }
        if matches!(self.queue, Queue::Allocated | Queue::Gone)
            && self.runner_exited
            && !self.requeued
            && f.attempt == 1
            && !f.receipt
        {
            // Batch shell requeue, node loss, or an administrator's requeuehold after failure.
            next.push(Self {
                queue: Queue::Pending,
                requeued: true,
                ..self
            });
        }
        if matches!(self.queue, Queue::Gone | Queue::Reused) && !self.accounting {
            next.push(Self {
                accounting: true,
                ..self
            });
        }
        if self.queue == Queue::Gone {
            next.push(Self {
                queue: Queue::Reused,
                ..self
            });
        }
    }

    // Oracle uses the actual actor's location, without applying the priority table.
    fn check(self, result: State) {
        let f = self.files;
        assert_eq!(result == State::Completed, f.receipt, "{self:?}");
        if f.receipt {
            return;
        }
        if self.queue != Queue::Empty {
            assert_ne!(result, State::NotLaunched);
        }
        if self.network != 1 {
            match self.queue {
                Queue::Pending => assert_eq!(result, State::Queued, "{self:?}"),
                Queue::Allocated => assert_eq!(result, State::Running, "{self:?}"),
                Queue::Reused => assert!(!matches!(result, State::Queued | State::Running)),
                _ => {}
            }
        }
        if f.launch == Launch::Prevented {
            assert_eq!(result, State::Canceled);
        }
        if self.queue == Queue::Empty && self.submit == Submit::Done && f.launch == Launch::Claimed
        {
            assert_eq!(
                result,
                if f.abandoned {
                    State::Canceled
                } else {
                    State::Unknown
                }
            );
        }
        if self.submit == Submit::Checked && f.cancel && !f.abandoned {
            assert_eq!(result, State::Unknown, "sbatch can still run");
        }
        if matches!(self.queue, Queue::Gone | Queue::Reused) && self.network != 1 {
            if f.end == Some(Outcome::Failed) {
                assert_eq!(result, State::Failed);
            } else if f.cancel && (self.accounting || f.withdrawn || f.runner) {
                assert_eq!(result, State::Canceled);
            }
        }
    }
}

fn fixture(path: &Path, facts: Files) -> Files {
    use serde_json::json;
    fs::create_dir(path).unwrap();
    if facts.launch != Launch::Unclaimed {
        assert!(create_record(&path.join("launch"), &OURS).unwrap());
        assert!(!create_record(&path.join("launch"), &"duplicate").unwrap());
    }
    let submission = match facts.launch {
        Launch::Accepted(id) => Some(Submission::Accepted(id)),
        Launch::Rejected => Some(Submission::Rejected("rejected".into())),
        Launch::Prevented => Some(Submission::Prevented),
        _ => None,
    };
    if let Some(record) = submission {
        create_record(&path.join("submission.json"), &record).unwrap();
    }
    for (present, name) in [
        (facts.cancel, "cancel"),
        (facts.abandoned, "abandoned"),
        (facts.runner, "runner.json"),
        (facts.withdrawn, "withdrawn"),
    ] {
        if present {
            fs::write(path.join(name), b"{}").unwrap();
        }
    }
    if facts.receipt {
        fs::create_dir(path.join("result")).unwrap();
        fs::write(path.join("result/receipt.json"), b"{}").unwrap();
    }
    let mut log = String::new();
    for number in 1..=facts.attempt {
        log.push_str(&format!("{}\n", json!({"type":"attempt","number":number})));
    }
    if let Some(outcome) = facts.end {
        log.push_str(&format!("{}\n", json!({"type":"end","outcome":outcome})));
    }
    fs::write(path.join("replies.jsonl"), log).unwrap();
    read_files(path).unwrap()
}

#[derive(Default)]
struct Node {
    next: Vec<usize>,
    parents: Vec<usize>,
}

#[test]
fn every_bounded_interleaving() {
    let root = tempfile::tempdir().unwrap();
    let mut views = HashMap::new();
    let mut worlds = vec![World::default()];
    let mut nodes = vec![Node::default()];
    let mut index = HashMap::from([(worlds[0], 0)]);
    let mut pos = 0;
    while pos < worlds.len() {
        let w = worlds[pos];
        let n = views.len();
        let files = *views
            .entry(w.files)
            .or_insert_with(|| fixture(&root.path().join(n.to_string()), w.files));
        assert_eq!(files, w.files);
        w.check(state(files, w.account()));
        // A reconnecting worker carries out the standing request even if the
        // submitter died before recording the ID, or the previous poll went offline.
        if files.cancel && matches!(w.queue, Queue::Pending | Queue::Allocated) {
            let online = World {
                network: 2,
                withdrawing: false,
                ..w
            };
            let started = online
                .next()
                .into_iter()
                .find(|child| child.withdrawing)
                .unwrap();
            let done = started
                .next()
                .into_iter()
                .find(|child| {
                    child.files.withdrawn && child.queue == Queue::Gone && !child.withdrawing
                })
                .unwrap();
            let terminal = state(done.files, done.account());
            assert!(matches!(
                terminal,
                State::Canceled | State::Completed | State::Failed
            ));
            assert_eq!(
                terminal == State::Failed,
                !files.receipt && files.end == Some(Outcome::Failed)
            );
        }
        for child in w.next() {
            let i = *index.entry(child).or_insert_with(|| {
                worlds.push(child);
                nodes.push(Node::default());
                worlds.len() - 1
            });
            nodes[pos].next.push(i);
            nodes[i].parents.push(pos);
        }
        pos += 1;
    }
    // Propagate the possible first observations forward. For each lookup, propagate
    // the possible last observations backward. This is exactly the product with one
    // three-step reader, factoring paths with identical observations.
    let order = topological(&nodes);
    let before = reachable(&order, &nodes, &worlds, |n| &n.next);
    let reverse: Vec<_> = order.iter().copied().rev().collect();
    let after = reachable(&reverse, &nodes, &worlds, |n| &n.parents);
    let mut triples = HashSet::new();
    let mut late = HashSet::new();
    let mut schedules = 0_u64;
    for (i, w) in worlds.iter().enumerate() {
        for first in &before[i] {
            for last in &after[i] {
                schedules += 1;
                let oracle = (*first == *last).then_some(state(w.files, w.account()));
                triples.insert((*first, w.account(), *last, oracle));
            }
        }
    }
    // The runner's files (its record, reply log and result folder) are written on the
    // compute node and can reach a reader elsewhere late, as Diamond's NFS view of GPFS
    // showed: as of any earlier moment, and the renamed result folder even after the end
    // record. Files that workers write are current. Both observations see the same view.
    for (i, w) in worlds.iter().enumerate() {
        for seen in &before[i] {
            for hidden in [false, true] {
                if hidden && !seen.receipt {
                    continue;
                }
                let view = Files {
                    runner: seen.runner,
                    attempt: seen.attempt,
                    end: seen.end,
                    receipt: seen.receipt && !hidden,
                    ..w.files
                };
                late.insert((view, w.account()));
            }
        }
    }
    for (view, lookup) in &late {
        let n = views.len();
        let read = *views
            .entry(*view)
            .or_insert_with(|| fixture(&root.path().join(n.to_string()), *view));
        assert_eq!(read, *view);
        let result = observed(*view, *lookup, *view).unwrap();
        assert_eq!(
            result == State::Completed,
            view.receipt,
            "{view:?} {lookup:?}"
        );
        // The launcher reports success, or the runner recorded completion, only after
        // the result was published: the job finished.
        if *lookup == Account::Finished || view.end == Some(Outcome::Completed) {
            assert!(
                !matches!(result, State::Interrupted | State::Canceled | State::Failed),
                "a finished job read {result:?} before its files were visible: {view:?} {lookup:?}"
            );
        }
    }
    for (first, lookup, last, oracle) in &triples {
        assert_eq!(observed(*first, *lookup, *last), *oracle);
    }
    eprintln!(
        "{} writer states, {} disk views, {} reader placements, {} distinct reader inputs with oracle, {} with the runner's files late",
        worlds.len(),
        views.len(),
        schedules,
        triples.len(),
        late.len()
    );
}

fn topological(nodes: &[Node]) -> Vec<usize> {
    let mut degree: Vec<_> = nodes.iter().map(|n| n.parents.len()).collect();
    let mut todo = VecDeque::from([0]);
    let mut order = Vec::new();
    while let Some(i) = todo.pop_front() {
        order.push(i);
        for j in &nodes[i].next {
            degree[*j] -= 1;
            if degree[*j] == 0 {
                todo.push_back(*j);
            }
        }
    }
    assert_eq!(order.len(), nodes.len(), "model bounds must give a DAG");
    order
}

fn reachable(
    order: &[usize],
    nodes: &[Node],
    worlds: &[World],
    edges: fn(&Node) -> &[usize],
) -> Vec<HashSet<Files>> {
    let mut sets: Vec<_> = worlds.iter().map(|w| HashSet::from([w.files])).collect();
    for &i in order {
        let seen = sets[i].clone();
        for j in edges(&nodes[i]) {
            sets[*j].extend(&seen);
        }
    }
    sets
}
