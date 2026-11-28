//! Exclusive use of one executor generation by a local job script.
use crate::Event;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::io;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
    mpsc::{self, Receiver, Sender, SyncSender},
};
use std::time::Duration;

#[derive(Clone)]
pub struct Scripts {
    events: SyncSender<Event>,
    ticket: Arc<AtomicU64>,
}

pub struct Stream {
    ticket: u64,
    events: SyncSender<Event>,
    replies: Receiver<Value>,
    first: Option<Value>,
}

pub(crate) enum Action {
    Acquire(u64, Sender<Value>),
    Send(u64, Value),
    Release(u64),
}

impl Scripts {
    pub(crate) fn new(events: SyncSender<Event>) -> Self {
        Self {
            events,
            ticket: Arc::new(AtomicU64::new(0)),
        }
    }
    pub fn acquire(&self, stopping: &dyn Fn() -> bool) -> io::Result<Stream> {
        let (send, replies) = mpsc::channel();
        let ticket = self.ticket.fetch_add(1, Ordering::Relaxed);
        self.events
            .send(Event::Inline(Action::Acquire(ticket, send)))
            .map_err(io::Error::other)?;
        let mut stream = Stream {
            ticket,
            events: self.events.clone(),
            replies,
            first: None,
        };
        // The ready belongs to the acquired generation and remains the first reply.
        loop {
            if stopping() {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "job canceled while waiting for the executor",
                ));
            }
            match stream.receive(Duration::from_millis(100))? {
                None => continue,
                Some(ready) => {
                    if ready["type"] != "ready" {
                        return Err(io::Error::other("executor ended before script admission"));
                    }
                    stream.first = Some(ready);
                    break;
                }
            }
        }
        Ok(stream)
    }
}

impl Stream {
    pub fn send(&mut self, line: &Value) -> io::Result<()> {
        self.events
            .send(Event::Inline(Action::Send(self.ticket, line.clone())))
            .map_err(io::Error::other)
    }

    pub fn receive(&mut self, timeout: Duration) -> io::Result<Option<Value>> {
        if let Some(ready) = self.first.take() {
            return Ok(Some(ready));
        }
        match self.replies.recv_timeout(timeout) {
            Ok(line) => Ok(Some(line)),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(e) => Err(io::Error::new(io::ErrorKind::BrokenPipe, e)),
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let _ = self
            .events
            .send(Event::Inline(Action::Release(self.ticket)));
    }
}

#[derive(Default)]
pub(crate) struct Queue {
    waiting: VecDeque<(u64, Sender<Value>)>,
    active: Option<(u64, Sender<Value>)>,
    // A script that drops early still owns its outstanding replies.
    requests: HashMap<String, (u64, Sender<Value>)>,
}

impl Queue {
    pub fn idle(&self) -> bool {
        self.active.is_none() && self.waiting.is_empty() && self.requests.is_empty()
    }
    pub fn wait(&mut self, id: u64, replies: Sender<Value>) {
        self.waiting.push_back((id, replies));
    }
    pub fn grant(&mut self, ready: &Value) {
        if self.active.is_none() && self.requests.is_empty() {
            while let Some((id, reply)) = self.waiting.pop_front() {
                if reply.send(ready.clone()).is_ok() {
                    self.active = Some((id, reply));
                    break;
                }
            }
        }
    }
    pub fn release(&mut self, ticket: u64) {
        if self.active.as_ref().is_some_and(|(id, _)| *id == ticket) {
            self.active = None;
        }
        self.waiting.retain(|(id, _)| *id != ticket);
    }
    pub fn submit(&mut self, ticket: u64, id: &str) -> Option<Sender<Value>> {
        let (owner, tx) = self.active.as_ref()?;
        if *owner != ticket {
            return None;
        }
        self.requests.insert(id.into(), (ticket, tx.clone()));
        Some(tx.clone())
    }
    pub fn reply(&mut self, id: &str, row: &Value, terminal: bool) -> bool {
        if let Some((_, send)) = self.requests.get(id) {
            let _ = send.send(row.clone());
            if terminal {
                self.requests.remove(id);
            }
            true
        } else {
            false
        }
    }
}
