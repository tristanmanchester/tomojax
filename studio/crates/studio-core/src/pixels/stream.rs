// Ported from T3 Code (MIT), packages/client-runtime/src/preview/serverBrowserStream.ts and apps/server/src/preview/ServerBrowser.ts at d2c44031; see THIRD_PARTY_NOTICES.md.
use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, PoisonError};

use super::Encoding;
use crate::sync::Locked;

type Backlog<T> = VecDeque<(Option<u32>, T)>;

/// One decode at a time. Each pane can replace its waiting moving frame.
/// Reliable items evict a moving frame, or wait for space. Closing wakes both ends.
pub struct FrameQueue<T> {
    capacity: usize,
    state: Mutex<Option<Backlog<T>>>,
    changed: Condvar,
}

impl<T> FrameQueue<T> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0);
        Self {
            capacity,
            state: Mutex::new(Some(VecDeque::new())),
            changed: Condvar::new(),
        }
    }

    pub fn push(&self, moving_slot: Option<u32>, value: T) {
        let mut state = self.state.locked();
        while let Some(items) = state.as_mut() {
            if moving_slot.is_some()
                && let Some(i) = items.iter().position(|(slot, _)| *slot == moving_slot)
            {
                items.remove(i);
            }
            if items.len() == self.capacity {
                if let Some(i) = items.iter().position(|(slot, _)| slot.is_some()) {
                    items.remove(i);
                } else if moving_slot.is_some() {
                    return;
                } else {
                    state = self
                        .changed
                        .wait(state)
                        .unwrap_or_else(PoisonError::into_inner);
                    continue;
                }
            }
            items.push_back((moving_slot, value));
            self.changed.notify_all();
            return;
        }
    }

    pub fn pop(&self) -> Option<T> {
        let mut state = self.state.locked();
        while let Some(items) = state.as_mut() {
            if let Some((_, value)) = items.pop_front() {
                self.changed.notify_all();
                return Some(value);
            }
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        None
    }

    pub fn close(&self) {
        *self.state.locked() = None;
        self.changed.notify_all();
    }
}

/// A request paces the next moving frame. A full-resolution transfer may be interrupted.
#[derive(Default)]
pub enum PaneStream {
    #[default]
    Idle,
    Waiting(u64, Encoding),
    Received(u64, Encoding),
}

impl PaneStream {
    pub fn next(&self, wanted: u64) -> Option<Encoding> {
        match *self {
            Self::Waiting(_, Encoding::Jpeg) => None,
            Self::Waiting(g, _) | Self::Received(g, Encoding::Float32) if g == wanted => None,
            Self::Received(g, Encoding::Jpeg) if g == wanted => Some(Encoding::Float32),
            _ => Some(Encoding::Jpeg),
        }
    }

    pub fn receive(&mut self, generation: u64, encoding: Encoding) -> bool {
        if matches!(*self, Self::Waiting(g,e) if (g,e) == (generation,encoding)) {
            *self = Self::Received(generation, encoding);
            true
        } else {
            false
        }
    }
}
