//! Locking for state that stays usable after a holder panics.
use std::sync::{Mutex, MutexGuard, PoisonError};

pub(crate) trait Locked<T> {
    /// Lock the mutex, taking its value as it stands if an earlier holder panicked.
    fn locked(&self) -> MutexGuard<'_, T>;
}

impl<T> Locked<T> for Mutex<T> {
    fn locked(&self) -> MutexGuard<'_, T> {
        self.lock().unwrap_or_else(PoisonError::into_inner)
    }
}
