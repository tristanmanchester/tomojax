//! Handles and image coordinates belong to exactly one incarnation of the executor.
use super::{ledger::Ledger, view::Placements};
use anyhow::Result;
use serde_json::{Value, json};
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicU64, Ordering},
};

#[derive(Debug, Clone)]
pub(super) struct Lost(String);
impl std::fmt::Display for Lost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for Lost {}

#[derive(Default)]
pub(super) struct Namespace {
    pub ledger: Mutex<Ledger>,
    pub placements: Mutex<Placements>,
    pub generation: u64,
    failure: OnceLock<Lost>,
}
impl Namespace {
    pub fn valid(&self) -> Result<()> {
        if let Some(e) = self.failure.get() {
            return Err(e.clone().into());
        }
        Ok(())
    }
}
#[derive(Default)]
pub(super) struct History {
    next: AtomicU64,
    pub renders: AtomicU64,
    report: Mutex<Value>,
}
impl History {
    pub fn namespace(&self) -> Arc<Namespace> {
        Arc::new(Namespace {
            generation: self.next.fetch_add(1, Ordering::Relaxed) + 1,
            ..Default::default()
        })
    }
    pub fn report(&self) -> Value {
        self.report.lock().unwrap().clone()
    }
    pub fn retire(&self, space: &Namespace, reason: &str) -> Lost {
        space
            .failure
            .get_or_init(|| {
                let lost = space
                    .ledger
                    .lock()
                    .unwrap()
                    .records
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>();
                *self.report.lock().unwrap() = json!({"reason":reason,"lost":lost});
                let names = if lost.is_empty() {
                    "none".into()
                } else {
                    lost.join(", ")
                };
                Lost(format!(
                    "{reason}. Lost references: {names}. Open inputs again."
                ))
            })
            .clone()
    }
}
