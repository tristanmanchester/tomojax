// Ported from T3 Code (MIT), packages/client-runtime/src/connection/registry.ts at d2c44031. See THIRD_PARTY_NOTICES.md.
use super::Owner;
use crate::engine::{EngineEvent, Wake};
use std::{collections::BTreeMap, path::PathBuf, time::Instant};

/// The client scopes each owner by connection name and retains it across drops.
pub struct Registry {
    worker: PathBuf,
    wake: Wake,
    required: &'static [&'static str],
    connections: BTreeMap<String, Owner>,
}
impl Registry {
    pub fn new(worker: PathBuf, required: &'static [&'static str], wake: Wake) -> Self {
        Self {
            worker,
            wake,
            required,
            connections: BTreeMap::new(),
        }
    }
    pub fn open(&mut self, name: &str) -> &mut Owner {
        self.connections
            .entry(name.to_owned())
            .or_insert_with_key(|key| {
                Owner::new(
                    self.worker.clone(),
                    key.clone(),
                    self.required,
                    self.wake.clone(),
                )
            })
    }
    pub fn remove(&mut self, name: &str) {
        self.connections.remove(name);
    }
    pub fn next_wake(&self) -> Option<Instant> {
        self.connections.values().filter_map(Owner::next_wake).min()
    }
    pub fn poll(&mut self) -> Vec<(String, EngineEvent)> {
        self.connections
            .iter_mut()
            .flat_map(|(name, connection)| {
                connection
                    .poll()
                    .into_iter()
                    .map(|event| (name.clone(), event))
            })
            .collect()
    }
}
