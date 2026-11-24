use std::collections::{BTreeMap, BTreeSet};

use crate::project::CapturedInput;
use crate::protocol::{Output, Product, Storage};

pub(super) enum SourceState {
    Waiting,
    Opening(String),
    Ready(Box<Output>),
    Failed(String),
}

pub(super) struct Source {
    pub input: CapturedInput,
    pub state: SourceState,
    touched: u64,
}

#[derive(Default)]
pub(super) struct Sources {
    pub entries: BTreeMap<String, Source>,
    retained: BTreeSet<String>,
    clock: u64,
}

pub(super) fn key(input: &CapturedInput) -> Result<String, String> {
    let bytes = serde_json::to_vec(input).map_err(|error| error.to_string())?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

impl Sources {
    pub fn ensure(&mut self, input: CapturedInput) -> Result<String, String> {
        let key = key(&input)?;
        self.clock += 1;
        let source = self.entries.entry(key.clone()).or_insert(Source {
            input,
            state: SourceState::Waiting,
            touched: self.clock,
        });
        source.touched = self.clock;
        Ok(key)
    }

    pub fn retain(&mut self, inputs: &[CapturedInput]) -> Result<(), String> {
        let keys = inputs.iter().map(key).collect::<Result<_, _>>()?;
        for input in inputs {
            self.ensure(input.clone())?;
        }
        self.retained = keys;
        Ok(())
    }

    pub fn product(&self, input: &CapturedInput) -> Result<Option<&Product>, String> {
        match self.entries.get(&key(input)?).map(|source| &source.state) {
            Some(SourceState::Ready(output)) => Ok(Some(&output.value)),
            Some(SourceState::Failed(error)) => Err(error.clone()),
            _ => Ok(None),
        }
    }

    pub fn source_ref(&self, input: &CapturedInput) -> Result<Option<Storage>, String> {
        let Some(product) = self.product(input)? else {
            return Ok(None);
        };
        if let Some(source) = product.scan_source() {
            return Ok(Some(source));
        }
        if let Product::Frames { handle, .. } = product {
            return Ok(Some(Storage::Member {
                handle: handle.clone(),
                member: "self".into(),
            }));
        }
        Err(
            "This result has no source scan. Select its original acquisition for processing."
                .into(),
        )
    }

    /// Explicitly retained sources and active preview inputs are needed work;
    /// retain at most two other handles for returning to a recent acquisition.
    pub fn trim(&mut self, preview_inputs: &BTreeSet<String>) -> Vec<Source> {
        let mut idle: Vec<_> = self
            .entries
            .iter()
            .filter(|(key, _)| !self.retained.contains(*key) && !preview_inputs.contains(*key))
            .map(|(key, source)| (source.touched, key.clone()))
            .collect();
        idle.sort_unstable();
        let remove = idle.len().saturating_sub(2);
        idle.into_iter()
            .take(remove)
            .filter_map(|(_, key)| self.entries.remove(&key))
            .collect()
    }

    pub fn clear(&mut self) -> Vec<Source> {
        self.retained.clear();
        std::mem::take(&mut self.entries).into_values().collect()
    }
}
