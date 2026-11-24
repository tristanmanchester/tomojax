//! Captured source context shared by edit-time checks and interactive previews.
//! Filesystem work runs once per source selection on a native I/O task.
use super::*;
use crate::project::CapturedInput;

#[derive(Default)]
pub(super) struct FormContext {
    key: String,
    generation: u64,
    active: Option<String>,
    selected_count: usize,
    inputs: Vec<CapturedInput>,
    pending: bool,
    error: Option<String>,
}

impl FormContext {
    pub(super) fn complete(&mut self, generation: u64, result: Result<Vec<CapturedInput>, String>) {
        if generation != self.generation {
            return;
        }
        self.pending = false;
        match result {
            Ok(inputs) => self.inputs = inputs,
            Err(error) => self.error = Some(error),
        }
    }
}

impl AppState {
    /// The selected assets in project order, or the active asset when nothing is
    /// selected. Preflight and submission both use this order: the first input
    /// sets the geometry or grid for multi-input operations.
    pub(super) fn selected_inputs(&self) -> Vec<String> {
        if self.model.selected_assets.is_empty() {
            return self.model.active_asset.iter().cloned().collect();
        }
        self.project.as_ref().map_or_else(Vec::new, |project| {
            project
                .document
                .assets
                .iter()
                .filter(|asset| self.model.selected_assets.contains(&asset.id))
                .map(|asset| asset.id.clone())
                .collect()
        })
    }

    pub(super) fn poll_form_context(&mut self) -> bool {
        let mut selected = self.selected_inputs();
        let Some(project) = &self.project else {
            return false;
        };
        let selected_count = selected.len();
        if let Some(active) = &self.model.active_asset
            && !selected.contains(active)
        {
            selected.push(active.clone());
        }
        let assets: Vec<_> = selected
            .iter()
            .filter_map(|id| project.document.asset(id).cloned())
            .collect();
        let key = json!({
            "root": project.root,
            "active": self.model.active_asset,
            "selected": selected,
            "sources": assets.iter().map(|asset| json!({
                "id": asset.id,
                "path": asset.path,
                "kind": asset.kind,
                "arrays": asset.arrays,
                "open": asset.open_parameters,
            })).collect::<Vec<_>>(),
        })
        .to_string();
        if key == self.form_context.key {
            return false;
        }
        let generation = self.form_context.generation + 1;
        self.form_context = FormContext {
            key,
            generation,
            active: self.model.active_asset.clone(),
            selected_count,
            pending: !assets.is_empty(),
            ..Default::default()
        };
        if !assets.is_empty() {
            let root = project.root.clone();
            self.spawn_local(None, move |cancelled| {
                let result = assets
                    .iter()
                    .map(|asset| {
                        check_cancel(&cancelled)?;
                        CapturedInput::from_asset(asset, &root)
                    })
                    .collect();
                LocalEvent::FormCaptured { generation, result }
            });
        }
        true
    }

    pub(super) fn current_form_input(&self) -> Option<CapturedInput> {
        self.form_context
            .inputs
            .iter()
            .find(|input| Some(&input.asset) == self.form_context.active.as_ref())
            .cloned()
    }

    pub(super) fn current_form_inputs(&self) -> &[CapturedInput] {
        &self.form_context.inputs[..self
            .form_context
            .selected_count
            .min(self.form_context.inputs.len())]
    }

    pub(super) fn form_input_error(&self) -> Option<&str> {
        self.form_context.error.as_deref()
    }

    pub(super) fn form_inputs_pending(&self) -> bool {
        self.form_context.pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_source_capture_cannot_replace_a_new_selection() {
        let mut context = FormContext {
            generation: 2,
            pending: true,
            ..Default::default()
        };
        context.complete(1, Err("old source missing".into()));
        assert!(context.pending);
        assert!(context.error.is_none());
        context.complete(2, Ok(Vec::new()));
        assert!(!context.pending);
    }
}
