//! The selected inputs as the open form sees them, shared by edit-time checks,
//! interactive previews and submission.
use super::*;
use crate::project::CapturedInput;

#[derive(Default)]
pub(super) struct FormContext {
    key: String,
    active: Option<String>,
    selected_count: usize,
    inputs: Vec<CapturedInput>,
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
        self.form_context = FormContext {
            key,
            active: self.model.active_asset.clone(),
            selected_count,
            inputs: assets
                .iter()
                .map(|asset| CapturedInput::from_asset(asset, project.data_root()))
                .collect(),
        };
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
}
