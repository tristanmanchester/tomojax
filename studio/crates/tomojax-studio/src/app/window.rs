//! The display window: the range the renderer uses and the two limit fields that edit it.
//!
//! A field is untouched while it still shows what Studio last wrote into it. An untouched
//! field stands for the exact limit behind that text and follows the window on screen; an
//! edited field is the user's draft and stays until they apply, reset or leave the image.
use rune::TextEdit;
use std::collections::BTreeMap;

/// What Studio last wrote into a limit field.
#[derive(Clone, Debug, Default, PartialEq)]
enum Written {
    /// Nothing: no window was on screen.
    #[default]
    Blank,
    /// `text`, a reading of the exact limit `exact`.
    Limit { text: String, exact: f32 },
}

impl Written {
    fn text(&self) -> &str {
        match self {
            Self::Blank => "",
            Self::Limit { text, .. } => text,
        }
    }
}

/// What a limit field holds now.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Reading<'a> {
    /// Untouched and blank: the automatic window.
    Blank,
    /// Untouched, standing for this exact limit.
    Limit(f32),
    /// Edited by the user, read as typed.
    Draft(&'a str),
}

#[derive(Clone, Debug, Default)]
struct Field {
    edit: TextEdit,
    written: Written,
}

impl Field {
    fn reading(&self) -> Reading<'_> {
        match &self.written {
            Written::Blank if self.edit.text.trim().is_empty() => Reading::Blank,
            Written::Limit { text, exact } if *text == self.edit.text => Reading::Limit(*exact),
            Written::Blank | Written::Limit { .. } => Reading::Draft(&self.edit.text),
        }
    }

    fn write(&mut self, written: Written) {
        self.edit.replace_text(written.text());
        self.written = written;
    }
}

/// The lower and upper limit fields.
const SIDES: [&str; 2] = ["low", "high"];

#[derive(Clone, Debug, Default)]
pub(crate) struct DisplayWindow {
    /// The committed window; `None` is the automatic window of the image on screen.
    applied: Option<[f32; 2]>,
    fields: [Field; 2],
}

impl DisplayWindow {
    pub(crate) fn applied(&self) -> Option<[f32; 2]> {
        self.applied
    }

    /// Field `side` (0 lower, 1 upper), for the view to show.
    pub(crate) fn field(&self, side: usize) -> &TextEdit {
        &self.fields[side].edit
    }

    /// Field `side`, for the view to edit.
    pub(crate) fn field_mut(&mut self, side: usize) -> &mut TextEdit {
        &mut self.fields[side].edit
    }

    pub(crate) fn reading(&self, side: usize) -> Reading<'_> {
        self.fields[side].reading()
    }

    /// Commits `window` (`None`, the automatic window) and shows it in both fields,
    /// dropping any draft.
    pub(crate) fn restore(&mut self, window: Option<[f32; 2]>) {
        self.applied = window;
        let written = match window {
            Some(window) => {
                let [low, high] = window_text(window[0], window[1]);
                [
                    Written::Limit {
                        text: low,
                        exact: window[0],
                    },
                    Written::Limit {
                        text: high,
                        exact: window[1],
                    },
                ]
            }
            None => [Written::Blank, Written::Blank],
        };
        for (field, written) in self.fields.iter_mut().zip(written) {
            field.write(written);
        }
    }

    /// Untouched fields show `shown`, the window on screen; drafts stay as typed.
    pub(crate) fn follow(&mut self, shown: Option<[f32; 2]>) {
        let Some(shown) = shown else {
            return;
        };
        let text = window_text(shown[0], shown[1]);
        for (side, field) in self.fields.iter_mut().enumerate() {
            match field.reading() {
                Reading::Blank | Reading::Limit(_) => field.write(Written::Limit {
                    text: text[side].clone(),
                    exact: shown[side],
                }),
                Reading::Draft(_) => {}
            }
        }
    }

    /// The window the fields describe: `None` when both are blank (automatic).
    pub(crate) fn parse(&self) -> Result<Option<[f32; 2]>, String> {
        let limit = |side: usize| -> Result<Option<f32>, String> {
            let missing = || format!("Enter a finite {} window limit", ["lower", "upper"][side]);
            match self.reading(side) {
                Reading::Blank => Ok(None),
                Reading::Limit(exact) => Ok(Some(exact)),
                Reading::Draft(text) if text.trim().is_empty() => Ok(None),
                Reading::Draft(text) => text.trim().parse().map(Some).map_err(|_| missing()),
            }
        };
        let window = match (limit(0)?, limit(1)?) {
            (None, None) => return Ok(None),
            (Some(low), Some(high)) => [low, high],
            (None, Some(_)) => return Err("Enter a finite lower window limit".into()),
            (Some(_), None) => return Err("Enter a finite upper window limit".into()),
        };
        if !window.iter().all(|limit| limit.is_finite()) || window[0] >= window[1] {
            return Err("Display window limits must be finite, with minimum below maximum".into());
        }
        Ok(Some(window))
    }

    /// Commits what the fields describe; an invalid draft changes nothing.
    pub(crate) fn apply(&mut self) -> Result<(), String> {
        let window = self.parse()?;
        self.restore(window);
        Ok(())
    }

    /// Records the fields in a project draft, so reopening keeps drafts and exact limits.
    pub(crate) fn save(
        &self,
        edits: &mut BTreeMap<String, String>,
        choices: &mut BTreeMap<String, String>,
    ) {
        for (side, field) in SIDES.iter().zip(&self.fields) {
            edits.insert(format!("studio:window_{side}"), field.edit.text.clone());
            choices.insert(
                format!("studio:prefill_{side}"),
                field.written.text().to_owned(),
            );
            if let Written::Limit { exact, .. } = field.written {
                choices.insert(format!("studio:exact_{side}"), exact.to_string());
            }
        }
    }

    /// Restores the fields [`DisplayWindow::save`] recorded, over the committed window.
    pub(crate) fn load(
        &mut self,
        edits: &BTreeMap<String, String>,
        choices: &BTreeMap<String, String>,
    ) {
        for (side, field) in SIDES.iter().zip(&mut self.fields) {
            if let Some(text) = choices.get(&format!("studio:prefill_{side}")) {
                field.written = match choices
                    .get(&format!("studio:exact_{side}"))
                    .and_then(|exact| exact.parse().ok())
                {
                    Some(exact) => Written::Limit {
                        text: text.clone(),
                        exact,
                    },
                    None => Written::Blank,
                };
            }
            if let Some(text) = edits.get(&format!("studio:window_{side}")) {
                field.edit.replace_text(text.clone());
            }
        }
    }
}

/// A display window as field text: rounded for reading while the rounded bounds stay in
/// order, else the shortest text that reads back as each exact bound.
pub(crate) fn window_text(low: f32, high: f32) -> [String; 2] {
    let rounded = [low, high].map(|v| crate::ui::compact_number(f64::from(v)));
    let read = |text: &str| text.parse::<f32>().unwrap_or(f32::NAN);
    if low >= high || read(&rounded[0]) < read(&rounded[1]) {
        rounded
    } else {
        [low.to_string(), high.to_string()]
    }
}
