//! Small application preferences; project-specific context lives in the project.
use rune::{Chord, KeyPlatform};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// The canonical binding text for a pressed key combination, or None while only modifiers are
/// down or the key cannot be bound.
pub fn chord_text(event: &rune::KeyEvent) -> Option<String> {
    use rune::Key;
    let key = match &event.key {
        Key::Command(c) => char_name(*c),
        Key::Text(text) if text.chars().count() == 1 => char_name(text.chars().next()?),
        Key::Function(n) => format!("f{n}"),
        Key::Enter => "enter".into(),
        Key::Tab => "tab".into(),
        Key::Backspace => "backspace".into(),
        Key::Delete => "delete".into(),
        Key::Insert => "insert".into(),
        Key::Home => "home".into(),
        Key::End => "end".into(),
        Key::PageUp => "pageup".into(),
        Key::PageDown => "pagedown".into(),
        Key::Left => "left".into(),
        Key::Right => "right".into(),
        Key::Up => "up".into(),
        Key::Down => "down".into(),
        _ => return None,
    };
    let mods = &event.mods;
    let mut text = String::new();
    for (held, name) in [(mods.cmd, "mod"), (mods.alt, "alt"), (mods.shift, "shift")] {
        if held {
            text.push_str(name);
            text.push('+');
        }
    }
    text.push_str(&key);
    rune::Chord::parse(&text).is_ok().then_some(text)
}

fn char_name(c: char) -> String {
    let c = c.to_ascii_lowercase();
    match c {
        ' ' => "space".into(),
        '+' => "plus".into(),
        ',' => "comma".into(),
        '-' => "minus".into(),
        '=' => "equal".into(),
        '.' => "period".into(),
        '/' => "slash".into(),
        '\\' => "backslash".into(),
        '`' => "backquote".into(),
        ';' => "semicolon".into(),
        '\'' => "quote".into(),
        '[' => "bracketleft".into(),
        ']' => "bracketright".into(),
        other => other.to_string(),
    }
}

/// A binding as this platform writes it ("Ctrl+," or "⌘,").
pub fn binding_label(binding: &str) -> String {
    rune::Chord::parse(binding)
        .map(|chord| chord.label())
        .unwrap_or_else(|_| binding.to_owned())
}

pub const COMMANDS: &[(&str, &str, &str, &str)] = &[
    ("new", "New project", "Project", "mod+n"),
    ("open", "Open project", "Project", "mod+o"),
    ("import", "Import scan", "Project", "mod+i"),
    ("folder", "Import folder", "Project", "mod+shift+i"),
    ("palette", "Search commands", "Workspace", "mod+k"),
    ("shortcuts", "Keyboard shortcuts", "Workspace", "mod+comma"),
    ("documentation", "Open documentation", "Help", "f1"),
    ("prepare", "Prepare workspace", "Workspace", "mod+1"),
    ("reconstruct", "Reconstruct workspace", "Workspace", "mod+2"),
    ("explore", "Explore workspace", "Workspace", "mod+3"),
    ("compare", "Compare workspace", "Workspace", "mod+4"),
    ("jobs", "Jobs workspace", "Workspace", "mod+5"),
    ("queue", "Queue current operation", "Processing", "mod+r"),
    (
        "simulate",
        "Configure simulation",
        "Processing",
        "mod+shift+n",
    ),
    ("export", "Export active result", "Project", "mod+e"),
    ("fit", "Fit image", "View", "f"),
    ("window", "Auto display window", "View", "w"),
    ("axial", "Axial plane", "View", "a"),
    ("coronal", "Coronal plane", "View", "c"),
    ("sagittal", "Sagittal plane", "View", "s"),
    ("volume", "3D volume", "View", "v"),
    ("previous", "Previous slice", "View", "left"),
    ("next", "Next slice", "View", "right"),
    ("undo", "Undo view change", "Workspace", "mod+z"),
    ("redo", "Redo view change", "Workspace", "mod+shift+z"),
];

#[derive(Default, Serialize, Deserialize)]
pub struct Preferences {
    #[serde(default)]
    pub recent: Vec<PathBuf>,
    #[serde(default)]
    pub shortcuts: BTreeMap<String, String>,
}
impl Preferences {
    pub fn path() -> PathBuf {
        if let Some(path) = std::env::var_os("TOMOJAX_STUDIO_SETTINGS") {
            return path.into();
        }
        // Tests never read or write the user's own settings.
        if cfg!(test) {
            return std::env::temp_dir()
                .join(format!("tomojax-studio-test-{}", std::process::id()))
                .join("studio.json");
        }
        let base = if cfg!(target_os = "windows") {
            std::env::var_os("APPDATA").map(PathBuf::from)
        } else {
            std::env::var_os("XDG_CONFIG_HOME")
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME").map(|p| {
                        PathBuf::from(p).join(if cfg!(target_os = "macos") {
                            "Library/Application Support"
                        } else {
                            ".config"
                        })
                    })
                })
        }
        .unwrap_or_else(std::env::temp_dir);
        base.join("tomojax").join("studio.json")
    }
    /// Where Studio keeps projects it creates itself, such as demo projects.
    pub fn data_dir() -> PathBuf {
        if cfg!(test) {
            return std::env::temp_dir()
                .join(format!("tomojax-studio-test-{}", std::process::id()));
        }
        let base = if cfg!(target_os = "windows") {
            std::env::var_os("APPDATA").map(PathBuf::from)
        } else if cfg!(target_os = "macos") {
            std::env::var_os("HOME").map(|p| PathBuf::from(p).join("Library/Application Support"))
        } else {
            std::env::var_os("XDG_DATA_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join(".local/share")))
        };
        base.unwrap_or_else(std::env::temp_dir).join("tomojax")
    }
    pub fn load() -> Result<Self, String> {
        let path = Self::path();
        if !path.exists() {
            return Ok(Self::default());
        }
        let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
        let preferences: Self = serde_json::from_slice(&bytes)
            .map_err(|e| format!("Could not read preferences: {e}"))?;
        for (id, binding) in &preferences.shortcuts {
            preferences.validate_binding(id, binding)?;
        }
        Ok(preferences)
    }
    pub fn save(&self) -> Result<(), String> {
        let path = Self::path();
        let parent = path.parent().ok_or("Invalid preferences path")?;
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        let mut file = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
        file.write_all(&serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?)
            .and_then(|()| file.as_file().sync_all())
            .map_err(|e| e.to_string())?;
        file.persist(path).map_err(|e| e.to_string())?;
        Ok(())
    }
    pub fn visit(&mut self, path: &Path) -> Result<(), String> {
        self.recent.retain(|p| p != path);
        self.recent.insert(0, path.into());
        self.recent.truncate(12);
        self.save()
    }
    pub fn binding<'a>(&'a self, id: &str, default: &'a str) -> &'a str {
        self.shortcuts
            .get(id)
            .map(String::as_str)
            .unwrap_or(default)
    }
    pub fn validate_binding(&self, id: &str, binding: &str) -> Result<(), String> {
        if !COMMANDS.iter().any(|c| c.0 == id) {
            return Err("Unknown command".into());
        }
        let chord = Chord::parse(binding).map_err(|e| e.to_string())?;
        let label = chord.label_on(KeyPlatform::current());
        for reserved in [
            "escape", "tab", "mod+c", "mod+v", "mod+x", "mod+a", "mod+w", "mod+q",
        ] {
            if label
                == Chord::parse(reserved)
                    .unwrap()
                    .label_on(KeyPlatform::current())
            {
                return Err(
                    "This key is reserved for navigation, editing or closing the window".into(),
                );
            }
        }
        for (other, title, _, default) in COMMANDS {
            if *other != id
                && Chord::parse(self.binding(other, default))
                    .is_ok_and(|c| c.label_on(KeyPlatform::current()) == label)
            {
                return Err(format!(
                    "{label} is already assigned to {title}. Choose another key or change that binding first."
                ));
            }
        }
        Ok(())
    }
    pub fn set_binding(&mut self, id: &str, binding: &str) -> Result<(), String> {
        self.validate_binding(id, binding)?;
        let canonical = Chord::parse(binding)
            .map_err(|e| e.to_string())?
            .to_string();
        let previous = self.shortcuts.insert(id.into(), canonical);
        if let Err(error) = self.save() {
            match previous {
                Some(value) => {
                    self.shortcuts.insert(id.into(), value);
                }
                None => {
                    self.shortcuts.remove(id);
                }
            }
            return Err(error);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_pressed_combination_becomes_canonical_binding_text() {
        use rune::{Key, KeyEvent, Mods};
        let press = |key: Key, mods: Mods| crate::preferences::chord_text(&KeyEvent { key, mods });
        let cmd = Mods {
            cmd: true,
            ctrl: true,
            ..Default::default()
        };
        let cmd_shift = Mods { shift: true, ..cmd };
        assert_eq!(
            press(Key::Command('r'), cmd_shift).as_deref(),
            Some("mod+shift+r")
        );
        assert_eq!(press(Key::Command(','), cmd).as_deref(), Some("mod+comma"));
        assert_eq!(
            press(Key::Function(5), Mods::default()).as_deref(),
            Some("f5")
        );
        assert_eq!(press(Key::Escape, cmd), None);
        assert_eq!(
            crate::preferences::binding_label("mod+comma"),
            rune::Chord::parse("mod+comma").unwrap().label()
        );
    }

    use super::*;
    #[test]
    fn conflicts_resolve_platform_aliases_and_reject_reserved_keys() {
        let preferences = Preferences::default();
        assert!(preferences.validate_binding("fit", "mod+o").is_err());
        assert!(preferences.validate_binding("fit", "escape").is_err());
        assert!(preferences.validate_binding("fit", "mod+alt+f").is_ok());
        #[cfg(not(target_os = "macos"))]
        assert!(preferences.validate_binding("fit", "ctrl+o").is_err());
    }
}
