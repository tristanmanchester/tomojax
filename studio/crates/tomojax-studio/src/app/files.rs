//! File selections belong to the current project and connection generation.
use super::*;
use crate::protocol::Listing;
use rune::{FileDialogOptions, TextEdit, Tx};
use studio_core::{connection::Profile, engine::FileReads, worker_path::WorkerPath};

const BROWSER_PAGE: usize = 128;

#[derive(Clone, Debug, PartialEq)]
pub enum Purpose {
    Import,
    Folder,
    Relink(String, bool),
    Results,
    FirstScan,
    Browse,
    Export {
        format: String,
        name: String,
        asset: String,
    },
}

#[derive(Clone)]
pub(crate) enum Panel {
    Closed,
    Connections,
    Editor,
    Browser(Purpose),
}

pub(crate) struct FilePanel {
    pub panel: Panel,
    pub names: Vec<String>,
    pub creating: bool,
    pub new_connection: Option<String>,
    pub editor: [TextEdit; 7],
    pub original: Option<Profile>,
    pub path: TextEdit,
    pub listing: Option<Listing>,
    pub selected: Option<(WorkerPath, u64)>,
    pub transfer: Option<String>,
    pub loading: bool,
    pub first_scan: Option<(WorkerPath, bool)>,
    revision: u64,
}

impl Default for FilePanel {
    fn default() -> Self {
        Self {
            panel: Panel::Closed,
            names: vec![],
            creating: false,
            new_connection: None,
            editor: Default::default(),
            original: None,
            path: Default::default(),
            listing: None,
            selected: None,
            transfer: None,
            loading: false,
            first_scan: None,
            revision: 0,
        }
    }
}
impl FilePanel {
    pub fn close_browser(&mut self) {
        self.panel = Panel::Closed;
        self.transfer = None;
        self.listing = None;
        self.selected = None;
        self.loading = false;
        self.revision += 1;
    }
}

pub(super) enum FileReply {
    Listed(u64, bool, Result<Listing, String>),
    Claimed(WorkerPath, Result<WorkerPath, String>),
    Transferred(Result<(String, Option<WorkerPath>), String>),
    Imported(Result<Vec<(PathBuf, bool)>, String>),
}

#[derive(Clone, Debug)]
pub enum Action {
    Connections(bool),
    Choose(Option<String>),
    Edit(Option<String>),
    Save,
    Close,
    Login,
    Retry,
    Browse(Purpose),
    List(Option<WorkerPath>),
    Output(WorkerPath),
    EnterPath,
    More,
    Select(WorkerPath, u64),
    Accept,
    Download,
    Upload,
}

impl AppState {
    fn machine(&self) -> &str {
        self.remote_project().unwrap_or("local")
    }
    fn browsing_files(&mut self) -> Result<FileReads, String> {
        if self.remote_project().is_some()
            && self
                .project
                .as_ref()
                .is_some_and(|p| p.document.results.is_none())
        {
            return self
                .engine
                .owner()
                .and_then(|o| o.engine())
                .map(Engine::files)
                .ok_or("Not connected".into());
        }
        self.file_reads().ok_or("Not connected".into())
    }

    pub(super) fn choose_results(&mut self) -> Result<(), String> {
        let files = self.browsing_files()?;
        let scan = self.files.first_scan.clone();
        let managed = self.ui.import_copy;
        let last = self
            .preferences
            .results_folders
            .get(self.machine())
            .cloned();
        let profile = Profile::load(self.machine())
            .ok()
            .and_then(|p| p.results)
            .map(|s| WorkerPath::from(s.as_str()));
        self.files.panel = Panel::Browser(Purpose::Results);
        let project = self
            .project
            .as_ref()
            .ok_or("Open a project")?
            .document
            .name
            .clone();
        self.begin_listing(false, move || {
            let directory = scan.as_ref().and_then(|(path, folder)| {
                if *folder {
                    Some(path.clone())
                } else {
                    path.parent()
                }
            });
            let source = managed_source(scan.as_ref(), managed);
            for path in [directory, last, profile].into_iter().flatten() {
                if let Ok(list) = files.list(Some(&path), None, BROWSER_PAGE)
                    && list.directory
                    && list.writable
                    && outside_source(&list.path, &project, source.as_ref())
                {
                    return Ok(list);
                }
            }
            files.list(None, None, BROWSER_PAGE)
        });
        Ok(())
    }

    fn begin_listing(
        &mut self,
        append: bool,
        read: impl FnOnce() -> Result<Listing, String> + Send + 'static,
    ) {
        self.files.revision += 1;
        let revision = self.files.revision;
        self.files.loading = true;
        self.files.selected = None;
        if !append {
            self.files.listing = None;
        }
        self.spawn_local(None, move |_| {
            LocalEvent::FilePanel(FileReply::Listed(revision, append, read()))
        });
    }
    fn list_folder(
        &mut self,
        path: Option<WorkerPath>,
        cursor: Option<String>,
    ) -> Result<(), String> {
        let files = self.browsing_files()?;
        self.begin_listing(cursor.is_some(), move || {
            files.list(path.as_ref(), cursor.as_deref(), BROWSER_PAGE)
        });
        Ok(())
    }
    pub(super) fn file_reply(&mut self, reply: FileReply) {
        let result = match reply {
            FileReply::Listed(rev, append, result) if rev == self.files.revision => {
                self.files.loading = false;
                result.map(|mut list| {
                    self.files.path = TextEdit::new(list.path.to_string());
                    if append && let Some(old) = &mut self.files.listing {
                        old.entries.append(&mut list.entries);
                        old.after = list.after;
                    } else {
                        if !list.directory
                            && let Some(file) = list.entries.first()
                        {
                            self.files.selected = Some((list.path.clone(), file.size));
                        }
                        self.files.listing = Some(list);
                    }
                })
            }
            FileReply::Listed(..) => return,
            FileReply::Claimed(parent, result) => {
                self.files.loading = false;
                result.and_then(|path| {
                    let project = self.project.as_mut().ok_or("No project")?;
                    project.document.results = Some(path);
                    project.changed()?;
                    project.flush()?;
                    self.preferences
                        .results_folders
                        .insert(self.machine().to_owned(), parent);
                    self.files.close_browser();
                    self.verified(Ok(()));
                    if let Some((scan, folder)) = self.files.first_scan.take() {
                        self.import_selected(vec![(scan.lexical().into(), folder)]);
                    }
                    self.preferences.save()
                })
            }
            FileReply::Transferred(result) => result.and_then(|(message, uploaded_to)| {
                self.model.status = message.clone();
                self.files.transfer = Some(message);
                if let Some(folder) = uploaded_to
                    && self
                        .files
                        .listing
                        .as_ref()
                        .is_some_and(|list| list.path == folder)
                {
                    self.list_folder(Some(folder), None)?;
                }
                Ok(())
            }),
            FileReply::Imported(result) => result.map(|paths| self.import_selected(paths)),
        };
        if let Err(error) = result {
            self.model.error = Some(error);
        }
    }

    fn edit_connection(&mut self, name: Option<String>) -> Result<(), String> {
        let profile = name
            .as_deref()
            .map(Profile::load)
            .transpose()
            .map_err(|e| e.to_string())?;
        let value = serde_json::to_value(&profile).map_err(|e| e.to_string())?;
        let hops = profile
            .as_ref()
            .map(|p| p.hops().join(" "))
            .unwrap_or_default();
        self.files.editor = [
            name.as_deref().unwrap_or(""),
            &hops,
            value["python"].as_str().unwrap_or(""),
            value["session"].as_str().unwrap_or(""),
            value["jobs"].as_str().unwrap_or(""),
            value["sbatch"].as_str().unwrap_or(""),
            value["results"].as_str().unwrap_or(""),
        ]
        .map(TextEdit::new);
        self.files.original = profile;
        self.files.panel = Panel::Editor;
        Ok(())
    }
    fn save_connection(&mut self, tx: &mut Tx) -> Result<(), String> {
        let [name, ssh, python, session, jobs, sbatch, results] =
            self.files.editor.each_ref().map(|edit| edit.text.trim());
        let mut draft = json!({"ssh":ssh.split_whitespace().collect::<Vec<_>>()});
        for (key, value) in [
            ("python", python),
            ("session", session),
            ("jobs", jobs),
            ("sbatch", sbatch),
            ("results", results),
        ] {
            if !value.is_empty() {
                draft[key] = json!(value);
            }
        }
        let profile = serde_json::from_value(draft).map_err(|e| e.to_string())?;
        Profile::save(name, self.files.original.as_ref(), &profile).map_err(|e| e.to_string())?;
        self.file_action(Action::Connections(self.files.creating), tx)?;
        Ok(())
    }

    pub(super) fn file_action(&mut self, action: Action, tx: &mut Tx) -> Result<(), String> {
        match action {
            Action::Connections(creating) => {
                self.files.names = Profile::catalog()
                    .map_err(|e| e.to_string())?
                    .into_keys()
                    .collect();
                self.files.panel = Panel::Connections;
                self.files.creating = creating;
            }
            Action::Choose(connection) => {
                if self.files.creating {
                    self.files.panel = Panel::Closed;
                    let options = FileDialogOptions {
                        title: "Create a project on this computer".into(),
                        file_name: Some("TomoJAX project".into()),
                        ..Default::default()
                    };
                    tx.save_file(options, move |s: &mut AppState, result, _| match result {
                        Ok(Some(path)) => {
                            s.files.new_connection = connection.clone();
                            s.files.first_scan = None;
                            if let Err(error) = s.open_project(&path, true) {
                                s.model.error = Some(error);
                            }
                            s.files.new_connection = None;
                        }
                        Err(e) => s.model.error = Some(e.to_string()),
                        Ok(None) => {}
                    });
                } else if self.data_offline()
                    && let Some(project) = &mut self.project
                    && project.document.connection.is_some()
                {
                    project.document.connection =
                        Some(connection.ok_or("Choose a connection to the project's results")?);
                    project.changed()?;
                    self.connection_epoch += 1;
                    self.files.close_browser();
                    self.select_project_source();
                }
            }
            Action::Edit(name) => self.edit_connection(name)?,
            Action::Save => self.save_connection(tx)?,
            Action::Retry => {
                if let Some(owner) = self.engine.owner_mut() {
                    owner.retry_now();
                }
            }
            Action::Login => {
                let bin = std::env::current_exe().map_err(|e| e.to_string())?;
                self.engine
                    .owner_mut()
                    .ok_or("Select a connection")?
                    .login(&bin)?;
            }
            Action::Close => self.files.close_browser(),
            Action::Browse(purpose) => {
                if self.data_offline() && !matches!(purpose, Purpose::Results | Purpose::FirstScan)
                {
                    return Err(
                        "Connect and select the project's results folder before browsing its data."
                            .into(),
                    );
                }
                if purpose == Purpose::Results {
                    return self.choose_results();
                }
                self.files.panel = Panel::Browser(purpose);
                let path = self
                    .preferences
                    .recent_folders
                    .get(self.machine())
                    .and_then(|v| v.first())
                    .cloned();
                self.list_folder(path, None)?;
            }
            Action::Output(path) => {
                self.files.panel = Panel::Browser(Purpose::Browse);
                self.list_folder(Some(path), None)?;
                self.ui.activity.close();
            }
            Action::List(path) => self.list_folder(path, None)?,
            Action::EnterPath => {
                self.list_folder(Some(self.files.path.text.as_str().into()), None)?
            }
            Action::More => {
                let list = self.files.listing.as_ref().ok_or("Choose a folder")?;
                self.list_folder(Some(list.path.clone()), list.after.clone())?;
            }
            Action::Select(path, size) => self.files.selected = Some((path, size)),
            Action::Accept => self.accept_file()?,
            Action::Download | Action::Upload => self.transfer_dialog(action, tx)?,
        }
        Ok(())
    }

    fn accept_file(&mut self) -> Result<(), String> {
        let Panel::Browser(purpose) = self.files.panel.clone() else {
            return Ok(());
        };
        if self.files.loading {
            return Ok(());
        }
        let list = self.files.listing.as_ref().ok_or("Choose a folder")?;
        let path = self
            .files
            .selected
            .as_ref()
            .map_or(&list.path, |(path, _)| path)
            .clone();
        let folder = self.files.selected.is_none() && list.directory;
        let parent = if folder {
            Some(path.clone())
        } else {
            path.parent()
        };
        if let Some(parent) = parent {
            let machine = self.machine().to_owned();
            let recent = self.preferences.recent_folders.entry(machine).or_default();
            recent.retain(|p| p != &parent);
            recent.insert(0, parent);
            recent.truncate(5);
            self.preferences.save()?;
        }
        match purpose {
            Purpose::FirstScan => {
                self.files.first_scan = Some((path, folder));
                return self.choose_results();
            }
            Purpose::Results => {
                if !folder || !list.writable {
                    return Err("Choose a writable folder for results".into());
                }
                let files = self.browsing_files()?;
                let doc = &self.project.as_ref().ok_or("Open a project")?.document;
                let source = managed_source(self.files.first_scan.as_ref(), self.ui.import_copy);
                if !outside_source(&path, &doc.name, source.as_ref()) {
                    return Err(
                        "Choose a results folder outside the acquisition being copied.".into(),
                    );
                }
                let cmd = json!({"type":"project_results","action":"claim","parent":path,"name":doc.name,"project":doc.id});
                self.files.loading = true;
                self.spawn_local(None, move |_| {
                    LocalEvent::FilePanel(FileReply::Claimed(
                        path,
                        files.request(cmd).and_then(|reply| match reply {
                            ResultValue::ProjectResults { path } => Ok(path),
                            _ => Err("Expected a results folder".into()),
                        }),
                    ))
                });
                return Ok(());
            }
            Purpose::Import => self.import_selected(vec![(path.lexical().into(), folder)]),
            Purpose::Folder => {
                if !folder {
                    return Err("Choose an acquisition folder".into());
                }
                let files = self.browsing_files()?;
                self.spawn_local(None, move |_| {
                    LocalEvent::FilePanel(FileReply::Imported(acquisitions(&files, &path)))
                });
            }
            Purpose::Relink(id, expected) => {
                if expected != folder {
                    return Err(format!(
                        "Choose a {}",
                        if expected { "folder" } else { "file" }
                    ));
                }
                self.relink(&id, path.lexical());
            }
            Purpose::Export {
                format,
                name,
                asset,
            } => {
                self.export_to_folder(&asset, &format, &name, &path, folder)?;
            }
            Purpose::Browse => {}
        }
        self.files.close_browser();
        Ok(())
    }

    fn export_to_folder(
        &mut self,
        asset: &str,
        format: &str,
        name: &str,
        path: &WorkerPath,
        folder: bool,
    ) -> Result<(), String> {
        if !folder {
            return Err("Choose a destination folder".into());
        }
        let dest = path.join(name).lexical().to_owned();
        if self.model.active_asset.as_deref() != Some(asset) {
            return Err("Select the result to export".into());
        }
        if self.interchange_format().is_some() {
            self.ui.export_path = Some(dest);
        } else {
            let input = self
                .project
                .as_ref()
                .ok_or("Open a project")?
                .capture(asset)?;
            self.queue_export(input, dest, format)?;
        }
        Ok(())
    }

    fn transfer_dialog(&mut self, action: Action, tx: &mut Tx) -> Result<(), String> {
        let files = self.file_reads().ok_or("Not connected")?;
        let epoch = self.connection_epoch;
        if matches!(action, Action::Download) {
            let (source, size) = self.files.selected.clone().ok_or("Choose a file")?;
            studio_core::transfer::check_size(size).map_err(|e| e.to_string())?;
            tx.save_file(
                FileDialogOptions {
                    title: "Copy to this computer".into(),
                    file_name: Some(source.name().into()),
                    ..Default::default()
                },
                move |s: &mut AppState, result, _| {
                    if epoch != s.connection_epoch {
                        return;
                    }
                    match result {
                        Ok(Some(dest)) => s.spawn_local(None, move |_| {
                            LocalEvent::FilePanel(FileReply::Transferred(
                                files.download(&source, size, &dest).map(|()| {
                                    (
                                        format!(
                                            "Copied {} to this computer: {}",
                                            source.name(),
                                            dest.display()
                                        ),
                                        None,
                                    )
                                }),
                            ))
                        }),
                        Err(e) => s.model.error = Some(e.to_string()),
                        Ok(None) => {}
                    }
                },
            );
        } else {
            let folder = self
                .files
                .listing
                .as_ref()
                .filter(|list| list.directory && list.writable)
                .ok_or("Choose a writable folder")?
                .path
                .clone();
            tx.open_file(
                FileDialogOptions {
                    title: "Copy from this computer".into(),
                    ..Default::default()
                },
                move |s: &mut AppState, result, _| {
                    if epoch != s.connection_epoch {
                        return;
                    }
                    match result {
                        Ok(Some(source)) => s.spawn_local(None, move |_| {
                            let result = source
                                .file_name()
                                .and_then(|s| s.to_str())
                                .ok_or("Choose a file with a UTF-8 name".into())
                                .and_then(|name| {
                                    let dest = folder.join(name);
                                    files.upload(&source, &dest).map(|()| {
                                        (
                                            format!("Copied {name} from this computer to {folder}"),
                                            Some(folder.clone()),
                                        )
                                    })
                                });
                            LocalEvent::FilePanel(FileReply::Transferred(result))
                        }),
                        Err(e) => s.model.error = Some(e.to_string()),
                        Ok(None) => {}
                    }
                },
            );
        }
        Ok(())
    }
}

fn acquisitions(files: &FileReads, folder: &WorkerPath) -> Result<Vec<(PathBuf, bool)>, String> {
    let mut answer = vec![];
    let mut after = None;
    loop {
        let page = files.list(Some(folder), after.as_deref(), 4096)?;
        for entry in page.entries {
            if entry.kind == "file" && (entry.hdf5 || project::is_hdf5(Path::new(&entry.name))) {
                answer.push((PathBuf::from(folder.join(&entry.name).lexical()), false));
            }
        }
        after = page.after;
        if after.is_none() {
            break;
        }
    }
    if answer.is_empty() {
        answer.push((PathBuf::from(folder.lexical()), true));
    }
    Ok(answer)
}

fn managed_source(scan: Option<&(WorkerPath, bool)>, managed: bool) -> Option<WorkerPath> {
    let (path, folder) = scan.filter(|_| managed)?;
    if *folder {
        Some(path.clone())
    } else if project::holds_whole_acquisition(path.lexical()) {
        None
    } else {
        path.parent()
    }
}
fn outside_source(parent: &WorkerPath, name: &str, source: Option<&WorkerPath>) -> bool {
    source.is_none_or(|source| !parent.join(name).lexical().starts_with(source.lexical()))
}
