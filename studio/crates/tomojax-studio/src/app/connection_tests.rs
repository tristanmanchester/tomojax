//! Drive the project owner, file dialogs and viewer against the real worker.
//! The child process has its own catalog, preferences and stdio relay.
use super::{
    files::{Action, Panel, Purpose},
    *,
};
use rune::Tx;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::process::Command;
use studio_core::{
    connection::{Owner, Profile, State},
    engine::WorkerCommand,
    worker_path::WorkerPath,
};

fn executable(path: &Path, script: &str) {
    fs::write(path, script).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}
fn wait(state: &mut AppState, why: &str, done: impl Fn(&AppState) -> bool) {
    let until = Instant::now() + Duration::from_secs(35);
    while !done(state) {
        state.poll();
        assert!(
            Instant::now() < until,
            "{why}: {} {:?} {} {:?}",
            state.connection_status(),
            state.model.error,
            state.model.backend.status,
            state.model.validation
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}
fn snap(state: AppState, name: &str) {
    let (out, size, scale): (PathBuf, String, String) =
        serde_json::from_str(&std::env::var("TOMOJAX_CONNECTION_CAPTURES").unwrap()).unwrap();
    tests::capture_native_viewport(state, &out.join(format!("{name}.png")), &size, &scale);
}

fn scan(path: &Path, offset: f32) {
    let file = hdf5::File::create(path).unwrap();
    let group = file.create_group("entry").unwrap();
    let data = group
        .new_dataset::<f32>()
        .shape([8, 96, 128])
        .create("projections")
        .unwrap();
    let values: Vec<_> = (0..8 * 96 * 128).map(|i| offset + i as f32).collect();
    data.write_raw(&values).unwrap();
}

#[test]
#[ignore = "drives the native worker through tools/slowlink.py with an isolated profile"]
fn named_project_reads_and_reconnects() {
    if let Ok(root) = std::env::var("TOMOJAX_CONNECTION_FIXTURE") {
        exercise(
            Path::new(&root),
            &std::env::var("TOMOJAX_CONNECTION_SCENE").unwrap_or_default(),
        );
        return;
    }
    let scenes: Vec<&str> = if std::env::var_os("TOMOJAX_CONNECTION_CAPTURES").is_some() {
        vec![
            "connections",
            "editor",
            "connecting",
            "closed",
            "login-needed",
            "logging-in",
            "blocked",
            "results",
            "browser",
            "copy-to",
            "copy-from",
            "starting",
            "connected",
            "executor-stopped",
            "not-live",
            "reconnected",
            "other-data",
        ]
    } else {
        vec![""]
    };
    for scene in scenes {
        let root = tempfile::tempdir().unwrap();
        let project_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../..")
            .canonicalize()
            .unwrap();
        let status = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "app::connection_tests::named_project_reads_and_reconnects",
                "--ignored",
                "--nocapture",
            ])
            .env("TOMOJAX_CONNECTION_FIXTURE", root.path())
            .env("TOMOJAX_CONNECTION_SCENE", scene)
            .env("XDG_CONFIG_HOME", root.path().join("config"))
            .env("TOMOJAX_STUDIO_SETTINGS", root.path().join("studio.json"))
            .env("TOMOJAX_TEST_WORKER", WorkerCommand::binary().unwrap())
            .env(
                "TOMOJAX_TEST_SLOWLINK",
                project_dir.join("tools/slowlink.py"),
            )
            .env(
                "TOMOJAX_TEST_PYTHON",
                std::env::var_os("TOMOJAX_TEST_PYTHON")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| project_dir.join(".venv/bin/python")),
            )
            .env_remove("TOMOJAX_STUDIO_INPUT")
            .env("JAX_PLATFORMS", "cpu")
            .status()
            .unwrap();
        assert!(status.success(), "scene {scene}");
    }
}

fn exercise(root: &Path, scene: &str) {
    let data = root.join("Beamline scans");
    fs::create_dir(&data).unwrap();
    let source = data.join("rotation.nxs");
    scan(&source, 0.);
    let fallback = root.join("Processing");
    fs::create_dir(&fallback).unwrap();
    let python = std::env::var("TOMOJAX_TEST_PYTHON").unwrap();
    let launcher = root.join("allocation");
    executable(
        &launcher,
        "#!/bin/sh\necho $$ > \"$TOMOJAX_CONNECTION_FIXTURE/allocation.pid\"\nwhile test ! -f \"$TOMOJAX_CONNECTION_FIXTURE/allocated\"; do sleep .05; done\nexec \"$@\"\n",
    );
    let profile: Profile =
        serde_json::from_value(json!({"python":python,"session":launcher,"results":fallback}))
            .unwrap();
    Profile::save("Beamline", None, &profile).unwrap();
    let mut state = AppState::init();
    let mut tx = Tx::default();
    if scene == "connections" || scene == "editor" {
        state
            .file_action(Action::Connections(true), &mut tx)
            .unwrap();
        if scene == "editor" {
            state
                .file_action(Action::Edit(Some("Beamline".into())), &mut tx)
                .unwrap();
        }
        return snap(state, scene);
    }
    let local = root.join("Mount review");
    state.files.new_connection = Some("Beamline".into());
    state.open_project(&local, true).unwrap();
    assert!(!local.join("inputs").exists());
    let relay = root.join("relay");
    executable(
        &relay,
        r#"#!/bin/sh
echo $$ > "$TOMOJAX_CONNECTION_FIXTURE/relay.pid"
case "$TOMOJAX_CONNECTION_SCENE" in
login-needed|logging-in) if test "$3" = --login; then sleep 5; else exit 3; fi ;;
blocked) echo 'Worker executable unavailable' >&2; exit 5 ;;
esac
exec "$TOMOJAX_TEST_PYTHON" "$TOMOJAX_TEST_SLOWLINK" -- "$TOMOJAX_TEST_WORKER" "$@"
"#,
    );
    state.engine = session::Session::Named {
        owner: Box::new(Owner::new(
            relay,
            "Beamline".into(),
            &["project_results", "list", "fetch"],
            state.wake.clone(),
        )),
        verified: false,
    };
    state.poll();
    match scene {
        "connecting" => return snap(state, scene),
        "closed" => {
            state.engine.owner_mut().unwrap().close();
            return snap(state, scene);
        }
        "blocked" | "login-needed" | "logging-in" => {
            wait(&mut state, "transport state", |s| {
                matches!(
                    s.connection_state(),
                    Some(State::Blocked | State::LoginNeeded)
                )
            });
            if scene == "logging-in" {
                state.file_action(Action::Login, &mut tx).unwrap();
            }
            state
                .file_action(Action::Connections(false), &mut tx)
                .unwrap();
            return snap(state, scene);
        }
        _ => {}
    }
    wait(&mut state, "results question", |s| {
        s.files.listing.is_some()
    });
    assert_eq!(
        state.files.listing.as_ref().unwrap().path.lexical(),
        fallback
    );
    state.files.first_scan = Some((WorkerPath::from(source.as_path()), false));
    state.choose_results().unwrap();
    wait(&mut state, "writable suggestion", |s| {
        s.files.listing.is_some()
    });
    assert_eq!(state.files.listing.as_ref().unwrap().path.lexical(), data);
    if scene == "results" {
        return snap(state, scene);
    }
    fs::set_permissions(&data, fs::Permissions::from_mode(0o555)).unwrap();
    state.choose_results().unwrap();
    wait(&mut state, "read-only suggestion", |s| {
        s.files.listing.is_some()
    });
    assert_eq!(
        state.files.listing.as_ref().unwrap().path.lexical(),
        fallback
    );
    fs::set_permissions(&data, fs::Permissions::from_mode(0o755)).unwrap();
    let header = data.join("scan.xtekct");
    fs::write(&header, "[XTekCT]").unwrap();
    state.files.first_scan = Some((header.as_path().into(), false));
    state.ui.import_copy = true;
    state.choose_results().unwrap();
    wait(&mut state, "managed folder suggestion", |s| {
        s.files.listing.is_some()
    });
    assert_eq!(
        state.files.listing.as_ref().unwrap().path.lexical(),
        fallback
    );
    state.ui.import_copy = false;
    state.files.first_scan = Some((source.as_path().into(), false));
    state.file_action(Action::Accept, &mut tx).unwrap();
    wait(
        &mut state,
        "lossless projection during executor startup",
        |s| s.imaging.plane(4).is_some_and(|p| p.lossless),
    );
    assert!(!state.model.backend.ready);
    assert_eq!(state.model.validation.phase, validation::Phase::Waiting);
    assert_eq!(state.imaging.plane(4).unwrap().values[1], 1.);
    assert!(local.join(project::MANIFEST).is_file());
    if scene == "starting" {
        return snap(state, scene);
    }
    state.files.panel = Panel::Browser(Purpose::Browse);
    state
        .file_action(Action::List(Some(data.as_path().into())), &mut tx)
        .unwrap();
    wait(&mut state, "file browser", |s| s.files.listing.is_some());
    let entries = &state.files.listing.as_ref().unwrap().entries;
    let size = entries
        .iter()
        .find(|e| e.name == "rotation.nxs" && e.hdf5)
        .unwrap()
        .size;
    if scene == "browser" {
        return snap(state, scene);
    }
    state
        .file_action(Action::Select(source.as_path().into(), size), &mut tx)
        .unwrap();
    let destination = root.join("Downloaded rotation.nxs");
    state = transfer(
        state,
        Action::Download,
        &destination,
        "Copy to this computer",
    );
    assert_eq!(fs::read(&source).unwrap(), fs::read(&destination).unwrap());
    if scene == "copy-to" {
        return snap(state, scene);
    }
    let duplicate = data.join(destination.file_name().unwrap());
    state = transfer(
        state,
        Action::Upload,
        &destination,
        "Copy from this computer",
    );
    assert_eq!(fs::read(&duplicate).unwrap(), fs::read(&source).unwrap());
    if scene == "copy-from" {
        return snap(state, scene);
    }
    let files = state.file_reads().unwrap();
    assert!(
        files
            .download(&source.as_path().into(), size, &destination)
            .is_err()
    );
    assert!(
        files
            .upload(&destination, &duplicate.as_path().into())
            .is_err()
    );
    state.files.close_browser();
    if scene == "executor-stopped" {
        let pid = fs::read_to_string(root.join("allocation.pid")).unwrap();
        assert!(
            Command::new("kill")
                .args(["-TERM", pid.trim()])
                .status()
                .unwrap()
                .success()
        );
        wait(&mut state, "executor stopped", |s| {
            s.model.backend.status.starts_with("TomoJAX stopped")
        });
        return snap(state, scene);
    }
    fs::write(root.join("allocated"), []).unwrap();
    wait(&mut state, "executor ready", |s| s.model.backend.ready);
    if scene == "connected" {
        return snap(state, scene);
    }
    state.dispatch(UiAction::SetProjectionIndex(5), &mut tx);
    let previous = state.imaging.plane(4).map(|p| p.id);
    let pid = fs::read_to_string(root.join("relay.pid")).unwrap();
    assert!(
        Command::new("kill")
            .args(["-TERM", pid.trim()])
            .status()
            .unwrap()
            .success()
    );
    wait(&mut state, "relay drop", |s| {
        matches!(s.connection_state(), Some(State::Waiting(_)))
    });
    let lost = Instant::now();
    if scene == "not-live" {
        return snap(state, scene);
    }
    scan(&source, 1_000_000.);
    wait(&mut state, "new worker and requested projection", |s| {
        if s.data_offline() {
            return false;
        }
        s.imaging.plane(4).is_some_and(|p| {
            if p.lossless {
                assert!(
                    p.values[0] >= 1_000_000.,
                    "a previous connection's data appeared"
                );
            }
            p.lossless && p.index == 5
        })
    });
    let plane = state.imaging.plane(4).unwrap();
    assert_ne!(Some(plane.id), previous);
    assert_eq!(plane.values[0], 1_000_000. + 5. * 96. * 128.);
    eprintln!(
        "Reconnected to the requested lossless plane in {:.3} s",
        lost.elapsed().as_secs_f64()
    );
    if scene == "reconnected" {
        return snap(state, scene);
    }
    if scene.is_empty() {
        exercise_kept_bundle(&mut state, root, &python);
    }
    let project = state.project.as_ref().unwrap();
    let results = project.document.results.clone().unwrap();
    let marker = results.join(&format!(".tomojax-project-{}", project.document.id));
    fs::remove_file(marker.lexical()).unwrap();
    state.shutdown();
    drop(state);
    let mut reopened = AppState::init();
    reopened.open_project(&local, false).unwrap();
    wait(&mut reopened, "wrong-machine refusal", |s| {
        s.model
            .error
            .as_ref()
            .is_some_and(|e| e.contains("other data"))
    });
    assert!(reopened.engine.as_ref().is_none());
    assert!(reopened.imaging.plane(4).is_none());
    if scene == "other-data" {
        return snap(reopened, scene);
    }
    reopened.shutdown();
}

fn transfer(state: AppState, action: Action, answer: &Path, title: &str) -> AppState {
    let mut rt = rune::Runtime::new(
        state,
        crate::ui::view,
        (1440., 900.),
        rune::Env::new(false, 1.),
    );
    rt.update(|s, tx| s.file_action(action, tx).unwrap());
    rt.frame(0.);
    assert_eq!(rt.file_dialog().unwrap().1.title, title);
    rt.resolve_file_dialog(Some(answer.to_owned())).unwrap();
    rt.frame(0.1);
    wait(&mut rt.state, title, |s| {
        s.local_tasks.is_empty() && !s.files.loading
    });
    assert!(
        rt.state.files.transfer.is_some(),
        "{:?}",
        rt.state.model.error
    );
    rt.state
}

fn exercise_kept_bundle(state: &mut AppState, root: &Path, python: &str) {
    let valid = root.join("synthetic.nxs");
    let output = Command::new(python).args(["-c", "import sys, tomojax as tj; tj.save(sys.argv[1], tj.simulate('shepp', size=16, n_views=24, seed=1).scan)"]).arg(&valid).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let tx = &mut Tx::default();
    state.ui.import_copy = true;
    state.files.panel = Panel::Browser(Purpose::Import);
    state
        .file_action(Action::List(Some(valid.as_path().into())), tx)
        .unwrap();
    wait(state, "choose scan to copy", |s| s.files.selected.is_some());
    state.file_action(Action::Accept, tx).unwrap();
    wait(state, "managed scan", |s| {
        s.model
            .active_asset
            .as_ref()
            .and_then(|id| s.project.as_ref()?.document.asset(id))
            .is_some_and(|a| a.managed && !a.arrays.is_empty())
    });
    let id = state.model.active_asset.clone().unwrap();
    let project = state.project.as_ref().unwrap();
    let stored = &project.document.asset(&id).unwrap().path;
    assert!(stored.lexical().starts_with("inputs"));
    assert_eq!(
        fs::read(&valid).unwrap(),
        fs::read(project.resolve(stored).lexical()).unwrap()
    );
    state.files.panel = Panel::Browser(Purpose::Relink(id.clone(), false));
    state
        .file_action(Action::List(Some(valid.as_path().into())), tx)
        .unwrap();
    wait(state, "choose replacement", |s| s.files.selected.is_some());
    state.file_action(Action::Accept, tx).unwrap();
    wait(state, "relinked scan", |s| s.metadata_pending == 0);
    assert_eq!(
        state
            .project
            .as_ref()
            .unwrap()
            .document
            .asset(&id)
            .unwrap()
            .path
            .lexical(),
        valid
    );
    state.set_operation("inspect", None);
    state.queue_operation("inspect").unwrap();
    wait(state, "inspection", |s| {
        s.project
            .as_ref()
            .unwrap()
            .document
            .jobs
            .last()
            .is_some_and(|j| !j.status.active())
    });
    let project = state.project.as_ref().unwrap();
    assert_eq!(
        project.document.jobs.last().unwrap().status,
        project::JobStatus::Completed
    );
    assert_eq!(
        project.document.asset(&id).unwrap().inspection["projection"]["found"],
        true
    );
    state.set_operation(
        "slice",
        Some(&json!({"axis":"z","index":8,"method":"fourier"})),
    );
    state.request_interactive("slice").unwrap();
    wait(state, "slice preview", |s| {
        s.model.preview.as_ref().is_some_and(|p| p.can_keep) && s.imaging.plane(1).is_some()
    });
    state.keep_interactive().unwrap();
    wait(state, "kept result", |s| {
        s.model.active_asset.as_deref() != Some(&id)
    });
    let kept = state.current_form_input().unwrap();
    let project = state.project.as_ref().unwrap();
    assert!(
        kept.path
            .lexical()
            .starts_with(project.data_root().join("kept"))
    );
    let bundle = root.join("Exported arrays");
    state.queue_export(kept, bundle.clone(), "array").unwrap();
    wait(state, "bundle export", |s| {
        s.project
            .as_ref()
            .unwrap()
            .document
            .jobs
            .last()
            .is_some_and(|j| !j.status.active())
    });
    let metadata: Value =
        serde_json::from_slice(&fs::read(bundle.join("arrays.json")).unwrap()).unwrap();
    let arrays = metadata["arrays"].as_object().unwrap();
    assert!(!arrays.is_empty());
    for array in arrays.values() {
        let dest = bundle.join(array["storage"]["path"].as_str().unwrap());
        assert!(fs::read(dest).unwrap().starts_with(b"\x93NUMPY"));
    }
}
