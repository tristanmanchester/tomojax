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
            "{why}: {} {:?} {} jobs {:?} live {:?} {:?}",
            state.connection_status(),
            state.model.error,
            state.model.backend.status,
            state
                .model
                .jobs
                .iter()
                .map(|j| (j.status, j.stage.as_str()))
                .collect::<Vec<_>>(),
            (
                state.live.loading(),
                state.live.error(),
                state.live.frame().map(|f| &f.info.job)
            ),
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
        isolated_scene(
            "app::connection_tests::named_project_reads_and_reconnects",
            scene,
        );
    }
}

fn isolated_scene(test: &str, scene: &str) {
    let root = tempfile::tempdir().unwrap();
    let project_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap();
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--ignored", "--nocapture"])
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
            .is_some_and(|e| e.contains("no results folder for this project"))
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

#[test]
#[ignore = "real slow link with a held fake Slurm allocation; TOMOJAX_TEST_BATCH_GPU exercises CUDA"]
fn batch_jobs_reconnect_with_the_session_executor_queued() {
    if let Ok(root) = std::env::var("TOMOJAX_BATCH_FIXTURE") {
        return batch_reconnect(Path::new(&root));
    }
    if std::env::var_os("TOMOJAX_CONNECTION_CAPTURES").is_some() {
        for scene in [
            "queued",
            "running",
            "finished",
            "failed",
            "interrupted",
            "unknown",
            "cancelled",
        ] {
            batch_child(scene);
        }
    } else {
        batch_child("");
    }
}

fn batch_child(scene: &str) {
    let root = tempfile::tempdir().unwrap();
    let tools = Path::new(env!("CARGO_MANIFEST_DIR")).join("../tomojax-jobs/tests/slurm");
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap();
    let status = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "app::connection_tests::batch_jobs_reconnect_with_the_session_executor_queued",
            "--ignored",
            "--nocapture",
        ])
        .env("TOMOJAX_BATCH_SCENE", scene)
        .env("TOMOJAX_BATCH_FIXTURE", root.path())
        .env("FAKE_SLURM", root.path().join("slurm"))
        .env(
            "PATH",
            std::env::join_paths(
                [tools]
                    .into_iter()
                    .chain(std::env::split_paths(&std::env::var_os("PATH").unwrap())),
            )
            .unwrap(),
        )
        .env("TOMOJAX_TEST_WORKER", WorkerCommand::binary().unwrap())
        .env("TOMOJAX_TEST_SLOWLINK", repo.join("tools/slowlink.py"))
        .env("TOMOJAX_TEST_PYTHON", repo.join(".venv/bin/python"))
        .env("XDG_CONFIG_HOME", root.path().join("config"))
        .env("TOMOJAX_STUDIO_SETTINGS", root.path().join("settings.json"))
        .env_remove("TOMOJAX_STUDIO_INPUT")
        .status()
        .unwrap();
    assert!(status.success());
}

fn batch_reconnect(root: &Path) {
    let python = std::env::var_os("TOMOJAX_TEST_PYTHON").unwrap();
    let gpu = std::env::var_os("TOMOJAX_TEST_BATCH_GPU").is_some();
    let batch = root.join("slurm");
    fs::create_dir(&batch).unwrap();
    fs::write(batch.join("hold"), "").unwrap();
    let session = root.join("session");
    executable(
        &session,
        r#"#!/bin/sh
while test -f "$TOMOJAX_BATCH_FIXTURE/hold-session"; do sleep .05; done
export JAX_PLATFORMS=cpu
exec "$@"
"#,
    );
    let interpreter = root.join("python");
    executable(
        &interpreter,
        r#"#!/bin/sh
if test -n "$SLURM_JOB_ID" && test -n "$TOMOJAX_TEST_BATCH_GPU"; then export JAX_PLATFORMS=cuda; fi
exec "$TOMOJAX_TEST_PYTHON" "$@"
"#,
    );
    let relay = root.join("relay");
    executable(
        &relay,
        r#"#!/bin/sh
exec "$TOMOJAX_TEST_PYTHON" "$TOMOJAX_TEST_SLOWLINK" -- "$TOMOJAX_TEST_WORKER" "$@"
"#,
    );
    let data = root.join("worker");
    fs::create_dir(&data).unwrap();
    let profile = serde_json::from_value(
        json!({"jobs":"slurm","python":interpreter,"session":session,"results":data}),
    )
    .unwrap();
    Profile::save("Batch", None, &profile).unwrap();
    let raw = data.join("Sphere scan.nxs");
    assert!(Command::new(python).args(["-c","import sys,tomojax as tj; tj.save(sys.argv[1],tj.simulate('sphere',size=16,n_views=24,seed=2).scan)"]).arg(&raw).env("JAX_PLATFORMS","cpu").status().unwrap().success());
    let project = root.join("Reconstruction study");
    let open = |create| {
        let mut state = AppState::init();
        if create {
            state.files.new_connection = Some("Batch".into());
        }
        state.open_project(&project, create).unwrap();
        state.engine = session::Session::Named {
            owner: Box::new(Owner::new(
                relay.clone(),
                "Batch".into(),
                &["submit", "job", "jobs", "job_control"],
                state.wake.clone(),
            )),
            verified: false,
        };
        state
    };
    let mut state = open(true);
    wait(&mut state, "results folder", |s| s.files.listing.is_some());
    state.files.first_scan = Some((raw.as_path().into(), false));
    state
        .file_action(Action::Accept, &mut Tx::default())
        .unwrap();
    wait(&mut state, "session CPU and input", |s| {
        s.model.backend.ready && s.current_form_input().is_some_and(|i| !i.arrays.is_empty())
    });
    let config = if gpu { "cuda" } else { "jax" };
    state.set_workspace(Workspace::Reconstruct);
    state.set_operation(
        "reconstruct",
        Some(&json!({"method":"fbp","config":{"type":"FBPConfig","backend":config}})),
    );
    if gpu {
        wait(&mut state, "hardware checked on the job", |s| {
            s.model.validation.phase == validation::Phase::Unavailable
                && s.model.validation.message.contains("job starts")
        });
    }
    state.ui.scheduler_options.replace_text("--time=00:03:00");
    let queued = |s: &AppState, id: &str| {
        s.project.as_ref().unwrap().document.jobs.iter().any(|j| {
            j.id == id
                && j.status == project::JobStatus::Queued
                && j.script.as_ref().and_then(|s| s.launcher.as_deref()) == Some("slurm")
        })
    };
    let scene = std::env::var("TOMOJAX_BATCH_SCENE").unwrap_or_default();
    if matches!(scene.as_str(), "running" | "interrupted") {
        // Long enough to still be running when its first solver preview arrives; the
        // capture cancels it afterwards.
        state.set_operation(
            "reconstruct",
            Some(&json!({"method":"sirt","iterations":360000})),
        );
    }
    state.queue_operation("reconstruct").unwrap();
    let id = state
        .project
        .as_ref()
        .unwrap()
        .document
        .jobs
        .last()
        .unwrap()
        .id
        .clone();
    wait(&mut state, "first allocation queued", |s| queued(s, &id));
    if !scene.is_empty() {
        return capture_batch_state(state, &batch, &raw, &scene, &id);
    }
    state.queue_operation("reconstruct").unwrap();
    let cancel = state
        .project
        .as_ref()
        .unwrap()
        .document
        .jobs
        .last()
        .unwrap()
        .id
        .clone();
    wait(&mut state, "second allocation queued", |s| {
        queued(s, &cancel)
    });
    state.queue_operation("reconstruct").unwrap();
    let abandon = state
        .project
        .as_ref()
        .unwrap()
        .document
        .jobs
        .last()
        .unwrap()
        .id
        .clone();
    wait(&mut state, "third allocation queued", |s| {
        queued(s, &abandon)
    });
    state.shutdown();
    drop(state);
    fs::write(root.join("hold-session"), "").unwrap();
    let time = Instant::now();
    let mut reopened = open(false);
    wait(&mut reopened, "project reconciled", |s| {
        s.launcher() == Some("slurm") && s.job_seen(&id) && s.job_seen(&cancel)
    });
    assert!(!reopened.model.backend.ready);
    reopened.control_job(&cancel, "cancel").unwrap();
    wait(&mut reopened, "cancel while Python is queued", |s| {
        s.project
            .as_ref()
            .unwrap()
            .document
            .jobs
            .iter()
            .any(|j| j.id == cancel && j.status == project::JobStatus::Canceled)
    });
    assert!(queued(&reopened, &id));
    fs::write(batch.join("offline"), "").unwrap();
    let has_status = |s: &AppState, id: &str, status| {
        s.model
            .jobs
            .iter()
            .any(|j| j.id == id && j.status == status)
    };
    wait(&mut reopened, "unknown allocation", |s| {
        has_status(s, &abandon, project::JobStatus::Unknown)
    });
    reopened.control_job(&abandon, "abandon").unwrap();
    wait(&mut reopened, "abandon recorded", |s| {
        has_status(s, &abandon, project::JobStatus::Canceled)
    });
    fs::remove_file(batch.join("offline")).unwrap();
    let dir = reopened.project.as_ref().unwrap().job_directory(&abandon);
    let submission: Value =
        serde_json::from_slice(&fs::read(dir.join("submission.json").lexical()).unwrap()).unwrap();
    let allocation = submission["accepted"].as_u64().unwrap();
    wait(
        &mut reopened,
        "standing cancel reached the rediscovered job",
        |_| batch.join(allocation.to_string()).join("cancel").exists(),
    );
    assert!(queued(&reopened, &id));
    fs::remove_file(batch.join("hold")).unwrap();
    wait(&mut reopened, "published volume", |s| {
        s.project
            .as_ref()
            .unwrap()
            .document
            .jobs
            .iter()
            .any(|j| j.id == id && j.status == project::JobStatus::Completed)
    });
    let p = reopened.project.as_ref().unwrap();
    let job = p.document.jobs.iter().find(|j| j.id == id).unwrap();
    let s = job.script.as_ref().unwrap();
    assert_eq!(s.options, "--time=00:03:00");
    if gpu {
        let activity = reopened.model.jobs.iter().find(|j| j.id == id).unwrap();
        assert!(activity.devices.contains("NVIDIA"), "{}", activity.devices);
        assert!(
            s.devices.iter().any(|d| d.contains("NVIDIA")),
            "{:?}",
            s.devices
        );
    }
    let asset = job.result.clone().unwrap();
    assert_eq!(
        p.document.assets.iter().filter(|a| a.id == asset).count(),
        1
    );
    let directory = p.job_directory(&id);
    assert_eq!(
        fs::read_to_string(directory.join("replies.jsonl").lexical())
            .unwrap()
            .matches("\"type\":\"attempt\"")
            .count(),
        1
    );
    assert_eq!(
        fs::read_to_string(batch.join("next")).unwrap().trim(),
        "103"
    );
    reopened.activate(asset);
    wait(&mut reopened, "open the reconnected result", |s| {
        s.imaging.plane(1).is_some()
    });
    eprintln!(
        "Reconnect, cancel and abandon the two other allocations, run and open the volume: {:.3}s",
        time.elapsed().as_secs_f64()
    );
    reopened.shutdown();
}

fn capture_batch_state(mut state: AppState, batch: &Path, scan: &Path, scene: &str, id: &str) {
    use project::JobStatus;
    let dir = state.project.as_ref().unwrap().job_directory(id);
    let wanted = match scene {
        "queued" => JobStatus::Queued,
        "unknown" => {
            fs::write(batch.join("offline"), "").unwrap();
            JobStatus::Unknown
        }
        "cancelled" => {
            state.control_job(id, "cancel").unwrap();
            JobStatus::Canceled
        }
        "failed" => {
            fs::remove_file(scan).unwrap();
            fs::remove_file(batch.join("hold")).unwrap();
            JobStatus::Failed
        }
        "finished" => {
            fs::remove_file(batch.join("hold")).unwrap();
            JobStatus::Completed
        }
        "running" | "interrupted" => {
            fs::remove_file(batch.join("hold")).unwrap();
            wait(&mut state, "solver iterations", |s| {
                s.project.as_ref().unwrap().document.jobs.iter().any(|j| {
                    j.id == id
                        && j.telemetry
                            .tasks
                            .iter()
                            .any(|t| t.unit == "iterations" && t.done > 0)
                })
            });
            if scene == "interrupted" {
                fs::write(batch.join("fail-requeue"), "").unwrap();
                let runner: Value =
                    serde_json::from_slice(&fs::read(dir.join("runner.json").lexical()).unwrap())
                        .unwrap();
                assert!(
                    Command::new("kill")
                        .arg("-TERM")
                        .arg(runner["pid"].as_u64().unwrap().to_string())
                        .status()
                        .unwrap()
                        .success()
                );
                JobStatus::Interrupted
            } else {
                JobStatus::Running
            }
        }
        _ => unreachable!(),
    };
    wait(&mut state, scene, |s| {
        s.model.jobs.iter().any(|j| {
            j.id == id
                && j.status == wanted
                && match scene {
                    "interrupted" => j.can_resume,
                    "running" => s.live.frame().is_some_and(|frame| frame.info.job == id),
                    "finished" => s.imaging.plane(1).is_some_and(|plane| plane.lossless),
                    _ => true,
                }
        })
    });
    state.ui.activity.show(vec![id.into()]);
    state.refresh_model();
    // Freeze the observed UI. Close this fixture's live allocation before its temporary
    // scheduler directory goes away; the native capture runs without an application poll.
    if wanted.active() {
        if scene == "unknown" {
            fs::remove_file(batch.join("offline")).unwrap();
        }
        state
            .file_reads()
            .unwrap()
            .request(json!({"type":"job_control","directory":dir,"action":"cancel"}))
            .unwrap();
        let until = Instant::now() + Duration::from_secs(30);
        while fs::read_to_string(batch.join("100/state"))
            .unwrap_or_default()
            .trim()
            != "CANCELLED"
        {
            assert!(Instant::now() < until);
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    let (out, size, scale): (PathBuf, String, String) =
        serde_json::from_str(&std::env::var("TOMOJAX_CONNECTION_CAPTURES").unwrap()).unwrap();
    let path = out.join(format!("activity-{scene}.png"));
    tests::render_native_viewport(state, &path, &size, &scale);
    let tree = fs::read_to_string(path.with_extension("tree.json")).unwrap();
    let control = match wanted {
        JobStatus::Completed => "open-job-result",
        JobStatus::Interrupted => "resume-job",
        JobStatus::Unknown => "abandon-job",
        JobStatus::Failed | JobStatus::Canceled => "retry-job",
        _ => "cancel-job",
    };
    assert!(
        tree.contains("jobs-drawer")
            && tree.contains(&format!("{control}-{id}"))
            && tree.contains("Scheduler options"),
        "{scene}: drawer, options or action missing"
    );
}

mod zoom;
