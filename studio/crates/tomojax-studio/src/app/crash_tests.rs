//! Kill a real Studio application state on each side of its two durable writes.
use super::*;
use crate::project::JobStatus;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::process::Command;

#[test]
#[ignore = "kills four Studio clients at controlled project writer boundaries"]
fn killed_studio_reconciles_each_save_boundary() {
    if let Ok(root) = std::env::var("TOMOJAX_CRASH_PROJECT") {
        return crash_child(
            Path::new(&root),
            &std::env::var("TOMOJAX_CRASH_BOUNDARY").unwrap(),
        )
        .unwrap();
    }
    for boundary in [
        "capture-before",
        "capture-after",
        "adopt-before",
        "adopt-after",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("project");
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "app::crash_tests::killed_studio_reconciles_each_save_boundary",
                "--ignored",
            ])
            .env("TOMOJAX_CRASH_PROJECT", &root)
            .env("TOMOJAX_CRASH_BOUNDARY", boundary)
            .env("TOMOJAX_STUDIO_SETTINGS", temp.path().join("settings.json"))
            .spawn()
            .unwrap();
        let until = Instant::now() + Duration::from_secs(60);
        while !temp.path().join("kill-now").exists() {
            assert!(
                child.try_wait().unwrap().is_none() && Instant::now() < until,
                "{boundary}"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        child.kill().unwrap();
        assert!(!child.wait().unwrap().success());
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();

        let mut state = AppState::init();
        state.open_project(&root, false).unwrap();
        let saved = state.project.as_ref().unwrap();
        assert_eq!(
            saved.document.assets.len(),
            usize::from(boundary == "adopt-after")
        );
        assert_eq!(
            saved.document.jobs.len(),
            usize::from(boundary != "capture-before")
        );
        let id = saved.document.jobs.first().map(|j| j.id.clone());
        if let Some(id) = id {
            let until = Instant::now() + Duration::from_secs(30);
            while !state.job_seen(&id) {
                state.poll();
                assert!(
                    Instant::now() < until,
                    "{boundary}: {:?}",
                    state.model.error
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            let p = state.project.as_ref().unwrap();
            if boundary == "capture-after" {
                assert_eq!(p.document.jobs[0].status, JobStatus::Interrupted);
            } else {
                assert_eq!(p.document.jobs[0].status, JobStatus::Completed);
                assert_eq!(p.document.assets.len(), 1);
                assert_eq!(
                    p.document.jobs[0].result.as_ref(),
                    Some(&p.document.assets[0].id)
                );
                let log = fs::read_to_string(p.job_directory(&id).join("replies.jsonl").lexical())
                    .unwrap();
                assert_eq!(log.matches("\"type\":\"attempt\"").count(), 1);
            }
        }
        if boundary.starts_with("capture") {
            assert!(fs::read_dir(root.join("jobs")).unwrap().next().is_none());
        }
        state.shutdown();
    }
}

fn crash_child(root: &Path, boundary: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut state = AppState::init();
    state.open_project(root, true)?;
    let until = Instant::now() + Duration::from_secs(45);
    while !state.model.backend.ready {
        state.poll();
        assert!(Instant::now() < until);
        std::thread::sleep(Duration::from_millis(5));
    }
    fs::create_dir_all(root.join("jobs"))?;
    state.project.as_mut().ok_or("project closed")?.flush()?;
    // The pre-save cases deliberately prevent the writer's atomic rename. Job directories
    // remain writable, so a premature submit or an unpersisted receipt is observable.
    let block_writer = || fs::set_permissions(root, fs::Permissions::from_mode(0o500));
    if boundary == "capture-before" {
        block_writer()?;
    }
    state.set_operation("simulate", Some(&json!({"size":12,"n_views":18,"seed":9})));
    state.queue_operation("simulate")?;
    let saved = state.project.as_mut().ok_or("project closed")?.flush();
    if boundary == "capture-before" {
        assert!(saved.is_err());
        state.poll();
    } else {
        saved?;
    }
    if boundary.starts_with("adopt") {
        if boundary == "adopt-before" {
            block_writer()?;
        }
        loop {
            state.poll();
            let job = &state
                .project
                .as_ref()
                .ok_or("project closed")?
                .document
                .jobs[0];
            if job.status == JobStatus::Completed {
                break;
            }
            assert!(
                Instant::now() < until && job.status.active(),
                "{boundary}: {:?} {:?}",
                job.error,
                state.model.error
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        let saved = state.project.as_mut().ok_or("project closed")?.flush();
        assert_eq!(saved.is_ok(), boundary == "adopt-after");
    }
    fs::write(
        root.parent()
            .ok_or("missing fixture parent")?
            .join("kill-now"),
        [],
    )?;
    loop {
        std::thread::park();
    }
}
