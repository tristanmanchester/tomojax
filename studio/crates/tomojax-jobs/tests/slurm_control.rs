use serde_json::json;
use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};
use tomojax_jobs::{
    executor::Executor,
    runner::InlineSession,
    script::Launcher,
    service::{Service, control, jobs, status},
};

struct NoSession;
impl InlineSession for NoSession {
    fn acquire(&self, _: &dyn Fn() -> bool) -> io::Result<Box<dyn Executor + '_>> {
        panic!("held batch job requested an executor")
    }
}

#[test]
fn a_reconnecting_poller_withdraws_only_the_lost_submission() {
    let flag = "TJ_TEST_LOST_SUBMISSION";
    if env::var_os(flag).is_none() {
        let root = tempfile::tempdir().unwrap();
        let scripts = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/slurm");
        let path = env::join_paths(
            [scripts]
                .into_iter()
                .chain(env::split_paths(&env::var_os("PATH").unwrap())),
        )
        .unwrap();
        let output = Command::new(env::current_exe().unwrap())
            .args([
                "--exact",
                "a_reconnecting_poller_withdraws_only_the_lost_submission",
                "--nocapture",
            ])
            .env("PATH", path)
            .env(flag, "1")
            .env("FAKE_SLURM", root.path())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let root = PathBuf::from(env::var_os("FAKE_SLURM").unwrap());
    let dir = root.join("jobs/hold");
    fs::write(root.join("hold"), "").unwrap();
    fs::write(root.join("lose-reply"), "").unwrap();
    let service = || {
        Service::new(
            Launcher::Slurm,
            "python3".into(),
            "/bin/false".into(),
            &[],
            Arc::new(NoSession),
            Arc::new(|_| panic!()),
        )
        .unwrap()
    };
    let script = [json!({"id":"hold.s","command":{"type":"run","operation":{"type":"simulate"}}})];
    assert_eq!(
        service().submit(&dir, &script, &[]).unwrap()["state"],
        "queued"
    );
    // The first poll reconstructed the ID from the name even though sbatch never returned it.
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(
            &fs::read(dir.join("submission.json")).unwrap()
        )
        .unwrap(),
        json!({"accepted":100})
    );
    assert_eq!(
        service().submit(&dir, &script, &[]).unwrap()["state"],
        "queued"
    );
    assert_eq!(fs::read_to_string(root.join("next")).unwrap().trim(), "101");
    // Another project can use the same human-chosen directory name, and both
    // sbatch replies can be lost. Recovery must still select the right allocation.
    let other = root.join("another/jobs/hold");
    service().submit(&other, &script, &[]).unwrap();
    control(&other, "cancel").unwrap();
    assert!(
        root.join("101/cancel").exists(),
        "cancel missed the second project"
    );
    assert!(
        !root.join("100/cancel").exists(),
        "cancel reached the first project"
    );
    fs::write(root.join("offline"), "").unwrap();
    assert_eq!(control(&dir, "cancel").unwrap()["state"], "unknown");
    assert!(!root.join("100/cancel").exists());
    fs::remove_file(root.join("offline")).unwrap();
    let _ = jobs(&root).unwrap();
    assert!(root.join("100/cancel").exists());
    let until = Instant::now() + Duration::from_secs(10);
    while status(&dir).unwrap()["state"] != "canceled" {
        assert!(Instant::now() < until);
        thread::sleep(Duration::from_millis(20));
    }
    fs::write(root.join("100/name"), "unrelated").unwrap();
    fs::write(root.join("100/state"), "RUNNING").unwrap();
    fs::remove_file(root.join("100/cancel")).unwrap();
    assert_eq!(jobs(&root).unwrap()[0]["state"], "canceled");
    assert!(!root.join("100/cancel").exists());
}
