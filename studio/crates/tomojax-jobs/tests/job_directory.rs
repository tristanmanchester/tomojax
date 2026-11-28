use serde_json::{Value, json};
use std::{fs, io::Write, sync::Barrier, thread};
use tomojax_jobs::{
    directory::read_files,
    log::{Log, page},
    publication::{self, Publication},
    script::{Launcher, prepare, publish_script},
    slurm::resources,
};

#[test]
fn competing_submitters_publish_one_script() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path().join("job");
    let barrier = Barrier::new(12);
    let submitted = thread::scope(|scope| {
        let tasks = (0..12)
            .map(|i| {
                let barrier = &barrier;
                let dir = &dir;
                scope.spawn(move || {
                    let req = json!({"id":"job.scan","command":{"type":"open","path":i%2}});
                    barrier.wait();
                    publish_script(dir, &[req]).is_ok()
                })
            })
            .collect::<Vec<_>>();
        tasks
            .into_iter()
            .filter_map(|t| t.join().ok())
            .filter(|won| *won)
            .count()
    });
    assert_eq!(submitted, 6);
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
}

#[test]
fn paging_leaves_torn_bytes_for_the_next_attempt() {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    let mut writer = Log::open(p).unwrap();
    let big = json!({"type":"attempt","number":1,"message":"x".repeat(1024)});
    writer.append(&big).unwrap();
    let advanced = |done| json!({"type":"progress","request":"job.r","record":{"event":"advanced","task":2,"done":done}});
    writer.append(&advanced(1)).unwrap();
    writer.append(&advanced(2)).unwrap();
    let (first, offset) = page(p, 0, 2).unwrap();
    assert_eq!(first, vec![big]);
    let (next, end) = page(p, offset, 4096).unwrap();
    assert_eq!(next, vec![advanced(2)]);
    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(p.join("replies.jsonl"))
        .unwrap();
    file.write_all(b"{\"type\":\"progress\",\"message\":\"\xe2\x82")
        .unwrap();
    assert_eq!(page(p, end, 1024).unwrap(), (vec![], end));
    assert_eq!(read_files(p).unwrap().attempt, 1);
    let mut restarted = Log::open(p).unwrap();
    let new = json!({"type":"attempt","number":2});
    restarted.append(&new).unwrap();
    assert_eq!(page(p, end, 1).unwrap().0, vec![new]);
    assert_eq!(read_files(p).unwrap().attempt, 2);
}

#[test]
fn a_stalled_reader_cannot_mix_preview_generations() {
    let root = tempfile::tempdir().unwrap();
    let original = root.path().join("input.npy");
    for n in 1..=3 {
        fs::write(&original, [n]).unwrap();
        let array = json!({"storage":{"type":"npy","path":original,"artifact":"a"},"axes":[{"coordinates":{"type":"coordinates","values":{"type":"npy","path":original,"artifact":"a"}}}]});
        let row = json!({"type":"progress","record":{"event":"preview","arrays":{"plane":array,"poses":array}}});
        publication::preview(root.path(), 2, n.into(), row).unwrap();
    }
    assert!(!root.path().join("preview/2-1/array-0.npy").exists());
    assert_eq!(
        fs::read(root.path().join("preview/2-2/array-0.npy")).unwrap(),
        [2]
    );
    let latest = publication::current_preview(root.path()).unwrap().unwrap();
    assert_eq!(latest["generation"], "2-3");
    assert!(!latest.to_string().contains("artifact"));
    let poses = latest
        .pointer("/record/arrays/poses/storage/path")
        .unwrap()
        .as_str()
        .unwrap();
    assert_eq!(fs::read(poses).unwrap(), [3]);
}

#[test]
fn info_arrays_coordinates_and_result_files_survive_work_removal() {
    let root = tempfile::tempdir().unwrap();
    let dir = root.path();
    let stage = dir.join("work/result");
    fs::create_dir_all(&stage).unwrap();
    let src = dir.join("work/session.npy");
    fs::write(&src, b"samples").unwrap();
    let storage = json!({"type":"npy","path":src,"artifact":"private"});
    let descriptor =
        json!({"storage":storage,"axes":[{"coordinates":{"type":"coordinates","values":storage}}]});
    let product = json!({"type":"output","value":{"type":"volume","volume":descriptor},
        "info":{"arrays":{"/solver/trace":descriptor},"summary":{}},"manifest":{"operation":"backproject","parameters":{"source":{"type":"member","member":"projections","handle":"job.project"}}}});
    let receipt =
        publication::receipt(dir, &json!({"type":"run"}), product, &|_| panic!()).unwrap();
    publication::publish(dir, &receipt).unwrap();
    fs::remove_dir_all(dir.join("work")).unwrap();
    assert!(!receipt.to_string().contains("private"));
    assert!(!receipt.to_string().contains("\"handle\""));
    for path in [
        "/arrays/volume/storage/path",
        "/info/arrays/~1solver~1trace/storage/path",
        "/arrays/volume/axes/0/coordinates/values/path",
    ] {
        assert_eq!(
            fs::read(receipt.pointer(path).unwrap().as_str().unwrap()).unwrap(),
            b"samples"
        );
    }
    assert_eq!(fs::read_dir(dir.join("result")).unwrap().count(), 2);
}

#[test]
fn publication_does_not_accept_an_in_memory_member_as_file_storage() {
    let tmp = tempfile::tempdir().unwrap();
    let mut p = Publication::new(tmp.path(), tmp.path());
    assert!(
        p.arrays(&mut json!({"storage":{"type":"member","handle":"a","member":"volume"}}))
            .is_err()
    );
}

#[test]
fn only_checkpointed_lines_and_resource_options_reach_the_launcher() {
    let tmp = tempfile::tempdir().unwrap();
    let line =
        |name: &str| json!({"id":"job.r","command":{"type":"run","operation":{"type":name}}});
    for op in [
        "align",
        "reconstruct",
        "reconstruct_slabs",
        "correct",
        "simulate",
    ] {
        let mut script = vec![line(op)];
        let checkpoint = prepare(tmp.path(), Launcher::Slurm, &mut script).unwrap();
        let expected = matches!(op, "align" | "reconstruct" | "reconstruct_slabs");
        assert_eq!(checkpoint.is_some(), expected);
        assert_eq!(
            script[0]["command"]["operation"]
                .get("checkpoint")
                .is_some(),
            expected
        );
    }
    let mut old: Value = line("reconstruct");
    old["command"]["operation"]["checkpoint"] = json!(tmp.path().join("other"));
    assert_eq!(
        prepare(tmp.path(), Launcher::Slurm, &mut [old])
            .unwrap()
            .unwrap(),
        tmp.path().join("other")
    );
    for forbidden in [
        "--clusters=other",
        "--array=1-2",
        "--job-name=x",
        "--output=elsewhere",
        "--wrap=true",
    ] {
        assert!(resources(&[forbidden.into()]).is_err());
    }
    assert_eq!(
        resources(&["-p".into(), "k11".into(), "--gres=gpu:1".into()]).unwrap(),
        ["--partition=k11", "--gres=gpu:1"]
    );
    let d = tmp.path().join("job");
    for input in [
        json!({"type":"npy","artifact":"token","path":"/some/file"}),
        json!({"type":"member","handle":"other.open","member":"self"}),
    ] {
        let mut bad = line("align");
        bad["command"]["operation"]["deep"] = json!({"from":[input]});
        assert!(publish_script(&d, &[bad]).is_err());
        assert!(!d.exists());
    }
}

#[test]
fn polling_never_contends_with_launch_or_with_other_readers() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use tomojax_jobs::locks::{held, try_lock};
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("lock");
    assert!(!held(&path).unwrap());
    assert!(!path.exists(), "a read created a job lock");
    let done = AtomicBool::new(false);
    let barrier = Barrier::new(5);
    thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                barrier.wait();
                while !done.load(Ordering::Acquire) {
                    held(&path).unwrap();
                }
            });
        }
        barrier.wait();
        let mut denied = false;
        for _ in 0..500 {
            let guard = try_lock(&path).unwrap();
            if guard.is_none() {
                denied = true;
                break;
            }
            assert!(held(&path).unwrap());
            drop(guard);
        }
        done.store(true, Ordering::Release);
        assert!(!denied, "a reader denied a new runner the lock");
    });
    assert!(!held(&path).unwrap());
}
