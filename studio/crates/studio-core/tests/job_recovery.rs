//! Exercise receipt replay against the project writer and the on-disk manifest.
use proptest::prelude::*;
use serde_json::json;
use std::path::Path;
use studio_core::project::{Asset, Job, JobStatus, Project, receipt::Reading, script::Script};

fn submit(p: &mut Project, op: &str, params: serde_json::Value, inputs: &[String]) -> String {
    let inputs = inputs.iter().map(|i| p.capture(i).unwrap()).collect();
    let mut job = Job::new(op, params, inputs);
    let intent = job.inputs.first().map(|i| {
        let a = p.document.asset_mut(&i.asset).unwrap();
        a.intent += 1;
        a.intent
    });
    job.script = Some(Script::build(&job, false, String::new(), intent).unwrap());
    let id = job.id.clone();
    p.document.jobs.push(job);
    id
}
fn row(p: &Project, id: &str, output: serde_json::Value) -> Reading {
    let mut receipt = output;
    receipt["manifest"] = json!({"operation":"open","parameters":{},"inputs":[]});
    serde_json::from_value(json!({"directory":p.job_directory(id),"launcher":"process","state":"completed","detail":"","cancel_requested":false,"receipt":receipt,"offset":87})).unwrap()
}
fn product(p: &Project, id: &str) -> Reading {
    row(
        p,
        id,
        json!({"type":"dataset","product":"simulation","path":p.job_directory(id).join("result/result.nxs"),"arrays":{}}),
    )
}

#[test]
fn kill_before_and_after_the_capture_and_adoption_saves() {
    const FLAG: &str = "TOMOJAX_JOB_SAVE_BOUNDARY";
    if let Ok(boundary) = std::env::var(FLAG) {
        let path = std::env::var_os("TOMOJAX_JOB_TEST_PROJECT").unwrap();
        let mut p = Project::open(Path::new(&path), true).unwrap();
        p.flush().unwrap();
        let id = submit(&mut p, "simulate", json!({}), &[]);
        if boundary != "capture-before" {
            p.changed().unwrap();
            p.flush().unwrap();
            if boundary.starts_with("adopt") {
                let row = product(&p, &id);
                p.apply_job(&id, 0, &row, None).unwrap();
                if boundary == "adopt-after" {
                    p.changed().unwrap();
                    p.flush().unwrap();
                }
            }
        }
        // Terminate without Project::drop or a graceful Studio shutdown.
        std::process::exit(23);
    }
    for boundary in [
        "capture-before",
        "capture-after",
        "adopt-before",
        "adopt-after",
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("project");
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "kill_before_and_after_the_capture_and_adoption_saves",
            ])
            .env(FLAG, boundary)
            .env("TOMOJAX_JOB_TEST_PROJECT", &path)
            .status()
            .unwrap();
        assert_eq!(child.code(), Some(23));
        let mut p = Project::open(&path, false).unwrap();
        if boundary == "capture-before" {
            assert!(p.document.jobs.is_empty());
            continue;
        }
        let id = p.document.jobs[0].id.clone();
        assert_eq!(
            p.document.assets.len(),
            usize::from(boundary == "adopt-after")
        );
        for _ in 0..2 {
            let offset = p.document.jobs[0].script.as_ref().unwrap().offset;
            let row = product(&p, &id);
            p.apply_job(&id, offset, &row, None).unwrap();
            p.changed().unwrap();
            p.flush().unwrap();
            drop(p);
            p = Project::open(&path, false).unwrap();
        }
        assert_eq!(p.document.assets.len(), 1);
        assert_eq!(p.document.jobs[0].status, JobStatus::Completed);
        assert_eq!(
            p.document.jobs[0].result.as_ref(),
            Some(&p.document.assets[0].id)
        );
    }
}

fn calibration(project: &mut Project, id: &str) {
    let reading = row(project, id, json!({"type":"source","product":"frames"}));
    project.apply_job(id, 0, &reading, None).unwrap();
}

fn exercise_recovery(steps: Vec<u8>, reverse: bool) {
    let tmp = tempfile::tempdir().unwrap();
    let mut p = Project::open(tmp.path(), true).unwrap();
    let source = Asset::input("raw.h5".into(), false);
    let asset = source.id.clone();
    p.document.assets.push(source);
    let input = std::slice::from_ref(&asset);
    let inspect = submit(&mut p, "inspect", json!({}), input);
    p.document
        .asset_mut(&asset)
        .unwrap()
        .relink("new.h5".into(), tmp.path());
    let older = submit(&mut p, "open", json!({"flats":900}), input);
    let later = submit(&mut p, "open", json!({"flats":1000}), input);
    let sim = submit(&mut p, "simulate", json!({}), &[]);
    p.changed().unwrap();
    p.flush().unwrap();

    for event in steps {
        match event {
            0 => calibration(&mut p, &later),
            1 => calibration(&mut p, &older),
            2 => {
                let stale = row(
                    &p,
                    &inspect,
                    json!({
                        "type":"report", "product":"inspection", "report":{"report":{"source":"stale"}}
                    }),
                );
                p.apply_job(&inspect, 0, &stale, None).unwrap();
            }
            3 => {
                let reading = product(&p, &sim);
                p.apply_job(&sim, 0, &reading, None).unwrap();
            }
            4 => {
                p.changed().unwrap();
                p.flush().unwrap();
            }
            _ => {
                drop(p);
                p = Project::open(tmp.path(), false).unwrap();
            }
        }
        let source = p.document.asset(&asset).unwrap();
        assert!(source.inspection.is_null());
        assert_ne!(source.open_parameters, json!({"flats":900}));
    }
    let order = if reverse {
        [&later, &older]
    } else {
        [&older, &later]
    };
    for id in order {
        calibration(&mut p, id);
    }
    let reading = product(&p, &sim);
    p.apply_job(&sim, 0, &reading, None).unwrap();
    assert_eq!(
        p.document.asset(&asset).unwrap().open_parameters,
        json!({"flats":1000})
    );
    assert_eq!(p.document.assets.len(), 2);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]
    #[test]
    fn repeated_pages_and_autosaves_keep_the_latest_intent(
        steps in prop::collection::vec(0u8..6, 1..40),
        reverse in any::<bool>(),
    ) {
        exercise_recovery(steps, reverse);
    }
}
