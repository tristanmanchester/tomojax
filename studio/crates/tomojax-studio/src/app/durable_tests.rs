use super::*;
use crate::project::{Job, JobStatus};
use std::process::Command;

fn finish(state: &mut AppState, id: &str) -> Job {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        state.poll();
        let job = state
            .project
            .as_ref()
            .unwrap()
            .document
            .jobs
            .iter()
            .find(|j| j.id == id)
            .unwrap();
        if !job.status.active() {
            assert_eq!(
                job.status,
                JobStatus::Completed,
                "{}: {:?}; {:?}",
                job.operation,
                job.error,
                state.model.error
            );
            return job.clone();
        }
        assert!(
            Instant::now() < deadline,
            "{}: {:?}; {:?}",
            job.operation,
            job.stage,
            state.model.error
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn run(state: &mut AppState, op: &str, params: Value, assets: &[&str]) -> Job {
    let inputs = assets
        .iter()
        .map(|id| state.project.as_ref().unwrap().capture(id).unwrap())
        .collect();
    let id = state
        .capture_jobs(vec![Job::new(op, params, inputs)])
        .unwrap()
        .pop()
        .unwrap();
    finish(state, &id)
}

#[test]
#[ignore = "runs every durable operation against a real CPU executor"]
fn script_products_and_projection_roles_survive_reopening() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let python = std::env::var_os("TOMOJAX_TEST_PYTHON").unwrap();
    let raw = root.join("raw.h5");
    let make=Command::new(&python).args(["-c","import sys,h5py,numpy as n; f=h5py.File(sys.argv[1],'w'); f['exchange/data']=n.random.default_rng(7).integers(430,980,(17,5,13),dtype=n.uint16); f['exchange/theta']=n.linspace(0,180,17,endpoint=False); f.close()"]).arg(&raw).status().unwrap();
    assert!(make.success());
    let mut state = AppState::init();
    state.open_project(&root.join("project"), true).unwrap();
    let asset = Asset::input(raw.clone(), false);
    let raw_id = asset.id.clone();
    state.project.as_mut().unwrap().document.assets.push(asset);
    let open = json!({"kind":"frames","flats":1000,"darks":10,"detector":{"nu":13,"nv":5,"du":1.,"dv":1.},"geometry":{"kind":"parallel"}});
    let inspected = run(&mut state, "inspect", json!({}), &[&raw_id]);
    assert!(inspected.result.is_none());
    assert!(
        state
            .project
            .as_ref()
            .unwrap()
            .document
            .asset(&raw_id)
            .unwrap()
            .inspection
            .is_object()
    );
    let frames = run(&mut state, "open", open.clone(), &[&raw_id]);
    assert_eq!(frames.result.as_ref(), Some(&raw_id));
    let capture = state.project.as_ref().unwrap().capture(&raw_id).unwrap();
    assert_eq!(capture.kind, "frames");
    assert_eq!(capture.open_parameters, open);
    let corrected = run(&mut state, "correct", json!({"epsilon":1e-6}), &[&raw_id])
        .result
        .unwrap();
    let reopened = run(&mut state, "open", json!({"kind":"scan"}), &[&corrected])
        .result
        .unwrap();
    assert_ne!(reopened, corrected);
    let plan = run(&mut state, "plan", json!({"method":"fbp"}), &[&corrected]);
    assert!(plan.stage.contains("reconstruction_plan"));
    for (op, params) in [
        ("crop", json!({"rows":[0,4],"cols":[1,12]})),
        ("bin", json!({"binning":2})),
        ("select", json!({"views":[0,1,2,3]})),
        ("poses", json!({"poses":null})),
    ] {
        let job = run(&mut state, op, params, &[&corrected]);
        assert_eq!(
            state
                .project
                .as_ref()
                .unwrap()
                .document
                .asset(job.result.as_ref().unwrap())
                .unwrap()
                .parent
                .as_ref(),
            Some(&corrected)
        );
    }
    let params = json!({"method":"fbp","config":{"type":"FBPConfig","backend":"jax","exterior":"zero"},"grid":{"nx":12,"ny":11,"nz":5,"vx":1.,"vy":1.,"vz":1.,"vol_origin":[-5.,-5.5,-2.]}});
    let slabs = run(&mut state, "reconstruct_slabs", params.clone(), &[&raw_id]);
    assert!(slabs.result.is_some());
    let _corrected_volume = run(&mut state, "reconstruct", params, &[&corrected])
        .result
        .unwrap();
    let align = run(
        &mut state,
        "align",
        json!({"method":"joint","fit":"translations","config":{"type":"AlignConfig","levels":[1],"iterations":2,"initial_image_iterations":2,"image_iterations":2,"refine_iterations":1,"tolerance":0.,"min_iterations":1}}),
        &[&corrected],
    );
    assert!(align.result.is_some());
    let geometry = run(
        &mut state,
        "simulate",
        json!({"phantom":"sphere","geometry":"cone","size":12,"n_views":19,"seed":9}),
        &[],
    )
    .result
    .unwrap();
    let second = run(&mut state, "open", json!({"kind":"scan"}), &[&geometry])
        .result
        .unwrap();
    let combined = run(&mut state, "combine", json!({}), &[&second, &geometry]);
    assert!(combined.result.is_some());
    let original = run(
        &mut state,
        "simulate",
        json!({"size":10,"n_views":17,"geometry":"cone","seed":2}),
        &[],
    )
    .result
    .unwrap();
    let volume = run(
        &mut state,
        "reconstruct",
        json!({"method":"fbp"}),
        &[&original],
    )
    .result
    .unwrap();
    let projected = run(&mut state, "project", json!({}), &[&volume, &geometry]);
    let projected_id = projected.result.unwrap();
    let p = state.project.as_ref().unwrap();
    let image = p.resolve(&p.document.asset(&volume).unwrap().path);
    let geom = p.resolve(&p.document.asset(&geometry).unwrap().path);
    let fwd = p.document.asset(&projected_id).unwrap();
    assert_eq!(fwd.parent.as_ref(), Some(&geometry));
    assert_eq!(fwd.provenance.as_ref().unwrap().operation, "project");
    assert!(p.document.asset(&geometry).unwrap().simulated);
    assert_eq!(
        fwd.provenance.as_ref().unwrap().parameters["grid"]["nx"],
        10
    );
    let fwd_path = p.resolve(&fwd.path);
    let projected_axes =
        serde_json::to_string(&p.capture(&projected_id).unwrap().arrays["projections"].axes)
            .unwrap();
    state.shutdown();
    drop(state);
    let mut state = AppState::init();
    state.open_project(&root.join("project"), false).unwrap();
    let back = run(
        &mut state,
        "backproject",
        json!({}),
        &[&geometry, &projected_id],
    )
    .result
    .unwrap();
    let p = state.project.as_ref().unwrap();
    let back = p.document.asset(&back).unwrap();
    assert_eq!(back.parent.as_ref(), Some(&projected_id));
    assert_eq!(back.provenance.as_ref().unwrap().operation, "backproject");
    let check = Command::new(&python)
        .args([
            "-c",
            r#"
import json,sys,numpy as np,tomojax as tj
from tomojax.core.registration import volume_axes
import msgspec
v=tj.load_reconstruction(sys.argv[1])
s=tj.load_simulation(sys.argv[2]).scan
p=np.load(sys.argv[3]); b=np.load(sys.argv[4])
np.testing.assert_array_equal(p,tj.project(s.geometry,v.volume,grid=v.grid))
np.testing.assert_array_equal(b,tj.backproject(s.geometry,p))
view,v_axis,u_axis=json.loads(sys.argv[5])
np.testing.assert_array_equal(np.load(view["coordinates"]["values"]["path"]),s.angles)
for axis,pitch,count,center in [(v_axis,s.detector.dv,s.detector.nv,s.detector.center[1]),(u_axis,s.detector.du,s.detector.nu,s.detector.center[0])]:
    assert axis["coordinates"]==dict(type="uniform",spacing=pitch,origin=center-(count-1)*pitch/2)
volume=json.loads(sys.argv[6]);assert volume["shape"]==list(b.shape)
assert volume["axes"]==msgspec.json.decode(msgspec.json.encode(volume_axes(s.grid,volume["axes"][0]["unit"])))
"#,
        ])
        .arg(image.lexical())
        .arg(geom.lexical())
        .arg(fwd_path.lexical())
        .arg(p.resolve(&back.path).lexical())
        .arg(projected_axes)
        .arg(serde_json::to_string(&back.arrays["volume"]).unwrap())
        .output()
        .unwrap();
    assert!(
        check.status.success(),
        "{}",
        String::from_utf8_lossy(&check.stderr)
    );
    let save = root.join("saved.nxs");
    let files = run(&mut state, "save", json!({"path":save}), &[&volume]);
    assert!(save.is_file());
    assert_eq!(files.outputs[0].lexical(), save);
    let tiff = root.join("volume.tif");
    let exported = run(
        &mut state,
        "export",
        json!({"path":tiff, "member":"volume"}),
        &[&volume],
    );
    assert!(tiff.is_file());
    assert!(exported.outputs.iter().any(|p| p.lexical() == tiff));
    state.shutdown();
}

#[test]
fn late_receipt_is_read_after_the_scheduler_reports_interruption() {
    let tmp = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state.project = Some(Project::open(tmp.path(), true).unwrap());
    let id = state
        .capture_jobs(vec![Job::new("simulate", json!({}), vec![])])
        .unwrap()
        .pop()
        .unwrap();
    state.jobs = JobSession::default(); // A newly connected reader.
    state.fail_job(&id, "Submit reply lost", false);
    state.job_reply(Pending::List, ResultValue::Jobs { jobs: vec![] });
    assert!(state.jobs.reads.contains_key(&id));
    assert!(state.jobs.waiting.is_empty());
    let dir = state.project.as_ref().unwrap().job_directory(&id);
    let mut row = json!({"state":"interrupted","detail":"COMPLETED","cancel_requested":false,"launcher":"slurm","directory":dir,"offset":1024});
    let read = |state: &mut AppState, offset, row: &Value| {
        state.job_reply(
            Pending::Read(id.clone(), offset),
            ResultValue::Job(Box::new(serde_json::from_value(row.clone()).unwrap())),
        )
    };
    read(&mut state, 0, &row);
    assert!(state.jobs.reads.contains_key(&id));
    row["state"] = json!("completed");
    row["receipt"] = json!({"type":"dataset","product":"simulation","path":dir.join("result/result.nxs"),"arrays":{},"manifest":{"operation":"simulate","parameters":{},"inputs":[]}});
    read(&mut state, 1024, &row);
    read(&mut state, 1024, &row);
    let p = state.project.as_ref().unwrap();
    assert_eq!(p.document.jobs[0].status, JobStatus::Completed);
    assert_eq!(p.document.assets.len(), 1);
}

#[test]
#[ignore = "pauses a real inline solve, cancels it and resumes its checkpoint"]
fn resume_a_paused_job_and_a_canceled_checkpoint() {
    let tmp = tempfile::tempdir().unwrap();
    let mut state = AppState::init();
    state.open_project(tmp.path(), true).unwrap();
    let scan = run(
        &mut state,
        "simulate",
        json!({"size":20,"n_views":30,"seed":7}),
        &[],
    )
    .result
    .unwrap();
    let input = state.project.as_ref().unwrap().capture(&scan).unwrap();
    let checkpoint = tmp.path().join("solve-checkpoint");
    let settings = json!({"method":"sirt","iterations":360,"checkpoint":checkpoint});
    let id = state
        .capture_jobs(vec![Job::new("reconstruct", settings, vec![input])])
        .unwrap()
        .pop()
        .unwrap();
    let wait = |s: &mut AppState, why: &str, predicate: &dyn Fn(&Job) -> bool| {
        let until = Instant::now() + Duration::from_secs(120);
        loop {
            s.poll();
            let job = s
                .project
                .as_ref()
                .unwrap()
                .document
                .jobs
                .iter()
                .find(|j| j.id == id)
                .unwrap();
            if predicate(job) {
                break;
            }
            assert!(
                Instant::now() < until && job.status.active(),
                "{why}: {:?}, {:?}, {:?}",
                job.status,
                job.error,
                s.model.error
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    };
    wait(&mut state, "solver began", &|j| {
        j.telemetry
            .tasks
            .iter()
            .any(|t| t.unit == "iterations" && t.done > 0)
    });
    state.control_job(&id, "pause").unwrap();
    wait(&mut state, "paused", &|j| j.status == JobStatus::Paused);
    state.control_job(&id, "resume").unwrap();
    wait(&mut state, "resumed", &|j| {
        j.status == JobStatus::Running && j.script.as_ref().is_some_and(|s| !s.pause_requested)
    });
    state.control_job(&id, "cancel").unwrap();
    wait(&mut state, "checkpoint saved", &|j| {
        j.status == JobStatus::Canceled
            && j.script
                .as_ref()
                .and_then(|s| s.checkpoint.as_ref())
                .is_some_and(|c| c.usable)
    });
    state.control_job(&id, "resume").unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    let resumed = loop {
        state.poll();
        let latest = state
            .project
            .as_ref()
            .unwrap()
            .document
            .jobs
            .last()
            .unwrap();
        if latest.id != id {
            break latest;
        }
        assert!(
            Instant::now() < deadline,
            "checkpoint confirmation: {:?}",
            state.model.error
        );
        std::thread::sleep(Duration::from_millis(5));
    };
    let lines = &resumed.script.as_ref().unwrap().lines;
    assert_eq!(
        lines[1]["command"]["operation"]["checkpoint"],
        json!(checkpoint)
    );
    let resumed_id = resumed.id.clone();
    assert!(finish(&mut state, &resumed_id).result.is_some());
    state.shutdown();
}
