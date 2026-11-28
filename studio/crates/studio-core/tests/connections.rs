//! Exercise the transport as a separate client, with a fake system ssh on PATH.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use studio_core::connection::connect;
use tempfile::TempDir;

// Re-enter this test in a fresh process so each SSH client gets its own PATH,
// signal handlers and inherited stdio, just as tomojax-worker connect does.
#[test]
fn driver() {
    let Ok(profile) = std::env::var("TEST_PROFILE") else {
        return;
    };
    let worker = std::env::var("TEST_WORKER").unwrap();
    let helper = std::env::var_os("TEST_ASKPASS").map(PathBuf::from);
    let mut args = vec![profile];
    if let Ok(action) = std::env::var("TEST_ACTION") {
        args.push(format!("--{action}"));
    }
    let code = connect(&args, |_| Ok(worker), helper.as_deref()).unwrap_or_else(|e| {
        eprintln!("{e}");
        e.status as i32
    });
    std::process::exit(code);
}

fn script(path: &Path, contents: &str) {
    // A concurrent test can fork between open and close. Write in a separate
    // process so it cannot inherit a writable executable and cause ETXTBSY.
    use std::{io::Write, process::Stdio};
    let mut writer = Command::new("sh")
        .args(["-c", r#"cat > "$1" && chmod 700 "$1""#, "write-script"])
        .arg(path)
        .stdin(Stdio::piped())
        .spawn()
        .unwrap();
    writer
        .stdin
        .take()
        .unwrap()
        .write_all(contents.as_bytes())
        .unwrap();
    assert!(writer.wait().unwrap().success());
}

struct Fixture {
    root: TempDir,
    config: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let config = root.path().join("tomojax");
        fs::create_dir(&config).unwrap();
        fs::create_dir(root.path().join("run")).unwrap();
        script(
            &root.path().join("ssh"),
            r#"#!/bin/sh
if [ "$1" = "-G" ]; then echo "hostname $3"; echo "controlpath /user/own-socket"; echo "controlpersist 3600"; echo "controlmaster auto"; exit; fi
printf '%s\0' "$@" >> "$LOG"
while [ "$1" != "--" ]; do shift; done
shift 2
exec sh -c "$1"
"#,
        );
        script(
            &root.path().join("worker ' ;"),
            r#"#!/bin/sh
printf '%s\0' "$@" > "$ARGUMENTS"
printf '%s\n' '{"type":"ready","protocol":1,"machine":"fixture","home":"/","directory":"/","host":"test","capabilities":[],"jobs":"inline"}'
printf '\000\377abc\n'
"#,
        );
        Self { root, config }
    }
    fn command(&self, profile: &str) -> Command {
        let mut c = Command::new(std::env::current_exe().unwrap());
        c.args(["--exact", "driver", "--nocapture"])
            .env("TEST_PROFILE", profile)
            .env("TEST_WORKER", self.root.path().join("worker ' ;"))
            .env("ARGUMENTS", self.root.path().join("arguments"))
            .env("XDG_CONFIG_HOME", self.root.path())
            .env("XDG_RUNTIME_DIR", self.root.path().join("run"))
            .env("LOG", self.root.path().join("argv"))
            .env(
                "PATH",
                format!("{}:/usr/bin:/bin", self.root.path().display()),
            );
        c
    }
    fn run(&self, profile: &str) -> Output {
        self.command(profile).output().unwrap()
    }
}

#[test]
fn local_one_and_two_hops_deliver_the_exact_worker_arguments() {
    for hops in [vec![], vec!["outer"], vec!["outer", "inner"]] {
        let f = Fixture::new();

        fs::write(f.config.join("connections.toml"), format!(
            "[fixture]\nssh={hops:?}\npython=\"/usr/bin/python3\"\nsession=\"env A='spaces and ; quotes'\"\nsbatch=\"-p k11\"\n"
        )).unwrap();
        let result = f.run("fixture");
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(result.stdout.ends_with(b"\x00\xffabc\n"));
        let argv = fs::read_to_string(f.root.path().join("arguments")).unwrap();
        let jobs = if hops.is_empty() { "inline" } else { "process" };
        assert_eq!(
            argv,
            format!(
                "serve\0--jobs\0{jobs}\0--idle-exit\x00120\0--sbatch\0-p k11\0--session\0env A='spaces and ; quotes'\0--python\0/usr/bin/python3\0"
            )
        );
        if !hops.is_empty() {
            let log = fs::read_to_string(f.root.path().join("argv")).unwrap();
            let options: Vec<_> = log.split('\0').collect();
            for host in &hops {
                assert!(options.windows(2).any(|pair| pair == ["--", *host]));
            }
            assert_eq!(
                options.iter().filter(|s| **s == "BatchMode=yes").count(),
                hops.len()
            );
            assert!(!options.contains(&"-S"));
            assert!(
                !options
                    .iter()
                    .any(|v| v.starts_with("ControlPath=") || v.starts_with("ControlPersist="))
            );
        } else {
            assert!(!f.root.path().join("argv").exists());
        }
    }
}

#[test]
fn scheduler_never_gates_the_handshake_and_a_missing_worker_blocks() {
    let f = Fixture::new();
    fs::write(
        f.config.join("connections.toml"),
        "[busy]\nssh=['outer']\nsession='false'\n",
    )
    .unwrap();
    assert!(f.run("busy").status.success());
    assert_eq!(
        f.command("busy")
            .env("TEST_WORKER", "/absent")
            .output()
            .unwrap()
            .status
            .code(),
        Some(5)
    );
}

mod openssh;

use proptest::prelude::*;
proptest! {
    #[test]
    fn an_argument_survives_every_shell(value in "[^\\x00-\\x08\\x0b-\\x1f\\x7f]{0,60}", depth in 0..=2usize) {
        let f = Fixture::new();
        let hops = &["outer", "inner"][..depth];
        let value_toml = serde_json::to_string(&value).unwrap();
        fs::write(
            f.config.join("connections.toml"),
            format!("[p]\nssh={hops:?}\nsession={value_toml}\n"),
        )
        .unwrap();
        let output = f.run("p");
        prop_assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let args = fs::read_to_string(f.root.path().join("arguments")).unwrap();
        prop_assert!(
            args.split_terminator('\0')
                .collect::<Vec<_>>()
                .ends_with(&["--session", &value])
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]
    #[test]
    fn reconnect_never_resends_an_admitted_request(drops in prop::collection::vec(any::<bool>(),2..12)) {
        use std::sync::Arc;
        use std::time::{Duration, Instant};
        use studio_core::connection::{Owner, State};
        let f = Fixture::new();
        let program = f.root.path().join("connector");
        let log = f.root.path().join("requests");
        // The connector records the operation *before* dropping the connection.
        // No terminal has reached the client at the instant of disconnection.
        script(
            &program,
            r#"#!/bin/sh
echo '{"type":"ready","protocol":1,"progress":2,"versions":{"tomojax":"test","jax":"test"},"capabilities":["describe","record"],"devices":[]}'
while IFS= read -r request; do
 case "$request" in *'"type":"record"'*) echo "$request" >> "$2"; case "$request" in *'"drop":true'*) exit 4;; esac;; esac
done"#,
        );
        let mut connection = Owner::new(
            program,
            log.to_string_lossy().into_owned(),
            &["record"],
            Arc::new(|| {}),
        );
        let wait = |c: &mut Owner, up: bool| {
            let end = Instant::now() + Duration::from_secs(3);
            loop {
                c.poll();
                if if up {
                    matches!(c.state(), State::Connected(_))
                } else {
                    matches!(c.state(), State::Waiting(_))
                } {
                    return;
                }
                assert!(Instant::now() < end, "{:?} {}", c.state(), c.diagnostic());
                std::thread::sleep(Duration::from_millis(2));
            }
        };
        wait(&mut connection, true);
        for (nonce, drop) in drops.iter().copied().chain([true]).enumerate() {
            connection
                .request(
                    serde_json::json!({"type":"record","nonce":nonce,"drop":drop}),
                    None,
                )
                .unwrap();
            if drop {
                wait(&mut connection, false);
                connection.retry_now();
                wait(&mut connection, true);
            }
        }
        connection.close();
        let nonces = fs::read_to_string(log)
            .unwrap()
            .lines()
            .map(|line| {
                serde_json::from_str::<serde_json::Value>(line).unwrap()["command"]["nonce"]
                    .as_u64()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        prop_assert_eq!(nonces, (0..=drops.len() as u64).collect::<Vec<_>>());
    }
}

#[test]
fn a_project_keeps_each_named_connection_and_its_login_independent() {
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use studio_core::connection::{Registry, State};
    let f = Fixture::new();
    let worker = f.root.path().join("connector");
    let a = f.root.path().join("a").to_string_lossy().into_owned();
    let b = f.root.path().join("b").to_string_lossy().into_owned();
    script(
        &worker,
        r#"#!/bin/sh
case "$3" in --login) touch "$2.login"; exit 0;; esac
echo start >> "$2"
case "$2" in */b) test -f "$2.login" || exit 3;; esac
echo '{"type":"ready","protocol":1,"machine":"machine","home":"/","directory":"/","host":"fixture","jobs":"inline","capabilities":["status"]}'
echo '{"type":"executor","state":"gone","detail":"allocation unavailable"}'
cat >/dev/null"#,
    );
    let mut registry = Registry::new(worker, &["status"], Arc::new(|| {}));
    registry.open(&a);
    registry.open(&b);
    let wait = |registry: &mut Registry, b_up: bool| {
        let until = Instant::now() + Duration::from_secs(4);
        loop {
            registry.poll();
            let first = registry.open(&a).state();
            let second = registry.open(&b).state();
            if matches!(first, State::Connected(_))
                && if b_up {
                    matches!(second, State::Connected(_))
                } else {
                    second == State::LoginNeeded
                }
            {
                break;
            }
            assert!(Instant::now() < until, "{first:?}, {second:?}");
            std::thread::sleep(Duration::from_millis(5));
        }
    };
    wait(&mut registry, false);
    registry.open(&b).login(Path::new("/unused")).unwrap();
    wait(&mut registry, true);
    registry.remove(&b);
    assert!(matches!(registry.open(&a).state(), State::Connected(_)));
    assert_eq!(fs::read_to_string(a).unwrap(), "start\n");
    assert_eq!(fs::read_to_string(b).unwrap(), "start\nstart\n");
}

#[test]
fn terminal_login_can_read_its_foreground_tty_and_restores_echo() {
    use std::{io::Write, process::Stdio};
    let f = Fixture::new();
    fs::write(f.config.join("connections.toml"), "[tty]\nssh=['outer']\n").unwrap();
    script(
        &f.root.path().join("ssh"),
        r#"#!/bin/sh
case " $* " in
*" -G "*) echo 'hostname outer'; exit 0 ;;
*" -O check "*) exit 0 ;;
esac
stty -echo </dev/tty || exit 1
IFS= read -r response </dev/tty && test "$response" = consent
"#,
    );
    let client = f.command("tty");
    let mut shell = Command::new("script");
    shell.args(["-q","-e","-c",r#"before=$(stty -g); "$DRIVER" --exact driver --nocapture && test "$before" = "$(stty -g)""#,"/dev/null"])
        .env("DRIVER",client.get_program()).env("TEST_ACTION","login").env_remove("SSH_ASKPASS_REQUIRE");
    shell.envs(client.get_envs().filter_map(|(k, v)| v.map(|v| (k, v))));
    let mut child = shell
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"consent\n").unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
}
