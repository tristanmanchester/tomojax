//! A real, private OpenSSH daemon, run under the account executing cargo test.
use super::{Fixture, script};
use std::{
    fs,
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

struct Server {
    f: Fixture,
    sshd: Child,
    bin: PathBuf,
}
impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.ssh().args(["-O", "exit", "outer"]).output();
        let _ = self.sshd.kill();
        let _ = self.sshd.wait();
    }
}
fn success(output: Output) {
    assert!(
        output.status.success(),
        "{}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}
fn key(path: &Path, pass: &str) {
    success(
        Command::new("ssh-keygen")
            .args(["-q", "-t", "ed25519", "-N", pass, "-f"])
            .arg(path)
            .output()
            .unwrap(),
    );
}
impl Server {
    fn start() -> Option<Self> {
        let Some(sshd) = ["/usr/sbin/sshd", "/usr/bin/sshd", "/sbin/sshd"]
            .into_iter()
            .find(|p| Path::new(p).is_file())
        else {
            eprintln!("Skipping OpenSSH integration test: sshd is not installed");
            return None;
        };
        let f = Fixture::new();
        // ControlPath has a short AF_UNIX path limit, independent of TMPDIR.
        let d = f.root.path();
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        key(&d.join("host"), "");
        key(&d.join("client"), "fixture-passphrase");
        let host = d.display();
        let bin = d.join("ssh");
        script(
            &bin,
            &format!("#!/bin/sh\nexec /usr/bin/ssh -F '{host}/client.conf' \"$@\""),
        );
        fs::write(d.join("client.conf"),format!("Host outer inner\n HostName 127.0.0.1\n Port {port}\n UserKnownHostsFile '{host}/known'\n GlobalKnownHostsFile /dev/null\n IdentitiesOnly yes\n IdentityAgent none\n IdentityFile '{host}/client'\nHost inner\n ControlPath none\nHost silent\n ProxyCommand sleep 90\n")).unwrap();
        script(
            &d.join("remote"),
            &format!(
                "#!/bin/sh\nexport PATH='{host}':/usr/bin:/bin\nexec /bin/sh -c \"$SSH_ORIGINAL_COMMAND\""
            ),
        );
        fs::write(d.join("sshd.conf"),format!("Port {port}\nListenAddress 127.0.0.1\nHostKey {host}/host\nPidFile {host}/sshd.pid\nAuthorizedKeysFile {host}/client.pub\nStrictModes no\nUsePAM no\nPasswordAuthentication no\nKbdInteractiveAuthentication no\nForceCommand {host}/remote\n")).unwrap();
        let mut child = Command::new(sshd)
            .args(["-D", "-e", "-f"])
            .arg(d.join("sshd.conf"))
            .stderr(Stdio::from(fs::File::create(d.join("daemon.log")).unwrap()))
            .spawn()
            .unwrap();
        let end = Instant::now() + Duration::from_secs(5);
        while TcpStream::connect(("127.0.0.1", port)).is_err() {
            assert!(
                child.try_wait().unwrap().is_none() && Instant::now() < end,
                "{}",
                fs::read_to_string(d.join("daemon.log")).unwrap()
            );
            thread::sleep(Duration::from_millis(10));
        }
        script(
            &d.join("ask"),
            &format!(
                r#"#!/bin/sh
case "$1" in
*authenticity*) echo trust >> '{host}/prompts'; echo yes ;;
*passphrase*) if [ -f '{host}/tried' ]; then echo right >> '{host}/prompts'; echo fixture-passphrase; else touch '{host}/tried'; echo wrong >> '{host}/prompts'; echo wrong; fi ;;
*) exit 1 ;;
esac"#
            ),
        );
        script(
            &d.join("worker ' ;"),
            r#"#!/bin/sh
printf '%s\n' '{"type":"ready","protocol":1,"machine":"fixture-id","host":"fixture","home":"/","directory":"/","capabilities":["status"],"jobs":"inline"}'
printf '%s\n' '{"type":"executor","state":"starting","detail":"queued"}'
printf '%s\n' '{"type":"executor","state":"ready","detail":{"protocol":1,"progress":2,"capabilities":[],"versions":{"tomojax":"test","jax":"test"},"devices":[]}}'
cat >/dev/null
"#,
        );
        fs::write(f.config.join("connections.toml"),"[one]\nssh=['outer']\n[two]\nssh=['outer','inner']\n[silent]\nssh=['outer','silent']\n").unwrap();
        Some(Self {
            f,
            sshd: child,
            bin,
        })
    }
    fn ssh(&self) -> Command {
        use std::os::unix::fs::FileTypeExt;
        let mut c = Command::new(&self.bin);
        let sockets = self.f.root.path().join("run/tomojax");
        if let Ok(mut entries) = fs::read_dir(sockets)
            && let Some(path) = entries.find_map(|result| {
                let entry = result.ok()?;
                entry.file_type().ok()?.is_socket().then(|| entry.path())
            })
        {
            c.arg("-S").arg(path);
        }
        c
    }
    fn action(&self, profile: &str, action: &str) -> Output {
        self.f
            .command(profile)
            .env("TEST_ACTION", action)
            .env("TEST_ASKPASS", self.f.root.path().join("ask"))
            .output()
            .unwrap()
    }
}

#[test]
fn encrypted_key_login_reuse_refusals_and_silent_inner_hop() {
    let Some(server) = Server::start() else {
        return;
    };
    let d = server.f.root.path();
    success(server.action("one", "login"));
    assert_eq!(
        fs::read_to_string(d.join("prompts")).unwrap(),
        "trust\nwrong\nright\n"
    );
    let checked = server.action("one", "check");
    assert!(String::from_utf8_lossy(&checked.stdout).contains("\"host\":\"fixture\""));
    success(checked);
    for action in ["check", "login"] {
        let inner = server.action("two", action);
        assert_eq!(
            inner.status.code(),
            Some(6),
            "{}",
            String::from_utf8_lossy(&inner.stderr)
        );
        let begin = Instant::now();
        let silent = server.action("silent", action);
        assert_eq!(
            silent.status.code(),
            Some(4),
            "{}",
            String::from_utf8_lossy(&silent.stderr)
        );
        assert!(begin.elapsed() < Duration::from_secs(32));
    }
    success(server.ssh().args(["-O", "exit", "outer"]).output().unwrap());
    let start = Instant::now();
    let denied = server.action("one", "check");
    assert_eq!(
        denied.status.code(),
        Some(3),
        "{}",
        String::from_utf8_lossy(&denied.stderr)
    );
    assert!(start.elapsed() < Duration::from_secs(2));
    assert_eq!(
        fs::read_to_string(d.join("prompts")).unwrap(),
        "trust\nwrong\nright\n"
    );
}

struct Resume(i32);
impl Drop for Resume {
    fn drop(&mut self) {
        // SAFETY: the paused fixture sshd session cannot exit and reuse its PID.
        unsafe {
            libc::kill(self.0, libc::SIGCONT);
        }
    }
}
#[test]
fn a_local_master_with_a_dead_network_is_retryable() {
    let Some(server) = Server::start() else {
        return;
    };
    success(server.action("one", "login"));
    let out = server.ssh().args(["outer", "echo $PPID"]).output().unwrap();
    assert!(out.status.success());
    let pid: i32 = String::from_utf8(out.stdout)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let cmd = fs::read_to_string(format!("/proc/{pid}/comm")).unwrap();
    assert!(cmd.trim().starts_with("sshd"), "{cmd}");
    // SAFETY: the fixture returned the PID of its own live session.
    assert_eq!(unsafe { libc::kill(pid, libc::SIGSTOP) }, 0);
    let _resume = Resume(pid);
    success(
        server
            .ssh()
            .args(["-O", "check", "outer"])
            .output()
            .unwrap(),
    );
    let start = Instant::now();
    let silent = server.action("one", "check");
    assert_eq!(
        silent.status.code(),
        Some(4),
        "{}",
        String::from_utf8_lossy(&silent.stderr)
    );
    assert!(String::from_utf8_lossy(&silent.stderr).contains("first hop did not answer"));
    assert!(start.elapsed() < Duration::from_secs(32));
}

struct Agent(Child);
impl Drop for Agent {
    fn drop(&mut self) {
        // SAFETY: this fixture agent is the unreaped leader of its private group.
        unsafe {
            libc::kill(-(self.0.id() as i32), libc::SIGKILL);
        }
        let _ = self.0.wait();
    }
}

#[test]
fn an_inner_key_waiting_for_agent_confirmation_times_out() {
    let Some(server) = Server::start() else {
        return;
    };
    success(server.action("one", "login"));
    let d = server.f.root.path();
    let socket = d.join("agent.sock");
    let ask = d.join("confirm");
    let proof = d.join("confirmation-requested");
    script(
        &ask,
        &format!("#!/bin/sh\ntouch '{}'\nexec sleep 45\n", proof.display()),
    );
    use std::os::unix::process::CommandExt;
    let _agent = Agent(
        Command::new("ssh-agent")
            .process_group(0)
            .args(["-D", "-a"])
            .arg(&socket)
            .env("SSH_ASKPASS", &ask)
            .env("SSH_ASKPASS_REQUIRE", "force")
            .env("DISPLAY", "fixture")
            .stdout(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let end = Instant::now() + Duration::from_secs(5);
    while !socket.exists() {
        assert!(Instant::now() < end);
        thread::sleep(Duration::from_millis(10));
    }
    success(
        Command::new("ssh-add")
            .arg("-c")
            .arg(d.join("client"))
            .env("SSH_AUTH_SOCK", &socket)
            .env("SSH_ASKPASS", d.join("ask"))
            .env("SSH_ASKPASS_REQUIRE", "force")
            .env("DISPLAY", "fixture")
            .output()
            .unwrap(),
    );
    let path = d.join("client.conf");
    let config = fs::read_to_string(&path).unwrap();
    fs::write(
        path,
        format!(
            "Host inner\n IdentityAgent '{}'\n{config}",
            socket.display()
        ),
    )
    .unwrap();
    let begin = Instant::now();
    thread::scope(|scope| {
        let handles = ["check", "login"].map(|action| scope.spawn(|| server.action("two", action)));
        for handle in handles {
            let result = handle.join().unwrap();
            assert_eq!(
                result.status.code(),
                Some(4),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
    });
    assert!(proof.exists());
    assert!(begin.elapsed() < Duration::from_secs(32));
}

fn wait_for_file(path: &Path) {
    let end = Instant::now() + Duration::from_secs(4);
    while !path.exists() {
        assert!(Instant::now() < end, "{} was never created", path.display());
        thread::sleep(Duration::from_millis(10));
    }
}
#[test]
fn killing_the_owner_before_ready_ends_its_worker_and_preserves_the_master() {
    let Some(server) = Server::start() else {
        return;
    };
    success(server.action("one", "login"));
    let root = server.f.root.path();
    script(
        &root.join("worker ' ;"),
        &format!(
            "#!/bin/sh\ntouch '{0}/started'\ncat >/dev/null\ntouch '{0}/stopped'\n",
            root.display()
        ),
    );
    let child = server.f.command("one");
    let mut owner = Command::new("sh");
    owner
        .args(["-c", "exec 3<&0; \"$@\" <&3 & wait", "owner"])
        .arg(child.get_program())
        .args(child.get_args());
    for (name, value) in child.get_envs() {
        if let Some(value) = value {
            owner.env(name, value);
        }
    }
    let mut owner = owner
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let _still_open = owner.stdin.take();
    wait_for_file(&root.join("started"));
    owner.kill().unwrap();
    owner.wait().unwrap();
    wait_for_file(&root.join("stopped"));
    success(
        server
            .ssh()
            .args(["-O", "check", "outer"])
            .output()
            .unwrap(),
    );
}

#[test]
fn cancelling_a_login_closes_its_active_prompt() {
    let Some(server) = Server::start() else {
        return;
    };
    let root = server.f.root.path();
    let pidfile = root.join("prompt-pid");
    script(
        &root.join("ask"),
        &format!(
            "#!/bin/sh\necho $$ > '{}'\nexec sleep 60\n",
            pidfile.display()
        ),
    );
    let mut login = server
        .f
        .command("one")
        .env("TEST_ACTION", "login")
        .env("TEST_ASKPASS", root.join("ask"))
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    wait_for_file(&pidfile);
    // SAFETY: login is our live, unreaped child.
    unsafe {
        libc::kill(login.id() as i32, libc::SIGTERM);
    }
    assert_eq!(login.wait().unwrap().code(), Some(130));
    let pid = fs::read_to_string(pidfile).unwrap();
    let stat = format!("/proc/{}/stat", pid.trim());
    assert!(fs::read_to_string(stat).map_or(true, |s| s.contains(") Z ")));
}
