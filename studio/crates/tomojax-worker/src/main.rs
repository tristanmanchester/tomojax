//! Worker process serving the files and TomoJAX executor on this machine.
mod executor;
mod files;
pub mod inline;
mod jobs;
use tomojax_io::{metadata, pixels};
mod session;
mod transfer;
mod wire;

use anyhow::{Context, Result, bail, ensure};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    mpsc::{self, SyncSender},
};
use std::thread;
use std::time::{Duration, Instant};

struct Options {
    python: String,
    session: Vec<String>,
    results: PathBuf,
    jobs: tomojax_jobs::script::Launcher,
    sbatch: Vec<String>,
    idle: Option<Duration>,
}

impl Options {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self> {
        let mut opts = Self {
            python: "python3".into(),
            session: Vec::new(),
            results: files::resolve("~")?,
            jobs: tomojax_jobs::script::Launcher::Inline,
            sbatch: Vec::new(),
            idle: None,
        };
        while let Some(key) = args.next() {
            let v = args
                .next()
                .with_context(|| format!("missing argument for {key}"))?;
            match key.as_str() {
                "--python" => opts.python = v,
                "--session" => {
                    opts.session =
                        shlex::split(&v).context("unclosed quote in executor launcher")?
                }
                "--results" => opts.results = files::resolve(&v)?,
                "--jobs" => {
                    opts.jobs = serde_json::from_value(serde_json::Value::String(v))
                        .context("unknown job launcher")?
                }
                "--sbatch" => {
                    opts.sbatch = shlex::split(&v).context("unclosed quote in scheduler options")?
                }
                "--idle-exit" => {
                    let secs = v.parse::<u64>()?;
                    ensure!(secs > 0, "idle exit must be positive");
                    opts.idle = Some(Duration::from_secs(secs));
                }
                _ => bail!("unknown serve argument {key}"),
            }
        }
        Ok(opts)
    }
}

enum Event {
    Inline(inline::Action),
    Input(wire::Input),
    Eof,
    Executor(u64, Vec<u8>),
    Gone(u64, String),
    Diagnostic(u64, String),
    Done(String, u64, Result<session::Output>),
    Progress(String, u64, u64),
    Written(io::Result<()>),
}

type Task = Box<dyn FnOnce() + Send>;
struct Pool {
    send: SyncSender<Task>,
}
impl Pool {
    fn new() -> Self {
        let (send, receive) = mpsc::sync_channel::<Task>(64);
        let receive = Arc::new(Mutex::new(receive));
        for _ in 0..4 {
            let receive = receive.clone();
            thread::spawn(move || {
                loop {
                    let work = receive.lock().expect("worker pool lock").recv();
                    match work {
                        Ok(work) => work(),
                        Err(_) => return,
                    }
                }
            });
        }
        Self { send }
    }
    fn submit(&self, work: impl FnOnce() + Send + 'static) -> Result<()> {
        self.send
            .try_send(Box::new(work))
            .map_err(|_| anyhow::anyhow!("file request queue is full"))
    }
}

fn read_requests(send: SyncSender<Event>) {
    thread::spawn(move || {
        let mut input = io::stdin().lock();
        loop {
            match wire::read(&mut input) {
                Ok(wire::Input::Eof) => break,
                Ok(item) => {
                    if send.send(Event::Input(item)).is_err() {
                        return;
                    }
                }
                Err(e) => {
                    eprintln!("invalid worker input: {e}");
                    break;
                }
            }
        }
        let _ = send.send(Event::Eof);
    });
}

fn serve(opts: Options) -> Result<()> {
    use rustix::event::{PollFd, PollFlags, Timespec, poll};
    studio_core::connection::process::handle_signals()?;
    let stdout = io::stdout();
    let mut disconnected = [PollFd::new(&stdout, PollFlags::empty())];
    let idle = opts.idle;
    let (events, incoming) = mpsc::sync_channel(256);
    read_requests(events.clone());
    let (writer, output) = mpsc::sync_channel::<Vec<u8>>(1);
    let send = events.clone();
    thread::spawn(move || {
        let mut stdout = io::stdout().lock();
        for bytes in output {
            let written = stdout.write_all(&bytes).and_then(|()| stdout.flush());
            if send.send(Event::Written(written)).is_err() {
                return;
            }
        }
    });
    let mut session = session::Session::new(opts, events)?;
    let mut last = Instant::now();
    let mut writing = false;
    loop {
        if poll(&mut disconnected, Some(&Timespec::default()))? > 0 {
            bail!("worker output closed");
        }
        if studio_core::connection::process::active().is_err()
            || idle.is_some_and(|d| last.elapsed() >= d)
        {
            session.stop();
        }
        session.tick()?;
        session.flush();
        if session.out.overloaded() {
            bail!("output backlog exceeds 16 MiB");
        }
        if !writing {
            if let Some((frame, done)) = session.out.take() {
                writer.send(frame)?;
                writing = true;
                if let Some(id) = done {
                    session.stream_done(&id);
                }
            } else if session.closing.is_some() && session.executor.is_none() {
                break;
            }
        }
        if let Some(at) = session.closing
            && at.elapsed() > Duration::from_secs(5)
        {
            if session.executor.take().is_some() {
                session.expired();
            }
            if at.elapsed() > Duration::from_secs(7) {
                bail!("worker output stalled during shutdown");
            }
        }
        match incoming.recv_timeout(Duration::from_millis(100)) {
            Ok(Event::Input(input)) => {
                last = Instant::now();
                if session.closing.is_none() {
                    session.input(input);
                }
            }
            Ok(Event::Inline(action)) => session.script(action),
            Ok(Event::Eof) => session.stop(),
            Ok(Event::Executor(g, bytes)) => {
                if let Err(e) = session.answer(g, bytes) {
                    session.gone(g, format!("{e:#}"));
                }
            }
            Ok(Event::Gone(g, e)) => session.gone(g, e),
            Ok(Event::Diagnostic(g, line)) => session.diagnostic(g, line),
            Ok(Event::Done(id, seq, result)) => session.done(&id, seq, result),
            Ok(Event::Progress(id, seq, n)) => session.progress(&id, seq, n),
            Ok(Event::Written(result)) => {
                result?;
                writing = false;
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    let mut args = std::env::args().skip(1);
    let result = match args.next().as_deref() {
        Some("connect") => {
            let result = studio_core::connection::connect(
                &args.collect::<Vec<_>>(),
                |profile| studio_core::connection::worker_path(profile, &std::env::current_exe()?),
                None,
            );
            return match result {
                Ok(code) => std::process::ExitCode::from(code as u8),
                Err(e) => {
                    eprintln!("{e}");
                    std::process::ExitCode::from(e.status as u8)
                }
            };
        }
        Some("job") => match jobs::run(args) {
            Ok(code) => return std::process::ExitCode::from(code),
            Err(error) => Err(error),
        },
        Some("serve") => Options::parse(args).and_then(serve),
        Some("--help" | "-h") => {
            println!(
                "Usage: tomojax-worker serve [--python PATH] [--session COMMAND] [--results DIR] [--jobs LAUNCHER] [--sbatch OPTIONS] [--idle-exit SECONDS]\n       tomojax-worker job DIR"
            );
            Ok(())
        }
        _ => Err(anyhow::anyhow!("expected worker mode: serve")),
    };
    match result {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e:#}");
            std::process::ExitCode::FAILURE
        }
    }
}
