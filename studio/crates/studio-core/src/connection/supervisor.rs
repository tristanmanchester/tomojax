// Ported from T3 Code (MIT), packages/client-runtime/src/connection/{supervisor,presentation}.ts at d2c44031. See THIRD_PARTY_NOTICES.md.
use super::auth::Login;
use crate::engine::{Engine, EngineEvent, Wake, WorkerCommand};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum State {
    Closed,
    Connecting,
    Connected(Instant),
    Waiting(Instant),
    LoginNeeded,
    LoggingIn,
    Blocked,
}

#[derive(Default)]
struct Retry {
    state: State,
    failures: u32,
}
impl Default for State {
    fn default() -> Self {
        Self::Waiting(Instant::now())
    }
}
impl Retry {
    fn begin(&mut self, now: Instant) -> bool {
        if let State::Waiting(at) = self.state
            && now >= at
        {
            self.state = State::Connecting;
            true
        } else {
            false
        }
    }
    fn retry_now(&mut self) {
        if !matches!(
            self.state,
            State::Connecting | State::LoggingIn | State::Connected(_)
        ) {
            self.state = State::Waiting(Instant::now());
        }
    }
    fn connected(&mut self, now: Instant) {
        self.state = State::Connected(now);
    }
    fn exited(&mut self, code: Option<i32>, now: Instant, jitter: f64) {
        if let State::Connected(since) = self.state
            && now.duration_since(since) >= Duration::from_secs(30)
        {
            self.failures = 0;
        }
        self.state = match code {
            Some(3) => State::LoginNeeded,
            Some(5 | 6) => State::Blocked,
            Some(130) => State::Closed,
            _ => {
                let ceiling = (2000u64 << self.failures.min(8)).min(300000);
                self.failures = self.failures.saturating_add(1);
                State::Waiting(
                    now + Duration::from_millis(
                        ((ceiling / 2) as f64 * (1.0 + jitter)).round() as u64
                    ),
                )
            }
        };
    }
}

/// Owns one session. Poll at next_wake and when the engine's waker fires.
/// Requests belong exclusively to the engine that admitted them.
pub struct Owner {
    worker: PathBuf,
    profile: String,
    required: &'static [&'static str],
    stop_code: Option<i32>,
    wake: Wake,
    retry: Retry,
    engine: Option<Engine>,
    login: Option<Login>,
    next_ping: Instant,
    ping: Option<(String, Instant)>,
    diagnostic: String,
}

impl Owner {
    pub fn new(
        worker: PathBuf,
        profile: String,
        required: &'static [&'static str],
        wake: Wake,
    ) -> Self {
        Self {
            worker,
            profile,
            required,
            stop_code: None,
            wake,
            retry: Retry::default(),
            engine: None,
            login: None,
            next_ping: Instant::now(),
            ping: None,
            diagnostic: String::new(),
        }
    }
    pub fn state(&self) -> State {
        self.retry.state
    }
    pub fn status_text(&self, now: Instant) -> String {
        match self.state() {
            State::Closed => "Not connected".into(),
            State::Connecting => format!("Connecting to {}", self.profile),
            State::Connected(_) => format!("Connected to {}", self.profile),
            State::Waiting(at) => format!(
                "Reconnecting in {} s",
                at.saturating_duration_since(now).as_secs_f64().ceil() as u64
            ),
            State::LoginNeeded => format!("{} needs a login", self.profile),
            State::LoggingIn => format!("Logging in to {}", self.profile),
            State::Blocked => self.diagnostic.clone(),
        }
    }
    pub fn diagnostic(&self) -> &str {
        &self.diagnostic
    }
    pub fn next_wake(&self) -> Option<Instant> {
        match self.state() {
            State::Waiting(at) => Some(at),
            State::LoggingIn => Some(Instant::now() + Duration::from_millis(50)),
            State::Connected(_) => Some(
                self.ping
                    .as_ref()
                    .map_or(self.next_ping, |(_, deadline)| *deadline),
            ),
            _ => None,
        }
    }
    pub fn retry_now(&mut self) {
        self.retry.retry_now();
        (self.wake)();
    }
    pub fn login(&mut self, askpass: &Path) -> Result<(), String> {
        if self.login.is_some() || self.engine.is_some() {
            return Err("The connection already has an active attempt".into());
        }
        self.login =
            Some(Login::start(&self.worker, &self.profile, askpass).map_err(|e| e.to_string())?);
        self.retry.state = State::LoggingIn;
        (self.wake)();
        Ok(())
    }

    fn poll_login(&mut self, now: Instant) {
        let Some(login) = self.login.as_mut() else {
            return;
        };
        match login.poll() {
            Ok(None) => return,
            Ok(Some((status, message))) => {
                self.diagnostic = message;
                if status.success() {
                    self.retry.state = State::Waiting(now);
                } else {
                    self.retry.exited(status.code(), now, fastrand::f64());
                }
            }
            Err(e) => {
                self.diagnostic = e.message;
                self.retry
                    .exited(Some(e.status as i32), now, fastrand::f64());
            }
        }
        self.login.take();
    }
    pub fn close(&mut self) {
        self.login.take();
        self.engine.take();
        self.ping = None;
        self.retry.state = State::Closed;
    }
    pub fn engine(&self) -> Option<&Engine> {
        self.engine.as_ref()
    }
    pub fn request(&mut self, command: Value, channel: Option<&str>) -> Result<String, String> {
        self.engine
            .as_mut()
            .ok_or("The worker is disconnected")?
            .request(command, channel)
    }

    pub fn poll(&mut self) -> Vec<EngineEvent> {
        let now = Instant::now();
        self.poll_login(now);
        if self.retry.begin(now) {
            self.stop_code = None;
            self.diagnostic.clear();
            match Engine::spawn(
                WorkerCommand::profile(&self.worker, &self.profile),
                self.wake.clone(),
            ) {
                Ok(engine) => self.engine = Some(engine),
                Err(error) => {
                    self.diagnostic = error;
                    self.retry.exited(Some(5), now, 0.0);
                }
            }
        }
        let Some(engine) = self.engine.as_mut() else {
            return Vec::new();
        };
        let events = engine.poll();
        for event in &events {
            match event {
                EngineEvent::Ready(ready) => {
                    let missing: Vec<_> = self
                        .required
                        .iter()
                        .copied()
                        .filter(|name| !ready.capabilities.iter().any(|c| c == name))
                        .collect();
                    if !missing.is_empty() {
                        self.diagnostic = format!(
                            "{} lacks required capabilities: {}",
                            self.profile,
                            missing.join(", ")
                        );
                        self.stop_code = Some(5);
                        engine.abort(&self.diagnostic);
                        continue;
                    }
                    self.retry.connected(now);
                    self.next_ping = now + Duration::from_secs(15);
                }
                EngineEvent::Diagnostic(message) => self.diagnostic = message.clone(),
                EngineEvent::Reply { response, .. }
                    if response.terminal()
                        && self
                            .ping
                            .as_ref()
                            .is_some_and(|(id, _)| response.request() == Some(id)) =>
                {
                    self.ping = None;
                    self.next_ping = now + Duration::from_secs(15);
                }
                EngineEvent::Disconnected { code, reason, .. } => {
                    self.retry
                        .exited(self.stop_code.take().or(*code), now, fastrand::f64());
                    if self.diagnostic.is_empty() {
                        self.diagnostic.clone_from(reason);
                    }
                    self.ping = None;
                    self.engine.take();
                    return events;
                }
                _ => {}
            }
        }
        if matches!(self.retry.state, State::Connected(_)) {
            if self
                .ping
                .as_ref()
                .is_some_and(|(_, deadline)| now >= *deadline)
            {
                self.stop_code = Some(4);
                engine.abort("Worker did not answer status within 30 s");
            } else if self.ping.is_none() && now >= self.next_ping {
                match engine.request(json!({"type":"status"}), None) {
                    Ok(id) => self.ping = Some((id, now + Duration::from_secs(30))),
                    Err(error) => {
                        self.stop_code = Some(4);
                        engine.abort(&error);
                    }
                }
            }
        }
        events
    }
}
#[cfg(test)]
#[path = "supervisor_tests.rs"]
mod tests;
