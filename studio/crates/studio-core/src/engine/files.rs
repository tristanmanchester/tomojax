//! Background file reads share the worker that issued session artifacts.
//! The pipe reader delivers their terminals directly, independently of UI polling.
use super::{Outgoing, Response, ResultValue};
use crate::sync::Locked;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, mpsc};

#[derive(Default)]
struct Routes {
    next: u64,
    pending: HashMap<String, mpsc::Sender<Response>>,
    closed: bool,
}

#[derive(Clone)]
pub struct FileReads {
    routes: Arc<Mutex<Routes>>,
    writer: mpsc::SyncSender<Outgoing>,
}

impl FileReads {
    pub(super) fn new(writer: mpsc::SyncSender<Outgoing>) -> Self {
        Self {
            routes: Arc::default(),
            writer,
        }
    }

    fn send(&self, command: Value) -> Result<(String, mpsc::Receiver<Response>), String> {
        let (send, receive) = mpsc::channel();
        let mut routes = self.routes.locked();
        if routes.closed {
            return Err("The worker file connection is closed".into());
        }
        routes.next += 1;
        let id = format!("file-{}", routes.next);
        let mut line =
            serde_json::to_vec(&json!({"id":id, "command":command})).map_err(|e| e.to_string())?;
        line.push(b'\n');
        self.writer
            .try_send(Outgoing::Line(line))
            .map_err(|e| e.to_string())?;
        routes.pending.insert(id.clone(), send);
        Ok((id, receive))
    }

    pub fn request(&self, command: Value) -> Result<ResultValue, String> {
        let (_, receive) = self.send(command)?;
        Self::result(
            receive
                .recv()
                .map_err(|_| "The worker file connection closed")?,
        )
    }

    pub fn copy(
        &self,
        command: Value,
        canceled: &std::sync::atomic::AtomicBool,
    ) -> Result<(), String> {
        let (id, receive) = self.send(command)?;
        let mut stopping = false;
        loop {
            match receive.recv_timeout(std::time::Duration::from_millis(50)) {
                Ok(response) => return Self::result(response).map(|_| ()),
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err("The worker file connection closed".into());
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if !stopping && canceled.load(std::sync::atomic::Ordering::Acquire) {
                        self.send(json!({"type":"control","target":id,"action":"cancel"}))?;
                        stopping = true;
                    }
                }
            }
        }
    }

    fn result(response: Response) -> Result<ResultValue, String> {
        match response {
            Response::Result { result, .. } => Ok(result),
            Response::Error { error, .. } => Err(error.message),
            Response::ProtocolError { message, .. } => Err(message),
            _ => Err("Expected a file request terminal".into()),
        }
    }

    pub(super) fn route(&self, response: Response) -> Option<Response> {
        if let Some(id) = response.request()
            && id.starts_with("file-")
        {
            if response.terminal()
                && let Some(send) = self.routes.locked().pending.remove(id)
            {
                let _ = send.send(response);
            }
            return None;
        }
        Some(response)
    }

    pub fn close(&self) {
        let mut routes = self.routes.locked();
        routes.closed = true;
        routes.pending.clear();
    }
}
