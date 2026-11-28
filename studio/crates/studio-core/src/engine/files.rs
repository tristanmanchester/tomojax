//! Background file reads share the worker that issued session artifacts.
//! The pipe reader delivers their terminals directly, independently of UI polling.
use super::{Outgoing, Response, ResultValue};
use crate::sync::Locked;
use crate::{
    protocol::Listing,
    transfer::{self, Upload},
    worker_path::WorkerPath,
};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, mpsc};

#[derive(Default)]
struct Routes {
    next: u64,
    pending: HashMap<String, mpsc::Sender<Response>>,
    downloads: HashMap<String, Result<Upload, String>>,
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
        self.start(command, None)
    }

    fn start(
        &self,
        command: Value,
        download: Option<Upload>,
    ) -> Result<(String, mpsc::Receiver<Response>), String> {
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
        if let Some(stage) = download {
            routes.downloads.insert(id.clone(), Ok(stage));
        }
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

    pub fn list(&self, path: Option<&WorkerPath>, after: Option<&str>) -> Result<Listing, String> {
        match self.request(json!({"type":"list","path":path,"after":after}))? {
            ResultValue::List(list) => Ok(list),
            _ => Err("Expected a file listing".into()),
        }
    }

    pub fn download(
        &self,
        source: &WorkerPath,
        size: u64,
        destination: &Path,
    ) -> Result<(), String> {
        let stage = Upload::download(destination.into(), size).map_err(|e| e.to_string())?;
        let (id, receive) = self.start(json!({"type":"download","path":source}), Some(stage))?;
        let terminal = receive.recv();
        let stage = self.routes.locked().downloads.remove(&id);
        let result = Self::result(terminal.map_err(|_| "The file connection closed")?)?;
        let ResultValue::Download { bytes, sha256 } = result else {
            return Err("Expected a download receipt".into());
        };
        stage
            .ok_or("The file connection closed")??
            .finish_download(bytes, sha256)
            .map_err(|e| e.to_string())
    }

    pub(super) fn download_block(&self, id: &str, bytes: &[u8]) -> bool {
        let mut routes = self.routes.locked();
        let Some(stage) = routes.downloads.get_mut(id) else {
            return false;
        };
        if let Ok(file) = stage
            && let Err(error) = file.write(bytes)
        {
            *stage = Err(error.to_string());
        }
        true
    }

    pub fn upload(&self, source: &Path, destination: &WorkerPath) -> Result<(), String> {
        use sha2::{Digest, Sha256};
        use std::io::{Read, Seek};
        let io = |e: std::io::Error| e.to_string();
        let mut file = std::fs::File::open(source).map_err(io)?;
        let meta = file.metadata().map_err(io)?;
        if !meta.is_file() {
            return Err("Choose a regular file to copy".into());
        }
        transfer::check_size(meta.len()).map_err(|e| e.to_string())?;
        let mut hash = Sha256::new();
        let read = std::io::copy(&mut (&mut file).take(meta.len() + 1), &mut hash).map_err(io)?;
        if read != meta.len() {
            return Err("The source changed during the copy".into());
        }
        file.rewind().map_err(io)?;
        let (id, receive) = self.send(json!({"type":"upload","path":destination,"bytes":read,"sha256":format!("{:x}",hash.finalize())}))?;
        let send = || -> Result<(), String> {
            let mut left = read;
            let mut data = vec![0u8; 64 << 10];
            while left > 0 {
                if let Ok(reply) = receive.try_recv() {
                    return Self::result(reply).map(|_| ());
                }
                let n = data.len().min(left as usize);
                file.read_exact(&mut data[..n]).map_err(io)?;
                let mut frame = serde_json::to_vec(&json!({"type":"block","request":id,"bytes":n}))
                    .map_err(|e| e.to_string())?;
                frame.push(b'\n');
                frame.extend_from_slice(&data[..n]);
                self.writer
                    .send(Outgoing::Line(frame))
                    .map_err(|e| e.to_string())?;
                left -= n as u64;
            }
            Self::result(receive.recv().map_err(|_| "The file connection closed")?).map(|_| ())
        };
        let result = {
            let mut send = send;
            send()
        };
        if result.is_err() {
            let _ = self.send(json!({"type":"control","target":id,"action":"cancel"}));
        }
        result
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
