// Ported from T3 Code (MIT, © 2026 T3 Tools Inc.), apps/server/src/preview/ServerBrowser.ts at d2c44031; see THIRD_PARTY_NOTICES.md.
use crate::transfer::Download;
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, Read};

pub const BLOCK: usize = 65536;
pub const CREDIT: usize = 262144;

pub enum Input {
    Request(Vec<u8>, Value),
    Block(String, Vec<u8>),
    Rejected(String),
    Eof,
}

pub fn read(input: &mut impl BufRead) -> Result<Input> {
    let mut bytes = Vec::new();
    let n = input.take((1 << 20) + 1).read_until(b'\n', &mut bytes)?;
    if n == 0 {
        return Ok(Input::Eof);
    }
    if n > 1 << 20 {
        if !bytes.ends_with(b"\n") {
            input.skip_until(b'\n')?;
        }
        return Ok(Input::Rejected("line exceeds 1 MiB".into()));
    }
    let value: Value = match serde_json::from_slice(&bytes) {
        Ok(value) => value,
        Err(err) => return Ok(Input::Rejected(err.to_string())),
    };
    if value["type"] == "block" {
        let size = value["bytes"]
            .as_u64()
            .filter(|&n| n > 0 && n <= BLOCK as u64)
            .context("invalid block length")?;
        let id = value["request"]
            .as_str()
            .context("missing block owner")?
            .to_owned();
        let mut payload = vec![0; size as usize];
        input.read_exact(&mut payload)?;
        Ok(Input::Block(id, payload))
    } else {
        if !bytes.ends_with(b"\n") {
            bytes.push(b'\n');
        }
        Ok(Input::Request(bytes, value))
    }
}

pub fn line(value: &Value) -> Vec<u8> {
    let mut bytes = value.to_string().into_bytes();
    bytes.push(b'\n');
    bytes
}

#[derive(Clone)]
pub struct Reply {
    pub id: String,
    pub channel: Value,
}

impl Reply {
    pub fn result(&self, result: Value) -> Value {
        json!({"type":"result","request":self.id,"channel":self.channel,"result":result})
    }
    pub fn error(&self, kind: &str, message: &str) -> Value {
        json!({"type":"error","request":self.id,"channel":self.channel,"error":{"event":"error","schema":2,"kind":kind,"message":message,"path":null,"checkpoint":null}})
    }
}

pub enum Payload {
    Frame {
        bytes: Vec<u8>,
        offset: usize,
        result: Value,
    },
    File(Download),
}

impl Payload {
    fn left(&self) -> usize {
        match self {
            Self::Frame { bytes, offset, .. } => bytes.len() - offset,
            Self::File(file) => file.left as usize,
        }
    }
    fn take(&mut self, size: usize) -> Result<Vec<u8>> {
        match self {
            Self::Frame { bytes, offset, .. } => {
                let block = bytes[*offset..*offset + size].to_vec();
                *offset += size;
                Ok(block)
            }
            Self::File(file) => file.read(size),
        }
    }
    fn finish(self) -> Value {
        match self {
            Self::Frame { result, .. } => result,
            Self::File(file) => json!({"type":"download","bytes":file.size,"sha256":file.digest()}),
        }
    }
}

pub struct Outbox {
    pub streams: HashMap<String, (Reply, Payload)>,
    lines: VecDeque<Vec<u8>>,
    last_advance: Option<(String, u64)>,
    backlog: usize,
    outstanding: usize,
    window: usize,
}

impl Default for Outbox {
    fn default() -> Self {
        Self {
            streams: HashMap::new(),
            lines: VecDeque::new(),
            last_advance: None,
            backlog: 0,
            outstanding: 0,
            window: CREDIT,
        }
    }
}

impl Outbox {
    pub fn window(&mut self, n: u64) -> Result<()> {
        ensure!(
            (65536..=1048576).contains(&n),
            "window must be 64 KiB to 1 MiB"
        );
        self.window = n as usize;
        Ok(())
    }
    pub fn credit(&mut self, n: u64) -> Result<()> {
        ensure!(
            n > 0 && n <= self.outstanding as u64,
            "credit exceeds outstanding pixel bytes"
        );
        self.outstanding -= n as usize;
        Ok(())
    }
    pub fn queue(&mut self, bytes: Vec<u8>) {
        self.last_advance = None;
        self.backlog += bytes.len();
        self.lines.push_back(bytes);
    }
    pub fn queue_record(&mut self, row: &Value, bytes: Vec<u8>) {
        let key = if row["type"] == "progress" && row["record"]["event"] == "advanced" {
            row["request"]
                .as_str()
                .zip(row["record"]["task"].as_u64())
                .map(|(id, task)| (id.to_owned(), task))
        } else {
            None
        };
        if key.is_some()
            && key == self.last_advance
            && let Some(prev) = self.lines.pop_back()
        {
            self.backlog -= prev.len();
        }
        self.queue(bytes);
        self.last_advance = key;
    }
    pub fn send(&mut self, value: Value) {
        let bytes = line(&value);
        self.queue_record(&value, bytes);
    }
    pub fn overloaded(&self) -> bool {
        self.backlog > 16 << 20
    }
    /// One write in flight. Reconsider the shortest stream at every block boundary.
    pub fn take(&mut self) -> Option<(Vec<u8>, Option<String>)> {
        if let Some(line) = self.lines.pop_front() {
            self.backlog -= line.len();
            return Some((line, None));
        }
        let (id, (_, payload)) = self.streams.iter_mut().min_by_key(|(_, (_, p))| p.left())?;
        let size = payload
            .left()
            .min(BLOCK)
            .min(self.window.saturating_sub(self.outstanding));
        if size == 0 && payload.left() > 0 {
            return None;
        }
        let id = id.clone();
        let data = payload.take(size);
        let mut frame = Vec::new();
        if let Ok(data) = &data
            && !data.is_empty()
        {
            frame = line(&json!({"type":"block","request":id,"bytes":size}));
            frame.extend_from_slice(data);
            self.outstanding += size;
        }
        if data.is_err() || payload.left() == 0 {
            let (reply, payload) = self.streams.remove(&id)?;
            let terminal = match data {
                Ok(_) => reply.result(payload.finish()),
                Err(err) => reply.error("input", &err.to_string()),
            };
            self.send(terminal);
            return Some((frame, Some(id)));
        }
        Some((frame, None))
    }
}
