//! A Rust session announces its machine immediately. Its Python executor can
//! be starting, ready or gone independently of the SSH transport.
use super::{Device, Versions};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ready {
    pub protocol: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pixel_credit: Option<usize>,
    pub capabilities: Vec<String>,
    #[serde(flatten)]
    pub peer: Peer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Peer {
    Worker {
        host: String,
        home: String,
        directory: String,
        jobs: Value,
    },
    Executor {
        progress: u8,
        versions: Versions,
        #[serde(default)]
        schema: String,
        #[serde(default)]
        root: String,
        devices: Vec<Device>,
    },
}

impl Ready {
    pub fn supported(&self) -> bool {
        self.protocol == 1 && !matches!(self.peer,Peer::Executor{progress,..} if progress!=2)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "state", content = "detail", rename_all = "snake_case")]
pub enum ExecutorState {
    Starting(Value),
    Ready(Ready),
    Gone(Value),
}
