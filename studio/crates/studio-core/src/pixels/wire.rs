use std::collections::HashMap;
use std::io::{BufRead, Read};
use std::time::{Duration, Instant};

use crate::protocol::{
    ChannelRevision, ErrorKind, ExecutorState, Peer, Progress, Response, ResultValue, WorkerError,
};

const BLOCK: usize = 64 << 10;
const LINE: u64 = 16 << 20;

pub(crate) struct WireReader<R> {
    input: R,
    version: Option<String>,
    partial: HashMap<String, Vec<u8>>,
    credit: CreditWindow,
}

#[derive(serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Flow {
    Credit(usize),
    Window(usize),
}

#[derive(Default)]
struct CreditWindow {
    target: Option<usize>,
    rate: Option<f64>,
}

impl CreditWindow {
    fn measure(&mut self, bytes: usize, transfer: Duration) -> Option<usize> {
        let current = self.target?;
        if bytes < BLOCK / 2 || transfer.is_zero() {
            return None;
        }
        let rate = bytes as f64 / transfer.as_secs_f64();
        // Smooth transfer time per byte, so a buffered block does not dominate.
        let rate = self
            .rate
            .map_or(rate, |old| 1. / (0.75 / old + 0.25 / rate));
        self.rate = Some(rate);
        let target = ((rate * 0.1 / BLOCK as f64).round() as usize).clamp(1, 16) * BLOCK;
        self.target = Some(target);
        (current != target).then_some(target)
    }
}

impl<R: BufRead> WireReader<R> {
    pub(crate) fn new(input: R) -> Self {
        Self {
            input,
            version: None,
            partial: HashMap::new(),
            credit: CreditWindow::default(),
        }
    }

    fn decode(&self, line: &[u8]) -> Result<Response, String> {
        let error = match serde_json::from_slice(line) {
            Ok(response) => return Ok(response),
            Err(error) => format!("Invalid worker response: {error}"),
        };
        #[derive(serde::Deserialize)]
        struct Envelope {
            #[serde(rename = "type")]
            kind: String,
            request: String,
            channel: Option<ChannelRevision>,
        }
        let row: Envelope = serde_json::from_slice(line).map_err(|_| error.clone())?;
        let Some(version) = &self.version else {
            return Err(error);
        };
        let message = format!("unsupported reply from TomoJAX {version}");
        match row.kind.as_str() {
            "progress" => Ok(Response::Progress {
                request: row.request,
                channel: row.channel,
                record: Progress::Note {
                    schema: 2,
                    task: None,
                    time: 0.0,
                    level: "warning".into(),
                    message,
                },
            }),
            "result" | "error" => Ok(Response::Error {
                request: row.request,
                channel: row.channel,
                error: WorkerError {
                    schema: 2,
                    event: "error".into(),
                    kind: ErrorKind::Input,
                    message,
                    path: None,
                    checkpoint: None,
                },
            }),
            _ => Err(error),
        }
    }

    #[cfg(test)]
    pub(crate) fn next(
        &mut self,
        mut flow: impl FnMut(Flow) -> Result<(), String>,
    ) -> Result<Option<Response>, String> {
        self.next_with_files(&mut flow, |_, _| false)
    }

    pub(crate) fn next_with_files(
        &mut self,
        mut flow: impl FnMut(Flow) -> Result<(), String>,
        mut download_block: impl FnMut(&str, &[u8]) -> bool,
    ) -> Result<Option<Response>, String> {
        let mut line = Vec::new();
        loop {
            line.clear();
            let read = self
                .input
                .by_ref()
                .take(LINE + 1)
                .read_until(b'\n', &mut line)
                .map_err(|e| e.to_string())?;
            if read == 0 {
                return Ok(None);
            }
            if read as u64 > LINE || !line.ends_with(b"\n") {
                return Err("Worker returned an oversized or incomplete protocol line".into());
            }
            let mut response = self.decode(&line)?;
            if let Response::Executor(ExecutorState::Ready(ready)) = &response
                && let Peer::Executor { versions, .. } = &ready.peer
            {
                self.version = Some(versions.tomojax.clone());
            }
            match &mut response {
                Response::Ready(ready) => {
                    if let Some(initial) = ready.pixel_credit {
                        if !(BLOCK..=1 << 20).contains(&initial) {
                            return Err("Invalid initial pixel credit".into());
                        }
                        self.credit.target = Some(initial);
                    }
                }
                Response::Block { request, bytes } => {
                    if !(1..=BLOCK).contains(bytes)
                        || (self.partial.len() >= 40 && !self.partial.contains_key(request))
                    {
                        return Err("Worker exceeded pixel block or active plane limit".into());
                    }
                    let before = Instant::now();
                    let mut block = vec![0; *bytes];
                    self.input
                        .read_exact(&mut block)
                        .map_err(|e| format!("Incomplete worker block: {e}"))?;
                    if !download_block(request, &block) {
                        let data = self.partial.entry(std::mem::take(request)).or_default();
                        if data.len() + bytes.to_owned() > 17 << 20 {
                            return Err("Worker pixel payload exceeds the plane limit".into());
                        }
                        data.extend_from_slice(&block);
                    }
                    if let Some(size) = self.credit.measure(*bytes, before.elapsed()) {
                        flow(Flow::Window(size))?;
                    }
                    flow(Flow::Credit(*bytes))?;
                    continue;
                }
                Response::Result {
                    request,
                    result: ResultValue::Pixels(pixels),
                    ..
                } => {
                    pixels.payload = self
                        .partial
                        .remove(request)
                        .ok_or("Pixel terminal without any blocks")?;
                    if pixels.payload.len() != pixels.bytes {
                        return Err("Incomplete worker pixel payload".into());
                    }
                }
                response if response.terminal() => {
                    if let Some(id) = response.request() {
                        self.partial.remove(id);
                    }
                }
                _ => {}
            }
            return Ok(Some(response));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn newer_records_only_affect_the_request_that_sent_them() {
        let lines = br#"{"type":"progress","request":"a","record":{"event":"new-event"}}
{"type":"result","request":"a","result":{"type":"new-result"}}
{"type":"result","request":"b","result":{"type":"ack"}}
"#;
        let mut reader = WireReader::new(&lines[..]);
        reader.version = Some("2.0".into());
        let mut next = || reader.next(|_| Ok(())).unwrap().unwrap();
        assert!(matches!(
            next(),
            Response::Progress {
                record: Progress::Note { .. },
                ..
            }
        ));
        assert!(
            matches!(next(),Response::Error{request,error,..} if request=="a" && error.message=="unsupported reply from TomoJAX 2.0")
        );
        assert!(matches!(next(),Response::Result{request,..} if request=="b"));
    }

    #[test]
    fn size_window_from_tenths_of_a_second_of_block_transfer() {
        let mut window = CreditWindow {
            target: Some(4 * BLOCK),
            rate: None,
        };
        let duration = Duration::from_secs_f64(BLOCK as f64 / 1_200_000.);
        assert_eq!(window.measure(BLOCK, duration), Some(2 * BLOCK));
        assert_eq!(window.measure(BLOCK, duration), None);
        for _ in 0..100 {
            window.measure(BLOCK, Duration::from_micros(500));
        }
        assert_eq!(window.target, Some(1 << 20));
        for _ in 0..100 {
            window.measure(BLOCK, Duration::from_secs(2));
        }
        assert_eq!(window.target, Some(BLOCK));
    }
}
