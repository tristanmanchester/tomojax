use std::collections::HashMap;
use std::io::{BufRead, Read};
use std::time::{Duration, Instant};

use crate::protocol::{Response, ResultValue};

const BLOCK: usize = 64 << 10;
const LINE: u64 = 16 << 20;

pub(crate) struct WireReader<R> {
    input: R,
    partial: HashMap<String, Vec<u8>>,
    credit: CreditWindow,
}

struct CreditWindow {
    available: usize,
    limit: usize,
    target: usize,
    rate: Option<f64>,
}

impl CreditWindow {
    fn consumed(&mut self, bytes: usize, transfer: Duration) -> Result<usize, String> {
        self.available = self
            .available
            .checked_sub(bytes)
            .ok_or("Worker exceeded its pixel credit")?;
        if bytes >= BLOCK / 2 && !transfer.is_zero() {
            let rate = bytes as f64 / transfer.as_secs_f64();
            // Smooth transfer time per byte, so a buffered block does not dominate.
            let rate = self
                .rate
                .map_or(rate, |old| 1. / (0.75 / old + 0.25 / rate));
            self.rate = Some(rate);
            self.target =
                ((rate * 0.1 / BLOCK as f64).round() as usize).clamp(1, self.limit / BLOCK) * BLOCK;
        }
        let grant = self.target.saturating_sub(self.available);
        self.available += grant;
        Ok(grant)
    }
}

impl<R: BufRead> WireReader<R> {
    pub(crate) fn new(input: R) -> Self {
        Self {
            input,
            partial: HashMap::new(),
            credit: CreditWindow {
                available: 4 * BLOCK,
                target: 4 * BLOCK,
                limit: 4 * BLOCK,
                rate: None,
            },
        }
    }

    pub(crate) fn next(
        &mut self,
        mut grant: impl FnMut(usize) -> Result<(), String>,
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
            let mut response: Response = serde_json::from_slice(&line)
                .map_err(|e| format!("Invalid worker response: {e}"))?;
            match &mut response {
                Response::Ready(ready) => {
                    if let Some(initial) = ready.pixel_credit {
                        if !(BLOCK..=1 << 20).contains(&initial) {
                            return Err("Invalid initial pixel credit".into());
                        }
                        self.credit.available = initial;
                        self.credit.target = initial;
                        self.credit.limit = 1 << 20;
                    }
                }
                Response::Block { request, bytes } => {
                    if !(1..=BLOCK).contains(bytes)
                        || (self.partial.len() >= 40 && !self.partial.contains_key(request))
                    {
                        return Err("Worker exceeded pixel block or active plane limit".into());
                    }
                    let data = self.partial.entry(std::mem::take(request)).or_default();
                    let end = data.len() + *bytes;
                    // An incompressible 16 MiB plane can grow slightly in the codec.
                    if end > 17 << 20 {
                        return Err("Worker pixel payload exceeds the plane limit".into());
                    }
                    let start = data.len();
                    data.resize(end, 0);
                    let before = Instant::now();
                    self.input
                        .read_exact(&mut data[start..])
                        .map_err(|e| format!("Incomplete worker pixel block: {e}"))?;
                    let credit = self.credit.consumed(*bytes, before.elapsed())?;
                    if credit > 0 {
                        grant(credit)?;
                    }
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
    fn new_worker_grows_credit_and_legacy_worker_withholds_only_consumed_bytes() {
        for (initial, limit) in [(4 * BLOCK, 4 * BLOCK), (BLOCK, 1 << 20)] {
            let mut window = CreditWindow {
                available: initial,
                target: initial,
                limit,
                rate: None,
            };
            let mut granted = 0;
            for _ in 0..4 {
                granted += window
                    .consumed(BLOCK, Duration::from_secs_f64(BLOCK as f64 / 1_200_000.))
                    .unwrap();
            }
            assert_eq!(window.available, 2 * BLOCK);
            assert_eq!(initial + granted, 6 * BLOCK);
            for _ in 0..100 {
                window.consumed(BLOCK, Duration::from_micros(500)).unwrap();
            }
            assert_eq!(window.available, limit);
        }
    }
}
