//! Page long replies without keeping temporary files or worker handles alive.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use uuid::Uuid;

const RETAINED_BYTES: usize = 64 << 20;
const LIFETIME: Duration = Duration::from_secs(900);
const PAGE: usize = 8000;
const REPLY_BUDGET: usize = 20_000;
struct Entry {
    id: String,
    bytes: String,
    expires: Instant,
}
#[derive(Default)]
pub struct Parts {
    entries: VecDeque<Entry>,
    size: usize,
}

impl Parts {
    fn expire(&mut self, now: Instant) {
        while self.entries.front().is_some_and(|e| e.expires <= now) {
            self.size -= self.entries.pop_front().unwrap().bytes.len();
        }
    }
    fn store(&mut self, raw: String, now: Instant) -> Option<String> {
        self.expire(now);
        if raw.len() > RETAINED_BYTES {
            return None;
        }
        while self.size + raw.len() > RETAINED_BYTES {
            self.size -= self.entries.pop_front().unwrap().bytes.len();
        }
        let id = Uuid::new_v4().simple().to_string();
        let uri = format!("tomojax://result/{id}/0");
        self.size += raw.len();
        self.entries.push_back(Entry {
            id,
            bytes: raw,
            expires: now + LIFETIME,
        });
        Some(uri)
    }
    pub fn read(&mut self, suffix: &str, now: Instant) -> Result<Value> {
        self.expire(now);
        let (id, n) = suffix.split_once('/').context("Missing result part")?;
        let found = self
            .entries
            .iter()
            .find(|e| e.id == id)
            .context("Result is no longer available. Repeat the call to get a new result.")?;
        let index = n.parse::<usize>()?;
        let start = index.checked_mul(PAGE).context("Invalid part number")?;
        ensure!(
            start < found.bytes.len(),
            "Unknown result part. Repeat the call to get a new result."
        );
        let end = (start + PAGE).min(found.bytes.len());
        let next =
            (end < found.bytes.len()).then(|| format!("tomojax://result/{id}/{}", index + 1));
        Ok(json!({"data":&found.bytes[start..end],"next":next}))
    }
    pub fn bounded(&mut self, reply: Value) -> Value {
        if cost(&reply) <= REPLY_BUDGET {
            return reply;
        }
        let mut raw = String::new();
        let mut stored = reply.clone();
        stored["resultType"] = json!("complete");
        for c in stored.to_string().chars() {
            if c.is_ascii() && c != '\x7f' {
                raw.push(c);
            } else {
                use std::fmt::Write;
                for unit in c.encode_utf16(&mut [0; 2]) {
                    write!(raw, "\\u{unit:04x}").unwrap();
                }
            }
        }
        let uri = self.store(raw, Instant::now());
        let message = uri.as_ref().map_or_else(
            || "Result exceeds the paging limit. Repeat the call with a smaller result.".into(),
            |uri| format!("Full result continues at {uri}"),
        );
        let fields = &reply["structuredContent"];
        let mut head = ["handle", "kind", "method", "next"]
            .into_iter()
            .filter_map(|k| fields[k].as_str())
            .map(|s| s.chars().take(160).collect::<String>())
            .collect::<Vec<_>>()
            .join(" ");
        if !head.is_empty() {
            head.push('\n');
        }
        head.push_str(&message);
        let content = uri.map_or_else(|| json!({"message":message}), |u| json!({"continuation":u}));
        let mut clipped = json!({"content":[{"type":"text","text":head}],"structuredContent":content,"isError":reply["isError"]});
        if let Some(fields) = fields.as_object() {
            let priority = [
                "handle", "members", "handles", "paths", "next", "kind", "shape", "unit", "method",
                "message",
            ];
            for key in priority.into_iter().chain(
                fields
                    .keys()
                    .map(String::as_str)
                    .filter(|k| !priority.contains(k)),
            ) {
                if let Some(v) = fields.get(key) {
                    clipped["structuredContent"][key] = v.clone();
                    if cost(&clipped) > REPLY_BUDGET {
                        clipped["structuredContent"]
                            .as_object_mut()
                            .unwrap()
                            .shift_remove(key);
                        if priority.contains(&key)
                            && let Some(prefix) = fitting_prefix(&mut clipped, key, v)
                        {
                            clipped["structuredContent"][key] = prefix;
                        }
                    }
                }
            }
        }
        let mut items = reply["content"].as_array().cloned().unwrap_or_default();
        items.sort_by_key(|v| v["type"] == "image");
        for item in items {
            let content = clipped["content"].as_array_mut().unwrap();
            let index = content.len() - 1;
            content.insert(index, item);
            if cost(&clipped) > REPLY_BUDGET {
                clipped["content"].as_array_mut().unwrap().remove(index);
            }
        }
        clipped
    }
}
fn fitting_prefix(reply: &mut Value, key: &str, value: &Value) -> Option<Value> {
    let len = match value {
        Value::Array(v) => v.len(),
        Value::String(v) => v.chars().count(),
        Value::Object(v) => v.len(),
        _ => return None,
    };
    let prefix = |n| match value {
        Value::Array(v) => Value::Array(v[..n].to_vec()),
        Value::String(v) => Value::String(v.chars().take(n).collect()),
        Value::Object(v) => Value::Object(
            v.iter()
                .take(n)
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        ),
        _ => unreachable!(),
    };
    let (mut lo, mut hi) = (0, len);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        reply["structuredContent"][key] = prefix(mid);
        if cost(reply) <= REPLY_BUDGET {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    reply["structuredContent"]
        .as_object_mut()?
        .shift_remove(key);
    (lo > 0).then(|| prefix(lo))
}

fn cost(value: &Value) -> usize {
    let mut counted = value.clone();
    counted["_meta"] = Value::Null;
    counted["resultType"] = json!("complete");
    let mut image_bytes = 0;
    let mut pixels = 0;
    if let Some(items) = counted["content"].as_array_mut() {
        for item in items {
            item["_meta"] = Value::Null;
            item["annotations"] = Value::Null;
            if item["type"] != "image" {
                continue;
            }
            let Some(Value::String(data)) = item.as_object_mut().and_then(|o| o.remove("data"))
            else {
                return usize::MAX;
            };
            use base64::{Engine, engine::general_purpose::STANDARD};
            let Some(h) = data.get(..32).and_then(|s| STANDARD.decode(s).ok()) else {
                return usize::MAX;
            };
            let w = u32::from_be_bytes(h[16..20].try_into().unwrap()) as usize;
            let h = u32::from_be_bytes(h[20..24].try_into().unwrap()) as usize;
            image_bytes += data.len();
            if w.max(h) > 1568 || image_bytes > 2 << 20 {
                return usize::MAX;
            }
            pixels += (w * h).div_ceil(750);
        }
    }
    // The Python SDK counts optional null fields on each content block and reply.
    counted.to_string().len() + pixels + 256
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expires_at_the_deadline_and_reassembles_escaped_text() {
        let now = Instant::now();
        let text = "x".repeat(PAGE - 2) + "\\ud83d\\ude00";
        let mut parts = Parts::default();
        let uri = parts.store(text.clone(), now).unwrap();
        let first = parts
            .read(uri.trim_start_matches("tomojax://result/"), now)
            .unwrap();
        let second = parts
            .read(
                first["next"]
                    .as_str()
                    .unwrap()
                    .trim_start_matches("tomojax://result/"),
                now,
            )
            .unwrap();
        assert_eq!(
            format!(
                "{}{}",
                first["data"].as_str().unwrap(),
                second["data"].as_str().unwrap()
            ),
            text
        );
        assert!(
            parts
                .read(uri.trim_start_matches("tomojax://result/"), now + LIFETIME)
                .is_err()
        );
        assert_eq!(parts.size, 0);
    }
}
