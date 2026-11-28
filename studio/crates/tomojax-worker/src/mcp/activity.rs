//! Progress for a single MCP call follows its worker requests, including nested reads.
use super::Context;
use rmcp::model::{ProgressNotificationParam, ProgressToken};
use serde_json::{Value, json};
use std::{
    collections::BTreeSet,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::Notify;

#[derive(Default)]
struct Record {
    stages: serde_json::Map<String, Value>,
    latest: Value,
    time: Option<String>,
    devices: BTreeSet<String>,
    notes: Vec<String>,
}
#[derive(Default, Clone)]
pub struct Activity {
    record: Arc<Mutex<Record>>,
    changed: Arc<Notify>,
}
tokio::task_local! {pub static CURRENT:Activity;}
impl Activity {
    pub fn record(&self, event: &Value) {
        let mut record = self.record.lock().unwrap();
        let id = event["task"].to_string();
        match event["event"].as_str() {
            Some("started") => {
                let label = [event["name"].as_str(), event["label"].as_str()]
                    .into_iter()
                    .flatten()
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ");
                record.stages.insert(id, json!(label));
            }
            Some("finished") => {
                record.stages.shift_remove(&id);
            }
            Some("placed") => {
                if let Some(devices) = event["devices"].as_array() {
                    record
                        .devices
                        .extend(devices.iter().filter_map(Value::as_str).map(str::to_owned));
                }
            }
            Some("note") => {
                if let Some(note) = event["message"].as_str() {
                    record.notes.push(note.to_owned());
                }
            }
            _ => {}
        }
        record.latest = event.clone();
        record.time = Some(jiff::Timestamp::now().to_string());
        self.changed.notify_one();
    }
    pub fn report(&self, data: &mut Value) {
        let record = self.record.lock().unwrap();
        data["devices"] = json!(record.devices);
        data["notes"] = json!(record.notes);
    }
    fn status(&self) -> (String, Option<f64>, Option<String>) {
        let record = self.record.lock().unwrap();
        let label = record
            .stages
            .values()
            .next_back()
            .and_then(Value::as_str)
            .unwrap_or("worker");
        let e = &record.latest;
        let message = if e["event"] == "advanced" {
            let mut position = format!("{label} {}", e["done"]);
            if !e["total"].is_null() {
                position += &format!("/{}", e["total"]);
            }
            if let Some(metrics) = e["metrics"].as_object() {
                for (name, v) in metrics {
                    position += &format!(" · {name} {v}");
                }
            }
            if let Some(eta) = e["eta"].as_f64() {
                position += &format!(" · eta {eta} s");
            }
            position
        } else {
            format!("waiting: {label}")
        };
        (message, e["eta"].as_f64(), record.time.clone())
    }
    pub fn update(&self, task: &mut Value, started: Instant) {
        let (message, eta, time) = self.status();
        if let Some(time) = time {
            task["lastUpdatedAt"] = json!(time);
            task["statusMessage"] = json!(message);
        }
        if started.elapsed().as_secs_f64() + eta.unwrap_or(0.) > 60. {
            task["pollIntervalMs"] = json!(10_000);
        }
    }
    pub async fn notify(&self, ctx: &Context, token: ProgressToken) {
        let start = Instant::now();
        loop {
            let (message, eta, _) = self.status();
            let elapsed = start.elapsed().as_secs_f64();
            let mut notice =
                ProgressNotificationParam::new(token.clone(), elapsed).with_message(message);
            notice.total = eta.map(|e| elapsed + e);
            if ctx.peer.notify_progress(notice).await.is_err() {
                return;
            }
            tokio::time::sleep(Duration::from_secs(1)).await;
            let _ = tokio::time::timeout(Duration::from_secs(24), self.changed.notified()).await;
        }
    }
}
